import asyncio
import ipaddress
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import httpx
from crawl4ai import CrawlerRunConfig, ProxyConfig

logger = logging.getLogger(__name__)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Why: When auto-proxy is on, silent fallback to direct connection leaks the
# server's real IP. Default REQUIRED=True so misconfig fails closed, not silently.
AUTO_PROXY_ENABLED = _as_bool(os.getenv("C4AI_AUTO_PROXY_ENABLED"), False)
AUTO_PROXY_REQUIRED = _as_bool(os.getenv("C4AI_AUTO_PROXY_REQUIRED"), True)


_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
        "instance-data",  # Azure IMDS host
    }
)


def _is_unsafe_ip(ip) -> bool:
    """True if IP belongs to a private / reserved / loopback / link-local range.

    Recurses into IPv4-mapped and IPv4-compatible IPv6 so the v4 semantics apply.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_unsafe_ip(ip.ipv4_mapped)
        as_int = int(ip)
        if 0 < as_int <= 0xFFFFFFFF:
            return _is_unsafe_ip(ipaddress.IPv4Address(as_int))
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_fetch_url(url: str) -> None:
    """Reject SSRF-prone QG_FETCH_URL values at startup.

    Strategy: disallow IP literals entirely, then DNS-resolve the hostname and
    block any answer in private/reserved space. This avoids numeric-form
    bypasses (decimal int / shorthand / IPv4-mapped IPv6).

    Known limitation: DNS rebinding between this startup check and httpx's
    per-request resolve. Full mitigation would need a custom transport that
    pins to the validated IP. TODO(2026-04-29 xiaodeng): revisit if QG endpoint
    moves to a less-trusted resolver path.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"QG_FETCH_URL scheme must be http or https, got {parsed.scheme!r}"
        )
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise ValueError("QG_FETCH_URL must include a hostname")
    if host in _BLOCKED_HOSTNAMES:
        raise ValueError(f"QG_FETCH_URL host {host!r} is blocked")

    try:
        ipaddress.ip_address(host)
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False
    if is_ip_literal:
        raise ValueError(
            "QG_FETCH_URL must be a DNS hostname, not an IP literal"
        )
    if all(c in "0123456789." for c in host):
        raise ValueError(
            f"QG_FETCH_URL host {host!r} looks numeric; use a real hostname"
        )

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(
            f"QG_FETCH_URL hostname did not resolve: {type(e).__name__}"
        ) from e
    for info in infos:
        addr = info[4][0]
        # IPv6 sockaddr may include zone id, e.g. "fe80::1%eth0"
        if "%" in addr:
            addr = addr.split("%", 1)[0]
        ip = ipaddress.ip_address(addr)
        if _is_unsafe_ip(ip):
            raise ValueError(
                f"QG_FETCH_URL host {host!r} resolves to blocked IP {addr}"
            )


def _safe_error_message(e: BaseException) -> str:
    """Avoid leaking URL/credentials from httpx exception strings into status."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    if isinstance(e, httpx.TimeoutException):
        return "fetch timeout"
    if isinstance(e, httpx.ConnectError):
        return "connect error"
    if isinstance(e, httpx.HTTPError):
        return f"http error: {type(e).__name__}"
    return type(e).__name__


@dataclass
class _TimedProxy:
    proxy: ProxyConfig
    expires_at: float

    def is_valid(self, min_ttl_seconds: int) -> bool:
        return (self.expires_at - time.time()) > min_ttl_seconds


class QGAutoProxyManager:
    """
    On-demand short-lived proxy manager for server-side auto injection.
    Designed for qg mode-2 style short-expiry proxies.
    """

    def __init__(self) -> None:
        self.fetch_url = os.getenv("QG_FETCH_URL", "").strip()
        self.auth_key = os.getenv("QG_AUTH_KEY")
        self.auth_pwd = os.getenv("QG_AUTH_PWD")
        self.ttl_seconds = int(os.getenv("PROXY_TTL_SECONDS", "55"))
        self.min_ttl_seconds = int(os.getenv("PROXY_MIN_TTL_SECONDS", "10"))
        self.min_ready = int(os.getenv("PROXY_MIN_READY", "2"))
        self.max_ready = int(os.getenv("PROXY_MAX_READY", "8"))
        self.fetch_timeout_seconds = float(os.getenv("QG_FETCH_TIMEOUT_SECONDS", "10"))
        self.max_response_bytes = int(os.getenv("QG_FETCH_MAX_BYTES", "1048576"))  # 1 MiB

        if self.min_ready < 1:
            raise ValueError("PROXY_MIN_READY must be >= 1")
        if self.max_ready < self.min_ready:
            raise ValueError("PROXY_MAX_READY must be >= PROXY_MIN_READY")

        if self.fetch_url:
            _validate_fetch_url(self.fetch_url)

        self._pool: List[_TimedProxy] = []
        self._lock = asyncio.Lock()
        self.last_fetch_at: Optional[float] = None
        self.last_fetch_ok: Optional[bool] = None
        self.last_error: Optional[str] = None
        self.last_fetch_count: int = 0
        self.total_acquired: int = 0  # 累计已消耗的代理数

    async def acquire(self) -> ProxyConfig:
        if not self.fetch_url:
            raise RuntimeError("QG_FETCH_URL is required when auto proxy is enabled")

        async with self._lock:
            self._drop_expired_locked()
            if len(self._pool) < self.min_ready:
                await self._refill_locked()

            if not self._pool:
                raise RuntimeError("No available proxy from QG endpoint")

            # 一次性代理：取出后不放回，用完即弃
            item = self._pool.pop(0)
            self.total_acquired += 1
            logger.info(
                "Proxy acquired: %s (pool remaining: %d, total used: %d)",
                item.proxy.server, len(self._pool), self.total_acquired,
            )
            return item.proxy

    def _drop_expired_locked(self) -> None:
        self._pool = [p for p in self._pool if p.is_valid(self.min_ttl_seconds)]

    async def _refill_locked(self) -> None:
        rows = await self._fetch_rows()
        parsed = self._rows_to_proxy_configs(rows)

        now = time.time()
        seen = {(p.proxy.server, p.proxy.username) for p in self._pool}
        for px in parsed:
            key = (px.server, px.username)
            if key in seen:
                continue
            self._pool.append(_TimedProxy(proxy=px, expires_at=now + self.ttl_seconds))
            seen.add(key)
            if len(self._pool) >= self.max_ready:
                break

    async def _fetch_rows(self) -> List[str]:
        self.last_fetch_at = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.fetch_timeout_seconds) as client:
                async with client.stream("GET", self.fetch_url) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "").lower()
                    body_bytes = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body_bytes.extend(chunk)
                        if len(body_bytes) > self.max_response_bytes:
                            raise ValueError(
                                f"QG response exceeded {self.max_response_bytes} bytes"
                            )
                body = body_bytes.decode("utf-8", errors="replace").strip()

            if "application/json" in content_type:
                rows = self._rows_from_json(json.loads(body))
            elif body.startswith("{") or body.startswith("["):
                try:
                    rows = self._rows_from_json(json.loads(body))
                except Exception:
                    rows = self._rows_from_text(body)
            else:
                rows = self._rows_from_text(body)

            self.last_fetch_ok = True
            self.last_error = None
            self.last_fetch_count = len(rows)
            return rows
        except Exception as e:
            self.last_fetch_ok = False
            self.last_error = _safe_error_message(e)
            self.last_fetch_count = 0
            raise

    @staticmethod
    def _rows_from_text(text: str) -> List[str]:
        rows: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line and "://" not in line:
                rows.extend([x.strip() for x in line.split(",") if x.strip()])
            else:
                rows.append(line)
        return rows

    @staticmethod
    def _rows_from_json(data: object) -> List[str]:
        # Common provider response styles.
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            for key in ("data", "list", "proxies", "result"):
                if key not in data:
                    continue
                value = data[key]
                if not isinstance(value, list):
                    continue
                out: List[str] = []
                for item in value:
                    if isinstance(item, str):
                        out.append(item.strip())
                    elif isinstance(item, dict):
                        ip = item.get("ip") or item.get("host")
                        port = item.get("port")
                        if ip and port:
                            out.append(f"{ip}:{port}")
                return [x for x in out if x]
        return []

    def _rows_to_proxy_configs(self, rows: Iterable[str]) -> List[ProxyConfig]:
        out: List[ProxyConfig] = []
        for row in rows:
            row = row.strip()
            if not row:
                continue

            if row.startswith(("http://", "https://", "socks5://")):
                out.append(
                    ProxyConfig(
                        server=row,
                        username=self.auth_key if self.auth_key else None,
                        password=self.auth_pwd if self.auth_pwd else None,
                    )
                )
                continue

            parts = row.split(":")

            # ip:port:user:pass
            if len(parts) == 4 and parts[1].isdigit():
                ip, port, user, pwd = parts
                out.append(ProxyConfig(server=f"http://{ip}:{port}", username=user, password=pwd))
                continue

            # ip:port + global auth key/pwd
            if len(parts) == 2 and parts[1].isdigit():
                ip, port = parts
                if self.auth_key and self.auth_pwd:
                    out.append(
                        ProxyConfig(
                            server=f"http://{ip}:{port}",
                            username=self.auth_key,
                            password=self.auth_pwd,
                        )
                    )
                else:
                    out.append(ProxyConfig(server=f"http://{ip}:{port}"))
                continue

        return out


_proxy_manager: Optional[QGAutoProxyManager] = None


def get_proxy_manager() -> QGAutoProxyManager:
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = QGAutoProxyManager()
    return _proxy_manager


def get_proxy_status() -> dict:
    manager = get_proxy_manager()
    now = time.time()

    total = len(manager._pool)
    valid = sum(1 for item in manager._pool if item.is_valid(manager.min_ttl_seconds))
    # Expiring soon: valid now, but less than configured ttl remaining.
    expiring_soon = sum(
        1
        for item in manager._pool
        if item.expires_at > now and (item.expires_at - now) <= manager.min_ttl_seconds
    )

    return {
        "enabled": AUTO_PROXY_ENABLED,
        "required": AUTO_PROXY_REQUIRED,
        "fetch_url_configured": bool(manager.fetch_url),
        "pool": {
            "total": total,
            "valid": valid,
            "expiring_soon": expiring_soon,
            "min_ready": manager.min_ready,
            "max_ready": manager.max_ready,
            "ttl_seconds": manager.ttl_seconds,
            "min_ttl_seconds": manager.min_ttl_seconds,
        },
        "usage": {
            "total_acquired": manager.total_acquired,
        },
        "fetch": {
            "last_fetch_at": manager.last_fetch_at,
            "last_fetch_ok": manager.last_fetch_ok,
            "last_fetch_count": manager.last_fetch_count,
            "last_error": manager.last_error,
            "timeout_seconds": manager.fetch_timeout_seconds,
        },
    }


async def maybe_apply_auto_proxy(run_config: CrawlerRunConfig) -> CrawlerRunConfig:
    """
    Inject proxy_config into run_config when:
    - auto proxy is enabled
    - caller did not set proxy_config / proxy_rotation_strategy explicitly
    """
    if not AUTO_PROXY_ENABLED:
        return run_config
    if run_config.proxy_config or run_config.proxy_rotation_strategy:
        return run_config

    try:
        proxy = await get_proxy_manager().acquire()
        run_config.proxy_config = proxy
        return run_config
    except Exception as e:
        # Why: full repr() of exception may include configured QG_FETCH_URL or
        # auth values that surface in API responses. Server logs keep the full
        # traceback; client only sees the sanitized class-level message.
        logger.warning("Auto proxy injection failed: %s", e)
        sanitized = _safe_error_message(e)
        if AUTO_PROXY_REQUIRED:
            raise RuntimeError(f"Auto proxy injection failed: {sanitized}") from e
        return run_config

