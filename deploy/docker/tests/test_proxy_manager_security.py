#!/usr/bin/env python3
"""Unit tests for QG auto-proxy SSRF / DNS-rebinding hardening.

Verifies:
- _validate_fetch_url returns (host, safe_ips) and rejects unsafe inputs.
- QGAutoProxyManager pins request URL to a startup-validated IP at fetch time,
  preserving SNI hostname and Host header so DNS rebinding cannot redirect
  the connect target between startup and request.
"""

import os
import socket
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

import proxy_manager


def _gai_result(*ips):
    """Build a fake getaddrinfo() return value with the given IPv4 addresses."""
    out = []
    for ip in ips:
        out.append((socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0)))
    return out


class TestValidateFetchUrl(unittest.TestCase):
    def test_returns_host_and_safe_ips(self):
        with patch.object(proxy_manager.socket, "getaddrinfo",
                          return_value=_gai_result("8.8.8.8", "1.1.1.1")):
            host, ips = proxy_manager._validate_fetch_url("https://qg.example.com/fetch")
        self.assertEqual(host, "qg.example.com")
        self.assertEqual(ips, ["8.8.8.8", "1.1.1.1"])

    def test_dedupes_repeated_ips(self):
        with patch.object(proxy_manager.socket, "getaddrinfo",
                          return_value=_gai_result("8.8.8.8", "8.8.8.8")):
            _, ips = proxy_manager._validate_fetch_url("https://qg.example.com/fetch")
        self.assertEqual(ips, ["8.8.8.8"])

    def test_rejects_ip_literal(self):
        with self.assertRaisesRegex(ValueError, "must be a DNS hostname"):
            proxy_manager._validate_fetch_url("https://8.8.8.8/fetch")

    def test_rejects_private_resolution(self):
        with patch.object(proxy_manager.socket, "getaddrinfo",
                          return_value=_gai_result("10.0.0.5")):
            with self.assertRaisesRegex(ValueError, "blocked IP 10.0.0.5"):
                proxy_manager._validate_fetch_url("https://qg.example.com/fetch")

    def test_rejects_imds(self):
        with patch.object(proxy_manager.socket, "getaddrinfo",
                          return_value=_gai_result("169.254.169.254")):
            with self.assertRaisesRegex(ValueError, "blocked IP 169.254.169.254"):
                proxy_manager._validate_fetch_url("https://qg.example.com/fetch")

    def test_rejects_decimal_int_form(self):
        # 2130706433 == 127.0.0.1; ipaddress.ip_address accepts the int form
        # which is the bypass we want to block.
        with self.assertRaisesRegex(ValueError, "looks numeric"):
            proxy_manager._validate_fetch_url("https://2130706433/fetch")

    def test_rejects_resolution_failure(self):
        with patch.object(proxy_manager.socket, "getaddrinfo",
                          side_effect=socket.gaierror(-2, "Name or service not known")):
            with self.assertRaisesRegex(ValueError, "did not resolve"):
                proxy_manager._validate_fetch_url("https://qg.example.com/fetch")

    def test_rejects_bad_scheme(self):
        with self.assertRaisesRegex(ValueError, "scheme must be"):
            proxy_manager._validate_fetch_url("ftp://qg.example.com/fetch")


class TestPinnedRequest(unittest.TestCase):
    """The fetched URL must connect to a pre-validated IP, not a runtime DNS lookup."""

    def _make_manager(self, env=None, gai=None):
        env = env or {"QG_FETCH_URL": "https://qg.example.com/fetch"}
        gai = gai if gai is not None else _gai_result("8.8.8.8", "1.1.1.1")
        with patch.dict(os.environ, env, clear=False), \
                patch.object(proxy_manager.socket, "getaddrinfo", return_value=gai):
            return proxy_manager.QGAutoProxyManager()

    def test_pins_request_to_validated_ip(self):
        m = self._make_manager()
        url, headers, ext = m._build_pinned_request()
        self.assertEqual(url.host, "8.8.8.8")
        self.assertEqual(url.scheme, "https")
        self.assertEqual(url.path, "/fetch")
        self.assertEqual(headers["Host"], "qg.example.com")
        self.assertEqual(ext["sni_hostname"], "qg.example.com")

    def test_round_robins_across_ips(self):
        m = self._make_manager()
        url1, *_ = m._build_pinned_request()
        url2, *_ = m._build_pinned_request()
        url3, *_ = m._build_pinned_request()
        self.assertEqual(
            [url1.host, url2.host, url3.host],
            ["8.8.8.8", "1.1.1.1", "8.8.8.8"],
        )

    def test_preserves_port_and_query(self):
        m = self._make_manager(env={"QG_FETCH_URL": "https://qg.example.com:8443/fetch?n=5"})
        url, headers, ext = m._build_pinned_request()
        self.assertEqual(url.host, "8.8.8.8")
        self.assertEqual(url.port, 8443)
        self.assertEqual(url.query, b"n=5")
        self.assertEqual(headers["Host"], "qg.example.com")

    def test_ipv6_pinned_url_brackets(self):
        gai6 = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "",
                 ("2001:4860:4860::8888", 0, 0, 0))]
        m = self._make_manager(gai=gai6)
        url, _, _ = m._build_pinned_request()
        # httpx canonicalizes IPv6 -- assert brackets in str form.
        self.assertIn("[2001:4860:4860::8888]", str(url))


class TestRefuseDnsRebinding(unittest.TestCase):
    """End-to-end: at fetch time the transport must connect to the pinned IP,
    even if a hostile resolver would now return a private IP.

    We patch the network backend so we observe the host that httpcore is
    trying to connect to. If pinning works, that host is the validated public
    IP we resolved at startup, never the runtime "rebinded" host.
    """

    def test_connect_target_is_pinned_ip(self):
        gai = _gai_result("8.8.8.8")
        env = {"QG_FETCH_URL": "https://qg.example.com/fetch"}

        with patch.dict(os.environ, env, clear=False), \
                patch.object(proxy_manager.socket, "getaddrinfo", return_value=gai):
            m = proxy_manager.QGAutoProxyManager()
        url, headers, ext = m._build_pinned_request()
        # The URL host being literal IP is what makes httpcore skip DNS:
        # AsyncHTTPConnection passes self._origin.host straight to connect_tcp.
        self.assertEqual(url.host, "8.8.8.8")
        # And SNI/Host carry the original hostname so TLS + vhost still match.
        self.assertEqual(ext["sni_hostname"], "qg.example.com")
        self.assertEqual(headers["Host"], "qg.example.com")


if __name__ == "__main__":
    unittest.main(verbosity=2)
