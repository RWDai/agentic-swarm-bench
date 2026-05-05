"""Shared utilities for proxy and recorder modules."""

from __future__ import annotations

_ANTHROPIC_HOSTS = ("api.anthropic.com", "anthropic.com")
_API_SUFFIXES = ("/v1/chat/completions", "/v1/messages", "/v1/responses")


def _strip_api_suffix(url: str) -> str:
    """Strip a trailing API path so callers can append the correct one."""
    trimmed = url.rstrip("/")
    for suffix in _API_SUFFIXES:
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def _detect_upstream_api(upstream_url: str, explicit: str | None) -> str:
    """Return 'anthropic' or 'openai' based on explicit flag or URL heuristic."""
    if explicit:
        return explicit
    from urllib.parse import urlparse

    parsed = urlparse(upstream_url)
    host = parsed.hostname or ""
    if any(host.endswith(h) for h in _ANTHROPIC_HOSTS):
        return "anthropic"
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1/messages"):
        return "anthropic"
    return "openai"
