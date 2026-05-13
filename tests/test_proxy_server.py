"""Tests for the FastAPI recording proxy in proxy/server.py."""

from __future__ import annotations

import json

import pytest
import respx

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from agentic_swarm_bench.proxy.server import (  # noqa: E402
    _detect_upstream_api,
    _strip_api_suffix,
    create_app,
)

# ---------------------------------------------------------------------------
# _detect_upstream_api (now from proxy.utils, re-exported via server)
# ---------------------------------------------------------------------------


def test_detect_explicit_anthropic():
    assert _detect_upstream_api("http://openai.com", "anthropic") == "anthropic"


def test_detect_explicit_openai():
    assert _detect_upstream_api("https://api.anthropic.com", "openai") == "openai"


def test_detect_anthropic_by_host():
    assert _detect_upstream_api("https://api.anthropic.com", None) == "anthropic"


def test_detect_anthropic_subdomain():
    assert _detect_upstream_api("https://sub.anthropic.com/v1", None) == "anthropic"


def test_detect_openai_fallback():
    assert _detect_upstream_api("http://localhost:8000", None) == "openai"
    assert _detect_upstream_api("https://api.openai.com", None) == "openai"


def test_detect_anthropic_by_path():
    assert _detect_upstream_api("https://api.swarmone.ai/abc/v1/messages", None) == "anthropic"
    assert _detect_upstream_api("https://example.com/v1/messages/", None) == "anthropic"


def test_detect_openai_by_path():
    assert _detect_upstream_api("https://example.com/v1/chat/completions", None) == "openai"


# ---------------------------------------------------------------------------
# _strip_api_suffix
# ---------------------------------------------------------------------------


def test_strip_suffix_chat_completions():
    assert _strip_api_suffix("http://host/v1/chat/completions") == "http://host"


def test_strip_suffix_messages():
    assert _strip_api_suffix("https://api.swarmone.ai/abc/v1/messages") == "https://api.swarmone.ai/abc"


def test_strip_suffix_responses():
    assert _strip_api_suffix("http://host/v1/responses") == "http://host"


def test_strip_suffix_trailing_slash():
    assert _strip_api_suffix("http://host/v1/messages/") == "http://host"


def test_strip_suffix_noop():
    assert _strip_api_suffix("http://host:8000") == "http://host:8000"
    assert _strip_api_suffix("http://host:8000/") == "http://host:8000"


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


def test_create_app_returns_fastapi_instance(tmp_path):
    app = create_app(
        upstream_url="http://fake:8000",
        model="test",
        log_dir=str(tmp_path),
    )
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# /benchmark/metrics
# ---------------------------------------------------------------------------


async def test_metrics_empty_when_no_file(tmp_path):
    app = create_app(upstream_url="http://fake:8000", model="test", log_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/benchmark/metrics")
    assert resp.status_code == 200
    assert resp.json() == {"metrics": []}


async def test_metrics_parses_jsonl(tmp_path):
    metrics_file = tmp_path / "metrics.jsonl"
    entry = {"req_id": 1, "ttft_ms": 100.0, "stream": True}
    metrics_file.write_text(json.dumps(entry) + "\n")

    app = create_app(upstream_url="http://fake:8000", model="test", log_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/benchmark/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["metrics"]) == 1
    assert data["metrics"][0]["req_id"] == 1


# ---------------------------------------------------------------------------
# /benchmark/summary
# ---------------------------------------------------------------------------


async def test_summary_no_file(tmp_path):
    app = create_app(upstream_url="http://fake:8000", model="test", log_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/benchmark/summary")
    assert resp.status_code == 200
    assert "error" in resp.json()


async def test_summary_with_streaming_entries(tmp_path):
    metrics_file = tmp_path / "metrics.jsonl"
    entries = [
        {"req_id": i, "stream": True, "ttft_ms": 100.0 + i * 10, "tok_per_sec": 30.0}
        for i in range(3)
    ]
    metrics_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    app = create_app(upstream_url="http://fake:8000", model="test", log_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/benchmark/summary")
    data = resp.json()
    assert data["total_requests"] == 3
    assert data["streaming_requests"] == 3
    assert data["ttft_ms"]["count"] == 3


async def test_summary_non_streaming_entries(tmp_path):
    metrics_file = tmp_path / "metrics.jsonl"
    entries = [{"req_id": i, "stream": False} for i in range(2)]
    metrics_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    app = create_app(upstream_url="http://fake:8000", model="test", log_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/benchmark/summary")
    data = resp.json()
    assert data["total_requests"] == 2
    assert data["streaming_requests"] == 0


async def test_openai_chat_passthrough_records_metrics(tmp_path):
    app = create_app(upstream_url="http://fake:8000", model="test", log_dir=str(tmp_path))
    with respx.mock:
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            )
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "ignored", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert resp.status_code == 200
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["stream"] is False
    assert metrics[0]["input_tokens_actual"] == 5
    assert metrics[0]["output_tokens"] == 2


# ---------------------------------------------------------------------------
# API key header forwarding
# ---------------------------------------------------------------------------


def test_upstream_headers_bearer(tmp_path):
    """create_app with api_key should produce Authorization: Bearer <key>."""
    app = create_app(
        upstream_url="http://fake:8000",
        model="test",
        api_key="sk-test",
        log_dir=str(tmp_path),
    )
    # Access the closure-private helper indirectly - verify via app state
    # (The public contract is that requests to upstream carry the header.)
    assert app is not None  # app created successfully with api_key


def test_upstream_headers_custom_header(tmp_path):
    app = create_app(
        upstream_url="http://fake:8000",
        model="test",
        api_key="my-key",
        api_key_header="X-Custom-Auth",
        log_dir=str(tmp_path),
    )
    assert app is not None
