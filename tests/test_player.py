"""Tests for player.py: token estimation, SSE streaming, retry logic, and reasoning fields."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.metrics.collector import RequestMetrics
from agentic_swarm_bench.scenarios.player import (
    _bucket_label,
    _build_headers,
    _compute_bucket_wall_time,
    _estimate_tokens,
    _replay_one_request,
    _slice_entries,
    _strip_cache_control,
)


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Token estimation from content length (~4 chars per token)."""

    def test_empty_string_returns_zero(self):
        assert _estimate_tokens("") == 0

    def test_none_returns_zero(self):
        assert _estimate_tokens(None) == 0

    def test_short_string_returns_minimum_one(self):
        assert _estimate_tokens("hi") == 1
        assert _estimate_tokens("a") == 1
        assert _estimate_tokens("abc") == 1

    def test_exactly_four_chars_returns_one(self):
        assert _estimate_tokens("abcd") == 1

    def test_eight_chars_returns_two(self):
        assert _estimate_tokens("abcdefgh") == 2

    def test_long_content_scales_linearly(self):
        text = "x" * 400
        assert _estimate_tokens(text) == 100

    def test_realistic_english_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = _estimate_tokens(text)
        assert tokens == len(text) // 4
        assert tokens >= 10

    def test_single_sse_chunk_with_many_tokens(self):
        """Anthropic sends 20+ tokens per SSE chunk. This must count > 1."""
        chunk_content = "Here is a response with many tokens in a single SSE event"
        tokens = _estimate_tokens(chunk_content)
        assert tokens > 10

    def test_reasoning_content_also_estimated(self):
        reasoning = "Let me think through this step by step. First, I need to consider..."
        tokens = _estimate_tokens(reasoning)
        assert tokens > 10


# ---------------------------------------------------------------------------
# _replay_one_request -- SSE streaming with token counting
# ---------------------------------------------------------------------------


def _make_sse_lines(chunks: list[dict], usage: dict | None = None) -> list[str]:
    """Build SSE `data: ...` lines from chunk dicts."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    if usage:
        lines.append(f"data: {json.dumps({'usage': usage, 'choices': []})}")
    lines.append("data: [DONE]")
    return lines


def _sse_chunk(content: str | None = None, reasoning: str | None = None) -> dict:
    """Build a single SSE chunk with optional content/reasoning."""
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"delta": delta}]}


def _sse_chunk_reasoning_field(reasoning: str) -> dict:
    """Build an SSE chunk using the 'reasoning' field (Together/GLM convention)."""
    return {"choices": [{"delta": {"reasoning": reasoning}}]}


class FakeStreamResponse:
    """Fake httpx streaming response for testing _replay_one_request."""

    def __init__(self, lines: list[str], status_code: int = 200, body: bytes = b""):
        self.lines = lines
        self.status_code = status_code
        self._body = body

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aread(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    """Fake httpx.AsyncClient that returns a FakeStreamResponse from .stream()."""

    def __init__(self, response: FakeStreamResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response


def _replay(**kwargs):
    """Shorthand: call _replay_one_request with sensible defaults, run sync."""
    defaults = dict(
        client=None,
        url="http://test/v1/chat/completions",
        model="test",
        headers={},
        messages=[{"role": "user", "content": "test"}],
        max_tokens=512,
        seq=0,
        timeout=30.0,
    )
    defaults.update(kwargs)
    return _run(_replay_one_request(**defaults))


def test_token_count_uses_content_length_not_chunk_count():
    """The old bug: each SSE chunk counted as 1 token.
    With 3 chunks of ~60 chars each, we should get ~45 tokens, not 3."""
    lines = _make_sse_lines(
        [
            _sse_chunk(content="Here is a detailed response with many tokens "),
            _sse_chunk(content="continuing the explanation with more details "),
            _sse_chunk(content="and finishing up with a final conclusion here"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens > 3, (
        f"Got {m.completion_tokens} tokens - still counting chunks as 1 each"
    )
    assert m.completion_tokens > 30


def test_usage_completion_tokens_overrides_estimate():
    """When the API provides usage.completion_tokens, it should override the estimate."""
    lines = _make_sse_lines(
        [_sse_chunk(content="short response")],
        usage={"prompt_tokens": 10, "completion_tokens": 71},
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens == 71
    assert m.prompt_tokens == 10


def test_no_usage_falls_back_to_estimate():
    """Without usage data, fall back to the char-based estimate."""
    text = "a" * 80  # 80 chars -> 20 estimated tokens
    lines = _make_sse_lines([_sse_chunk(content=text)])
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens == 20


def test_stream_options_include_usage_in_payload():
    """Payload should include stream_options: {include_usage: true}."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(client=CapturingClient())
    assert "stream_options" in captured_payload
    assert captured_payload["stream_options"]["include_usage"] is True


# ---------------------------------------------------------------------------
# Reasoning token field name handling
# ---------------------------------------------------------------------------


def test_reasoning_content_field():
    """Anthropic/DeepSeek convention: delta.reasoning_content."""
    lines = _make_sse_lines(
        [
            _sse_chunk(reasoning="Let me think about this problem carefully"),
            _sse_chunk(content="The answer is 42"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.thinking_tokens > 0
    assert m.ttft_thinking_ms > 0
    assert m.ttft_visible_ms > 0


def test_reasoning_field_together_convention():
    """Together/GLM convention: delta.reasoning (not reasoning_content)."""
    lines = _make_sse_lines(
        [
            _sse_chunk_reasoning_field("Step 1: analyze the problem carefully"),
            _sse_chunk(content="Final answer is here"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.thinking_tokens > 0, "Together-style 'reasoning' field was not counted"
    assert m.ttft_thinking_ms > 0


def test_no_reasoning_tokens_when_content_only():
    """Regular (non-reasoning) response should have zero thinking tokens."""
    lines = _make_sse_lines(
        [
            _sse_chunk(content="Just a normal response without thinking"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.thinking_tokens == 0
    assert m.ttft_thinking_ms == 0


# ---------------------------------------------------------------------------
# Rate limit retry (HTTP 429)
# ---------------------------------------------------------------------------


def test_http_429_records_error_when_retries_disabled():
    """With max_retries=0 (default), 429 is recorded as an error, no retry."""
    resp = FakeStreamResponse([], status_code=429, body=b"rate limited")
    m = _replay(client=FakeClient(resp), max_retries=0)
    assert m.error is not None
    assert "429" in m.error


def test_http_429_retries_then_succeeds():
    """With retries enabled, 429 should be retried with backoff."""
    call_count = 0

    class RetryClient:
        def stream(self, method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeStreamResponse([], status_code=429, body=b"rate limited")
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="success after retry")]))

    with patch("agentic_swarm_bench.scenarios.player.asyncio.sleep", new_callable=AsyncMock):
        m = _replay(client=RetryClient(), max_retries=2)
    assert call_count == 2
    assert m.error is None
    assert m.completion_tokens > 0


def test_stream_options_retry_on_rejection():
    """If a provider rejects stream_options, retry without it."""
    call_count = 0
    captured_payloads = []

    class RejectStreamOptionsClient:
        def stream(self, method, url, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(dict(json or {}))
            if call_count == 1 and "stream_options" in (json or {}):
                return FakeStreamResponse(
                    [],
                    status_code=400,
                    body=b'{"error": "stream_options is not supported"}',
                )
            return FakeStreamResponse(
                _make_sse_lines([_sse_chunk(content="ok without stream_options")])
            )

    m = _replay(client=RejectStreamOptionsClient(), max_retries=1)
    assert call_count == 2
    assert "stream_options" in captured_payloads[0]
    assert "stream_options" not in captured_payloads[1]
    assert m.error is None


def test_http_500_not_retried():
    """Non-429/non-stream_options errors should not be retried."""
    call_count = 0

    class Error500Client:
        def stream(self, method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeStreamResponse([], status_code=500, body=b"internal error")

    m = _replay(client=Error500Client(), max_retries=3)
    assert call_count == 1
    assert "500" in m.error


def test_429_exhausts_all_retries():
    """If every attempt gets 429, we should see max_retries+1 attempts and an error."""
    call_count = 0

    class Always429Client:
        def stream(self, method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeStreamResponse([], status_code=429, body=b"rate limited")

    with patch("agentic_swarm_bench.scenarios.player.asyncio.sleep", new_callable=AsyncMock):
        m = _replay(client=Always429Client(), max_retries=2)
    assert call_count == 3  # initial + 2 retries
    assert m.error is not None
    assert "429" in m.error


# ---------------------------------------------------------------------------
# Payload construction edge cases
# ---------------------------------------------------------------------------


def test_openai_endpoint_uses_max_completion_tokens():
    """OpenAI endpoints should use max_completion_tokens, not max_tokens."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(
        client=CapturingClient(),
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-5.4",
        max_tokens=1024,
    )
    assert "max_completion_tokens" in captured_payload
    assert "max_tokens" not in captured_payload


def test_non_openai_endpoint_uses_max_tokens():
    """Non-OpenAI endpoints should use max_tokens."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(
        client=CapturingClient(),
        url="https://api.anthropic.com/v1/chat/completions",
        model="claude",
        max_tokens=1024,
    )
    assert "max_tokens" in captured_payload
    assert "max_completion_tokens" not in captured_payload


def test_max_tokens_passed_through():
    """max_tokens should be forwarded to the API without capping."""
    captured_payload = {}

    class CapturingClient:
        def stream(self, method, url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

    _replay(client=CapturingClient(), max_tokens=16384)
    assert captured_payload["max_tokens"] == 16384


# ---------------------------------------------------------------------------
# _bucket_label
# ---------------------------------------------------------------------------


class TestBucketLabel:
    def test_fresh(self):
        assert _bucket_label(0) == "fresh"
        assert _bucket_label(5000) == "fresh"
        assert _bucket_label(9999) == "fresh"

    def test_short(self):
        assert _bucket_label(10000) == "short"
        assert _bucket_label(29999) == "short"

    def test_medium(self):
        assert _bucket_label(30000) == "medium"
        assert _bucket_label(54999) == "medium"

    def test_long(self):
        assert _bucket_label(55000) == "long"
        assert _bucket_label(84999) == "long"

    def test_full(self):
        assert _bucket_label(85000) == "full"
        assert _bucket_label(149999) == "full"

    def test_xl(self):
        assert _bucket_label(150000) == "xl"
        assert _bucket_label(299999) == "xl"

    def test_xxl(self):
        assert _bucket_label(300000) == "xxl"
        assert _bucket_label(1000000) == "xxl"


# ---------------------------------------------------------------------------
# _compute_bucket_wall_time
# ---------------------------------------------------------------------------


class TestComputeBucketWallTime:
    def test_empty(self):
        assert _compute_bucket_wall_time([]) == 0.0

    def test_single_user(self):
        reqs = [
            RequestMetrics(user_id=0, total_time_s=1.0),
            RequestMetrics(user_id=0, total_time_s=2.0),
        ]
        assert _compute_bucket_wall_time(reqs) == 3.0

    def test_multiple_user_ids_take_max(self):
        reqs = [
            RequestMetrics(user_id=0, total_time_s=1.0),
            RequestMetrics(user_id=0, total_time_s=2.0),  # user 0 total: 3.0
            RequestMetrics(user_id=1, total_time_s=1.5),
            RequestMetrics(user_id=1, total_time_s=1.0),  # user 1 total: 2.5
        ]
        assert _compute_bucket_wall_time(reqs) == 3.0


# ---------------------------------------------------------------------------
# _slice_entries
# ---------------------------------------------------------------------------


class TestSliceEntries:
    def _make_entry(self, prompt_tokens=None, content_len=400):
        entry = MagicMock()
        entry.prompt_tokens = prompt_tokens
        entry.messages = [{"content": "x" * content_len}]
        return entry

    def test_no_budget_returns_all(self):
        entries = [self._make_entry() for _ in range(5)]
        result = _slice_entries(entries, slice_tokens=None)
        assert len(result) == 5

    def test_budget_limits_entries(self):
        entries = [self._make_entry(prompt_tokens=1000) for _ in range(10)]
        result = _slice_entries(entries, slice_tokens=2500)
        assert len(result) == 2

    def test_always_includes_first_entry_even_if_exceeds(self):
        entries = [self._make_entry(prompt_tokens=5000)]
        result = _slice_entries(entries, slice_tokens=100)
        assert len(result) == 1

    def test_falls_back_to_char_estimate(self):
        entries = [self._make_entry(prompt_tokens=None, content_len=400) for _ in range(5)]
        result = _slice_entries(entries, slice_tokens=250)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Timing metrics
# ---------------------------------------------------------------------------


def test_ttft_and_tok_per_sec_computed():
    """Verify that TTFT and tok/s are computed on a successful stream."""
    lines = _make_sse_lines(
        [
            _sse_chunk(content="hello world response with enough tokens to measure"),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.ttft_ms > 0
    assert m.total_time_s > 0
    assert m.error is None


def test_context_tokens_estimated_from_messages():
    """context_tokens should be estimated from message content length."""
    content = "x" * 800  # 200 estimated tokens
    lines = _make_sse_lines([_sse_chunk(content="ok")])
    m = _replay(
        client=FakeClient(FakeStreamResponse(lines)),
        messages=[{"role": "user", "content": content}],
    )
    assert m.context_tokens == 200


def test_timeout_records_error():
    """Wall-clock timeout should produce an error metric."""

    class HangingResponse:
        status_code = 200

        async def aiter_lines(self):
            await asyncio.sleep(999)
            yield "data: [DONE]"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class HangingClient:
        def stream(self, method, url, **kwargs):
            return HangingResponse()

    m = _replay(client=HangingClient(), timeout=0.01)
    assert m.error is not None
    assert "timeout" in m.error.lower() or "Timeout" in m.error


def test_empty_stream_returns_zero_tokens():
    """A stream that sends no content chunks should have 0 completion tokens."""
    lines = _make_sse_lines([])
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.completion_tokens == 0


def test_mixed_content_and_reasoning_tokens():
    """Both content tokens and reasoning tokens should be counted separately."""
    reasoning_text = "Let me think step by step about this problem"
    content_text = "The answer is forty two"
    lines = _make_sse_lines(
        [
            _sse_chunk(reasoning=reasoning_text),
            _sse_chunk(content=content_text),
        ]
    )
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    expected_reasoning = _estimate_tokens(reasoning_text)
    expected_total = expected_reasoning + _estimate_tokens(content_text)

    assert m.thinking_tokens == expected_reasoning
    assert m.completion_tokens == expected_total


def test_malformed_json_in_sse_is_skipped():
    """Malformed JSON in SSE should be silently skipped, not crash."""
    lines = [
        "data: {invalid json}",
        f"data: {json.dumps(_sse_chunk(content='valid chunk'))}",
        "data: [DONE]",
    ]
    m = _replay(client=FakeClient(FakeStreamResponse(lines)))
    assert m.error is None
    assert m.completion_tokens > 0


# ---------------------------------------------------------------------------
# _strip_cache_control
# ---------------------------------------------------------------------------


class TestStripCacheControl:
    """Stripping cache_control from serialized content blocks."""

    def test_string_dict_with_cache_control(self):
        content = json.dumps({
            "tool_use_id": "abc",
            "content": "result",
            "cache_control": {"type": "ephemeral"},
        })
        result = _strip_cache_control(content)
        parsed = json.loads(result)
        assert "cache_control" not in parsed
        assert parsed["tool_use_id"] == "abc"
        assert parsed["content"] == "result"

    def test_string_list_with_cache_control(self):
        content = json.dumps([
            {"type": "text", "text": "hello"},
            {"type": "tool_result", "cache_control": {"type": "ephemeral"}},
        ])
        result = _strip_cache_control(content)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert "cache_control" not in parsed[0]
        assert "cache_control" not in parsed[1]

    def test_list_content_with_cache_control(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "tool_result", "cache_control": {"type": "ephemeral"}},
        ]
        result = _strip_cache_control(content)
        assert isinstance(result, list)
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]

    def test_plain_string_unchanged(self):
        assert _strip_cache_control("hello world") == "hello world"

    def test_empty_string_unchanged(self):
        assert _strip_cache_control("") == ""

    def test_string_without_cache_control_unchanged(self):
        content = json.dumps({"tool_use_id": "abc", "content": "result"})
        assert _strip_cache_control(content) == content

    def test_non_json_string_with_keyword_unchanged(self):
        content = 'this mentions "cache_control" but is not JSON'
        assert _strip_cache_control(content) == content

    def test_none_passthrough(self):
        assert _strip_cache_control(None) is None


# ---------------------------------------------------------------------------
# Anthropic replay path -- SSE parsing, payload construction, headers
# ---------------------------------------------------------------------------


def _anthropic_event(event_type: str, data: dict) -> bytes:
    """Build one Anthropic-style SSE event as bytes."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _anthropic_text_delta(text: str, index: int = 0) -> bytes:
    return _anthropic_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def _anthropic_thinking_delta(thinking: str, index: int = 0) -> bytes:
    return _anthropic_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "thinking_delta", "thinking": thinking},
    })


def _anthropic_input_json_delta(partial_json: str, index: int = 1) -> bytes:
    return _anthropic_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })


def _anthropic_message_start(input_tokens: int = 100) -> bytes:
    return _anthropic_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "usage": {"input_tokens": input_tokens},
        },
    })


def _anthropic_message_delta(output_tokens: int = 50) -> bytes:
    return _anthropic_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": output_tokens},
    })


class FakeAnthropicStreamResponse:
    """Fake httpx streaming response that yields bytes for Anthropic SSE."""

    def __init__(self, events: list[bytes], status_code: int = 200, body: bytes = b""):
        self.status_code = status_code
        self._events = events
        self._body = body

    async def aiter_bytes(self):
        for event in self._events:
            yield event

    async def aread(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeAnthropicClient:
    """Fake httpx.AsyncClient that returns a FakeAnthropicStreamResponse."""

    def __init__(self, response: FakeAnthropicStreamResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response


def _replay_anthropic(**kwargs):
    """Shorthand: call _replay_one_request with Anthropic defaults, run sync."""
    defaults = dict(
        client=None,
        url="https://api.anthropic.com",
        model="claude-test",
        headers={"anthropic-version": "2023-06-01"},
        messages=[{"role": "user", "content": "test"}],
        max_tokens=512,
        seq=0,
        timeout=30.0,
        upstream_api="anthropic",
    )
    defaults.update(kwargs)
    return _run(_replay_one_request(**defaults))


class TestAnthropicReplayTemperature:
    """Anthropic replay payload must include temperature matching the OAI path."""

    def test_anthropic_payload_includes_temperature(self):
        captured_payload = {}

        class CapturingClient:
            def stream(self, method, url, json=None, **kwargs):
                captured_payload.update(json or {})
                return FakeAnthropicStreamResponse([
                    _anthropic_message_start(),
                    _anthropic_text_delta("ok"),
                    _anthropic_message_delta(output_tokens=1),
                ])

        _replay_anthropic(client=CapturingClient())
        assert "temperature" in captured_payload
        assert captured_payload["temperature"] == 0.7

    def test_oai_payload_temperature_matches_anthropic(self):
        """Both paths must use the same temperature value."""
        captured_oai = {}
        captured_anth = {}

        class OaiCapture:
            def stream(self, method, url, json=None, **kwargs):
                captured_oai.update(json or {})
                return FakeStreamResponse(_make_sse_lines([_sse_chunk(content="ok")]))

        class AnthCapture:
            def stream(self, method, url, json=None, **kwargs):
                captured_anth.update(json or {})
                return FakeAnthropicStreamResponse([
                    _anthropic_message_start(),
                    _anthropic_text_delta("ok"),
                    _anthropic_message_delta(output_tokens=1),
                ])

        _replay(client=OaiCapture())
        _replay_anthropic(client=AnthCapture())
        assert captured_oai["temperature"] == captured_anth["temperature"]


class TestAnthropicInputJsonDelta:
    """Anthropic tool-use input_json_delta events must be tracked for timing."""

    def test_input_json_delta_counted_for_ttft(self):
        """When tool input is the first content, TTFT must be set."""
        events = [
            _anthropic_message_start(),
            _anthropic_input_json_delta('{"query":'),
            _anthropic_input_json_delta('"hello"}'),
            _anthropic_message_delta(output_tokens=10),
        ]
        m = _replay_anthropic(client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)))
        assert m.ttft_ms > 0, "TTFT should be set from input_json_delta"
        assert m.error is None

    def test_input_json_delta_then_text_both_tracked(self):
        """Both tool input and text deltas should contribute to timing."""
        events = [
            _anthropic_message_start(),
            _anthropic_input_json_delta('{"q":"v"}'),
            _anthropic_text_delta("Here is the result"),
            _anthropic_message_delta(output_tokens=20),
        ]
        m = _replay_anthropic(client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)))
        assert m.ttft_ms > 0
        assert m.completion_tokens == 20
        assert len(m.itl_ms) > 0, "Should have ITL entries from multiple deltas"

    def test_text_only_no_tool_input(self):
        """Text-only responses should work exactly as before."""
        text = "Here is a detailed response with enough content"
        events = [
            _anthropic_message_start(),
            _anthropic_text_delta(text),
            _anthropic_message_delta(output_tokens=15),
        ]
        m = _replay_anthropic(client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)))
        assert m.ttft_ms > 0
        assert m.completion_tokens == 15
        assert m.error is None


class TestAnthropicResponseChunks:
    """response_chunks should collect text but NOT tool input."""

    def test_text_collected_in_response_chunks(self):
        events = [
            _anthropic_message_start(),
            _anthropic_text_delta("Hello "),
            _anthropic_text_delta("world"),
            _anthropic_message_delta(output_tokens=5),
        ]
        response_chunks: list[str] = []
        _replay_anthropic(
            client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)),
            response_chunks=response_chunks,
        )
        assert "".join(response_chunks) == "Hello world"

    def test_tool_input_not_in_response_chunks(self):
        events = [
            _anthropic_message_start(),
            _anthropic_input_json_delta('{"key":"value"}'),
            _anthropic_text_delta("Result text"),
            _anthropic_message_delta(output_tokens=10),
        ]
        response_chunks: list[str] = []
        _replay_anthropic(
            client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)),
            response_chunks=response_chunks,
        )
        assert "".join(response_chunks) == "Result text"

    def test_thinking_collected_in_thinking_chunks(self):
        events = [
            _anthropic_message_start(),
            _anthropic_thinking_delta("Let me think..."),
            _anthropic_text_delta("The answer"),
            _anthropic_message_delta(output_tokens=10),
        ]
        thinking_chunks: list[str] = []
        response_chunks: list[str] = []
        _replay_anthropic(
            client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)),
            response_chunks=response_chunks,
            thinking_chunks=thinking_chunks,
        )
        assert "".join(thinking_chunks) == "Let me think..."
        assert "".join(response_chunks) == "The answer"


class TestAnthropicBetaHeader:
    """_build_headers must include anthropic-beta when provided."""

    def test_no_beta_header_when_not_provided(self):
        config = BenchmarkConfig(api_key="test-key")
        headers = _build_headers(config, upstream_api="anthropic")
        assert "anthropic-beta" not in headers
        assert headers["anthropic-version"] == "2023-06-01"

    def test_beta_header_included_when_provided(self):
        config = BenchmarkConfig(api_key="test-key")
        headers = _build_headers(
            config,
            upstream_api="anthropic",
            anthropic_beta="prompt-caching-2024-07-31,extended-thinking-2025-04-14",
        )
        assert headers["anthropic-beta"] == "prompt-caching-2024-07-31,extended-thinking-2025-04-14"
        assert headers["anthropic-version"] == "2023-06-01"

    def test_beta_header_ignored_for_openai(self):
        config = BenchmarkConfig(api_key="test-key")
        headers = _build_headers(
            config,
            upstream_api="openai",
            anthropic_beta="extended-thinking-2025-04-14",
        )
        assert "anthropic-beta" not in headers
        assert "anthropic-version" not in headers

    def test_anthropic_api_key_via_x_api_key(self):
        config = BenchmarkConfig(api_key="sk-ant-123", api_key_header="Authorization")
        headers = _build_headers(config, upstream_api="anthropic")
        assert headers["x-api-key"] == "sk-ant-123"
        assert "Authorization" not in headers

    def test_anthropic_custom_key_header(self):
        config = BenchmarkConfig(api_key="custom-key", api_key_header="x-custom-key")
        headers = _build_headers(config, upstream_api="anthropic")
        assert headers["x-custom-key"] == "custom-key"


class TestAnthropicRetryLogic:
    """Anthropic path should handle retries the same as OAI."""

    def test_http_429_retries(self):
        call_count = 0

        class RetryClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return FakeAnthropicStreamResponse(
                        [], status_code=429, body=b"rate limited"
                    )
                return FakeAnthropicStreamResponse([
                    _anthropic_message_start(),
                    _anthropic_text_delta("success"),
                    _anthropic_message_delta(output_tokens=5),
                ])

        with patch("agentic_swarm_bench.scenarios.player.asyncio.sleep", new_callable=AsyncMock):
            m = _replay_anthropic(client=RetryClient(), max_retries=2)
        assert call_count == 2
        assert m.error is None

    def test_http_500_not_retried(self):
        call_count = 0

        class Error500Client:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return FakeAnthropicStreamResponse(
                    [], status_code=500, body=b"internal error"
                )

        m = _replay_anthropic(client=Error500Client(), max_retries=3)
        assert call_count == 1
        assert "500" in m.error


class TestAnthropicMessageDeltaUsage:
    """message_delta output_tokens should override estimated token count."""

    def test_output_tokens_overrides_estimate(self):
        events = [
            _anthropic_message_start(input_tokens=200),
            _anthropic_text_delta("short"),
            _anthropic_message_delta(output_tokens=42),
        ]
        m = _replay_anthropic(client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)))
        assert m.completion_tokens == 42
        assert m.prompt_tokens == 200

    def test_prompt_tokens_include_cache_fields(self):
        start_event = _anthropic_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "usage": {
                    "input_tokens": 50,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 25,
                },
            },
        })
        events = [
            start_event,
            _anthropic_text_delta("ok"),
            _anthropic_message_delta(output_tokens=5),
        ]
        m = _replay_anthropic(client=FakeAnthropicClient(FakeAnthropicStreamResponse(events)))
        assert m.prompt_tokens == 175  # 50 + 100 + 25
