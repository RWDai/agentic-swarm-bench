"""Recording proxy server for agentic benchmarks.

Sits between an agent (Claude Code, etc.) and an OpenAI-compatible endpoint.
Translates Anthropic Messages API to OpenAI Chat Completions API, records timing.

Usage:
    python -m agentic_swarm_bench.proxy.server --upstream http://localhost:8000 --port 19000
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from agentic_swarm_bench.proxy.context import pad_messages_to_target
from agentic_swarm_bench.proxy.translators import (
    anthropic_to_openai,
    make_anthropic_stream_events,
    openai_to_anthropic_response,
)
from agentic_swarm_bench.proxy.utils import _detect_upstream_api, _strip_api_suffix

JsonDict = dict[str, Any]

uvicorn: Any = None
FastAPI: Any = None
JSONResponse: Any = None
Response: Any = None
StreamingResponse: Any = None

if TYPE_CHECKING:
    from fastapi import Request as FastAPIRequest
else:
    FastAPIRequest = Any

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi import Request as FastAPIRequest
    from fastapi.responses import JSONResponse, Response, StreamingResponse
except ImportError:
    pass

HAS_FASTAPI = uvicorn is not None


def create_app(
    upstream_url: str,
    model: str,
    api_key: str = "",
    api_key_header: str = "Authorization",
    context_target_tokens: int = 0,
    log_dir: str = "./traces",
    upstream_api: str | None = None,
) -> Any:
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI and uvicorn are required for the proxy. "
            "Install with: pip install agentic-swarm-bench[proxy]"
        )

    app = FastAPI(title="agentic-swarm-bench Recording Proxy")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_log = log_path / "metrics.jsonl"

    is_anthropic_upstream = _detect_upstream_api(upstream_url, upstream_api) == "anthropic"

    state = {"request_counter": 0}

    def _upstream_headers() -> JsonDict:
        headers = {"Content-Type": "application/json"}
        if not api_key:
            return headers
        if api_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers[api_key_header] = api_key
        return headers

    @app.get("/benchmark/metrics")
    async def get_metrics():
        if not metrics_log.exists():
            return {"metrics": []}
        lines = metrics_log.read_text().strip().split("\n")
        return {"metrics": [json.loads(line) for line in lines if line]}

    @app.get("/benchmark/summary")
    async def get_summary():
        if not metrics_log.exists():
            return {"error": "No metrics yet"}
        lines = metrics_log.read_text().strip().split("\n")
        entries = [json.loads(line) for line in lines if line]
        streaming = [e for e in entries if e.get("stream")]
        if not streaming:
            return {"total_requests": len(entries), "streaming_requests": 0}

        def _stats(vals):
            if not vals:
                return {}
            vals = sorted(vals)
            n = len(vals)
            return {
                "count": n,
                "min": round(vals[0], 2),
                "max": round(vals[-1], 2),
                "mean": round(sum(vals) / n, 2),
                "median": round(vals[n // 2], 2),
                "p95": round(vals[int(n * 0.95)], 2) if n > 1 else round(vals[0], 2),
            }

        return {
            "total_requests": len(entries),
            "streaming_requests": len(streaming),
            "ttft_ms": _stats([e["ttft_ms"] for e in streaming if e.get("ttft_ms")]),
            "tok_per_sec": _stats([e["tok_per_sec"] for e in streaming if e.get("tok_per_sec")]),
            "prefill_tok_per_sec": _stats(
                [e["prefill_tok_per_sec"] for e in streaming if e.get("prefill_tok_per_sec")]
            ),
        }

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def proxy(request: FastAPIRequest, path: str):
        state["request_counter"] += 1
        req_id = state["request_counter"]

        body = await request.body()
        body_json = None
        is_streaming = False
        clean_path = path.rstrip("/")
        is_messages_api = clean_path in ("v1/messages",)
        is_openai_chat_api = clean_path in ("v1/chat/completions", "chat/completions")
        is_openai_responses_api = clean_path in ("v1/responses", "responses")

        try:
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False)
        except Exception:
            pass

        t_start = time.perf_counter()

        if is_messages_api and body_json and is_anthropic_upstream:
            return await _handle_anthropic_passthrough(
                request, req_id, body_json, is_streaming, t_start, metrics_log,
            )

        if is_messages_api and body_json:
            return await _handle_messages(
                req_id,
                body_json,
                is_streaming,
                t_start,
                upstream_url,
                model,
                api_key,
                context_target_tokens,
                metrics_log,
                log_path,
            )

        if is_openai_chat_api and body_json:
            return await _handle_openai_chat(
                req_id,
                body_json,
                is_streaming,
                t_start,
                upstream_url,
                model,
                api_key,
                metrics_log,
            )

        if is_openai_responses_api and body_json:
            return await _handle_openai_responses(
                req_id,
                body_json,
                is_streaming,
                t_start,
                upstream_url,
                model,
                api_key,
                metrics_log,
            )

        target = f"{upstream_url}/{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=body,
            )

        fwd_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
        }
        return Response(content=resp.content, status_code=resp.status_code, headers=fwd_headers)

    _ANTHROPIC_FORWARD_HEADERS = {"anthropic-version", "anthropic-beta"}

    async def _handle_anthropic_passthrough(
        request: FastAPIRequest, req_id: int, body_json: JsonDict, is_streaming: bool,
        t_start: float, metrics_log_path: Path,
    ):
        """Forward Anthropic requests natively to an Anthropic upstream, capture metrics."""
        target_url = _strip_api_suffix(upstream_url) + "/v1/messages"
        headers = _upstream_headers()
        for h in _ANTHROPIC_FORWARD_HEADERS:
            val = request.headers.get(h)
            if val:
                headers[h] = val
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"

        if model:
            body_json["model"] = model

        metrics: JsonDict = {
            "req_id": req_id,
            "stream": is_streaming,
            "timestamp": datetime.now().isoformat(),
        }

        if not is_streaming:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                resp = await client.post(target_url, json=body_json, headers=headers)
            t_end = time.perf_counter()
            metrics["total_time_s"] = round(t_end - t_start, 3)
            if resp.status_code != 200:
                metrics["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
                print(f"[proxy] ERROR: Upstream returned HTTP {resp.status_code}")
            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            fwd_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            }
            return Response(content=resp.content, status_code=resp.status_code, headers=fwd_headers)

        async def _stream_anthropic():
            ttft = None
            token_count = 0
            first_time = None
            last_time = None
            upstream_error = None

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    "POST", target_url, json=body_json, headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        error_body = body.decode(errors='replace')[:500]
                        upstream_error = f"HTTP {resp.status_code}: {error_body}"
                        print(f"[proxy] ERROR: Upstream returned {upstream_error}")
                        yield body
                    else:
                        buf = b""
                        async for chunk_bytes in resp.aiter_bytes():
                            yield chunk_bytes

                            buf += chunk_bytes
                            while b"\n" in buf:
                                raw_line, buf = buf.split(b"\n", 1)
                                line = raw_line.decode("utf-8", errors="replace").strip()
                                if not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                try:
                                    data_obj = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue

                                now = time.perf_counter()
                                event_type = data_obj.get("type", "")

                                if event_type == "content_block_delta":
                                    delta = data_obj.get("delta", {})
                                    delta_type = delta.get("type", "")
                                    dt = delta_type
                                    is_text = dt == "text_delta" and delta.get("text")
                                    is_json = dt == "input_json_delta" and delta.get("partial_json")
                                    is_think = dt == "thinking_delta" and delta.get("thinking")
                                    has_content = is_text or is_json or is_think
                                    if has_content:
                                        if first_time is None:
                                            first_time = now
                                            ttft = (now - t_start) * 1000
                                        last_time = now
                                        token_count += 1

                                elif event_type == "message_delta":
                                    usage = data_obj.get("usage", {})
                                    if usage.get("output_tokens"):
                                        token_count = usage["output_tokens"]

                                elif event_type == "message_start":
                                    msg = data_obj.get("message", {})
                                    usage = msg.get("usage", {})
                                    metrics["input_tokens_actual"] = (
                                        usage.get("input_tokens", 0)
                                        + usage.get("cache_read_input_tokens", 0)
                                        + usage.get("cache_creation_input_tokens", 0)
                                    )

            t_end = time.perf_counter()
            metrics["ttft_ms"] = round(ttft, 2) if ttft else None
            metrics["total_time_s"] = round(t_end - t_start, 3)
            metrics["output_tokens"] = token_count
            if upstream_error:
                metrics["error"] = upstream_error

            if first_time and last_time and token_count > 1:
                decode_time = last_time - first_time
                metrics["decode_time_s"] = round(decode_time, 3)
                if decode_time > 0:
                    metrics["tok_per_sec"] = round(token_count / decode_time, 2)
                else:
                    metrics["tok_per_sec"] = 0
            else:
                metrics["tok_per_sec"] = 0

            input_for_prefill = metrics.get("input_tokens_actual", 0)
            if ttft and input_for_prefill:
                metrics["prefill_tok_per_sec"] = round(input_for_prefill / (ttft / 1000), 2)

            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        if StreamingResponse is None:
            raise ImportError("FastAPI response classes are unavailable")
        return StreamingResponse(_stream_anthropic(), media_type="text/event-stream")

    return app


async def _handle_openai_chat(
    req_id: int,
    body_json: JsonDict,
    is_streaming: bool,
    t_start: float,
    upstream_url: str,
    model: str,
    api_key: str,
    metrics_log: Path,
):
    """Handle /v1/chat/completions pass-through and record OpenAI metrics."""
    if model:
        body_json["model"] = model

    upstream = f"{_strip_api_suffix(upstream_url)}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not is_streaming:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            response = await client.post(upstream, json=body_json, headers=headers)
        t_end = time.perf_counter()
        metrics: JsonDict = {
            "req_id": req_id,
            "stream": False,
            "timestamp": datetime.now().isoformat(),
            "total_time_s": round(t_end - t_start, 3),
        }
        try:
            payload = response.json()
            usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            usage = {}
        if usage.get("prompt_tokens"):
            metrics["input_tokens_actual"] = usage["prompt_tokens"]
        if usage.get("completion_tokens"):
            metrics["output_tokens"] = usage["completion_tokens"]
        if response.status_code != 200:
            metrics["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
            print(f"[proxy] ERROR: OpenAI upstream returned HTTP {response.status_code}")
        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        fwd_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
        }
        if Response is None:
            raise ImportError("FastAPI response classes are unavailable")
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=fwd_headers,
        )

    metrics: JsonDict = {
        "req_id": req_id,
        "stream": True,
        "timestamp": datetime.now().isoformat(),
    }

    async def _stream_openai():
        ttft = None
        token_count = 0
        first_content_time = None
        last_content_time = None
        input_tokens_actual = 0
        output_tokens_actual = 0
        upstream_error = None

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", upstream, json=body_json, headers=headers) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    upstream_error = (
                        f"HTTP {response.status_code}: "
                        f"{body.decode(errors='replace')[:500]}"
                    )
                    print(f"[proxy] ERROR: OpenAI upstream returned {upstream_error}")
                    yield body
                else:
                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        buffer += chunk.decode(errors="replace")
                        lines = buffer.splitlines(keepends=True)
                        if lines and not lines[-1].endswith(("\n", "\r")):
                            buffer = lines.pop()
                        else:
                            buffer = ""
                        for raw_line in lines:
                            line = raw_line.strip()
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            usage = event.get("usage")
                            if usage:
                                input_tokens_actual = usage.get(
                                    "prompt_tokens", input_tokens_actual,
                                )
                                output_tokens_actual = usage.get(
                                    "completion_tokens", output_tokens_actual,
                                )

                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content") or delta.get("reasoning_content")
                                if not content:
                                    continue
                                now = time.perf_counter()
                                if first_content_time is None:
                                    first_content_time = now
                                    ttft = (now - t_start) * 1000
                                last_content_time = now
                                token_count += 1

        if output_tokens_actual > 0:
            token_count = output_tokens_actual

        t_end = time.perf_counter()
        metrics["ttft_ms"] = round(ttft, 2) if ttft else None
        metrics["total_time_s"] = round(t_end - t_start, 3)
        metrics["output_tokens"] = token_count
        if upstream_error:
            metrics["error"] = upstream_error
        if input_tokens_actual:
            metrics["input_tokens_actual"] = input_tokens_actual

        if first_content_time and last_content_time and token_count > 1:
            decode_time = last_content_time - first_content_time
            metrics["decode_time_s"] = round(decode_time, 3)
            metrics["tok_per_sec"] = round(token_count / decode_time, 2) if decode_time > 0 else 0
        else:
            metrics["tok_per_sec"] = 0

        input_for_prefill = input_tokens_actual or 0
        if ttft and input_for_prefill:
            metrics["prefill_tok_per_sec"] = round(input_for_prefill / (ttft / 1000), 2)

        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    if StreamingResponse is None:
        raise ImportError("FastAPI response classes are unavailable")
    return StreamingResponse(_stream_openai(), media_type="text/event-stream")


async def _handle_openai_responses(
    req_id: int,
    body_json: JsonDict,
    is_streaming: bool,
    t_start: float,
    upstream_url: str,
    model: str,
    api_key: str,
    metrics_log: Path,
):
    """Handle Codex /v1/responses requests through a chat-completions upstream.

    Codex CLI v0.130 speaks OpenAI Responses API. Most ASB targets are
    OpenAI-compatible chat-completions servers, so the proxy translates the
    subset Codex needs for text/tool turns and emits Responses-shaped SSE.
    """
    chat_body = _responses_to_chat_completions(body_json, model)
    upstream = f"{_strip_api_suffix(upstream_url)}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response_id = "resp_" + uuid.uuid4().hex

    if not is_streaming:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            response = await client.post(upstream, json=chat_body, headers=headers)
        t_end = time.perf_counter()
        metrics = _openai_response_metrics(req_id, False, t_start, t_end, response)
        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if response.status_code != 200:
            if JSONResponse is None:
                raise ImportError("FastAPI response classes are unavailable")
            return JSONResponse(
                content=_responses_error_payload(response_id, response.status_code, response.text),
                status_code=502,
            )
        payload = response.json()
        if JSONResponse is None:
            raise ImportError("FastAPI response classes are unavailable")
        return JSONResponse(content=_chat_completion_to_responses(payload, response_id))

    metrics: JsonDict = {
        "req_id": req_id,
        "stream": True,
        "timestamp": datetime.now().isoformat(),
    }

    async def _stream_responses():
        ttft = None
        token_count = 0
        first_content_time = None
        last_content_time = None
        input_tokens_actual = 0
        output_tokens_actual = 0
        upstream_error = None
        text_parts: list[str] = []
        tool_calls: dict[int, JsonDict] = {}

        yield _responses_sse(
            "response.created",
            {"type": "response.created", "response": {"id": response_id}},
        )

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", upstream, json=chat_body, headers=headers) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    upstream_error = (
                        f"HTTP {response.status_code}: "
                        f"{body.decode(errors='replace')[:500]}"
                    )
                    print(f"[proxy] ERROR: OpenAI upstream returned {upstream_error}")
                    yield _responses_sse(
                        "response.failed",
                        _responses_error_payload(response_id, response.status_code, upstream_error),
                    )
                else:
                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        buffer += chunk.decode(errors="replace")
                        lines = buffer.splitlines(keepends=True)
                        if lines and not lines[-1].endswith(("\n", "\r")):
                            buffer = lines.pop()
                        else:
                            buffer = ""
                        for raw_line in lines:
                            line = raw_line.strip()
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            usage = event.get("usage")
                            if usage:
                                input_tokens_actual = usage.get(
                                    "prompt_tokens", input_tokens_actual,
                                )
                                output_tokens_actual = usage.get(
                                    "completion_tokens", output_tokens_actual,
                                )

                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content") or delta.get("reasoning_content")
                                if content:
                                    now = time.perf_counter()
                                    if first_content_time is None:
                                        first_content_time = now
                                        ttft = (now - t_start) * 1000
                                    last_content_time = now
                                    token_count += 1
                                    text_parts.append(content)
                                    yield _responses_sse(
                                        "response.output_text.delta",
                                        {"type": "response.output_text.delta", "delta": content},
                                    )

                                for tc_delta in delta.get("tool_calls", []):
                                    idx = int(tc_delta.get("index", 0))
                                    tool_call = tool_calls.setdefault(
                                        idx,
                                        {
                                            "type": "function_call",
                                            "call_id": (
                                                tc_delta.get("id")
                                                or f"call_{uuid.uuid4().hex[:8]}"
                                            ),
                                            "name": "",
                                            "arguments": "",
                                        },
                                    )
                                    if tc_delta.get("id"):
                                        tool_call["call_id"] = tc_delta["id"]
                                    function_delta = tc_delta.get("function", {})
                                    if function_delta.get("name"):
                                        tool_call["name"] = function_delta["name"]
                                    if function_delta.get("arguments"):
                                        tool_call["arguments"] += function_delta["arguments"]

        if output_tokens_actual > 0:
            token_count = output_tokens_actual

        text = "".join(text_parts)
        if text:
            message_id = "msg_" + uuid.uuid4().hex[:24]
            yield _responses_sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "id": message_id,
                        "content": [{"type": "output_text", "text": text}],
                    },
                },
            )
        for tool_call in tool_calls.values():
            yield _responses_sse(
                "response.output_item.done",
                {"type": "response.output_item.done", "item": tool_call},
            )
        yield _responses_sse("response.completed", _responses_completed_payload(response_id))

        t_end = time.perf_counter()
        metrics["ttft_ms"] = round(ttft, 2) if ttft else None
        metrics["total_time_s"] = round(t_end - t_start, 3)
        metrics["output_tokens"] = token_count
        if upstream_error:
            metrics["error"] = upstream_error
        if input_tokens_actual:
            metrics["input_tokens_actual"] = input_tokens_actual
        if first_content_time and last_content_time and token_count > 1:
            decode_time = last_content_time - first_content_time
            metrics["decode_time_s"] = round(decode_time, 3)
            metrics["tok_per_sec"] = round(token_count / decode_time, 2) if decode_time > 0 else 0
        else:
            metrics["tok_per_sec"] = 0
        if ttft and input_tokens_actual:
            metrics["prefill_tok_per_sec"] = round(input_tokens_actual / (ttft / 1000), 2)
        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    if StreamingResponse is None:
        raise ImportError("FastAPI response classes are unavailable")
    return StreamingResponse(_stream_responses(), media_type="text/event-stream")


def _responses_to_chat_completions(body: JsonDict, model: str) -> JsonDict:
    messages: list[JsonDict] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    for item in body.get("input", []):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            role = str(item.get("role", "user"))
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            messages.append({"role": role, "content": _responses_content_text(item.get("content"))})
        elif item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": str(item.get("call_id", "")),
                "content": _responses_content_text(item.get("output")),
            })

    if not messages:
        messages.append({"role": "user", "content": ""})

    chat_body: JsonDict = {
        "model": model or str(body.get("model", "")),
        "messages": messages,
        "stream": bool(body.get("stream", True)),
        "stream_options": {"include_usage": True},
    }
    if body.get("max_output_tokens"):
        chat_body["max_tokens"] = body["max_output_tokens"]

    tools = _responses_tools_to_chat_tools(body.get("tools"))
    if tools:
        chat_body["tools"] = tools
    return chat_body


def _responses_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("output_text") or block.get("input_text")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _responses_tools_to_chat_tools(tools: object) -> list[JsonDict]:
    if not isinstance(tools, list):
        return []
    converted: list[JsonDict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _chat_completion_to_responses(payload: JsonDict, response_id: str) -> JsonDict:
    choice = payload.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    output: list[JsonDict] = []
    content = message.get("content")
    if content:
        output.append({
            "type": "message",
            "role": "assistant",
            "id": "msg_" + uuid.uuid4().hex[:24],
            "content": [{"type": "output_text", "text": content}],
        })
    for tc in message.get("tool_calls", []):
        function = tc.get("function", {})
        output.append({
            "type": "function_call",
            "call_id": tc.get("id", "call_" + uuid.uuid4().hex[:8]),
            "name": function.get("name", ""),
            "arguments": function.get("arguments", ""),
        })
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "output": output,
        "usage": _responses_usage(payload.get("usage", {})),
    }


def _openai_response_metrics(
    req_id: int, is_streaming: bool, t_start: float, t_end: float, response: httpx.Response,
) -> JsonDict:
    metrics: JsonDict = {
        "req_id": req_id,
        "stream": is_streaming,
        "timestamp": datetime.now().isoformat(),
        "total_time_s": round(t_end - t_start, 3),
    }
    try:
        payload = response.json()
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        usage = {}
    if usage.get("prompt_tokens"):
        metrics["input_tokens_actual"] = usage["prompt_tokens"]
    if usage.get("completion_tokens"):
        metrics["output_tokens"] = usage["completion_tokens"]
    if response.status_code != 200:
        metrics["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
        print(f"[proxy] ERROR: OpenAI upstream returned HTTP {response.status_code}")
    return metrics


def _responses_usage(usage: object) -> JsonDict:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens) or 0),
    }


def _responses_completed_payload(response_id: str) -> JsonDict:
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": None,
                "output_tokens": 0,
                "output_tokens_details": None,
                "total_tokens": 0,
            },
        },
    }


def _responses_error_payload(response_id: str, status_code: int, message: str) -> JsonDict:
    return {
        "type": "response.failed",
        "response": {
            "id": response_id,
            "error": {"code": f"http_{status_code}", "message": message[:500]},
        },
    }


def _responses_sse(event: str, payload: JsonDict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


async def _handle_messages(
    req_id,
    body_json,
    is_streaming,
    t_start,
    upstream_url,
    model,
    api_key,
    context_target_tokens,
    metrics_log,
    log_path,
):
    """Handle /v1/messages: translate Anthropic -> OpenAI, record metrics."""
    anthropic_model = body_json.get("model", "unknown")
    oai_body = anthropic_to_openai(body_json, model)

    if context_target_tokens > 0:
        oai_body["messages"] = pad_messages_to_target(
            oai_body["messages"],
            context_target_tokens,
        )

    upstream = f"{_strip_api_suffix(upstream_url)}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not is_streaming:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            response = await client.post(upstream, json=oai_body, headers=headers)
        t_end = time.perf_counter()
        metrics = {"req_id": req_id, "stream": False, "total_time_s": round(t_end - t_start, 3)}

        if response.status_code != 200:
            error_msg = (
                f"Upstream returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
            print(f"[proxy] ERROR: {error_msg}")
            metrics["error"] = error_msg
            anthropic_resp = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": (
                        f"Upstream at {upstream} returned HTTP {response.status_code}. "
                        f"The endpoint may not support OpenAI-compatible /v1/chat/completions."
                    ),
                },
            }
        else:
            oai_resp = response.json()
            if "choices" in oai_resp:
                anthropic_resp = openai_to_anthropic_response(oai_resp, anthropic_model)
            else:
                anthropic_resp = oai_resp

        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if JSONResponse is None:
            raise ImportError("FastAPI response classes are unavailable")
        return JSONResponse(
            content=anthropic_resp,
            status_code=200 if response.status_code == 200 else 502,
        )

    msg_id = "msg_" + uuid.uuid4().hex[:24]
    metrics: JsonDict = {
        "req_id": req_id,
        "stream": True,
        "timestamp": datetime.now().isoformat(),
    }

    async def _stream():
        ttft = None
        token_count = 0
        first_content_time = None
        last_content_time = None
        input_tokens_actual = 0
        output_tokens_actual = 0
        upstream_error = None

        for event in make_anthropic_stream_events(anthropic_model, msg_id):
            yield event.encode()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", upstream, json=oai_body, headers=headers) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    upstream_error = (
                        f"Upstream returned HTTP {response.status_code}: "
                        f"{body.decode(errors='replace')[:500]}"
                    )
                    print(f"[proxy] ERROR: {upstream_error}")
                    error_text = (
                        f"[Proxy error] Upstream at {upstream} returned HTTP "
                        f"{response.status_code}. The endpoint may not support "
                        f"OpenAI-compatible /v1/chat/completions."
                    )
                    delta_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": error_text},
                    }
                    sse = f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                    yield sse.encode()
                else:
                    async for chunk in response.aiter_lines():
                        if not chunk.startswith("data: "):
                            continue
                        data_str = chunk[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        now = time.perf_counter()

                        usage = event.get("usage")
                        if usage:
                            input_tokens_actual = usage.get(
                                "prompt_tokens", input_tokens_actual,
                            )
                            output_tokens_actual = usage.get(
                                "completion_tokens", output_tokens_actual,
                            )

                        for choice in event.get("choices", []):
                            content = choice.get("delta", {}).get("content")
                            if not content:
                                continue
                            if first_content_time is None:
                                first_content_time = now
                                ttft = (now - t_start) * 1000
                            last_content_time = now
                            token_count += 1

                            delta_event = {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": content},
                            }
                            sse = f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                            yield sse.encode()

        if output_tokens_actual > 0:
            token_count = output_tokens_actual

        block_stop = {"type": "content_block_stop", "index": 0}
        yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n".encode()

        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": token_count},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode()
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'.encode()

        t_end = time.perf_counter()
        metrics["ttft_ms"] = round(ttft, 2) if ttft else None
        metrics["total_time_s"] = round(t_end - t_start, 3)
        metrics["output_tokens"] = token_count
        if upstream_error:
            metrics["error"] = upstream_error
        if input_tokens_actual:
            metrics["input_tokens_actual"] = input_tokens_actual

        if first_content_time and last_content_time and token_count > 1:
            decode_time = last_content_time - first_content_time
            metrics["decode_time_s"] = round(decode_time, 3)
            metrics["tok_per_sec"] = round(token_count / decode_time, 2) if decode_time > 0 else 0
        else:
            metrics["tok_per_sec"] = 0

        input_for_prefill = input_tokens_actual or 0
        if ttft and input_for_prefill:
            metrics["prefill_tok_per_sec"] = round(input_for_prefill / (ttft / 1000), 2)

        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    if StreamingResponse is None:
        raise ImportError("FastAPI response classes are unavailable")
    return StreamingResponse(_stream(), media_type="text/event-stream")


def run_proxy(
    upstream_url: str = "http://localhost:8000",
    port: int = 19000,
    model: str = "default",
    api_key: str = "",
    api_key_header: str = "Authorization",
    context_target_tokens: int = 0,
    log_dir: str = "./traces",
    upstream_api: str | None = None,
):
    """Start the recording proxy server."""
    if not HAS_FASTAPI:
        raise ImportError("Install proxy deps: pip install agentic-swarm-bench[proxy]")

    detected_api = _detect_upstream_api(upstream_url, upstream_api)

    app = create_app(
        upstream_url=upstream_url,
        model=model,
        api_key=api_key,
        api_key_header=api_key_header,
        context_target_tokens=context_target_tokens,
        log_dir=log_dir,
        upstream_api=upstream_api,
    )
    print(f"agentic-swarm-bench proxy on :{port} -> {upstream_url}")
    print(f"  Model: {model}")
    print(f"  Upstream API: {detected_api}")
    print(f"  Context target: {context_target_tokens} tokens (0 = no padding)")
    print(f"  Traces: {log_dir}")
    print(f"  Metrics: http://localhost:{port}/benchmark/summary")
    if uvicorn is None:
        raise ImportError("uvicorn is unavailable")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
