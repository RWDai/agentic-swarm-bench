"""Tests for the Claude Code agent orchestration in runner/claude_code.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import respx

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.runner.claude_code import (
    AgentTaskResult,
    _agent_report_payload,
    _build_agent_command,
    _cleanup_workdir,
    _duration_stats,
    _generate_agent_markdown_report,
    _preflight_check,
    _save_agent_outputs,
    _start_proxy,
    _stop_proxy,
)
from agentic_swarm_bench.scenarios.schedule import Schedule


def _dict_value(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _list_value(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _parse_json_payload(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(starts):
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue

        trailing = text[start + end :].strip()
        if trailing:
            continue
        return _dict_value(payload)
    raise AssertionError("No complete JSON payload found in captured output")


def test_build_agent_command_uses_claude_print_by_default():
    cfg = BenchmarkConfig(model="test-model", proxy_port=19000)

    assert _build_agent_command("claude", "do task", config=cfg) == [
        "claude",
        "--print",
        "do task",
    ]


def test_build_agent_command_uses_codex_exec_with_proxy_config():
    cfg = BenchmarkConfig(model="test-model", proxy_port=19000)

    assert _build_agent_command("codex", "do task", config=cfg) == [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "-c",
        'openai_base_url="http://localhost:19000/v1"',
        "-c",
        'model="test-model"',
        "do task",
    ]


def test_build_agent_command_appends_prompt_to_custom_command():
    cfg = BenchmarkConfig(model="test-model", proxy_port=19000)

    assert _build_agent_command("my-agent --flag 'two words'", "do task", config=cfg) == [
        "my-agent",
        "--flag",
        "two words",
        "do task",
    ]

# ---------------------------------------------------------------------------
# _preflight_check
# ---------------------------------------------------------------------------


async def test_preflight_404_returns_false():
    with respx.mock:
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(404)
        )
        result = await _preflight_check("http://fake:8000", "openai")
    assert result is False


async def test_preflight_200_returns_true():
    with respx.mock:
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        result = await _preflight_check("http://fake:8000", "openai")
    assert result is True


async def test_preflight_400_returns_true():
    """Any non-404 response means the path exists; preflight passes."""
    with respx.mock:
        respx.post("http://fake:8000/v1/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": "bad model"})
        )
        result = await _preflight_check("http://fake:8000", "openai")
    assert result is True


async def test_preflight_connect_error_returns_false():
    with respx.mock:
        respx.post("http://unreachable:9999/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        result = await _preflight_check("http://unreachable:9999", "openai")
    assert result is False


async def test_preflight_anthropic_path():
    with respx.mock:
        respx.post("http://fake:8000/v1/messages").mock(
            return_value=httpx.Response(200, json={})
        )
        result = await _preflight_check("http://fake:8000", "anthropic")
    assert result is True


async def test_preflight_404_anthropic_returns_false():
    with respx.mock:
        respx.post("http://fake:8000/v1/messages").mock(
            return_value=httpx.Response(404)
        )
        result = await _preflight_check("http://fake:8000", "anthropic")
    assert result is False


# ---------------------------------------------------------------------------
# _cleanup_workdir
# ---------------------------------------------------------------------------


def test_cleanup_removes_logs_when_keep_false(tmp_path):
    log = tmp_path / "slot0_p1_r0.log"
    log.write_text("some log")
    task_dir = tmp_path / "slot0_p1_r0"
    task_dir.mkdir()

    _cleanup_workdir(tmp_path, keep_logs=False)

    assert not log.exists()
    assert not task_dir.exists()


def test_cleanup_keeps_logs_when_keep_true(tmp_path):
    log = tmp_path / "slot0_p1_r0.log"
    log.write_text("some log")
    task_dir = tmp_path / "slot0_p1_r0"
    task_dir.mkdir()

    _cleanup_workdir(tmp_path, keep_logs=True)

    assert log.exists()
    assert task_dir.exists()


def test_cleanup_keeps_non_empty_task_dirs(tmp_path):
    task_dir = tmp_path / "slot0_p1_r0"
    task_dir.mkdir()
    (task_dir / "output.txt").write_text("data")

    _cleanup_workdir(tmp_path, keep_logs=False)

    assert task_dir.exists()


def test_cleanup_keeps_metrics_and_summary(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("{}")
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "slot0_p1_r0.log").write_text("log")

    _cleanup_workdir(tmp_path, keep_logs=False)

    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert not (tmp_path / "slot0_p1_r0.log").exists()


# ---------------------------------------------------------------------------
# _stop_proxy (now async; wraps asyncio.subprocess.Process)
# ---------------------------------------------------------------------------


async def test_stop_proxy_terminates_proc():
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)
    await _stop_proxy(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_awaited_once()


async def test_stop_proxy_skips_when_already_exited():
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    await _stop_proxy(mock_proc)
    mock_proc.terminate.assert_not_called()


async def test_stop_proxy_kills_on_timeout():
    """If proxy doesn't exit within 5s, _stop_proxy must SIGKILL it."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)

    async def _fake_wait_for(coro, timeout):
        # Close the coroutine we were handed so it never produces an
        # "unawaited coroutine" warning, then simulate the timeout.
        coro.close()
        raise asyncio.TimeoutError()

    with patch(
        "agentic_swarm_bench.runner.claude_code.asyncio.wait_for",
        new=_fake_wait_for,
    ):
        await _stop_proxy(mock_proc)

    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# _start_proxy (now async)
# ---------------------------------------------------------------------------


async def test_start_proxy_returns_none_when_fastapi_missing():
    config = BenchmarkConfig(endpoint="http://fake:8000", model="test", proxy_port=19001)
    # Setting a sys.modules key to None causes `import` of that name to raise ImportError.
    with patch.dict("sys.modules", {"agentic_swarm_bench.proxy.server": None}):
        result = await _start_proxy(config)

    assert result is None


# ---------------------------------------------------------------------------
# Agent task completion reporting
# ---------------------------------------------------------------------------


def test_duration_stats_includes_p90_and_p95():
    results = [
        AgentTaskResult(task_id=f"P{i}", task_tier="medium", elapsed_s=float(i))
        for i in range(1, 6)
    ]

    stats = _duration_stats(results)

    assert stats["count"] == 5
    assert stats["p50"] == 3.0
    assert stats["p90"] == 4.6
    assert stats["p95"] == 4.8


def test_agent_report_payload_groups_completion_by_tier_task_and_tag(tmp_path):
    results = [
        AgentTaskResult(
            task_id="P1",
            task_tier="trivial",
            task_tags=["python", "basics"],
            elapsed_s=1.0,
            returncode=0,
            stdout_chars=10,
        ),
        AgentTaskResult(
            task_id="P2",
            task_tier="easy",
            task_tags=["python", "cli"],
            elapsed_s=9.0,
            returncode=0,
            stdout_chars=20,
        ),
    ]

    payload = _agent_report_payload(
        config=BenchmarkConfig(endpoint="http://fake:8000", model="test-model"),
        agent_cmd="claude",
        schedule=Schedule(repetitions=1, max_concurrent=2, policy="round_robin"),
        task_results=results,
        proxy_summary={"total_requests": 3, "streaming_requests": 3},
        workdir=tmp_path,
    )

    assert payload["model"] == "test-model"
    assert payload["users"] == 2
    completion = _dict_value(payload["completion_time_s"])
    assert completion["p50"] == 5.0
    assert completion["p90"] == 8.2

    tiers = [_dict_value(row) for row in _list_value(payload["by_tier"])]
    tasks = [_dict_value(row) for row in _list_value(payload["by_task"])]
    tags = [_dict_value(row) for row in _list_value(payload["by_tag"])]
    assert {row["name"] for row in tiers} == {"trivial", "easy"}
    assert {row["name"] for row in tasks} == {"P1", "P2"}
    assert "python" in {row["name"] for row in tags}


def test_agent_markdown_report_contains_completion_percentiles(tmp_path):
    payload = _agent_report_payload(
        config=BenchmarkConfig(endpoint="http://fake:8000", model="test-model"),
        agent_cmd="claude",
        schedule=Schedule(repetitions=1, max_concurrent=1, policy="round_robin"),
        task_results=[
            AgentTaskResult(
                task_id="P1",
                task_tier="trivial",
                task_tags=["python"],
                elapsed_s=2.0,
                returncode=0,
                stdout_chars=10,
            )
        ],
        proxy_summary={
            "total_requests": 1,
            "streaming_requests": 1,
            "ttft_ms": {"median": 100, "p95": 150},
        },
        workdir=tmp_path,
    )

    report = _generate_agent_markdown_report(payload, json_path="agent.json")

    assert "Task Completion Time" in report
    assert "P90" in report
    assert "P95" in report
    assert "By Task Tier" in report
    assert "LLM Request Summary" in report


def test_save_agent_outputs_writes_markdown_and_json(tmp_path):
    payload = _agent_report_payload(
        config=BenchmarkConfig(endpoint="http://fake:8000", model="test-model"),
        agent_cmd="claude",
        schedule=Schedule(repetitions=1, max_concurrent=1, policy="round_robin"),
        task_results=[
            AgentTaskResult(
                task_id="P1",
                task_tier="trivial",
                elapsed_s=1.0,
                returncode=0,
                stdout_chars=10,
            )
        ],
        proxy_summary=None,
        workdir=tmp_path,
    )

    output = tmp_path / "agent-report.md"
    _save_agent_outputs(str(output), payload)

    assert output.exists()
    assert (tmp_path / "agent-report.json").exists()


def test_save_agent_outputs_path_matrix(tmp_path):
    payload = _agent_report_payload(
        config=BenchmarkConfig(endpoint="http://fake:8000", model="test-model"),
        agent_cmd="claude",
        schedule=Schedule(repetitions=1, max_concurrent=1, policy="round_robin"),
        task_results=[
            AgentTaskResult(
                task_id="P1",
                task_tier="trivial",
                elapsed_s=1.0,
                returncode=0,
                stdout_chars=10,
            )
        ],
        proxy_summary=None,
        workdir=tmp_path,
    )

    cases = [
        ("agent-report.json", True),
        ("agent-report.md", True),
        ("agent-report", False),
    ]
    for output_path, expect_markdown in cases:
        output = tmp_path / output_path
        _save_agent_outputs(str(output), payload)

        if expect_markdown:
            assert output.exists()
        else:
            assert not output.exists()

        if output.suffix == ".json":
            json_path = output
        elif output.suffix == ".md":
            json_path = output.with_suffix(".json")
        else:
            json_path = Path(str(output) + ".json")
        assert json_path.exists()
        assert json.loads(json_path.read_text()) == payload
