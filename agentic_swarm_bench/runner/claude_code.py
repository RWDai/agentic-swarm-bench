"""Agent runner: orchestrates Claude Code (or similar) through the recording proxy.

The runner models the workload exactly like ``asb replay``:

    schedule-task = (task, execution_index)
        One CLI invocation of the agent on one task.

    With T tasks and R repetitions, there are T*R schedule-tasks. A
    Schedule (R, J, policy, seed) orders them into a single pending list
    L; then a pool of J parallel subprocess workers pulls items off L
    until drained. Nothing ever waits for a batch peer.

This matches Mike's work-queue model from docs/SCHEDULING.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
from rich.console import Console
from rich.table import Table

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.scenarios.schedule import (
    Schedule,
    build_execution_queue,
    run_work_queue,
)
from agentic_swarm_bench.tasks.registry import get_tasks, select_task_mix

JsonDict = dict[str, object]
DurationStats = dict[str, float | int]

console = Console()


@dataclass
class AgentTaskResult:
    """One real agent invocation on one benchmark task."""

    task_id: str
    task_tier: str
    task_tags: list[str] = field(default_factory=list)
    repetition_id: int = 0
    slot_id: int = 0
    elapsed_s: float = 0.0
    returncode: int | None = None
    timed_out: bool = False
    empty_stdout: bool = False
    stdout_chars: int = 0
    stderr_chars: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.error is None
            and not self.timed_out
            and not self.empty_stdout
            and self.returncode == 0
        )

    def to_dict(self) -> JsonDict:
        return {
            "task_id": self.task_id,
            "task_tier": self.task_tier,
            "task_tags": self.task_tags,
            "repetition_id": self.repetition_id,
            "slot_id": self.slot_id,
            "elapsed_s": round(self.elapsed_s, 3),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "empty_stdout": self.empty_stdout,
            "stdout_chars": self.stdout_chars,
            "stderr_chars": self.stderr_chars,
            "succeeded": self.succeeded,
            "error": self.error,
        }


async def run_agent_benchmark(
    config: BenchmarkConfig,
    agent_cmd: str = "claude",
    schedule: Schedule | None = None,
) -> None:
    """Run agentic benchmark: start proxy, feed a scheduled workload through agents.

    ``schedule`` controls (R, J, policy, seed). When omitted, defaults to
    one repetition per task, one worker at a time, sequential order --
    matching pre-3.2.0 behavior.
    """
    _runner_checked = False
    if not _runner_checked and not shutil.which(agent_cmd):
        console.print(
            f"[red]Error: '{agent_cmd}' not found in PATH.[/red]\n"
            f"Install Claude Code: npm install -g @anthropic-ai/claude-code"
        )
        return

    if schedule is None:
        schedule = Schedule(repetitions=1, max_concurrent=1, policy="sequential")

    tasks: list[JsonDict] = get_tasks(task_range=config.task_range)
    if not tasks:
        tasks = get_tasks(task_range="p1-p10")
    tasks = select_task_mix(
        tasks,
        mix=config.task_mix,
        count=config.task_count,
        seed=schedule.seed,
    )

    workdir = Path(tempfile.mkdtemp(prefix="agentic-swarm-bench-"))

    execution_queue = build_execution_queue(tasks, schedule)
    total_schedule_tasks = len(execution_queue)

    console.print("\n[bold]agentic-swarm-bench agent[/bold]")
    console.print(f"  Upstream: {config.endpoint}")
    console.print(f"  Model: {config.model}")
    console.print(f"  Agent: {agent_cmd}")
    console.print(f"  Proxy port: {config.proxy_port}")
    console.print(f"  Tasks: {len(tasks)}")
    if config.task_mix or config.task_count:
        console.print(f"  Task mix: {config.task_mix or 'balanced'}")
        if config.task_count:
            console.print(f"  Distinct task count: {config.task_count}")
    console.print(f"  Schedule: {schedule.policy}"
                  f" × {schedule.repetitions} reps"
                  f" (max {schedule.max_concurrent} parallel agents)")
    if schedule.seed is not None:
        console.print(f"  Seed: {schedule.seed}")
    console.print(f"  Total schedule-tasks: {total_schedule_tasks}")
    console.print(f"  Workdir: {workdir}")

    from agentic_swarm_bench.proxy.server import _detect_upstream_api

    detected_api = _detect_upstream_api(config.endpoint, config.upstream_api)
    console.print(f"  Upstream API: {detected_api}")

    proxy_proc = await _start_proxy(config, log_dir=str(workdir))
    if proxy_proc is None:
        return

    try:
        await asyncio.sleep(2)

        if not await _preflight_check(config.endpoint, detected_api):
            await _stop_proxy(proxy_proc)
            return

        env = os.environ.copy()
        _runner_configured_env = False
        if not _runner_configured_env:
            env["ANTHROPIC_BASE_URL"] = f"http://localhost:{config.proxy_port}"
            env["ANTHROPIC_AUTH_TOKEN"] = "agentic-swarm-bench"
            env["ANTHROPIC_MODEL"] = config.model
            env["CLAUDE_MODEL"] = config.model
            env["CODEX_API_KEY"] = "agentic-swarm-bench"
            env["OPENAI_API_KEY"] = "agentic-swarm-bench"
            env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        state = _AgentRunState()

        async def _run_schedule_task(
            sched_task: tuple[JsonDict, int],
            slot_id: int,
        ) -> AgentTaskResult:
            task, exec_idx = sched_task
            return await _run_one_agent_task(
                task=task,
                exec_idx=exec_idx,
                slot_id=slot_id,
                total=total_schedule_tasks,
                agent_cmd=agent_cmd,
                config=config,
                env=env,
                workdir=workdir,
                timeout=config.timeout,
                state=state,
            )

        task_results = await run_work_queue(
            execution_queue,
            _run_schedule_task,
            max_concurrent=schedule.max_concurrent,
        )

        _drain_proxy_stderr(proxy_proc, workdir)
        summary = await _fetch_and_save_summary(config.proxy_port, workdir)
        _print_task_completion_summary(task_results)
        _print_results(
            summary,
            workdir,
            empty_count=state.empty_count,
            total_tasks=total_schedule_tasks,
        )
        payload = _agent_report_payload(
            config=config,
            agent_cmd=agent_cmd,
            schedule=schedule,
            task_results=task_results,
            proxy_summary=summary,
            workdir=workdir,
        )
        if config.output:
            _save_agent_outputs(config.output, payload)
        if getattr(config, "json_stdout", False):
            sys.stdout.write(json.dumps(payload, indent=2))
            sys.stdout.write("\n")

        _cleanup_workdir(workdir, keep_logs=(state.empty_count > 0))

    finally:
        await _stop_proxy(proxy_proc)


class _AgentRunState:
    """Mutable counters shared across concurrent agent workers."""

    def __init__(self) -> None:
        self.empty_count = 0
        self.completed = 0
        self._lock = asyncio.Lock()

    async def record_completion(self, empty: bool) -> int:
        async with self._lock:
            self.completed += 1
            if empty:
                self.empty_count += 1
            return self.completed


async def _run_one_agent_task(
    *,
    task: JsonDict,
    exec_idx: int,
    slot_id: int,
    total: int,
    agent_cmd: str,
    config: BenchmarkConfig,
    env: dict[str, str],
    workdir: Path,
    timeout: float,
    state: _AgentRunState,
) -> AgentTaskResult:
    """Launch one agent subprocess for one schedule-task and capture its output."""
    task_id = str(task["id"])
    task_tier = str(task.get("tier", "unknown"))
    raw_tags = task.get("tags", [])
    task_tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
    prompt = str(task["prompt"])

    task_dir = workdir / f"slot{slot_id}_{task_id}_r{exec_idx}"
    task_dir.mkdir(parents=True, exist_ok=True)

    label = f"{task_id}#r{exec_idx}"
    preview = prompt[:70]
    console.print(f"\n  [slot {slot_id}] start {label}: {preview}...")

    t_start = time.perf_counter()
    _cmd_argv = _build_agent_command(agent_cmd, prompt, config=config)
    try:
        proc = await asyncio.create_subprocess_exec(
            *_cmd_argv,
            cwd=str(task_dir),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        console.print(f"    [red][slot {slot_id}] spawn failed: {e}[/red]")
        await state.record_completion(empty=True)
        return AgentTaskResult(
            task_id=task_id,
            task_tier=task_tier,
            task_tags=task_tags,
            repetition_id=exec_idx,
            slot_id=slot_id,
            elapsed_s=elapsed,
            empty_stdout=True,
            error=f"spawn failed: {e}",
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        returncode = proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        elapsed = time.perf_counter() - t_start
        console.print(
            f"    [slot {slot_id}] [yellow]{label} timed out after {elapsed:.0f}s[/yellow]"
        )
        await state.record_completion(empty=True)
        return AgentTaskResult(
            task_id=task_id,
            task_tier=task_tier,
            task_tags=task_tags,
            repetition_id=exec_idx,
            slot_id=slot_id,
            elapsed_s=elapsed,
            timed_out=True,
            empty_stdout=True,
            error=f"timed out after {elapsed:.0f}s",
        )

    elapsed = time.perf_counter() - t_start

    log_file = workdir / f"slot{slot_id}_{task_id}_r{exec_idx}.log"
    log_file.write_text(stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""))

    empty = not stdout.strip()
    completed = await state.record_completion(empty=empty)

    status = f"exit={returncode}"
    if empty and stderr.strip():
        preview_err = stderr.strip()[:150]
        console.print(
            f"    [slot {slot_id}] [{completed}/{total}] {label} {status} "
            f"{elapsed:.1f}s [dim yellow](empty stdout; stderr: {preview_err})[/dim yellow]"
        )
    elif empty:
        console.print(
            f"    [slot {slot_id}] [{completed}/{total}] {label} {status} "
            f"{elapsed:.1f}s [yellow](empty stdout)[/yellow]"
        )
    else:
        console.print(
            f"    [slot {slot_id}] [{completed}/{total}] {label} {status} {elapsed:.1f}s"
        )

    return AgentTaskResult(
        task_id=task_id,
        task_tier=task_tier,
        task_tags=task_tags,
        repetition_id=exec_idx,
        slot_id=slot_id,
        elapsed_s=elapsed,
        returncode=returncode,
        empty_stdout=empty,
        stdout_chars=len(stdout),
        stderr_chars=len(stderr),
        error=None if returncode == 0 else f"exit={returncode}",
    )


def _build_agent_command(
    agent_cmd: str,
    prompt: str,
    *,
    config: BenchmarkConfig,
) -> list[str]:
    """Build a non-interactive agent command for known agent CLIs."""
    parts = shlex.split(agent_cmd)
    if not parts:
        raise ValueError("agent_cmd must not be empty")

    executable = Path(parts[0]).name
    if executable == "codex" and len(parts) == 1:
        return [
            *parts,
            "exec",
            "-c",
            f'openai_base_url="http://localhost:{config.proxy_port}/v1"',
            "-c",
            f'model="{config.model}"',
            prompt,
        ]
    if executable == "claude" and len(parts) == 1:
        return [*parts, "--print", prompt]
    return [*parts, prompt]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 3)
    rank = percentile * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac, 3)


def _duration_stats(results: list[AgentTaskResult]) -> DurationStats:
    values = [r.elapsed_s for r in results]
    if not values:
        return {
            "count": 0,
            "min": 0,
            "mean": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "max": 0,
        }
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 3),
    }


def _success_summary(results: list[AgentTaskResult]) -> JsonDict:
    total = len(results)
    succeeded = sum(1 for r in results if r.succeeded)
    failed = total - succeeded
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": round(succeeded / total, 4) if total else 0,
    }


def _group_results(
    results: list[AgentTaskResult],
    key_fn: Callable[[AgentTaskResult], str],
) -> list[JsonDict]:
    groups: dict[str, list[AgentTaskResult]] = {}
    for result in results:
        key = key_fn(result) or "unknown"
        groups.setdefault(key, []).append(result)
    return [
        {
            "name": name,
            "completion_time_s": _duration_stats(items),
            "successful_completion_time_s": _duration_stats([r for r in items if r.succeeded]),
            "success": _success_summary(items),
        }
        for name, items in sorted(groups.items())
    ]


def _group_results_by_tag(results: list[AgentTaskResult]) -> list[JsonDict]:
    groups: dict[str, list[AgentTaskResult]] = {}
    for result in results:
        tags = result.task_tags or ["untagged"]
        for tag in tags:
            groups.setdefault(tag, []).append(result)
    return [
        {
            "name": name,
            "completion_time_s": _duration_stats(items),
            "successful_completion_time_s": _duration_stats([r for r in items if r.succeeded]),
            "success": _success_summary(items),
        }
        for name, items in sorted(groups.items())
    ]


def _agent_report_payload(
    *,
    config: BenchmarkConfig,
    agent_cmd: str,
    schedule: Schedule,
    task_results: list[AgentTaskResult],
    proxy_summary: JsonDict | None,
    workdir: Path,
) -> JsonDict:
    successful = [r for r in task_results if r.succeeded]
    return {
        "model": config.model,
        "endpoint": config.endpoint,
        "agent": agent_cmd,
        "users": schedule.max_concurrent,
        "schedule": {
            "repetitions": schedule.repetitions,
            "max_concurrent": schedule.max_concurrent,
            "policy": schedule.policy,
            "seed": schedule.seed,
        },
        "workdir": str(workdir),
        "success": _success_summary(task_results),
        "completion_time_s": _duration_stats(task_results),
        "successful_completion_time_s": _duration_stats(successful),
        "by_tier": _group_results(task_results, lambda r: r.task_tier),
        "by_task": _group_results(task_results, lambda r: r.task_id),
        "by_tag": _group_results_by_tag(task_results),
        "task_results": [r.to_dict() for r in task_results],
        "proxy_summary": proxy_summary,
    }


def _print_task_completion_summary(task_results: list[AgentTaskResult]) -> None:
    if not task_results:
        return

    stats = _duration_stats(task_results)
    success = _success_summary(task_results)
    console.print("\n[bold]Task completion time[/bold]")
    console.print(
        f"  {success['succeeded']}/{success['total']} succeeded  "
        f"p50 {stats['p50']:.1f}s  p90 {stats['p90']:.1f}s  p95 {stats['p95']:.1f}s"
    )

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Tier", style="bold")
    for col in ["Count", "OK", "P50", "P90", "P95", "Max"]:
        table.add_column(col, justify="right")
    for row in _group_results(task_results, lambda r: r.task_tier):
        data = _expect_duration_stats(row["completion_time_s"])
        row_success = _expect_json_dict(row["success"])
        ok = row_success["succeeded"]
        total = row_success["total"]
        table.add_row(
            str(row["name"]),
            str(data["count"]),
            f"{ok}/{total}",
            f"{data['p50']:.1f}s",
            f"{data['p90']:.1f}s",
            f"{data['p95']:.1f}s",
            f"{data['max']:.1f}s",
        )
    console.print(table)


def _save_agent_outputs(output: str, payload: JsonDict) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output.endswith(".json"):
        out_path.write_text(json.dumps(payload, indent=2))
        console.print(f"\n  [dim]Saved →[/dim]  {out_path}")
        return

    json_path = out_path.with_suffix(".json") if out_path.suffix else Path(str(out_path) + ".json")
    json_path.write_text(json.dumps(payload, indent=2))
    console.print(f"\n  [dim]Saved →[/dim]  {json_path}")

    if output.endswith(".md"):
        out_path.write_text(_generate_agent_markdown_report(payload, json_path=str(json_path)))
        console.print(f"  [dim]Saved →[/dim]  {out_path}")


def _generate_agent_markdown_report(payload: JsonDict, json_path: str | None = None) -> str:
    schedule = _expect_json_dict(payload["schedule"])
    lines = [
        "# AgenticSwarmBench Agent Report",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|:--|:--|",
        f"| Model | `{payload['model']}` |",
        f"| Endpoint | `{payload['endpoint']}` |",
        f"| Agent | `{payload['agent']}` |",
        f"| Users | {payload['users']} |",
        f"| Policy | `{schedule['policy']}` |",
        f"| Repetitions | {schedule['repetitions']} |",
        f"| Workdir | `{payload['workdir']}` |",
    ]
    if json_path:
        lines.append(f"| Raw results | `{json_path}` |")

    success = _expect_json_dict(payload["success"])
    completion = _expect_duration_stats(payload["completion_time_s"])
    lines.extend(
        [
            "",
            "## Task Completion Time",
            "",
            f"Completed **{success['succeeded']} / {success['total']}** schedule-tasks.",
            "",
            "| Scope | Count | OK | P50 | P90 | P95 | Max |",
            "|:--|--:|--:|--:|--:|--:|--:|",
            _markdown_completion_row("overall", completion, success),
        ]
    )

    by_tier = _expect_list(payload["by_tier"])
    if by_tier:
        lines.extend(["", "## By Task Tier", "", "| Tier | Count | OK | P50 | P90 | P95 | Max |"])
        lines.append("|:--|--:|--:|--:|--:|--:|--:|")
        for row_obj in by_tier:
            row = _expect_json_dict(row_obj)
            lines.append(
                _markdown_completion_row(
                    str(row["name"]),
                    _expect_duration_stats(row["completion_time_s"]),
                    _expect_json_dict(row["success"]),
                )
            )

    by_task = _expect_list(payload["by_task"])
    if by_task:
        lines.extend(["", "## By Task", "", "| Task | Count | OK | P50 | P90 | P95 | Max |"])
        lines.append("|:--|--:|--:|--:|--:|--:|--:|")
        for row_obj in by_task:
            row = _expect_json_dict(row_obj)
            lines.append(
                _markdown_completion_row(
                    str(row["name"]),
                    _expect_duration_stats(row["completion_time_s"]),
                    _expect_json_dict(row["success"]),
                )
            )

    proxy = _expect_json_dict(payload.get("proxy_summary") or {})
    if proxy and "error" not in proxy:
        lines.extend(["", "## LLM Request Summary", ""])
        lines.append(f"Total requests: **{proxy.get('total_requests', 0)}**")
        lines.append(f"Streaming requests: **{proxy.get('streaming_requests', 0)}**")
        for key, label in [
            ("ttft_ms", "TTFT (ms)"),
            ("tok_per_sec", "Decode tok/s"),
            ("prefill_tok_per_sec", "Prefill tok/s"),
        ]:
            data = proxy.get(key)
            if isinstance(data, dict):
                lines.append(
                    f"- {label}: median {data.get('median', 0)}, p95 {data.get('p95', 0)}"
                )

    lines.append("")
    return "\n".join(lines)


def _markdown_completion_row(name: str, stats: DurationStats, success: JsonDict) -> str:
    return (
        f"| `{name}` | {stats['count']} | {success['succeeded']}/{success['total']} | "
        f"{stats['p50']:.3f}s | {stats['p90']:.3f}s | {stats['p95']:.3f}s | "
        f"{stats['max']:.3f}s |"
    )


def _expect_json_dict(value: object) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _expect_duration_stats(value: object) -> DurationStats:
    if not isinstance(value, dict):
        return _duration_stats([])
    return {
        "count": int(value.get("count", 0)),
        "min": float(value.get("min", 0)),
        "mean": float(value.get("mean", 0)),
        "p50": float(value.get("p50", 0)),
        "p90": float(value.get("p90", 0)),
        "p95": float(value.get("p95", 0)),
        "max": float(value.get("max", 0)),
    }


def _expect_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


async def _preflight_check(endpoint: str, upstream_api: str) -> bool:
    """Verify the upstream endpoint is reachable and serves the expected API."""
    if upstream_api == "anthropic":
        url = f"{endpoint.rstrip('/')}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "preflight-check",
        }
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
    else:
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }

    console.print(f"\n  Pre-flight check: {url}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code == 404:
                alt = "openai" if upstream_api == "anthropic" else "anthropic"
                console.print(
                    f"  [bold red]FAILED: {url} returned 404[/bold red]\n"
                    f"  [red]The endpoint does not serve this path.\n"
                    f"  Try --upstream-api={alt} if the endpoint "
                    f"uses a different API format.[/red]"
                )
                return False
            console.print(f"  [green]OK[/green] (HTTP {resp.status_code})")
            return True
    except httpx.ConnectError:
        console.print(
            f"  [bold red]FAILED: Cannot connect to {endpoint}[/bold red]\n"
            f"  [red]Is the server running?[/red]"
        )
        return False
    except Exception as e:
        console.print(
            f"  [yellow]Warning: pre-flight check failed ({e}), "
            f"continuing anyway[/yellow]"
        )
        return True


def _drain_proxy_stderr(proxy_proc: asyncio.subprocess.Process, workdir: Path) -> None:
    """Read any buffered proxy stderr and save it for diagnostics."""
    # Avoid blocking on the live proxy stderr pipe. The proxy is stopped in
    # ``_stop_proxy`` after summaries are fetched; request metrics and summary
    # files remain the primary diagnostics.
    if proxy_proc.stderr is not None:
        (workdir / "proxy.log").touch(exist_ok=True)


async def _fetch_and_save_summary(proxy_port: int, workdir: Path) -> JsonDict | None:
    """Fetch the proxy summary while the proxy is still alive, and save to workdir."""
    url = f"http://localhost:{proxy_port}/benchmark/summary"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                summary = _expect_json_dict(resp.json())
                summary_path = workdir / "summary.json"
                summary_path.write_text(json.dumps(summary, indent=2))
                return summary
    except Exception:
        pass

    metrics_path = workdir / "metrics.jsonl"
    if metrics_path.exists():
        return {"note": "Proxy unreachable, but metrics.jsonl saved", "path": str(metrics_path)}
    return None


def _print_results(
    summary: JsonDict | None, workdir: Path, empty_count: int = 0, total_tasks: int = 0
) -> None:
    """Print a clear results section after the benchmark run."""
    console.print("\n[bold]─── Results ───[/bold]")
    console.print(f"  Workdir: {workdir}")

    metrics_path = workdir / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        line_count = len(lines)
        error_count = sum(1 for line in lines if '"error"' in line)
        console.print(f"  Metrics: {metrics_path} ({line_count} requests)")
        if error_count:
            console.print(
                f"  [red]Upstream errors: {error_count}/{line_count}"
                f" requests failed[/red]"
            )
    else:
        console.print(
            "  [yellow]No metrics.jsonl found - proxy may not "
            "have received requests[/yellow]"
        )

    if empty_count and total_tasks:
        console.print(
            f"  [yellow]Empty output: {empty_count}/{total_tasks} "
            f"schedule-tasks produced no stdout[/yellow]"
        )

    if not summary or "error" in summary:
        console.print("  [yellow]No proxy summary available[/yellow]")
        return

    total = summary.get("total_requests", 0)
    streaming = summary.get("streaming_requests", 0)
    console.print(f"  Total requests: {total}  (streaming: {streaming})")

    ttft = summary.get("ttft_ms")
    tps = summary.get("tok_per_sec")
    prefill = summary.get("prefill_tok_per_sec")

    if not any([ttft, tps, prefill]):
        console.print("  [dim]No streaming stats recorded[/dim]")
        return

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    for col in ["Count", "Min", "Mean", "Median", "P95", "Max"]:
        table.add_column(col, justify="right")

    for label, data in [("TTFT (ms)", ttft), ("Decode tok/s", tps), ("Prefill tok/s", prefill)]:
        if not isinstance(data, dict):
            continue
        table.add_row(
            label,
            str(data.get("count", "")),
            str(data.get("min", "")),
            str(data.get("mean", "")),
            str(data.get("median", "")),
            str(data.get("p95", "")),
            str(data.get("max", "")),
        )

    console.print(table)
    console.print(f"\n  Summary saved: {workdir / 'summary.json'}")


def _cleanup_workdir(workdir: Path, keep_logs: bool) -> None:
    """Remove debug artifacts from the workdir. Keep metrics and summary."""
    if keep_logs:
        return
    for log_file in workdir.glob("*.log"):
        log_file.unlink()
    for task_dir in workdir.glob("slot*"):
        if task_dir.is_dir() and not any(task_dir.iterdir()):
            task_dir.rmdir()


async def _stop_proxy(proxy_proc: asyncio.subprocess.Process) -> None:
    """Gracefully stop the proxy subprocess."""
    if proxy_proc.returncode is not None:
        return
    proxy_proc.terminate()
    try:
        await asyncio.wait_for(proxy_proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proxy_proc.kill()
        await proxy_proc.wait()


async def _start_proxy(
    config: BenchmarkConfig,
    log_dir: str = "./traces",
) -> asyncio.subprocess.Process | None:
    """Start the recording proxy as a subprocess."""
    try:
        import agentic_swarm_bench.proxy.server  # noqa: F401
    except ImportError:
        console.print("[red]Proxy deps missing. Run: pip install agentic-swarm-bench[proxy][/red]")
        return None

    script = (
        "import json, sys; "
        "from agentic_swarm_bench.proxy.server import run_proxy; "
        "args = json.loads(sys.argv[1]); "
        "run_proxy(**args)"
    )
    args_json = json.dumps(
        {
            "upstream_url": config.endpoint,
            "port": config.proxy_port,
            "model": config.model,
            "api_key": config.api_key,
            "api_key_header": config.api_key_header,
            "log_dir": log_dir,
            "upstream_api": config.upstream_api,
        }
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        args_json,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    console.print(f"  Proxy started (PID {proc.pid})")
    return proc
