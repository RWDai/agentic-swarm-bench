"""Load and filter benchmark tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

JsonTask = dict[str, Any]

TASKS_FILE = Path(__file__).parent / "tasks.json"

TIERS = ["trivial", "easy", "medium", "hard", "expert"]

TIER_RANGES = {
    "trivial": (1, 10),
    "easy": (11, 25),
    "medium": (26, 50),
    "hard": (51, 75),
    "expert": (76, 100),
}


def load_all_tasks() -> list[JsonTask]:
    with open(TASKS_FILE) as f:
        return json.load(f)


def parse_task_range(spec: str) -> tuple[int, int]:
    """Parse a range like 'p1-p25', 'P51-P75', '1-50', or 'p10'.

    Returns (start, end) inclusive.
    """
    spec = spec.strip().lower()

    if spec in TIERS:
        return TIER_RANGES[spec]

    spec = spec.replace("p", "")
    if "-" in spec:
        parts = spec.split("-", 1)
        return int(parts[0]), int(parts[1])

    n = int(spec)
    return n, n


def _task_number(task: JsonTask) -> int:
    return int(task["id"].lstrip("Pp"))


def parse_task_mix(spec: Optional[str]) -> dict[str, float]:
    """Parse a tier mix such as ``balanced`` or ``trivial=1,easy=2``."""
    if spec is None or spec.strip().lower() in {"", "balanced"}:
        return {}

    weights: dict[str, float] = {}
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                "task mix entries must use tier=weight, e.g. "
                "trivial=1,easy=2,medium=4"
            )
        tier, raw_weight = part.split("=", 1)
        tier = tier.strip().lower()
        if tier not in TIERS:
            raise ValueError(f"unknown task tier in mix: {tier!r}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"invalid weight for tier {tier!r}: {raw_weight!r}") from exc
        if weight <= 0:
            raise ValueError(f"task mix weight for tier {tier!r} must be > 0")
        weights[tier] = weight

    if not weights:
        return {}
    return weights


def select_task_mix(
    tasks: list[JsonTask],
    *,
    mix: Optional[str] = None,
    count: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[JsonTask]:
    """Select a deterministic tier-balanced or weighted subset of tasks.

    ``tasks`` is already filtered by range/tags. ``count`` controls how many
    distinct tasks are selected; when omitted, all candidates remain eligible.
    The scheduler still controls repetitions and concurrency afterward.
    """
    if not tasks:
        return []
    if count is not None and count < 1:
        raise ValueError(f"task_count must be >= 1, got {count}")

    target_count = min(count or len(tasks), len(tasks))
    weights = parse_task_mix(mix)

    grouped: dict[str, list[JsonTask]] = {tier: [] for tier in TIERS}
    for task in sorted(tasks, key=_task_number):
        grouped.setdefault(str(task.get("tier", "")), []).append(task)

    active_tiers = [tier for tier in TIERS if grouped.get(tier)]
    if weights:
        active_tiers = [tier for tier in TIERS if tier in weights and grouped.get(tier)]
        missing = [tier for tier in weights if not grouped.get(tier)]
        if missing:
            raise ValueError(
                "task mix references tiers outside the selected task range: "
                + ",".join(sorted(missing))
            )
    if not active_tiers:
        return []

    effective_weights = {tier: weights.get(tier, 1.0) for tier in active_tiers}
    quotas = _allocate_tier_quotas(grouped, effective_weights, target_count)

    rng = random.Random(seed)
    selected: list[JsonTask] = []
    for tier in TIERS:
        quota = quotas.get(tier, 0)
        if quota <= 0:
            continue
        tier_tasks = list(grouped[tier])
        if seed is not None:
            rng.shuffle(tier_tasks)
        selected.extend(tier_tasks[:quota])

    return sorted(selected, key=_task_number)


def _allocate_tier_quotas(
    grouped: dict[str, list[JsonTask]],
    weights: dict[str, float],
    target_count: int,
) -> dict[str, int]:
    remaining_capacity = {tier: len(grouped[tier]) for tier in weights}
    quotas = {tier: 0 for tier in weights}
    remaining = target_count

    while remaining > 0:
        available = [tier for tier, capacity in remaining_capacity.items() if capacity > 0]
        if not available:
            break

        total_weight = sum(weights[tier] for tier in available)
        raw_allocations = {
            tier: remaining * weights[tier] / total_weight for tier in available
        }
        additions = {
            tier: min(int(raw_allocations[tier]), remaining_capacity[tier])
            for tier in available
        }
        allocated = sum(additions.values())

        if allocated == 0:
            ranked = sorted(
                available,
                key=lambda tier: (-raw_allocations[tier], TIERS.index(tier)),
            )
            for tier in ranked:
                if remaining == 0:
                    break
                additions[tier] += 1
                allocated += 1
                remaining -= 1
                remaining_capacity[tier] -= 1
                quotas[tier] += 1
            continue

        for tier, addition in additions.items():
            quotas[tier] += addition
            remaining_capacity[tier] -= addition
        remaining -= allocated

        leftovers = sorted(
            available,
            key=lambda tier: (
                -(raw_allocations[tier] - int(raw_allocations[tier])),
                TIERS.index(tier),
            ),
        )
        for tier in leftovers:
            if remaining == 0:
                break
            if remaining_capacity[tier] <= 0:
                continue
            quotas[tier] += 1
            remaining_capacity[tier] -= 1
            remaining -= 1

    return quotas


def filter_tasks(
    tasks: list[JsonTask],
    task_range: Optional[str] = None,
    tier: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[JsonTask]:
    """Filter tasks by range, tier, or tags."""
    result = tasks

    if task_range:
        start, end = parse_task_range(task_range)
        result = [t for t in result if start <= _task_number(t) <= end]

    if tier:
        result = [t for t in result if t["tier"] == tier]

    if tags:
        tag_set = set(tags)
        result = [t for t in result if tag_set.intersection(t.get("tags", []))]

    return result


def get_tasks(
    task_range: Optional[str] = None,
    tier: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[JsonTask]:
    """Load and filter tasks in one call."""
    all_tasks = load_all_tasks()
    return filter_tasks(all_tasks, task_range=task_range, tier=tier, tags=tags)
