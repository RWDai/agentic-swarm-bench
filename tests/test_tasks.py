"""Tests for task loading and filtering."""

from agentic_swarm_bench.tasks.registry import (
    filter_tasks,
    get_tasks,
    load_all_tasks,
    parse_task_mix,
    parse_task_range,
    select_task_mix,
)


def test_load_all_tasks():
    tasks = load_all_tasks()
    assert len(tasks) == 110


def test_task_ids_are_sequential():
    tasks = load_all_tasks()
    for i, task in enumerate(tasks, start=1):
        assert task["id"] == f"P{i}"


def test_all_tiers_present():
    tasks = load_all_tasks()
    tiers = {t["tier"] for t in tasks}
    assert tiers == {"trivial", "easy", "medium", "hard", "expert"}


def test_tier_counts():
    tasks = load_all_tasks()
    counts = {}
    for t in tasks:
        counts[t["tier"]] = counts.get(t["tier"], 0) + 1
    assert counts["trivial"] == 11
    assert counts["easy"] == 17
    assert counts["medium"] == 28
    assert counts["hard"] == 27
    assert counts["expert"] == 27


def test_every_task_has_required_fields():
    tasks = load_all_tasks()
    for t in tasks:
        assert "id" in t
        assert "tier" in t
        assert "prompt" in t
        assert "tags" in t
        assert "max_output_tokens" in t
        assert len(t["prompt"]) > 10


def test_parse_task_range_p_notation():
    assert parse_task_range("p1-p25") == (1, 25)
    assert parse_task_range("P51-P75") == (51, 75)


def test_parse_task_range_numeric():
    assert parse_task_range("1-50") == (1, 50)


def test_parse_task_range_single():
    assert parse_task_range("p10") == (10, 10)


def test_parse_task_range_tier_name():
    assert parse_task_range("trivial") == (1, 10)
    assert parse_task_range("expert") == (76, 100)


def test_filter_by_range():
    tasks = load_all_tasks()
    filtered = filter_tasks(tasks, task_range="p1-p10")
    assert len(filtered) == 10
    assert all(t["tier"] == "trivial" for t in filtered)


def test_filter_by_tier():
    tasks = load_all_tasks()
    filtered = filter_tasks(tasks, tier="hard")
    assert len(filtered) == 27


def test_filter_by_tags():
    tasks = load_all_tasks()
    filtered = filter_tasks(tasks, tags=["basics"])
    assert len(filtered) >= 5
    assert all("basics" in t["tags"] for t in filtered)


def test_filter_by_language_tags():
    tasks = load_all_tasks()
    ts_tasks = filter_tasks(tasks, tags=["typescript"])
    assert len(ts_tasks) >= 3
    rust_tasks = filter_tasks(tasks, tags=["rust"])
    assert len(rust_tasks) >= 3
    go_tasks = filter_tasks(tasks, tags=["go"])
    assert len(go_tasks) >= 2


def test_get_tasks_combined():
    result = get_tasks(task_range="p1-p25", tier="easy")
    for t in result:
        assert t["tier"] == "easy"
        num = int(t["id"][1:])
        assert 1 <= num <= 25


def test_parse_task_mix_balanced_and_weighted():
    assert parse_task_mix(None) == {}
    assert parse_task_mix("balanced") == {}
    assert parse_task_mix("trivial=1,easy=2,medium=4") == {
        "trivial": 1.0,
        "easy": 2.0,
        "medium": 4.0,
    }


def test_parse_task_mix_rejects_invalid_values():
    for spec in ("trivial", "unknown=1", "easy=0", "medium=-1", "hard=nope"):
        try:
            parse_task_mix(spec)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {spec!r}")


def test_select_task_mix_balanced_one_per_tier():
    tasks = get_tasks(task_range="p1-p100")

    selected = select_task_mix(tasks, mix="balanced", count=5)

    assert [task["tier"] for task in selected] == [
        "trivial",
        "easy",
        "medium",
        "hard",
        "expert",
    ]


def test_select_task_mix_weighted_by_tier():
    tasks = get_tasks(task_range="p1-p100")

    selected = select_task_mix(
        tasks,
        mix="trivial=1,easy=2,medium=4,hard=2,expert=1",
        count=10,
    )

    counts: dict[str, int] = {}
    for task in selected:
        counts[task["tier"]] = counts.get(task["tier"], 0) + 1
    assert counts == {
        "trivial": 1,
        "easy": 2,
        "medium": 4,
        "hard": 2,
        "expert": 1,
    }


def test_select_task_mix_seed_changes_task_identity_deterministically():
    tasks = get_tasks(task_range="p1-p25")

    selected_a = select_task_mix(tasks, mix="trivial=1,easy=1", count=4, seed=1)
    selected_b = select_task_mix(tasks, mix="trivial=1,easy=1", count=4, seed=1)
    selected_c = select_task_mix(tasks, mix="trivial=1,easy=1", count=4, seed=2)

    assert [task["id"] for task in selected_a] == [task["id"] for task in selected_b]
    assert [task["id"] for task in selected_a] != [task["id"] for task in selected_c]
