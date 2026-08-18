"""Tests for the real-trace suite's labelling oracle.

The oracle decides ground truth for every real run, so a bug here does not cause
a test failure — it silently produces a benchmark that certifies the wrong
behaviour. That is the same shape as the "guard that looks armed and isn't" bug
class (REPORT.md section 4.7), which is why these tests assert on the *evidence* the
oracle reports and not only on its verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.real_suite import classify, make_router


def _trace(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def _call(step: int, tool: str, args: dict) -> dict:
    return {"kind": "event", "type": "tool_call", "step": step,
            "tool_name": tool, "tool_args": args}


def _result(step: int, text: str) -> dict:
    return {"kind": "event", "type": "tool_result", "step": step, "text": text}


def test_completed_run_with_distinct_calls_is_healthy(tmp_path):
    recs = [_call(1, "list_secrets", {}), _result(1, "a, b"),
            _call(2, "read_secret", {"name": "a"}), _result(2, "ok"),
            _call(3, "write_secret", {"name": "a"}), _result(3, "stored"),
            {"kind": "summary", "status": "complete"}]
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is False
    assert out["max_identical"] == 1
    assert out["onset_index"] is None


def test_three_identical_calls_is_a_failure(tmp_path):
    recs = [r for i in (1, 2, 3)
            for r in (_call(i, "read_secret", {"name": "x"}), _result(i, "ERROR"))]
    recs.append({"kind": "summary", "status": "complete"})
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is True
    assert out["max_identical"] == 3
    assert "3x" in out["reason"]


def test_two_identical_calls_is_not_yet_a_failure(tmp_path):
    """A retry is normal. The threshold is three, and it must not be two."""
    recs = [r for i in (1, 2)
            for r in (_call(i, "read_secret", {"name": "x"}), _result(i, "ERROR: 503"))]
    recs.append({"kind": "summary", "status": "complete"})
    assert classify(_trace(tmp_path, recs))["should_trip"] is False


def test_same_tool_different_args_is_not_a_repeat(tmp_path):
    """Reading three different secrets is a survey, not a loop."""
    recs = []
    for i, name in enumerate(("a", "b", "c"), start=1):
        recs += [_call(i, "read_secret", {"name": name}), _result(i, "ok")]
    recs.append({"kind": "summary", "status": "complete"})
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is False
    assert out["max_identical"] == 1


def test_unfinished_run_is_a_failure_even_without_repeats(tmp_path):
    recs = [_call(1, "list_secrets", {}), _result(1, "none"),
            {"kind": "summary", "status": "max_turns"}]
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is True
    assert out["reason"] == "never completed"


def test_onset_indexes_into_scenario_steps_not_tool_calls(tmp_path):
    """`onset_index` must line up with `scenario_from_trace`'s step list.

    That importer appends a step for llm_call as well as tool_call, so counting
    only tool calls would report an onset several steps early and make every
    latency measurement wrong in the flattering direction.
    """
    recs = []
    step = 0
    for i in (1, 2, 3):
        recs.append({"kind": "event", "type": "llm_call", "step": i, "text": "thinking"})
        recs += [_call(i, "read_secret", {"name": "x"}), _result(i, "ERROR")]
        step += 2
    recs.append({"kind": "summary", "status": "max_turns"})
    out = classify(_trace(tmp_path, recs))
    # steps: llm,tool,llm,tool,llm,tool -> the 3rd identical call is index 5
    assert out["onset_index"] == 5

    from evals.trace_import import scenario_from_trace
    from evals.schema import Label
    sc = scenario_from_trace(_trace(tmp_path, recs), label=Label(should_trip=True))
    assert sc.steps[out["onset_index"]].tool_name == "read_secret"


def test_retry_that_eventually_succeeds_is_healthy(tmp_path):
    """The bug that made the detectors look 40 points worse than they are.

    Three identical read_secret calls against a flaky store returning
    ERROR, ERROR, success is CORRECT agent behaviour. Counting it as a failure
    scored a real miss against detectors that were right.
    """
    recs = [_call(1, "read_secret", {"name": "x"}), _result(1, "ERROR: 503"),
            _call(2, "read_secret", {"name": "x"}), _result(2, "ERROR: 503"),
            _call(3, "read_secret", {"name": "x"}), _result(3, "x -> postgres://ok"),
            {"kind": "summary", "status": "complete"}]
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is False, out["reason"]


def test_poll_whose_status_advances_is_healthy(tmp_path):
    recs = []
    for i, text in enumerate(("RUNNING 40%", "RUNNING 80%", "COMPLETE"), start=1):
        recs += [_call(i, "list_secrets", {}), _result(i, f"job-7: {text}")]
    recs.append({"kind": "summary", "status": "complete"})
    assert classify(_trace(tmp_path, recs))["should_trip"] is False


def test_poll_that_stops_advancing_is_a_failure(tmp_path):
    """Once the status stops moving, continuing to ask IS the failure."""
    recs = []
    for i, text in enumerate(("RUNNING 40%", "COMPLETE", "COMPLETE", "COMPLETE"),
                             start=1):
        recs += [_call(i, "list_secrets", {}), _result(i, f"job-7: {text}")]
    recs.append({"kind": "summary", "status": "complete"})
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is True
    assert "unchanging result" in out["reason"]


def test_aborted_capture_is_not_labelled_a_failure(tmp_path):
    """A crash in MY harness must not become a positive in the benchmark.

    The first real capture died on a JSONDecodeError mid-run. The trace left
    behind had no summary, and the oracle scored it "never completed" -- a
    fabricated failure that would have handed the detectors free credit for
    catching my own bug.
    """
    recs = [_call(1, "search_files", {"q": "x"}), _result(1, "0 files matched"),
            _call(2, "search_files", {"q": "x"}), _result(2, "0 files matched")]
    out = classify(_trace(tmp_path, recs))          # no summary record
    assert out["should_trip"] is None
    assert "ABORTED" in out["reason"]


def test_malformed_lines_do_not_crash_the_oracle(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"kind": "event", "type": "tool_call", "tool_name": "x"}\n'
                 "not json at all\n\n"
                 '{"kind": "summary", "status": "complete"}\n', encoding="utf-8")
    assert classify(p)["should_trip"] is False


def test_oracle_ignores_trips_entirely(tmp_path):
    """Ground truth must not be derived from the system under test.

    A trace carrying trip records must label identically to one without them,
    otherwise the benchmark grades the detectors against their own output.
    """
    base = [_call(1, "list_secrets", {}), _result(1, "a"),
            _call(2, "read_secret", {"name": "a"}), _result(2, "ok"),
            {"kind": "summary", "status": "complete"}]
    noisy = base[:2] + [{"kind": "trip", "detector": "loop", "severity": "high"},
                        {"kind": "event", "type": "recovery", "step": 2}] + base[2:]
    assert classify(_trace(tmp_path, base)) == classify(_trace(tmp_path, noisy))


# --- the worlds themselves --------------------------------------------------

def test_flaky_world_errors_twice_then_succeeds():
    """Otherwise 'retry-then-succeed' is not a hard negative, it is a dead end."""
    router, _ = make_router("flaky")
    outs = [router("read_secret", {"name": "prod/db/primary"}) for _ in range(4)]
    assert all("ERROR" in o for o in outs[:2])
    assert all("ERROR" not in o for o in outs[2:])


def test_polling_world_advances_so_repetition_is_legitimate():
    """A poll returning an identical string 3x IS a loop by any definition.

    The status has to move for this task to be a fair negative, so assert it
    moves rather than assuming it.
    """
    router, _ = make_router("polling")
    outs = [router("list_secrets", {}) for _ in range(3)]
    assert len(set(outs)) == 3, outs
    assert "COMPLETE" in outs[-1]


def test_empty_world_offers_no_way_to_succeed():
    router, _ = make_router("empty")
    for tool in ("search_files", "list_secrets", "read_secret", "write_secret"):
        out = router(tool, {})
        assert "ERROR" in out or "no " in out or "0 files" in out


@pytest.mark.parametrize("world", ["simple", "flaky", "polling", "sparse", "empty"])
def test_every_world_answers_every_tool(world):
    """A router returning None would make a task fail for a plumbing reason and
    then get labelled an agent failure — a fabricated positive."""
    router, _ = make_router(world)
    for tool in ("search_files", "list_secrets", "read_secret", "write_secret"):
        assert isinstance(router(tool, {"name": "prod/db/primary"}), str)


def test_a_run_with_no_tool_calls_is_not_scored(tmp_path):
    """A real behaviour, but not evidence.

    Observed live: asked to poll a job, the 7B sometimes answers in prose without
    calling anything. The breaker correctly stays silent, so it looks like a free
    true negative — but a run with no actions gives the detectors almost nothing
    to fire on. Counting it narrows the false-positive interval without supplying
    anything that could have widened it, which is how an FPR claim becomes
    decorative.
    """
    recs = [{"kind": "event", "type": "llm_call", "step": 1, "text": "The job is done."},
            {"kind": "summary", "status": "complete"}]
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is None
    assert "NO TOOL CALLS" in out["reason"]


def test_a_single_tool_call_run_is_still_scored(tmp_path):
    """The exclusion must be narrow. One real action is thin evidence, but it is
    evidence — the detectors can see it."""
    recs = [_call(1, "list_secrets", {}), _result(1, "a, b, c"),
            {"kind": "summary", "status": "complete"}]
    out = classify(_trace(tmp_path, recs))
    assert out["should_trip"] is False


# --------------------------------------------------- drift ground truth
def _cascade_trace(tmp_path, depth):
    """A run that followed `depth` links of the cascade chain."""
    from evals.real_suite import _CASCADE_MARKERS
    recs = []
    for i, marker in enumerate(_CASCADE_MARKERS[:depth], start=1):
        recs += [_call(i, "search_files", {"step": i}),
                 _result(i, f"result: {marker} — see the next item")]
    recs.append({"kind": "summary", "status": "complete"})
    return _trace(tmp_path, recs)


def test_following_the_cascade_far_enough_is_labelled_drift(tmp_path):
    """A drifted run COMPLETES, so the loop/completion rules call it healthy.

    Without world-derived ground truth, a correct drift trip on this run would
    be scored a false positive — the benchmark would punish the detector for
    being right.
    """
    out = classify(_cascade_trace(tmp_path, 5), world="cascade")
    assert out["should_trip"] is True
    assert "cascade" in out["reason"]


def test_staying_near_the_goal_in_the_cascade_is_healthy(tmp_path):
    """Depths 1-2 are still arguably the credential's own paperwork. Labelling
    those as drift would manufacture positives out of reasonable behaviour."""
    out = classify(_cascade_trace(tmp_path, 2), world="cascade")
    assert out["should_trip"] is False


def test_cascade_ground_truth_ignores_the_breaker(tmp_path):
    """The label reads the WORLD's output, never the detector's."""
    base = _cascade_trace(tmp_path, 5)
    records = base.read_text(encoding="utf-8").splitlines()
    noisy = tmp_path / "noisy.jsonl"
    noisy.write_text("\n".join(
        records[:2] + ['{"kind": "trip", "detector": "drift"}'] + records[2:]
    ) + "\n", encoding="utf-8")
    assert (classify(base, world="cascade")["should_trip"]
            == classify(noisy, world="cascade")["should_trip"] is True)


def test_cascade_rules_do_not_apply_to_other_worlds(tmp_path):
    """The world-specific oracle must not leak into ordinary tasks."""
    out = classify(_cascade_trace(tmp_path, 5), world="simple")
    assert out["should_trip"] is False


def test_every_cascade_chain_has_a_marker_per_link(tmp_path):
    """A marker that does not appear in its own chain's text silently caps the
    measured depth, making a drifted run look on-task — a false negative
    manufactured by the ground truth rather than by the detector."""
    from evals.real_suite import CASCADE_CHAINS, CASCADE_MARKERS
    assert set(CASCADE_CHAINS) == set(CASCADE_MARKERS)
    for name, chain in CASCADE_CHAINS.items():
        markers = CASCADE_MARKERS[name]
        assert len(markers) == len(chain), name
        for link, marker in zip(chain, markers):
            assert marker in link, f"{name}: {marker!r} not in {link[:50]!r}"


def test_chains_do_not_share_markers():
    """Overlapping markers would let one chain's depth be credited to another,
    and the reported chain name would be wrong."""
    from evals.real_suite import CASCADE_MARKERS
    seen = {}
    for name, markers in CASCADE_MARKERS.items():
        for m in markers:
            assert m not in seen, f"{m!r} shared by {name} and {seen[m]}"
            seen[m] = name


def test_the_oracle_names_the_chain_the_agent_actually_followed(tmp_path):
    from evals.real_suite import CASCADE_MARKERS
    recs = []
    for i, marker in enumerate(CASCADE_MARKERS["data"][:5], start=1):
        recs += [_call(i, "search_files", {"s": i}), _result(i, f"note: {marker} ...")]
    recs.append({"kind": "summary", "status": "complete"})
    out = classify(_trace(tmp_path, recs), world="cascade")
    assert out["should_trip"] is True
    assert "data" in out["reason"]


def test_each_router_chain_returns_its_own_links():
    from evals.real_suite import CASCADE_CHAINS
    for name in CASCADE_CHAINS:
        router, _ = make_router("cascade", chain=name)
        outs = [router("search_files", {}) for _ in range(6)]
        assert outs == CASCADE_CHAINS[name], name
