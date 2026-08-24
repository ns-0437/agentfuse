"""Tests for the generator-entropy audit's own repeat-detection logic.

Worth testing directly: the first version of this audit flagged
gen_benign_retry as reproducing the gen_long_sparse_benign artifact, and it
was a false alarm -- the repeated triple was three "HTTP 503" results, which
LoopDetector deliberately tolerates as a healthy retry (retry_multiplier).
These pin the corrected behaviour so a future edit can't quietly reintroduce
that false positive.
"""

from __future__ import annotations

from evals.audit_generator_entropy import (
    REPEAT_THRESHOLD, RETRY_MULTIPLIER,
    worst_error_repeat_run, worst_nonerror_repeat_run,
)
from evals.schema import Scenario, StepSpec, Label


def _scenario(steps: list[StepSpec]) -> Scenario:
    return Scenario(id="t", title="t", family="t", goal="g", steps=steps,
                    description="t", label=Label(should_trip=False))


def _tool(name, args, result, progress=False) -> StepSpec:
    return StepSpec(kind="tool", tool_name=name, tool_args=args, result=result,
                    progress=progress)


def test_three_identical_nonerror_results_is_the_gen_long_sparse_benign_shape():
    steps = [_tool("search", {"q": "x"}, "found: a") for _ in range(3)]
    assert worst_nonerror_repeat_run(_scenario(steps)) == REPEAT_THRESHOLD


def test_error_shaped_repeats_do_not_count_as_nonerror_and_get_more_slack():
    """The exact case that produced a false alarm in the first version."""
    steps = [_tool("build", {"env": "x"}, "HTTP 503 service unavailable")
             for _ in range(REPEAT_THRESHOLD)]
    assert worst_nonerror_repeat_run(_scenario(steps)) == 0
    assert worst_error_repeat_run(_scenario(steps)) == REPEAT_THRESHOLD
    # Below the retry-tolerant threshold -- a real audit run would not flag this.
    assert worst_error_repeat_run(_scenario(steps)) < REPEAT_THRESHOLD * RETRY_MULTIPLIER


def test_progress_resets_the_run_even_mid_repeat():
    steps = [_tool("poll", {}, "pending"), _tool("poll", {}, "pending"),
             _tool("poll", {}, "advanced", progress=True),
             _tool("poll", {}, "pending"), _tool("poll", {}, "pending")]
    assert worst_nonerror_repeat_run(_scenario(steps)) == 2


def test_distinct_args_are_not_a_repeat():
    steps = [_tool("search", {"q": "a"}, "found"),
            _tool("search", {"q": "b"}, "found"),
            _tool("search", {"q": "c"}, "found")]
    assert worst_nonerror_repeat_run(_scenario(steps)) == 1
