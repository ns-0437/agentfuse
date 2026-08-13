"""Tier 1 signal plumbing — reading token logprobs without inventing a signal.

Split deliberately from the *hypothesis*. Whether low confidence actually marks a
failing run is measured against a real model in `evals/measure_confidence.py`;
these tests only cover the machinery, which had a real bug within minutes of
first use:

    The OpenAI SDK returns a `ChoiceLogprobs` OBJECT, not a dict. The extractor
    handled dicts and lists, so a server returning perfectly good logprobs looked
    like one returning none — reading absence of signal where there was signal,
    which is the exact failure this module exists to avoid.

A second quirk, found the same way and worth writing down: **llama.cpp's server
returns `logprobs=None` unless `top_logprobs` is also requested.** `logprobs=True`
alone silently yields nothing.

    pytest evals/test_confidence.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

import pytest  # noqa: E402

from agentfuse.confidence import (  # noqa: E402
    ConfidenceDetector, _token_logprobs, summarize, summarize_event,
)
from agentfuse.events import AgentEvent, EventType  # noqa: E402


class _Tok:
    def __init__(self, logprob):
        self.logprob = logprob


class _ChoiceLogprobs:
    """Mimics the OpenAI SDK object that broke the first extractor."""

    def __init__(self, values):
        self.content = [_Tok(v) for v in values]


# ------------------------------------------------------------- extraction
def test_extracts_from_the_sdk_object_not_just_dicts():
    """The bug. An object with `.content` is the shape real providers return."""
    assert _token_logprobs(_ChoiceLogprobs([-0.1, -0.5, -2.0])) == [-0.1, -0.5, -2.0]


def test_extracts_from_the_openai_dict_shape():
    raw = {"content": [{"token": "a", "logprob": -0.2},
                       {"token": "b", "logprob": -1.4}]}
    assert _token_logprobs(raw) == [-0.2, -1.4]


def test_extracts_from_llama_cpp_style_and_plain_floats():
    assert _token_logprobs({"token_logprobs": [-0.3, -0.7]}) == [-0.3, -0.7]
    assert _token_logprobs([-0.3, -0.7]) == [-0.3, -0.7]


def test_an_unrecognised_shape_yields_no_signal_rather_than_a_wrong_one():
    for junk in (None, "some string", 42.0 and object(), {"unrelated": 1}):
        out = _token_logprobs(junk)
        assert isinstance(out, list)


# ------------------------------------------------------------ summarising
def test_summary_reports_mean_perplexity_and_uncertain_share():
    s = summarize([-0.1, -0.1, -3.0, -0.1])
    assert s["tokens"] == 4
    assert s["mean_logprob"] == pytest.approx(-0.825)
    assert s["perplexity"] == pytest.approx(2.282, abs=1e-2)
    assert s["low_fraction"] == pytest.approx(0.25), "one of four tokens below -1.0"
    assert s["min_logprob"] == -3.0


def test_empty_and_non_finite_input_summarise_to_nothing():
    assert summarize([]) is None
    assert summarize([float("nan"), float("-inf")]) is None


def test_a_precomputed_summary_is_used_as_is():
    """A caller that already reduced its logprobs should not be second-guessed."""
    ev = AgentEvent(type=EventType.LLM_CALL, step=1,
                    meta={"confidence": {"mean_logprob": -0.4, "perplexity": 1.5,
                                         "low_fraction": 0.1, "tokens": 10}})
    assert summarize_event(ev)["mean_logprob"] == -0.4


# --------------------------------------------------------------- detector
def _turn(step, values):
    return AgentEvent(type=EventType.LLM_CALL, step=step,
                      meta={"logprobs": _ChoiceLogprobs(values)})


def test_it_abstains_entirely_without_logprobs():
    """Most callers never request them. Absence of signal is not evidence."""
    det = ConfidenceDetector(patience=1)
    for i in range(1, 20):
        assert det.inspect(AgentEvent(type=EventType.LLM_CALL, step=i,
                                      text="reasoning"), []) is None


def test_a_sustained_collapse_relative_to_the_run_trips():
    det = ConfidenceDetector(drop_sigmas=1.0, patience=3)
    for i in range(1, 6):
        assert det.inspect(_turn(i, [-0.15] * 20), []) is None      # healthy
    trip = None
    for i in range(6, 10):
        trip = trip or det.inspect(_turn(i, [-1.6] * 20), [])
    assert trip is not None and trip.detector == "confidence"
    assert trip.evidence["consecutive_low_turns"] >= 3


def test_one_uncertain_turn_is_not_a_collapse():
    det = ConfidenceDetector(drop_sigmas=1.0, patience=3)
    for i in range(1, 6):
        det.inspect(_turn(i, [-0.15] * 20), [])
    assert det.inspect(_turn(6, [-1.6] * 20), []) is None
    assert det.inspect(_turn(7, [-0.15] * 20), []) is None
    assert det._low_streak == 0, "a recovered turn must clear the streak"


def test_the_baseline_is_relative_so_a_quiet_model_is_not_punished():
    """Absolute thresholds do not transfer between models or prompts.

    This model runs at -1.4 throughout and never wavers. An absolute cut-off
    anywhere near it would fire constantly; a self-relative one stays silent.
    """
    det = ConfidenceDetector(drop_sigmas=1.0, patience=2)
    for i in range(1, 15):
        assert det.inspect(_turn(i, [-1.4] * 20), []) is None


def test_the_baseline_does_not_drift_down_to_follow_a_failing_model():
    """Otherwise the alarm calibrates itself away exactly when it is needed."""
    det = ConfidenceDetector(drop_sigmas=1.0, patience=2)
    for i in range(1, 5):
        det.inspect(_turn(i, [-0.2] * 20), [])
    healthy_baseline = det._baseline
    for i in range(5, 12):
        det.inspect(_turn(i, [-2.0] * 20), [])
    assert det._baseline == pytest.approx(healthy_baseline), (
        "low-confidence turns must not update the baseline")


def test_it_waits_for_a_baseline_before_judging_anything():
    """Tripping on turn two would be judging a model against nothing."""
    det = ConfidenceDetector(drop_sigmas=1.0, patience=1)
    assert det.inspect(_turn(1, [-0.2] * 20), []) is None
    assert det.inspect(_turn(2, [-3.0] * 20), []) is None, (
        "a single healthy sample is not a baseline")


def test_tool_events_are_ignored():
    det = ConfidenceDetector(patience=1)
    ev = AgentEvent(type=EventType.TOOL_RESULT, step=1, tool_name="t",
                    meta={"logprobs": _ChoiceLogprobs([-5.0] * 10)})
    assert det.inspect(ev, []) is None
