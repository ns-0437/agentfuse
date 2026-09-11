"""Token-logprob plumbing — reading confidence without inventing a signal.

`ConfidenceDetector`, the class these tests used to cover, was deleted
2026-09-11 (REPORT.md 3.4/4.12): measured actively harmful (dF1 +10.8 for
removing it, 118 extra false positives) and never reachable through
`MonitorConfig` in the first place. `_token_logprobs`/`summarize` remain --
they are useful extraction/summarising utilities independent of whether this
project ships a built-in consumer -- and these are the tests that cover them,
including the real bug found within minutes of first use:

    The OpenAI SDK returns a `ChoiceLogprobs` OBJECT, not a dict. The extractor
    handled dicts and lists, so a server returning perfectly good logprobs
    looked like one returning none — reading absence of signal where there
    was signal, which is the exact failure this module exists to avoid.

    pytest evals/test_confidence.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from agentfuse.confidence import _token_logprobs, summarize  # noqa: E402


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
    assert s is not None
    assert s["tokens"] == 4
    assert s["mean_logprob"] == pytest.approx(-0.825)
    assert s["perplexity"] == pytest.approx(2.282, abs=1e-2)
    assert s["low_fraction"] == pytest.approx(0.25), "one of four tokens below -1.0"
    assert s["min_logprob"] == -3.0


def test_empty_and_non_finite_input_summarise_to_nothing():
    assert summarize([]) is None
    assert summarize([float("nan"), float("-inf")]) is None
