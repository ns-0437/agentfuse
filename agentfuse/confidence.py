"""Token-logprob plumbing — reading a model's own confidence, without a consumer.

This module reads the one internal signal a closed API will hand over: **token
logprobs**. OpenAI's chat completions return them on request, and llama.cpp's
server implements the same field, so this works on hosted and self-hosted models
alike without needing weights. ``agentfuse.adapters.openai_sdk.guarded_tool_loop``
attaches a summary of them to ``AgentEvent.meta["confidence"]`` when
``logprobs=True``, independent of anything reading it downstream.

What used to read it: a Tier 1 detector that tripped on a self-relative
confidence collapse. Measured against a real model (``evals/measure_confidence.py``,
now removed with it) and found **actively harmful** — ablation put it at ΔF1
+10.8 for REMOVING it, recall unchanged at 97.6% (it caught nothing the
behavioural detectors missed), and 118 additional false positives. Deleted
2026-09-11 rather than kept unreachable: it was never wired into
``MonitorConfig`` in the first place, so no one could have enabled it by
accident, but keeping fully-built, tested, provably-harmful code around with
no way to turn it on was worse than removing it. The measurement is preserved
in full in REPORT.md sections 3.4/4.12, which is the permanent record.

The extraction/summarising functions below remain because they are useful on
their own — raw per-turn confidence, for whoever wants to build something with
it — independent of whether this project ships a built-in consumer.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional


# --------------------------------------------------------------- extraction
def _token_logprobs(raw: Any) -> list[float]:
    """Pull a flat list of per-token logprobs out of the shapes providers use.

    Tolerant on purpose: OpenAI nests them under ``content[].logprob``,
    llama.cpp's server returns ``token_logprobs``, and a caller may already have
    reduced them to a list of floats. Returning [] for anything unrecognised
    means an unsupported provider degrades to "no signal" rather than to a wrong
    one.
    """
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, dict):
        for key in ("content", "tokens", "token_logprobs"):
            if key in raw:
                return _token_logprobs(raw[key])
        return []
    # The OpenAI SDK hands back a `ChoiceLogprobs` OBJECT, not a dict — its
    # tokens live on `.content`. Missing this made a server that was returning
    # perfectly good logprobs look like one that returned none at all, which is
    # the failure mode this whole module is supposed to avoid: reading absence
    # of signal where there is signal.
    for attr in ("content", "token_logprobs", "tokens"):
        nested = getattr(raw, attr, None)
        if nested is not None:
            return _token_logprobs(nested)
    if isinstance(raw, (list, tuple)):
        out: list[float] = []
        for item in raw:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, dict) and "logprob" in item:
                try:
                    out.append(float(item["logprob"]))
                except (TypeError, ValueError):
                    continue
            elif hasattr(item, "logprob"):
                try:
                    out.append(float(item.logprob))
                except (TypeError, ValueError):
                    continue
        return out
    return []


def summarize(logprobs: Iterable[float]) -> Optional[dict]:
    """Reduce a turn's token logprobs to the few numbers worth keeping.

    ``mean`` is the headline. ``perplexity`` is its exponential, which is the
    form most people have intuitions about. ``low_fraction`` is the share of
    tokens the model was genuinely unsure of — a turn can average well and still
    contain a handful of coin-flips, and that pattern is different from uniform
    mild uncertainty.
    """
    values = [float(v) for v in logprobs if v is not None and math.isfinite(v)]
    if not values:
        return None
    mean = sum(values) / len(values)
    return {
        "tokens": len(values),
        "mean_logprob": mean,
        "perplexity": math.exp(-mean),
        "low_fraction": sum(1 for v in values if v < -1.0) / len(values),
        "min_logprob": min(values),
    }
