"""Tier 1 of the signal ladder — what the model's own token probabilities say.

Every detector in this project so far reads **behaviour**: which tools were
called, whether state moved, how many tokens went out. That is Tier 0, and it is
deliberately model-agnostic. Its blind spot is equally deliberate: an agent that
is confidently wrong and an agent that is guessing produce identical tool traces,
because the difference lives inside the model rather than in what it did.

Tier 1 reads the one internal signal a closed API will hand over: **token
logprobs**. OpenAI's chat completions return them on request, and llama.cpp's
server implements the same field, so this works on hosted and self-hosted models
alike without needing weights.

Why the threshold is relative, not absolute
-------------------------------------------
Mean logprob is not comparable across models, or even across prompts to the same
model: a terse factual answer and a long piece of reasoning sit at different
baselines by construction. A constant here would be a magic number that transfers
nowhere.

So confidence is judged **against the run's own history**, the same principle as
:mod:`agentfuse.calibration`: learn what this model's confidence looks like while
it is demonstrably doing fine, then trip when it falls well below that. A drop
relative to a self-established baseline is a claim that survives changing models;
"mean logprob below -1.2" is not.

What this does NOT assume
-------------------------
That low confidence means failure. That is a hypothesis, and it is tested in
``evals/measure_confidence.py`` against a real local model rather than asserted
here — the same order the embedding work followed, where measuring first showed a
33M model ranked drift as *more* similar than on-task text and would have made
things worse. If the separation does not hold, the honest outcome is that this
detector does not ship.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

from .events import AgentEvent, EventType
from .detectors.base import Detector, Severity, Trip


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


def summarize_event(event: AgentEvent) -> Optional[dict]:
    """Confidence stats for one event, from raw logprobs or a precomputed dict."""
    meta = event.meta or {}
    pre = meta.get("confidence")
    if isinstance(pre, dict) and "mean_logprob" in pre:
        return pre
    if "mean_logprob" in meta:          # a caller that did the reduction itself
        return summarize([meta["mean_logprob"]])
    return summarize(_token_logprobs(meta.get("logprobs")))


# ---------------------------------------------------------------- detector
class ConfidenceDetector(Detector):
    """Trips when the model's own certainty collapses relative to this run.

    Abstains entirely when no event carries logprobs, which is the common case:
    most callers do not request them, and a detector that invented a signal from
    their absence would be worse than one that stays quiet.
    """

    name = "confidence"

    #: Turns of evidenced-healthy reasoning before a baseline is trusted. Lower
    #: than it looks: each turn contributes hundreds of tokens.
    MIN_SAMPLES = 3

    def __init__(self, drop_sigmas: float = 1.0, patience: int = 3,
                 alpha: float = 0.3, calibrator=None, min_drop: float = 0.03):
        #: How far below baseline counts as a collapse, measured in standard
        #: deviations of THIS RUN's own healthy turns rather than in nats.
        #:
        #: The first version used an absolute 0.6 nats, and measuring it showed
        #: why that was wrong twice over. On a real 3B model the actual gaps are
        #: 0.09-0.16 nats, so 0.6 could never fire — a detector tuned above its
        #: own effect size, which is the same defect NoProgressDetector had. And
        #: an absolute figure would not transfer anyway: effect sizes differ by
        #: model just as baselines do.
        #:
        #: In sigmas the measurement reads cleanly: healthy turns had sd 0.068,
        #: and the three failure modes sat 1.3, 2.1 and 2.3 sigmas below the
        #: healthy mean. At 1.0 sigma all three clear the bar, and requiring
        #: `patience` CONSECUTIVE low turns makes chance triggering ~0.4%
        #: rather than 16%.
        self.drop_sigmas = drop_sigmas
        #: Floor on the absolute drop, for a model so deterministic that its
        #: healthy variance is ~0. Without it, sigma collapses and any wobble
        #: reads as catastrophic.
        self.min_drop = min_drop
        #: Consecutive low-confidence turns before tripping. One uncertain turn
        #: is normal; a sustained run of them is the claim.
        self.patience = patience
        self.alpha = alpha            # EMA weight on the newest healthy turn
        self.calibrator = calibrator
        self._baseline: Optional[float] = None
        self._samples = 0
        self._low_streak = 0
        self._last: Optional[dict] = None
        self._healthy: list[float] = []      # recent healthy means, for sigma

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        if event.type is not EventType.LLM_CALL:
            return None
        stats = summarize_event(event)
        if stats is None:
            return None               # no signal; say nothing
        self._last = stats
        mean = stats["mean_logprob"]

        if self._baseline is None:
            self._baseline = mean
            self._samples = 1
            self._healthy = [mean]
            return None

        threshold = self._baseline - self._effective_drop()
        if mean < threshold:
            self._low_streak += 1
        else:
            # Only healthy turns update the baseline, for the same reason
            # AdaptiveCalibrator only learns from evidenced-healthy stretches: a
            # baseline that drifts down to follow a struggling model calibrates
            # the alarm away exactly when it is needed.
            self._low_streak = 0
            self._samples += 1
            self._baseline = (1 - self.alpha) * self._baseline + self.alpha * mean
            self._healthy.append(mean)
            del self._healthy[:-24]      # recent history is what sigma should reflect
            return None

        if self._samples < self.MIN_SAMPLES or self._low_streak < self.patience:
            return None

        return Trip(
            detector=self.name,
            severity=Severity.TRIP,
            reason=(
                f"Model confidence has collapsed for {self._low_streak} consecutive "
                f"turns: mean logprob {mean:.2f} against a run baseline of "
                f"{self._baseline:.2f} (perplexity {stats['perplexity']:.1f}, "
                f"{stats['low_fraction']:.0%} of tokens uncertain). The agent is "
                f"guessing rather than reasoning."
            ),
            evidence={
                "mean_logprob": round(mean, 3),
                "baseline": round(self._baseline, 3),
                "drop": round(self._baseline - mean, 3),
                "drop_threshold": round(self._effective_drop(), 3),
                "perplexity": round(stats["perplexity"], 2),
                "low_token_fraction": round(stats["low_fraction"], 3),
                "consecutive_low_turns": self._low_streak,
            },
        )

    def _effective_drop(self) -> float:
        """How far below baseline is a collapse, in this run's own units."""
        if len(self._healthy) < 2:
            # No spread observed yet, so sigma is meaningless. Fall back to the
            # floor rather than to zero, which would trip on the first wobble.
            return max(self.min_drop, self.drop_sigmas * 0.05)
        mean = sum(self._healthy) / len(self._healthy)
        var = sum((v - mean) ** 2 for v in self._healthy) / len(self._healthy)
        return max(self.min_drop, self.drop_sigmas * math.sqrt(var))

    def reset(self) -> None:
        self._low_streak = 0

    @property
    def status(self) -> dict:
        return {"baseline": self._baseline, "samples": self._samples,
                "low_streak": self._low_streak, "last": self._last}
