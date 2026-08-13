"""Scoring the *steering*, not just the detection.

Until now the benchmark measured one half of the product. AgentFuse's central
claim is not "it notices failures" — it is "it notices failures **and steers the
agent out of them**". The second half had never been measured at all.

Two things are scored here:

  1. **Steering quality** — a deterministic rubric over the instruction the
     recovery engine produced. Does it name the actual failure? Does it re-anchor
     to the original objective? Does it prescribe a *different* action rather than
     "try again"? Is the action type right for the severity?

  2. **Recovery efficacy** — whether the run actually reaches its objective after
     the steering is injected. Scenarios may declare a ``recovery_branch``: the
     trajectory the agent follows *if* it receives usable guidance. Steering that
     fails the rubric does not unlock the branch, so a vague nudge scores as a
     failed recovery rather than a free win.

Honest limitation: with the deterministic mock backend, the rubric is close to a
regression test — the mock was written to produce well-formed instructions and
largely passes. Its real value appears when a live reasoning model is plugged in
(``RecoveryEngine(backend="real")``), where vague or hallucinated steering is a
genuine risk. It also stops a future refactor from silently degrading the mock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from agentfuse.recovery import SteeringPath, RecoveryAction

# Phrases that signal the instruction merely restates the failure instead of
# prescribing a way out of it.
_VAGUE = re.compile(
    r"\b(try again|retry|keep going|continue as before|proceed normally|"
    r"do your best|be careful|think harder)\b", re.I)

# Phrases that signal a concrete change of course.
#
# This list was widened after it was caught scoring its own authors' vocabulary.
# The original set was written alongside the deterministic templates, so it
# matched their exact wording and missed semantically identical phrasings from a
# real model: `realign` matched but `re-align` did not, and "do not repeat any
# previous actions involving searching files" — an explicit, named prohibition —
# contained no listed keyword at all and scored as prescribing no action.
#
# That is measurement bias in favour of the templates, in a comparison whose
# entire purpose is templates-versus-model. Measured effect of the fix on the
# local 3B model: actionable 55% -> 80%. The templates are unaffected, since
# they already matched.
_ACTIONABLE = re.compile(
    r"\b(stop|different|instead|another|alternative|switch|discard|abandon|"
    r"re-?read|re-?align|re-?anchor|verify|list your assumptions|choose|halt|"
    r"escalat|avoid|refrain|cease|rather than|no longer|"
    r"(do not|don't|never)\s+(repeat|call|use|invoke|retry|attempt|continue))",
    re.I)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (text or "").lower())}


@dataclass
class SteeringScore:
    """Rubric result for a single steering instruction."""

    diagnostic: bool = False        # names the specific failing behaviour
    goal_anchored: bool = False     # re-anchors to the original objective
    actionable: bool = False        # prescribes a concrete different action
    not_vague: bool = False         # avoids "try again" style non-advice
    action_appropriate: bool = False  # inject vs escalate matches the severity
    novel: bool = True              # differs from the previous steering attempt
    reasons: list[str] = field(default_factory=list)

    CHECKS = ("diagnostic", "goal_anchored", "actionable", "not_vague",
              "action_appropriate", "novel")

    @property
    def score(self) -> float:
        return sum(bool(getattr(self, c)) for c in self.CHECKS) / len(self.CHECKS)

    @property
    def usable(self) -> bool:
        """Would this instruction plausibly move a competent agent forward?

        Requires the three load-bearing properties — a concrete alternative
        action, anchored to the real goal, that isn't empty encouragement.
        """
        return self.actionable and self.goal_anchored and self.not_vague

    def to_dict(self) -> dict:
        d = {c: bool(getattr(self, c)) for c in self.CHECKS}
        d.update({"score": round(self.score, 3), "usable": self.usable,
                  "reasons": self.reasons})
        return d


def score_steering(path: SteeringPath, *, original_goal: str,
                   trip_detector: str, trip_severity: str,
                   failing_tool: Optional[str] = None,
                   previous_instructions: Optional[list[str]] = None) -> SteeringScore:
    """Apply the rubric to one steering instruction."""
    s = SteeringScore()
    text = path.instruction or ""
    low = text.lower()

    # 1. Diagnostic — does it name what actually went wrong?
    if failing_tool and failing_tool.lower() in low:
        s.diagnostic = True
    elif trip_detector == "drift" and any(w in low for w in ("drift", "objective", "tangent", "off")):
        s.diagnostic = True
    elif trip_detector == "progress" and any(w in low for w in ("assumption", "progress", "state", "false")):
        s.diagnostic = True
    elif trip_detector == "spend" and any(w in low for w in ("budget", "token", "burn", "spend", "ceiling")):
        s.diagnostic = True
    if not s.diagnostic:
        s.reasons.append(f"does not name the {trip_detector} failure concretely")

    # 2. Goal-anchored — does it point back at the real objective?
    goal_words = _content_words(original_goal)
    overlap = len(goal_words & _content_words(text))
    if overlap >= max(2, int(0.15 * len(goal_words))):
        s.goal_anchored = True
    else:
        s.reasons.append("does not re-anchor to the original objective")

    # 3/4. Actionable, and not empty encouragement.
    s.actionable = bool(_ACTIONABLE.search(text))
    if not s.actionable:
        s.reasons.append("prescribes no concrete alternative action")
    s.not_vague = not _VAGUE.search(text)
    if not s.not_vague:
        s.reasons.append("contains non-advice such as 'try again'")

    # 5. Action type must match severity: a hard ceiling is not steerable.
    if trip_severity == "critical":
        s.action_appropriate = path.action in (RecoveryAction.ESCALATE, RecoveryAction.ABORT)
        if not s.action_appropriate:
            s.reasons.append("critical trip should escalate, not inject")
    else:
        s.action_appropriate = path.action in (
            RecoveryAction.INJECT, RecoveryAction.ESCALATE, RecoveryAction.ABORT)

    # 6. Novelty — repeating a steer that already failed is not a recovery.
    for prev in (previous_instructions or []):
        a, b = _content_words(prev), _content_words(text)
        if a and b and len(a & b) / max(1, len(a | b)) > 0.9:
            s.novel = False
            s.reasons.append("near-identical to a previous steering attempt")
            break

    return s


@dataclass
class RecoveryOutcome:
    """Did the run actually get back on track?"""

    attempted: bool = False       # steering was produced at all
    usable: bool = False          # the instruction passed the rubric
    recovered: bool = False       # the run reached its objective afterwards
    escalated: bool = False       # handed to a human (a correct outcome, not a win)
    steering_scores: list[SteeringScore] = field(default_factory=list)
    tokens_to_recovery: int = 0

    @property
    def mean_quality(self) -> float:
        if not self.steering_scores:
            return 0.0
        return sum(s.score for s in self.steering_scores) / len(self.steering_scores)

    def to_dict(self) -> dict:
        return {"attempted": self.attempted, "usable": self.usable,
                "recovered": self.recovered, "escalated": self.escalated,
                "mean_quality": round(self.mean_quality, 3),
                "tokens_to_recovery": self.tokens_to_recovery,
                "scores": [s.to_dict() for s in self.steering_scores]}
