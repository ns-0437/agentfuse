"""Ground-truth schema for the AgentFuse evaluation harness.

A *scenario* is a scripted agent trajectory plus a ground-truth label saying
whether the circuit breaker **should** have fired, which detector should have
caught it, and by when. Scenarios are deterministic and replayed offline, so the
whole suite runs in CI with no API key and no cost.

Two label classes matter equally:

  * **Positives** — a genuine long-horizon failure. We measure whether we catch
    it, *which* detector caught it, and *how fast* (wasted steps before the trip).
  * **Hard negatives** — a healthy run that superficially resembles a failure
    (a legitimate retry, a sub-goal that reads as drift, polling that is really
    making progress). These decide the false-positive rate, which is what
    actually determines whether anyone trusts the breaker in production.

Token accounting is derived from the scenario itself rather than hand-entered:
everything after the failure *onset* is waste, so savings = tokens the run would
have burned after the step where we tripped.

Pure stdlib — the harness must never be harder to run than the library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """What supervision costs, so 'tokens saved' is a *net* number.

    A breaker that saves 5k tokens but spends 6k supervising is a regression,
    and the report must be able to say so.
    """

    recovery_call_tokens: int = 1500   # one steering call to the reasoning model
    drift_probe_tokens: int = 0        # per-probe embedding cost (0 in lexical mode)
    false_positive_penalty: int = 1500  # a bogus trip costs a wasted recovery call

    @property
    def label(self) -> str:
        return (f"recovery={self.recovery_call_tokens}tok · "
                f"drift_probe={self.drift_probe_tokens}tok · "
                f"fp_penalty={self.false_positive_penalty}tok")


DEFAULT_COST = CostModel()


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
@dataclass
class StepSpec:
    """One scripted action in a trajectory.

    ``kind="tool"``  -> emits TOOL_CALL, then TOOL_RESULT (+ STATE_UPDATE if
                        ``progress`` is set, meaning the working state advanced).
    ``kind="think"`` -> emits a single LLM_CALL carrying reasoning text.
    """

    kind: str                          # "tool" | "think"
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    text: Optional[str] = None         # reasoning text / tool output
    goal: Optional[str] = None         # the agent's *currently interpreted* goal
    tokens_in: int = 0
    tokens_out: int = 0
    result: Optional[str] = None       # tool return value
    progress: bool = False             # did this genuinely advance the task?
    node: str = "agent"

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


def tool(name: str, args: dict, *, result: str = "ok", progress: bool = False,
         tokens_in: int = 700, tokens_out: int = 160, goal: Optional[str] = None,
         node: str = "agent") -> StepSpec:
    """Shorthand for a tool-calling step."""
    return StepSpec(kind="tool", tool_name=name, tool_args=args, result=result,
                    progress=progress, tokens_in=tokens_in, tokens_out=tokens_out,
                    goal=goal, node=node)


def think(text: str, *, tokens_in: int = 900, tokens_out: int = 220,
          as_goal: bool = True, progress: bool = False, node: str = "agent") -> StepSpec:
    """Shorthand for a reasoning turn.

    ``as_goal`` publishes the text as the agent's interpreted goal, which is what
    the drift detector probes. Set it False for reasoning that shouldn't be read
    as a statement of intent.
    """
    return StepSpec(kind="think", text=text, goal=text if as_goal else None,
                    tokens_in=tokens_in, tokens_out=tokens_out, progress=progress,
                    node=node)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
@dataclass
class Label:
    """What *should* happen, independent of what our code currently does."""

    should_trip: bool
    detector: Optional[str] = None      # expected detector on positives
    onset_index: Optional[int] = None   # step index where the failure truly begins
    detect_by_index: Optional[int] = None  # detecting later than this is "late"
    known_gap: bool = False             # we expect to MISS this today (documented weakness)
    note: str = ""


@dataclass
class Scenario:
    """A scripted trajectory plus its ground truth."""

    id: str
    title: str
    family: str                 # loop | drift | progress | spend | benign
    goal: str                   # the original objective handed to the agent
    steps: list[StepSpec]
    label: Label
    description: str = ""
    config: dict = field(default_factory=dict)  # MonitorConfig overrides

    # The trajectory the agent follows *if* it receives usable steering. This is
    # what makes recovery measurable at all: without it we could only ever score
    # detection, and "it self-heals" would remain an unverified claim. Steering
    # that fails the quality rubric does not unlock this branch.
    recovery_branch: list[StepSpec] = field(default_factory=list)
    # The failing tool a correct steering instruction ought to name.
    failing_tool: Optional[str] = None

    # Which ladder rung this agent actually responds to — the ground truth that
    # makes recovery measurable rather than circular. Previously the recovery
    # branch unlocked whenever our own rubric approved our own instruction,
    # which measured nothing: the rubric and the instruction templates were
    # written together, so it always passed. Here the agent's responsiveness is
    # independent of what we generate, so the ladder has to actually FIND the
    # rung that works. ``None`` means nothing steers this agent and the only
    # correct outcome is escalation.
    responds_to: Optional[str] = "re-anchor"

    # -- token accounting ----------------------------------------------
    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    def tokens_after_index(self, idx: int) -> int:
        """Tokens the run would still burn *after* step ``idx`` (0-based).

        This is the saving unlocked by tripping at ``idx``: the breaker halts the
        run there, so everything downstream is never spent.
        """
        if idx is None or idx < 0:
            return 0
        return sum(s.tokens for s in self.steps[idx + 1:])

    @property
    def wasted_tokens_if_undetected(self) -> int:
        """Tokens burned from the failure onset to the end of the run."""
        if self.label.onset_index is None:
            return 0
        return sum(s.tokens for s in self.steps[self.label.onset_index:])


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    """Outcome of replaying one scenario through a monitor."""

    scenario_id: str
    family: str
    should_trip: bool
    known_gap: bool
    tripped: bool
    trip_detector: Optional[str] = None
    trip_step_index: Optional[int] = None   # index into steps[] where it fired
    trip_severity: Optional[str] = None
    all_trips: list[dict] = field(default_factory=list)
    recoveries: int = 0
    tokens_spent: int = 0        # tokens actually consumed before the halt
    tokens_saved: int = 0        # tokens avoided by halting early
    supervision_cost: int = 0    # what the supervision itself cost
    steps_late: Optional[int] = None  # steps between onset and detection
    recovery: Optional["object"] = None  # RecoveryOutcome, when a branch existed

    # -- confusion-matrix classification -------------------------------
    @property
    def outcome(self) -> str:
        if self.should_trip and self.tripped:
            return "TP"
        if self.should_trip and not self.tripped:
            return "FN"
        if not self.should_trip and self.tripped:
            return "FP"
        return "TN"

    @property
    def attribution_correct(self) -> Optional[bool]:
        """Did the *right* detector catch it? (positives only)"""
        if not (self.should_trip and self.tripped):
            return None
        return self.trip_detector == self.expected_detector

    expected_detector: Optional[str] = None

    @property
    def net_tokens(self) -> int:
        """Net token benefit: saved minus what supervision cost."""
        return self.tokens_saved - self.supervision_cost
