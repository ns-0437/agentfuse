"""CI gate for the AgentFuse eval suite.

Two suites, two jobs:

  * **Generated (536 scenarios)** — the statistically meaningful one. Confidence
    intervals here are narrow enough (±5-6 points) that a real regression is
    distinguishable from noise. All rate assertions run against this.
  * **Hand-written (16 scenarios)** — kept as *named* regression cases, so a
    specific bug that was once fixed can be asserted by name.

These tests do not pretend the system is finished: ``progress`` recall is 0% and
the false-positive rate is 10.8%, both recorded in ``baseline.json``. What they
enforce is that nothing silently gets worse, and that the detectors keep beating
a rate-matched random control.

    pytest evals/test_eval.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.ablation import run_ablation, control_significance  # noqa: E402
from evals.generators import generate_suite  # noqa: E402
from evals.metrics import score  # noqa: E402
from evals.runner import run_suite  # noqa: E402
from evals.scenarios import ALL_SCENARIOS, POSITIVES, NEGATIVES  # noqa: E402
from evals.stats import wilson  # noqa: E402

BASELINE = json.loads((ROOT / "evals" / "baseline.json").read_text(encoding="utf-8"))
GEN = BASELINE["generated_suite"]
HAND = BASELINE["handwritten_suite"]


@pytest.fixture(scope="module")
def generated():
    scenarios = generate_suite(n_per_generator=GEN["n_per_generator"], seed=GEN["seed"])
    results = run_suite(scenarios)
    return scenarios, results, score(results, {s.id: s for s in scenarios})


@pytest.fixture(scope="module")
def handwritten():
    results = run_suite(ALL_SCENARIOS)
    return results, score(results, {s.id: s for s in ALL_SCENARIOS})


# ------------------------------------------------------------ suite sanity
def test_generated_suite_is_large_enough(generated):
    scenarios, _, _ = generated
    assert len(scenarios) >= 400, (
        "the suite must be big enough for meaningful confidence intervals")


def test_generated_suite_is_reproducible():
    a = generate_suite(n_per_generator=5, seed=99)
    b = generate_suite(n_per_generator=5, seed=99)
    assert [s.id for s in a] == [s.id for s in b]
    assert [len(s.steps) for s in a] == [len(s.steps) for s in b]


def test_suite_is_balanced(generated):
    _, _, m = generated
    pos, neg = m.tp + m.fn, m.fp + m.tn
    assert pos >= 150 and neg >= 150, "need enough of both classes to score either"


def test_confidence_intervals_are_tight_enough(generated):
    """A benchmark whose intervals span 30 points cannot detect a regression."""
    _, _, m = generated
    assert m.recall_ci().width < 0.15, f"recall CI too wide: {m.recall_ci().render()}"
    assert m.fpr_ci().width < 0.15, f"FPR CI too wide: {m.fpr_ci().render()}"


def test_scenarios_have_unique_ids(generated):
    scenarios, _, _ = generated
    ids = [s.id for s in scenarios] + [s.id for s in ALL_SCENARIOS]
    assert len(ids) == len(set(ids))


def test_positives_declare_ground_truth():
    for s in POSITIVES:
        assert s.label.detector, f"{s.id} must name the detector that should catch it"
        assert s.label.onset_index is not None, f"{s.id} must declare a failure onset"


def test_hard_negatives_exist():
    assert len(NEGATIVES) >= 5, "without hard negatives the FPR is unmeasurable"


# --------------------------------------------------- regression gates (generated)
def test_recall_floor(generated):
    _, _, m = generated
    assert m.recall >= GEN["floor"]["recall"], f"recall regressed to {m.recall:.3f}"


def test_precision_floor(generated):
    _, _, m = generated
    assert m.precision >= GEN["floor"]["precision"], f"precision regressed to {m.precision:.3f}"


def test_f1_floor(generated):
    _, _, m = generated
    assert m.f1 >= GEN["floor"]["f1"], f"F1 regressed to {m.f1:.3f}"


def test_false_positive_ceiling(generated):
    """The number that decides whether anyone leaves the breaker switched on."""
    _, _, m = generated
    assert m.false_positive_rate <= GEN["ceiling"]["false_positive_rate"], (
        f"false-positive rate worsened to {m.false_positive_rate:.3f}")


def test_attribution_floor(generated):
    """Catching a loop via the spend guard produces wrong steering advice."""
    _, _, m = generated
    assert m.attribution_accuracy >= GEN["floor"]["attribution_accuracy"]


def test_no_premature_trip_regression(generated):
    """Trips landing before the failure begins would also fire on healthy runs."""
    _, _, m = generated
    assert m.detected_premature <= GEN["ceiling"]["detected_premature"]


def test_supervision_pays_for_itself(generated):
    _, _, m = generated
    assert m.net_tokens > 0, "supervision now costs more than the waste it prevents"
    assert m.net_tokens >= GEN["floor"]["net_tokens"]
    assert m.roi >= GEN["floor"]["roi"]


# --------------------------------------------- regression gates (hand-written)
def test_handwritten_no_new_misses(handwritten):
    results, _ = handwritten
    known = set(HAND["known_misses"])
    new = {r.scenario_id for r in results if r.outcome == "FN"} - known
    assert not new, f"newly missed failures: {sorted(new)}"


def test_handwritten_no_new_false_positives(handwritten):
    results, _ = handwritten
    # Entries are "scenario_id:: explanation", so split off the id.
    known = {e.split("::")[0].strip() for e in HAND["known_false_positives"]}
    new = {r.scenario_id for r in results if r.outcome == "FP"} - known
    assert not new, f"new false positives on healthy runs: {sorted(new)}"


# ------------------------------------------------------------ validity
def test_detectors_significantly_beat_random_control(generated):
    """The core validity check, after AE Studio's frequency-matched control design.

    One lucky seed proves nothing, so the control is run across many seeds and the
    difference is tested empirically.
    """
    scenarios, _, m = generated
    test, _rate = control_significance(scenarios, m,
                                       seeds=BASELINE["significance"]["seeds"])
    assert test.significant, (
        f"detectors do not significantly beat a rate-matched random control: {test.render()}")


def test_ablation_covers_every_detector(generated):
    scenarios, _, _ = generated
    _, rows = run_ablation(scenarios[:120])
    labels = {r.label for r in rows}
    for name in ("loop", "drift", "progress", "spend"):
        assert f"ablate {name}" in labels


def test_wilson_interval_sanity():
    i = wilson(7, 9)
    assert i.low < i.point < i.high
    assert 0.0 <= i.low and i.high <= 1.0
    assert wilson(0, 0).n == 0


# ------------------------------------------------------------ recovery
# Until these existed, AgentFuse's central claim - that it steers agents back on
# track - had never been measured. Detection was the only thing under test.
def test_recovery_is_actually_measured(generated):
    _, _, m = generated
    assert m.recovery_eligible > 50, (
        "not enough scenarios declare a recovery branch to measure steering at all")


def test_recovery_rate_floor(generated):
    _, _, m = generated
    assert m.recovery_rate >= GEN["floor"]["recovery_rate"], (
        f"recovery rate regressed to {m.recovery_rate:.3f} "
        f"({m.recovery_succeeded}/{m.recovery_eligible})")


def test_steering_quality_floor(generated):
    _, _, m = generated
    assert m.mean_steering_quality >= GEN["floor"]["steering_quality"]
    assert m.steering_usable_rate >= GEN["floor"]["steering_usable_rate"]


def test_vague_steering_does_not_unlock_recovery():
    """A nudge that names nothing concrete must not score as a free win."""
    from agentfuse.recovery import SteeringPath, RecoveryAction
    from evals.steering import score_steering
    vague = SteeringPath(action=RecoveryAction.INJECT,
                         instruction="Something went wrong. Try again and be careful.",
                         rationale="", confidence=0.5)
    s = score_steering(vague, original_goal="Rotate the production database credential.",
                       trip_detector="loop", trip_severity="trip", failing_tool="search_files")
    assert not s.usable, "vague steering must not count as usable guidance"


def test_good_steering_scores_usable():
    from agentfuse.recovery import SteeringPath, RecoveryAction
    from evals.steering import score_steering
    good = SteeringPath(
        action=RecoveryAction.INJECT,
        instruction=("STOP repeating `search_files` with the same arguments - it is not "
                     "advancing the task. Re-read your original objective: rotate the "
                     "production database credential. Choose a DIFFERENT tool."),
        rationale="", confidence=0.8)
    s = score_steering(good, original_goal="Rotate the production database credential.",
                       trip_detector="loop", trip_severity="trip", failing_tool="search_files")
    assert s.usable and s.diagnostic and s.goal_anchored and s.actionable


def test_critical_trips_must_escalate_not_steer():
    from agentfuse.recovery import SteeringPath, RecoveryAction
    from evals.steering import score_steering
    wrong = SteeringPath(action=RecoveryAction.INJECT, instruction="Keep going within budget.",
                         rationale="", confidence=0.5)
    s = score_steering(wrong, original_goal="x", trip_detector="spend",
                       trip_severity="critical")
    assert not s.action_appropriate
