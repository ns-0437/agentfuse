"""CI gate for the AgentFuse eval suite.

The suite currently has real, documented weaknesses (see ``evals/baseline.json``
and ``evals/results/REPORT.md``). These tests do not pretend otherwise. What they
enforce is that the numbers never get *worse* without someone noticing, and that
the detectors keep beating a rate-matched random control.

    pytest evals/test_eval.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.ablation import run_ablation  # noqa: E402
from evals.metrics import score  # noqa: E402
from evals.runner import run_suite  # noqa: E402
from evals.scenarios import ALL_SCENARIOS, POSITIVES, NEGATIVES  # noqa: E402

BASELINE = json.loads((ROOT / "evals" / "baseline.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def suite():
    by_id = {s.id: s for s in ALL_SCENARIOS}
    results = run_suite(ALL_SCENARIOS)
    return results, score(results, by_id)


# --------------------------------------------------------------- sanity
def test_every_scenario_runs(suite):
    results, _ = suite
    assert len(results) == len(ALL_SCENARIOS)
    assert all(r.outcome in ("TP", "FP", "FN", "TN") for r in results)


def test_suite_has_hard_negatives():
    """A benchmark without hard negatives cannot measure false positives."""
    assert len(NEGATIVES) >= 5, "need a meaningful set of healthy-run scenarios"
    assert len(POSITIVES) >= 5


def test_scenarios_have_unique_ids():
    ids = [s.id for s in ALL_SCENARIOS]
    assert len(ids) == len(set(ids))


def test_positives_declare_onset_and_detector():
    for s in POSITIVES:
        assert s.label.detector, f"{s.id} must name the detector that should catch it"
        assert s.label.onset_index is not None, f"{s.id} must declare a failure onset"


# ------------------------------------------------------- regression gates
def test_recall_not_below_floor(suite):
    _, m = suite
    assert m.recall >= BASELINE["floor"]["recall"], (
        f"recall regressed to {m.recall:.3f} (floor {BASELINE['floor']['recall']})")


def test_precision_not_below_floor(suite):
    _, m = suite
    assert m.precision >= BASELINE["floor"]["precision"], (
        f"precision regressed to {m.precision:.3f}")


def test_false_positive_rate_not_worse(suite):
    _, m = suite
    assert m.false_positive_rate <= BASELINE["ceiling"]["false_positive_rate"], (
        f"false-positive rate worsened to {m.false_positive_rate:.3f}")


def test_net_token_benefit_stays_positive(suite):
    """Supervision must cost less than the waste it prevents, or it's a liability."""
    _, m = suite
    assert m.net_tokens > 0, "supervision now costs more than it saves"
    assert m.net_tokens >= BASELINE["floor"]["net_tokens"]


def test_no_new_false_positives(suite):
    results, _ = suite
    known = set(BASELINE["known_false_positives"])
    new = {r.scenario_id for r in results if r.outcome == "FP"} - known
    assert not new, f"new false positives on healthy runs: {sorted(new)}"


def test_no_new_misses(suite):
    results, _ = suite
    known = set(BASELINE["known_misses"])
    new = {r.scenario_id for r in results if r.outcome == "FN"} - known
    assert not new, f"newly missed failures: {sorted(new)}"


def test_no_new_premature_trips(suite):
    _, m = suite
    assert m.detected_premature <= BASELINE["ceiling"]["detected_premature"], (
        "more detectors are now firing before the failure begins")


# ------------------------------------------------------------- ablation
def test_detectors_beat_random_control():
    """The core validity check, after AE Studio's matched-control design.

    If a rate-matched random detector scores as well as ours, our F1 is an
    artefact of how often we trip, not of what we detect.
    """
    full, rows = run_ablation(ALL_SCENARIOS)
    control = [r for r in rows if r.label.startswith("random control")][0]
    assert full.f1 > control.metrics.f1, (
        f"detectors ({full.f1:.3f}) do not beat random control ({control.metrics.f1:.3f})")


def test_ablation_reports_every_detector():
    _, rows = run_ablation(ALL_SCENARIOS)
    labels = {r.label for r in rows}
    for name in ("loop", "drift", "progress", "spend"):
        assert f"ablate {name}" in labels
