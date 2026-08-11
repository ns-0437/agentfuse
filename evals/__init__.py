"""AgentFuse evaluation harness.

Answers the question the project could not previously answer: *does the breaker
actually work?* Scenarios carry ground truth, hard negatives measure the
false-positive rate, and an ablation study (after AE Studio's ESR methodology)
establishes each detector's causal contribution against a rate-matched random
control.

Run it:  ``python evals/run_eval.py --json``
"""

from .schema import Scenario, StepSpec, Label, ScenarioResult, CostModel, tool, think
from .metrics import Metrics, score, family_rates
from .runner import run_scenario, run_suite
from .ablation import run_ablation, RandomControlDetector

__all__ = [
    "Scenario", "StepSpec", "Label", "ScenarioResult", "CostModel", "tool", "think",
    "Metrics", "score", "family_rates",
    "run_scenario", "run_suite",
    "run_ablation", "RandomControlDetector",
]
