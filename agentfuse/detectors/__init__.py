"""Failure-mode detectors for AgentFuse.

Each detector is an independent sensor for one long-horizon failure mode:

  * :class:`LoopDetector`       — repetitive identical tool calls
  * :class:`DriftDetector`      — interpreted goal drifting from the system prompt
  * :class:`SpendDetector`      — token / cost budget + burn-rate guard
  * :class:`NoProgressDetector` — busy activity with no state advance (logic traps)
  * :class:`RateOfProgressDetector` — state advancing on every step, converging
    on none (the Zeno trap, which the binary progress test cannot see)
"""

from .base import Detector, Trip, Severity
from .loop import LoopDetector
from .drift import DriftDetector
from .spend import SpendDetector
from .progress import NoProgressDetector
from .rate import RateOfProgressDetector

__all__ = [
    "Detector",
    "Trip",
    "Severity",
    "LoopDetector",
    "DriftDetector",
    "SpendDetector",
    "NoProgressDetector",
    "RateOfProgressDetector",
]
