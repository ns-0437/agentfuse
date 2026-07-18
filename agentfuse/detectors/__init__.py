"""Failure-mode detectors for AgentFuse.

Each detector is an independent sensor for one long-horizon failure mode:

  * :class:`LoopDetector`       — repetitive identical tool calls
  * :class:`DriftDetector`      — interpreted goal drifting from the system prompt
  * :class:`SpendDetector`      — token / cost budget + burn-rate guard
  * :class:`NoProgressDetector` — busy activity with no state advance (logic traps)
"""

from .base import Detector, Trip, Severity
from .loop import LoopDetector
from .drift import DriftDetector
from .spend import SpendDetector
from .progress import NoProgressDetector

__all__ = [
    "Detector",
    "Trip",
    "Severity",
    "LoopDetector",
    "DriftDetector",
    "SpendDetector",
    "NoProgressDetector",
]
