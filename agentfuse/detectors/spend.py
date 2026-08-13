"""Token / cost budget detector.

Long-running autonomy fails economically before it fails logically: a stuck
agent burns tokens indefinitely. This tracks cumulative spend and, separately,
the *burn rate* over a recent window. A hard ceiling trips CRITICAL (escalate);
a runaway burn rate trips a normal steering event so a recovery can cut the
waste before the ceiling is hit.

A note on ``max_cost_usd``, because it was decorative for most of this project's
life. Nothing populated ``AgentEvent.cost_usd``, so ``_total_cost`` stayed at 0.0
and the ceiling could not fire: measured, a monitor with ``max_cost_usd=1.0``
burned 11.94 million tokens without tripping, reporting ``$0.00`` throughout. The
detector now prices events itself from :mod:`agentfuse.pricing` when the caller
has not already done so.

Tokens whose model has no known price are counted as **unpriced**, never as free.
Treating an unknown model as $0 would rebuild the original bug exactly — spend
accumulating at zero under a ceiling that never fires — so an unenforceable cost
ceiling warns at construction and is reported in ``totals``.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Optional

from ..events import AgentEvent
from ..pricing import estimate_cost, price_for
from .base import Detector, Trip, Severity


class SpendDetector(Detector):
    name = "spend"

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        burst_window: int = 6,
        burst_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ):
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.burst_window = burst_window
        self.burst_tokens = burst_tokens
        #: Used to price events that arrive without a cost. An event may also
        #: name its own model in ``meta['model']``, which takes precedence.
        self.model = model
        self._total_tokens = 0
        self._total_cost = 0.0
        self._unpriced_tokens = 0
        self._recent: deque[int] = deque(maxlen=burst_window)

        # The common case is not a mistyped model, it is no model at all: the
        # operator sets a dollar ceiling, nothing can price the tokens, and the
        # ceiling silently never fires. Say so at construction, when it can
        # still be fixed, rather than at the end of a run that overspent.
        if max_cost_usd is not None and price_for(model) is None:
            detail = (f"no price is known for model {model!r}" if model
                      else "no model was given, so tokens cannot be priced")
            warnings.warn(
                f"max_cost_usd={max_cost_usd} is set but {detail}; the cost "
                f"ceiling CANNOT be enforced. Pass model= to MonitorConfig, set "
                f"AGENTFUSE_PRICING_FILE, or populate cost_usd on each event.",
                RuntimeWarning, stacklevel=2)

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        step_tokens = event.tokens_in + event.tokens_out
        self._total_tokens += step_tokens

        # A caller-supplied cost always wins; it knows more than a price table.
        if event.cost_usd:
            self._total_cost += event.cost_usd
        elif step_tokens:
            model = (event.meta or {}).get("model") or self.model
            cost = estimate_cost(model, event.tokens_in, event.tokens_out)
            if cost is None:
                self._unpriced_tokens += step_tokens
            else:
                self._total_cost += cost

        if step_tokens:
            self._recent.append(step_tokens)

        # Hard ceilings -> escalate, don't just steer.
        if self.max_tokens is not None and self._total_tokens >= self.max_tokens:
            return Trip(
                detector=self.name,
                severity=Severity.CRITICAL,
                reason=f"Token budget exhausted: {self._total_tokens} >= {self.max_tokens}.",
                evidence={"total_tokens": self._total_tokens, "limit": self.max_tokens},
            )
        if self.max_cost_usd is not None and self._total_cost >= self.max_cost_usd:
            return Trip(
                detector=self.name,
                severity=Severity.CRITICAL,
                reason=f"Cost budget exhausted: ${self._total_cost:.4f} >= ${self.max_cost_usd:.4f}.",
                evidence={"total_cost_usd": round(self._total_cost, 4),
                          "limit": self.max_cost_usd,
                          "unpriced_tokens": self._unpriced_tokens},
            )

        # Burn-rate guard -> steer before the ceiling.
        if self.burst_tokens is not None and len(self._recent) == self.burst_window:
            burned = sum(self._recent)
            if burned >= self.burst_tokens:
                return Trip(
                    detector=self.name,
                    severity=Severity.TRIP,
                    reason=(
                        f"Token burn rate spiking: {burned} tokens over the last "
                        f"{self.burst_window} steps (>= {self.burst_tokens})."
                    ),
                    evidence={"window_tokens": burned, "window": self.burst_window},
                )
        return None

    def reset(self) -> None:
        self._recent.clear()

    @property
    def totals(self) -> dict:
        return {
            "tokens": self._total_tokens,
            "cost_usd": round(self._total_cost, 4),
            # Non-zero means part of this run was never priced, so the dollar
            # figure is a FLOOR and any cost ceiling is under-enforced. Reported
            # rather than hidden: silence here is what made the ceiling
            # decorative in the first place.
            "unpriced_tokens": self._unpriced_tokens,
            "cost_is_complete": self._unpriced_tokens == 0,
        }
