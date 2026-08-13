"""Token → dollar conversion, so the cost ceiling is not decorative.

``SpendDetector`` has always accepted ``max_cost_usd``. Nothing ever populated
``AgentEvent.cost_usd``, so ``_total_cost`` stayed at 0.0 for the life of every
run and the ceiling could not fire. Measured before this module existed: a
monitor with ``max_cost_usd=1.0`` burned **11.94 million tokens** without
tripping once, reporting ``$0.00`` throughout.

That is the same failure as a restart resetting the budget — a guard that looks
armed and is not — and it is arguably worse, because it fails silently on the
axis the operator explicitly asked to be protected on.

Unknown models are not free
---------------------------
The important decision here is what to do about a model we have no price for.
Returning ``0.0`` would be convenient and would recreate the exact bug: spend
accumulates at zero, the ceiling never fires, and nothing says so.
:func:`estimate_cost` therefore returns ``None`` for an unpriced model, and
``SpendDetector`` treats "I cannot price this" as a condition to surface rather
than as "this is free".

These prices go stale
---------------------
This table is a **convenience default, not a source of truth.** Provider pricing
changes without notice, and a hardcoded table in a library is wrong the moment it
ships. It is dated, and overriding it takes no code change:

    AGENTFUSE_PRICING_FILE=my_prices.json

    {"gpt-4.1": {"input_per_1m": 2.0, "output_per_1m": 8.0}}

Anyone relying on the dollar ceiling for real money should set that file from
their provider's current published rates and treat the bundled numbers as a
placeholder. Costs computed here are estimates for *guardrail* purposes — they
are not billing figures and will not reconcile with an invoice.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    input_per_1m: float
    output_per_1m: float

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_1m + tokens_out * self.output_per_1m) / 1e6


#: When the bundled table was last checked. Read this before trusting a number.
PRICES_AS_OF = "2026-05"

#: Keyed by model-name PREFIX, longest match wins, so dated releases like
#: ``gpt-4.1-2025-04-14`` resolve without needing their own entry.
PRICES: dict[str, ModelPrice] = {
    "gpt-4.1-nano":    ModelPrice(0.10, 0.40),
    "gpt-4.1-mini":    ModelPrice(0.40, 1.60),
    "gpt-4.1":         ModelPrice(2.00, 8.00),
    "gpt-4o-mini":     ModelPrice(0.15, 0.60),
    "gpt-4o":          ModelPrice(2.50, 10.00),
    "o4-mini":         ModelPrice(1.10, 4.40),
    "o3-mini":         ModelPrice(1.10, 4.40),
    "o3":              ModelPrice(2.00, 8.00),
    # Anything served locally costs no API money. Electricity is real but it is
    # not what a spend ceiling is protecting against.
    "local":           ModelPrice(0.0, 0.0),
    "qwen":            ModelPrice(0.0, 0.0),
    "llama":           ModelPrice(0.0, 0.0),
    "mock":            ModelPrice(0.0, 0.0),
}

_overrides_loaded = False


def _load_overrides() -> None:
    """Merge ``AGENTFUSE_PRICING_FILE`` over the bundled table, once."""
    global _overrides_loaded
    if _overrides_loaded:
        return
    _overrides_loaded = True
    path = os.getenv("AGENTFUSE_PRICING_FILE")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for name, entry in data.items():
            PRICES[name] = ModelPrice(float(entry["input_per_1m"]),
                                      float(entry["output_per_1m"]))
    except Exception:
        # A malformed override must not take down the run. The consequence is
        # that the bundled defaults apply, which unknown-model handling below
        # will surface if it matters.
        pass


def price_for(model: Optional[str]) -> Optional[ModelPrice]:
    """Longest-prefix price lookup, or ``None`` when the model is unknown."""
    if not model:
        return None
    _load_overrides()
    name = model.strip().lower()
    # Strip a provider prefix such as "openai/gpt-4.1" or a local file path.
    for sep in ("/", "\\"):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    best: Optional[str] = None
    for key in PRICES:
        if name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is None:
        # Substring fallback catches "models/qwen2.5-3b-instruct-q4_k_m.gguf".
        for key in PRICES:
            if key in name and (best is None or len(key) > len(best)):
                best = key
    return PRICES[best] if best else None


def estimate_cost(model: Optional[str], tokens_in: int,
                  tokens_out: int) -> Optional[float]:
    """USD for one call, or ``None`` when the model has no known price.

    ``None`` is deliberate and load-bearing: a caller that treats it as ``0.0``
    silently disarms whatever budget depends on it.
    """
    price = price_for(model)
    return None if price is None else price.cost(tokens_in, tokens_out)


def known_models() -> list[str]:
    _load_overrides()
    return sorted(PRICES)
