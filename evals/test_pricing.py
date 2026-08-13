"""Cost accounting — is the dollar ceiling real, or decorative?

`SpendDetector` has always accepted `max_cost_usd`. Nothing ever populated
`AgentEvent.cost_usd`, so `_total_cost` stayed at 0.0 and the ceiling could not
fire. Measured before this was fixed: a monitor with `max_cost_usd=1.0` burned
**11.94 million tokens** without tripping once, reporting `$0.00` throughout.

That is the same class of failure as a restart resetting the budget — a guard
that looks armed and is not — and worse in one respect: it failed silently on
the exact axis the operator asked to be protected on.

The tests that matter here are the ones about NOT knowing a price. Treating an
unknown model as free is the convenient choice and it rebuilds the original bug
exactly, so most of what follows checks that ignorance is surfaced rather than
rounded down to zero.

    pytest evals/test_pricing.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

import pytest  # noqa: E402

from agentfuse import (  # noqa: E402
    AgentEvent, CircuitBreakerMonitor, DirectiveKind, EventType, MonitorConfig, Tracer,
)
from agentfuse import pricing  # noqa: E402
from agentfuse.detectors import SpendDetector  # noqa: E402
from agentfuse.pricing import ModelPrice, estimate_cost, price_for  # noqa: E402

GOAL = "Reconcile the ledger."


def _burn(mon, steps=400, tokens_in=50_000, tokens_out=10_000):
    for s in range(1, steps + 1):
        d = mon.observe(AgentEvent(type=EventType.LLM_CALL, step=s, node="a", text=GOAL,
                                   tokens_in=tokens_in, tokens_out=tokens_out))
        if d.kind is not DirectiveKind.CONTINUE:
            return s
    return None


# ------------------------------------------------------- the regression itself
def test_the_dollar_ceiling_actually_fires():
    """The bug this module exists for: 11.94M tokens, $1 ceiling, no trip."""
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, max_cost_usd=1.0,
                      model="gpt-4.1", drift_threshold=0.0),
        tracer=Tracer(None, False))
    assert _burn(mon) is not None, "the cost ceiling never fired"
    assert mon.spend_totals["cost_usd"] >= 1.0


def test_an_unpriceable_run_says_so_instead_of_reporting_zero():
    """Silence here is what made the ceiling decorative for months."""
    with pytest.warns(RuntimeWarning, match="CANNOT be enforced"):
        mon = CircuitBreakerMonitor(
            MonitorConfig(original_goal=GOAL, echo=False, max_cost_usd=1.0,
                          model=None, drift_threshold=0.0),
            tracer=Tracer(None, False))
    _burn(mon, steps=20)
    totals = mon.spend_totals
    assert totals["unpriced_tokens"] > 0
    assert totals["cost_is_complete"] is False, (
        "a run nobody could price must not report a complete cost")


def test_no_warning_when_the_ceiling_is_enforceable(recwarn):
    CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, max_cost_usd=1.0,
                      model="gpt-4.1"), tracer=Tracer(None, False))
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_no_warning_when_no_dollar_ceiling_is_requested(recwarn):
    """Most runs never set max_cost_usd; they must not be nagged."""
    CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                          tracer=Tracer(None, False))
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# --------------------------------------------------------------- the lookup
def test_unknown_models_return_none_not_zero():
    """`None` is load-bearing: a caller treating it as 0.0 disarms the budget."""
    assert estimate_cost("some-model-that-does-not-exist", 1_000_000, 0) is None
    assert price_for("some-model-that-does-not-exist") is None
    assert price_for(None) is None


def test_longest_prefix_wins_so_dated_releases_resolve():
    assert price_for("gpt-4.1-mini-2025-04-14") == price_for("gpt-4.1-mini")
    assert price_for("gpt-4.1-mini") != price_for("gpt-4.1")


def test_provider_prefixes_and_local_paths_resolve():
    assert price_for("openai/gpt-4o-mini") == price_for("gpt-4o-mini")
    local = price_for("models/qwen2.5-3b-instruct-q4_k_m.gguf")
    assert local is not None and local.cost(1_000_000, 1_000_000) == 0.0, (
        "a locally served model costs no API money")


def test_cost_is_split_between_input_and_output_rates():
    p = ModelPrice(input_per_1m=2.0, output_per_1m=8.0)
    assert p.cost(1_000_000, 0) == pytest.approx(2.0)
    assert p.cost(0, 1_000_000) == pytest.approx(8.0)
    assert p.cost(500_000, 250_000) == pytest.approx(1.0 + 2.0)


def test_a_caller_supplied_cost_always_wins():
    """The caller knows its real invoice; a price table is only an estimate."""
    det = SpendDetector(model="gpt-4.1")
    det.inspect(AgentEvent(type=EventType.LLM_CALL, step=1,
                           tokens_in=1_000_000, tokens_out=0, cost_usd=99.0), [])
    assert det.totals["cost_usd"] == pytest.approx(99.0)


def test_per_event_model_overrides_the_run_default():
    """A supervisor and its agent are usually different models."""
    det = SpendDetector(model="gpt-4.1")
    det.inspect(AgentEvent(type=EventType.LLM_CALL, step=1, tokens_in=1_000_000,
                           tokens_out=0, meta={"model": "gpt-4.1-nano"}), [])
    assert det.totals["cost_usd"] == pytest.approx(
        price_for("gpt-4.1-nano").cost(1_000_000, 0))


# ------------------------------------------------------------- staying current
def test_prices_can_be_overridden_without_a_code_change(tmp_path, monkeypatch):
    """A hardcoded table in a library is wrong the moment it ships."""
    f = tmp_path / "prices.json"
    f.write_text(json.dumps({"my-private-model":
                             {"input_per_1m": 1.0, "output_per_1m": 3.0}}))
    monkeypatch.setenv("AGENTFUSE_PRICING_FILE", str(f))
    monkeypatch.setattr(pricing, "_overrides_loaded", False)
    try:
        assert estimate_cost("my-private-model", 1_000_000, 0) == pytest.approx(1.0)
    finally:
        pricing.PRICES.pop("my-private-model", None)


def test_a_malformed_override_does_not_break_the_run(tmp_path, monkeypatch):
    f = tmp_path / "prices.json"
    f.write_text("{ this is not json")
    monkeypatch.setenv("AGENTFUSE_PRICING_FILE", str(f))
    monkeypatch.setattr(pricing, "_overrides_loaded", False)
    assert price_for("gpt-4.1") is not None, "bundled defaults must survive"


def test_the_table_is_dated_because_it_goes_stale():
    """If this is ever removed, nobody can tell how old the numbers are."""
    assert pricing.PRICES_AS_OF, "the bundled price table must carry an as-of date"
