"""RecoveryEngine's backend auto-selection — the one place a silent choice bills money.

Before this fix, ``RecoveryEngine(backend=None)`` (which is exactly what a
plain ``CircuitBreakerMonitor(config)`` constructs, since nothing in the
public examples passes an explicit ``recovery=``) chose ``backend="real"``
whenever ``OPENAI_API_KEY`` happened to be set in the environment — with no
other signal. That variable is the single most common one an integrator's
OWN agent already needs for something that has nothing to do with
AgentFuse's recovery step, so its mere presence is not consent to (a) billed
API calls or (b) the measurably worse recovery ladder (REPORT.md
3.4/4.12/8.1: the reasoning-model backend loses to the deterministic
templates at every model size tested).

These tests pin the fix: an unrelated key must never silently upgrade the
backend, and both deliberate opt-in paths (a self-hosted base_url, or the
explicit AGENTFUSE_RECOVERY_BACKEND flag) still work.

    pytest evals/test_recovery_backend.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import CircuitBreakerMonitor, MonitorConfig  # noqa: E402
from agentfuse.recovery import RecoveryEngine  # noqa: E402


def _clear_recovery_env(monkeypatch):
    """Isolate the exact signals RecoveryEngine's auto-selection reads."""
    for key in ("OPENAI_API_KEY", "AGENTFUSE_OFFLINE", "AGENTFUSE_RECOVERY_BACKEND",
               "AGENTFUSE_LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


def test_an_unrelated_api_key_does_not_silently_upgrade_the_backend(monkeypatch):
    """The regression this file exists to prevent."""
    _clear_recovery_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-the-users-OWN-agent-needs")
    eng = RecoveryEngine()
    assert eng.backend == "mock", (
        "an OPENAI_API_KEY set for an unrelated reason must not silently opt "
        "the user into billed calls and the measurably worse reasoning-model "
        "recovery backend")


def test_with_no_signals_at_all_the_default_is_mock(monkeypatch):
    _clear_recovery_env(monkeypatch)
    eng = RecoveryEngine()
    assert eng.backend == "mock"


def test_explicit_backend_argument_is_never_overridden(monkeypatch):
    _clear_recovery_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    eng = RecoveryEngine(backend="mock")
    assert eng.backend == "mock", "an explicit backend= must win regardless of env"


def test_agentfuse_recovery_backend_env_var_is_a_real_explicit_opt_in(monkeypatch):
    """The decision logic, isolated from whether `openai` happens to be
    installed. `_make_client` gracefully falls back to mock when the SDK is
    genuinely unavailable (correct: CI's own "test" job installs the package
    with zero extras specifically to verify the stdlib-only core, so `openai`
    is NOT installed there -- this test found that gap live, failing in CI
    while passing locally where openai happens to be installed for other
    tests). Stubbing _make_client isolates "which env var wins" from "is the
    SDK importable", which is the thing this test actually means to check.
    """
    _clear_recovery_env(monkeypatch)
    monkeypatch.setattr(RecoveryEngine, "_make_client", lambda self: object())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-but-syntactically-present")
    monkeypatch.setenv("AGENTFUSE_RECOVERY_BACKEND", "real")
    eng = RecoveryEngine()
    assert eng.backend == "real", (
        "the dedicated, AgentFuse-specific opt-in must still work -- this is "
        "the intentional replacement for the old OPENAI_API_KEY auto-detect")


def test_a_self_hosted_base_url_is_still_a_deliberate_signal(monkeypatch):
    """AGENTFUSE_LLM_BASE_URL is AgentFuse-specific -- nobody sets it by accident.

    _make_client stubbed for the same reason as the test above: this checks
    which signal the decision logic honours, not whether openai is installed.
    """
    _clear_recovery_env(monkeypatch)
    monkeypatch.setattr(RecoveryEngine, "_make_client", lambda self: object())
    monkeypatch.setenv("AGENTFUSE_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    eng = RecoveryEngine()
    assert eng.backend == "real"


def test_offline_mode_still_forces_mock_even_with_explicit_opt_in(monkeypatch):
    """AGENTFUSE_OFFLINE is the hard override -- belt and braces against billing."""
    _clear_recovery_env(monkeypatch)
    monkeypatch.setenv("AGENTFUSE_OFFLINE", "1")
    monkeypatch.setenv("AGENTFUSE_RECOVERY_BACKEND", "real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    eng = RecoveryEngine()
    assert eng.backend == "mock", (
        "AGENTFUSE_OFFLINE must be the hard override no other signal can beat")


def test_the_actual_end_to_end_path_a_real_user_hits(monkeypatch):
    """CircuitBreakerMonitor(config) with no explicit recovery=, exactly what
    every public example constructs -- not RecoveryEngine directly. This
    file's own module docstring claims this is "exactly what a plain
    CircuitBreakerMonitor(config) constructs"; this test is what actually
    checks that claim end-to-end instead of only testing RecoveryEngine in
    isolation.
    """
    _clear_recovery_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-the-users-OWN-agent-needs")
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal="test goal", echo=False))
    assert mon.recovery.backend == "mock", (
        "the real construction path a user actually calls must resolve to "
        "mock, not just RecoveryEngine() in isolation")
