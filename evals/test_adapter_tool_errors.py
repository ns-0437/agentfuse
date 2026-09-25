"""Tool exceptions must preserve the adapter's safety contract."""

import pytest

from agentfuse.adapters.openai_sdk import guarded_tool_loop
from agentfuse.monitor import Directive
from evals.test_adapters_untested import FakeOpenAI, _Msg, _Resp, _ToolCall


class QuietMonitor:
    def observe(self, event):
        return Directive()

    def finish(self, status):
        return {"status": status}


def tool_turn(name):
    return _Resp(_Msg("working", [_ToolCall(name, {"id": 42}, 0)]))


@pytest.mark.parametrize("effects", [None, {"create_invoice": "write"}])
def test_unsafe_tool_exception_blocks_run_without_a_ledger(effects):
    client = FakeOpenAI([tool_turn("create_invoice"), _Resp(_Msg("done"))])
    effects = []

    def write_then_timeout(*_):
        effects.append("possibly committed")
        raise TimeoutError("secret-in-error-message")

    result = guarded_tool_loop(
        client, "fake", "Assist", "Create invoice", [], write_then_timeout,
        monitor=QuietMonitor(), max_turns=3, tool_effects=effects)
    assert result["status"] == "recovery_blocked"
    assert result["blocked_tool"] == "create_invoice"
    assert "outcome is unknown" in result["blocked_reason"]
    assert "secret-in-error-message" not in result["blocked_reason"]
    assert effects == ["possibly committed"]
    assert len(client.seen) == 1


@pytest.mark.parametrize("effect", ["read", "idempotent"])
def test_declared_safe_tool_error_reaches_agent_and_can_retry(effect):
    client = FakeOpenAI([
        tool_turn("get_invoice"), tool_turn("get_invoice"),
        _Resp(_Msg("Invoice 42 found")),
    ])
    attempts = []

    def flaky_tool(*_):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("private-api-token")
        return "invoice 42"

    result = guarded_tool_loop(
        client, "fake", "Assist", "Find invoice", [], flaky_tool,
        monitor=QuietMonitor(), max_turns=3,
        tool_effects={"get_invoice": effect})
    assert result["status"] == "complete"
    assert result["output"] == "Invoice 42 found"
    assert len(attempts) == 2
    assert any("ERROR: get_invoice failed (TimeoutError)" in str(m.get("content"))
               for m in client.seen[1])
    assert all("private-api-token" not in str(m) for turn in client.seen
               for m in turn)
