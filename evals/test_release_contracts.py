"""Execution guarantees from the independent September reliability review."""
import os
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse import AgentEvent, CircuitBreakerMonitor, EventType, MonitorConfig
from agentfuse.monitor import DirectiveKind, MonitorMode


def test_sdk_returns_the_agent_output_with_run_summary():
    from evals.test_adapters_untested import FakeOpenAI, _Resp, _Msg
    from agentfuse.adapters.openai_sdk import guarded_tool_loop
    result = guarded_tool_loop(FakeOpenAI([_Resp(_Msg("Invoice reconciled"))]),
        "gpt-4o", "Assist", "Task", [], lambda *_: None, echo=False)
    assert result["status"] == "complete"
    assert result["output"] == "Invoice reconciled"


def test_sdk_does_not_rerun_after_declared_external_write():
    from evals.test_adapters_untested import FakeOpenAI, _loop_turns
    from agentfuse.adapters.openai_sdk import guarded_tool_loop
    from agentfuse.monitor import Directive
    class TripAfterResult:
        def observe(self, event):
            if event.type is EventType.TOOL_RESULT:
                return Directive(DirectiveKind.INJECT, steering_text="change plan")
            return Directive()
        def finish(self, status):
            return {"status": status}
    writes = []
    result = guarded_tool_loop(FakeOpenAI(_loop_turns(tool="create_invoice")),
        "gpt-4o", "Assist", "Task", [],
        lambda *_: writes.append("committed") or "invoice created",
        tool_effects={"create_invoice": "write"}, monitor=TripAfterResult(),
        max_turns=6, echo=False)
    assert result["status"] == "recovery_blocked"
    assert result["blocked_tool"] == "create_invoice"
    assert len(writes) == 1


def test_monitor_modes_separate_observation_enforcement_and_recovery():
    def decision(mode):
        mon = CircuitBreakerMonitor(MonitorConfig(original_goal="task", mode=mode,
            echo=False, max_tokens=1))
        return mon.observe(AgentEvent(type=EventType.LLM_CALL, tokens_in=2))
    assert decision(MonitorMode.OBSERVE).kind is DirectiveKind.CONTINUE
    assert decision(MonitorMode.ENFORCE).kind is DirectiveKind.ABORT
    assert decision(MonitorMode.RECOVER).kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT)


def test_long_runs_keep_bounded_working_history():
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal="task", echo=False,
        mode=MonitorMode.OBSERVE, history_limit=10, route_history_limit=3), detectors=[])
    for step in range(30):
        mon.observe(AgentEvent(type=EventType.ROUTE, step=step, node=f"node-{step}"))
    assert len(mon.history) == 10
    assert mon.history[-1].step == 29
    assert mon.route_history == ["node-27", "node-28", "node-29"]


def test_doctor_reports_core_and_machine_readable_capabilities(capsys):
    from agentfuse.cli import main
    assert main(["doctor", "--json"]) == 0
    report = __import__("json").loads(capsys.readouterr().out)
    assert report["core"] == "ok"
    assert "drift_backend" in report
    assert set(report["integrations"]) == {"agents_sdk", "langgraph", "openai"}


def test_sdk_progress_requires_application_evidence():
    import asyncio
    import pytest
    from types import SimpleNamespace as NS
    pytest.importorskip("agents")
    from agentfuse.adapters.agentkit_hooks import FuseRunHooks
    hooks = FuseRunHooks("Create invoice", echo=False)
    asyncio.run(hooks.on_tool_end(None, NS(name="a"), NS(name="create"),
                                 "ERROR secret-not-found"))
    assert hooks.monitor.history[-1].state is None
    hooks.progress_validator = lambda name, result: {"invoice_id": result["id"]}
    asyncio.run(hooks.on_tool_end(None, NS(name="a"), NS(name="create"), {"id": 123}))
    assert hooks.monitor.history[-1].state == {"invoice_id": 123}


def test_real_agents_tool_pairs_retain_call_ids():
    import pytest
    pytest.importorskip("agents")
    from evals.test_adapters import _drive
    hooks, output, _ = _drive(echo=False)
    calls = {e.meta["call_id"]: e.tool_name for e in hooks.monitor.history
             if e.type is EventType.TOOL_CALL}
    results = [e for e in hooks.monitor.history if e.type is EventType.TOOL_RESULT]
    assert results and all(e.meta["call_id"] for e in results)
    assert all(calls[e.meta["call_id"]] == e.tool_name for e in results)
