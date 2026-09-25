"""Execution guarantees from the independent September reliability review."""
import os
import pytest
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


@pytest.mark.parametrize("effects", [None, {"create_invoice": "write"}])
def test_sdk_does_not_rerun_after_external_write(effects):
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
        tool_effects=effects, monitor=TripAfterResult(),
        max_turns=6, echo=False)
    assert result["status"] == "recovery_blocked"
    assert result["blocked_tool"] == "create_invoice"
    assert len(writes) == 1


@pytest.mark.parametrize("trip_on", [EventType.LLM_CALL, EventType.TOOL_CALL])
def test_sdk_blocks_later_rerun_after_unknown_tool_effect(trip_on):
    from evals.test_adapters_untested import FakeOpenAI, _loop_turns
    from agentfuse.adapters.openai_sdk import guarded_tool_loop
    from agentfuse.monitor import Directive

    class TripOnSecondTurn:
        def __init__(self):
            self.calls = 0

        def observe(self, event):
            if event.type is EventType.LLM_CALL:
                self.calls += 1
            if self.calls == 2 and event.type is trip_on:
                return Directive(DirectiveKind.INJECT, steering_text="change plan")
            return Directive()

        def finish(self, status):
            return {"status": status}

    writes = []
    result = guarded_tool_loop(FakeOpenAI(_loop_turns(tool="create_invoice")),
        "gpt-4o", "Assist", "Task", [],
        lambda *_: writes.append("committed") or "invoice created",
        monitor=TripOnSecondTurn(), max_turns=4)
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


def test_config_normalizes_mode_strings_and_rejects_bad_values():
    import pytest
    from typing import Any, cast

    config = MonitorConfig(original_goal="task", mode=cast(Any, "observe"),
                           echo=False, max_tokens=1)
    assert config.mode is MonitorMode.OBSERVE
    decision = CircuitBreakerMonitor(config).observe(
        AgentEvent(type=EventType.LLM_CALL, tokens_in=2))
    assert decision.kind is DirectiveKind.CONTINUE

    with pytest.raises(ValueError, match="original_goal"):
        MonitorConfig(original_goal="   ")
    with pytest.raises(ValueError, match="checkpoint_every"):
        MonitorConfig(original_goal="task", checkpoint_every=0)
    with pytest.raises(ValueError, match="mode must be one of"):
        MonitorConfig(original_goal="task", mode=cast(Any, "shadow"))
    with pytest.raises(ValueError, match="max_cost_usd"):
        MonitorConfig(original_goal="task", max_cost_usd=-1.0)


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


def test_quickstart_proves_recovery_and_can_write_a_trace(tmp_path, capsys):
    import json
    from agentfuse.cli import main
    trace = tmp_path / "nested" / "quickstart.jsonl"
    assert main(["quickstart", "--json", "--trace", str(trace)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["detected"] == "loop"
    assert result["directive"] == "inject"
    assert result["status"] == "complete"
    assert result["recoveries"] == 1
    assert trace.exists()
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert any(record.get("kind") == "trip" for record in records)
    assert any(record.get("kind") == "recovery" for record in records)


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
