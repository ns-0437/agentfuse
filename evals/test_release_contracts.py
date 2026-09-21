"""Execution guarantees from the independent September reliability review."""
import os
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse import AgentEvent, CircuitBreakerMonitor, EventType, MonitorConfig
from agentfuse.detectors import DriftDetector, SpendDetector
from agentfuse.monitor import DirectiveKind, MonitorMode


def test_hard_budget_wins_over_drift_and_counts_every_event():
    spend = SpendDetector(max_tokens=150)
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal="database", echo=False),
        detectors=[DriftDetector(original_goal="database", threshold=0.9), spend])
    for step in (1, 2):
        decision = mon.observe(AgentEvent(type=EventType.LLM_CALL, step=step,
            text="gardening roses", goal="gardening roses", tokens_in=100))
    assert spend.totals["tokens"] == mon.total_tokens == 200
    assert decision.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT)


def test_sdk_forwards_model_for_dollar_limit():
    from evals.test_adapters_untested import FakeOpenAI, _Resp, _Msg
    from agentfuse.adapters.openai_sdk import guarded_tool_loop
    result = guarded_tool_loop(FakeOpenAI([_Resp(_Msg("Done"), tin=1000, tout=1000)]),
        "gpt-4o", "Assist", "Task", [], lambda *_: None,
        max_cost_usd=0.000001, echo=False)
    assert result["status"] == "escalated"
    assert result["unpriced_tokens"] == 0
    assert result["total_cost_usd"] > 0


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


def test_sdk_anchors_to_user_task_with_explicit_override(monkeypatch):
    from evals.test_adapters_untested import FakeOpenAI, _Resp, _Msg
    from agentfuse.adapters import openai_sdk
    goals = []
    real = openai_sdk.CircuitBreakerMonitor
    def capture(config):
        goals.append(config.original_goal)
        return real(config)
    monkeypatch.setattr(openai_sdk, "CircuitBreakerMonitor", capture)
    for options in ({}, {"original_goal": "explicit task"}):
        openai_sdk.guarded_tool_loop(FakeOpenAI([_Resp(_Msg("Done"))]),
            "gpt-4o", "You are helpful", "Reconcile invoices", [], lambda *_: None,
            echo=False, **options)
    assert goals == ["Reconcile invoices", "explicit task"]


def test_langgraph_registered_callback_observes_real_tool():
    import pytest
    pytest.importorskip("langchain_core")
    from langchain_core.tools import StructuredTool
    from agentfuse.adapters.langgraph import FuseCallbackHandler
    handler = FuseCallbackHandler("read value", echo=False)
    tool = StructuredTool.from_function(lambda value: value, name="read",
                                       description="Read a value")
    assert tool.invoke({"value": "hello"}, config={"callbacks": [handler]}) == "hello"
    assert [e.type for e in handler.monitor.history] == [EventType.TOOL_CALL, EventType.TOOL_RESULT]


def test_langgraph_interleaved_results_keep_identity():
    from agentfuse.adapters.langgraph import FuseCallbackHandler
    handler = FuseCallbackHandler("task", echo=False)
    handler.on_tool_start({"name": "a"}, "{}", run_id="a1")
    handler.on_tool_start({"name": "b"}, "{}", run_id="b1")
    handler.on_tool_end("a result", run_id="a1")
    handler.on_tool_error(ValueError("failed"), run_id="b1")
    results = handler.monitor.history[-2:]
    assert [(e.tool_name, e.meta["call_id"]) for e in results] == [("a", "a1"), ("b", "b1")]
    assert handler._inflight == {}


def test_langgraph_halt_prevents_next_real_tool_dispatch():
    import pytest
    pytest.importorskip("langchain_core")
    from langchain_core.tools import StructuredTool
    from agentfuse.adapters.langgraph import FuseCallbackHandler, FuseHalt
    from agentfuse.monitor import Directive
    class Stop:
        def observe(self, event):
            return Directive(DirectiveKind.ABORT)
    calls = []
    handler = FuseCallbackHandler("task", monitor=Stop())
    tool = StructuredTool.from_function(lambda value: calls.append(value),
                                       name="write", description="Write a value")
    for _ in range(2):
        with pytest.raises(FuseHalt):
            tool.invoke({"value": "x"}, config={"callbacks": [handler]})
    assert calls == []
    with pytest.raises(FuseHalt):
        handler.supervisor_node({})


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
