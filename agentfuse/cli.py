"""Dependency-free diagnostics and a first-run AgentFuse demonstration."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from . import __version__
from .embedding import describe
from .events import AgentEvent, EventType
from .monitor import CircuitBreakerMonitor, DirectiveKind, MonitorConfig


def doctor() -> dict:
    """Return capabilities without importing optional runtime SDKs."""
    return {
        "agentfuse_version": __version__,
        "core": "ok",
        "drift_backend": describe(),
        "integrations": {
            "agents_sdk": importlib.util.find_spec("agents") is not None,
            "langgraph": importlib.util.find_spec("langgraph") is not None,
            "openai": importlib.util.find_spec("openai") is not None,
        },
    }


def quickstart(trace_path: Optional[str] = None) -> dict:
    """Run a small, offline loop-and-recovery scenario.

    The scenario uses the real monitor and deterministic recovery ladder. It
    makes no model or network calls and writes nothing unless ``trace_path`` is
    supplied. The returned dictionary is stable enough for installation smoke
    tests and intentionally avoids exposing internal detector objects.
    """
    goal = "Find the customer record, then produce a support summary."
    monitor = CircuitBreakerMonitor(MonitorConfig(
        original_goal=goal,
        loop_threshold=3,
        echo=False,
        jsonl_path=trace_path,
    ))

    directive = None
    for step in range(1, 4):
        monitor.observe(AgentEvent(
            type=EventType.TOOL_CALL,
            step=step,
            node="support-agent",
            tool_name="search_customers",
            tool_args={"email": "missing@example.com"},
            goal=goal,
        ))
        directive = monitor.observe(AgentEvent(
            type=EventType.TOOL_RESULT,
            step=step,
            node="support-agent",
            tool_name="search_customers",
            text="no customer found",
        ))

    tripped = directive is not None and directive.kind is DirectiveKind.INJECT
    if tripped:
        monitor.observe(AgentEvent(
            type=EventType.RESUME,
            step=4,
            node="supervisor",
            text="steering accepted; using customer id from the ticket",
        ))
        monitor.observe(AgentEvent(
            type=EventType.TOOL_RESULT,
            step=5,
            node="support-agent",
            tool_name="get_customer",
            text="customer 42 found",
            state={"customer_id": 42},
        ))
        monitor.observe(AgentEvent(
            type=EventType.COMPLETE,
            step=6,
            node="support-agent",
            text="support summary produced",
            state={"summary_ready": True},
        ))

    result = monitor.finish("complete" if tripped else "failed")
    return {
        "ok": tripped and result["status"] == "complete",
        "status": result["status"],
        "detected": "loop" if tripped else None,
        "directive": directive.kind.value if directive is not None else None,
        "recoveries": result["recoveries"],
        "steps": result["steps"],
        "trace_path": str(Path(trace_path).resolve()) if trace_path else None,
    }


def adapter_quickstart(trace_path: Optional[str] = None) -> dict:
    """Exercise the real OpenAI-compatible adapter with a scripted offline model."""
    from .adapters.openai_sdk import guarded_tool_loop

    goal = "Find customer 42 and produce a support summary."
    tool_calls: list[str] = []
    requested_tools: list[str] = []

    class ScriptedCompletions:
        def create(self, *, messages: list[dict], **kwargs: object) -> SimpleNamespace:
            steered = any("CIRCUIT BREAKER STEERING" in str(m.get("content", ""))
                          for m in messages)
            customer_found = any(m.get("role") == "tool" and "customer 42" in
                                 str(m.get("content", "")) for m in messages)
            if customer_found:
                content, calls = "Support summary ready for customer 42.", None
            else:
                name = "get_customer" if steered else "search_customers"
                requested_tools.append(name)
                arguments = {"customer_id": 42} if steered else {"email": "missing@example.com"}
                content = "Checking the customer record."
                calls = [SimpleNamespace(
                    id=f"call_{len(messages)}", function=SimpleNamespace(
                        name=name, arguments=json.dumps(arguments)))]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=content, tool_calls=calls))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=ScriptedCompletions()))

    def route(name: str, args: dict) -> str:
        tool_calls.append(name)
        if name == "search_customers":
            return "no customer found"
        if name == "get_customer" and args == {"customer_id": 42}:
            return "customer 42 found"
        raise ValueError(f"unexpected tool call: {name}")

    result = guarded_tool_loop(
        client, model="offline-scripted", system_prompt="Help the support team.",
        user_input=goal, tools=[], tool_router=route, max_turns=8,
        tool_effects={"search_customers": "read", "get_customer": "read"},
        loop_threshold=3, echo=False, jsonl_path=trace_path,
    )
    recovered = (result.get("status") == "complete"
                 and result.get("recoveries", 0) >= 1
                 and requested_tools.count("search_customers") >= 3)
    return {
        "ok": recovered and "get_customer" in tool_calls,
        "status": result["status"],
        "recoveries": result.get("recoveries", 0),
        "output": result.get("output"),
        "requested_tools": requested_tools,
        "tool_calls": tool_calls,
        "trace_path": str(Path(trace_path).resolve()) if trace_path else None,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="agentfuse")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor", help="show installed supervision capabilities")
    check.add_argument("--json", action="store_true", dest="as_json")
    demo = sub.add_parser("quickstart", help="run an offline loop-recovery smoke test")
    demo.add_argument("--json", action="store_true", dest="as_json")
    demo.add_argument("--trace", metavar="PATH", help="write the JSONL trace to PATH")
    demo.add_argument("--adapter", choices=["monitor", "openai"], default="monitor",
                      help="exercise the monitor or the OpenAI-compatible adapter")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = doctor()
        if args.as_json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(f"AgentFuse {report['agentfuse_version']} - core {report['core']}")
            print(f"Drift: {report['drift_backend']}")
            for name, available in report["integrations"].items():
                print(f"{name}: {'available' if available else 'not installed'}")
        return 0

    result = adapter_quickstart(args.trace) if args.adapter == "openai" else quickstart(args.trace)
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"AgentFuse offline quickstart ({args.adapter})")
        if result["ok"]:
            print("PASS: repeated tool loop detected")
            print("PASS: deterministic steering issued")
            print("PASS: supervised run recovered and completed")
            if args.adapter == "openai":
                print(f"Output: {result['output']}")
        else:
            print("FAIL: the expected loop-recovery path did not complete")
        if result["trace_path"]:
            print(f"Trace: {result['trace_path']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
