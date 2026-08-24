"""AgentFuse — a logical circuit breaker for long-range autonomous agents.

Drop a ``CircuitBreakerMonitor`` above any agent's execution graph. It watches
the telemetry every framework already emits — tool calls, graph routes, state
changes, token spend — and trips when the agent falls into the long-horizon
failure modes: infinite tool loops, goal drift, logical traps, runaway spend.
On a trip it freezes state and climbs a fixed escalation ladder of corrections
— always separate from the agent being supervised — then injects that
correction and resumes. A *separate reasoning model* can write the correction
text instead of the deterministic ladder; this is opt-in, and measured against
every real model tested so far, it currently loses to the fixed templates
(see REPORT.md section 8.1).

Quick start::

    from agentfuse import CircuitBreakerMonitor, MonitorConfig, AgentEvent, EventType

    mon = CircuitBreakerMonitor(MonitorConfig(original_goal="..."))
    directive = mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=1,
                                       tool_name="search", tool_args={"q": "x"}))
    if directive.kind is DirectiveKind.INJECT:
        agent.add_system_message(directive.steering_text)

See ``agentfuse.adapters`` for framework-native wiring (AgentKit / OpenAI SDK /
LangGraph).
"""

from .events import AgentEvent, EventType, ExecutionSnapshot, stable_hash
from .detectors import (
    Detector,
    Trip,
    Severity,
    LoopDetector,
    DriftDetector,
    SpendDetector,
    NoProgressDetector,
)
from .recovery import RecoveryEngine, SteeringPath, RecoveryAction
from .tracer import Tracer
from .monitor import (
    CircuitBreakerMonitor,
    MonitorConfig,
    Directive,
    DirectiveKind,
)

__version__ = "0.1.0"

__all__ = [
    "AgentEvent",
    "EventType",
    "ExecutionSnapshot",
    "stable_hash",
    "Detector",
    "Trip",
    "Severity",
    "LoopDetector",
    "DriftDetector",
    "SpendDetector",
    "NoProgressDetector",
    "RecoveryEngine",
    "SteeringPath",
    "RecoveryAction",
    "Tracer",
    "CircuitBreakerMonitor",
    "MonitorConfig",
    "Directive",
    "DirectiveKind",
    "__version__",
]
