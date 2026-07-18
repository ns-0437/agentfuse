"""Plain OpenAI SDK adapter.

For teams running a hand-rolled tool-use loop against the OpenAI Responses/Chat
API (no framework). ``guarded_tool_loop`` runs the loop for you and consults the
breaker each turn; if a steering directive comes back it injects the correction
as a system message and continues. This shows the breaker is framework-agnostic:
the exact same monitor powers AgentKit, LangGraph, and raw SDK code.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..events import AgentEvent, EventType
from ..monitor import CircuitBreakerMonitor, MonitorConfig, DirectiveKind


def guarded_tool_loop(
    client: Any,
    model: str,
    system_prompt: str,
    user_input: str,
    tools: list[dict],
    tool_router: Callable[[str, dict], Any],
    max_turns: int = 40,
    monitor: Optional[CircuitBreakerMonitor] = None,
    **config_kwargs: Any,
) -> dict:
    """Run a guarded manual tool-use loop against the OpenAI Chat Completions API.

    ``tool_router(name, args)`` executes a tool and returns its result. Pricing
    for spend tracking can be passed via ``config_kwargs`` (max_tokens/max_cost).
    """
    mon = monitor or CircuitBreakerMonitor(
        MonitorConfig(original_goal=system_prompt, **config_kwargs)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    step = 0
    for _ in range(max_turns):
        step += 1
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) if usage else 0
        tout = getattr(usage, "completion_tokens", 0) if usage else 0

        msg = choice.message
        directive = mon.observe(AgentEvent(
            type=EventType.LLM_CALL, step=step, text=msg.content or "",
            tokens_in=tin, tokens_out=tout, goal=msg.content or None,
        ))
        messages.append(msg.model_dump())

        if _apply_directive(mon, directive, messages) == "stop":
            return mon.finish("escalated")

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return mon.finish("complete")

        for tc in tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments or "{}")
            step += 1
            d = mon.observe(AgentEvent(
                type=EventType.TOOL_CALL, step=step,
                tool_name=tc.function.name, tool_args=args,
            ))
            if _apply_directive(mon, d, messages) == "stop":
                return mon.finish("escalated")

            result = tool_router(tc.function.name, args)
            mon.observe(AgentEvent(
                type=EventType.TOOL_RESULT, step=step,
                tool_name=tc.function.name, text=str(result)[:200],
                state={"last_tool": tc.function.name, "result": str(result)[:200]},
            ))
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": str(result),
            })

    return mon.finish("max_turns")


def _apply_directive(mon, directive, messages) -> str:
    if directive.kind is DirectiveKind.INJECT and directive.steering_text:
        messages.append({"role": "system", "content": f"[CIRCUIT BREAKER STEERING] {directive.steering_text}"})
        return "continue"
    if directive.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT):
        return "stop"
    return "continue"
