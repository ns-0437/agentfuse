"""Plain OpenAI SDK adapter.

For teams running a hand-rolled tool-use loop against the OpenAI Responses/Chat
API (no framework). ``guarded_tool_loop`` runs the loop for you and consults the
breaker each turn; if a steering directive comes back it injects the correction
as a system message and continues. This shows the breaker is framework-agnostic:
the exact same monitor powers AgentKit, LangGraph, and raw SDK code.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..confidence import _token_logprobs, summarize
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
    tool_choice: Any = "auto",
    logprobs: bool = False,
    intervention: str = "system",
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
    # Kept so the "rerun" intervention can rebuild the conversation from its
    # starting point rather than appending to a transcript full of the failure.
    baseline = list(messages)
    step = 0
    for _ in range(max_turns):
        step += 1
        # tool_choice is sent explicitly rather than left to the default.
        # OpenAI itself defaults to "auto" when tools are present, so this
        # changes nothing there — but other OpenAI-compatible servers do not.
        # llama.cpp's chatml-function-calling handler returns NO tool calls at
        # all unless it is set, which makes a tool-using agent look like one
        # that simply chose to answer in prose. Found by pointing this adapter
        # at a real local model and watching it never call a tool.
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if logprobs:
            # top_logprobs is not redundant: llama.cpp's server returns
            # logprobs=None unless it is set, which makes a live signal look
            # absent. Off by default — most callers do not want the extra
            # payload, and a detector that reads it is opt-in too.
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 1
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) if usage else 0
        tout = getattr(usage, "completion_tokens", 0) if usage else 0

        msg = choice.message
        # The agent's own confidence on THIS turn, attached to the event so a
        # supervisor can read it. This is the only place it can be captured —
        # once the response is discarded the signal is gone — and it is what
        # makes the internal/external merge measurable at all.
        meta: dict = {}
        if logprobs:
            stats = summarize(_token_logprobs(getattr(choice, "logprobs", None)))
            if stats:
                meta["confidence"] = stats
        directive = mon.observe(AgentEvent(
            type=EventType.LLM_CALL, step=step, text=msg.content or "",
            tokens_in=tin, tokens_out=tout, goal=msg.content or None,
            meta=meta,
        ))
        # Append a CLEAN assistant message rather than the raw model_dump().
        #
        # model_dump() carries provider-specific extras — refusal, audio,
        # annotations, a legacy function_call — and `content: None`. OpenAI's own
        # API accepts all of that back. Other OpenAI-compatible servers do not:
        # llama.cpp's returns a 500 with seven Pydantic validation errors,
        # because its schema requires assistant content to be a string. Echoing
        # a provider's response verbatim assumes only that provider will ever
        # read it, which is the opposite of what an adapter is for.
        assistant: dict = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant)

        outcome = _apply_directive(mon, directive, messages, intervention, baseline)
        if outcome == "stop":
            return mon.finish("escalated")
        if outcome == "restart":
            continue

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return mon.finish("complete")

        restart = False
        for tc in tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments or "{}")
            step += 1
            d = mon.observe(AgentEvent(
                type=EventType.TOOL_CALL, step=step,
                tool_name=tc.function.name, tool_args=args,
                # The SDK's own call id, carried on both the call and its
                # result. It is what lets the breaker pair them exactly when the
                # model issues several tool calls in one turn, which it does by
                # default. It must be on both events or the pair never matches.
                meta={"call_id": tc.id},
            ))
            outcome = _apply_directive(mon, d, messages, intervention, baseline)
            if outcome == "stop":
                return mon.finish("escalated")
            if outcome == "restart":
                restart = True
                break

            result = tool_router(tc.function.name, args)
            d = mon.observe(AgentEvent(
                type=EventType.TOOL_RESULT, step=step,
                tool_name=tc.function.name, text=str(result)[:200],
                state={"last_tool": tc.function.name, "result": str(result)[:200]},
                meta={"call_id": tc.id},
            ))
            # The directive from the RESULT was previously discarded. That is
            # where the loop detector now fires — it deliberately waits for the
            # outcome rather than judging a call it has not seen the result of —
            # so dropping it meant steering was generated, logged, and never
            # actually applied. The trace showed a recovery; the agent never
            # received one.
            outcome = _apply_directive(mon, d, messages, intervention, baseline)
            if outcome == "stop":
                return mon.finish("escalated")
            if outcome == "restart":
                restart = True
                break
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": str(result),
            })
        if restart:
            continue

    return mon.finish("max_turns")


def _apply_directive(mon, directive, messages, intervention: str = "system",
                     baseline: Optional[list] = None) -> str:
    """Deliver a steering correction. HOW it is delivered is a real variable.

    Measured 2026-08-16 against a real Qwen2.5-7B: appending the correction as a
    system message and continuing the conversation was obeyed **1 time in 41**.
    The instructions were the deterministic templates, the ones our own rubric
    scores at 100% "usable" — so the failure was not the wording. See REPORT.md
    §3.5.

    The AgentKit adapter, whose one captured run self-healed, differs in two
    ways at once: it delivers the steer as a USER message, and it ABORTS the
    in-flight run so the failing turns are discarded rather than accumulated.
    Those two were confounded in the only evidence we had, so both are exposed
    here separately:

      "system"     append as a system message, continue  (the 2.4% baseline)
      "user"       append as a user message, continue    (isolates the role)
      "rerun"      restart from the original prompt + steer, discarding the
                   failing turns entirely  (isolates the mechanism)
      "drop_tool"  as "user", and the caller removes the failing tool from the
                   schema  (a hard constraint rather than a request)

    A model that has just emitted the same call four times is being asked to
    contradict its own visible history; "rerun" removes that history instead of
    arguing with it.
    """
    if directive.kind is DirectiveKind.INJECT and directive.steering_text:
        text = f"[CIRCUIT BREAKER STEERING] {directive.steering_text}"
        if intervention == "rerun" and baseline is not None:
            # Discard the failing turns. Everything the model is being told to
            # stop doing is in those turns, and leaving them in place means the
            # correction has to out-argue several rounds of the agent's own
            # committed behaviour.
            messages[:] = list(baseline) + [{"role": "user", "content": text}]
            # "restart", not "continue". Rewriting history is only half of an
            # abort: the caller must also abandon the rest of THIS turn. Any
            # tool results still pending belong to an assistant message that no
            # longer exists, and appending them produces a tool message with a
            # tool_call_id nothing matches — which the API rejects outright.
            return "restart"
        elif intervention in ("user", "drop_tool"):
            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "system", "content": text})
        return "continue"
    if directive.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT):
        return "stop"
    return "continue"
