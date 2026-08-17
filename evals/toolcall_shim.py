"""Recover tool calls that llama.cpp's server leaves sitting in `content`.

`probe_termination.py` established that neither llama-cpp-python serving config
can capture a faithful agent trace:

    chatml-function-calling   parses tool calls, but CANNOT TERMINATE. Handed the
                              answer in the transcript it calls the tool again,
                              so every run ends in repetition and every capture
                              looks like a loop.
    native chat template      terminates correctly, but emits its tool calls as
                              raw `<tool_call>{...}</tool_call>` text that the
                              server never parses into `tool_calls`, so the agent
                              never acts.

This shim takes the second config and fixes the one thing wrong with it. The
model's output is already correct — the right function name, the right arguments
— and only the server's parsing step is missing. Reconstructing the call from
that text is therefore *fidelity*, not distortion: it recovers what the model
actually said. Nothing about the model's behaviour is altered, which is the whole
requirement for a capture rig, since anything the rig invents becomes a fact in
the benchmark.

It lives in `evals/` rather than in `agentfuse/adapters/` deliberately. This is a
workaround for one server's bug, and putting server-specific text-scraping on the
public adapter surface would make every future llama.cpp change a breaking change
for library users. Capture infrastructure can carry that risk; the shipped
adapter should not.

Usage — wrap the client, change nothing else:

    client = ToolCallShim(OpenAI(base_url=..., api_key="not-needed"))
    guarded_tool_loop(client, ...)
"""

from __future__ import annotations

import json
import re
from typing import Any

# Non-greedy, and tolerant of a missing closing tag: a truncated generation still
# carries a usable call, and dropping it would silently turn a tool-using turn
# into a "the agent stopped" datapoint.
_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def _loads_lenient(blob: str) -> Any:
    """Parse a tool-call payload, tolerating the template's doubled braces.

    Qwen's bundled template renders `{{"name": ...}}` — an escaping bug in the
    jinja source that doubles the outer braces. Left unhandled it makes every
    tool call unparseable, which would look exactly like a model that emits
    malformed calls.
    """
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    if blob.startswith("{{") and blob.endswith("}}"):
        try:
            return json.loads(blob[1:-1])
        except json.JSONDecodeError:
            pass
    return None


def parse_tool_calls(content: str) -> tuple[list[dict], str]:
    """Return (calls, remaining_prose) extracted from a content string.

    `remaining_prose` matters: a turn can legitimately carry both reasoning and a
    call, and discarding the text would corrupt the LLM_CALL event the drift
    detector reads.
    """
    if not content or "<tool_call>" not in content:
        return [], content or ""

    calls: list[dict] = []
    for i, blob in enumerate(_BLOCK.findall(content)):
        obj = _loads_lenient(blob.strip())
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        # The adapter does json.loads(arguments), so this must be a STRING.
        # Passing a dict raises TypeError deep inside the tool loop.
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {})
        calls.append({"id": f"call_{i + 1}", "name": name, "arguments": args})

    prose = _BLOCK.sub("", content).strip()
    return calls, prose


class ToolCallShim:
    """Wraps an OpenAI client so `<tool_call>` text becomes real `tool_calls`.

    Only the response is rewritten. Requests pass through untouched, and so do
    `usage` and `logprobs`, because the confidence detector reads them and a
    rebuilt response object would quietly drop them.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _Chat(client)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _Chat:
    def __init__(self, client: Any) -> None:
        self.completions = _Completions(client)


class _Completions:
    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        resp = self._client.chat.completions.create(**kwargs)
        for choice in getattr(resp, "choices", []) or []:
            _patch_choice(choice)
        return resp


def _patch_choice(choice: Any) -> None:
    """Mutate one choice in place, preserving every other field.

    In-place mutation rather than reconstruction is deliberate: rebuilding the
    object means enumerating its fields, and any field forgotten today silently
    disappears from every future trace.
    """
    msg = getattr(choice, "message", None)
    if msg is None or getattr(msg, "tool_calls", None):
        return  # already parsed by the server; nothing to recover
    calls, prose = parse_tool_calls(getattr(msg, "content", None) or "")
    if not calls:
        return

    try:
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall, Function)
        built = [ChatCompletionMessageToolCall(
            id=c["id"], type="function",
            function=Function(name=c["name"], arguments=c["arguments"]))
            for c in calls]
    except Exception:                                   # noqa: BLE001
        # Older/newer SDK layouts move these types around. A duck-typed stand-in
        # keeps capture working rather than failing the whole run on an import.
        built = [_DuckToolCall(c) for c in calls]

    msg.tool_calls = built
    msg.content = prose
    choice.finish_reason = "tool_calls"


class _DuckToolCall:
    def __init__(self, c: dict) -> None:
        self.id = c["id"]
        self.type = "function"
        self.function = _DuckFn(c["name"], c["arguments"])


class _DuckFn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments
