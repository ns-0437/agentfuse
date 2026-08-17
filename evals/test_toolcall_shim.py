"""Tests for the llama.cpp tool-call recovery shim.

This shim sits between the model and every captured trace, so a bug here does not
surface as a crash — it fabricates or drops agent actions and those become facts
in the benchmark. The tests therefore pin the exact malformed shapes observed
from the real server, not idealised ones.
"""

from __future__ import annotations

import json

import pytest

from evals.toolcall_shim import ToolCallShim, parse_tool_calls


def test_plain_prose_is_left_alone():
    calls, prose = parse_tool_calls("The secrets are a, b and c.")
    assert calls == []
    assert prose == "The secrets are a, b and c."


def test_empty_content_does_not_crash():
    assert parse_tool_calls("") == ([], "")
    assert parse_tool_calls(None) == ([], "")


def test_well_formed_block_is_recovered():
    calls, prose = parse_tool_calls(
        '<tool_call>\n{"name": "list_secrets", "arguments": {"scope": "production"}}\n'
        "</tool_call>")
    assert len(calls) == 1
    assert calls[0]["name"] == "list_secrets"
    assert json.loads(calls[0]["arguments"]) == {"scope": "production"}
    assert prose == ""


def test_doubled_braces_from_the_qwen_template_are_recovered():
    """The exact string the real server returned. Qwen's bundled template has a
    jinja escaping bug that doubles the outer braces; unhandled, every tool call
    would look malformed and the model would look broken."""
    calls, _ = parse_tool_calls(
        '<tool_call>\n{{"name": "list_secrets", "arguments": {"scope": "production"}}}\n'
        "</tool_call>")
    assert len(calls) == 1
    assert calls[0]["name"] == "list_secrets"
    assert json.loads(calls[0]["arguments"]) == {"scope": "production"}


def test_unclosed_block_is_still_recovered():
    """A truncated generation still carries a usable call. Dropping it would turn
    a tool-using turn into a false 'the agent stopped and answered' datapoint."""
    calls, _ = parse_tool_calls(
        '<tool_call>\n{"name": "read_secret", "arguments": {"name": "a"}}')
    assert len(calls) == 1
    assert calls[0]["name"] == "read_secret"


def test_prose_alongside_a_call_is_preserved():
    """The LLM_CALL event's text feeds the drift detector; discarding reasoning
    would silently change what that detector sees."""
    calls, prose = parse_tool_calls(
        'I should check the store first.\n'
        '<tool_call>{"name": "list_secrets", "arguments": {}}</tool_call>')
    assert len(calls) == 1
    assert prose == "I should check the store first."


def test_multiple_calls_get_distinct_ids():
    calls, _ = parse_tool_calls(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>')
    assert [c["name"] for c in calls] == ["a", "b"]
    assert len({c["id"] for c in calls}) == 2, "duplicate ids collapse lanes in LoopDetector"


def test_arguments_are_always_a_json_string():
    """The adapter calls json.loads(arguments). A dict raises TypeError deep in
    the tool loop, which reads as a model failure rather than a shim bug."""
    for blob in ('{"name": "x", "arguments": {"k": 1}}',
                 '{"name": "x", "arguments": "{\\"k\\": 1}"}',
                 '{"name": "x"}'):
        calls, _ = parse_tool_calls(f"<tool_call>{blob}</tool_call>")
        assert isinstance(calls[0]["arguments"], str)
        json.loads(calls[0]["arguments"])


def test_garbage_inside_a_block_is_skipped_not_guessed():
    calls, _ = parse_tool_calls("<tool_call>not json at all</tool_call>")
    assert calls == []


def test_block_without_a_name_is_skipped():
    calls, _ = parse_tool_calls('<tool_call>{"arguments": {"a": 1}}</tool_call>')
    assert calls == []


# --- the client wrapper -----------------------------------------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Choice:
    def __init__(self, msg, finish_reason="stop"):
        self.message, self.finish_reason = msg, finish_reason
        self.logprobs = "LOGPROBS-SENTINEL"


class _Resp:
    def __init__(self, choice):
        self.choices = [choice]
        self.usage = "USAGE-SENTINEL"


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp
        self.seen = None

        outer = self

        class _C:
            def create(self, **kw):
                outer.seen = kw
                return outer._resp

        class _Chat:
            completions = _C()

        self.chat = _Chat()


def _shimmed(content, tool_calls=None):
    resp = _Resp(_Choice(_Msg(content, tool_calls)))
    client = _FakeClient(resp)
    out = ToolCallShim(client).chat.completions.create(model="m", messages=[])
    return out, client


def test_shim_converts_text_into_real_tool_calls():
    out, _ = _shimmed('<tool_call>{"name": "list_secrets", "arguments": {}}</tool_call>')
    msg = out.choices[0].message
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "list_secrets"
    assert out.choices[0].finish_reason == "tool_calls"


def test_shim_preserves_usage_and_logprobs():
    """Rebuilding the response instead of mutating it would drop these, and the
    spend and confidence detectors read them."""
    out, _ = _shimmed('<tool_call>{"name": "x", "arguments": {}}</tool_call>')
    assert out.usage == "USAGE-SENTINEL"
    assert out.choices[0].logprobs == "LOGPROBS-SENTINEL"


def test_shim_does_not_touch_already_parsed_calls():
    existing = [type("T", (), {"id": "call_9", "type": "function",
                               "function": _Fn("already", "{}")})()]
    out, _ = _shimmed("", existing)
    assert out.choices[0].message.tool_calls is existing


def test_shim_leaves_genuine_prose_alone():
    """A real termination must survive the shim, or it re-creates the very bug
    the shim exists to fix."""
    out, _ = _shimmed("The secrets are a, b and c.")
    assert out.choices[0].message.tool_calls is None
    assert out.choices[0].finish_reason == "stop"


def test_shim_passes_requests_through_unmodified():
    _, client = _shimmed("hi")
    assert client.seen["model"] == "m"
    assert client.seen["messages"] == []


def test_shim_proxies_unknown_attributes():
    client = _FakeClient(_Resp(_Choice(_Msg("hi"))))
    client.base_url = "http://x/v1"
    assert ToolCallShim(client).base_url == "http://x/v1"
