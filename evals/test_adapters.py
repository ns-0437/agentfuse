"""Adapter coverage — the layer the eval was blind to.

The scenario suite drives ``CircuitBreakerMonitor`` directly. That is the right
call for measuring *detection quality*, but it means every adapter — the code
users actually import — was completely untested. A bug in ``FuseRunHooks`` that
dropped tool calls on the floor would leave every benchmark number untouched
while making the library useless in practice.

These tests drive the real ``openai-agents`` SDK: a real ``Agent``, real
``@function_tool``s, the real ``Runner``, and the real ``FuseRunHooks``. Only
token generation is stubbed (a scripted model), so the tests are hermetic and
free while still exercising the genuine lifecycle.

    pytest evals/test_adapters.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

agents = pytest.importorskip("agents", reason="pip install openai-agents")

from agents import (  # noqa: E402
    Agent, Runner, Model, ModelProvider, ModelResponse, Usage, RunConfig, function_tool,
)
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
)

from agentfuse import DirectiveKind, EventType  # noqa: E402
from agentfuse.adapters.agentkit_hooks import FuseRunHooks, BreakerInterrupt  # noqa: E402

GOAL = ("Rotate the production database credential: locate the active connection "
        "string, generate a new secret, and update the secret store.")


@function_tool
def search_files(directory: str, pattern: str) -> str:
    """Search a directory for files matching a glob pattern."""
    return "0 files matched"


@function_tool
def secret_manager_get(name: str) -> str:
    """Fetch a secret from the managed secret store by name."""
    return f"secret-rotated-token for '{name}': new credential provisioned"


class ScriptedModel(Model):
    """Loops on search_files until steered, then takes the correct action."""

    def __init__(self):
        self.calls = 0

    def _usage(self):
        return Usage(requests=1, input_tokens=900, output_tokens=180, total_tokens=1080)

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, *, previous_response_id=None,
                           conversation_id=None, prompt=None, **kwargs):
        self.calls += 1
        blob = str(input)
        steered = "CIRCUIT BREAKER STEERING" in blob
        got_secret = "secret-rotated-token" in blob

        if steered and got_secret:
            msg = ResponseOutputMessage(
                id=f"m{self.calls}", role="assistant", status="completed", type="message",
                content=[ResponseOutputText(text="Credential rotated.", type="output_text",
                                            annotations=[])])
            return ModelResponse(output=[msg], usage=self._usage(),
                                 response_id=f"r{self.calls}", request_id=None)

        name, args = ("secret_manager_get", {"name": "prod/db/primary"}) if steered else \
                     ("search_files", {"directory": "./config", "pattern": "*.conn"})
        fc = ResponseFunctionToolCall(type="function_call", name=name,
                                      arguments=json.dumps(args),
                                      call_id=f"c{self.calls}", id=f"f{self.calls}",
                                      status="completed")
        return ModelResponse(output=[fc], usage=self._usage(),
                             response_id=f"r{self.calls}", request_id=None)

    async def stream_response(self, *a, **k):
        raise NotImplementedError
        yield  # pragma: no cover


class _Provider(ModelProvider):
    def __init__(self, m): self._m = m
    def get_model(self, name): return self._m


def _drive(max_attempts: int = 6, **fuse_kwargs):
    """Run the agent under supervision; return (fuse, final_output, attempts)."""
    async def go():
        agent = Agent(name="rotator", instructions=GOAL,
                      tools=[search_files, secret_manager_get])
        fuse = FuseRunHooks(original_goal=GOAL, loop_threshold=3, **fuse_kwargs)
        cfg = RunConfig(model_provider=_Provider(ScriptedModel()))
        items = [{"role": "user", "content": "Rotate the credential."}]
        for attempt in range(max_attempts):
            try:
                res = await Runner.run(agent, items, hooks=fuse, run_config=cfg, max_turns=12)
                return fuse, str(res.final_output), attempt + 1
            except BreakerInterrupt as bi:
                if bi.directive.kind is DirectiveKind.INJECT:
                    items.append({"role": "user",
                                  "content": f"[CIRCUIT BREAKER STEERING] {fuse.take_steering()}"})
                    continue
                return fuse, None, attempt + 1
        return fuse, None, max_attempts
    return asyncio.run(go())


# ---------------------------------------------------------------- tests
def test_hooks_observe_the_real_lifecycle():
    """Tool calls and args must actually reach the monitor through the adapter."""
    fuse, _out, _n = _drive()
    events = fuse.monitor.history
    assert events, "adapter fed no events to the monitor"

    tool_calls = [e for e in events if e.type is EventType.TOOL_CALL]
    assert tool_calls, "no TOOL_CALL events — the adapter is dropping tool calls"
    assert any(e.tool_name == "search_files" for e in tool_calls)
    # Arguments must survive the round-trip, or loop detection cannot fingerprint.
    assert any((e.tool_args or {}).get("directory") == "./config" for e in tool_calls)


def test_token_usage_reaches_the_monitor():
    """Without usage the spend detector is blind."""
    fuse, _out, _n = _drive()
    assert fuse.monitor.total_tokens > 0, "no token usage propagated through the adapter"


def test_breaker_interrupts_a_runaway_real_run():
    fuse, _out, attempts = _drive()
    assert fuse.monitor.tracer.trips >= 1, "breaker never fired on a genuine loop"
    assert attempts > 1, "run was never interrupted, so steering was never exercised"


def test_agent_self_heals_end_to_end():
    """The whole point: interrupted, steered, resumed, completed."""
    fuse, out, _n = _drive()
    assert out is not None, "run never completed after steering"
    assert "rotated" in out.lower()
    assert fuse.monitor.tracer.recoveries >= 1


def test_steering_is_consumed_once():
    """A steering instruction must not be replayed into the next turn twice."""
    fuse, _out, _n = _drive()
    assert fuse.take_steering() is None, "pending steering leaked after the run"


def test_hard_stop_escalates_instead_of_looping_forever():
    """With recoveries exhausted the adapter must hand back control, not spin."""
    fuse, out, attempts = _drive(max_attempts=3, max_recoveries=0)
    assert out is None, "expected an escalation, not a completion"
    assert attempts <= 3
