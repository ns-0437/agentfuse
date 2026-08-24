"""REAL OpenAI AgentKit run, supervised by AgentFuse.

Unlike the simulated demos, this drives the genuine `openai-agents` SDK: a real
`Agent`, real `@function_tool`s, the real `Runner`, and AgentFuse's real
`FuseRunHooks` observing the live lifecycle. The agent falls into an infinite
tool loop (keeps calling `search_files` for a config that doesn't exist); the
circuit breaker — watching the actual SDK hooks — trips, aborts the runaway run,
climbs the deterministic escalation ladder for a steering path, injects it into
the conversation, and re-runs. The agent then takes the correct action
(`secret_manager_get`) and completes.

Only the *model's token generation* is stubbed (a `ScriptedModel`) so the run is
hermetic and free. Everything else is the real SDK. To run against a real model,
delete the ScriptedModel / RunConfig override and set OPENAI_API_KEY — the hooks
and breaker code are identical:

    result = await Runner.run(agent, input_items, hooks=fuse, max_turns=12)

Requires:  pip install openai-agents
Run:       python examples/real_agentkit_run.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import (
    Agent, Runner, Model, ModelProvider, ModelResponse, Usage, RunConfig, function_tool,
)
from openai.types.responses import (
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
)

from agentfuse.adapters.agentkit_hooks import FuseRunHooks, BreakerInterrupt
from agentfuse import DirectiveKind

GOAL = (
    "Rotate the production database credential: locate the active connection "
    "string, generate a new secret, and update the secret store."
)


# --- real AgentKit tools -------------------------------------------------
@function_tool
def search_files(directory: str, pattern: str) -> str:
    """Search a directory for files matching a glob pattern."""
    # The trap: the connection string is NOT a file — this always comes back empty.
    return "0 files matched"


@function_tool
def secret_manager_get(name: str) -> str:
    """Fetch a secret from the managed secret store by name."""
    return f"secret-rotated-token for '{name}': new credential provisioned and stored"


# --- hermetic stub model (the ONLY simulated piece) ----------------------
class ScriptedModel(Model):
    """Stands in for GPT so the run is free. Behavior mirrors a real model that
    loops until it receives a corrective steering message, then course-corrects.
    """

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
        secret_fetched = "secret-rotated-token" in blob

        if steered and secret_fetched:
            msg = ResponseOutputMessage(
                id=f"msg_{self.calls}", role="assistant", status="completed", type="message",
                content=[ResponseOutputText(text="Credential rotated successfully.",
                                            type="output_text", annotations=[])],
            )
            return ModelResponse(output=[msg], usage=self._usage(),
                                 response_id=f"r_{self.calls}", request_id=None)

        if steered:
            fc = ResponseFunctionToolCall(
                type="function_call", name="secret_manager_get",
                arguments=json.dumps({"name": "prod/db/primary"}),
                call_id=f"call_{self.calls}", id=f"fc_{self.calls}", status="completed")
            return ModelResponse(output=[fc], usage=self._usage(),
                                 response_id=f"r_{self.calls}", request_id=None)

        # Default: the doomed, repeated search (the infinite loop).
        fc = ResponseFunctionToolCall(
            type="function_call", name="search_files",
            arguments=json.dumps({"directory": "./config", "pattern": "*.conn"}),
            call_id=f"call_{self.calls}", id=f"fc_{self.calls}", status="completed")
        return ModelResponse(output=[fc], usage=self._usage(),
                             response_id=f"r_{self.calls}", request_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("streaming not used in this demo")
        yield  # pragma: no cover


class _Provider(ModelProvider):
    def __init__(self, model): self._model = model
    def get_model(self, model_name): return self._model


async def main() -> None:
    print("\n" + "=" * 72)
    print("AgentFuse x OpenAI AgentKit — REAL Runner + real hooks + real tools")
    print("Scenario: live agent stuck in an infinite tool loop, then self-healed")
    print("=" * 72)
    print(f"OBJECTIVE: {GOAL}\n")

    agent = Agent(name="rotator", instructions=GOAL, tools=[search_files, secret_manager_get])
    fuse = FuseRunHooks(
        original_goal=GOAL, loop_threshold=3, max_recoveries=3,
        max_tokens=500_000, jsonl_path="runs/real_agentkit.jsonl",
    )

    scripted = ScriptedModel()
    run_config = RunConfig(model_provider=_Provider(scripted))  # remove for a real model
    input_items = [{"role": "user", "content": "Please rotate the production DB credential now."}]

    for attempt in range(6):
        try:
            result = await Runner.run(agent, input_items, hooks=fuse,
                                      run_config=run_config, max_turns=12)
            fuse.monitor.observe(_complete_event(fuse))
            totals = fuse.finish("complete")
            print(f"\n>> FINAL OUTPUT: {result.final_output}")
            print(f">> Self-healed live AgentKit run — trips: {totals['trips']} | "
                  f"recoveries: {totals['recoveries']} | steps: {totals['steps']}")
            print(">> Trace: runs/real_agentkit.jsonl")
            return
        except BreakerInterrupt as bi:
            if bi.directive.kind is DirectiveKind.INJECT:
                steer = fuse.take_steering()
                # Inject the steering instruction into the conversation and retry —
                # the transferable pattern that also works against a real model.
                input_items.append({"role": "user", "content": f"[CIRCUIT BREAKER STEERING] {steer}"})
                print("  >> steering injected into conversation; re-running agent\n")
                continue
            # PAUSE / ABORT -> stop and hand back to a human.
            print(f"\n>> HARD STOP: {bi.directive.kind.value} — escalating to a human.")
            fuse.finish("escalated")
            return

    fuse.finish("incomplete")
    print("\n>> Did not converge within attempt budget.")


def _complete_event(fuse):
    from agentfuse import AgentEvent, EventType
    return AgentEvent(type=EventType.COMPLETE, step=fuse._step + 1, node="rotator",
                      text="credential rotated", state={"rotated": True})


if __name__ == "__main__":
    asyncio.run(main())
