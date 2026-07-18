"""REAL GPT run supervised by AgentFuse — a genuine model, genuinely looping.

This is the fully-real counterpart to ``real_agentkit_run.py``: no scripted
model. A real OpenAI model drives a real ``openai-agents`` Agent with real tools,
and AgentFuse's real ``FuseRunHooks`` supervise the live run.

Scenario chosen so a *capable* model actually exhibits the failure mode: the
agent is told to monitor a batch job until it reports COMPLETED, polling
``check_job_status``. The job never completes (it always returns PENDING), so a
compliant model polls the same call over and over — a realistic infinite loop.
The breaker trips on the live hooks, a separate reasoning model produces a
steering path ("stop polling, the job is stuck, report it"), we inject that into
the conversation, and the model then calls ``report_status`` and finishes.

Cost: a handful of small model calls (typically a few cents). Bounded by the
loop threshold + a hard token ceiling.

Setup (your key never goes in source or chat):
    1) create  agentfuse/.env   containing:   OPENAI_API_KEY=sk-...
       (optionally  AGENTFUSE_MODEL=gpt-4o-mini  and  AGENTFUSE_RECOVERY_MODEL=o4-mini)
    2) run:   python examples/real_gpt_run.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Sets os.environ; never prints values."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is not set.\n"
          "Create agentfuse/.env with a line:  OPENAI_API_KEY=sk-...\n"
          "(your key stays local; it is never written to source or printed).")
    raise SystemExit(1)

from agents import Agent, Runner, function_tool  # noqa: E402
from agentfuse.adapters.agentkit_hooks import FuseRunHooks, BreakerInterrupt  # noqa: E402
from agentfuse import DirectiveKind  # noqa: E402

MODEL = os.getenv("AGENTFUSE_MODEL", "gpt-4o-mini")

GOAL = (
    "You are monitoring batch job 'nightly-etl-42'. Poll its status with "
    "check_job_status and DO NOT report until the job status is COMPLETED. "
    "Once it is COMPLETED, call report_status with a one-line summary. Keep "
    "checking until it completes."
)

_poll_count = {"n": 0}


@function_tool
def check_job_status(job_id: str) -> str:
    """Return the current status of a batch job."""
    _poll_count["n"] += 1
    # The trap: this job is wedged and never reaches COMPLETED.
    return f"job '{job_id}' status: PENDING (still running, check again shortly)"


@function_tool
def report_status(summary: str) -> str:
    """Report the final outcome of the monitored job."""
    return f"Report filed: {summary}"


async def main() -> None:
    print("\n" + "=" * 72)
    print(f"AgentFuse x REAL GPT ({MODEL}) — live Runner, live hooks, real model")
    print("Scenario: agent polls a job that never completes -> real infinite loop")
    print("=" * 72)
    print(f"OBJECTIVE: {GOAL}\n")

    agent = Agent(name="job-monitor", model=MODEL,
                  instructions=GOAL, tools=[check_job_status, report_status])
    fuse = FuseRunHooks(
        original_goal=GOAL,
        loop_threshold=4,          # allow a few honest polls before calling it a loop
        max_recoveries=2,
        max_tokens=40_000,         # hard ceiling -> escalate rather than burn budget
        jsonl_path="runs/real_gpt.jsonl",
    )

    input_items = [{"role": "user", "content": "Begin monitoring the job now."}]

    for attempt in range(5):
        try:
            result = await Runner.run(agent, input_items, hooks=fuse, max_turns=12)
            from agentfuse import AgentEvent, EventType
            fuse.monitor.observe(AgentEvent(type=EventType.COMPLETE, step=fuse._step + 1,
                                            node="job-monitor", text=str(result.final_output)[:120],
                                            state={"done": True}))
            totals = fuse.finish("complete")
            print(f"\n>> FINAL OUTPUT: {result.final_output}")
            print(f">> Real GPT run self-healed — trips: {totals['trips']} | "
                  f"recoveries: {totals['recoveries']} | tokens: {totals['total_tokens']} | "
                  f"polls before break: {_poll_count['n']}")
            print(">> Trace: runs/real_gpt.jsonl")
            return
        except BreakerInterrupt as bi:
            if bi.directive.kind is DirectiveKind.INJECT:
                steer = fuse.take_steering()
                input_items.append({"role": "user",
                                    "content": f"[CIRCUIT BREAKER STEERING] {steer}"})
                print("  >> steering injected into conversation; re-running agent\n")
                continue
            print(f"\n>> HARD STOP: {bi.directive.kind.value} — escalating to a human.")
            fuse.finish("escalated")
            return

    fuse.finish("incomplete")
    print("\n>> Did not converge within attempt budget.")


if __name__ == "__main__":
    asyncio.run(main())
