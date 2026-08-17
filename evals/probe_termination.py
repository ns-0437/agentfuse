"""Can the model stop, or does the harness make stopping impossible?

Every real trace captured so far ends in repetition, including tasks as simple as
"list the secrets and report them" (10 identical `list_secrets` calls, then
max_turns). Two explanations fit that evidence and they have opposite
consequences:

  A. The model is genuinely weak and loops. The traces are valid agent failures
     and the corpus is real data.
  B. The serving stack cannot emit a content-only reply while `tools` are
     present. Then termination is impossible BY CONSTRUCTION, every captured
     "loop" is an artifact of my own rig, and the real-trace corpus is worthless
     as evidence about agents.

Nothing downstream is safe to interpret until this is settled, so this probe
settles it in isolation — no monitor, no agent loop, no detectors.

Test 1  A question answerable with no tool at all, tools present.
Test 2  A conversation where the tool has ALREADY returned the answer. Any
        competent agent replies in prose here; a rig that cannot stop will call
        the tool again.
Test 3  Same as 2 with no tools attached — the control. If prose appears only
        here, the tools parameter is what suppresses termination, which is
        finding B.
"""

from __future__ import annotations

import argparse
import os
import sys

TOOLS = [{
    "type": "function",
    "function": {
        "name": "list_secrets",
        "description": "List secret names in a scope.",
        "parameters": {"type": "object",
                       "properties": {"scope": {"type": "string"}},
                       "required": ["scope"]},
    },
}]


def show(label: str, choice, tool_names=("list_secrets",)) -> bool:
    """Print what came back. True only for GENUINE prose.

    Non-empty content is NOT sufficient. Asked "what is 2+2?" this stack returned
    `finish=stop` with content `'functions.list_secrets:'` — a tool call that
    leaked into the content channel because the template's parser failed. Scoring
    that as "the model answered in prose" would count a broken tool call as
    evidence of healthy termination and invert the verdict, which is the same
    bug class this project tracks: a check that looks armed and isn't.
    """
    msg = choice.message
    calls = getattr(msg, "tool_calls", None) or []
    content = (getattr(msg, "content", None) or "").strip()
    if calls:
        names = ", ".join(c.function.name for c in calls)
        print(f"  {label:<34} TOOL CALL ({names})  finish={choice.finish_reason}")
        return False
    leaked = content.startswith("functions.") or any(
        content.startswith(f"{n}:") or content.startswith(f"functions.{n}")
        for n in tool_names)
    kind = "LEAKED TOOL CALL" if leaked else "PROSE"
    print(f"  {label:<34} {kind}: {content[:64]!r}  finish={choice.finish_reason}")
    return bool(content) and not leaked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("AGENTFUSE_LLM_BASE_URL",
                                                    "http://127.0.0.1:8080/v1"))
    ap.add_argument("--model", default=os.getenv("AGENTFUSE_RECOVERY_MODEL", "local"))
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    def ask(messages, tools):
        kw = {"model": args.model, "messages": messages, "temperature": 0.0}
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        return client.chat.completions.create(**kw).choices[0]

    print("=" * 78)
    print("TERMINATION PROBE — can this stack answer without calling a tool?")
    print("=" * 78)

    results = {}

    # 1. No tool is needed or relevant.
    results["no_tool_needed"] = show("1. trivia, tools attached", ask(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is 2 + 2? Answer in one word."}], TOOLS))

    # 2. The answer is already in the transcript. Calling again is the failure.
    answered = [
        {"role": "system", "content": "You are a helpful assistant. When you have "
                                      "the answer, reply to the user directly."},
        {"role": "user", "content": "List the secrets in the production scope and "
                                    "report them."},
        # content MUST be "" not None: llama.cpp's schema requires a string and
        # returns a 500 with seven validation errors otherwise. The adapter
        # already knows this (openai_sdk.py:105); the probe has to match it.
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "list_secrets",
                                      "arguments": '{"scope": "production"}'}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "prod/db/primary, prod/db/replica, prod/api/token"},
    ]
    results["answer_in_hand"] = show("2. answer already returned", ask(answered, TOOLS))

    # 3. Control: identical state, tools removed.
    control = [m for m in answered]
    control[2] = {"role": "assistant", "content": "I called list_secrets."}
    control[3] = {"role": "user", "content": "Result: prod/db/primary, "
                                             "prod/db/replica, prod/api/token"}
    results["control_no_tools"] = show("3. same state, NO tools (control)",
                                       ask(control, None))

    # 4. The other half of the requirement. A stack that never calls a tool also
    # "terminates" perfectly, and would pass tests 1-3 while being useless for a
    # tool-using agent. Fixing termination by breaking tool calls would trade one
    # artifact for a worse one, so it has to be checked in the same run.
    needs_tool = show("4. tool genuinely required", ask(
        [{"role": "system", "content": "You are an agent. Use the provided tools."},
         {"role": "user", "content": "List the secrets in the production scope."}],
        TOOLS))
    results["calls_when_needed"] = not needs_tool  # prose here would be the failure

    print("\n" + "-" * 78)
    if not results["calls_when_needed"]:
        print("WARNING: no tool call even when one was required. This stack cannot")
        print("  drive a tool-using agent; termination results below are moot.")
        return 1
    # Test 2 is the decisive one: the answer is already in the transcript, so a
    # further tool call cannot be justified by missing information. Test 1 is
    # weaker evidence — a trivia question is off-distribution for a tool agent.
    stops_with_tools = results["answer_in_hand"]
    if stops_with_tools:
        print("VERDICT: the stack CAN terminate with tools attached.")
        print("  => The captured loops are real model behaviour, not a rig artifact.")
        print("     The corpus stands. Healthy runs need better prompting or a")
        print("     stronger model, not a different harness.")
    elif results["control_no_tools"]:
        print("VERDICT: prose ONLY without tools. Termination is suppressed by the")
        print("  `tools` parameter in this serving stack.")
        print("  => FINDING B. Every captured 'loop' is an artifact of my own rig.")
        print("     The real-trace corpus cannot be used as evidence about agent")
        print("     behaviour until captures are redone on a stack that can stop.")
    else:
        print("VERDICT: no prose in ANY condition — the probe itself is suspect")
        print("  (server, template, or model wiring). Fix that before concluding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
