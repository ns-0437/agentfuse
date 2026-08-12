"""Recovery / steering engine.

When the breaker trips, we freeze the execution state and hand it to a *separate*
reasoning model — deliberately not the agent that drifted, because a model in a
logical trap is the worst judge of its own trap. That model returns a
``SteeringPath``: a concrete corrective instruction plus a decision on whether to
inject-and-resume, or escalate to a human.

Backends:
  * ``real``  — calls the OpenAI Responses API with a reasoning model
    (default ``o4-mini``; override with ``AGENTFUSE_RECOVERY_MODEL``). Requests a
    strict JSON steering path.
  * ``mock``  — a deterministic rule-based steerer that inspects the trip type
    and synthesizes a sensible recovery. Lets the whole system be demoed offline
    and keeps CI hermetic.

The engine auto-selects ``real`` when ``OPENAI_API_KEY`` is present, else ``mock``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .env import load_env, offline_mode
from .events import ExecutionSnapshot


class RecoveryAction(str, Enum):
    INJECT = "inject"      # inject the steering instruction and resume
    ESCALATE = "escalate"  # pause and hand to a human
    ABORT = "abort"        # unrecoverable — stop the run


@dataclass
class SteeringPath:
    action: RecoveryAction
    instruction: str            # the corrective message injected into the agent
    rationale: str              # why the supervisor chose this
    confidence: float = 0.5
    backend: str = "mock"
    raw: dict = field(default_factory=dict)


SUPERVISOR_SYSTEM = (
    "You are a supervising reasoning model acting as a circuit breaker for an "
    "autonomous agent that has been paused mid-run. You are NOT the agent; you "
    "are an independent overseer. You are given a frozen snapshot of the agent's "
    "state and the reason its execution was tripped. Diagnose the root cause and "
    "produce a single, concrete steering instruction that will realign the agent "
    "with its ORIGINAL objective and break it out of the failure mode. "
    "Prefer 'inject' when a corrective nudge can plausibly recover the run; use "
    "'escalate' when human judgment or credentials are required; use 'abort' only "
    "when the objective is impossible or unsafe to continue. "
    "Respond with STRICT JSON: "
    '{"action": "inject|escalate|abort", "instruction": "...", "rationale": "...", '
    '"confidence": 0.0-1.0}'
)


class RecoveryEngine:
    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None):
        self.model = model or os.getenv("AGENTFUSE_RECOVERY_MODEL", "o4-mini")
        if backend is None:
            load_env()
            backend = ("mock" if offline_mode()
                       else ("real" if os.getenv("OPENAI_API_KEY") else "mock"))
        self.backend = backend
        self._client = self._make_client() if backend == "real" else None
        if backend == "real" and self._client is None:
            self.backend = "mock"  # graceful fallback if SDK missing

    def _make_client(self):
        load_env()  # pick up a key from .env if the shell has none
        try:
            from openai import OpenAI  # type: ignore

            return OpenAI()
        except Exception:
            return None

    def recover(self, snapshot: ExecutionSnapshot) -> SteeringPath:
        if self.backend == "real":
            try:
                return self._recover_real(snapshot)
            except Exception as e:  # never let recovery itself crash the run
                path = self._recover_mock(snapshot)
                path.rationale = f"[fell back to mock: {e}] " + path.rationale
                return path
        return self._recover_mock(snapshot)

    # --- real reasoning-model backend -----------------------------------
    def _recover_real(self, snapshot: ExecutionSnapshot) -> SteeringPath:
        client = self._client
        user = snapshot.to_prompt_context()
        resp = client.responses.create(  # type: ignore[union-attr]
            model=self.model,
            input=[
                {"role": "system", "content": SUPERVISOR_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        text = getattr(resp, "output_text", None) or self._extract_text(resp)
        data = self._parse_json(text)
        return SteeringPath(
            action=RecoveryAction(data.get("action", "inject")),
            instruction=data.get("instruction", "Re-read your original objective and take a different next action."),
            rationale=data.get("rationale", ""),
            confidence=float(data.get("confidence", 0.6)),
            backend=f"real:{self.model}",
            raw=data,
        )

    @staticmethod
    def _extract_text(resp) -> str:
        try:
            parts = []
            for item in resp.output:  # type: ignore[attr-defined]
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        parts.append(t)
            return "".join(parts)
        except Exception:
            return ""

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    # --- deterministic offline backend ----------------------------------
    def _recover_mock(self, snapshot: ExecutionSnapshot) -> SteeringPath:
        det = snapshot.trip_detector
        goal = snapshot.original_goal

        if det == "loop":
            tool = snapshot.trip_evidence.get("tool", "that tool")
            return SteeringPath(
                action=RecoveryAction.INJECT,
                instruction=(
                    f"STOP repeating `{tool}` with the same arguments — it has "
                    f"returned the same result every time and is not advancing the "
                    f"task. Re-read your original objective: \"{goal}\". Choose a "
                    f"DIFFERENT next action or a different tool/argument that moves "
                    f"the state forward. If the information you need does not exist "
                    f"via this tool, note that assumption as false and try another path."
                ),
                rationale="Repetitive identical tool call with no state progress; break the loop by forcing a different action.",
                confidence=0.8,
                backend="mock",
            )
        if det == "drift":
            cur = snapshot.trip_evidence.get("current_goal", "")
            return SteeringPath(
                action=RecoveryAction.INJECT,
                instruction=(
                    f"You have drifted from your assigned objective. Your recent "
                    f"focus (\"{cur}\") is not what you were asked to do. Discard "
                    f"that tangent and realign strictly to the ORIGINAL objective: "
                    f"\"{goal}\". State your next action in terms of that objective only."
                ),
                rationale="Interpreted goal diverged from the system prompt; re-anchor to original objective.",
                confidence=0.75,
                backend="mock",
            )
        if det == "progress":
            return SteeringPath(
                action=RecoveryAction.INJECT,
                instruction=(
                    f"You have taken several actions without changing the working "
                    f"state — you may be reasoning from a false assumption. List the "
                    f"assumptions your last few steps depended on, mark the weakest "
                    f"one as possibly false, and take a concrete action that would "
                    f"verify it, in service of: \"{goal}\"."
                ),
                rationale="Activity without state progress signals a logical trap; force assumption-checking.",
                confidence=0.65,
                backend="mock",
            )
        if det == "spend":
            sev_critical = snapshot.trip_evidence.get("limit") is not None and (
                "total_tokens" in snapshot.trip_evidence or "total_cost_usd" in snapshot.trip_evidence
            )
            if sev_critical:
                return SteeringPath(
                    action=RecoveryAction.ESCALATE,
                    instruction=(
                        "Budget ceiling reached. Halting autonomous execution and "
                        "escalating to a human with a summary of progress so far."
                    ),
                    rationale="Hard budget ceiling hit; human decision required before spending more.",
                    confidence=0.9,
                    backend="mock",
                )
            return SteeringPath(
                action=RecoveryAction.INJECT,
                instruction=(
                    f"Your token burn rate is spiking without matching progress. "
                    f"Switch to the most direct plan to achieve \"{goal}\": take the "
                    f"single highest-value action next and avoid exploratory tool calls."
                ),
                rationale="Burn-rate spike; compress the plan to conserve budget.",
                confidence=0.6,
                backend="mock",
            )

        return SteeringPath(
            action=RecoveryAction.INJECT,
            instruction=f"Pause and re-read your original objective: \"{goal}\". Take a different next action.",
            rationale="Generic recovery.",
            confidence=0.4,
            backend="mock",
        )
