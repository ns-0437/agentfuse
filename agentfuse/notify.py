"""Escalation delivery — telling an actual human the run stopped.

The breaker escalates when steering has been exhausted or a hard ceiling is hit.
Until now "escalate to a human" meant returning a ``PAUSE`` directive and printing
to the console. For a supervisor whose entire premise is **unattended** runs of
hours to days, that is a notification nobody receives: the console belongs to a
process that is no longer being watched, which is precisely why the breaker
exists.

That makes it the fourth instance of one bug class in this project — a guard that
looks armed and is not. The others were a restart resetting the spend counter,
``max_cost_usd`` never firing, and ``NoProgressDetector`` being structurally
inert. All four passed review; all four were found by asking what the guard
actually does at the moment it is supposed to work.

Delivery is verified, not assumed
---------------------------------
A notifier that fails silently rebuilds the same bug one layer up, so
:meth:`Notifier.send` returns whether delivery *succeeded* and the monitor
records it. A run that escalated but could not reach anyone reports
``escalation_delivered: False`` rather than looking identical to one that did.
Escalating with no channel configured warns, because the operator almost
certainly believes someone will be told.

What leaves the machine
-----------------------
An escalation payload naturally contains the agent's reasoning and tool output.
Posting that to an external URL is data egress, so it is treated as such: free
text is passed through :mod:`agentfuse.sanitize`, truncated, and
``include_agent_text=False`` drops it entirely for deployments where the trace is
sensitive. What remains — run id, detector, step, reason, and the operator's own
goal — is what a human needs to decide whether to intervene.

Failure never propagates
------------------------
A webhook outage must not take down the run it is reporting on. Every send is
wrapped, bounded by a timeout and a small retry budget, and returns ``False``
instead of raising. The run is halting anyway, so a brief block is acceptable;
an unbounded one is not.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

from .redact import redact
from .sanitize import sanitize

#: Free-text fields are cut to this before leaving the process.
MAX_TEXT = 800


@dataclass
class Notification:
    """Why a human is being asked to look, and at what."""

    run_id: str
    reason: str
    detector: str
    step: int
    goal: str
    action: str                       # escalate | abort
    instruction: str = ""             # what the supervisor would have said
    evidence: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    def payload(self, include_agent_text: bool = True) -> dict:
        """JSON body. Agent-produced text is sanitised, truncated, or omitted."""
        def clean(text: str) -> str:
            # Redact before sanitize and before truncation. A credential cut in
            # half by the length limit is no longer matchable and would leave
            # the machine as a fragment. This is the most exposed of the three
            # egress paths — an outbound POST to a third party.
            return sanitize(redact(str(text or "")))[:MAX_TEXT]

        body: dict[str, Any] = {
            "source": "agentfuse",
            "run_id": self.run_id,
            "action": self.action,
            "detector": self.detector,
            "step": self.step,
            # The goal is the operator's own text, not the agent's, so it is not
            # sanitised — doing so would mangle the one field they wrote.
            "goal": str(self.goal)[:MAX_TEXT],
            "totals": self.totals,
        }
        if include_agent_text:
            body["reason"] = clean(self.reason)
            body["instruction"] = clean(self.instruction)
            body["evidence"] = {k: clean(v) if isinstance(v, str) else v
                                for k, v in (self.evidence or {}).items()}
        return body


class Notifier(Protocol):
    """Returns whether the human was actually reached."""

    def send(self, note: Notification) -> bool: ...


class ConsoleNotifier:
    """The previous behaviour, kept explicit rather than implicit.

    Reports success, because writing to a console that someone *is* watching is a
    real delivery. It is the default only because it cannot fail; it is not a
    substitute for a channel that reaches someone who is asleep.
    """

    def __init__(self, write: Optional[Callable[[str], None]] = None):
        self._write = write or print

    def send(self, note: Notification) -> bool:
        self._write(
            f"[agentfuse] ESCALATION run={note.run_id} step={note.step} "
            f"detector={note.detector} action={note.action}: {note.reason}")
        return True


class WebhookNotifier:
    """POST the escalation as JSON. Stdlib only — no new dependency.

    Works with anything that accepts a JSON body: Slack and Discord incoming
    webhooks, PagerDuty Events, an internal endpoint.
    """

    def __init__(self, url: str, timeout: float = 5.0, retries: int = 2,
                 headers: Optional[dict] = None, include_agent_text: bool = True,
                 opener: Optional[Callable] = None):
        self.url = url
        self.timeout = timeout
        # Total attempts = retries + 1. Deliberately small: the run is stopping
        # and a human is waiting, so a long retry ladder helps nobody.
        self.retries = retries
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.include_agent_text = include_agent_text
        self.last_error: Optional[str] = None
        self._opener = opener or urllib.request.urlopen

    def send(self, note: Notification) -> bool:
        body = json.dumps(note.payload(self.include_agent_text)).encode("utf-8")
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body,
                                             headers=self.headers, method="POST")
                with self._opener(req, timeout=self.timeout) as resp:
                    code = getattr(resp, "status", None) or resp.getcode()
                    if 200 <= int(code) < 300:
                        self.last_error = None
                        return True
                    self.last_error = f"HTTP {code}"
            except Exception as e:            # noqa: BLE001 — see module docstring
                self.last_error = f"{type(e).__name__}: {e}"
        return False


class MultiNotifier:
    """Fan out to several channels. Succeeds if *any* channel delivered.

    Any, not all: the question this answers is "was a human reached", and one
    working channel answers it yes. Requiring all would report failure whenever a
    redundant backup was misconfigured, which inverts the meaning.
    """

    def __init__(self, notifiers: Sequence[Notifier]):
        self.notifiers = list(notifiers)

    def send(self, note: Notification) -> bool:
        delivered = False
        for n in self.notifiers:
            try:
                delivered = bool(n.send(note)) or delivered
            except Exception:                 # noqa: BLE001
                continue
        return delivered


def build_notifier(webhook_url: Optional[str] = None, echo: bool = True,
                   **kwargs) -> Optional[Notifier]:
    """Assemble the channels a config asks for, or ``None`` if it asks for none."""
    channels: list[Notifier] = []
    if webhook_url:
        channels.append(WebhookNotifier(webhook_url, **kwargs))
    if echo:
        channels.append(ConsoleNotifier())
    if not channels:
        return None
    return channels[0] if len(channels) == 1 else MultiNotifier(channels)
