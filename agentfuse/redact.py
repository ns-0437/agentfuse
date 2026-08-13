"""Credential redaction on every path agent output can escape through.

:mod:`agentfuse.sanitize` defends the *supervisor* against prompt injection —
agent text trying to issue instructions. It has nothing to say about secrets, and
secrets are a separate problem with a separate direction of travel: not "can this
text control us", but "is this text safe to write down".

An agent doing real work reads credentials. It fetches a connection string,
receives an API key from a secret manager, gets a bearer token in a header. That
text reaches three places, all of them durable or remote:

  1. the JSONL trace on disk, verbatim, forever;
  2. the supervisor prompt, which is sent to a model provider;
  3. the escalation webhook, an outbound POST to a third party.

Path 3 was added on 2026-08-13 and made this materially worse: a project that
previously leaked to a local file now leaks over the network, and nothing in the
codebase noticed.

Matching philosophy
-------------------
Conservative on shape, aggressive on known formats. Provider key formats
(``sk-``, ``AKIA``, ``ghp_``, ``xox``…) are unambiguous and matched outright.
Generic assignments are matched only when a *naming* signal is present —
``api_key=``, ``password:`` — so ordinary prose is untouched. Bare high-entropy
strings are matched only above 32 characters, because this project's own
identifiers are 12-character hashes and redacting those would destroy the traces
this exists to protect.

Redaction is **lossy and deliberate**: the value is replaced by a labelled
marker, so a reader can see that a credential was present and of what kind
without recovering it. Length is not preserved — leaking a secret's length is a
smaller leak, not no leak.

This will miss things. A credential with no recognisable format, no naming
context, and under 32 characters is indistinguishable from ordinary text. The
honest mitigation is defence in depth: ``escalation_include_agent_text=False``
keeps the trace off the wire entirely, and that remains the only complete answer
for a sensitive deployment.
"""

from __future__ import annotations

import re
from typing import Pattern

_MARK = "[REDACTED:{}]"

#: (label, pattern). Order matters: specific formats before generic ones, so a
#: recognisable key is labelled by provider rather than as a generic secret.
_RULES: list[tuple[str, Pattern[str]]] = [
    # -- PEM blocks. Matched first; they contain base64 that later rules would
    #    shred into a hundred separate markers.
    ("private-key", re.compile(
        r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?-----END[ A-Z]*PRIVATE KEY-----",
        re.DOTALL)),
    # -- Provider key formats, unambiguous enough to match on shape alone.
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("aws-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe-key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]+")),
    # -- Credentials embedded in a URL: postgres://user:secret@host/db
    ("url-credentials", re.compile(r"(?<=://)([^\s:/@]+):([^\s:/@]+)(?=@)")),
    # -- Authorization headers.
    ("bearer-token", re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9_\-\.=]{16,}")),
    # -- Named assignments. The NAME is the signal; without it, ordinary prose
    #    full of long words would be shredded.
    ("secret-assignment", re.compile(
        r"(?i)\b(api[_-]?key|apikey|secret|password|passwd|pwd|access[_-]?token|"
        r"auth[_-]?token|client[_-]?secret|private[_-]?key)"
        r"(\s*[:=]\s*)(\"[^\"]{4,}\"|'[^']{4,}'|[^\s,;}\)]{4,})")),
    # -- Long high-entropy blobs with no other signal. 32+ only: this project's
    #    own hashes are 12 chars and must survive.
    ("high-entropy", re.compile(r"\b(?=[A-Za-z0-9+/_\-]{32,}\b)"
                                r"(?=[^\s]*[0-9])(?=[^\s]*[A-Za-z])"
                                r"[A-Za-z0-9+/_\-]{32,}={0,2}\b")),
]


def redact(text: str) -> str:
    """Replace anything that looks like a credential with a labelled marker."""
    if not text:
        return text or ""
    out = str(text)
    for label, pattern in _RULES:
        if label == "secret-assignment":
            # Keep the name and separator so the trace still reads sensibly;
            # replace only the value.
            out = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MARK.format(label)}", out)
        elif label == "url-credentials":
            out = pattern.sub(_MARK.format(label), out)
        else:
            out = pattern.sub(_MARK.format(label), out)
    return out


def redact_obj(value):
    """Redact recursively through the containers events and traces actually use."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        t = type(value)
        return t(redact_obj(v) for v in value)
    return value


def contains_secret(text: str) -> bool:
    """Whether redaction would change anything — useful for tests and alerts."""
    return redact(text) != (text or "")
