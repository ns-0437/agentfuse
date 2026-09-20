"""Credential redaction on every path agent output escapes through.

`sanitize.py` asks "can this text control the supervisor". This asks the other
question — "is this text safe to write down" — and nothing was asking it. An
agent doing real work reads connection strings, secret-manager responses and
bearer tokens, and that text reached three durable or remote places verbatim:

  1. the JSONL trace on disk,
  2. the supervisor prompt, sent to a model provider,
  3. the escalation webhook, an outbound POST to a third party.

Path 3 was added on 2026-08-13, which turned a local-file leak into a network
one. Nothing in the codebase noticed, which is why the tests below are written
against the *exit points* rather than against the regexes: a pattern library
nothing calls is worth nothing.

The false-positive tests matter as much as the detection ones. Redaction is
lossy, and a matcher that eats ordinary text destroys the traces this exists to
protect — including this project's own 12-character hashes.

    pytest evals/test_redact.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

import pytest  # noqa: E402

from agentfuse import AgentEvent, EventType, Tracer  # noqa: E402
from agentfuse.events import ExecutionSnapshot, stable_hash  # noqa: E402
from agentfuse.notify import Notification  # noqa: E402
from agentfuse.redact import contains_secret, redact, redact_obj  # noqa: E402

GOAL = "Rotate the production database credential."

# Fixtures are ASSEMBLED AT RUNTIME rather than written as literals.
#
# Not cosmetic: the first version spelled them out, and GitHub's push protection
# blocked the push because the Slack fixture looked like a live token. It was
# right to. A repository that contains strings indistinguishable from real
# credentials trains everyone reading it to wave through exactly the thing this
# module exists to catch, and the fix is to stop committing them — not to add an
# exemption. Joining fragments means no complete credential-shaped string ever
# appears in the file, while the value the tests see is unchanged.
_ALPHA = "abcdefghijklmnopqrstuvwxyz012345"


def _k(*parts: str) -> str:
    return "".join(parts)


SECRETS = [
    ("openai", _k("sk-", "proj-", _ALPHA)),
    ("openai-legacy", _k("sk-", _ALPHA)),
    ("aws", _k("AKIA", "IOSFODNN7", "EXAMPLE")),
    ("github", _k("ghp", "_", _ALPHA, "6789")),
    ("github-pat", _k("github", "_pat_", _ALPHA)),
    ("slack", _k("xox", "b-", "123456789012-", "abcdefghijklmnop")),
    ("google", _k("AIza", "SyA1234567890", "abcdefghijklmnopqrstuvw")),
    ("stripe", _k("sk", "_live_", "abcdefghijklmnopqrstuvwx")),
    ("jwt", _k("eyJ", "hbGciOiJIUzI1NiJ9.", "eyJzdWIiOiIxMjM0NTY3ODkwIn0.",
               "dozjgNryP4J3jVmNHl0w5N")),
]
OPENAI_KEY = SECRETS[0][1]


@pytest.mark.parametrize("label,secret", SECRETS, ids=[s[0] for s in SECRETS])
def test_known_key_formats_are_removed(label, secret):
    out = redact(f"the tool returned {secret} for you")
    assert secret not in out, f"{label} key survived redaction"
    assert "REDACTED" in out


def test_credentials_inside_a_connection_string():
    out = redact("postgres://svc_user:s3cr3tP4ss@db.internal:5432/orders")
    assert "s3cr3tP4ss" not in out
    assert "db.internal" in out, "the host is diagnostic and should survive"


def test_named_assignments_keep_the_name_and_lose_the_value():
    """A trace saying `password=[REDACTED]` is still readable; one saying
    nothing at all is not."""
    out = redact('api_key="abcd1234efgh" and password: hunter2xyz')
    assert "abcd1234efgh" not in out and "hunter2xyz" not in out
    assert "api_key" in out and "password" in out


def test_pem_blocks_collapse_to_one_marker():
    pem = ("-----BEGIN RSA PRIVATE KEY-----\n"
           "MIIEowIBAAKCAQEAx7Zk9s0aQ1QwWq3nT8vB\nabcdef0123456789+/==\n"
           "-----END RSA PRIVATE KEY-----")
    out = redact(f"key material:\n{pem}\ndone")
    assert "MIIEowIBAAKCAQEA" not in out
    assert out.count("REDACTED") == 1, "a PEM block should not shred into many markers"
    assert "done" in out


def test_bearer_tokens():
    out = redact("Authorization: Bearer abcdefghijklmnop0123456789")
    assert "abcdefghijklmnop0123456789" not in out


# ------------------------------------------------- what must NOT be redacted
def test_ordinary_prose_is_untouched():
    text = ("The agent called search_files with pattern *.conn and received "
            "0 files matched, then retried twice before succeeding.")
    assert redact(text) == text


def test_our_own_identifiers_survive():
    """stable_hash is 12 chars. Redacting those would destroy every trace."""
    h = stable_hash({"tool": "search_files"})
    assert len(h) == 12
    line = f"signature t:{h} repeats 3x"
    assert redact(line) == line, "a 12-char internal hash must not be redacted"


def test_run_ids_and_step_numbers_survive():
    line = "run-4f2a9c71b0de step 42 tokens 11940000 cost $1.08"
    assert redact(line) == line


def test_a_long_plain_word_is_not_a_secret():
    """The high-entropy rule requires BOTH letters and digits, so prose is safe."""
    assert redact("antidisestablishmentarianismandmorewordshere") == \
        "antidisestablishmentarianismandmorewordshere"


def test_contains_secret_reports_honestly():
    assert contains_secret(f"key {OPENAI_KEY}")
    assert not contains_secret("nothing sensitive at all here")


def test_redact_obj_walks_containers():
    out = redact_obj({"args": {"token": SECRETS[3][1]},
                      "list": [SECRETS[2][1], 3], "n": 7})
    assert SECRETS[3][1] not in json.dumps(out)
    assert SECRETS[2][1] not in json.dumps(out)
    assert out["n"] == 7


# --------------------------------------------- the exit points, end to end
# A pattern library nothing calls is worth nothing, so each of the three egress
# paths is driven directly.
LEAK = f"connect with {OPENAI_KEY} and password=hunter2xyz"


def test_the_jsonl_trace_on_disk_has_no_secrets(tmp_path):
    path = tmp_path / "run.jsonl"
    tracer = Tracer(jsonl_path=str(path), echo=False)
    tracer.event(AgentEvent(type=EventType.TOOL_RESULT, step=1, tool_name="get_secret",
                            tool_args={"vault": LEAK}, text=LEAK))
    tracer.close()
    blob = path.read_text(encoding="utf-8")
    assert OPENAI_KEY not in blob
    assert "hunter2xyz" not in blob, "tool ARGUMENTS carry secrets too"
    assert "get_secret" in blob, "the trace must still be readable"


def test_the_supervisor_prompt_has_no_secrets():
    """This packet is sent to a model provider — it is egress like any other."""
    snap = ExecutionSnapshot(
        step=3, original_goal=GOAL, current_goal=None, total_tokens=1,
        total_cost_usd=0.0, route_history=["agent"],
        recent_events=[{"step": 2, "tool_name": "get_secret",
                        "tool_args": {"name": "db"}, "text": LEAK}],
        trip_reason=f"tool returned {LEAK}", trip_detector="loop",
        trip_evidence={"result": LEAK})
    out = snap.to_prompt_context()
    assert OPENAI_KEY not in out
    assert "hunter2xyz" not in out
    assert GOAL in out, "the operator's own objective must survive intact"


def test_the_escalation_payload_has_no_secrets():
    note = Notification(run_id="r1", reason=LEAK, detector="spend", step=9,
                        goal=GOAL, action="escalate", instruction=LEAK,
                        evidence={"result": LEAK})
    blob = json.dumps(note.payload())
    assert OPENAI_KEY not in blob
    assert "hunter2xyz" not in blob


def test_redaction_happens_before_truncation():
    """A credential cut in half by a length limit is no longer matchable.

    Padding pushes the secret past the 800-char cut, so a payload that truncated
    first would ship a recognisable fragment.
    """
    note = Notification(run_id="r", reason="x" * 780 + " " + SECRETS[0][1],
                        detector="loop", step=1, goal=GOAL, action="escalate")
    body = json.dumps(note.payload())
    assert OPENAI_KEY[:18] not in body, "a truncated secret fragment escaped"


def test_long_space_free_string_redacts_in_linear_time():
    """A 16 KB token with no whitespace took ~4s (quadratic lookahead) and this runs on
    every trace write."""
    import time
    start = time.perf_counter()
    redact("a-" * 8000)
    redact("ab" * 8000)
    assert time.perf_counter() - start < 1.0


def test_high_entropy_needs_letters_and_digits():
    mixed = "aB3" * 12
    assert redact(f"key {mixed} end") == "key [REDACTED:high-entropy] end"
    assert redact("x" * 40) == "x" * 40, "letters only: ordinary text, left alone"
    assert redact("1" * 40) == "1" * 40, "digits only: an id or number, left alone"
    assert redact("abcdef123456") == "abcdef123456", "12-char project hash must survive"


@pytest.mark.parametrize("text, secret", [
    ("db_password=hunter2hunter2", "hunter2hunter2"),
    ("DB_PASSWORD: hunter2hunter2", "hunter2hunter2"),
    ("my_api_key = abcd1234efgh", "abcd1234efgh"),
    ("STRIPE_CLIENT_SECRET=abcdef123456", "abcdef123456"),
    ("mypassword=hunter2hunter2", "hunter2hunter2"),
])
def test_prefixed_secret_names_are_redacted(text, secret):
    """A bare word boundary before `password` never matched inside `db_password`, so the
    most common real-world config names slipped through."""
    out = redact(text)
    assert secret not in out
    assert "[REDACTED:secret-assignment]" in out


def test_prefixed_names_do_not_swallow_ordinary_text():
    assert redact("the password policy is strict") == "the password policy is strict"
    url = "password_reset_url=https://example.com/reset"
    assert redact(url) == url


def test_prefixed_names_stay_linear_on_long_hyphenated_runs():
    import time
    start = time.perf_counter()
    redact("a-" * 20000)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("text, secret", [
    ('{"password": "hunter2hunter2"}', "hunter2hunter2"),
    ("{'api_key': 'abcd1234efgh'}", "abcd1234efgh"),
    ('{"db_password":"hunter2hunter2","user":"bob"}', "hunter2hunter2"),
    ('{"client_secret": "abcdef123456", "x": 1}', "abcdef123456"),
])
def test_json_style_quoted_names_are_redacted(text, secret):
    """`\"password\": \"...\"` has a closing quote between the name and the colon, so the
    assignment rule never matched JSON -- the shape tool arguments actually arrive in."""
    out = redact(text)
    assert secret not in out
    assert "[REDACTED:secret-assignment]" in out


def test_json_redaction_keeps_unrelated_fields():
    out = redact('{"password": "hunter2hunter2", "user": "bob"}')
    assert '"user": "bob"' in out


def test_trip_evidence_json_is_redacted():
    """events.py builds EVIDENCE with json.dumps, i.e. quoted names."""
    evidence = json.dumps({"tool": "connect", "password": "hunter2hunter2"})
    assert "hunter2hunter2" not in redact(evidence)


def test_redact_obj_redacts_values_under_sensitive_keys():
    """The trace writer calls redact_obj on the structured record. A bare value like
    "hunter2hunter2" has no shape to match, so the key name is the only signal."""
    rec = {"tool": "connect", "args": {"host": "db", "password": "hunter2hunter2",
                                       "api_key": "abcd1234", "refresh_token": "zzzz9999"}}
    out = redact_obj(rec)
    assert out["args"]["password"] == "[REDACTED:secret-assignment]"
    assert out["args"]["api_key"] == "[REDACTED:secret-assignment]"
    assert out["args"]["refresh_token"] == "[REDACTED:secret-assignment]"
    assert out["args"]["host"] == "db"
    assert out["tool"] == "connect"


def test_redact_obj_leaves_token_counts_and_short_values_alone():
    rec = {"max_tokens": "1024", "total_tokens": 50, "prompt_tokens": 12, "password": "x"}
    assert redact_obj(rec) == rec


def test_redact_obj_reaches_secrets_in_nested_lists():
    rec = {"calls": [{"headers": {"Authorization": "Bearer abc123def456ghi789"}}]}
    assert "abc123def456ghi789" not in json.dumps(redact_obj(rec))


def test_basic_auth_header_is_redacted():
    out = redact("Authorization: Basic dXNlcjpwYXNzd29yZDEyMzQ1")
    assert "dXNlcjpwYXNzd29yZDEyMzQ1" not in out
    assert out.startswith("Authorization: ")


def test_basic_auth_in_json_headers_is_redacted():
    out = redact('{"Authorization": "Basic dXNlcjpwYXNzd29yZDEyMzQ1"}')
    assert "dXNlcjpwYXNzd29yZDEyMzQ1" not in out


def test_the_word_basic_in_prose_is_left_alone():
    text = "Authorization is basic internationalization of the request"
    assert redact(text) == text


def test_url_password_containing_a_colon_is_redacted():
    out = redact("postgres://user:pa:ss@host/db")
    assert "pa:ss" not in out
    assert out.endswith("@host/db")
