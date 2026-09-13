"""agentfuse/env.py — .env loading, offline detection, and the diagnostic report.

Zero test coverage existed for this module before this file. Written while
fixing describe()'s output to stop implying a bare OPENAI_API_KEY means the
reasoning-model recovery backend or hosted embeddings are in use — REPORT.md
3.34 found those had quietly diverged from what the key's presence implies,
and check_env.py (the user-facing diagnostic script) prints describe()'s
output directly, so an inaccurate report there misleads exactly the person
trying to understand their own setup.

    pytest evals/test_env.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import env  # noqa: E402


def _isolate(monkeypatch):
    for key in ("OPENAI_API_KEY", "AGENTFUSE_OFFLINE", "AGENTFUSE_RECOVERY_BACKEND",
               "AGENTFUSE_LLM_BASE_URL", "AGENTFUSE_EMBED_BACKEND"):
        monkeypatch.delenv(key, raising=False)


# ----------------------------------------------------------- read_env_text
def test_reads_utf16le_with_bom_the_way_powershell_redirect_writes_it(tmp_path):
    """The documented `printf ... > .env` on Windows PowerShell writes this.

    read_env_text()'s own docstring says this bug previously took nine other
    tests down with a UnicodeDecodeError -- none of them pinned the UTF-16
    case itself, so this does. Found again 2026-09-12: check_env.py had its
    own naive read_text(encoding="utf-8") that reproduced the exact bug this
    function was written to fix, garbling this project's own .env into
    mojibake and reporting the key as entirely missing.
    """
    p = tmp_path / ".env"
    p.write_bytes("OPENAI_API_KEY=sk-proj-utf16-test\n".encode("utf-16"))
    text = env.read_env_text(p)
    assert "OPENAI_API_KEY=sk-proj-utf16-test" in text


def test_reads_utf8_with_bom(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes("﻿OPENAI_API_KEY=sk-proj-utf8-bom-test\n".encode("utf-8"))
    text = env.read_env_text(p)
    assert "OPENAI_API_KEY=sk-proj-utf8-bom-test" in text


def test_reads_plain_utf8_without_a_bom(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes(b"OPENAI_API_KEY=sk-proj-plain-utf8-test\n")
    text = env.read_env_text(p)
    assert "OPENAI_API_KEY=sk-proj-plain-utf8-test" in text


def test_load_env_picks_up_a_key_from_a_utf16_file(tmp_path, monkeypatch):
    """The end-to-end path check_env.py and every example script relies on."""
    _isolate(monkeypatch)
    p = tmp_path / ".env"
    p.write_bytes("OPENAI_API_KEY=sk-proj-e2e-utf16-test\n".encode("utf-16"))
    assert env.load_env(path=p) is True
    assert os.environ.get("OPENAI_API_KEY") == "sk-proj-e2e-utf16-test"


# --------------------------------------------------------------- offline_mode
def test_offline_mode_reads_the_env_var_case_insensitively(monkeypatch):
    for value in ("1", "true", "True", "YES", "yes"):
        monkeypatch.setenv("AGENTFUSE_OFFLINE", value)
        assert env.offline_mode() is True
    monkeypatch.setenv("AGENTFUSE_OFFLINE", "0")
    assert env.offline_mode() is False


def test_offline_mode_forces_has_openai_key_false_even_with_a_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-looking-key")
    monkeypatch.setenv("AGENTFUSE_OFFLINE", "1")
    assert env.has_openai_key() is False


# ------------------------------------------------------- describe() accuracy
def test_describe_does_not_claim_real_recovery_from_a_bare_key(monkeypatch):
    """The exact staleness this file was written to catch (REPORT.md 3.34).

    A key alone must never make describe() claim the reasoning-model
    recovery backend is in use -- that requires an explicit opt-in.
    """
    _isolate(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-the-users-own-agent")
    out = env.describe()
    assert "recovery      : mock" in out, (
        f"describe() must report the backend actually in use, not one implied "
        f"by the key's presence:\n{out}")


def test_describe_reports_real_recovery_only_with_explicit_opt_in(monkeypatch):
    """_make_client stubbed so this checks describe()'s reporting logic, not
    whether `openai` happens to be installed in the current environment --
    found failing in CI's minimal-install "test" job for exactly that reason
    before this fix (RecoveryEngine gracefully falls back to mock when the
    SDK genuinely is not importable, which is correct production behavior;
    the test just needs to not depend on which environment it runs in)."""
    _isolate(monkeypatch)
    from agentfuse.recovery import RecoveryEngine
    monkeypatch.setattr(RecoveryEngine, "_make_client", lambda self: object())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
    monkeypatch.setenv("AGENTFUSE_RECOVERY_BACKEND", "real")
    out = env.describe()
    assert "recovery      : real" in out


def test_describe_with_no_key_still_reports_the_real_backend_choice(monkeypatch):
    _isolate(monkeypatch)
    # This repo's own dev .env carries a real key -- has_openai_key() would
    # reload it via load_env() the moment the in-process var is cleared, so
    # the "truly no key anywhere" case has to also stub out the file lookup.
    monkeypatch.setattr(env, "find_env_file", lambda start=None: None)
    monkeypatch.setattr(env, "load_env", lambda *a, **k: False)
    out = env.describe()
    assert "OPENAI_API_KEY: NOT SET" in out
    assert "recovery      : mock" in out
