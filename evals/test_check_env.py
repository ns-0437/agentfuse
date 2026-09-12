"""check_env.py — the user-facing .env diagnostic script.

Found live against this project's own real .env file while fixing this
script's UTF-16 handling: it also silently missed the OTHER documented .env
failure mode, a literal backslash-n instead of a real newline (PowerShell's
`>` redirect does not interpret `\\n`, so `printf 'A=1\\nB=2\\n' > .env`
writes the escape sequences verbatim). ``load_env()`` already works around
this internally, but check_env.py's own line-by-line diagnostic parsed the
raw text directly and would report a corrupted key as "all good" -- exactly
backwards for a script whose whole job is catching this before it causes a
confusing 401 with no obvious cause.

    pytest evals/test_check_env.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import check_env  # noqa: E402


def test_flags_a_literal_backslash_n_instead_of_reporting_all_good(tmp_path, monkeypatch, capsys):
    """The exact bug found live in this project's own .env file.

    Before this fix, this shape of file printed "All good" -- the API key
    value silently carried another variable's name and a literal \\n glued
    onto its end, which would fail authentication with a 401 pointing
    nowhere near the real cause (agentfuse/env.py's load_env() docstring).
    """
    p = tmp_path / ".env"
    p.write_text("OPENAI_API_KEY=sk-proj-realkey123\\nAGENTFUSE_RECOVERY_MODEL=o4-mini\\n",
                 encoding="utf-8")
    monkeypatch.setattr(check_env, "find_env_file", lambda: p)
    monkeypatch.setattr(check_env, "load_env", lambda: True)
    monkeypatch.setattr(check_env, "describe", lambda: "OPENAI_API_KEY: sk-proj…y123")

    check_env.main()
    out = capsys.readouterr().out
    assert "backslash-n" in out, (
        f"a literal \\n in the file must be flagged as a problem, not reported "
        f"as \"All good\":\n{out}")
    assert "All good" not in out


def test_a_normal_file_with_real_newlines_reports_no_backslash_n_problem(tmp_path, monkeypatch, capsys):
    p = tmp_path / ".env"
    p.write_text("OPENAI_API_KEY=sk-proj-realkey123\nAGENTFUSE_MODEL=gpt-4o-mini\n",
                 encoding="utf-8")
    monkeypatch.setattr(check_env, "find_env_file", lambda: p)
    monkeypatch.setattr(check_env, "load_env", lambda: True)
    monkeypatch.setattr(check_env, "describe", lambda: "OPENAI_API_KEY: sk-proj…y123")

    check_env.main()
    out = capsys.readouterr().out
    assert "unexpected name" not in out
    assert "All good" in out


def test_the_new_explicit_opt_in_vars_are_not_flagged_as_unexpected(tmp_path, monkeypatch, capsys):
    """A reader following REPORT.md 3.34's own fix into their .env must not
    have this script call it a mistake -- the regression this test prevents."""
    p = tmp_path / ".env"
    p.write_text(
        "OPENAI_API_KEY=sk-proj-realkey123\n"
        "AGENTFUSE_RECOVERY_BACKEND=real\n"
        "AGENTFUSE_EMBED_BACKEND=none\n",
        encoding="utf-8")
    monkeypatch.setattr(check_env, "find_env_file", lambda: p)
    monkeypatch.setattr(check_env, "load_env", lambda: True)
    monkeypatch.setattr(check_env, "describe", lambda: "OPENAI_API_KEY: sk-proj…y123")

    check_env.main()
    out = capsys.readouterr().out
    assert "unexpected name" not in out, (
        f"AGENTFUSE_RECOVERY_BACKEND/AGENTFUSE_EMBED_BACKEND must be recognised, "
        f"not reported as a mistake:\n{out}")
    assert "All good" in out
