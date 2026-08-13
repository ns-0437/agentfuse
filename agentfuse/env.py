"""Zero-dependency ``.env`` loading.

Only ``examples/real_gpt_run.py`` knew how to read a ``.env`` file, so a key
sitting in the project root was invisible to the library, the evals and every
other example. That made "run it against a real model" needlessly fiddly.

This module centralises it: call :func:`load_env` once at entry and the key is
available to the recovery engine, the drift detector's embedding backend, and
the eval harness alike.

Deliberately not ``python-dotenv`` — the core stays dependency-free, and the
parsing needed here is a dozen lines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED = False


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from ``start`` looking for a ``.env`` file."""
    here = (start or Path(__file__).resolve().parent).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _read_env_text(env_path) -> str:
    """Read a .env whatever encoding the operating system wrote it in.

    Reading it as plain UTF-8 was a real bug, and a Windows-shaped one. The
    documented way to create this file is a shell redirect, and PowerShell's
    ``>`` writes **UTF-16LE with a BOM** by default — so the most likely way a
    Windows user follows the instructions produces a file the loader cannot
    read. It did not degrade either: `load_env()` raised UnicodeDecodeError out
    of every import path that touches it, taking nine tests down with it.

    Tries UTF-8, then the BOM-sniffing codecs, then falls back to replacing
    undecodable bytes — a mangled line is skipped by the KEY=value parser
    below, which is a far better outcome than an exception from a helper whose
    entire job is "find the key if there is one".
    """
    raw = env_path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        # utf-16 happily decodes UTF-8 bytes into mojibake; a real .env always
        # contains "=", so use that as the sanity check rather than trusting
        # the decoder not to have guessed.
        if "=" in text:
            return text
    return raw.decode("utf-8", errors="replace")


def load_env(path: Optional[Path] = None, override: bool = False) -> bool:
    """Load ``KEY=value`` pairs from a ``.env`` file into ``os.environ``.


    Existing environment variables win unless ``override`` is set, so a value
    exported in the shell is never silently replaced by a stale file. Returns
    whether a file was found. Never logs values.
    """
    global _LOADED
    env_path = path or find_env_file()
    if env_path is None or not env_path.is_file():
        return False

    # Split on literal backslash-n as well as real newlines.
    #
    # Windows PowerShell does not interpret \n in a redirect, so the documented
    # `printf 'A=1\nB=2\n' > .env` produces ONE line containing the escape
    # sequences verbatim. The parser then read the entire file as a single
    # KEY=value pair and handed back an API key with
    # "\nAGENTFUSE_RECOVERY_MODEL=o4-mini\n" glued to the end — a key that
    # looks present, has a plausible length, and fails authentication with a
    # 401 that points nowhere near the real cause. Measured on this machine:
    # key length 200 instead of 164, and the second variable silently absent.
    text = _read_env_text(env_path).replace("\\n", "\n")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value

    _LOADED = True
    return True


def offline_mode() -> bool:
    """True when network calls are explicitly disabled.

    The benchmark must be free and deterministic by default. Without this an
    ordinary `pytest` run on a machine that happens to have a key configured
    would fire thousands of real embedding and reasoning calls and bill the
    user, purely as a side effect of the key existing.
    """
    return os.getenv("AGENTFUSE_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def has_openai_key() -> bool:
    """True when a usable OpenAI key is present (loading ``.env`` if needed)."""
    if offline_mode():
        return False
    if not os.getenv("OPENAI_API_KEY"):
        load_env()
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith("sk-your")


def describe() -> str:
    """A short, safe status line. Never reveals the key itself."""
    env_path = find_env_file()
    if has_openai_key():
        key = os.environ["OPENAI_API_KEY"]
        masked = f"{key[:7]}…{key[-4:]}" if len(key) > 14 else "set"
        return (f"OPENAI_API_KEY: {masked}\n"
                f"  source        : {env_path if _LOADED else 'shell environment'}\n"
                f"  agent model   : {os.getenv('AGENTFUSE_MODEL', 'gpt-4o-mini')}\n"
                f"  recovery model: {os.getenv('AGENTFUSE_RECOVERY_MODEL', 'o4-mini')}\n"
                f"  embed model   : {os.getenv('AGENTFUSE_EMBED_MODEL', 'text-embedding-3-small')}")
    return ("OPENAI_API_KEY: NOT SET — AgentFuse will run in offline mode\n"
            "  (mock recovery + lexical drift; everything still works, "
            "just without real-model validation)")


if __name__ == "__main__":
    load_env()
    print(describe())
