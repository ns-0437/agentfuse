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

    for raw in env_path.read_text(encoding="utf-8").splitlines():
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


def has_openai_key() -> bool:
    """True when a usable OpenAI key is present (loading ``.env`` if needed)."""
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
