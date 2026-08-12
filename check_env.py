"""Check that AgentFuse can see your API key — safe to run and screenshot.

Prints a masked key and diagnoses the common .env mistakes (a bare key with no
variable name, smart quotes, a placeholder left in place). Never prints the key.

    python check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentfuse.env import find_env_file, load_env, describe  # noqa: E402

EXPECTED = ("OPENAI_API_KEY", "AGENTFUSE_MODEL",
            "AGENTFUSE_RECOVERY_MODEL", "AGENTFUSE_EMBED_MODEL")


def main() -> int:
    env_path = find_env_file()
    print()
    if env_path is None:
        print("No .env file found. Create one in the agentfuse/ folder.")
        return 1

    print(f".env found: {env_path}")
    problems: list[str] = []
    seen: set[str] = set()

    for i, raw in enumerate(env_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            # By far the most common failure: the key pasted on its own line.
            hint = " (looks like a bare API key — it needs OPENAI_API_KEY= in front)" \
                if line.lower().startswith("sk-") else ""
            problems.append(f"line {i}: no '=' sign{hint}")
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        seen.add(key)
        if key not in EXPECTED:
            problems.append(f"line {i}: unexpected name {key!r}")
        if value.upper().startswith(("SK-YOUR", "SK-PROJ-YOUR", "SK-PROJ-PASTE")):
            problems.append(f"line {i}: {key} still holds the placeholder text")
        if value.startswith(("'", '"', "“", "‘")):
            problems.append(f"line {i}: {key} value is quoted — remove the quotes")

    if "OPENAI_API_KEY" not in seen:
        problems.append("OPENAI_API_KEY line is missing entirely")

    print()
    load_env()
    print(describe())

    if problems:
        print("\nProblems found:")
        for p in problems:
            print(f"  - {p}")
        print("\nEasiest fix — open the file and edit it directly:")
        print("    notepad .env")
        print("\nIt should look exactly like this (no quotes, no spaces around '='):")
        print("    OPENAI_API_KEY=sk-proj-...")
        print("    AGENTFUSE_MODEL=gpt-4o-mini")
        print("    AGENTFUSE_RECOVERY_MODEL=o4-mini")
        print("    AGENTFUSE_EMBED_MODEL=text-embedding-3-small")
        return 1

    print("\nAll good — AgentFuse can see your key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
