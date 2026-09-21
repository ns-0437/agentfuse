"""Small, dependency-free diagnostics for an AgentFuse installation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from typing import Optional

from . import __version__
from .embedding import describe


def doctor() -> dict:
    """Return capabilities without importing optional runtime SDKs."""
    return {
        "agentfuse_version": __version__,
        "core": "ok",
        "drift_backend": describe(),
        "integrations": {
            "agents_sdk": importlib.util.find_spec("agents") is not None,
            "langgraph": importlib.util.find_spec("langgraph") is not None,
            "openai": importlib.util.find_spec("openai") is not None,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="agentfuse")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor", help="show installed supervision capabilities")
    check.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = doctor()
    if args.as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"AgentFuse {report['agentfuse_version']} - core {report['core']}")
        print(f"Drift: {report['drift_backend']}")
        for name, available in report["integrations"].items():
            print(f"{name}: {'available' if available else 'not installed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
