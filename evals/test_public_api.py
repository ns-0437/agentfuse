"""The package's own advertised surface — does it actually import?

Found by inspection, not by a failing test: `from agentfuse import
RateOfProgressDetector` raised ImportError even though the class is documented,
implemented, and re-exported from `agentfuse.detectors`. The top-level package
had simply never re-exported it. Nothing here previously would have caught
that — every other test imports what it needs directly from the submodule it
lives in, so the top-level `__init__.py` had no test surface of its own at all.

    pytest evals/test_public_api.py -v
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agentfuse  # noqa: E402
from agentfuse.detectors import __all__ as DETECTOR_EXPORTS  # noqa: E402


def test_every_name_in_all_is_actually_importable():
    """`__all__` is a promise; every name in it must resolve to something."""
    missing = [name for name in agentfuse.__all__ if not hasattr(agentfuse, name)]
    assert not missing, f"__all__ names {missing} but agentfuse.py has no such attribute"


def test_all_five_detectors_are_exported_from_the_top_level_package():
    """The exact regression: agentfuse.detectors exported all five; the
    top-level package silently dropped one of them."""
    for name in DETECTOR_EXPORTS:
        if name in ("Detector", "Trip", "Severity"):
            continue          # base contract types, not detector classes
        assert hasattr(agentfuse, name), (
            f"{name} is exported from agentfuse.detectors but not from the "
            f"top-level agentfuse package")
        assert name in agentfuse.__all__, f"{name} importable but missing from __all__"


def test_the_docstring_names_all_five_failure_modes():
    """The module docstring is a user-facing claim about what the library
    catches; a detector that exists but isn't named in it is invisible to
    someone skimming `help(agentfuse)`."""
    doc = agentfuse.__doc__ or ""
    for phrase in ("loop", "drift", "logical trap", "Zeno", "spend"):
        assert phrase.lower() in doc.lower(), f"docstring never mentions {phrase!r}"


def test_the_all_extra_is_a_superset_of_every_other_extra():
    """Found the same way: `[all]` was missing the entire `langgraph` extra.
    A generic check beats another one-off list, so this can't silently drop
    a future extra the same way.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    all_pkgs = {pkg.split(">=")[0].split("==")[0] for pkg in extras["all"]}
    for name, pkgs in extras.items():
        if name == "all":
            continue
        for pkg in pkgs:
            base = pkg.split(">=")[0].split("==")[0]
            assert base in all_pkgs, (
                f"extra {name!r} requires {base!r}, which is missing from [all]")


def test_adapters_package_imports_without_openai_agents_installed():
    """The promise `adapters/__init__.py` now documents in a comment: two of
    three adapters must not require the optional `openai-agents` package.

    `agentkit_hooks.py` (FuseRunHooks/BreakerInterrupt) hard-imports `agents`
    at module level, so it is deliberately excluded from this package's
    unconditional imports. This simulates `agents` being absent and confirms
    the promise actually holds rather than trusting the comment.
    """
    import builtins
    import importlib

    real_import = builtins.__import__

    def blocking_import(name, *a, **kw):
        if name == "agents" or name.startswith("agents."):
            raise ImportError(f"simulated: {name!r} is not installed")
        return real_import(name, *a, **kw)

    for mod in list(sys.modules):
        if mod == "agents" or mod.startswith("agentfuse.adapters"):
            del sys.modules[mod]

    builtins.__import__ = blocking_import
    try:
        adapters_module = importlib.import_module("agentfuse.adapters")
        assert hasattr(adapters_module, "AgentKitBreaker")
        assert hasattr(adapters_module, "guarded_tool_loop")
        assert hasattr(adapters_module, "FuseCallbackHandler")
    finally:
        builtins.__import__ = real_import
        for mod in list(sys.modules):
            if mod.startswith("agentfuse.adapters"):
                del sys.modules[mod]
        importlib.import_module("agentfuse.adapters")
