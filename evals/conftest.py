"""Shared pytest fixtures for the eval suite.

``requires_embeddings`` — for tests whose assertions are about drift
*accuracy*, not drift *mechanism*. Lexical similarity is a documented,
intentionally weaker fallback (see ``agentfuse/detectors/drift.py``'s module
docstring): it cannot reliably separate gradual drift from a legitimate
paraphrase. A handful of tests here compare a real captured trace's outcome,
or a hand-written regression case, against a specific expected verdict --
those numbers were measured against the real embedder and have no reason to
hold under the fallback. Running them anyway does not test anything: pass or
fail, the result is a property of which backend happened to be installed,
not of the code.

Found the hard way: CI's "Tests" matrix job (9 cells, every supported OS x
Python combination) installs the package with no extras, by design, to keep
it a fast portability check rather than a second full accuracy suite --
the separate "benchmark gate" job installs ``.[embeddings]`` once and is
where drift accuracy is actually meant to be validated. Nobody had marked the
embedding-dependent tests as such, so the "Tests" job silently ran them
against the lexical fallback and failed the same four ways on every cell,
every run, for 6 days (2026-08-17 onward) before this was noticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.embedding import get_embedder  # noqa: E402

_, _EMBED_MODE = get_embedder()
_HAS_EMBEDDINGS = _EMBED_MODE.startswith("embedding")

requires_embeddings = pytest.mark.skipif(
    not _HAS_EMBEDDINGS,
    reason=(
        f"needs a real embedder, not the lexical fallback (resolved mode: "
        f"{_EMBED_MODE!r}). Install fastembed to run this locally: "
        f"pip install -e '.[embeddings]'"
    ),
)
