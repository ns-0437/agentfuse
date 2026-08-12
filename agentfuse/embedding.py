"""Embedding backends — local by default, API only if you ask for it.

The drift detector needs to know whether the agent's current focus still means
the same thing as its objective. Lexical overlap cannot express that, and the
measurements are unambiguous about it.

Measured on the case that matters — gradual drift, where the agent keeps using
the goal's vocabulary while its intent slides away:

    signal                       on-task   paraphrase   GRADUAL DRIFT   separable?
    lexical (difflib+jaccard)      0.323       0.332          0.276     ~0.05 window
    bge-small-en-v1.5   (33M)      0.712       0.764          0.756     NO - inverted
    bge-base-en-v1.5   (110M)      0.708       0.769          0.665     YES, gap +0.043

Two things follow, and both are worth stating because they are counterintuitive:

1. **Model size has a floor, and it is not where you would guess.** The 33M model
   is not merely weaker — it ranks gradual drift as *more* similar to the goal
   than genuinely on-task text. Any threshold built on it fires in the wrong
   direction. 110M works. Nothing here needs a billion parameters; sentence
   similarity is not a task that rewards scale the way generation does.

2. **Local beats API for this workload.** The model is ~120MB of ONNX, runs on
   CPU in about 4ms per sentence, needs no key, costs nothing, and cannot leak
   the agent's reasoning to a third party. An embedding call per turn against a
   hosted API would dominate the supervision latency budget (measured at ~40us
   per event) and add a network failure mode to a component whose entire job is
   to stay up when things go wrong.

So the resolution order is: an explicitly injected embedder, then a local model,
then a hosted one, then lexical. ``AGENTFUSE_OFFLINE`` disables the *hosted*
backend only — a local ONNX model spends no money and touches no network, so
disabling it would confuse "don't bill me" with "don't think".
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from .env import load_env, offline_mode

#: Default local model. 110M parameters, 768 dimensions, ~120MB on disk.
#: bge-small is deliberately NOT the default despite being faster: it scores
#: gradual drift as closer to the goal than on-task text, which is worse than
#: having no signal at all.
DEFAULT_LOCAL_MODEL = "BAAI/bge-base-en-v1.5"

_local_cache: dict[str, Callable[[str], list[float]]] = {}


def local_embedder(model_name: Optional[str] = None) -> Optional[Callable[[str], list[float]]]:
    """A CPU embedder backed by fastembed's ONNX runtime, or None if unavailable.

    The model is loaded once per process and cached: construction takes a second
    or two, per-sentence inference about 4ms.
    """
    name = model_name or os.getenv("AGENTFUSE_LOCAL_EMBED_MODEL", DEFAULT_LOCAL_MODEL)
    if name in _local_cache:
        return _local_cache[name]

    try:
        from fastembed import TextEmbedding  # type: ignore
    except ImportError:
        return None

    try:
        model = TextEmbedding(model_name=name)
    except Exception:
        return None

    def embed(text: str) -> list[float]:
        return list(next(iter(model.embed([text[:8000]]))))

    _local_cache[name] = embed
    return embed


def openai_embedder() -> Optional[Callable[[str], list[float]]]:
    """A hosted embedder, if a key is configured and network use is permitted."""
    load_env()
    if offline_mode() or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        model = os.getenv("AGENTFUSE_EMBED_MODEL", "text-embedding-3-small")

        def embed(text: str) -> list[float]:
            return client.embeddings.create(model=model, input=text[:8000]).data[0].embedding

        return embed
    except Exception:
        return None


def get_embedder(prefer: Optional[str] = None) -> tuple[Optional[Callable[[str], list[float]]], str]:
    """Resolve the best available embedder. Returns ``(embedder, mode)``.

    ``prefer`` may be ``"local"``, ``"openai"`` or ``"none"``; otherwise the order
    is local, then hosted, then nothing (leaving the caller on lexical).
    """
    prefer = prefer or os.getenv("AGENTFUSE_EMBED_BACKEND", "auto")

    if prefer == "none":
        return None, "lexical"
    if prefer == "openai":
        e = openai_embedder()
        return (e, "embedding:openai") if e else (None, "lexical")
    if prefer == "local":
        e = local_embedder()
        return (e, "embedding:local") if e else (None, "lexical")

    e = local_embedder()
    if e:
        return e, "embedding:local"
    e = openai_embedder()
    if e:
        return e, "embedding:openai"
    return None, "lexical"


def describe() -> str:
    """Human-readable summary of which backend would be used, and why."""
    _, mode = get_embedder()
    if mode == "embedding:local":
        return (f"local ONNX embeddings ({DEFAULT_LOCAL_MODEL}) — free, offline, "
                f"~4ms/sentence on CPU")
    if mode == "embedding:openai":
        return "hosted OpenAI embeddings — billed per call, adds network latency"
    return ("lexical fallback — cannot separate gradual drift from a legitimate "
            "paraphrase; install fastembed for the local model")
