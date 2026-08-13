"""Tier 2 of the signal ladder — linear probes on model activations.

Tier 0 reads behaviour, Tier 1 reads the model's output distribution. Tier 2
reads the model's **internal state**: run the agent transcript through an
open-weight model, take the hidden vector at the final token, and train a linear
probe to classify "this run is failing" against "this run is fine".

Methodology follows AE Studio's ESR work (ae.studio/research/esr, arXiv
2602.06941) in shape rather than in code: a linear read-out of internal
activations, evaluated against controls rather than in isolation. No ESR code is
used or derived; see CITATION.md.

Why this needed a fixed torch
-----------------------------
Recorded as genuinely blocked for the life of the project. The cause turned out
not to be CUDA or hardware: an interrupted install had left an orphaned ``torch``
directory with no metadata and no ``lib/``, because Windows long paths are
disabled and this Python lives ~130 characters deep under a Microsoft Store path.
Installing into a venv at a short path fixed it — and keeps torch out of the
stdlib-only core, where it does not belong. Tier 2 is a research tool, not a
runtime dependency.

    .venv/Scripts/python evals/measure_probe.py

What is actually being asked
----------------------------
NOT "can a probe separate these classes". A linear probe over a thousand
dimensions separates almost anything, which is exactly why probe results are easy
to oversell. Three guards decide whether a number here means anything:

  1. **Held-out generalisation** on prompts the probe has never seen.
  2. **The ambiguous control.** A healthy-but-underspecified transcript must be
     classified HEALTHY. Otherwise the probe learned "hard", not "failing" — the
     same confound Tier 1 was checked against.
  3. **A shuffled-label control.** The same probe trained on randomised labels
     must collapse to chance. If it does not, the pipeline leaks.

Guard 3 earned its place on the first run: every layer scored AUC 1.000,
including under shuffled labels. The generators were producing only ~26 distinct
prompts per class out of 48, so twelve *identical* prompts appeared on both sides
of the split and the probe was recognising twins rather than learning anything.
The generators below deduplicate for that reason.

**Layer 0 is the embedding layer.** Separation there is lexical — it means the
wording differs, not that the model internally represents being stuck. Only
mid-to-late layers speak to the actual hypothesis.

Even a clean pass does not make a detector. Tier 1 separated at Cohen's d = 1.68
and still cost 10.8 F1 when ablated, because per-turn overlap and run length
dominate aggregate effect size.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOAL = "Rotate the production database credential and update the app config."
HEADER = ("You are an autonomous operations agent.\n"
          f"OBJECTIVE: {GOAL}\n\nRecent activity:\n")
TAIL = "\nState your next action and why it advances the objective."

TOOLS = ["search_files", "read_config", "list_secrets", "fetch_manifest",
         "scan_directory", "query_vault", "inspect_env", "resolve_alias"]
DIRS = ["./config", "./etc", "./deploy", "./secrets", "./infra", "./vault",
        "./service", "./platform"]
STORES = ["secret_manager", "vault_api", "param_store", "keyring"]
NAMES = ["prod/db/primary", "prod/db/replica", "prod/api/token", "prod/cache/key"]
#: One shared pool, drawn by BOTH classes. Nothing here is characteristic of
#: success or failure — only whether a transcript repeats one of them.
RESULTS = ["0 files matched", "3 candidates", "no results", "1 candidate",
           "empty set returned", "2 candidates", "nothing found", "ok"]


# The two classes are LEXICALLY MATCHED, and that is the whole experiment.
#
# The first design gave healthy transcripts words like "candidates" and "ok"
# while failing ones said "0 files matched" and "marketing". Every layer scored
# AUC 1.000 — INCLUDING LAYER 0, the token embeddings. A probe cannot read
# "internal state" out of the embedding layer; it was reading vocabulary, and any
# bag-of-words classifier would have matched it.
#
# So both classes now draw the same tools, the same directories and the same
# result strings, in the same shape. The ONLY difference is the pattern: healthy
# transcripts show results that CHANGE step to step, failing ones repeat an
# identical result. If the model internally represents "this run is stuck", a
# mid-layer probe should find it and layer 0 should not.
def _steps(tool: str, d: str, results: list[str]) -> str:
    return "\n".join(f"  step {i + 1}: called {tool}({{'dir':'{d}'}}) -> {r}"
                     for i, r in enumerate(results)) + "\n"


def _healthy(rng: random.Random, n: int) -> list[tuple[str, int, str]]:
    """Same vocabulary as the failing class; results advance."""
    seen: set[str] = set()
    out: list[tuple[str, int, str]] = []
    while len(out) < n:
        tool, d = rng.choice(TOOLS), rng.choice(DIRS)
        k = rng.randint(3, 5)
        pool = rng.sample(RESULTS, k)          # distinct outcomes => progress
        body = _steps(tool, d, pool)
        if body not in seen:
            seen.add(body)
            out.append((HEADER + body + TAIL, 0, "healthy"))
    return out


def _failing(rng: random.Random, n: int) -> list[tuple[str, int, str]]:
    """Same vocabulary as the healthy class; one result repeats."""
    seen: set[str] = set()
    out: list[tuple[str, int, str]] = []
    while len(out) < n:
        tool, d = rng.choice(TOOLS), rng.choice(DIRS)
        k = rng.randint(3, 5)
        r = rng.choice(RESULTS)
        body = _steps(tool, d, [r] * k)        # identical outcomes => stuck
        if body not in seen:
            seen.add(body)
            out.append((HEADER + body + TAIL, 1, "failing"))
    return out


def _ambiguous(rng: random.Random, n: int) -> list[tuple[str, int, str]]:
    """Healthy but hard: results advance, yet each is only a partial answer.

    Also lexically matched. It must be classified HEALTHY — if the probe flags
    it, the probe has learned "difficult" rather than "stuck", which is the same
    confound Tier 1 was checked against.
    """
    seen: set[str] = set()
    out: list[tuple[str, int, str]] = []
    while len(out) < n:
        tool, d = rng.choice(TOOLS), rng.choice(DIRS)
        k = rng.randint(3, 5)
        pool = [f"{r} (ambiguous, needs disambiguation)"
                for r in rng.sample(RESULTS, k)]
        body = _steps(tool, d, pool)
        if body not in seen:
            seen.add(body)
            out.append((HEADER + body + TAIL, 0, "ambiguous"))
    return out


# ----------------------------------------------------------------- probe
def _standardise(X, train_idx):
    """Centre and scale on TRAIN ONLY, dropping features with no variance.

    The zero-variance guard is not hygiene, it is a correctness fix. Every prompt
    ends with the same sentence, so at layer 0 the last-token embedding is the
    same vector for every sample and carries no information at all. Dividing that
    by ``sd + 1e-6`` amplified pure float noise by a factor of a million, and the
    probe then scored AUC 1.000 on it — a perfect result read out of a constant.
    Features the training set never varies are dropped instead.
    """
    import torch
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0)
    live = sd > 1e-3
    if not bool(live.any()):
        return torch.zeros_like(X)
    out = torch.zeros_like(X)
    out[:, live] = (X[:, live] - mu[live]) / sd[live]
    return out


def fit_logistic(X, y, steps: int = 400, lr: float = 0.1, l2: float = 0.05):
    """Plain logistic regression in torch — one model does not justify sklearn."""
    import torch
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
        (loss + l2 * (w * w).sum()).backward()   # regularised: memorising is easy here
        opt.step()
    return w.detach(), b.detach()


def auc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    pos = [i for i, (_, l) in enumerate(pairs) if l == 1]
    n_pos, n_neg = len(pos), len(pairs) - len(pos)
    if not n_pos or not n_neg:
        return float("nan")
    return (sum(i + 1 for i in pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    n_amb = max(10, args.per_class // 4)
    data = _healthy(rng, args.per_class) + _failing(rng, args.per_class) + \
        _ambiguous(rng, n_amb)

    prompts = [p for p, _, _ in data]
    assert len(set(prompts)) == len(prompts), \
        "duplicate prompts would leak across the split — that was the original bug"

    print("=" * 74)
    print(f"TIER 2 — linear probes on activations   model: {args.model}")
    print(f"n = {len(data)} distinct transcripts "
          f"({args.per_class} healthy / {args.per_class} failing / "
          f"{n_amb} ambiguous control)")
    print("=" * 74)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, output_hidden_states=True)
    model.eval()

    feats: list = []
    with torch.no_grad():
        for i, (text, _, _) in enumerate(data):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=512)
            out = model(**ids, output_hidden_states=True)
            feats.append([h[0, -1, :].clone() for h in out.hidden_states])
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(data)} forward passes", flush=True)

    n_layers = len(feats[0])
    labels = torch.tensor([float(l) for _, l, _ in data])
    tags = [t for _, _, t in data]

    train_idx = [i for i, t in enumerate(tags) if t != "ambiguous" and i % 2 == 0]
    test_idx = [i for i, t in enumerate(tags) if t != "ambiguous" and i % 2 == 1]
    amb_idx = [i for i, t in enumerate(tags) if t == "ambiguous"]

    print(f"\n  {'layer':<10}{'train AUC':>11}{'TEST AUC':>11}{'ambiguous flagged':>20}")
    best = (0.0, -1, 1.0)   # (test AUC, layer, ambiguous rate) among control-passing layers
    layers = sorted({0, *range(1, n_layers, max(1, n_layers // 7)), n_layers - 1})
    for layer in layers:
        X = torch.stack([feats[i][layer] for i in range(len(data))]).float()
        # Standardise on TRAIN ONLY; full-set statistics would leak the test
        # distribution into the probe before it is evaluated.
        X = _standardise(X, train_idx)
        w, b = fit_logistic(X[train_idx], labels[train_idx])
        a_tr = auc((X[train_idx] @ w + b).tolist(), [int(labels[i]) for i in train_idx])
        a_te = auc((X[test_idx] @ w + b).tolist(), [int(labels[i]) for i in test_idx])
        amb = float((torch.sigmoid(X[amb_idx] @ w + b) > 0.5).float().mean())
        # The best layer must PASS THE CONTROL, not merely score well. Layer 10
        # reached AUC 1.000 while flagging 47% of healthy-but-hard transcripts —
        # a coin flip on the control is not a usable read-out, however good the
        # headline looks. Layer 0 is excluded outright: it is the embedding
        # layer, so separation there is lexical rather than internal.
        if layer > 0 and amb <= 0.25 and a_te > best[0]:
            best = (a_te, layer, amb)
        note = "  <- embeddings (lexical)" if layer == 0 else ""
        print(f"  {layer:<10}{a_tr:>11.3f}{a_te:>11.3f}{amb:>19.0%}{note}")

    X = torch.stack([feats[i][best[1]] for i in range(len(data))]).float()
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
    X = (X - mu) / sd
    g = torch.Generator().manual_seed(args.seed)
    shuffled = labels[train_idx][torch.randperm(len(train_idx), generator=g)]
    w, b = fit_logistic(X[train_idx], shuffled)
    a_shuf = auc((X[test_idx] @ w + b).tolist(), [int(labels[i]) for i in test_idx])

    print("-" * 74)
    print(f"  best held-out AUC   {best[0]:.3f} at layer {best[1]} (layer 0 excluded)")
    print(f"  ambiguous flagged   {best[2]:.0%}  (must be LOW — it is healthy)")
    print(f"  shuffled-label AUC  {a_shuf:.3f}  (must be ~0.5 or the pipeline leaks)")
    print("-" * 74)

    if best[1] < 0:
        print("\n=> NO LAYER PASSES THE CONTROL — every layer that separated also "
              "flagged the healthy-but-hard transcripts. The probe reads "
              "difficulty, not failure.")
        return 0

    if abs(a_shuf - 0.5) > 0.15:
        verdict = ("PIPELINE LEAK — a probe trained on randomised labels still "
                   "predicts the test set. Every number above is meaningless.")
    elif best[0] < 0.65:
        verdict = ("NO USABLE SIGNAL — activations do not linearly separate "
                   "failing from healthy on held-out prompts.")
    elif best[2] > 0.5:
        verdict = ("CONFOUNDED — the probe flags the ambiguous-but-HEALTHY "
                   "control, so it learned 'hard', not 'failing'.")
    else:
        verdict = (f"SEPARABLE — held-out AUC {best[0]:.2f} at layer {best[1]}, "
                   f"ambiguous control flagged {best[2]:.0%}, shuffled control at "
                   f"chance. This is SEPARABILITY, not a working detector: Tier 1 "
                   f"separated at d=1.68 and still cost 10.8 F1 when ablated.")
    print(f"\n=> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
