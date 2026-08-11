# Citations & prior work

## Endogenous Steering Resistance (AE Studio)

AgentFuse's evaluation harness adapts the **experimental methodology** from AE
Studio's work on Endogenous Steering Resistance:

```bibtex
@article{mckenzie2026esr,
  title={Endogenous Resistance to Activation Steering in Language Models:
         Evidence for Internal Consistency Monitoring in Llama-3.3-70B},
  author={McKenzie, Alex and Pepper, Keenan and Servaes, Stijn and
          Leitgab, Martin and Cubuktepe, Murat and Vaiana, Mike and
          de Lucena, Diogo and Rosenblatt, Judd and Graziano, Michael S. A.},
  journal={arXiv preprint arXiv:2602.06941},
  year={2026}
}
```

- Paper: <https://arxiv.org/abs/2602.06941>
- Project page: <https://www.ae.studio/research/esr>
- Code: <https://github.com/agencyenterprise/endogenous-steering-resistance> (Apache-2.0)

### What we took, precisely

Two pieces of **experimental design**, implemented independently in
[`evals/ablation.py`](evals/ablation.py):

1. **Leave-one-out ablation to establish causal contribution.** They zero-ablated
   26 SAE latents and measured the resulting 25% drop in self-correction. We
   disable one detector at a time and measure the delta in recall / F1 / net
   tokens, so each detector's contribution is a measurement rather than a claim.
   This is what revealed that our `NoProgressDetector` contributes exactly 0.0 F1.

2. **A frequency-matched random control.** They controlled against random latents
   *matched for activation frequency*, and reported that a control ablation at
   4.2% fell within error bars of the 3.8% baseline. We run a random detector
   rate-matched to our own trip frequency across 25 seeds and test the difference
   empirically. Without this, a system that simply trips often would post a
   respectable F1.

Their reporting of a **7,892-trial** baseline is also why our first 16-scenario
suite was scrapped and rebuilt as a 536-scenario generated benchmark with
confidence intervals.

### What we did **not** take

**No code from their repository is used, vendored, copied, or derived from.**
AgentFuse's harness is an independent implementation. We have not run their
experiments — reproducing them requires 2×H100 GPUs (≥90GB VRAM), Llama-3.3-70B,
and a patched vLLM fork, none of which we have access to. Their numbers are cited
as published, not reproduced by us.

Citing this work implies no endorsement of AgentFuse by AE Studio or the authors.

### How the two lines of work relate

ESR studies an **intrinsic** signal: a model's own capacity to notice it has been
pushed off-topic and self-correct mid-generation. AgentFuse is an **extrinsic**
control: an independent supervisor outside the model that trips deterministically
on observable behaviour.

They are complementary rather than competing, and neither is sufficient alone:

- ESR fires at roughly **3.8%** on Llama-3.3-70B (≤1.1% on smaller models, 18.3%
  after fine-tuning). A safety mechanism that engages one time in twenty-five is
  a discovered tendency, not a control lever.
- An external monitor is deterministic and model-agnostic, but is blind to the
  model's internal state and can only reason about observable behaviour.

The interesting synthesis, which is on the AgentFuse roadmap as the "signal
ladder", is that ESR demonstrates the internal off-topic signal genuinely exists
and is causally implicated — the model simply acts on it rarely. Extracting that
signal directly (rather than waiting for the model to act on it) and fusing it
with behavioural evidence is strictly better than either alone.
