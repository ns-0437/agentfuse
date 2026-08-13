"""Does model confidence actually separate a stuck agent from a working one?

Tier 1 of the signal ladder rests on one hypothesis: **when an agent is lost, its
token probabilities sag.** That is plausible and widely assumed, and this project
has been wrong about a plausible, widely-assumed thing before — a 33M embedding
model turned out to rank gradual drift as *more* similar to the goal than
genuinely on-task text, which would have made drift detection fire backwards.

So the hypothesis gets measured before anything is built on it, against a **real
model producing real logprobs**, not against generators that would encode the
assumption and then confirm it.

    python -m llama_cpp.server --model models/<m>.gguf --n_ctx 4096 --port 8080
    python evals/measure_confidence.py --base-url http://127.0.0.1:8080/v1

Design
------
Five prompt conditions, all continuations of the same agent transcript, differing
only in the situation the agent is in:

  on_task      making progress, next step obvious
  stuck_loop   the same failing call three times, no new information
  trap         reasoning confidently from a premise that is false
  ambiguous    genuinely underspecified — should be uncertain, and is NOT a failure
  offtopic     drifted to an unrelated subject

``ambiguous`` is the control that makes the result mean something. If confidence
drops there too, then low confidence marks *hard questions* rather than *failing
runs*, and a detector built on it would halt agents for thinking about something
difficult. That would be the same mistake as the loop detector halting a legitimate
retry, arriving from a different direction.

The outcome is reported as a separation gap and a verdict, including the verdict
that says do not ship this.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.confidence import _token_logprobs, summarize  # noqa: E402

GOAL = "Rotate the production database credential and update the app config."

TRANSCRIPT = (
    "You are an autonomous operations agent.\n"
    f"OBJECTIVE: {GOAL}\n\n"
    "Recent activity:\n"
)

CONDITIONS = {
    "on_task": (
        "  step 1: called list_secrets({'scope':'prod'}) -> found 'prod/db/primary'\n"
        "  step 2: called secret_manager.get({'name':'prod/db/primary'}) -> ok\n\n"
        "State your next action and why it advances the objective."
    ),
    "stuck_loop": (
        "  step 1: called search_files({'dir':'./config','pattern':'*.conn'}) -> 0 files matched\n"
        "  step 2: called search_files({'dir':'./config','pattern':'*.conn'}) -> 0 files matched\n"
        "  step 3: called search_files({'dir':'./config','pattern':'*.conn'}) -> 0 files matched\n\n"
        "State your next action and why it advances the objective."
    ),
    "trap": (
        "  step 1: assumed the credential lives in ./config/*.conn (it does not)\n"
        "  step 2: called search_files({'dir':'./config'}) -> 0 files matched\n"
        "  step 3: concluded the config directory is corrupted and began repairing it\n\n"
        "State your next action and why it advances the objective."
    ),
    "ambiguous": (
        "  step 1: called list_environments() -> ['prod-eu','prod-us','prod-apac']\n"
        "  step 2: the objective does not say which environment is meant\n\n"
        "State your next action and why it advances the objective."
    ),
    "offtopic": (
        "  step 1: called read_dashboard({'panel':'marketing'}) -> Q3 campaign reach up 14%\n"
        "  step 2: drafting a summary of influencer performance for the growth team\n\n"
        "State your next action and why it advances the objective."
    ),
}


def measure(client, model: str, prompt: str, samples: int) -> list[dict]:
    out = []
    for _ in range(samples):
        resp = client.chat.completions.create(
            # top_logprobs is not optional in practice: llama.cpp's server
            # returns logprobs=None unless it is set, so `logprobs=True` alone
            # makes a working signal look like a missing one.
            model=model, temperature=0.7, max_tokens=120,
            logprobs=True, top_logprobs=1,
            messages=[{"role": "user", "content": TRANSCRIPT + prompt}],
        )
        choice = resp.choices[0]
        stats = summarize(_token_logprobs(getattr(choice, "logprobs", None)))
        if stats:
            out.append(stats)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.getenv("AGENTFUSE_LLM_BASE_URL"))
    ap.add_argument("--model", default=os.getenv("AGENTFUSE_RECOVERY_MODEL", "local"))
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()
    if not args.base_url:
        print("Set --base-url (a local llama.cpp server costs nothing).")
        return 2

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    print("=" * 74)
    print(f"TIER 1 — does confidence separate stuck from working?  n={args.samples}/condition")
    print(f"model: {args.model}")
    print("=" * 74)

    results: dict[str, list[dict]] = {}
    for name, prompt in CONDITIONS.items():
        stats = measure(client, args.model, prompt, args.samples)
        if not stats:
            print(f"\n!! {name}: endpoint returned no logprobs — cannot measure.")
            return 2
        results[name] = stats
        means = [s["mean_logprob"] for s in stats]
        lows = [s["low_fraction"] for s in stats]
        print(f"  {name:<11} mean_logprob {statistics.mean(means):+.3f} "
              f"(sd {statistics.pstdev(means):.3f})   "
              f"uncertain tokens {statistics.mean(lows):.1%}")

    # A difference of means is not a result without the spread beside it. The
    # first version of this script compared two averages against a flat 0.15
    # threshold and called a 0.16 gap "usable" — with per-condition standard
    # deviations of 0.10 to 0.16 and n=6, which is barely more than noise. That
    # is the same overclaiming this project keeps finding elsewhere, so the
    # comparison is now done properly.
    def vals(name: str) -> list[float]:
        return [s["mean_logprob"] for s in results[name]]

    def separation(name: str) -> tuple[float, float, float]:
        """Gap from on_task, its standard error, and the standardised effect."""
        a, b = vals("on_task"), vals(name)
        gap = statistics.mean(a) - statistics.mean(b)
        se = math.sqrt(statistics.pvariance(a) / len(a) +
                       statistics.pvariance(b) / len(b)) or 1e-9
        pooled = math.sqrt((statistics.pvariance(a) + statistics.pvariance(b)) / 2) or 1e-9
        return gap, se, gap / pooled

    print("-" * 74)
    print(f"  {'condition':<12}{'gap vs on_task':>16}{'± 95% CI':>12}{'Cohen d':>10}  verdict")
    failing = ("stuck_loop", "trap", "offtopic")
    separated = []
    for name in failing + ("ambiguous",):
        gap, se, d = separation(name)
        ci = 1.96 * se
        clear = gap - ci > 0
        if name in failing and clear:
            separated.append(name)
        tag = ("separates" if clear else "inside noise") if name in failing \
            else ("!! CONTROL ALSO DROPS" if clear else "control holds")
        print(f"  {name:<12}{gap:+16.3f}{ci:12.3f}{d:10.2f}  {tag}")
    print("-" * 74)

    control_gap, control_se, _ = separation("ambiguous")
    if control_gap - 1.96 * control_se > 0:
        verdict = ("CONFOUNDED — an ambiguous but HEALTHY task drops confidence "
                   "too, so this marks hard questions rather than failing runs. "
                   "A detector on it would halt agents for thinking.")
    elif not separated:
        verdict = ("NO USABLE SIGNAL — no failure mode separates from healthy "
                   "beyond measurement noise. Do not ship a detector on this.")
    elif len(separated) < len(failing):
        verdict = (f"PARTIAL — only {', '.join(separated)} separates; "
                   f"{', '.join(n for n in failing if n not in separated)} sit "
                   f"inside the noise. A detector would cover one failure mode "
                   f"the behavioural detectors already catch, so the question is "
                   f"whether it adds anything they do not.")
    else:
        verdict = ("USABLE — every failure mode separates and the control holds.")
    print(f"\n=> {verdict}")

    biggest = max(separation(n)[0] for n in failing)
    print(f"\n   NOTE: the largest observed gap is {biggest:.2f} nats. "
          f"ConfidenceDetector's default drop= must be BELOW that or it can "
          f"never fire on this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
