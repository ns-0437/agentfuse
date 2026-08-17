"""Re-derive the labels on committed captures from the traces themselves.

The four `real_*.json` captures were labelled `should_trip=True, detector=loop`
from runs taken under `--chat_format chatml-function-calling`, which cannot
terminate. All four carry the identical fingerprint — ten identical calls, one
distinct call in the whole run — because that is the serving bug's signature,
not four independent agent failures. `rotate_findable`, whose world contains the
secret and whose agent was supposed to succeed, was among them.

Those labels are asserted in `test_fidelity.py::test_captured_traces_are_scored`,
so the suite was passing by confirming that the detectors catch a bug in my own
capture rig. Re-capturing without relabelling would simply flip that test to
failing while leaving the wrong ground truth in place.

Labels come from `real_suite.classify()` — the same behavioural oracle the real
suite uses — so both corpora are judged by one rule instead of one being
hand-labelled and the other automatic. Human judgement is not being removed;
`trace_import` still argues that labelling should be a human call. What is being
removed is the ability to keep a label that the trace no longer supports.

    python evals/relabel_captures.py            # show what would change
    python evals/relabel_captures.py --write    # apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.real_suite import classify  # noqa: E402

CAPTURED = ROOT / "evals" / "captured"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="apply the new labels")
    args = ap.parse_args()

    changed = 0
    for spec_path in sorted(CAPTURED.glob("*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or "trace" not in spec:
            continue
        trace = Path(spec["trace"])
        if not trace.is_absolute():
            trace = CAPTURED / trace
        if not trace.exists():
            print(f"{spec_path.name:<28} trace missing, skipped")
            continue

        obs = classify(trace)
        if obs["should_trip"] is None:
            print(f"{spec_path.name:<28} {obs['reason']} — left alone")
            continue

        old = spec["label"].get("should_trip")
        new = obs["should_trip"]
        mark = "" if old == new else "   <-- CHANGED"
        print(f"{spec_path.name:<28} should_trip {old} -> {new}{mark}")
        print(f"    {obs['reason']} "
              f"(calls={obs['tool_calls']}, status={obs['status']})")
        if old == new:
            continue
        changed += 1

        label = {"should_trip": new, "note": obs["reason"]}
        # A documented disagreement survives relabelling. It was written down by
        # hand after investigating the trace, and the oracle cannot re-derive
        # that judgement from behaviour.
        if spec["label"].get("known_gap"):
            label["known_gap"] = True
            label["note"] = spec["label"].get("note", label["note"])
        if new:
            # Only positives carry a detector expectation and an onset; asserting
            # either on a healthy run would invent a failure that is not there.
            label["detector"] = spec["label"].get("detector", "loop")
            label["onset_index"] = obs["onset_index"]
        spec["label"] = label
        if args.write:
            spec_path.write_text(json.dumps(spec, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")

    print(f"\n{changed} label(s) would change" if not args.write
          else f"\n{changed} label(s) rewritten")
    if changed and not args.write:
        print("Re-run with --write to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
