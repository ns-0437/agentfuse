"""Latency budget — the cost users feel, which tokens alone don't capture.

The benchmark measures token economics but has never measured *time*. That is a
real omission: a supervisor sitting in the hot path of every agent step adds
wall-clock latency to every action, and if that overhead is large enough people
will switch it off no matter how good the F1 is.

What is measured here:

  * per-event overhead of ``monitor.observe()`` — the number that actually lands
    in a user's request path,
  * the same with each detector ablated, so an expensive detector is visible,
  * the drift detector in isolation, since it is the only one doing real text
    work (and would do network I/O in embeddings mode).

Reported as median and p95. Means hide exactly the tail that users notice.

Caveat: this measures the *offline lexical* drift path. With embeddings enabled
the drift probe becomes a network call, which will dominate everything here —
that budget must be measured separately once a live backend is wired in.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from agentfuse import (
    AgentEvent, EventType, CircuitBreakerMonitor, MonitorConfig, Tracer,
)
from agentfuse.detectors import LoopDetector, DriftDetector, NoProgressDetector, SpendDetector

from agentfuse.recovery import RecoveryEngine

from .generators import generate_suite
from .runner import DEFAULT_CONFIG, _events_for_step


class _SilentTracer(Tracer):
    """Tracer that records nothing and, crucially, prints nothing.

    The base Tracer prints trip/recovery panels unconditionally, which would put
    console I/O inside the timed region and make every measurement meaningless.
    """

    def __init__(self) -> None:
        super().__init__(jsonl_path=None, echo=False)

    def trip(self, event, trip) -> None:  # type: ignore[override]
        self.trips += 1

    def recovery(self, path) -> None:  # type: ignore[override]
        self.recoveries += 1

    def summary(self, totals) -> None:  # type: ignore[override]
        pass


@dataclass
class LatencyStats:
    label: str
    n: int
    median_us: float
    p95_us: float
    max_us: float
    total_ms: float

    def render(self) -> str:
        return (f"  {self.label:<26}{self.median_us:>9.1f}{self.p95_us:>10.1f}"
                f"{self.max_us:>10.1f}{self.total_ms:>11.1f}")

    def to_dict(self) -> dict:
        return {"label": self.label, "n": self.n,
                "median_us": round(self.median_us, 2), "p95_us": round(self.p95_us, 2),
                "max_us": round(self.max_us, 2), "total_ms": round(self.total_ms, 2)}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[k]


def measure(label: str, scenarios, disabled: set[str] | None = None,
            include_trips: bool = False) -> LatencyStats:
    """Time ``observe()`` across a set of scenarios.

    By default this reports **steady-state** cost only: events that did not trip
    the breaker. That is the number in every request path, and it is what a user
    pays on the overwhelming majority of steps.

    Trip-handling is a different animal — it runs the whole recovery engine — so
    mixing the two produces a meaningless average dominated by a handful of
    outliers. Pass ``include_trips=True`` to measure that path separately.
    """
    disabled = disabled or set()
    samples: list[float] = []

    for sc in scenarios:
        cfg = {**DEFAULT_CONFIG, **sc.config}
        detectors = []
        if "loop" not in disabled:
            detectors.append(LoopDetector(threshold=cfg["loop_threshold"]))
        if "drift" not in disabled:
            detectors.append(DriftDetector(original_goal=sc.goal,
                                           threshold=cfg["drift_threshold"]))
        if "progress" not in disabled:
            detectors.append(NoProgressDetector(patience=cfg["stall_patience"]))
        if "spend" not in disabled:
            detectors.append(SpendDetector(max_tokens=cfg["max_tokens"],
                                           burst_window=cfg["burst_window"],
                                           burst_tokens=cfg["burst_tokens"]))
        tracer = _SilentTracer()
        mon = CircuitBreakerMonitor(
            config=MonitorConfig(original_goal=sc.goal, echo=False),
            detectors=detectors,
            recovery=RecoveryEngine(backend="mock"),
            tracer=tracer,
        )
        for idx, step in enumerate(sc.steps):
            for ev in _events_for_step(step, idx + 1):
                before = tracer.trips
                t0 = time.perf_counter()
                mon.observe(ev)
                dt = (time.perf_counter() - t0) * 1e6  # microseconds
                tripped = tracer.trips > before
                if tripped == include_trips:
                    samples.append(dt)

    return LatencyStats(
        label=label,
        n=len(samples),
        median_us=statistics.median(samples) if samples else 0.0,
        p95_us=_percentile(samples, 0.95),
        max_us=max(samples) if samples else 0.0,
        total_ms=sum(samples) / 1000.0,
    )


def run_latency_suite(n_per_generator: int = 8, seed: int = 20260812) -> list[LatencyStats]:
    """Full latency profile: everything on, then one detector at a time removed."""
    scenarios = generate_suite(n_per_generator=n_per_generator, seed=seed)
    measure("warmup", scenarios[:20])  # prime imports/caches before timing
    out = [measure("all detectors", scenarios)]
    for name in ("loop", "drift", "progress", "spend"):
        out.append(measure(f"without {name}", scenarios, disabled={name}))
    out.append(measure("monitor only (no detectors)", scenarios,
                       disabled={"loop", "drift", "progress", "spend"}))
    # The rare path, measured separately: a trip runs the whole recovery engine.
    out.append(measure("TRIP handling (rare path)", scenarios, include_trips=True))
    return out


def render(stats: list[LatencyStats]) -> str:
    lines = ["", "LATENCY  (supervision overhead per observed event)", "-" * 78,
             f"  {'variant':<26}{'median us':>9}{'p95 us':>10}{'max us':>10}{'total ms':>11}"]
    lines += [s.render() for s in stats]
    base = next((s for s in stats if s.label == "all detectors"), None)
    bare = next((s for s in stats if s.label.startswith("monitor only")), None)
    if base and bare:
        overhead = base.median_us - bare.median_us
        lines.append("")
        if abs(overhead) < 5.0:
            # Do not dress up noise as a measurement: if the delta is smaller
            # than run-to-run jitter, the honest statement is that detector cost
            # is unmeasurably small next to the monitor's own bookkeeping.
            lines.append(f"  detector cost is below measurement noise "
                         f"(all={base.median_us:.1f}us vs bare={bare.median_us:.1f}us); "
                         f"the ~{base.median_us:.0f}us is monitor bookkeeping, not detection")
        else:
            lines.append(f"  detector cost: {overhead:.1f} us median per event "
                         f"({base.median_us:.1f} total vs {bare.median_us:.1f} bare)")
        lines += [
            f"  VERDICT: ~{base.median_us:.0f}us/event is negligible against an LLM call "
            f"(1e5-1e6 us).",
            "  Supervision is not a latency problem in offline mode.",
            "  NOTE: this is the lexical drift path. With embeddings the drift probe",
            "  becomes a network call and will dominate everything measured here.",
        ]
    return "\n".join(lines)
