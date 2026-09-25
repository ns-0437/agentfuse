# Evaluate traces from outside AgentFuse's own task suite

The bundled 1,018-case synthetic benchmark and captured Qwen runs are useful
regression tests. They are not independent evidence that AgentFuse works on
other teams' agents. This procedure keeps a separate corpus and scores it
offline without a model call or paid service.

1. Collect complete JSONL traces from ordinary agent tasks with
   `MonitorConfig(jsonl_path="runs/task-001.jsonl")`. Include healthy runs,
   legitimate retries, and polling, as well as failures. Each trace needs its
   original goal, events, and one final summary record. Keep the raw traces
   private if they contain customer data or secrets.
2. Label each run from its task outcome and external evidence **before**
   looking at AgentFuse's trip decision. Write down why it was a failure or a
   healthy run. Do not copy the breaker's trip into the label.
3. Put a `labels.json` next to the traces. Paths resolve relative to the
   manifest:

   ```json
   [
     {
       "id": "task-001",
       "trace": "task-001.jsonl",
       "label": {
         "should_trip": false,
         "note": "The retry returned a result and the user task completed."
       }
     },
     {
       "id": "task-002",
       "trace": "task-002.jsonl",
       "label": {
         "should_trip": true,
         "detector": "loop",
         "note": "The same lookup returned the same empty result until the run hit its limit."
       }
     }
   ]
   ```

4. Run `python -m evals.score_external path/to/labels.json`, or add `--json`
   for a machine-readable report. The command uses the same replay runner as
   the bundled benchmark, forces offline mode, and rejects missing or malformed
   evidence. It reports confusion counts, precision, recall, false-positive
   rate, Wilson intervals, and trace lengths. A metric with no eligible cases
   is `null`, not zero.

This scores **detection on the supplied trajectories**, not whether steering
actually rescued an agent. Keep a holdout set of new runs when changing
thresholds; tuning on the same traces and then reporting their score would
recreate the self-authored-benchmark problem. A few dozen runs can expose gross
failures, but cannot establish a tiny false-positive rate with confidence.
