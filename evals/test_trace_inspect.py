"""A trace inspection must be useful without mistaking stale output for success."""

import json

from agentfuse.cli import main
from agentfuse.trace_inspect import inspect_trace


def test_inspect_real_quickstart_trace_without_payloads(tmp_path, capsys):
    trace = tmp_path / "run.jsonl"
    assert main(["quickstart", "--trace", str(trace), "--json"]) == 0
    capsys.readouterr()

    assert main(["inspect", str(trace), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert report["has_final_summary"] is True
    assert report["trips"] == report["recoveries"] == 1
    assert report["detectors"] == {"loop": 1}
    assert report["events"] > 0
    assert "missing@example.com" not in json.dumps(report)
    assert "original_goal" not in report


def test_inspect_does_not_treat_stale_summary_as_completed_run(tmp_path):
    trace = tmp_path / "resumed.jsonl"
    trace.write_text('\n'.join([
        json.dumps({"kind": "summary", "status": "complete", "steps": 2}),
        json.dumps({"kind": "event", "type": "tool_call", "step": 3,
                    "tool_args": {"password": "secret"}}),
    ]) + '\n', encoding="utf-8")

    report = inspect_trace(str(trace))
    assert report["status"] == "incomplete"
    assert report["has_final_summary"] is False
    assert report["steps"] == 3
    assert "secret" not in json.dumps(report)

    trace.write_text(json.dumps({"kind": "summary", "status": "complete"}) + "\n"
                     + json.dumps({"kind": "meta", "original_goal": "new run"}) + "\n",
                     encoding="utf-8")
    assert inspect_trace(str(trace))["status"] == "incomplete"


def test_inspect_rejects_malformed_and_missing_trace_cleanly(tmp_path, capsys):
    trace = tmp_path / "bad.jsonl"
    trace.write_text('{"kind": "meta"}\nnot-json\n', encoding="utf-8")
    assert main(["inspect", str(trace)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "line 2" in captured.err

    assert main(["inspect", str(tmp_path / "missing.jsonl")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read trace" in captured.err
