"""An external corpus must never look better because bad evidence was skipped."""

import json

import pytest

from evals.score_external import load_cases, main, score_manifest


def write_trace(path, *, repeated=False):
    records = [{"kind": "meta", "original_goal": "Find account 42"}]
    for step in range(1, 5 if repeated else 2):
        records.extend([
            {"kind": "event", "type": "tool_call", "step": step,
             "tool_name": "lookup_account", "tool_args": {"id": 42}},
            {"kind": "event", "type": "tool_result", "step": step,
             "text": "not found"},
        ])
    records.append({"kind": "summary", "status": "max_turns" if repeated else "complete"})
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n",
                    encoding="utf-8")


def write_manifest(path, cases):
    path.write_text(json.dumps(cases), encoding="utf-8")


def case(case_id, trace, should_trip, note="labelled from task outcome"):
    return {"id": case_id, "trace": trace,
            "label": {"should_trip": should_trip, "note": note}}


def test_external_scorer_reports_both_classes_and_uncertainty(tmp_path, capsys):
    write_trace(tmp_path / "healthy.jsonl")
    write_trace(tmp_path / "failure.jsonl", repeated=True)
    manifest = tmp_path / "labels.json"
    write_manifest(manifest, [case("healthy", "healthy.jsonl", False),
                              case("failure", "failure.jsonl", True)])
    report = score_manifest(manifest)
    assert report["cases"] == 2
    assert report["tp"] == 1 and report["fn"] == 0
    assert report["tn"] == 1 and report["fp"] == 0
    assert report["recall"]["n"] == 1
    assert report["false_positive_rate"]["n"] == 1
    assert main([str(manifest), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 2


@pytest.mark.parametrize("bad_case,match", [
    (case("missing", "gone.jsonl", False), "missing trace"),
    (case("bad-label", "trace.jsonl", "false"), "manual boolean label"),
    (case("no-note", "trace.jsonl", False, note=""), "manual boolean label"),
])
def test_external_scorer_rejects_incomplete_evidence(tmp_path, bad_case, match):
    write_trace(tmp_path / "trace.jsonl")
    manifest = tmp_path / "labels.json"
    write_manifest(manifest, [bad_case])
    with pytest.raises(ValueError, match=match):
        load_cases(manifest)


def test_external_scorer_rejects_duplicate_ids_and_malformed_trace(tmp_path):
    write_trace(tmp_path / "trace.jsonl")
    manifest = tmp_path / "labels.json"
    one = case("same", "trace.jsonl", False)
    write_manifest(manifest, [one, one])
    with pytest.raises(ValueError, match="unique"):
        load_cases(manifest)

    (tmp_path / "trace.jsonl").write_text('{"kind":"event"}\ninvalid\n',
                                          encoding="utf-8")
    write_manifest(manifest, [one])
    with pytest.raises(ValueError, match="malformed trace"):
        load_cases(manifest)


def test_external_scorer_rejects_aborted_trace_without_summary(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"kind":"meta","original_goal":"Find account 42"}\n'
                     '{"kind":"event","type":"tool_call"}\n',
                     encoding="utf-8")
    manifest = tmp_path / "labels.json"
    write_manifest(manifest, [case("aborted", "trace.jsonl", True)])
    with pytest.raises(ValueError, match="exactly one summary"):
        load_cases(manifest)


def test_manifest_cannot_change_original_goal(tmp_path):
    write_trace(tmp_path / "trace.jsonl")
    manifest = tmp_path / "labels.json"
    spec = case("changed-goal", "trace.jsonl", False)
    spec["goal"] = "Do an unrelated task"
    write_manifest(manifest, [spec])
    with pytest.raises(ValueError, match="goal differs"):
        load_cases(manifest)
