import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_mac_pipeline as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def test_report_completeness_rejects_duplicates_and_missing():
    suite = {"conversations": [{"id": "a", "turns": [{"user": "question"}]}]}
    turn = {"conversationID": "a", "turn": 1, "user": "question"}
    runner.verify_turns({"turns": [turn]}, suite)
    for turns in ([], [turn, turn], [{**turn, "user": "different"}]):
        with pytest.raises(ValueError):
            runner.verify_turns({"turns": turns}, suite)


@pytest.mark.parametrize("export_fails", [False, True, "spawn_error"])
def test_sequential_pipeline_preserves_export_failure_and_completed_content_failures(tmp_path, monkeypatch, export_fails):
    suite = {"conversations": [{"id": "a", "turns": [{"user": "question"}]}]}
    report = {"turns": [{"conversationID": "a", "turn": 1, "user": "question", "groundingTitles": []}]}
    suite_path = write(tmp_path / "suite.json", suite)
    descriptor = {"path": suite_path}
    protocol = {"evaluator": {"path": "/not/executed/eval"}, "suite": suite,
                "calibration_suite": descriptor, "merged_suite": descriptor, "source_suites": [],
                "sources": {"zim": descriptor, "streetzim": descriptor},
                "models": {r: {"artifact": "/not/loaded/weights", "manifest": descriptor} for r in ("reference", "bonsai")}}
    args = SimpleNamespace(run_dir=str(tmp_path / "run"), protocol=write(tmp_path / "protocol.json", protocol),
                           capture_dir=str(tmp_path / "capture"), calibration_report=write(tmp_path / "cal.json", report),
                           wait_seconds=1)
    write(tmp_path / "capture/capture-run.json", {"status": "completed"})
    monkeypatch.setattr(runner, "verify_file", lambda record: None)
    monkeypatch.setattr(runner, "verify_model_bytes", lambda models: None)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if any(str(x).endswith("mac_collect.py") for x in command):
            if export_fails == "spawn_error":
                raise OSError("fixture subprocess failed to start")
            return SimpleNamespace(returncode=2 if export_fails else 0)
        if "--probe-discuss" in command:
            write(Path(command[command.index("--report-json") + 1]), report)
            write(Path(command[command.index("--capture-model-inputs") + 1]) / "capture-run.json", {"status": "completed"})
            return SimpleNamespace(returncode=1)  # completed but quality assertions failed
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    runner.main(args)
    status = runner.read(tmp_path / "run/status.json")
    assert status["status"] == ("comparison_completed_calibration_failed" if export_fails else "completed")
    assert status["stages"]["bonsai_comparison"]["returncode"] == 1
    assert len([c for c in calls if "--probe-discuss" in c]) == 2
    assert ("activation_validation" in status["stages"]) == (not export_fails)
    assert status["stages"]["activation_export"]["status"] == ("failed" if export_fails else "completed")
    assert all(not any("gcloud" in x or "docker" in x for x in c) for c in calls)


def test_malformed_protocol_records_failure(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text("not json")
    args = SimpleNamespace(run_dir=str(tmp_path / "run"), protocol=str(path))
    with pytest.raises(ValueError):
        runner.main(args)
    assert runner.read(tmp_path / "run/status.json")["status"] == "failed"


def test_model_bytes_independently_checked_even_without_collector(tmp_path):
    from prepare_comparison import artifact
    weights = tmp_path / "weights"
    weights.mkdir()
    files = {}
    for n in range(18):
        path = weights / f"model-{n}.safetensors"
        path.write_bytes(b"original fixture")
        files[path.name] = {k: v for k, v in artifact(path).items() if k != "path"}
    reference = write(tmp_path / "reference.json", {
        "status": "validated", "model": "Qwen/Qwen3.8-27B", "precision": "BF16",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "verified_files": files})
    bonsai_file = tmp_path / "bonsai.gguf"
    bonsai_file.write_bytes(b"bonsai fixture")
    bonsai = write(tmp_path / "bonsai.json", artifact(bonsai_file))
    models = {"reference": {"artifact": str(weights), "manifest": {"path": reference}},
              "bonsai": {"artifact": str(bonsai_file), "manifest": {"path": bonsai}}}
    runner.verify_model_bytes(models)
    bonsai_file.write_bytes(b"altered weights")
    with pytest.raises(ValueError, match="changed"):
        runner.verify_model_bytes(models)
