"""Run local activation export and paired evaluation after app capture finishes.

Sequential Metal workloads; never provisions or contacts a cloud service.
Every stage retains its own logs and failure state. Comparison can still run
when calibration export fails, without calling the overall pipeline complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_turns(report, suite):
    expected = {(c["id"], n + 1): t["user"] for c in suite["conversations"] for n, t in enumerate(c["turns"])}
    actual = {}
    for turn in report["turns"]:
        key = (turn["conversationID"], turn["turn"])
        if key in actual or expected.get(key) != turn["user"]:
            raise ValueError("Duplicate, extra or mismatched reported turn")
        actual[key] = turn["user"]
    if actual != expected:
        raise ValueError("Incomplete reported conversation set")


def verify_file(record):
    path = Path(record["path"])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Frozen artifact changed: {path}")


def verify_model_bytes(models):
    reference = models["reference"]
    manifest = read(reference["manifest"]["path"])
    if (manifest.get("status"), manifest.get("model"), manifest.get("revision"), manifest.get("precision")) != (
            "validated", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "BF16"):
        raise ValueError("Wrong full-precision reference manifest")
    files = manifest["verified_files"]
    if len(files) != 18 or {p.name for p in Path(reference["artifact"]).glob("*.safetensors")} != set(files):
        raise ValueError("Missing/extra reference weight files")
    for name, info in files.items():
        if Path(name).name != name:
            raise ValueError("Unsafe weight path")
        verify_file({**info, "path": str(Path(reference["artifact"]) / name)})
    bonsai = models["bonsai"]
    manifest = read(bonsai["manifest"]["path"])
    verify_file({"path": bonsai["artifact"], "bytes": manifest["bytes"], "sha256": manifest["sha256"]})


def main(args):
    root = Path(args.run_dir).resolve()
    root.mkdir(exist_ok=False)
    state_path = root / "status.json"
    state = {"status": "waiting_for_calibration_conversations", "stages": {},
             "started_unix": time.time(), "cloud_resources": "none; local execution only"}
    write(state_path, state)
    scripts = Path(__file__).resolve().parent

    def stage(name, command, allowed=(0,)):
        state["status"] = name
        detail = {"status": "running", "started_unix": time.time(), "command": [str(x) for x in command]}
        state["stages"][name] = detail
        write(state_path, state)
        print(f"Starting {name}", flush=True)
        try:
            with (root / f"{name}.log").open("x") as log:
                result = subprocess.run([str(x) for x in command], env=env, stdout=log, stderr=subprocess.STDOUT)
            detail.update(returncode=result.returncode, status="completed" if result.returncode in allowed else "failed")
            if detail["status"] == "failed":
                raise RuntimeError(f"{name} exited {result.returncode}; see its retained log")
        except BaseException as error:
            detail.update(status="failed", error=str(error))
            raise
        finally:
            detail["elapsed_seconds"] = time.time() - detail["started_unix"]
            write(state_path, state)

    try:
        protocol = read(args.protocol)
        write(root / "protocol.json", protocol)
        env = {**os.environ, "DYLD_FRAMEWORK_PATH": str(Path(protocol["evaluator"]["path"]).parent)}
        deadline = time.monotonic() + args.wait_seconds
        capture = Path(args.capture_dir)
        while read(capture / "capture-run.json").get("status") == "running":
            if time.monotonic() >= deadline:
                raise TimeoutError("Calibration conversation capture did not finish by deadline")
            time.sleep(5)
        if read(capture / "capture-run.json").get("status") != "completed":
            raise ValueError("Calibration capture failed")
        calibration_report = read(args.calibration_report)
        write(root / "calibration-answers.json", calibration_report)
        calibration_suite = read(protocol["calibration_suite"]["path"])
        verify_turns(calibration_report, calibration_suite)
        for item in [protocol["evaluator"], protocol["merged_suite"], protocol["calibration_suite"],
                     *protocol["sources"].values(), *protocol["source_suites"],
                     *(v["manifest"] for v in protocol["models"].values())]:
            verify_file(item)
        if read(protocol["merged_suite"]["path"]) != protocol["suite"]:
            raise ValueError("Frozen suite content mismatch")
        verify_model_bytes(protocol["models"])
        heldout_titles = {s.casefold() for c in protocol["suite"]["conversations"] for t in c["turns"]
                          for s in t.get("expected_source_titles", [])}
        overlaps = [{"conversation": t["conversationID"], "turn": t["turn"], "title": title}
                    for t in calibration_report["turns"] for title in t["groundingTitles"]
                    if title.casefold() in heldout_titles]
        write(root / "split-audit.json", {"reported_grounding_title_overlaps": overlaps,
              "scope": "Actual calibration grounding titles vs held-out expected titles; full prompt/source review remains required"})
        calibration_error = None
        try:
            if overlaps:
                raise ValueError("Calibration retrieved held-out source titles; review split-audit before export")
            stage("activation_export", [sys.executable, scripts / "mac_collect.py", "--capture-dir", capture,
                  "--model-dir", protocol["models"]["reference"]["artifact"],
                  "--validation", protocol["models"]["reference"]["manifest"]["path"],
                  "--output-dir", root / "activations"])
            stage("activation_validation", [sys.executable, scripts / "validate_capture.py", root / "activations",
                  "--report", root / "activation-validation.json"])
        except Exception as error:
            calibration_error = str(error)
            state["calibration_error"] = calibration_error
            write(state_path, state)
        for role in ("bonsai", "reference"):
            model = protocol["models"][role]
            command = [protocol["evaluator"]["path"], "--probe-discuss", "--zim", protocol["sources"]["zim"]["path"],
                       "--streetzim", protocol["sources"]["streetzim"]["path"], "--suite", protocol["merged_suite"]["path"],
                       "--temperature", "0", "--top-p", "1", "--top-k", "0", "--seed", "42",
                       "--capture-model-inputs", root / f"{role}-inputs", "--report-json", root / f"{role}-answers.json",
                       "--qwen-bf16-dir" if role == "reference" else "--gguf", model["artifact"]]
            # Exit 1 means completed content assertions failed, not incomplete execution.
            stage(f"{role}_comparison", command, allowed=(0, 1))
            verify_turns(read(root / f"{role}-answers.json"), protocol["suite"])
            if read(root / f"{role}-inputs/capture-run.json").get("status") != "completed":
                raise ValueError(f"Incomplete {role} comparison input capture")
        stage("comparison_report", [sys.executable, scripts / "compare_app_reports.py",
              "--reference", root / "reference-answers.json", "--bonsai", root / "bonsai-answers.json",
              "--reference-capture", root / "reference-inputs", "--bonsai-capture", root / "bonsai-inputs",
              "--protocol", args.protocol, "--output", root / "comparison.json"])
        state["status"] = "completed" if calibration_error is None else "comparison_completed_calibration_failed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started_unix"]
        write(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("capture-dir", "calibration-report", "protocol", "run-dir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    main(parser.parse_args())
