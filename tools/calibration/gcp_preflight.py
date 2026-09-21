"""Prepare the historical one-hour L4 synthetic probe; paid launch is disabled.

The approved workflow requires a prebuilt runtime, full calibration package,
real model smoke and durable resume before paid execution can be enabled.
The retained execution code/tests document ownership and cleanup semantics.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import uuid

PROJECT = "tiltastech-zimfo"
ZONE = "us-central1-a"
IMAGE = "common-cu129-ubuntu-2204-nvidia-580-v20260909"
FILES = ("streamed_linear_probe.py", "cuda_preflight.py", "bootstrap.sh",
         "tests/test_streamed_linear_probe.py")


def create_command(name, owner):
    return ["gcloud", "compute", "instances", "create", name,
            f"--project={PROJECT}", f"--zone={ZONE}", "--machine-type=g2-standard-8",
            "--provisioning-model=SPOT", "--instance-termination-action=DELETE",
            "--max-run-duration=1h", "--no-restart-on-failure",
            "--maintenance-policy=TERMINATE", "--reservation-affinity=none",
            "--boot-disk-size=100GB", "--boot-disk-type=pd-balanced", "--boot-disk-auto-delete",
            f"--image={IMAGE}", "--image-project=deeplearning-platform-release",
            "--network=default", "--no-service-account", "--no-scopes",
            f"--labels=purpose=zimfo-cuda-preflight,run-token={owner}",
            "--quiet", "--format=json"]


def call(argv, timeout=120, check=True):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"Command failed: {shlex.join(argv)}\n{result.stderr[-3000:]}")
    return result


def describe(name):
    result = call(["gcloud", "compute", "instances", "describe", name,
                   f"--project={PROJECT}", f"--zone={ZONE}", "--format=json", "--quiet"], check=False)
    if result.returncode:
        if "was not found" in result.stderr or "not found" in result.stderr.lower():
            return None
        raise RuntimeError(result.stderr[-3000:])
    return json.loads(result.stdout)


def delete_owned(name, owner):
    instance = describe(name)
    if instance is not None:
        if instance.get("labels", {}).get("run-token") != owner:
            raise RuntimeError("Refusing cleanup: instance ownership label does not match")
        call(["gcloud", "compute", "instances", "delete", name,
              f"--project={PROJECT}", f"--zone={ZONE}", "--quiet"], timeout=180)
        if describe(name) is not None:
            raise RuntimeError("Instance still present after deletion")
    # The launch creates only this auto-delete boot disk. Verify it is gone too.
    disk = call(["gcloud", "compute", "disks", "describe", name,
                 f"--project={PROJECT}", f"--zone={ZONE}", "--format=json", "--quiet"], check=False)
    if disk.returncode == 0 or "not found" not in disk.stderr.lower():
        raise RuntimeError("Could not verify deletion of the auto-delete boot disk")
    return "instance_and_disk_deleted"


def validate_results(path):
    # Read only the expected JSON member; never extract an untrusted tar archive.
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name == "results/cuda.json"]
        if len(members) != 1 or not members[0].isfile() or members[0].size > 1024 * 1024:
            raise RuntimeError("Missing, duplicate or invalid CUDA report")
        report = json.load(archive.extractfile(members[0]))
    if (report.get("status"), report.get("device"), report.get("profile")) != ("passed", "cuda", "target"):
        raise RuntimeError("CUDA target profile did not pass")
    parities = [c["parity"] for c in report.get("checks", []) if "parity" in c]
    if len(parities) != 2 or {p.get("dtype") for p in parities} != {"torch.float32", "torch.bfloat16"}:
        raise RuntimeError("Missing CUDA precision parity checks")
    for parity in parities:
        errors = parity.get("relative_l2_errors", {})
        limit = .05 if parity["dtype"] == "torch.bfloat16" else 1e-4
        if set(errors) != {"output", "input_gradient", "allocation_gradient"} or any(
            not math.isfinite(v) or not 0 <= v <= limit for v in errors.values()
        ):
            raise RuntimeError("CUDA gradient parity failed")
    projections = [c["projection"] for c in report.get("checks", []) if "projection" in c]
    expected = {(t, w, o) for t in (2048, 4096, 7330) for w, o in ((5120, 17408), (17408, 5120))}
    if len(projections) != 6 or {(p["tokens"], p["input_width"], p["output_width"]) for p in projections} != expected:
        raise RuntimeError("Incomplete target-shape coverage")
    for projection in projections:
        if projection.get("dtype") != "torch.bfloat16" or projection.get("candidate_count") != 7:
            raise RuntimeError("Wrong precision or candidate-count coverage")
        if len(projection["steps"]) < 3:
            raise RuntimeError("Incomplete memory-lifetime checks")
        for step in projection["steps"]:
            for key in ("peak_allocated", "peak_reserved"):
                if not 0 < step.get(key, float("inf")) <= 20 * 1024**3:
                    raise RuntimeError("Invalid or excessive reported CUDA memory")
    return report


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "launch.json").exists():
        raise ValueError("Use a fresh output directory; do not overwrite a previous run")
    root = Path(__file__).resolve().parent
    token = uuid.uuid4().hex[:12]
    name = "zimfo-cuda-" + token
    hashes = {}
    with tarfile.open(directory / "harness.tgz", "w:gz") as archive:
        for relative in FILES:
            path = root / relative
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative)
    manifest = {"status": "prepared_not_launched", "instance": name, "owner": token,
                "paid_execution_enabled": False,
                "project": PROJECT, "zone": ZONE, "files_sha256": hashes,
                "create_command": create_command(name, token), "max_runtime_seconds": 3600,
                "estimate_usd_one_hour": .53, "suggested_approval_budget_usd": 2,
                "estimate_note": "Indicative Sept 19 public pricing, not a billing guarantee; no automatic retries"}
    (directory / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def execute(directory, manifest):
    require_launch_ready()
    directory = Path(directory).resolve()
    name, owner = manifest["instance"], manifest["owner"]
    if describe(name) is not None:
        raise RuntimeError("Instance already exists; refusing to reuse it")
    def persist():
        (directory / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    remote_attempted = False
    manifest["status"] = "launch_requested"
    persist()
    try:
        remote_attempted = True
        call(manifest["create_command"], timeout=180)
        manifest["status"] = "waiting_for_ssh"
        persist()
        deadline = time.monotonic() + 300
        while True:
            ready = call(["gcloud", "compute", "ssh", name, f"--project={PROJECT}",
                          f"--zone={ZONE}", "--quiet", "--ssh-flag=-oConnectTimeout=10",
                          "--command=true"], timeout=30, check=False)
            if ready.returncode == 0: break
            if time.monotonic() >= deadline: raise RuntimeError("SSH readiness timed out")
            time.sleep(5)
        call(["gcloud", "compute", "scp", str(directory / "harness.tgz"), f"{name}:harness.tgz",
              f"--project={PROJECT}", f"--zone={ZONE}", "--quiet"])
        manifest["status"] = "running"
        persist()
        # Fixed command; local user-supplied strings are not interpolated into a shell.
        command = ("mkdir -p zimfo-preflight && tar -xzf harness.tgz -C zimfo-preflight && "
                   "cd zimfo-preflight && "
                   "timeout --signal=TERM --kill-after=30s 2400s bash bootstrap.sh > console.log 2>&1; "
                   "status=$?; cd ~/zimfo-preflight; "
                   "tar -czf ~/preflight-results.tgz results console.log; exit $status")
        result = call(["gcloud", "compute", "ssh", name, f"--project={PROJECT}",
                       f"--zone={ZONE}", "--quiet", f"--command={command}"], timeout=2500, check=False)
        manifest["remote_exit_code"] = result.returncode
        manifest["status"] = "passed" if result.returncode == 0 else "failed"
        (directory / "ssh-output.log").write_text(result.stdout + result.stderr)
    except BaseException as error:
        manifest["status"] = "failed_or_interrupted"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if remote_attempted:
            try:
                fetched = call(["gcloud", "compute", "scp", f"{name}:preflight-results.tgz",
                                str(directory / "results.tgz"), f"--project={PROJECT}",
                                f"--zone={ZONE}", "--quiet"], timeout=45, check=False)
                manifest["results_retrieved"] = fetched.returncode == 0
                if fetched.returncode == 0:
                    try:
                        validate_results(directory / "results.tgz")
                        manifest["cuda_report_validated"] = True
                    except Exception as error:
                        manifest["cuda_report_validated"] = False
                        manifest["report_error"] = str(error)
            except Exception as error:
                manifest["results_retrieved"] = False
                manifest["retrieval_error"] = str(error)
            try:
                manifest["cleanup"] = delete_owned(name, owner)
            except Exception as error:
                manifest["cleanup"] = "unverified"
                manifest["cleanup_error"] = str(error)
                print("CLEANUP NOT VERIFIED. Check instance and disk; API-side one-hour deletion remains configured.")
        persist()
    if manifest["status"] != "passed" or not manifest.get("cuda_report_validated") or manifest.get("cleanup") == "unverified":
        manifest["status"] = "failed"
        persist()
        raise RuntimeError("Run, result retrieval, or cleanup did not complete; inspect launch.json")


def require_launch_ready():
    # The old one-hour projection run is not the approved full-model smoke.
    # Keep paid execution closed until the real runner, image and durable
    # resume path replace it. There is deliberately no CLI bypass flag.
    raise RuntimeError(
        "Paid launch disabled: prebuilt Linux runtime, complete calibration package, "
        "real GSQ/RCO integration and durable checkpoint/resume validation are not ready. "
        "The synthetic projection archive is preparation evidence only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true", help="Creates paid resources; use only after approval")
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(shlex.join(manifest["create_command"]), flush=True)
    if args.execute:
        execute(args.output_dir, manifest)
