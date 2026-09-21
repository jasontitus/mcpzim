"""Run the reviewed solver from already prepared disks. Never install or pull."""
import hashlib
import fcntl
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

METADATA = "http://metadata.google.internal/computeMetadata/v1/"
MOUNT = Path("/mnt/zimfo-inputs")
DEVICE = "/dev/disk/by-id/google-zimfo-inputs"


def metadata(path):
    request = urllib.request.Request(METADATA + path, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read(1024 * 1024)


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=60).stdout.strip()


def verify_prepared_driver(proof):
    if command(["uname", "-r"]) != proof["kernel"]:
        raise ValueError("Prepared kernel changed")
    if command(["modinfo", "-F", "version", "nvidia"]) != proof["driver"]:
        raise ValueError("Prepared driver changed")
    if command(["modinfo", "-F", "license", "nvidia"]) != "Dual MIT/GPL":
        raise ValueError("Blackwell requires the open NVIDIA kernel module (Dual MIT/GPL)")


def serial_summary(summary, fd=1):
    """Best effort bounded diagnostics; a slow serial reader cannot stop CUDA."""
    data = (json.dumps(summary, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > 2048:
        return False
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            return os.write(fd, data) == len(data)
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
    except OSError:
        return False


def progress_summary(output, elapsed, running):
    """Read bounded local evidence and expose only known phase/numeric fields."""
    output = Path(output)
    summary = {"event": "zimfo_solver_progress", "elapsed_seconds": int(elapsed),
               "process_running": running}
    phases = {"preflight", "initialize", "smoke_gsq", "smoke_rco", "embedding", "gsq", "head", "rco"}
    try:
        with (output / "status.json").open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        status = json.loads(data) if len(data) <= 1024 * 1024 else {}
        if not isinstance(status, dict): status = {}
        if status.get("status") in {"running", "failed", "checkpointed", "ready_for_packaging", "quantization_stages_completed"}:
            summary["job_status"] = status["status"]
        progress = status.get("progress", {})
        if not isinstance(progress, dict): progress = {}
        if progress.get("phase") in phases: summary["phase"] = progress["phase"]
        for name in ("optimizer_updates", "validation_optimizer_updates", "production_optimizer_updates"):
            value = progress.get(name)
            if type(value) is int and 0 <= value < 10**12: summary[name] = value
        smoke = status.get("smoke", {})
        if isinstance(smoke, dict) and type(smoke.get("passed")) is bool:
            summary["smoke_passed"] = smoke["passed"]
    except (OSError, ValueError, TypeError):
        pass
    try:
        with (output / "solver.log").open("rb") as stream:
            stream.seek(0, os.SEEK_END); summary["log_bytes"] = stream.tell()
            stream.seek(max(0, summary["log_bytes"] - 16384))
            lines = stream.read(16384).splitlines()
        for line in reversed(lines):
            try: update = json.loads(line)
            except (ValueError, UnicodeError): continue
            if not isinstance(update, dict) or not isinstance(update.get("stage"), str) or update["stage"] not in phases:
                continue
            # Timing events also contain a stage/block, but are not optimizer
            # progress. Keep looking for the last completed update in this
            # bounded log tail rather than hiding its cursor behind a timer.
            if type(update.get("global_step")) is not int or not 0 <= update["global_step"] < 10**12:
                continue
            summary["last_update_stage"] = update["stage"]
            for name in ("global_step", "block", "epoch", "sequence"):
                value = update.get(name)
                if type(value) is int and 0 <= value < 10**12: summary[name] = value
            value = update.get("last_loss", update.get("loss"))
            if type(value) in (float, int) and abs(value) <= 1e300 and math.isfinite(value): summary["last_loss"] = value
            break
    except OSError:
        pass
    return summary


def run_solver(args, output, timeout=3000, poll_seconds=1, heartbeat_seconds=15, emit=serial_summary, log_path=None):
    """Keep complete stdout on disk, periodically expose small serial summaries.

    There is no stdout pipe to fill and no periodic network/GCS operation. The
    caller's finally block stops the named Docker container after this CLI exits.
    """
    started = time.monotonic(); last_emit = float("-inf"); previous = None
    with (Path(log_path) if log_path is not None else Path(output) / "solver.log").open("xb") as log:
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                code = child.poll(); elapsed = time.monotonic() - started
                summary = progress_summary(output, elapsed, code is None)
                # Log bytes/loss can change every step; limit serial writes to
                # transitions in job phase/status plus a15-second heartbeat.
                transition = (summary.get("phase"), summary.get("job_status"), code)
                if transition != previous or elapsed - last_emit >= heartbeat_seconds:
                    emit(summary); previous = transition; last_emit = elapsed
                if code is not None:
                    return subprocess.CompletedProcess(args, code)
                if elapsed >= timeout:
                    raise subprocess.TimeoutExpired(args, timeout)
                time.sleep(min(poll_seconds, timeout - elapsed))
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
            log.flush(); os.fsync(log.fileno())


def solver_command(config):
    receipt = config["ready"]
    return ["docker", "run", "--name", config["run_id"], "--rm", "--pull=never", "--gpus=all", "--network=host",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--shm-size=8g",
            "--mount", f"type=bind,src={MOUNT},dst={MOUNT}",
            "--mount", f"type=bind,src={MOUNT / 'prepared'},dst={MOUNT / 'prepared'},readonly",
            receipt["production_image"], "python", "-m", "solver.job",
            "--inputs", str(MOUNT / "prepared"),
            "--input-commit-sha256", receipt["input_commit"]["sha256"],
            "--output", str(MOUNT / "jobs" / config["run_id"]),
            "--bucket", receipt["bucket"], "--prefix", "runs/" + config["run_id"],
            "--runtime-sha256", receipt["production_image"].split("@sha256:")[1],
            "--target-bytes", "4000000000", "--deadline-seconds", "2700",
            "--checkpoint-seconds", "120"]


def publish(config, report):
    # One terminal object; model/checkpoint objects are handled by the solver.
    token = json.loads(metadata("instance/service-accounts/default/token"))["access_token"]
    query = urllib.parse.urlencode({"uploadType": "media", "name": f"runs/{config['run_id']}/vm-result.json",
                                    "ifGenerationMatch": 0})
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{config['ready']['bucket']}/o?{query}"
    request = urllib.request.Request(url, data=json.dumps(report, sort_keys=True).encode(), method="POST",
                                    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read(1024 * 1024)


def validate_solver_status(status, expected_sequence_tokens=None, *, scientific_only=False):
    """Fail closed: successful process exit is not scientific or recovery proof."""
    if not isinstance(status, dict) or status.get("schema_version") != 1 or status.get("status") not in ("checkpointed", "ready_for_packaging"):
        raise ValueError("Missing or unsuccessful solver report")
    smoke = status.get("smoke", {})
    if (smoke.get("passed") is not True or smoke.get("model") != "Qwen/Qwen3.8-27B"
            or smoke.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            or not isinstance(smoke.get("sequence_tokens"), int) or smoke["sequence_tokens"] <= 0):
        raise ValueError("Missing original-model smoke identity")
    if expected_sequence_tokens is not None and smoke["sequence_tokens"] != expected_sequence_tokens:
        raise ValueError("Smoke did not cover the largest captured invocation")
    cuda = smoke.get("cuda", {})
    if (cuda.get("available") is not True or not cuda.get("device") or not cuda.get("capability")
            or any(not isinstance(cuda.get(key), (int, float)) or not math.isfinite(cuda[key]) or cuda[key] <= 0
                   for key in ("total_memory_bytes", "peak_allocated_bytes", "peak_reserved_bytes"))
            or cuda["peak_allocated_bytes"] > cuda["total_memory_bytes"]
            or cuda["peak_reserved_bytes"] > cuda["total_memory_bytes"]):
        raise ValueError("CUDA memory evidence incomplete")
    for name in ("linear_attention_gsq", "full_attention_gsq", "full_model_rco"):
        check = smoke.get(name, {})
        norm = check.get("gradient_norm")
        if check.get("passed") is not True or not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm <= 0:
            raise ValueError("Missing real-model backward/update evidence: " + name)
        if name != "full_model_rco" and (not isinstance(check.get("loss"), (int, float)) or not math.isfinite(check["loss"])):
            raise ValueError("Missing finite GSQ objective")
    if smoke["full_model_rco"].get("full_vocabulary") is not True:
        raise ValueError("RCO smoke did not use full vocabulary")
    resume = smoke.get("checkpoint_resume", {})
    if resume.get("passed") is not True or resume.get("next_update_matches") is not True or not resume.get("snapshot"):
        raise ValueError("Real CUDA checkpoint resume unproven")
    if scientific_only:
        if not valid_commit(resume.get("commit", {})):
            raise ValueError("No valid generation-pinned smoke commit")
        return status
    progress = status.get("progress", {})
    if not isinstance(progress.get("optimizer_updates"), int) or progress["optimizer_updates"] <= 0 or not progress.get("phase"):
        raise ValueError("No useful optimizer progress")
    durable = progress.get("durable_checkpoint", {})
    if not durable.get("snapshot"):
        raise ValueError("No durable progress snapshot")
    commits = [resume.get("commit", {}), durable.get("receipt", {}).get("commit", {})]
    for commit in commits:
        if (not commit.get("object") or not str(commit.get("generation", "")).isdigit()
                or int(commit["generation"]) <= 0 or not isinstance(commit.get("bytes"), int) or commit["bytes"] <= 0
                or len(commit.get("sha256", "")) != 64):
            raise ValueError("No valid generation-pinned checkpoint commit")
    return status


def verify_remote_commits(config, status):
    token = json.loads(metadata("instance/service-accounts/default/token"))["access_token"]
    commits = [status["smoke"]["checkpoint_resume"]["commit"],
               status["progress"]["durable_checkpoint"]["receipt"]["commit"]]
    seen = set()
    for commit in commits:
        identity = (commit["object"], str(commit["generation"]))
        if identity in seen:
            continue
        seen.add(identity)
        ancestors = [config["continuation"][key] for key in ("source_commit", "smoke_commit")] if config.get("mode") == "continuation" else []
        if (not valid_commit(commit) or (not commit["object"].startswith("runs/" + config["run_id"] + "/") and commit not in ancestors)) or commit["bytes"] > 4*1024*1024:
            raise ValueError("Checkpoint commit outside this run or exceeds bounded manifest size")
        query = urllib.parse.urlencode({"alt": "media", "generation": commit["generation"],
                                        "ifGenerationMatch": commit["generation"]})
        key = urllib.parse.quote(commit["object"], safe="")
        request = urllib.request.Request(f"https://storage.googleapis.com/storage/v1/b/{config['ready']['bucket']}/o/{key}?{query}",
                                         headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(commit["bytes"] + 1)
        if len(data) != commit["bytes"] or hashlib.sha256(data).hexdigest() != commit["sha256"]:
            raise ValueError("Durable checkpoint commit could not be independently verified")


def valid_commit(commit):
    return (isinstance(commit, dict) and isinstance(commit.get("object"), str)
            and re.fullmatch(r"runs/[a-zA-Z0-9/_-]+(?:\.json)", commit["object"]) is not None
            and str(commit.get("generation", "")).isdigit() and int(commit["generation"]) > 0
            and type(commit.get("bytes")) is int and 0 < commit["bytes"] <= 4*1024*1024
            and isinstance(commit.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", commit["sha256"]) is not None)


def validate_continuation_plan(plan, source, ready):
    if (plan.get("schema_version") != 1 or plan.get("status") != "continuation_preparation_only"
            or plan.get("runtime_image") != ready["production_image"]
            or plan.get("runtime_sha256") != ready["production_image"].split("@sha256:")[1]
            or source.get("runtime_sha256") != plan["runtime_sha256"]
            or source.get("input_commit_sha256") != ready["input_commit"]["sha256"]
            or plan.get("selected_checkpoint") != source.get("progress", {}).get("durable_checkpoint", {}).get("receipt", {}).get("commit")
            or type(plan.get("recovery_required_free_bytes")) is not int
            or plan["recovery_required_free_bytes"] < 10*1024**3
            or not isinstance(plan.get("files"), dict)):
        raise ValueError("Continuation plan lost input/runtime/source binding")
    # Reuse scientific validation while recognizing that source status may be
    # a failed supervisor with a valid committed production checkpoint.
    proof = dict(source); proof["status"] = "checkpointed"
    validate_solver_status(proof, scientific_only=True)
    if not valid_commit(plan["selected_checkpoint"]) or not valid_commit(source["smoke"]["checkpoint_resume"]["commit"]):
        raise ValueError("Invalid exact ancestor checkpoint")


def validate_staging_receipt(config):
    c = config["continuation"]; proof = c.get("staging_receipt")
    if (not isinstance(proof, dict) or proof.get("status") != "continuation_staged"
            or proof.get("data_disk_id") != config["ready"]["data_disk"]["id"]
            or proof.get("plan_path") != c["plan_path"] or proof.get("plan_sha256") != c["plan_sha256"]
            or proof.get("files") != c["files"]
            or type(proof.get("free_bytes")) is not int or proof["free_bytes"] < c["recovery_required_free_bytes"]):
        raise ValueError("CPU-staged continuation receipt with sufficient disk headroom required before GPU launch")
    if c.get("checkpoint_mode") == "local-spool" and (
            proof.get("checkpoint_mode") != "local-spool" or proof.get("publisher_sha256") != c["publisher_sha256"]
            or proof.get("spool_min_free_bytes") != c["spool_min_free_bytes"]
            or proof["free_bytes"] < c["spool_min_free_bytes"]):
        raise ValueError("CPU preparation has not verified publisher/spool capacity")


def continuation_deadlines(absolute_deadline, now=None):
    now = time.time() if now is None else now
    hard = datetime.fromisoformat(absolute_deadline).timestamp()
    # Provider deletion remains final backstop. Leave independent shutdown,
    # diagnostics and owned-container cleanup headroom outside controller cap.
    remaining = int(hard-now)
    result = {"guest_seconds":remaining-180, "bootstrap_seconds":remaining-360,
              "controller_seconds":remaining-600}
    if result["controller_seconds"] < 1200:
        raise ValueError("Insufficient approved runtime remains for safe continuation")
    return result


def validate_continuation_status(config, status):
    c = config["continuation"]
    if (not isinstance(status, dict) or status.get("schema_version") != 1
            or status.get("status") not in ("checkpointed", "quantization_stages_completed")
            or status.get("runtime_image") != c["runtime_image"]
            or status.get("runtime_sha256") != c["runtime_image"].split("@sha256:")[1]
            or status.get("input_commit_sha256") != config["ready"]["input_commit"]["sha256"]
            or status.get("packaged_model_validated") is not False
            or status.get("progress", {}).get("optimizer_update_totals_known") is not False
            or status.get("smoke", {}).get("checkpoint_resume", {}).get("commit") != c["smoke_commit"]
            or hashlib.sha256(json.dumps(status.get("smoke"),sort_keys=True,separators=(",", ":")).encode()).hexdigest() != c["smoke_sha256"]):
        raise ValueError("Invalid continuation outcome or identity")
    durable = status.get("progress", {}).get("durable_checkpoint", {})
    commit = durable.get("receipt", {}).get("commit", {})
    if not durable.get("snapshot") or not valid_commit(commit):
        raise ValueError("Continuation lacks generation-pinned durable checkpoint")
    if commit != c["source_commit"] and not commit["object"].startswith("runs/"+config["run_id"]+"/"):
        raise ValueError("Continuation checkpoint outside exact ancestry/current namespace")
    if status["status"] == "quantization_stages_completed" and (status.get("progress", {}).get("phase") != "rco" or not isinstance(status.get("allocation"), dict)):
        raise ValueError("Continuation completion lacks RCO allocation")
    return status


def verify_staged_continuation(config):
    c = config["continuation"]; validate_staging_receipt(config)
    directory = Path(c["plan_path"])
    expected_parent = MOUNT/"continuation-plans"
    if directory.is_symlink() or directory.resolve().parent != expected_parent.resolve():
        raise ValueError("Unexpected staged continuation directory")
    if "continuation-plan.json" in c["files"]: raise ValueError("Plan cannot override its own binding")
    descriptors = {"continuation-plan.json": {"sha256":c["plan_sha256"]}, **c["files"]}
    for name, item in descriptors.items():
        path = directory/name
        if path.is_symlink() or directory.resolve() not in path.resolve().parents or path.stat().st_size > 2*1024*1024:
            raise ValueError("Unsafe staged continuation evidence")
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"] or ("bytes" in item and len(data) != item["bytes"]):
            raise ValueError("CPU-staged continuation evidence changed")
    plan=json.loads((directory/"continuation-plan.json").read_text())
    source=json.loads((directory/"reconciled-status.json").read_text())
    validate_continuation_plan(plan,source,config["ready"])
    if plan["selected_checkpoint"] != c["source_commit"] or source["smoke"]["checkpoint_resume"]["commit"] != c["smoke_commit"]:
        raise ValueError("Changed continuation ancestry")
    if shutil.disk_usage(MOUNT).free < max(c["recovery_required_free_bytes"],c.get("spool_min_free_bytes",0)):
        raise ValueError("Insufficient live recovery disk headroom")
    controller=Path("/opt/zimfo-gpu/continuation.py")
    if controller.is_symlink() or hashlib.sha256(controller.read_bytes()).hexdigest() != c["controller_sha256"]:
        raise ValueError("Continuation controller changed")
    if c.get("checkpoint_mode") == "local-spool":
        publisher=Path("/opt/zimfo-gpu/checkpoint_bridge.py")
        if publisher.is_symlink() or hashlib.sha256(publisher.read_bytes()).hexdigest() != c["publisher_sha256"]:
            raise ValueError("Checkpoint publisher changed")


def cleanup_owned_container(config, ledger):
    try:
        if ledger.is_symlink() or ledger.stat().st_size > 4096: raise ValueError("Unsafe container ledger")
        item=json.loads(ledger.read_text())
        if item.get("owner_run_id") != config["run_id"] or not re.fullmatch(r"continue-[a-f0-9]{12}-(prepare|embedding|gsq|head|rco)",item.get("container_name", "")):
            raise ValueError("Container ledger ownership mismatch")
        names=[item["container_name"]]
        publisher=item.get("publisher_container_name")
        if publisher is not None:
            if publisher != item["container_name"]+"-publisher" or not re.fullmatch(r"continue-[a-f0-9]{12}-(embedding|gsq|head|rco)-publisher",publisher):
                raise ValueError("Publisher ledger ownership mismatch")
            names.append(publisher)
        subprocess.run(["docker","stop","--time=90",*names],capture_output=True,timeout=100,check=False)
    except FileNotFoundError: pass
    except (ValueError, OSError, subprocess.TimeoutExpired):
        serial_summary({"event":"zimfo_container_cleanup","status":"owned_stop_failed"})


def execute_continuation(config, gpu):
    verify_staged_continuation(config)
    output = MOUNT/"jobs"/config["run_id"]
    if output.exists(): raise FileExistsError("Continuation output must be new")
    supervision=MOUNT/"supervision"/config["run_id"]
    supervision.mkdir(parents=True,exist_ok=False)
    ledger=supervision/"active-container.json"
    deadline=metadata("instance/attributes/zimfo-absolute-deadline").decode().strip()
    if deadline != config.get("absolute_deadline"):
        raise ValueError("Absolute deadline differs from launch configuration")
    limits=continuation_deadlines(deadline)
    args=["python3","/opt/zimfo-gpu/continuation.py","execute","--plan-dir",config["continuation"]["plan_path"],
          "--output",str(output),"--workspace",str(MOUNT),"--inputs",str(MOUNT/"prepared"),
          "--deadline-seconds",str(limits["controller_seconds"]),
          "--checkpoint-reserve-seconds",str(config["continuation"]["checkpoint_reserve_seconds"]),
          "--write-prefix","runs/"+config["run_id"],"--owned-container-ledger",str(ledger),"--owner-run-id",config["run_id"]]
    c=config["continuation"]
    if c.get("checkpoint_mode") == "local-spool":
        args += ["--checkpoint-mode","local-spool","--publisher-script","/opt/zimfo-gpu/checkpoint_bridge.py",
                 "--publisher-sha256",c["publisher_sha256"],"--spool-min-free-bytes",str(c["spool_min_free_bytes"])]
    try:
        result=run_solver(args,output,timeout=limits["controller_seconds"]+120,log_path=supervision/"supervisor.log")
    finally: cleanup_owned_container(config,ledger)
    if result.returncode: raise RuntimeError("Continuation controller failed; inspect retained evidence")
    status=json.loads((output/"status.json").read_text())
    validate_continuation_status(config,status); verify_remote_commits(config,status)
    return {"status":"validated_continuation_run","solver_returncode":result.returncode,"solver_status":status,"gpu":gpu}


def execute(config):
    if metadata("instance/machine-type").decode().split("/")[-1] != "g4-standard-48":
        raise ValueError("Wrong GPU machine type")
    receipt = config["ready"]
    verify_prepared_driver(receipt["runtime_proof"])
    MOUNT.mkdir(exist_ok=True)
    if not MOUNT.is_mount():
        if command(["blkid", "-s", "LABEL", "-o", "value", DEVICE]) != "zimfo-inputs":
            raise ValueError("Prepared disk label mismatch; refusing to format")
        command(["mount", "-o", "noatime", DEVICE, str(MOUNT)])
    validation = json.loads((MOUNT / "prepared/restore-validation.json").read_text())
    if validation != receipt["restore_validation"] or validation["input_commit_sha256"] != receipt["input_commit"]["sha256"]:
        raise ValueError("Prepared inputs differ from reviewed receipt")
    inspection = json.loads(command(["docker", "image", "inspect", receipt["production_image"]]))[0]
    if receipt["production_image"] not in inspection["RepoDigests"] or inspection["Architecture"] != "amd64":
        raise ValueError("Prepared container missing or different")
    gpu = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    lines = gpu.splitlines()
    if len(lines) != 1 or "RTX PRO 6000" not in lines[0] or float(lines[0].split(",")[1]) < 90000:
        raise ValueError("Expected one 96GB RTX PRO 6000")
    if config.get("mode") == "continuation": return execute_continuation(config,gpu)
    output = MOUNT / "jobs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = run_solver(solver_command(config), output)
    finally:
        # A timed-out Docker CLI can leave its container alive. Address only this
        # unique run; give the solver a bounded opportunity to commit its state.
        try:
            subprocess.run(["docker", "stop", "--time=45", config["run_id"]],
                           capture_output=True, timeout=55, check=False)
        except (subprocess.TimeoutExpired, OSError):
            # Keep the original solver exception. The independent VM deadline
            # and startup shutdown trap remain the final resource backstop.
            serial_summary({"event": "zimfo_container_cleanup", "status": "stop_command_failed"})
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    if result.returncode != 0:
        raise RuntimeError(f"Solver failed with exit {result.returncode}; inspect retained solver.log/status.json")
    manifest = json.loads((MOUNT / "prepared/calibration/manifest.json").read_text())
    validate_solver_status(status, max(sequence["tokens"] for sequence in manifest["sequences"]))
    verify_remote_commits(config, status)
    return {"status": "validated_checkpointed_run", "solver_returncode": result.returncode,
            "solver_status": status, "gpu": gpu}


def main():
    config = json.loads(Path("/opt/zimfo-gpu/config.json").read_text())
    expected = config.pop("config_sha256")
    if hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != expected:
        raise ValueError("GPU config changed")
    config["config_sha256"] = expected
    started = time.monotonic()
    try:
        report = execute(config)
    except BaseException as error:
        report = {"status": "failed", "error": str(error)}
    report.update(run_id=config["run_id"], config_sha256=expected, elapsed_seconds=time.monotonic()-started)
    path = (MOUNT if MOUNT.is_mount() else Path("/opt/zimfo-gpu")) / (config["run_id"] + "-result.json")
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    command(["sync"])
    print(json.dumps(report), flush=True)
    publish(config, report)


if __name__ == "__main__":
    main()
