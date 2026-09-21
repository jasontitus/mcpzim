"""One explicitly bounded Spot launch using prepared boot and data disks."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid

from cloud_prep.prepare import PROJECT, ZONE, SA, TAG, REGISTRY, ensure_ingress_denied, run

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def validate_ready(ready):
    if (ready.get("status") != "prepared_disks_detached" or ready.get("project") != PROJECT
            or ready.get("zone") != ZONE or ready.get("gpu_smoke_passed") is not False
            or ready.get("restore_validation", {}).get("status") != "validated"
            or ready["restore_validation"].get("input_commit_sha256") != ready["input_commit"]["sha256"]):
        raise ValueError("A complete, matching CPU preparation receipt is required")
    if not re.fullmatch(re.escape(REGISTRY) + r"[a-z0-9][a-z0-9_./-]*@sha256:[0-9a-f]{64}", ready["production_image"]):
        raise ValueError("Production image must use immutable private digest")
    for key in ("boot_disk", "data_disk"):
        disk = ready[key]
        if (not re.fullmatch(r"zimfo-prep-[a-f0-9]{12}-(boot|inputs)", disk["name"])
                or not str(disk["id"]).isdigit()
                or not disk["self_link"].endswith(f"/projects/{PROJECT}/zones/{ZONE}/disks/{disk['name']}")):
            raise ValueError("Unexpected prepared disk identity")


def prepare(receipt_path, output, *, max_seconds=3600, continuation_plan=None,
            staged_plan_path=None, continuation_staging=None, checkpoint_mode="gcs", spool_min_free_bytes=None):
    ready = json.loads(Path(receipt_path).read_text())
    validate_ready(ready)
    if type(max_seconds) is not int or not 1800 <= max_seconds <= 86400:
        raise ValueError("Explicit runtime cap must be 1800..86400 seconds")
    if checkpoint_mode not in ("gcs", "local-spool"):
        raise ValueError("Unknown checkpoint mode")
    if checkpoint_mode == "local-spool" and (continuation_plan is None or type(spool_min_free_bytes) is not int or spool_min_free_bytes < 10*1024**3):
        raise ValueError("Local spool needs explicit continuation and disk admission bound")
    continuation = None
    if continuation_plan is not None:
        from cloud_gpu.bootstrap import validate_continuation_plan
        plan_path = Path(continuation_plan).resolve()
        controller = HERE.parent / "continuation.py"
        plan = json.loads(plan_path.read_text())
        source = json.loads((plan_path.parent / "reconciled-status.json").read_text())
        validate_continuation_plan(plan, source, ready)
        if not staged_plan_path or not re.fullmatch(r"/mnt/zimfo-inputs/continuation-plans/[a-zA-Z0-9_-]+", staged_plan_path):
            raise ValueError("Explicit CPU-staged continuation plan directory required")
        continuation = {"plan_path": staged_plan_path, "plan_sha256": sha(plan_path),
            "controller_sha256": sha(controller), "files": plan["files"],
            "runtime_image": plan["runtime_image"],
            "source_commit": plan["selected_checkpoint"],
            "smoke_commit": source["smoke"]["checkpoint_resume"]["commit"],
            "smoke_sha256": hashlib.sha256(json.dumps(source["smoke"],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
            "recovery_required_free_bytes": plan["recovery_required_free_bytes"],
            "checkpoint_reserve_seconds": 900,
            "staging_receipt": json.loads(Path(continuation_staging).read_text()) if continuation_staging else None}
        continuation["checkpoint_mode"] = checkpoint_mode
        if checkpoint_mode == "local-spool":
            continuation.update(publisher_sha256=sha(HERE.parent / "checkpoint_bridge.py"),
                                spool_min_free_bytes=spool_min_free_bytes)
        for name, descriptor in plan["files"].items():
            evidence = plan_path.parent / name
            if name == "continuation-plan.json" or evidence.is_symlink() or plan_path.parent not in evidence.resolve().parents or sha(evidence) != descriptor["sha256"] or evidence.stat().st_size != descriptor["bytes"]:
                raise ValueError("Changed continuation evidence")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {"run_id": "zimfo-gpu-" + uuid.uuid4().hex[:12], "ready": ready,
              "max_seconds": max_seconds, "mode": "continuation" if continuation else "initial",
              "ready_sha256": sha(receipt_path), "bootstrap_sha256": sha(HERE / "bootstrap.py"),
              "startup_sha256": sha(HERE / "startup.sh"),
              "shutdown_sha256": sha(HERE / "shutdown-diagnostics.sh")}
    if continuation:
        config["continuation"] = continuation
        (output / "continuation.py").write_bytes(controller.read_bytes())
        (output / "continuation-plan.json").write_bytes(plan_path.read_bytes())
        if checkpoint_mode == "local-spool":
            (output / "checkpoint_bridge.py").write_bytes((HERE.parent / "checkpoint_bridge.py").read_bytes())
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    save(output / "config.json", config)
    for name in ("startup.sh", "bootstrap.py", "shutdown-diagnostics.sh"):
        (output / name).write_bytes((HERE / name).read_bytes())
    save(output / "plan.json", {"status": "prepared_not_launched", "config": config,
                               "max_seconds": max_seconds, "spot": True, "machine_type": "g4-standard-48",
                               "gpu_compute_usd_per_hour_estimate": 1.7429,
                               "storage_and_network_extra": True, "automatic_retries": 0})
    return config


def create_command(config, directory, deadline):
    ready = config["ready"]
    extra = ",zimfo-continuation-controller=" + str(directory / "continuation.py") if config.get("mode") == "continuation" else ""
    if config.get("continuation", {}).get("checkpoint_mode") == "local-spool":
        extra += ",zimfo-checkpoint-publisher=" + str(directory / "checkpoint_bridge.py")
    return ["gcloud", "compute", "instances", "create", config["run_id"], f"--project={PROJECT}", f"--zone={ZONE}",
            "--machine-type=g4-standard-48", "--provisioning-model=SPOT", "--maintenance-policy=TERMINATE",
            "--instance-termination-action=DELETE", "--no-restart-on-failure", f"--termination-time={deadline}",
            f"--disk=name={ready['boot_disk']['name']},boot=yes,auto-delete=no,device-name=zimfo-boot",
            f"--disk=name={ready['data_disk']['name']},auto-delete=no,device-name=zimfo-inputs,mode=rw",
            f"--service-account={SA}", "--scopes=https://www.googleapis.com/auth/cloud-platform",
            "--network-interface=network=default,subnet=default,nic-type=GVNIC", f"--tags={TAG}",
            f"--labels=zimfo-purpose=quantization,zimfo-run={config['run_id']}",
            "--metadata=enable-oslogin=true,block-project-ssh-keys=true,serial-port-enable=false",
            f"--metadata-from-file=startup-script={directory / 'startup.sh'},zimfo-gpu-bootstrap={directory / 'bootstrap.py'},zimfo-gpu-config={directory / 'config.json'},shutdown-script={directory / 'shutdown-diagnostics.sh'}{extra}",
            "--format=json"]


def execute(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if plan["status"] != "prepared_not_launched" or config != plan["config"]:
        raise ValueError("Already attempted or changed plan; inspect same named VM, never blindly retry")
    validate_ready(config["ready"])
    unhashed = {k: v for k, v in config.items() if k != "config_sha256"}
    if hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != config["config_sha256"]:
        raise ValueError("Launch configuration changed")
    if config.get("mode") == "continuation":
        from cloud_gpu.bootstrap import validate_staging_receipt
        continuation = config["continuation"]
        if sha(directory / "continuation.py") != continuation["controller_sha256"] or sha(directory / "continuation-plan.json") != continuation["plan_sha256"]:
            raise ValueError("Continuation controller or plan changed")
        if continuation.get("checkpoint_mode") == "local-spool" and sha(directory / "checkpoint_bridge.py") != continuation["publisher_sha256"]:
            raise ValueError("Checkpoint publisher changed")
        validate_staging_receipt(config)
    for name, key in (("startup.sh", "startup_sha256"), ("bootstrap.py", "bootstrap_sha256"), ("shutdown-diagnostics.sh", "shutdown_sha256")):
        if sha(directory / name) != config[key]:
            raise ValueError("Startup changed after plan preparation")
    for key in ("boot_disk", "data_disk"):
        expected = config["ready"][key]
        disk = run(["gcloud", "compute", "disks", "describe", expected["name"], f"--project={PROJECT}", f"--zone={ZONE}", "--format=json"])
        if (str(disk["id"]) != expected["id"] or disk.get("users") or disk["status"] != "READY"
                or disk.get("labels", {}).get("zimfo-run") != config["ready"]["run_id"]):
            raise ValueError("Prepared disk replaced, in use, or ownership changed")
    instances = run(["gcloud", "compute", "instances", "list", f"--project={PROJECT}", "--format=json"])
    if any(i.get("labels", {}).get("zimfo-purpose") == "quantization" for i in instances):
        raise ValueError("Existing quantization VM needs inspection before any launch")
    ensure_ingress_denied()
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=config.get("max_seconds", 3600))).isoformat(timespec="seconds")
    # The provider deadline is written once, before creation, and is shared by
    # guest supervision. It is separately bound into the launch plan.
    config["absolute_deadline"] = deadline
    config.pop("config_sha256", None)
    config["config_sha256"] = hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    plan["config"] = config
    save(directory / "config.json",config)
    (directory / "absolute-deadline.txt").write_text(deadline)
    command = create_command(config, directory, deadline)
    meta = next(i for i, item in enumerate(command) if item.startswith("--metadata-from-file="))
    command[meta] += ",zimfo-absolute-deadline=" + str(directory / "absolute-deadline.txt")
    plan.update(status="launch_requested", absolute_deadline=deadline, create_command=command)
    save(directory / "plan.json", plan)  # no uncertain-create automatic retry
    result = run(command)
    save(directory / "instance.json", result)
    plan["status"] = "launched"
    save(directory / "plan.json", plan)
    return plan


def finish(directory):
    """Collect the final report and delete only this stopped VM, retaining disks."""
    from gcs_checkpoints import make_client
    from cloud_gpu.bootstrap import validate_solver_status, validate_continuation_status
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    if plan["status"] not in ("launched", "launch_requested"):
        raise ValueError("No launch to finish")
    config = plan["config"]
    bucket = make_client(PROJECT, use_gcloud=True).bucket(config["ready"]["bucket"])
    blob = bucket.get_blob(f"runs/{config['run_id']}/vm-result.json")
    if blob is None or blob.size > 1024*1024:
        raise ValueError("No bounded terminal report yet; inspect instance/serial logs")
    report = json.loads(blob.download_as_bytes(if_generation_match=int(blob.generation)))
    if report.get("run_id") != config["run_id"] or report.get("config_sha256") != config["config_sha256"]:
        raise ValueError("Terminal report does not belong to reviewed run")
    expected_status = "validated_continuation_run" if config.get("mode") == "continuation" else "validated_checkpointed_run"
    valid = report.get("status") == expected_status
    if valid:
        if config.get("mode") == "continuation": validate_continuation_status(config, report.get("solver_status"))
        else: validate_solver_status(report.get("solver_status"))
    save(directory / "result.json", report)
    inspected = subprocess.run(["gcloud", "compute", "instances", "describe", config["run_id"],
        f"--project={PROJECT}", f"--zone={ZONE}", "--format=json"], capture_output=True, text=True)
    if inspected.returncode:
        if "was not found" not in inspected.stderr and "notFound" not in inspected.stderr:
            raise RuntimeError("Could not verify instance state")
    else:
        instance = json.loads(inspected.stdout)
        if (instance.get("labels", {}).get("zimfo-run") != config["run_id"]
                or instance.get("labels", {}).get("zimfo-purpose") != "quantization"
                or instance.get("status") != "TERMINATED"):
            raise ValueError("Instance must be this owned, stopped job before cleanup")
        for role in ("boot_disk", "data_disk"):
            disk = config["ready"][role]
            matches = [d for d in instance["disks"] if d["source"] == disk["self_link"]]
            if len(matches) != 1 or matches[0].get("autoDelete") is not False:
                raise ValueError("Attached disk identity or preservation changed")
        run(["gcloud", "compute", "instances", "delete", config["run_id"], f"--project={PROJECT}",
             f"--zone={ZONE}", "--keep-disks=all", "--quiet", "--format=json"])
    retained = {}
    for role in ("boot_disk", "data_disk"):
        expected = config["ready"][role]
        disk = run(["gcloud", "compute", "disks", "describe", expected["name"], f"--project={PROJECT}", f"--zone={ZONE}", "--format=json"])
        if str(disk["id"]) != expected["id"] or disk.get("users"):
            raise ValueError("Retained disk identity/detachment failed")
        retained[role] = expected
    plan.update(status="finished_validated" if valid else "finished_failed", retained_disks=retained)
    save(directory / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--continuation-plan")
    parser.add_argument("--staged-plan-path")
    parser.add_argument("--continuation-staging")
    parser.add_argument("--checkpoint-mode",choices=("gcs","local-spool"),default="gcs")
    parser.add_argument("--spool-min-free-bytes",type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    if args.execute and args.finish:
        parser.error("Choose execute or finish")
    if args.finish:
        result = finish(args.output)
    elif args.execute:
        result = execute(args.output)
    else:
        if not args.receipt:
            parser.error("--receipt required when preparing")
        result = prepare(args.receipt, args.output, max_seconds=args.max_seconds,
                         continuation_plan=args.continuation_plan, staged_plan_path=args.staged_plan_path,
                         continuation_staging=args.continuation_staging, checkpoint_mode=args.checkpoint_mode,
                         spool_min_free_bytes=args.spool_min_free_bytes)
    print(json.dumps(result, indent=2))
