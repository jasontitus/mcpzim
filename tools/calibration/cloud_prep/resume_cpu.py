"""Freeze and execute one controlled CPU restore restart, preserving both disks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from cloud_prep.prepare import PROJECT, ZONE, HERE, read, run, sha, verify_prepared_files, create_command


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = read(source / "plan.json")
    verify_prepared_files(original, source)
    if original["status"] != "launched" or original["machine_type"] != "c4-standard-2":
        raise ValueError("Expected the active reviewed C4 CPU preparation")
    output.mkdir(parents=True, exist_ok=False)
    for local, target in ((HERE / "resume_guest.py", "bootstrap.py"),
                          (HERE / "startup.sh", "startup.sh"),
                          (HERE.parent / "restore_inputs.py", "restore_inputs.py")):
        (output / target).write_bytes(local.read_bytes())
    config = dict(original["config"])
    config.pop("config_sha256")
    attempt = "zimfo-resume-" + uuid.uuid4().hex[:12]
    config.update(bootstrap_sha256=sha(output / "bootstrap.py"), startup_sha256=sha(output / "startup.sh"),
                  restore_source_sha256=sha(output / "restore_inputs.py"), resume_container=attempt,
                  result_object=f"preparation/{original['run_id']}/{attempt}/result.json")
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan = dict(original)
    plan.update(status="continuation_prepared", config=config, result_object=config["result_object"],
                continuation_source=str(source), continuation_source_plan_sha256=sha(source / "plan.json"),
                preserve_disks=True, max_vm_runtime_seconds=3600)
    plan["create_command"] = create_command(plan, output)
    plan["disk_commands"] = []
    for name, value in (("config.json", config), ("plan.json", plan)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    return plan


def check_instance(instance, plan):
    if (instance.get("name") != plan["run_id"]
            or instance.get("labels", {}).get("zimfo-purpose") != "cpu-preparation"
            or instance.get("labels", {}).get("zimfo-run") != plan["run_id"]
            or not instance.get("machineType", "").endswith("/c4-standard-2")
            or instance.get("guestAccelerators")
            or instance.get("scheduling", {}).get("maxRunDuration", {}).get("seconds") != "3600"
            or instance.get("scheduling", {}).get("instanceTerminationAction") != "DELETE"):
        raise ValueError("Unexpected CPU VM ownership, hardware or runtime limit")
    disks = instance.get("disks", [])
    if ({disk["source"].split("/")[-1] for disk in disks} != {plan["boot_disk"], plan["input_disk"]}
            or any(disk.get("autoDelete", True) for disk in disks)):
        raise ValueError("Disk preservation or attachment identity changed")


def execute(directory):
    from gcs_checkpoints import make_client
    directory = Path(directory).resolve()
    plan = read(directory / "plan.json")
    if plan["status"] != "continuation_prepared":
        raise ValueError("Already attempted; inspect the same VM before any retry")
    verify_prepared_files(plan, directory)
    if sha(directory / "restore_inputs.py") != plan["config"]["restore_source_sha256"]:
        raise ValueError("Frozen continuation source changed")
    source = Path(plan["continuation_source"])
    if sha(source / "plan.json") != plan["continuation_source_plan_sha256"]:
        raise ValueError("Original plan changed")
    original = read(source / "plan.json")
    bucket = make_client(PROJECT, use_gcloud=True).bucket(plan["config"]["bucket"])
    if bucket.get_blob(original["result_object"]) is not None:
        raise ValueError("Original preparation has a terminal report; inspect instead of interrupting")
    common = [f"--project={PROJECT}", f"--zone={ZONE}"]
    inspected = run(["gcloud", "compute", "instances", "describe", plan["run_id"], *common, "--format=json"])
    check_instance(inspected, plan)
    if inspected["status"] != "RUNNING":
        raise ValueError("Expected the currently running CPU preparation")
    for role in ("boot_disk", "input_disk"):
        disk = run(["gcloud", "compute", "disks", "describe", plan[role], *common, "--format=json"])
        expected = original["recovery"]["disks"][role]
        if str(disk["id"]) != str(expected["id"]) or disk.get("labels", {}).get("zimfo-run") != plan["run_id"]:
            raise ValueError("Original retained disk identity changed")
    plan["status"] = "continuation_stop_requested"
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    run(["gcloud", "compute", "instances", "stop", plan["run_id"], *common, "--quiet", "--format=json"])
    inspected = run(["gcloud", "compute", "instances", "describe", plan["run_id"], *common, "--format=json"])
    check_instance(inspected, plan)
    if inspected["status"] != "TERMINATED":
        raise ValueError("Old writer has not stopped")
    metadata = ",".join((f"startup-script={directory / 'startup.sh'}", f"zimfo-prep-bootstrap={directory / 'bootstrap.py'}",
                         f"zimfo-prep-config={directory / 'config.json'}", f"zimfo-restore-source={directory / 'restore_inputs.py'}"))
    run(["gcloud", "compute", "instances", "add-metadata", plan["run_id"], *common,
         "--metadata-from-file=" + metadata, "--format=json"])
    plan["status"] = "launch_requested"
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    result = run(["gcloud", "compute", "instances", "start", plan["run_id"], *common, "--format=json"])
    (directory / "instance.json").write_text(json.dumps(result, indent=2) + "\n")
    plan["status"] = "launched"
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute and not args.source:
        parser.error("--source is required when preparing")
    print(json.dumps(execute(args.output) if args.execute else prepare(args.source, args.output), indent=2))
