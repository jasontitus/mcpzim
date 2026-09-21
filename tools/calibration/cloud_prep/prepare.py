"""Prepare a reviewable CPU-only staging VM plan; execute only explicitly.

No GPU machine/accelerator options exist. Readiness requires a committed input
manifest and immutable final solver image. Both disks survive VM deletion.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

PROJECT = "tiltastech-zimfo"
ZONE = "us-central1-b"
IMAGE_PROJECT = "deeplearning-platform-release"
OS_IMAGE = "common-cu129-ubuntu-2204-nvidia-580-v20260909"
OS_IMAGE_ID = "3612508179781164991"
SA = "zimfo-quant-worker@tiltastech-zimfo.iam.gserviceaccount.com"
TAG = "zimfo-cpu-prep"
FIREWALL = "zimfo-cpu-prep-deny-ingress"
REGISTRY = "us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/"
HERE = Path(__file__).resolve().parent
CPU_MACHINES = {"n4-standard-2": .0907, "n4-standard-4": .1814, "c4-standard-2": .096866}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Cloud command failed (exit {result.returncode}): {result.stderr.strip()[:16384]}")
    return json.loads(result.stdout)


def build_plan(status_path, manifest_path, image, output):
    status, manifest = read(status_path), read(manifest_path)
    if status.get("status") != "staged" or manifest.get("status") != "staged":
        raise ValueError("Committed input staging is required before CPU VM planning")
    commit = status["commit"]
    if (type(commit.get("bytes")) is not int or not 0 < commit["bytes"] <= 1024 * 1024
            or sha(manifest_path) != commit["sha256"] or Path(manifest_path).stat().st_size != commit["bytes"]
            or status["bucket"] != manifest["bucket"]
            or manifest["revision"] != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"):
        raise ValueError("Staging receipt does not bind the exact input manifest")
    if not re.fullmatch(re.escape(REGISTRY) + r"[a-z0-9][a-z0-9_./-]*@sha256:[0-9a-f]{64}", image):
        raise ValueError("Final private solver image must be pinned by digest, not tag")
    if not str(commit.get("generation", "")).isdigit() or int(commit["generation"]) <= 0:
        raise ValueError("Committed generation required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = "zimfo-prep-" + uuid.uuid4().hex[:12]
    config = {"run_id": run_id, "bucket": status["bucket"], "input_commit": commit, "image": image,
              "machine_type": "n4-standard-2",
              "bootstrap_sha256": sha(HERE / "bootstrap.py"), "startup_sha256": sha(HERE / "startup.sh")}
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "bootstrap.py").write_bytes((HERE / "bootstrap.py").read_bytes())
    (output / "startup.sh").write_bytes((HERE / "startup.sh").read_bytes())
    plan = {"schema_version": 1, "status": "prepared_not_launched", "run_id": run_id,
            "project": PROJECT, "zone": ZONE, "machine_type": "n4-standard-2", "gpu_count": 0,
            "max_vm_runtime_seconds": 3600, "config": config,
            "input_disk": run_id + "-inputs", "boot_disk": run_id + "-boot",
            "input_snapshot": run_id + "-inputs", "boot_image": run_id + "-boot",
            "os_image_id": OS_IMAGE_ID, "preserve_disks": True,
            "result_object": f"preparation/{run_id}/result.json",
            "cost": {"cpu_vm_usd_per_hour": .0907, "vm_max_hours": 1,
                     "note": "Storage356GiB, IPv4, registry/GCS and later snapshots extra; disk charges persist after VM deletion"}}
    plan["create_command"] = create_command(plan, output)
    plan["disk_commands"] = disk_commands(plan)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def create_command(plan, directory):
    machine = plan["machine_type"]
    if machine not in CPU_MACHINES:
        raise ValueError("Only explicitly reviewed small CPU machine types are allowed")
    return ["gcloud", "compute", "instances", "create", plan["run_id"], f"--project={PROJECT}", f"--zone={ZONE}",
            f"--machine-type={machine}", "--provisioning-model=STANDARD", "--max-run-duration=1h",
            "--instance-termination-action=DELETE", "--no-restart-on-failure", "--maintenance-policy=TERMINATE",
            f"--disk=name={plan['boot_disk']},boot=yes,auto-delete=no,device-name=zimfo-boot",
            f"--disk=name={plan['input_disk']},device-name=zimfo-inputs,auto-delete=no,mode=rw",
            f"--service-account={SA}", "--scopes=https://www.googleapis.com/auth/cloud-platform",
            "--network-interface=network=default,subnet=default,nic-type=GVNIC", f"--tags={TAG}",
            f"--labels=zimfo-purpose=cpu-preparation,zimfo-run={plan['run_id']}",
            "--metadata=enable-oslogin=true,block-project-ssh-keys=true,serial-port-enable=false",
            f"--metadata-from-file=startup-script={directory / 'startup.sh'},zimfo-prep-bootstrap={directory / 'bootstrap.py'},zimfo-prep-config={directory / 'config.json'}",
            "--format=json"]


def disk_commands(plan):
    if plan.get("recovery"):
        return []
    common = [f"--project={PROJECT}", f"--zone={ZONE}", "--type=hyperdisk-balanced",
              f"--labels=zimfo-purpose=cpu-preparation,zimfo-run={plan['run_id']}", "--format=json"]
    return [["gcloud", "compute", "disks", "create", plan["boot_disk"], "--size=100GB",
             "--provisioned-iops=3000", "--provisioned-throughput=140",
             f"--image={OS_IMAGE}", f"--image-project={IMAGE_PROJECT}", *common],
            ["gcloud", "compute", "disks", "create", plan["input_disk"], "--size=256GB",
             "--provisioned-iops=3000", "--provisioned-throughput=750", *common]]


def verify_prepared_files(plan, directory):
    config = read(directory / "config.json")
    if config != plan["config"]:
        raise ValueError("Prepared configuration changed after review")
    if config.get("machine_type", "n4-standard-2") != plan["machine_type"]:
        raise ValueError("Guest machine expectation differs from CPU launch plan")
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != config["config_sha256"]:
        raise ValueError("Prepared configuration checksum changed")
    for name in ("bootstrap", "startup"):
        suffix = ".py" if name == "bootstrap" else ".sh"
        if sha(directory / (name + suffix)) != config[name + "_sha256"]:
            raise ValueError("Prepared startup code changed after review")
    if plan["create_command"] != create_command(plan, directory) or plan["disk_commands"] != disk_commands(plan):
        raise ValueError("VM command differs from fixed CPU-only plan")


def instance_absent(plan):
    instances = run(["gcloud", "compute", "instances", "list", f"--project={PROJECT}",
                     f"--filter=name={plan['run_id']}", "--format=json"])
    if not isinstance(instances, list) or instances:
        raise ValueError("Recovery requires confirmed absence of the original VM")


def verify_recovery_disk(plan, disk, role, expected_id):
    size = 100 if role == "boot_disk" else 256
    throughput = 140 if role == "boot_disk" else 750
    suffix = f"/projects/{PROJECT}/zones/{ZONE}/disks/{plan[role]}"
    if (str(disk.get("id")) != str(expected_id) or disk.get("name") != plan[role]
            or disk.get("status") != "READY" or disk.get("users")
            or disk.get("labels", {}).get("zimfo-run") != plan["run_id"]
            or disk.get("labels", {}).get("zimfo-purpose") != "cpu-preparation"
            or int(disk.get("sizeGb", 0)) != size
            or not disk.get("selfLink", "").endswith(suffix)
            or not disk.get("type", "").endswith(f"/zones/{ZONE}/diskTypes/hyperdisk-balanced")
            or int(disk.get("provisionedIops", 0)) != 3000
            or int(disk.get("provisionedThroughput", 0)) != throughput):
        raise ValueError("Existing recovery disk identity/ownership/state differs")
    if role == "boot_disk":
        if str(disk.get("sourceImageId")) != OS_IMAGE_ID:
            raise ValueError("Recovery boot disk source image differs")
    elif any(disk.get(key) for key in ("sourceImage", "sourceSnapshot", "sourceDisk")):
        raise ValueError("Recovery input disk must be originally blank; guest also verifies signatures")


def recovery_resources(plan):
    instance_absent(plan)
    if set(plan["recovery"]["disks"]) != {"boot_disk", "input_disk"}:
        raise ValueError("Recovery requires both exact existing disk identities")
    result = {}
    for role, expected in plan["recovery"]["disks"].items():
        disk = run(["gcloud", "compute", "disks", "describe", plan[role],
                    f"--project={PROJECT}", f"--zone={ZONE}", "--format=json"])
        verify_recovery_disk(plan, disk, role, expected["id"])
        for key in ("creationTimestamp", "lastAttachTimestamp", "lastDetachTimestamp"):
            if disk.get(key) != expected.get(key):
                raise ValueError("Recovery disk was recreated or attached after the reviewed recovery plan")
        result[role] = disk
    return result


def build_recovery(directory, image, output, boot_disk_id, input_disk_id, machine_type="n4-standard-4"):
    """Read-only cloud inspection, new local plan; preserve the failed attempt."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    plan = read(directory / "plan.json")
    if plan["status"] != "launch_requested" or plan.get("recovery"):
        raise ValueError("Recovery requires an original attempted plan, not an automatic retry")
    verify_prepared_files(plan, directory)
    if machine_type not in CPU_MACHINES:
        raise ValueError("Only explicitly reviewed small CPU machine types are allowed")
    if not re.fullmatch(re.escape(REGISTRY) + r"[a-z0-9][a-z0-9_./-]*@sha256:[0-9a-f]{64}", image):
        raise ValueError("Recovery solver image must be an immutable private digest")
    instance_absent(plan)
    target = f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/instances/{plan['run_id']}"
    operations = run(["gcloud", "compute", "operations", "list", f"--project={PROJECT}",
                      f"--filter=targetLink={target}", "--format=json"])
    matches = [op for op in operations if op.get("targetLink") == target and op.get("operationType") == "insert"]
    if not matches: raise ValueError("No definite original VM create operation found")
    operation = max(matches, key=lambda op: op.get("insertTime", ""))
    errors = operation.get("error", {}).get("errors", [])
    if (operation.get("status") != "DONE" or not errors
            or any(e.get("code") not in ("ZONE_RESOURCE_POOL_EXHAUSTED", "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS") for e in errors)):
        raise ValueError("VM create did not definitely fail from resource exhaustion")
    disks = {}
    for role, identity in (("boot_disk", boot_disk_id), ("input_disk", input_disk_id)):
        disk = run(["gcloud", "compute", "disks", "describe", plan[role],
                    f"--project={PROJECT}", f"--zone={ZONE}", "--format=json"])
        verify_recovery_disk(plan, disk, role, identity)
        end = datetime.fromisoformat(operation["endTime"])
        start = datetime.fromisoformat(operation["startTime"])
        for key in ("lastAttachTimestamp", "lastDetachTimestamp"):
            if disk.get(key) and not start <= datetime.fromisoformat(disk[key]) <= end:
                raise ValueError("Disk attachment history differs from the failed create attempt")
        disks[role] = disk
    # No existing files are amended. This new directory preserves exact prior
    # inputs before writing an independently reviewable recovery configuration.
    output.mkdir(parents=True, exist_ok=False)
    prior = output / "previous-attempt"; prior.mkdir()
    for name in ("plan.json", "config.json", "bootstrap.py", "startup.sh"):
        shutil.copyfile(directory / name, prior / name)
    plan["recovery"] = {"source_directory": str(directory), "source_plan_sha256": sha(directory / "plan.json"),
                        "failed_operation": operation, "disks": disks,
                        "vm_absence_verified": True, "guest_checks_input_disk_signatures": True}
    plan.update(status="prepared_not_launched", machine_type=machine_type)
    plan["cost"]["cpu_vm_usd_per_hour"] = CPU_MACHINES[machine_type]
    config = plan["config"]
    config.update(image=image, machine_type=machine_type, bootstrap_sha256=sha(HERE / "bootstrap.py"),
                  startup_sha256=sha(HERE / "startup.sh"))
    config.pop("config_sha256")
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for name in ("bootstrap.py", "startup.sh"): shutil.copyfile(HERE / name, output / name)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    plan["disk_commands"] = []
    plan["create_command"] = create_command(plan, output)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    verify_prepared_files(plan, output)
    return plan


def ensure_ingress_denied():
    command = ["gcloud", "compute", "firewall-rules", "describe", FIREWALL, f"--project={PROJECT}", "--format=json"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        # Do not interpret arbitrary auth/network errors as absence.
        if "was not found" not in result.stderr and "notFound" not in result.stderr:
            raise RuntimeError("Cannot verify dedicated ingress firewall")
        return run(["gcloud", "compute", "firewall-rules", "create", FIREWALL, f"--project={PROJECT}",
                    "--network=default", "--direction=INGRESS", "--priority=0", "--action=DENY",
                    "--rules=all", "--source-ranges=0.0.0.0/0", f"--target-tags={TAG}", "--format=json"])
    rule = json.loads(result.stdout)
    if (rule.get("direction") != "INGRESS" or rule.get("priority") != 0 or rule.get("disabled", False)
            or rule.get("denied") != [{"IPProtocol": "all"}]
            or rule.get("targetTags") != [TAG] or rule.get("sourceRanges") != ["0.0.0.0/0"]
            or not rule.get("network", "").endswith("/networks/default")):
        raise ValueError("Existing dedicated firewall differs; refusing to alter it")


def execute(directory):
    directory = Path(directory).resolve()
    plan = read(directory / "plan.json")
    if plan["status"] != "prepared_not_launched":
        raise ValueError("Plan has already been attempted; inspect retained resources before retry")
    verify_prepared_files(plan, directory)
    if plan.get("recovery"):
        recovery_resources(plan)
    image = run(["gcloud", "compute", "images", "describe", OS_IMAGE, f"--project={IMAGE_PROJECT}", "--format=json"])
    if str(image["id"]) != OS_IMAGE_ID or image["status"] != "READY":
        raise ValueError("Pinned driver OS image changed")
    # Read-only checks happen before any firewall/VM operation.
    from gcs_checkpoints import make_client
    client = make_client(PROJECT, use_gcloud=True)
    commit = plan["config"]["input_commit"]
    blob = client.bucket(plan["config"]["bucket"]).blob(commit["object"], generation=int(commit["generation"]))
    data = blob.download_as_bytes(if_generation_match=int(commit["generation"]))
    if len(data) != commit["bytes"] or hashlib.sha256(data).hexdigest() != commit["sha256"]:
        raise ValueError("Remote input commit failed verification")
    run(["gcloud", "artifacts", "docker", "images", "describe", plan["config"]["image"], f"--project={PROJECT}", "--format=json"])
    ensure_ingress_denied()
    plan["status"] = "launch_requested"
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for command in plan["disk_commands"]:
        run(command)
    # An uncertain create response is NOT retried with a new VM name.
    result = run(plan["create_command"])
    plan["status"] = "launched"
    (directory / "instance.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def snapshot_recipe(plan):
    """Reviewable commands; caller first verifies report and a stopped/deleted VM.

    No VM/disk deletion is generated. The future GPU VM clones these immutable
    prepared artifacts, avoiding installs, image pulls or input downloads.
    """
    return {"input_snapshot": ["gcloud", "compute", "snapshots", "create", plan["input_snapshot"],
        f"--source-disk={plan['input_disk']}", f"--source-disk-zone={ZONE}", f"--project={PROJECT}", "--storage-location=us-central1"],
        "prepared_boot_image": ["gcloud", "compute", "images", "create", plan["boot_image"],
        f"--source-disk={plan['boot_disk']}", f"--source-disk-zone={ZONE}", f"--project={PROJECT}", "--storage-location=us-central1"]}


def ready_receipt(plan, report, boot, data):
    if (report.get("status") != "prepared_cpu_only" or report.get("run_id") != plan["run_id"]
            or report.get("config_sha256") != plan["config"]["config_sha256"]
            or report.get("image") != plan["config"]["image"]
            or report.get("input_commit") != plan["config"]["input_commit"]
            or report.get("gpu_attached") is not False or report.get("gpu_smoke_passed") is not False):
        raise ValueError("Preparation report does not prove this CPU-only plan")
    if (not report.get("driver", "").startswith("580.")
            or report.get("restore_validation", {}).get("status") != "validated"
            or not report.get("container_probe", {}).get("cuda", "").startswith("13.")):
        raise ValueError("Missing driver/container/input validation")
    disks = {}
    for role, disk, expected_size in (("boot_disk", boot, 100), ("data_disk", data, 256)):
        expected_name = plan["boot_disk"] if role == "boot_disk" else plan["input_disk"]
        if (disk.get("name") != expected_name or disk.get("users")
                or disk.get("labels", {}).get("zimfo-run") != plan["run_id"]
                or int(disk.get("sizeGb", 0)) != expected_size or disk.get("status") != "READY"
                or not disk.get("type", "").endswith("/hyperdisk-balanced")
                or not str(disk.get("id", "")).isdigit()):
            raise ValueError("Prepared disk ownership, identity or detach check failed")
        disks[role] = {"name": expected_name, "id": str(disk["id"]), "self_link": disk["selfLink"],
                       "size_gib": expected_size, "type": "hyperdisk-balanced"}
    return {"schema_version": 1, "status": "prepared_disks_detached", "run_id": plan["run_id"],
            "project": PROJECT, "zone": ZONE, "source_image_id": OS_IMAGE_ID, **disks,
            "production_image": report["image"], "input_commit": report["input_commit"],
            "bucket": plan["config"]["bucket"], "config_sha256": report["config_sha256"],
            "runtime_proof": {key: report[key] for key in ("kernel", "driver", "driver_module", "toolkit", "container_probe")},
            "restore_validation": report["restore_validation"], "gpu_smoke_passed": False,
            "paths": {"mount": "/mnt/zimfo-inputs", "model": "/mnt/zimfo-inputs/prepared/model",
                      "calibration": "/mnt/zimfo-inputs/prepared/calibration",
                      "container_workdir": "/opt/zimfo-runtime", "data_device": "/dev/disk/by-id/google-zimfo-inputs"}}


def finish(directory):
    """Validate success, delete only this labelled VM, retain/detach both disks."""
    directory = Path(directory).resolve()
    plan = read(directory / "plan.json")
    if plan["status"] not in ("launched", "launch_requested"):
        raise ValueError("This plan has not been launched")
    verify_prepared_files(plan, directory)
    from gcs_checkpoints import make_client
    bucket = make_client(PROJECT, use_gcloud=True).bucket(plan["config"]["bucket"])
    blob = bucket.get_blob(plan["result_object"])
    if blob is None or blob.size > 1024 * 1024:
        raise ValueError("No bounded final preparation report")
    report = json.loads(blob.download_as_bytes(if_generation_match=int(blob.generation)))
    # Check report identity before even considering a VM action.
    if (report.get("status") != "prepared_cpu_only" or report.get("run_id") != plan["run_id"]
            or report.get("config_sha256") != plan["config"]["config_sha256"]):
        raise ValueError("Preparation did not complete; retain resources for diagnosis")
    described = subprocess.run(["gcloud", "compute", "instances", "describe", plan["run_id"],
        f"--project={PROJECT}", f"--zone={ZONE}", "--format=json"], capture_output=True, text=True)
    if described.returncode == 0:
        instance = json.loads(described.stdout)
        if instance.get("labels", {}).get("zimfo-run") != plan["run_id"]:
            raise ValueError("Refusing to delete VM without exact ownership label")
        if instance.get("status") != "TERMINATED":
            raise ValueError("Wait until startup shuts the CPU VM down before detaching disks")
        attached = instance.get("disks", [])
        if ({entry["source"].split("/")[-1] for entry in attached} != {plan["boot_disk"], plan["input_disk"]}
                or any(entry.get("autoDelete", True) for entry in attached)):
            raise ValueError("Unexpected disks or deletion policy; no cleanup performed")
        subprocess.run(["gcloud", "compute", "instances", "delete", plan["run_id"],
            f"--project={PROJECT}", f"--zone={ZONE}", "--keep-disks=all", "--quiet"], check=True)
    elif "was not found" not in described.stderr and "notFound" not in described.stderr:
        raise RuntimeError("Cannot verify whether owned CPU VM exists")
    disks = [run(["gcloud", "compute", "disks", "describe", plan[role], f"--project={PROJECT}",
                  f"--zone={ZONE}", "--format=json"]) for role in ("boot_disk", "input_disk")]
    receipt = ready_receipt(plan, report, *disks)
    receipt["preparation_report_generation"] = str(blob.generation)
    (directory / "ready.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan_parser = sub.add_parser("plan")
    for key in ("input-status", "input-manifest", "image", "output"):
        plan_parser.add_argument("--" + key, required=True)
    recovery_parser = sub.add_parser("recover")
    recovery_parser.add_argument("directory", type=Path)
    for key in ("image", "output", "boot-disk-id", "input-disk-id"):
        recovery_parser.add_argument("--" + key, required=True)
    recovery_parser.add_argument("--machine-type", choices=sorted(CPU_MACHINES), default="n4-standard-4")
    for mode in ("execute", "finish", "snapshot-recipe"):
        sub.add_parser(mode).add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        result = build_plan(args.input_status, args.input_manifest, args.image, args.output)
    elif args.mode == "recover":
        result = build_recovery(args.directory, args.image, args.output, args.boot_disk_id,
                                args.input_disk_id, args.machine_type)
    elif args.mode == "execute":
        result = execute(args.directory)
    elif args.mode == "finish":
        result = finish(args.directory)
    else:
        result = snapshot_recipe(read(args.directory / "plan.json"))
    print(json.dumps(result, indent=2))
