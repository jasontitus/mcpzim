"""CPU-only continuation of a verified, interrupted input restore; never formats."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tempfile
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
    print(json.dumps({"command": args}), flush=True)
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=60).stdout.strip()


def restore_command(config, staging):
    return ["docker", "run", "--name", config["resume_container"], "--rm", "--pull=never",
            "--runtime=runc", "--network=host", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--mount", f"type=bind,src={MOUNT},dst=/data",
            "--mount", "type=bind,src=/opt/zimfo-prep/restore_inputs.py,dst=/opt/zimfo-app/restore_inputs.py,readonly",
            config["image"], "python", "-u", "-m", "restore_inputs", "--manifest", "/data/inputs-manifest.json",
            "--expected-manifest-sha256", config["input_commit"]["sha256"],
            "--destination", "/data/prepared", "--resume-staging", "/data/" + staging.name,
            "--adopt-legacy-staging"]


def prepare(config):
    machine = metadata("instance/machine-type").decode().split("/")[-1]
    if machine != "c4-standard-2" or machine != config["machine_type"]:
        raise ValueError("Continuation requires the reviewed CPU-only machine")
    if int(command(["blockdev", "--getsize64", DEVICE])) != 256 * 1024**3:
        raise ValueError("Unexpected data disk capacity")
    if command(["blkid", "-s", "LABEL", "-o", "value", DEVICE]) != "zimfo-inputs":
        raise ValueError("Expected existing input filesystem; will not format")
    MOUNT.mkdir(exist_ok=True)
    if not MOUNT.is_mount():
        command(["mount", "-o", "noatime", DEVICE, str(MOUNT)])
    kernel = command(["uname", "-r"])
    driver = command(["modinfo", "-F", "version", "nvidia"])
    module = command(["modinfo", "-F", "filename", "nvidia"])
    license_value = command(["modinfo", "-F", "license", "nvidia"])
    if not driver.startswith("580.") or f"/{kernel}/" not in module or license_value != "Dual MIT/GPL":
        raise ValueError("Expected matching NVIDIA580 open kernel module")
    toolkit = command(["nvidia-container-cli", "--version"])
    if config.get("cache_only"):
        token = json.loads(metadata("instance/service-accounts/default/token"))["access_token"]
        with tempfile.TemporaryDirectory(prefix="zimfo-cache-auth-", dir="/run") as auth:
            subprocess.run(["docker", "--config", auth, "login", "-u", "oauth2accesstoken", "--password-stdin",
                            "https://us-central1-docker.pkg.dev"], input=token, text=True,
                           capture_output=True, check=True, timeout=60)
            subprocess.run(["docker", "--config", auth, "pull", config["image"]],
                           capture_output=True, check=True, timeout=300)
    inspection = json.loads(command(["docker", "image", "inspect", config["image"]]))[0]
    if config["image"] not in inspection["RepoDigests"] or inspection["Architecture"] != "amd64":
        raise ValueError("Reviewed runtime missing; continuation never pulls or installs")
    manifest = MOUNT / "inputs-manifest.json"
    if manifest.is_symlink() or hashlib.sha256(manifest.read_bytes()).hexdigest() != config["input_commit"]["sha256"]:
        raise ValueError("Existing committed input manifest differs")
    benchmark = None
    if config.get("cache_only"):
        validation = json.loads((MOUNT / "prepared/restore-validation.json").read_text())
        if validation != config["expected_restore_validation"]:
            raise ValueError("Prepared input receipt changed")
        code = """import hashlib,json,sys,time
from gcs_checkpoints import GoogleStorageBackend,GCS_TRANSFER_BYTES
entry=json.loads(sys.argv[1]);backend=GoogleStorageBackend(sys.argv[2]);start=time.monotonic();count=0;digest=hashlib.sha256()
assert GCS_TRANSFER_BYTES==64*1024*1024
with backend.open(entry['object'],int(entry['generation'])) as stream:
 while data:=stream.read(1024*1024):
  count+=len(data);digest.update(data)
  if count>entry['bytes']:raise ValueError('Remote object exceeds descriptor')
elapsed=time.monotonic()-start
assert count==entry['bytes'] and digest.hexdigest()==entry['sha256']
print(json.dumps({'bytes':count,'sha256':digest.hexdigest(),'seconds':elapsed,'MiB_per_second':count/elapsed/1024**2,'range_bytes':GCS_TRANSFER_BYTES}))
"""
        result = subprocess.run(["docker", "run", "--rm", "--runtime=runc", "--network=host", "--pull=never",
            config["image"], "python", "-c", code, json.dumps(config["benchmark_object"]), config["bucket"]],
            text=True, capture_output=True, check=True, timeout=600)
        benchmark = json.loads(result.stdout)
        if benchmark["MiB_per_second"] < 32:
            raise ValueError("Full-size checkpoint readback remains too slow: " + json.dumps(benchmark))
        print(json.dumps({"transport_benchmark":benchmark}), flush=True)
    else:
        source = metadata("instance/attributes/zimfo-restore-source")
        if hashlib.sha256(source).hexdigest() != config["restore_source_sha256"]:
            raise ValueError("Continuation source changed")
        Path("/opt/zimfo-prep/restore_inputs.py").write_bytes(source)
        candidates = list(MOUNT.glob(".input-restore-*"))
        if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_dir():
            raise ValueError("Need exactly one existing ordinary restore directory")
        if (MOUNT / "prepared").exists():
            raise ValueError("Restore already completed; collect original receipt")
        args = restore_command(config, candidates[0])
        print(json.dumps({"command": args}), flush=True)
        try:
            subprocess.run(args, check=True, timeout=2400)
        finally:
            subprocess.run(["docker", "stop", "--time=20", config["resume_container"]],
                           capture_output=True, timeout=30, check=False)
    validation = json.loads((MOUNT / "prepared/restore-validation.json").read_text())
    if validation.get("status") != "validated" or validation.get("input_commit_sha256") != config["input_commit"]["sha256"]:
        raise ValueError("Restored inputs did not validate")
    probe = json.loads(command(["docker", "run", "--rm", "--runtime=runc", "--network=none", config["image"],
        "python", "-c", "import json,torch,solver.job,solver.run; assert not torch.cuda.is_available(); "
        "assert torch.version.cuda.startswith('13.'); print(json.dumps({'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'gpu_available':False}))"]))
    return {"schema_version": 1, "run_id": config["run_id"], "status": "prepared_cpu_only",
            "config_sha256": config["config_sha256"], "gpu_attached": False,
            "input_commit": config["input_commit"], "image": config["image"],
            "kernel": kernel, "driver": driver, "driver_module": module, "driver_license": license_value,
            "toolkit": toolkit, "container_probe": probe, "restore_validation": validation,
            "gpu_smoke_passed": False, "restore_source_sha256": config.get("restore_source_sha256"),
            "transport_benchmark": benchmark}


def main():
    config = json.loads(metadata("instance/attributes/zimfo-prep-config"))
    expected = config.pop("config_sha256")
    if hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != expected:
        raise ValueError("Continuation configuration changed")
    config["config_sha256"] = expected
    started = time.monotonic()
    try:
        result = prepare(config)
    except BaseException as error:
        result = {"status": "failed", "run_id": config["run_id"], "config_sha256": expected,
                  "error": str(error), "gpu_smoke_passed": False}
    result["elapsed_seconds"] = time.monotonic() - started
    destination = MOUNT if MOUNT.is_mount() else Path("/opt/zimfo-prep")
    with (destination / (config["resume_container"] + "-result.json")).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result), flush=True)
    command(["sync"])
    token = json.loads(metadata("instance/service-accounts/default/token"))["access_token"]
    query = urllib.parse.urlencode({"uploadType": "media", "name": config["result_object"], "ifGenerationMatch": 0})
    request = urllib.request.Request(f"https://storage.googleapis.com/upload/storage/v1/b/{config['bucket']}/o?{query}",
        data=json.dumps(result).encode(), method="POST", headers={"Authorization": "Bearer " + token,
                                                                "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read(1024 * 1024)
    return 0 if result["status"] == "prepared_cpu_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
