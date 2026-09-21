"""CPU guest preparation; stdlib bootstrap, no GPU or quantization execution.

Configuration and this program arrive through instance metadata. The worker
identity reads the pinned inputs/container and creates one final status object.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

METADATA = "http://metadata.google.internal/computeMetadata/v1/"
MOUNT = Path("/mnt/zimfo-inputs")
DEVICE = "/dev/disk/by-id/google-zimfo-inputs"
CHUNK = 1024 * 1024


def metadata(path):
    request = urllib.request.Request(METADATA + path, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read(1024 * 1024)


def token():
    return json.loads(metadata("instance/service-accounts/default/token"))["access_token"]


def command(args, *, private_input=None):
    # stdin may be a short-lived registry token: never print it or command output
    # from authentication. All other commands have ordinary bounded status logs.
    if private_input is None:
        print(json.dumps({"command": args}), flush=True)
    result = subprocess.run(args, input=private_input, text=True, capture_output=True, timeout=2400)
    if result.returncode:
        if private_input is not None:
            raise RuntimeError("Registry authentication failed")
        raise RuntimeError(f"Command failed: {args[0]}: {result.stderr[-3000:]}")
    return result.stdout.strip()


def download_descriptor(bucket, entry, destination):
    if (type(entry.get("bytes")) is not int or not 0 < entry["bytes"] <= 1024 * 1024
            or not str(entry.get("generation", "")).isdigit()):
        raise ValueError("Invalid committed manifest descriptor")
    query = urllib.parse.urlencode({"alt": "media", "generation": entry["generation"],
                                    "ifGenerationMatch": entry["generation"]})
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(entry['object'], safe='')}?{query}"
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token()})
    digest, count = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as output:
        while data := response.read(CHUNK):
            count += len(data)
            if count > entry["bytes"]:
                raise ValueError("Input manifest exceeds declared size")
            digest.update(data)
            output.write(data)
        output.flush()
        os.fsync(output.fileno())
    if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError("Input manifest checksum mismatch")


def publish_status(config, report):
    body = json.dumps(report, sort_keys=True, indent=2).encode()
    key = f"preparation/{config['run_id']}/result.json"
    query = urllib.parse.urlencode({"uploadType": "media", "name": key, "ifGenerationMatch": 0})
    request = urllib.request.Request(
        f"https://storage.googleapis.com/upload/storage/v1/b/{config['bucket']}/o?{query}",
        data=body, method="POST", headers={"Authorization": "Bearer " + token(),
                                         "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read(1024 * 1024)


def mount_inputs():
    # This device is a newly created, separately labelled data disk. Never format
    # a disk with any existing filesystem/signature; preserve evidence on failure.
    deadline = time.monotonic() + 60
    while not Path(DEVICE).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Dedicated input disk did not appear")
        time.sleep(1)
    if not Path(DEVICE).is_block_device() or int(command(["blockdev", "--getsize64", DEVICE])) != 256 * 1024**3:
        raise RuntimeError("Input device is not the dedicated 256GiB block disk")
    signatures = command(["wipefs", "--no-act", "--json", DEVICE])
    if json.loads(signatures).get("signatures"):
        raise RuntimeError("Input disk is not empty; refusing format/reuse")
    command(["mkfs.ext4", "-L", "zimfo-inputs", "-m", "0", DEVICE])
    MOUNT.mkdir(exist_ok=True)
    command(["mount", "-o", "noatime", DEVICE, str(MOUNT)])


def prepare(config):
    machine = metadata("instance/machine-type").decode().split("/")[-1]
    if machine not in ("n4-standard-2", "n4-standard-4", "c4-standard-2") or machine != config.get("machine_type", "n4-standard-2"):
        raise ValueError("CPU preparation requires the exact reviewed small CPU machine")
    mount_inputs()
    result = {"schema_version": 1, "run_id": config["run_id"], "status": "preparing",
              "config_sha256": config["config_sha256"], "gpu_attached": False,
              "input_commit": config["input_commit"], "image": config["image"]}
    # Install missing host utilities here on cheap CPU time, never in GPU startup.
    if shutil.which("docker") is None:
        command(["apt-get", "update", "-qq"])
        command(["apt-get", "install", "-y", "--no-install-recommends", "docker.io"])
    command(["systemctl", "enable", "--now", "docker"])
    if shutil.which("nvidia-ctk") is None:
        # The pinned NVIDIA driver image must provide the configured package
        # repository. No curl-to-shell installer or driver rebuild is attempted.
        command(["apt-get", "update", "-qq"])
        command(["apt-get", "install", "-y", "--no-install-recommends", "nvidia-container-toolkit"])
    # Persist the frozen prepared environment. GPU startup must not spend time
    # in unattended package jobs or silently change the reviewed kernel/driver.
    installed_units = {line.split()[0] for line in command(
        ["systemctl", "list-unit-files", "--no-legend", "--plain"]).splitlines() if line.split()}
    update_units = [unit for unit in ("apt-daily.service", "apt-daily.timer", "apt-daily-upgrade.service",
        "apt-daily-upgrade.timer", "unattended-upgrades.service") if unit in installed_units]
    if update_units:
        command(["systemctl", "mask", "--now", *update_units])
    driver = command(["modinfo", "-F", "version", "nvidia"])
    if not driver.startswith("580."):
        raise RuntimeError("Pinned CUDA 13 preparation requires the expected 580 driver")
    module = command(["modinfo", "-F", "filename", "nvidia"])
    kernel = command(["uname", "-r"])
    if f"/{kernel}/" not in module:
        raise RuntimeError("NVIDIA module does not match running kernel")
    command(["nvidia-ctk", "runtime", "configure", "--runtime=docker"])
    command(["systemctl", "restart", "docker"])
    runtimes = json.loads(command(["docker", "info", "--format", "{{json .Runtimes}}"]));
    if "nvidia" not in runtimes:
        raise RuntimeError("Docker NVIDIA runtime was not configured")
    toolkit = command(["nvidia-container-cli", "--version"])
    # /run is temporary memory, so credentials cannot enter the boot snapshot.
    with tempfile.TemporaryDirectory(prefix="zimfo-registry-", dir="/run") as auth:
        command(["docker", "--config", auth, "login", "-u", "oauth2accesstoken", "--password-stdin",
                 "https://us-central1-docker.pkg.dev"], private_input=token())
        command(["docker", "--config", auth, "pull", config["image"]])
    inspection = json.loads(command(["docker", "image", "inspect", config["image"]]))[0]
    if config["image"] not in inspection["RepoDigests"] or inspection["Architecture"] != "amd64":
        raise RuntimeError("Pulled image identity/architecture mismatch")
    image_probe = json.loads(command(["docker", "run", "--rm", "--runtime=runc", "--network=none",
        config["image"], "python", "-c",
        "import json,torch,google.cloud.storage,solver.run,solver.job,restore_inputs; "
        "assert not torch.cuda.is_available(); "
        "assert torch.version.cuda.startswith('13.'); "
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_available':False}))"]))
    manifest = MOUNT / "inputs-manifest.json"
    download_descriptor(config["bucket"], config["input_commit"], manifest)
    command(["docker", "run", "--rm", "--runtime=runc", "--network=host", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--mount", f"type=bind,src={MOUNT},dst=/data",
        config["image"], "python", "-m", "restore_inputs", "--manifest", "/data/inputs-manifest.json",
        "--expected-manifest-sha256", config["input_commit"]["sha256"], "--destination", "/data/prepared"])
    validation = json.loads((MOUNT / "prepared/restore-validation.json").read_text())
    if validation.get("status") != "validated":
        raise RuntimeError("Input restore did not validate")
    result.update(status="prepared_cpu_only", kernel=kernel, driver=driver, driver_module=module,
                  toolkit=toolkit, container_probe=image_probe, restore_validation=validation,
                  gpu_smoke_passed=False)
    return result


def main():
    config = json.loads(metadata("instance/attributes/zimfo-prep-config"))
    expected = config.pop("config_sha256")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError("Preparation config checksum mismatch")
    config["config_sha256"] = expected
    started = time.monotonic()
    try:
        report = prepare(config)
    except BaseException as error:
        report = {"status": "failed", "run_id": config["run_id"], "config_sha256": expected,
                  "error": str(error), "gpu_smoke_passed": False}
    report["elapsed_seconds"] = time.monotonic() - started
    path = MOUNT / "preparation.json" if MOUNT.is_mount() else Path("/opt/zimfo-prep/preparation.json")
    with path.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report), flush=True)
    command(["sync"])
    try:
        publish_status(config, report)  # one final object; progress stays on serial/local logs
    finally:
        subprocess.run(["shutdown", "-h", "now"], check=False)
    return 0 if report["status"] == "prepared_cpu_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
