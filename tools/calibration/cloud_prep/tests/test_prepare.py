import hashlib
import json
from pathlib import Path

import pytest

from cloud_prep import prepare, bootstrap


@pytest.fixture
def inputs(tmp_path):
    manifest = tmp_path / "inputs-manifest.json"
    manifest.write_text(json.dumps({"status": "staged", "bucket": "test-bucket",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}))
    receipt = {"object": "inputs/test/inputs-manifest.json", "generation": "123",
               "sha256": prepare.sha(manifest), "bytes": manifest.stat().st_size}
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "staged", "bucket": "test-bucket", "commit": receipt}))
    return status, manifest, prepare.REGISTRY + "solver@sha256:" + "a" * 64


def test_plan_is_local_cpu_only_bounded_and_preserves_disks(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **k: pytest.fail("Plan contacted cloud"))
    plan = prepare.build_plan(*inputs, tmp_path / "plan")
    command = plan["create_command"]
    assert "--machine-type=n4-standard-2" in command
    assert "--max-run-duration=1h" in command
    assert "--instance-termination-action=DELETE" in command
    assert "--no-restart-on-failure" in command
    assert "--tags=zimfo-cpu-prep" in command
    assert not any("accelerator" in arg or "g4-" in arg or "spot" in arg.lower() for arg in command)
    assert len([arg for arg in command if arg.startswith("--disk=") and "auto-delete=no" in arg]) == 2
    assert all("--type=hyperdisk-balanced" in cmd for cmd in plan["disk_commands"])
    assert "--size=100GB" in plan["disk_commands"][0]
    assert "--size=256GB" in plan["disk_commands"][1]
    prepare.verify_prepared_files(plan, tmp_path / "plan")
    assert plan["status"] == "prepared_not_launched"


@pytest.mark.parametrize("fault", ["uploading", "changed_manifest", "tag", "external_registry", "no_generation"])
def test_readiness_gates_reject_before_vm_plan(inputs, tmp_path, fault):
    status, manifest, image = inputs
    data = prepare.read(status)
    if fault == "uploading":
        data["status"] = "uploading"
    elif fault == "changed_manifest":
        manifest.write_text(manifest.read_text() + " ")
    elif fault == "tag":
        image = prepare.REGISTRY + "solver:latest"
    elif fault == "external_registry":
        image = "docker.io/evil/solver@sha256:" + "a" * 64
    else:
        data["commit"]["generation"] = 0
    status.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepare.build_plan(status, manifest, image, tmp_path / "plan")


@pytest.mark.parametrize("file", ["bootstrap.py", "startup.sh", "config.json"])
def test_changed_prepared_files_refuse_launch(inputs, tmp_path, file):
    folder = tmp_path / "plan"
    plan = prepare.build_plan(*inputs, folder)
    if file == "config.json":
        config = prepare.read(folder / file)
        config["image"] += "changed"
        (folder / file).write_text(json.dumps(config))
    else:
        with (folder / file).open("a") as stream:
            stream.write("\n# modified\n")
    with pytest.raises(ValueError):
        prepare.verify_prepared_files(plan, folder)


def test_changed_command_cannot_request_gpu(inputs, tmp_path):
    folder = tmp_path / "plan"
    plan = prepare.build_plan(*inputs, folder)
    plan["create_command"].append("--accelerator=count=1,type=nvidia-l4")
    with pytest.raises(ValueError):
        prepare.verify_prepared_files(plan, folder)


def test_firewall_refuses_existing_unrelated_rule(monkeypatch):
    class Result:
        returncode, stderr = 0, ""
        stdout = json.dumps({"direction": "INGRESS", "priority": 1000, "allowed": [{"IPProtocol": "tcp"}]})
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(ValueError):
        prepare.ensure_ingress_denied()


def test_auth_failure_is_not_firewall_absence(monkeypatch):
    class Result:
        returncode, stderr, stdout = 1, "Authentication failed", ""
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.setattr(prepare, "run", lambda *a: pytest.fail("Created resource after auth failure"))
    with pytest.raises(RuntimeError):
        prepare.ensure_ingress_denied()


def evidence(plan):
    report = {"status": "prepared_cpu_only", "run_id": plan["run_id"],
              "config_sha256": plan["config"]["config_sha256"], "image": plan["config"]["image"],
              "input_commit": plan["config"]["input_commit"], "gpu_attached": False, "gpu_smoke_passed": False,
              "driver": "580.82.09", "kernel": "6.8.0-1040-gcp", "driver_module": "/lib/modules/module.ko",
              "toolkit": "1.17", "container_probe": {"cuda": "13.0", "gpu_available": False},
              "restore_validation": {"status": "validated"}}
    disks = [{"name": plan[role], "id": str(index + 1), "selfLink": "https://example/disk",
              "sizeGb": str(size), "type": "zones/us-central1-b/diskTypes/hyperdisk-balanced",
              "status": "READY", "labels": {"zimfo-run": plan["run_id"]}}
             for index, (role, size) in enumerate((("boot_disk", 100), ("input_disk", 256)))]
    return report, disks


def test_ready_receipt_binds_disks_runtime_and_inputs(inputs, tmp_path):
    plan = prepare.build_plan(*inputs, tmp_path / "plan")
    report, disks = evidence(plan)
    ready = prepare.ready_receipt(plan, report, *disks)
    assert ready["status"] == "prepared_disks_detached"
    assert ready["boot_disk"]["id"] == "1" and ready["data_disk"]["id"] == "2"
    assert ready["source_image_id"] == "3612508179781164991"
    assert ready["gpu_smoke_passed"] is False
    assert ready["paths"]["model"].endswith("prepared/model")


@pytest.mark.parametrize("fault", ["attached", "wrong_owner", "wrong_size", "failed_report", "wrong_image", "driver"])
def test_ready_receipt_refuses_unproven_disks(inputs, tmp_path, fault):
    plan = prepare.build_plan(*inputs, tmp_path / "plan")
    report, disks = evidence(plan)
    if fault == "attached":
        disks[0]["users"] = ["running-cpu-vm"]
    elif fault == "wrong_owner":
        disks[1]["labels"]["zimfo-run"] = "not-ours"
    elif fault == "wrong_size":
        disks[0]["sizeGb"] = "10"
    elif fault == "failed_report":
        report["status"] = "failed"
    elif fault == "wrong_image":
        report["image"] = "wrong"
    else:
        report["driver"] = "570.1"
    with pytest.raises(ValueError):
        prepare.ready_receipt(plan, report, *disks)


def test_snapshot_recipe_does_not_delete_or_create_vms(inputs, tmp_path):
    plan = prepare.build_plan(*inputs, tmp_path / "plan")
    commands = prepare.snapshot_recipe(plan)
    assert all("delete" not in command and "instances" not in command for command in commands.values())


def test_guest_rejects_wrong_machine_before_format(monkeypatch):
    monkeypatch.setattr(bootstrap, "metadata", lambda path: b"zones/test/machineTypes/g4-standard-48")
    monkeypatch.setattr(bootstrap, "mount_inputs", lambda: pytest.fail("Touched disk on GPU host"))
    with pytest.raises(ValueError, match="CPU"):
        bootstrap.prepare({})


@pytest.mark.parametrize('machine', ['n4-standard-2', 'n4-standard-4', 'c4-standard-2'])
def test_guest_success_runs_no_gpu_and_verifies_container_inputs(tmp_path, monkeypatch, machine):
    monkeypatch.setattr(bootstrap, "MOUNT", tmp_path)
    monkeypatch.setattr(bootstrap, "mount_inputs", lambda: None)
    monkeypatch.setattr(bootstrap, "metadata", lambda path: ('zones/test/machineTypes/' + machine).encode())
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "/usr/bin/available")
    monkeypatch.setattr(bootstrap, "token", lambda: "secret-test-token")
    monkeypatch.setattr(bootstrap, "download_descriptor", lambda bucket, entry, target: target.write_text("{}"))
    # Avoid using system /run on the Mac; this is a fake credential temp dir only.
    original_tempdir = bootstrap.tempfile.TemporaryDirectory
    monkeypatch.setattr(bootstrap.tempfile, "TemporaryDirectory", lambda **kwargs: original_tempdir(dir=tmp_path))
    image = prepare.REGISTRY + "solver@sha256:" + "a" * 64
    commands = []

    def command(args, private_input=None):
        commands.append(args)
        if args[:4] == ["modinfo", "-F", "version", "nvidia"]:
            return "580.82.09"
        if args[:4] == ["modinfo", "-F", "filename", "nvidia"]:
            return "/lib/modules/kernel-test/updates/nvidia.ko"
        if args == ["uname", "-r"]:
            return "kernel-test"
        if args[:2] == ["systemctl", "list-unit-files"]:
            return "apt-daily.timer enabled\napt-daily-upgrade.timer enabled\n"
        if args[:2] == ["docker", "info"]:
            return '{"nvidia":{}}'
        if args[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"RepoDigests": [image], "Architecture": "amd64"}])
        if args[:2] == ["docker", "run"] and "-c" in args:
            return '{"cuda":"13.0","gpu_available":false}'
        if "restore_inputs" in args:
            (tmp_path / "prepared").mkdir()
            (tmp_path / "prepared/restore-validation.json").write_text('{"status":"validated"}')
        return "ok"

    monkeypatch.setattr(bootstrap, "command", command)
    result = bootstrap.prepare({"run_id": "test", "config_sha256": "a" * 64, "machine_type": machine,
        "image": image, "bucket": "bucket", "input_commit": {"sha256": "b" * 64}})
    assert result["status"] == "prepared_cpu_only"
    assert result["gpu_smoke_passed"] is False
    assert all("--gpus" not in arg for command in commands for arg in command)
    docker_runs = [command for command in commands if command[:2] == ["docker", "run"]]
    assert all("--runtime=runc" in command for command in docker_runs)
    assert any("--expected-manifest-sha256" in command for command in docker_runs)
    assert ["systemctl", "mask", "--now", "apt-daily.timer", "apt-daily-upgrade.timer"] in commands
    restore = next(command for command in docker_runs if "restore_inputs" in command)
    assert restore[restore.index("python") + 1:restore.index("python") + 3] == ["-m", "restore_inputs"]


def test_startup_verifies_bootstrap_before_execution():
    startup = (prepare.HERE / "startup.sh").read_text()
    assert startup.index("actual != config['bootstrap_sha256']") < startup.index("timeout --signal=TERM")
    assert "--on-active=55m" in startup


@pytest.fixture
def failed_attempt(inputs, tmp_path, monkeypatch):
    old = tmp_path/'original'
    plan = prepare.build_plan(*inputs, old)
    plan['status']='launch_requested'
    (old/'plan.json').write_text(json.dumps(plan))
    _, disks = evidence(plan)
    resources={}
    for role,disk in zip(('boot_disk','input_disk'),disks):
        disk.update(selfLink=f"https://www.googleapis.com/compute/v1/projects/{prepare.PROJECT}/zones/{prepare.ZONE}/disks/{plan[role]}",
                    type=f"https://www.googleapis.com/compute/v1/projects/{prepare.PROJECT}/zones/{prepare.ZONE}/diskTypes/hyperdisk-balanced",
                    provisionedIops='3000',provisionedThroughput='140' if role=='boot_disk' else '750',
                    creationTimestamp='2026-09-19T15:59:42-07:00',
                    lastAttachTimestamp='2026-09-19T15:59:54-07:00',
                    lastDetachTimestamp='2026-09-19T15:59:55-07:00')
        disk['labels']['zimfo-purpose']='cpu-preparation'
        if role=='boot_disk':disk['sourceImageId']=prepare.OS_IMAGE_ID
        resources[plan[role]]=disk
    operation={'name':'operation-test','targetLink':f"https://www.googleapis.com/compute/v1/projects/{prepare.PROJECT}/zones/{prepare.ZONE}/instances/{plan['run_id']}",
        'operationType':'insert','status':'DONE','insertTime':'2026-09-19T15:59:48-07:00',
        'startTime':'2026-09-19T15:59:48-07:00','endTime':'2026-09-19T15:59:56-07:00',
        'error':{'errors':[{'code':'ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS'}]}}
    calls=[]
    def cloud(command):
        calls.append(command)
        if command[2:4]==['instances','list']:return []
        if command[2:4]==['operations','list']:return [operation]
        if command[2:4]==['disks','describe']:return resources[command[4]]
        pytest.fail('Recovery attempted mutation/unexpected command: '+str(command))
    monkeypatch.setattr(prepare,'run',cloud)
    return old,plan,resources,operation,calls


@pytest.mark.parametrize('machine,rate',[('n4-standard-4',.1814),('c4-standard-2',.096866)])
def test_recovery_reuses_exact_disks_preserves_original_and_has_no_mutation(failed_attempt,inputs,tmp_path,machine,rate):
    old,original,resources,operation,calls=failed_attempt
    original_bytes={p.name:p.read_bytes() for p in old.iterdir()}
    output=tmp_path/'recovery'
    image=prepare.REGISTRY+'runtime@sha256:'+'b'*64
    recovered=prepare.build_recovery(old,image,output,'1','2',machine)
    assert recovered['run_id']==original['run_id']
    assert recovered['boot_disk']==original['boot_disk'] and recovered['input_disk']==original['input_disk']
    assert recovered['disk_commands']==[]
    assert '--machine-type='+machine in recovered['create_command']
    assert '--max-run-duration=1h' in recovered['create_command']
    assert recovered['config']['image']==image
    assert recovered['cost']['cpu_vm_usd_per_hour']==rate
    assert all(p.read_bytes()==original_bytes[p.name] for p in old.iterdir())
    assert all((output/'previous-attempt'/name).read_bytes()==data for name,data in original_bytes.items())
    prepare.verify_prepared_files(recovered,output)
    prepare.recovery_resources(recovered)
    assert all(command[3] in ('describe','list') for command in calls)


@pytest.mark.parametrize('fault',['vm_exists','unknown_failure','not_done','wrong_disk_id','attached','wrong_image','later_attachment','gpu_machine'])
def test_recovery_refuses_ambiguous_or_changed_resources(failed_attempt,inputs,tmp_path,monkeypatch,fault):
    old,original,resources,operation,calls=failed_attempt
    machine='n4-standard-4';boot_id='1'
    if fault=='vm_exists':monkeypatch.setattr(prepare,'instance_absent',lambda _:(_ for _ in ()).throw(ValueError('VM exists')))
    elif fault=='unknown_failure':operation['error']['errors'][0]['code']='INTERNAL_ERROR'
    elif fault=='not_done':operation['status']='RUNNING'
    elif fault=='wrong_disk_id':boot_id='999'
    elif fault=='attached':resources[original['input_disk']]['users']=['some-vm']
    elif fault=='wrong_image':resources[original['boot_disk']]['sourceImageId']='999'
    elif fault=='later_attachment':resources[original['boot_disk']]['lastAttachTimestamp']='2026-09-19T16:00:01-07:00'
    elif fault=='gpu_machine':machine='g4-standard-48'
    output=tmp_path/'refused'
    with pytest.raises(ValueError):prepare.build_recovery(old,inputs[2],output,boot_id,'2',machine)
    assert not output.exists()


def test_recovery_execute_rechecks_disk_attachment_history(failed_attempt,inputs,tmp_path):
    old,original,resources,operation,calls=failed_attempt
    recovered=prepare.build_recovery(old,inputs[2],tmp_path/'recovery','1','2')
    # Simulate a new API response, not mutation of the retained evidence object.
    disk=dict(resources[original['input_disk']]);disk['lastDetachTimestamp']='2026-09-19T16:03:00-07:00'
    resources[original['input_disk']]=disk
    with pytest.raises(ValueError,match='attached'):prepare.recovery_resources(recovered)


def test_run_surfaces_captured_cloud_failure_stderr(monkeypatch):
    class Result:
        returncode=1;stdout='';stderr='ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS: no CPUs'
    monkeypatch.setattr(prepare.subprocess,'run',lambda *a,**kw:Result())
    with pytest.raises(RuntimeError,match='ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS'):
        prepare.run(['gcloud','compute','instances','describe','test'])


def test_guest_rejects_wrong_reviewed_cpu_size(monkeypatch):
    monkeypatch.setattr(bootstrap,'metadata',lambda _:b'zones/test/machineTypes/n4-standard-4')
    monkeypatch.setattr(bootstrap,'mount_inputs',lambda:pytest.fail('Touched disk before machine check'))
    with pytest.raises(ValueError):bootstrap.prepare({'machine_type':'n4-standard-2'})
