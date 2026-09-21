import copy
import json
import hashlib
import io
from pathlib import Path
import pytest
from types import SimpleNamespace

from cloud_gpu.launch import create_command, validate_ready, prepare, execute
from cloud_gpu.bootstrap import solver_command, validate_solver_status


def receipt():
    base = "zimfo-prep-abcdef123456"
    return {"status": "prepared_disks_detached", "project": "tiltastech-zimfo", "zone": "us-central1-b",
            "run_id": base, "gpu_smoke_passed": False,
            "input_commit": {"sha256": "c" * 64}, "restore_validation": {"status": "validated", "input_commit_sha256": "c"*64},
            "production_image": "us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/runtime@sha256:" + "a"*64,
            "bucket": "tiltastech-zimfo-quantization-us-central1",
            **{key: {"name": base + suffix, "id": number,
                     "self_link": f"https://www.googleapis.com/compute/v1/projects/tiltastech-zimfo/zones/us-central1-b/disks/{base}{suffix}"}
               for key, suffix, number in (("boot_disk", "-boot", "1"), ("data_disk", "-inputs", "2"))}}


def test_spot_deadline_and_both_disks_preserved():
    args = create_command({"ready": receipt(), "run_id": "zimfo-gpu-123"}, Path("/tmp/plan"), "2026-09-20T00:00:00+00:00")
    assert "--provisioning-model=SPOT" in args
    assert "--termination-time=2026-09-20T00:00:00+00:00" in args
    assert "--instance-termination-action=DELETE" in args
    assert "--network-interface=network=default,subnet=default,nic-type=GVNIC" in args
    disks = [arg for arg in args if arg.startswith("--disk=")]
    assert len(disks) == 2 and all("auto-delete=no" in disk for disk in disks)


@pytest.mark.parametrize("change", ["mutable_image", "incomplete", "wrong_inputs", "wrong_disk_project"])
def test_bad_receipts_rejected(change):
    r = receipt()
    if change == "mutable_image": r["production_image"] = r["production_image"].split("@")[0] + ":latest"
    elif change == "incomplete": r["status"] = "preparing"
    elif change == "wrong_inputs": r["restore_validation"]["input_commit_sha256"] = "d"*64
    else: r["boot_disk"]["self_link"] = r["boot_disk"]["self_link"].replace("tiltastech-zimfo", "other")
    with pytest.raises(ValueError): validate_ready(r)


def test_changed_startup_fails_before_cloud_calls(tmp_path, monkeypatch):
    import cloud_gpu.launch as launch
    r = tmp_path / "ready.json"; r.write_text(json.dumps(receipt()))
    target = tmp_path / "plan"
    prepare(r, target)
    (target / "startup.sh").write_text("changed")
    monkeypatch.setattr(launch, "run", lambda _: pytest.fail("cloud call before local integrity check"))
    with pytest.raises(ValueError, match="Startup changed"): execute(target)


def test_no_pull_bound_job_and_durable_workdir():
    args = solver_command({"ready": receipt(), "run_id": "zimfo-gpu-123"})
    assert "--pull=never" in args
    assert args[args.index("--deadline-seconds")+1] == "2700"
    assert args[args.index("--output")+1].startswith("/mnt/zimfo-inputs/jobs/")
    assert args[args.index("--checkpoint-seconds")+1] == "120"
    assert "type=bind,src=/mnt/zimfo-inputs/prepared,dst=/mnt/zimfo-inputs/prepared,readonly" in args


def successful_status():
    commit = {"object": "runs/test/commit.json", "generation": "123", "bytes": 123, "sha256": "a"*64}
    stage = {"passed": True, "gradient_norm": 0.4, "loss": 0.2}
    return {"schema_version": 1, "status": "checkpointed", "smoke": {
        "passed": True, "model": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "sequence_tokens": 4096, "cuda": {"available": True, "device": "RTX PRO 6000", "capability": [12, 0],
            "total_memory_bytes": 96*1024**3, "peak_allocated_bytes": 80*1024**3, "peak_reserved_bytes": 85*1024**3},
        "linear_attention_gsq": stage.copy(), "full_attention_gsq": stage.copy(),
        "full_model_rco": {**stage, "full_vocabulary": True},
        "checkpoint_resume": {"passed": True, "next_update_matches": True, "snapshot": "s", "commit": commit}},
        "progress": {"phase": "gsq", "optimizer_updates": 3, "durable_checkpoint": {"snapshot": "s", "receipt": {"commit": commit}}}}


def test_status_requires_actual_proofs():
    status = successful_status()
    assert validate_solver_status(status, 4096) is status
    for invalid in (None, {}, {"schema_version": 1, "status": "failed"}):
        with pytest.raises(ValueError): validate_solver_status(invalid, 4096)
    for section, field, bad in (("full_model_rco", "full_vocabulary", False),
            ("checkpoint_resume", "next_update_matches", False),
            ("linear_attention_gsq", "gradient_norm", float("nan")),
            ("cuda", "available", False)):
        invalid = copy.deepcopy(status); invalid["smoke"][section][field] = bad
        with pytest.raises(ValueError): validate_solver_status(invalid, 4096)
    with pytest.raises(ValueError, match="largest"): validate_solver_status(status, 5000)
    invalid = copy.deepcopy(status); invalid["progress"]["optimizer_updates"] = 0
    with pytest.raises(ValueError): validate_solver_status(invalid, 4096)


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_finish_preserves_both_disks_and_checks_owner(tmp_path, monkeypatch, wrong_owner):
    import cloud_gpu.launch as launch
    import gcs_checkpoints
    r = tmp_path / "receipt.json"; r.write_text(json.dumps(receipt()))
    target = tmp_path / "plan"; config = prepare(r, target)
    plan_path = target / "plan.json"; plan = json.loads(plan_path.read_text()); plan["status"] = "launched"
    plan_path.write_text(json.dumps(plan))
    report = {"status": "failed", "run_id": config["run_id"], "config_sha256": config["config_sha256"]}
    blob = SimpleNamespace(size=200, generation="3", download_as_bytes=lambda **kw: json.dumps(report).encode())
    monkeypatch.setattr(gcs_checkpoints, "make_client", lambda *a, **k: SimpleNamespace(bucket=lambda _: SimpleNamespace(get_blob=lambda _: blob)))
    instance = {"status": "TERMINATED", "labels": {"zimfo-run": "other" if wrong_owner else config["run_id"],
        "zimfo-purpose": "quantization"}, "disks": [{"source": receipt()[role]["self_link"], "autoDelete": False}
                                                    for role in ("boot_disk", "data_disk")]}
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(instance)))
    calls = []
    def run(args):
        calls.append(args)
        if "delete" in args: return []
        key = "boot_disk" if args[4].endswith("-boot") else "data_disk"
        return {"id": receipt()[key]["id"], "users": []}
    monkeypatch.setattr(launch, "run", run)
    if wrong_owner:
        with pytest.raises(ValueError, match="owned"): launch.finish(target)
        assert not calls
    else:
        result = launch.finish(target)
        assert result["status"] == "finished_failed"
        assert "--keep-disks=all" in calls[0]
        assert len(result["retained_disks"]) == 2


@pytest.mark.parametrize("corrupt", [False, True])
def test_remote_commit_is_read_by_generation_and_verified(monkeypatch, corrupt):
    from cloud_gpu import bootstrap
    status = successful_status()
    payload = b'{"checkpoint":"complete"}'
    commit = {"object": "runs/test/commits/s.json", "generation": 123,
              "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    status["smoke"]["checkpoint_resume"]["commit"] = commit
    status["progress"]["durable_checkpoint"]["receipt"]["commit"] = commit
    monkeypatch.setattr(bootstrap, "metadata", lambda _: b'{"access_token":"test-only"}')
    calls = []
    def open_url(request, **kw):
        calls.append(request.full_url)
        return io.BytesIO(b"changed" if corrupt else payload)
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", open_url)
    config = {"run_id": "test", "ready": receipt()}
    if corrupt:
        with pytest.raises(ValueError, match="verified"): bootstrap.verify_remote_commits(config, status)
    else:
        bootstrap.verify_remote_commits(config, status)
    assert len(calls) == 1  # identical commit references do not cause tiny repeated reads
    assert "generation=123" in calls[0] and "ifGenerationMatch=123" in calls[0]


def test_progress_summary_is_bounded_and_does_not_disclose_raw_output(tmp_path):
    from cloud_gpu import bootstrap
    secret='DO_NOT_SERIALIZE_PROMPT_OR_TOKEN'
    (tmp_path/'status.json').write_text(json.dumps({'status':'running','error':secret,
        'progress':{'phase':'gsq','optimizer_updates':4,'secret':secret},'smoke':{'passed':True}}))
    (tmp_path/'solver.log').write_text(secret+'\n'+json.dumps({'stage':'gsq','global_step':5,'block':2,
        'last_loss':.3,'prompt':secret})+'\n'+json.dumps({'stage':[]})+'\n')
    summary=bootstrap.progress_summary(tmp_path,30,True)
    assert summary['phase']=='gsq' and summary['global_step']==5 and summary['block']==2
    assert summary['smoke_passed'] is True and secret not in json.dumps(summary)
    (tmp_path/'status.json').write_bytes(b'x'*(1024*1024+1))
    (tmp_path/'solver.log').write_bytes(b'x'*100000)
    summary=bootstrap.progress_summary(tmp_path,31,True)
    assert summary['log_bytes']==100000 and 'phase' not in summary
    assert len(json.dumps(summary))<2048


def test_timing_event_does_not_hide_latest_optimizer_cursor(tmp_path):
    from cloud_gpu import bootstrap
    events = [
        {"stage": "gsq", "global_step": 87, "block": 0, "sequence": 87, "last_loss": .2},
        {"stage": "gsq", "operation": "cache_propagation", "block": 0, "seconds": 1.3},
        {"stage": "gsq", "operation": "checkpoint_publish", "seconds": 400},
    ]
    (tmp_path / 'solver.log').write_text(''.join(json.dumps(item)+'\n' for item in events))
    summary = bootstrap.progress_summary(tmp_path, 2300, True)
    assert summary['global_step'] == 87 and summary['sequence'] == 87
    assert summary['block'] == 0 and summary['last_loss'] == .2
    assert 'operation' not in summary


def test_serial_summary_full_pipe_never_blocks_and_restores_flags():
    import os
    import fcntl
    import time
    from cloud_gpu import bootstrap
    readfd,writefd=os.pipe()
    try:
        original=fcntl.fcntl(writefd,fcntl.F_GETFL)
        fcntl.fcntl(writefd,fcntl.F_SETFL,original|os.O_NONBLOCK)
        while True:
            try:os.write(writefd,b'x'*4096)
            except BlockingIOError:break
        fcntl.fcntl(writefd,fcntl.F_SETFL,original)
        before=fcntl.fcntl(writefd,fcntl.F_GETFL)
        started=time.monotonic()
        assert bootstrap.serial_summary({'event':'test'},fd=writefd) is False
        assert time.monotonic()-started<.5
        assert fcntl.fcntl(writefd,fcntl.F_GETFL)==before
    finally:
        os.close(readfd);os.close(writefd)


def test_real_child_keeps_complete_log_and_emits_phase_before_exit(tmp_path):
    import sys
    from cloud_gpu import bootstrap
    script="""import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
(root/'status.json').write_text(json.dumps({'status':'running','progress':{'phase':'smoke_gsq'}}))
print('retained-only-secret',flush=True)
sys.stderr.write('stderr retained too\\n');sys.stderr.flush()
time.sleep(.15)
(root/'status.json').write_text(json.dumps({'status':'failed','progress':{'phase':'smoke_gsq'},'error':'private-error'}))
"""
    events=[]
    result=bootstrap.run_solver([sys.executable,'-c',script,str(tmp_path)],tmp_path,
                                timeout=3,poll_seconds=.02,heartbeat_seconds=.05,emit=events.append)
    assert result.returncode==0
    assert (tmp_path/'solver.log').read_text()=='retained-only-secret\nstderr retained too\n'
    assert any(e.get('phase')=='smoke_gsq' and e['process_running'] for e in events)
    assert events[-1]['process_running'] is False and events[-1]['job_status']=='failed'
    assert 'private-error' not in json.dumps(events) and 'retained-only-secret' not in json.dumps(events)
    assert len(events)<15


def test_real_child_timeout_terminates_cli_and_preserves_log(tmp_path):
    import os
    import sys
    import subprocess
    import time
    from cloud_gpu import bootstrap
    script="import os,sys,time;print(os.getpid(),flush=True);time.sleep(20)"
    started=time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        bootstrap.run_solver([sys.executable,'-c',script],tmp_path,
            timeout=.2,poll_seconds=.01,heartbeat_seconds=.05,emit=lambda _:None)
    assert time.monotonic()-started<2
    pid=int((tmp_path/'solver.log').read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)


@pytest.mark.parametrize('license',['NVIDIA','Dual MIT/GPL'])
def test_blackwell_requires_open_kernel_module(monkeypatch,license):
    from cloud_gpu import bootstrap
    calls=[]
    def command(args):
        calls.append(args)
        if args==['uname','-r']:return 'kernel'
        if args==['modinfo','-F','version','nvidia']:return '580.126'
        if args==['modinfo','-F','license','nvidia']:return license
        pytest.fail('Unexpected driver query')
    monkeypatch.setattr(bootstrap,'command',command)
    if license=='NVIDIA':
        with pytest.raises(ValueError,match='open NVIDIA'):
            bootstrap.verify_prepared_driver({'kernel':'kernel','driver':'580.126'})
    else:bootstrap.verify_prepared_driver({'kernel':'kernel','driver':'580.126'})
    assert len(calls)==3


def test_solver_timeout_always_stops_only_named_container_without_masking_error(tmp_path,monkeypatch):
    import subprocess
    from cloud_gpu import bootstrap
    ready=receipt();ready['runtime_proof']={'kernel':'kernel','driver':'580.126'}
    (tmp_path/'prepared').mkdir()
    (tmp_path/'prepared/restore-validation.json').write_text(json.dumps(ready['restore_validation']))
    monkeypatch.setattr(bootstrap,'MOUNT',tmp_path)
    monkeypatch.setattr(bootstrap,'metadata',lambda _:b'zones/test/machineTypes/g4-standard-48')
    def command(args):
        if args==['uname','-r']:return 'kernel'
        if args[:3]==['modinfo','-F','version']:return '580.126'
        if args[:3]==['modinfo','-F','license']:return 'Dual MIT/GPL'
        if args[0]=='blkid':return 'zimfo-inputs'
        if args[0]=='mount':return ''
        if args[:3]==['docker','image','inspect']:
            return json.dumps([{'RepoDigests':[ready['production_image']],'Architecture':'amd64'}])
        if args[0]=='nvidia-smi':return 'NVIDIA RTX PRO 6000, 96000, 580.126'
        pytest.fail(str(args))
    monkeypatch.setattr(bootstrap,'command',command)
    def timeout(*a,**kw):raise subprocess.TimeoutExpired(['solver-command'],3000)
    monkeypatch.setattr(bootstrap,'run_solver',timeout)
    stopped=[]
    def stop(args,**kw):
        stopped.append((args,kw))
        raise subprocess.TimeoutExpired(args,55)
    monkeypatch.setattr(bootstrap.subprocess,'run',stop)
    monkeypatch.setattr(bootstrap,'serial_summary',lambda _:False)
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        bootstrap.execute({'ready':ready,'run_id':'zimfo-gpu-unique'})
    assert failure.value.cmd==['solver-command']
    assert stopped[0][0]==['docker','stop','--time=45','zimfo-gpu-unique']
    assert stopped[0][1]['timeout']==55


def test_diagnostics_are_attached_before_launch_and_frozen(tmp_path, monkeypatch):
    import cloud_gpu.launch as launch
    r=tmp_path/'ready.json';r.write_text(json.dumps(receipt()))
    output=tmp_path/'plan';config=prepare(r,output)
    command=create_command(config,output,'2026-09-20T03:00:00+00:00')
    metadata=next(arg for arg in command if arg.startswith('--metadata-from-file='))
    assert 'shutdown-script='+str(output/'shutdown-diagnostics.sh') in metadata
    assert launch.sha(output/'shutdown-diagnostics.sh')==config['shutdown_sha256']
    (output/'shutdown-diagnostics.sh').write_text('changed')
    monkeypatch.setattr(launch,'run',lambda _:pytest.fail('cloud call before integrity check'))
    with pytest.raises(ValueError,match='Startup changed'):execute(output)
