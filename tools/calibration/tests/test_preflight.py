import argparse
import json
import tarfile
import io
from types import SimpleNamespace

import pytest

import cuda_preflight
import gcp_preflight


def test_prepare_uploads_only_harness_and_never_calls_cloud(tmp_path, monkeypatch):
    monkeypatch.setattr(gcp_preflight, "call", lambda *a, **k: pytest.fail("Preparation contacted cloud"))
    manifest = gcp_preflight.prepare(tmp_path)
    assert manifest["status"] == "prepared_not_launched"
    with tarfile.open(tmp_path / "harness.tgz") as archive:
        assert set(archive.getnames()) == set(gcp_preflight.FILES)
    with pytest.raises(ValueError): gcp_preflight.prepare(tmp_path)


def test_launch_is_one_spot_vm_with_server_deadline():
    argv = gcp_preflight.create_command("test-vm", "test-token")
    for flag in ("--provisioning-model=SPOT", "--max-run-duration=1h",
                 "--instance-termination-action=DELETE", "--boot-disk-auto-delete",
                 "--no-restart-on-failure", "--no-service-account", "--no-scopes",
                 "--machine-type=g2-standard-8", "--project=tiltastech-zimfo"):
        assert flag in argv


def test_cleanup_refuses_unrelated_instance(monkeypatch):
    monkeypatch.setattr(gcp_preflight, "describe", lambda name: {"labels": {"run-token": "someone-else"}})
    monkeypatch.setattr(gcp_preflight, "call", lambda *a, **k: pytest.fail("Deleted unrelated VM"))
    with pytest.raises(RuntimeError, match="ownership"):
        gcp_preflight.delete_owned("vm", "ours")


def test_failed_create_still_attempts_owned_cleanup(tmp_path, monkeypatch):
    # Isolate cleanup semantics; the public launcher remains disabled until
    # the full-model readiness conditions are actually implemented and met.
    monkeypatch.setattr(gcp_preflight, "require_launch_ready", lambda: None)
    manifest = gcp_preflight.prepare(tmp_path)
    monkeypatch.setattr(gcp_preflight, "describe", lambda name: None)
    monkeypatch.setattr(gcp_preflight, "call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API timeout")))
    cleaned = []
    monkeypatch.setattr(gcp_preflight, "delete_owned", lambda n, o: cleaned.append(n) or "absent")
    with pytest.raises(RuntimeError, match="API timeout"):
        gcp_preflight.execute(tmp_path, manifest)
    assert cleaned == [manifest["instance"]]
    result = json.loads((tmp_path / "launch.json").read_text())
    assert result["status"] == "failed_or_interrupted"
    assert result["cleanup"] == "absent"


def test_unready_launch_cannot_contact_cloud(tmp_path, monkeypatch):
    manifest = gcp_preflight.prepare(tmp_path)
    monkeypatch.setattr(gcp_preflight, "describe", lambda *a: pytest.fail("Cloud contacted before ready"))
    monkeypatch.setattr(gcp_preflight, "call", lambda *a, **k: pytest.fail("Cloud contacted before ready"))
    with pytest.raises(RuntimeError, match="Paid launch disabled"):
        gcp_preflight.execute(tmp_path, manifest)
    assert json.loads((tmp_path / "launch.json").read_text())["status"] == "prepared_not_launched"


def test_target_cannot_pass_using_cpu(tmp_path):
    args = argparse.Namespace(profile="target", device="cpu", output=str(tmp_path / "result.json"))
    with pytest.raises(ValueError, match="requires CUDA"):
        cuda_preflight.run(args)


def test_failure_report_is_not_success(tmp_path, monkeypatch):
    args = argparse.Namespace(profile="tiny", device="cpu", output=str(tmp_path / "result.json"))
    def fail(*a): raise AssertionError("bad gradient")
    monkeypatch.setattr(cuda_preflight, "parity", fail)
    with pytest.raises(AssertionError): cuda_preflight.run(args)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "failed"
    assert "bad gradient" in report["error"]


def test_nonfinite_cannot_pass_relative_error():
    import torch
    with pytest.raises(AssertionError):
        cuda_preflight.relative_error(torch.tensor(float("nan")), torch.tensor(1.))


def test_absent_vm_does_not_hide_leftover_disk(monkeypatch):
    monkeypatch.setattr(gcp_preflight, "describe", lambda name: None)
    monkeypatch.setattr(gcp_preflight, "call", lambda *a, **k: SimpleNamespace(returncode=0, stdout="{}", stderr=""))
    with pytest.raises(RuntimeError, match="boot disk"):
        gcp_preflight.delete_owned("vm", "ours")


@pytest.mark.parametrize("report", [
    {"status": "passed", "device": "cpu", "profile": "tiny"},
    {"status": "passed", "device": "cuda", "profile": "target", "checks": []},
    {"status": "running", "device": "cuda", "profile": "target"},
])
def test_incomplete_or_cpu_report_cannot_pass_launch(tmp_path, report):
    data = json.dumps(report).encode()
    with tarfile.open(tmp_path / "results.tgz", "w:gz") as archive:
        entry = tarfile.TarInfo("results/cuda.json"); entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    with pytest.raises(RuntimeError): gcp_preflight.validate_results(tmp_path / "results.tgz")
