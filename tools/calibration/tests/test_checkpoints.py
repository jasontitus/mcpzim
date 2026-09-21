import hashlib
import json
import random

import numpy as np
import pytest
import torch

from checkpoints import (CheckpointError, LocalCheckpointStore, REQUIRED_ROLES,
                         capture_rng_state, exclusive_store_lock, restore_rng_state)


@pytest.fixture
def identity():
    return {"baseline_repo": "Qwen/Qwen3.8-27B", "baseline_revision": "a" * 40,
            "solver_revision": "b" * 40, "calibration_sha256": "1" * 64,
            "solver_config_sha256": "2" * 64, "candidate_database_sha256": "3" * 64,
            "runtime_sha256": "4" * 64}


@pytest.fixture
def payloads(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    result = {}
    for role in REQUIRED_ROLES:
        result[role] = directory / role
        result[role].write_bytes(role.encode())
    return result


def test_publish_restore_and_idempotent_retry(tmp_path, identity, payloads):
    store = LocalCheckpointStore(tmp_path / "store")
    manifest = store.publish("step-0001", identity, payloads)
    assert store.publish("step-0001", identity, payloads) == manifest
    assert store.committed_snapshots() == ["step-0001"]
    restored = tmp_path / "restored"
    assert store.restore("step-0001", identity, restored) == manifest
    for role, source in payloads.items():
        assert (restored / role).read_bytes() == source.read_bytes()
    with pytest.raises(CheckpointError, match="already exists"):
        store.restore("step-0001", identity, restored)


def test_upload_interruption_does_not_publish_and_retry_succeeds(tmp_path, identity, payloads, monkeypatch):
    store = LocalCheckpointStore(tmp_path / "store")
    upload = store._upload
    count = 0

    def fail_second(source):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("spot disappeared")
        return upload(source)

    monkeypatch.setattr(store, "_upload", fail_second)
    with pytest.raises(OSError):
        store.publish("step-1", identity, payloads)
    assert store.committed_snapshots() == []
    assert list(store.objects.iterdir())  # orphan does not count as a commit
    monkeypatch.setattr(store, "_upload", upload)
    store.publish("step-1", identity, payloads)
    store.verify("step-1", identity)


@pytest.mark.parametrize("after_commit", [False, True])
def test_commit_interruption_is_old_or_complete(tmp_path, identity, payloads, monkeypatch, after_commit):
    store = LocalCheckpointStore(tmp_path / "store")
    store.publish("old", identity, payloads)
    install = store._install

    def interrupt(temp, destination):
        if destination.parent == store.commits:
            if after_commit:
                install(temp, destination)
            raise OSError("process lost")
        install(temp, destination)

    monkeypatch.setattr(store, "_install", interrupt)
    with pytest.raises(OSError):
        store.publish("new", identity, payloads)
    store.verify("old", identity)
    assert ("new" in store.committed_snapshots()) == after_commit
    if after_commit:
        store.verify("new", identity)


@pytest.mark.parametrize("remove", [False, True])
def test_corrupt_or_missing_payload_fails_before_restore_exposure(tmp_path, identity, payloads, remove):
    store = LocalCheckpointStore(tmp_path / "store")
    manifest = store.publish("step", identity, payloads)
    obj = store.objects / manifest["payloads"]["optimizer"]["sha256"]
    if remove:
        obj.unlink()
    else:
        obj.write_bytes(b"broken")
    with pytest.raises((CheckpointError, FileNotFoundError)):
        store.restore("step", identity, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


@pytest.mark.parametrize("field", ["baseline_revision", "solver_revision", "calibration_sha256",
                                   "solver_config_sha256", "candidate_database_sha256", "runtime_sha256"])
def test_wrong_run_identity_rejected(tmp_path, identity, payloads, field):
    store = LocalCheckpointStore(tmp_path / "store")
    store.publish("step", identity, payloads)
    wrong = {**identity, field: "c" * len(identity[field])}
    with pytest.raises(CheckpointError, match="identity"):
        store.restore("step", wrong, tmp_path / "restore")


def test_cannot_replace_commit(tmp_path, identity, payloads):
    store = LocalCheckpointStore(tmp_path / "store")
    previous = store.publish("step", identity, payloads)
    payloads["solver"].write_bytes(b"new weights")
    with pytest.raises(CheckpointError, match="Immutable"):
        store.publish("step", identity, payloads)
    assert store.verify("step", identity) == previous


def test_reject_incomplete_and_traversal(tmp_path, identity, payloads):
    store = LocalCheckpointStore(tmp_path / "store")
    with pytest.raises(CheckpointError):
        store.publish("../bad", identity, payloads)
    with pytest.raises(CheckpointError):
        store.publish("step", identity, {"solver": payloads["solver"]})
    with pytest.raises(CheckpointError):
        store.publish("step", identity, {**payloads, "../escape": payloads["solver"]})
    store.publish("step", identity, payloads)
    path = store.commits / "step.json"
    value = json.loads(path.read_text())
    value["payloads"]["solver"]["sha256"] = "../../elsewhere"
    path.write_text(json.dumps(value))
    with pytest.raises(CheckpointError):
        store.verify("step", identity)


def test_source_change_during_copy_rejected(tmp_path, identity, payloads, monkeypatch):
    import checkpoints
    store = LocalCheckpointStore(tmp_path / "store")
    original_copy = checkpoints.shutil.copyfileobj

    def mutate(source, target, **kwargs):
        original_copy(source, target, **kwargs)
        payloads["optimizer"].write_bytes(b"changed after copy")

    monkeypatch.setattr(checkpoints.shutil, "copyfileobj", mutate)
    with pytest.raises(CheckpointError, match="Source changed"):
        store._upload(payloads["optimizer"])


def test_restore_rechecks_payload_after_verification(tmp_path, identity, payloads, monkeypatch):
    store = LocalCheckpointStore(tmp_path / "store")
    manifest = store.publish("step", identity, payloads)
    original = store.verify

    def corrupt_after_verify(*args):
        result = original(*args)
        (store.objects / manifest["payloads"]["optimizer"]["sha256"]).write_bytes(b"corrupt")
        return result

    monkeypatch.setattr(store, "verify", corrupt_after_verify)
    with pytest.raises(CheckpointError, match="during restore"):
        store.restore("step", identity, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()
    assert list(tmp_path.glob(".restore-*")) == []


def test_resume_reproduces_next_adam_update_and_random_streams(tmp_path, identity):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    parameter = torch.nn.Parameter(torch.randn(4))
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    def update(param, opt, sched):
        target = torch.rand_like(param) + random.random() + float(np.random.random())
        opt.zero_grad()
        (param - target).square().mean().backward()
        opt.step()
        sched.step()
        return target

    for _ in range(3):
        update(parameter, optimizer, scheduler)
    sources = tmp_path / "sources"
    sources.mkdir()
    states = {"solver": {"parameter": parameter.detach()}, "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(), "rng": capture_rng_state(),
              "progress": {"completed_updates": 3, "next_update": 3, "boundary": "update_complete"}}
    payloads = {}
    for role, value in states.items():
        path = sources / role
        torch.save(value, path)
        payloads[role] = path
    store = LocalCheckpointStore(tmp_path / "durable")
    store.publish("step-3", identity, payloads)
    expected_target = update(parameter, optimizer, scheduler)
    expected_parameter = parameter.detach().clone()
    expected_moment = optimizer.state[parameter]["exp_avg"].clone()

    restored = tmp_path / "resume"
    store.restore("step-3", identity, restored)
    load = lambda role: torch.load(restored / role, weights_only=True)
    param2 = torch.nn.Parameter(load("solver")["parameter"])
    opt2 = torch.optim.Adam([param2], lr=0.05)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=2, gamma=0.8)
    opt2.load_state_dict(load("optimizer"))
    sched2.load_state_dict(load("scheduler"))
    assert load("progress")["next_update"] == 3
    restore_rng_state(load("rng"))  # last, after constructors and loads
    actual_target = update(param2, opt2, sched2)
    assert torch.equal(actual_target, expected_target)
    assert torch.equal(param2, expected_parameter)
    assert torch.equal(opt2.state[param2]["exp_avg"], expected_moment)
    assert sched2.state_dict() == scheduler.state_dict()


def test_rng_refuses_changed_cuda_topology(monkeypatch):
    state = capture_rng_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(state["torch_cuda"]) + 1)
    with pytest.raises(CheckpointError, match="device count"):
        restore_rng_state(state)


def test_mps_rng_state_round_trips_exactly():
    """A run that used MPS records its generator, and that state replays bit-exactly.

    This test previously asserted the opposite — that restore *refuses* an MPS
    state — on the belief that MPS replay was unproven. Measurement showed the
    belief was wrong: `torch.mps.get_rng_state`/`set_rng_state` round-trip
    exactly (capture -> draw -> restore -> draw reproduces identical values), and
    `torch.manual_seed` does position the MPS generator. Refusing the restore
    made **every** checkpoint from an MPS run unresumable, which is worse than
    the misalignment it was guarding against.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    assert "mps" not in capture_rng_state(), "a CPU-only run must not record MPS state"
    tensor = torch.ones(1024, 1024, device="mps")
    assert tensor.sum().item() == 1024 * 1024
    torch.mps.synchronize()
    _ = torch.rand(4, device="mps")  # advance the generator off its start state
    state = capture_rng_state()
    assert state["mps"].dtype == torch.uint8 and state["mps"].numel() > 0

    expected = torch.rand(6, device="mps").cpu()
    restore_rng_state(state)
    assert torch.equal(torch.rand(6, device="mps").cpu(), expected), (
        "MPS RNG state did not replay exactly; exact resume is unsound on this build"
    )


def test_cpu_only_run_checkpoint_round_trips_on_an_mps_capable_machine():
    """Availability alone must not make a CPU checkpoint unresumable."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    # A pure-CPU run must not *require* MPS state to restore. It may carry an MPS
    # entry if MPS memory happens to be live (an unrelated earlier allocation is
    # enough to trigger capture), so the assertion is on the restore succeeding,
    # not on the key being absent.
    state = capture_rng_state()
    random.seed(11)
    expected = (random.random(), torch.rand(3))
    restore_rng_state(state)
    random.seed(11)
    assert random.random() == expected[0]
    torch.testing.assert_close(torch.rand(3), expected[1], rtol=0, atol=0)


def test_store_lock_excludes_a_second_writer_and_prune(tmp_path, identity, payloads):
    # publish installs objects before its manifest, so a concurrent reclamation
    # could delete objects the newest checkpoint is about to reference. Both
    # sides take the same lock; the loser refuses instead of proceeding.
    from solver.tools.prune_store import prune
    store = LocalCheckpointStore(tmp_path / "store")
    store.publish("step-0001", identity, payloads)
    with exclusive_store_lock(store.root, timeout_seconds=0.3):
        with pytest.raises(CheckpointError, match="held by another process"):
            store.publish("step-0002", identity, payloads, timeout_seconds=0.3)
        with pytest.raises(CheckpointError, match="held by another process"):
            prune(store.root, apply=True, keep=1, timeout_seconds=0.3)
        with pytest.raises(CheckpointError, match="held by another process"):
            store.restore("step-0001", identity, tmp_path / "restored", timeout_seconds=0.3)


def test_prune_spares_in_flight_uploads_and_keeps_the_newest_chain(tmp_path, identity, payloads):
    from solver.tools.prune_store import prune
    store = LocalCheckpointStore(tmp_path / "store")
    for tag in ("first", "second"):
        variant = {}
        for role in REQUIRED_ROLES:
            path = tmp_path / f"{tag}-{role}"
            path.write_bytes(tag.encode() + role.encode())
            variant[role] = path
        store.publish(f"step-{tag}", identity, variant)
    orphan = store.objects / ".upload-in-flight"
    orphan.write_bytes(b"partial copy")
    before = {p.name for p in store.objects.iterdir()}
    assert prune(store.root, apply=True, keep=1) == 0
    after = {p.name for p in store.objects.iterdir()}
    assert ".upload-in-flight" in after
    assert len(before) - len(after) == len(REQUIRED_ROLES)
    store.verify("step-second", identity)
    with pytest.raises(CheckpointError):
        store.verify("step-first", identity)
