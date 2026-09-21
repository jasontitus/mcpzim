import hashlib
import io
import json
import random
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from checkpoints import CheckpointError, REQUIRED_ROLES
from gcs_checkpoints import GCSCheckpointStore, GoogleStorageBackend


class MemoryBackend:
    def __init__(self):
        self.objects = {}
        self.generation = 0
        self.opened = []

    def create(self, key, stream, size):
        if key not in self.objects:
            self.generation += 1
            content = stream.read()
            assert len(content) == size
            self.objects[key] = (self.generation, content)
        return self.stat(key)

    def stat(self, key, generation=None):
        actual, content = self.objects[key]
        if generation is not None and actual != generation:
            raise FileNotFoundError("Missing requested generation")
        return {"object": key, "generation": actual, "bytes": len(content)}

    def open(self, key, generation):
        self.stat(key, generation)
        self.opened.append((key, generation))
        return io.BytesIO(self.objects[key][1])

    def list(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]


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


@pytest.fixture
def store():
    return GCSCheckpointStore("test-bucket", "tests/run", backend=MemoryBackend())


def test_generation_bound_roundtrip_and_retry(store, identity, payloads, tmp_path):
    receipt = store.publish("step-1", identity, payloads)
    assert store.publish("step-1", identity, payloads) == receipt
    assert store.committed_snapshots() == ["step-1"]
    result = store.restore("step-1", identity, tmp_path / "restored", commit_generation=receipt["commit"]["generation"])
    assert result == receipt
    for role, source in payloads.items():
        assert (tmp_path / "restored" / role).read_bytes() == source.read_bytes()
    assert all(generation > 0 for _, generation in store.backend.opened)
    assert len(store.backend.objects) == 2  # one bulk state object plus one commit


def test_immutable_extra_archive_reused_without_upload(store, identity, payloads, tmp_path, monkeypatch):
    archive = tmp_path / "completed-block.tar"
    archive.write_bytes(b"frozen completed block archive")
    payloads["candidate_block_0"] = archive
    first = store.publish("step-1", identity, payloads)
    extra = first["manifest"]["payloads"]["candidate_block_0"]
    original = store.backend.create
    writes = []

    def record(key, *args):
        writes.append(key)
        return original(key, *args)

    monkeypatch.setattr(store.backend, "create", record)
    payloads["progress"].write_bytes(b"next optimizer update")
    second = store.publish("step-2", identity, payloads)
    assert second["manifest"]["payloads"]["candidate_block_0"] == extra
    assert extra["object"] not in writes
    assert len(writes) == 2
    store.restore("step-2", identity, tmp_path / "restore")
    assert (tmp_path / "restore/candidate_block_0").read_bytes() == archive.read_bytes()


@pytest.mark.parametrize("fault", ["traversal", "symlink", "duplicate", "wrong_content"])
def test_bundle_extraction_rejects_unsafe_members(store, identity, payloads, tmp_path, fault):
    receipt = store.publish("step", identity, payloads)
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        for role in sorted(REQUIRED_ROLES):
            data = payloads[role].read_bytes()
            name = "../escape" if fault == "traversal" and role == "solver" else role
            info = tarfile.TarInfo(name)
            info.size = len(data)
            if fault == "symlink" and role == "solver":
                info.type, info.linkname, info.size = tarfile.SYMTYPE, "/tmp/escape", 0
                archive.addfile(info)
            else:
                if fault == "wrong_content" and role == "solver":
                    data = b"x" * len(data)
                archive.addfile(info, io.BytesIO(data))
                if fault == "duplicate" and role == "solver":
                    archive.addfile(info, io.BytesIO(data))
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(CheckpointError):
        store._extract_bundle(archive_path, receipt["manifest"]["payloads"], staging)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("when", ["payload", "before_commit", "after_commit"])
def test_interrupted_write_retry_is_old_or_complete(store, identity, payloads, monkeypatch, when):
    store.publish("old", identity, payloads)
    original = store.backend.create

    def interrupted(key, stream, size):
        if ((when == "payload" and "/objects/" in key)
                or (when in ("before_commit", "after_commit") and "/commits/new.json" in key)):
            if when == "after_commit":
                original(key, stream, size)
            raise ConnectionError("Spot process/network vanished")
        return original(key, stream, size)

    monkeypatch.setattr(store.backend, "create", interrupted)
    with pytest.raises(ConnectionError):
        store.publish("new", identity, payloads)
    assert ("new" in store.committed_snapshots()) == (when == "after_commit")
    store.verify("old", identity)
    monkeypatch.setattr(store.backend, "create", original)
    store.publish("new", identity, payloads)
    store.verify("new", identity)


def test_conflicting_existing_commit_never_overwritten(store, identity, payloads):
    receipt = store.publish("step", identity, payloads)
    payloads["solver"].write_bytes(b"other solver state")
    with pytest.raises(CheckpointError):
        store.publish("step", identity, payloads)
    assert store.verify("step", identity) == receipt


def test_existing_hash_object_is_not_trusted(store, identity, payloads):
    receipt = store.publish("original", identity, payloads)
    entry = receipt["manifest"]["state_bundle"]
    store.backend.objects[entry["object"]] = (entry["generation"], b"x" * entry["bytes"])
    with pytest.raises(CheckpointError, match="checksum"):
        store.publish("step", identity, payloads)
    assert store.committed_snapshots() == ["original"]


def test_source_changed_then_restored_while_bundling_cannot_commit(store, identity, payloads, monkeypatch):
    original = tarfile.TarFile.addfile

    def mutated(archive, info, stream=None):
        if info.name == "solver":
            # Producer's source hashes before/after match, but actual read bytes
            # differ. The completed archive member hash must catch this.
            return original(archive, info, io.BytesIO(b"x" * info.size))
        return original(archive, info, stream)

    monkeypatch.setattr(tarfile.TarFile, "addfile", mutated)
    with pytest.raises(CheckpointError, match="member checksum"):
        store.publish("step", identity, payloads)
    assert store.backend.objects == {}


def test_manifest_payloads_must_be_object(store, identity, payloads):
    receipt = store.publish("step", identity, payloads)
    manifest = receipt["manifest"]
    manifest["payloads"] = list(manifest["payloads"])
    key = receipt["commit"]["object"]
    store.backend.objects[key] = (receipt["commit"]["generation"], json.dumps(manifest).encode())
    with pytest.raises(CheckpointError, match="state roles"):
        store.verify("step", identity)


@pytest.mark.parametrize("fault", ["wrong_generation", "missing", "corrupt", "wrong_identity", "wrong_commit_generation"])
def test_restore_fails_without_exposing_partial_destination(store, identity, payloads, tmp_path, fault):
    receipt = store.publish("step", identity, payloads)
    entry = receipt["manifest"]["state_bundle"]
    key = entry["object"]
    target_identity = identity
    generation = receipt["commit"]["generation"]
    if fault == "wrong_generation":
        store.backend.objects[key] = (1000, store.backend.objects[key][1])
    elif fault == "missing":
        del store.backend.objects[key]
    elif fault == "corrupt":
        store.backend.objects[key] = (entry["generation"], b"x" * entry["bytes"])
    elif fault == "wrong_identity":
        target_identity = {**identity, "calibration_sha256": "e" * 64}
    else:
        generation += 1
    with pytest.raises((CheckpointError, FileNotFoundError, KeyError)):
        store.restore("step", target_identity, tmp_path / "restore", commit_generation=generation)
    assert not (tmp_path / "restore").exists()
    assert not list(tmp_path.glob(".gcs-restore-*"))


def test_restore_network_failure_cleans_staging(store, identity, payloads, tmp_path, monkeypatch):
    store.publish("step", identity, payloads)
    original = store.backend.open

    class BrokenReader(io.BytesIO):
        def read(self, size=-1):
            raise ConnectionError("interrupted range download")

    def open_(key, generation):
        return BrokenReader() if "/objects/" in key else original(key, generation)

    monkeypatch.setattr(store.backend, "open", open_)
    with pytest.raises(ConnectionError):
        store.restore("step", identity, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()
    assert not list(tmp_path.glob(".gcs-restore-*"))


def test_scan_uses_bounded_reads(store, identity, payloads, monkeypatch, tmp_path):
    import gcs_checkpoints
    monkeypatch.setattr(gcs_checkpoints, "CHUNK_BYTES", 4)
    original = store.backend.open

    class LimitedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 4
            return super().read(size)

    monkeypatch.setattr(store.backend, "open", lambda key, generation: LimitedReader(original(key, generation).read()))
    store.publish("step", identity, payloads)
    store.restore("step", identity, tmp_path / "restore")


def test_payload_replaced_after_commit_metadata_read_does_not_mix_generations(store, identity, payloads, tmp_path, monkeypatch):
    receipt = store.publish("step", identity, payloads)
    original = store.backend.stat
    payload = receipt["manifest"]["state_bundle"]

    def mutate(key, generation=None):
        result = original(key, generation)
        if "/commits/" in key:
            store.backend.objects[payload["object"]] = (999, b"changed")
        return result

    monkeypatch.setattr(store.backend, "stat", mutate)
    with pytest.raises(FileNotFoundError):
        store.restore("step", identity, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


def test_sdk_boundary_requires_create_only_and_pinned_download():
    from google.api_core.exceptions import PreconditionFailed
    calls = []

    class FakeBlob:
        generation, size = 7, 4

        def upload_from_file(self, stream, **kwargs):
            assert kwargs["if_generation_match"] == 0
            assert kwargs["checksum"] == "crc32c"
            calls.append("create")
            raise PreconditionFailed("already present")

        def reload(self, **kwargs):
            calls.append(("stat", kwargs["if_generation_match"]))

        def open(self, mode, **kwargs):
            assert kwargs["if_generation_match"] == 7
            assert kwargs["raw_download"] is True
            assert kwargs["chunk_size"] == 64 * 1024 * 1024
            calls.append("read")
            return io.BytesIO(b"data")

    class Bucket:
        def blob(self, key, **kwargs):
            calls.append((key, kwargs))
            return FakeBlob()

    class Client:
        def bucket(self, _):
            return Bucket()

    backend = GoogleStorageBackend("bucket", client=Client())
    assert backend.create("key", io.BytesIO(b"data"), 4)["generation"] == 7
    with backend.open("key", 7) as stream:
        assert stream.read() == b"data"
    assert ("key", {"generation": 7}) in calls
    assert ("key", {"chunk_size": 64 * 1024 * 1024}) in calls


def test_network_retry_jitter_does_not_change_solver_rng(store, identity, payloads, monkeypatch, tmp_path):
    original_create, original_open = store.backend.create, store.backend.open

    def create(*args):
        random.random()
        return original_create(*args)

    def open_(*args):
        random.random()
        return original_open(*args)

    monkeypatch.setattr(store.backend, "create", create)
    monkeypatch.setattr(store.backend, "open", open_)
    state = random.getstate()
    store.publish("step", identity, payloads)
    assert random.getstate() == state


    store.restore("step", identity, tmp_path / "restore")
    assert random.getstate() == state

    def failed(*args):
        random.random()
        raise ConnectionError("network stopped")

    monkeypatch.setattr(store.backend, "create", failed)
    with pytest.raises(ConnectionError):
        store.publish("failed", identity, payloads)
    assert random.getstate() == state


def test_background_checkpointing_is_rejected_without_rng_rewind(store, identity, payloads):
    state = random.getstate()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.publish, "step", identity, payloads)
        with pytest.raises(CheckpointError, match="synchronous"):
            future.result()
    assert random.getstate() == state
    assert store.backend.objects == {}


@pytest.mark.parametrize("prefix", ["", "/root", "a/../b", "a//b", "a/", "a/."])
def test_invalid_run_prefix_rejected(prefix):
    with pytest.raises(CheckpointError):
        GCSCheckpointStore("bucket", prefix, backend=MemoryBackend())
