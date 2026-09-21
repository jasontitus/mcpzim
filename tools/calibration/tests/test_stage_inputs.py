import hashlib
from pathlib import Path
import tarfile

import pytest
from google.api_core.exceptions import PreconditionFailed

from stage_inputs import fingerprint, pack, upload, verify_archive


def test_archive_reproducible_and_detects_changed_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "tensor").write_bytes(b"original")
    expected = {"tensor": hashlib.sha256(b"original").hexdigest()}
    a, b = tmp_path / "a.tar", tmp_path / "b.tar"
    pack(source, expected, a)
    pack(source, expected, b)
    assert fingerprint(a) == fingerprint(b)
    verify_archive(a, expected)
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(a, {"tensor": "0" * 64})
    with pytest.raises(FileExistsError):
        pack(source, expected, a)


def test_archive_rejects_traversal_links_and_extra_members(tmp_path):
    (tmp_path / "data").write_bytes(b"data")
    (tmp_path / "link").symlink_to(tmp_path / "data")
    for names in [["../data"], [str(tmp_path / "data")], ["link"]]:
        with pytest.raises(ValueError):
            pack(tmp_path, names, tmp_path / "bad.tar")
    archive = tmp_path / "extra.tar"
    pack(tmp_path, ["data"], archive)
    with pytest.raises(ValueError, match="Unexpected"):
        verify_archive(archive, {})
    with pytest.raises(ValueError, match="Missing"):
        verify_archive(archive, {"data": fingerprint(tmp_path / "data")["sha256"], "missing": "0"*64})


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name, self.metadata = store, name, None

    def upload_from_filename(self, filename, **kwargs):
        assert kwargs["if_generation_match"] == 0
        assert kwargs["checksum"] == "crc32c"
        if self.name in self.store:
            raise PreconditionFailed("exists")
        info = fingerprint(filename)
        self.store[self.name] = {"size": info["bytes"], "crc32c": info["crc32c"],
                                 "generation": 1, "metadata": self.metadata}

    def reload(self):
        for k, v in self.store[self.name].items():
            setattr(self, k, v)


class FakeBucket:
    def __init__(self):
        self.store = {}

    def blob(self, name, **kwargs):
        return FakeBlob(self.store, name)


def test_immutable_upload_retries_and_remote_conflict(tmp_path):
    source = tmp_path / "weights"
    source.write_bytes(b"verified data")
    bucket = FakeBucket()
    result = upload(bucket, "immutable", source, fingerprint(source))
    assert upload(bucket, "immutable", source) == result
    assert result["generation"] == "1"
    bucket.store["immutable"]["crc32c"] = "corrupt"
    with pytest.raises(ValueError, match="Remote"):
        upload(bucket, "immutable", source)


def test_local_hash_mismatch_rejected_before_upload(tmp_path):
    source = tmp_path / "weights"
    source.write_bytes(b"altered")
    bucket = FakeBucket()
    with pytest.raises(ValueError, match="Local"):
        upload(bucket, "immutable", source, {"bytes": 7, "sha256": "0"*64})
    assert not bucket.store
