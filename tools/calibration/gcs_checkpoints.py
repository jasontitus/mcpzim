"""Durable GCS checkpoint transport, manifest last and generation pinned.

Requires google-cloud-storage==3.4.1. Default credentials come from the host or
VM identity; no keys or tokens are serialized. The store never creates buckets,
changes IAM, launches compute, or deletes objects. Source files must be frozen.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import functools
from contextlib import nullcontext
from pathlib import Path
import random
import shutil
import tarfile
import tempfile
import threading

from checkpoints import (CHUNK_BYTES, MAX_MANIFEST_BYTES, REQUIRED_ROLES, HASH,
                         CheckpointError, canonical_json, digest_file,
                         _identity, _name, _sync_directory)


# Network range size is independent of the small local hash/extraction buffer.
# A1MiB remote range caused minutes of idle GPU time per checkpoint in the
# first full-model run.64MiB remains bounded while amortizing request latency.
GCS_TRANSFER_BYTES = 64 * 1024 * 1024


def _preserve_python_rng(function):
    """SDK retry jitter must not perturb the solver's next random draw.

    Checkpoint calls are synchronous at a solver boundary. This is not suitable
    for background checkpoint threads sharing the process-global training RNG.
    """
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise CheckpointError("Checkpoint I/O requires a synchronous main-thread optimizer boundary; background uploads are unsupported")
        state = random.getstate()
        try:
            return function(*args, **kwargs)
        finally:
            random.setstate(state)
    return wrapped


def make_client(project, *, use_gcloud=False):
    """ADC on compute; opt-in refreshable gcloud login on the preparation Mac.

    The access token travels only through a captured pipe into client memory.
    Never persist this object, print its token, or include subprocess stderr in
    public logs. A worker VM should use its restricted service account via ADC.
    """
    from google.cloud import storage
    if not use_gcloud:
        return storage.Client(project=project)
    import datetime
    import subprocess
    from google.auth.credentials import Credentials
    from google.auth.exceptions import RefreshError

    class GcloudCredentials(Credentials):
        def refresh(self, request):
            result = subprocess.run(["gcloud", "auth", "print-access-token", "--quiet",
                                     f"--project={project}"], capture_output=True, text=True, timeout=90)
            token = result.stdout.strip()
            if result.returncode != 0 or not token or any(c.isspace() for c in token):
                raise RefreshError("Existing gcloud login did not supply an access token")
            self.token = token
            # gcloud may reuse a token with less remaining life; an HTTP 401 also
            # triggers the SDK's normal refresh. Do not log/export token contents.
            self.expiry = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(minutes=40)

    return storage.Client(project=project, credentials=GcloudCredentials())


class GoogleStorageBackend:
    """Small SDK boundary, also injectable for fault tests without cloud access."""
    def __init__(self, bucket_name, project=None, client=None):
        from google.cloud import storage
        from google.cloud.storage.retry import DEFAULT_RETRY
        self.client = client if client is not None else storage.Client(project=project)
        self.bucket = self.client.bucket(bucket_name)
        self.retry = DEFAULT_RETRY.with_timeout(120)

    def stat(self, key, generation=None):
        blob = self.bucket.blob(key, generation=generation)
        blob.reload(if_generation_match=generation, retry=self.retry, timeout=60)
        return {"object": key, "generation": int(blob.generation), "bytes": int(blob.size)}

    def create(self, key, stream, size):
        from google.api_core.exceptions import PreconditionFailed
        blob = self.bucket.blob(key, chunk_size=GCS_TRANSFER_BYTES)
        try:
            # The zero precondition makes retries safe. Uploads larger than the
            # SDK's multipart threshold use bounded64MiB resumable chunks.
            blob.upload_from_file(stream, rewind=True, size=size,
                                  if_generation_match=0, checksum="crc32c",
                                  retry=self.retry, timeout=60)
            return {"object": key, "generation": int(blob.generation), "bytes": int(blob.size)}
        except PreconditionFailed:
            # Existing data is NOT assumed equal; the store downloads and hashes
            # this generation before accepting an idempotent retry.
            return self.stat(key)

    def open(self, key, generation):
        return self.bucket.blob(key, generation=generation).open(
            "rb", chunk_size=GCS_TRANSFER_BYTES, if_generation_match=generation,
            retry=self.retry, timeout=60, raw_download=True)

    def list(self, prefix):
        return (blob.name for blob in self.client.list_blobs(self.bucket, prefix=prefix))


class GCSCheckpointStore:
    def __init__(self, bucket_name, prefix, *, project=None, client=None, backend=None, staging_dir=None):
        if (not isinstance(prefix, str) or not prefix or prefix.startswith("/")
                or any(part in ("", ".", "..") for part in prefix.split("/"))):
            raise CheckpointError("A nonempty safe run prefix is required")
        for part in prefix.split("/"):
            _name(part)
        if not isinstance(bucket_name, str) or not bucket_name or "/" in bucket_name:
            raise CheckpointError("Invalid bucket name")
        self.bucket_name, self.prefix = bucket_name, prefix
        self.staging_dir = staging_dir
        self.backend = backend if backend is not None else GoogleStorageBackend(bucket_name, project, client)
        self._verified_sources = {}

    def _scan(self, entry, target=None, expected_digest=None):
        """Read one immutable generation, enforcing size and SHA-256 ourselves.

        Range reads cannot rely on the SDK's whole-object checksum machinery.
        A bounded scan independently verifies every byte before publication/use.
        """
        digest, size = hashlib.sha256(), 0
        with self.backend.open(entry["object"], entry["generation"]) as stream:
            while chunk := stream.read(CHUNK_BYTES):
                size += len(chunk)
                if size > entry["bytes"]:
                    raise CheckpointError("Remote object exceeds declared size")
                digest.update(chunk)
                if target is not None:
                    target.write(chunk)
        if size != entry["bytes"] or (expected_digest is not None and digest.hexdigest() != expected_digest):
            raise CheckpointError("Remote object size/checksum mismatch")
        return {**entry, "sha256": digest.hexdigest()}

    def _put(self, key, stream, size, digest):
        entry = self.backend.create(key, stream, size)
        if entry["object"] != key or entry["bytes"] != size or type(entry["generation"]) is not int or entry["generation"] <= 0:
            raise CheckpointError("Remote immutable object conflict")
        return self._scan(entry, expected_digest=digest)

    def _payload(self, source):
        source = Path(source).resolve()
        stat = source.stat()
        fingerprint = (str(source), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._verified_sources.get(fingerprint)
        if cached is not None:
            # GCS generations have immutable bytes. Reuse only a previously
            # SHA-verified generation and an unchanged caller-owned frozen file;
            # metadata check confirms it still exists. No repeated archive writes.
            current = self.backend.stat(cached["object"], cached["generation"])
            if current != {key: cached[key] for key in ("object", "generation", "bytes")}:
                raise CheckpointError("Previously verified archive generation changed")
            return cached
        digest, size = digest_file(source)
        with source.open("rb") as stream:
            result = self._put(f"{self.prefix}/objects/{digest}", stream, size, digest)
        if digest_file(source) != (digest, size):
            raise CheckpointError("Source changed while uploading checkpoint")
        self._verified_sources[fingerprint] = result
        return result

    @_preserve_python_rng
    def publish(self, snapshot, identity, payloads):
        _name(snapshot)
        identity = _identity(identity)
        if not REQUIRED_ROLES <= set(payloads) or len(payloads) > 256:
            raise CheckpointError("Missing required state roles or too many payloads")
        for role, path in payloads.items():
            _name(role)
            if not Path(path).is_file():
                raise CheckpointError(f"Missing payload: {role}")
        # Batch the small state roles into one deterministic object. Completed
        # candidate/cache archives supplied as extra roles remain separate and
        # content-addressed, allowing reuse across later checkpoints.
        with tempfile.TemporaryDirectory(prefix="gcs-checkpoint-", dir=self.staging_dir) as directory:
            archive_path = Path(directory) / "state.tar"
            members = {}
            with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
                for role in sorted(REQUIRED_ROLES):
                    source = Path(payloads[role])
                    digest, size = digest_file(source)
                    info = tarfile.TarInfo(role)
                    info.size, info.mode, info.mtime = size, 0o600, 0
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
                    if digest_file(source) != (digest, size):
                        raise CheckpointError("Source changed while bundling checkpoint")
                    members[role] = {"member": role, "sha256": digest, "bytes": size}
            # Check the bytes actually copied into the tar, not just the source
            # before/after: this also catches a changed-then-restored source.
            self._extract_bundle(archive_path, members)
            bundle = self._payload(archive_path)
        manifest = {"schema": 3, "snapshot": snapshot, "identity": identity,
                    "bucket": self.bucket_name, "prefix": self.prefix, "state_bundle": bundle,
                    "payloads": {**members, **{role: self._payload(path)
                                 for role, path in sorted(payloads.items()) if role not in REQUIRED_ROLES}}}
        encoded = canonical_json(manifest)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise CheckpointError("Manifest too large")
        # No commit exists until every payload's *remote* bytes passed SHA-256.
        commit = self._put(f"{self.prefix}/commits/{snapshot}.json", io.BytesIO(encoded),
                           len(encoded), hashlib.sha256(encoded).hexdigest())
        return {"manifest": manifest, "commit": commit}

    @_preserve_python_rng
    def committed_snapshots(self):
        prefix = f"{self.prefix}/commits/"
        result = []
        for key in self.backend.list(prefix):
            suffix = key.removeprefix(prefix)
            if key.startswith(prefix) and suffix.endswith(".json") and "/" not in suffix:
                result.append(_name(suffix[:-5]))
        return sorted(result)

    def _manifest(self, snapshot, expected_identity, generation=None):
        _name(snapshot)
        expected_identity = _identity(expected_identity)
        if generation is not None and (type(generation) is not int or generation <= 0):
            raise CheckpointError("Invalid commit generation")
        entry = self.backend.stat(f"{self.prefix}/commits/{snapshot}.json", generation)
        if (type(entry["generation"]) is not int or entry["generation"] <= 0
                or (generation is not None and entry["generation"] != generation)
                or not 0 < entry["bytes"] <= MAX_MANIFEST_BYTES):
            raise CheckpointError("Invalid commit size/generation")
        buffer = io.BytesIO()
        commit = self._scan(entry, buffer)
        try:
            manifest = json.loads(buffer.getvalue())
            if (not isinstance(manifest, dict)
                    or set(manifest) != {"schema", "snapshot", "identity", "bucket", "prefix", "state_bundle", "payloads"}
                    or manifest["schema"] != 3 or manifest["snapshot"] != snapshot
                    or manifest["identity"] != expected_identity
                    or manifest["bucket"] != self.bucket_name or manifest["prefix"] != self.prefix):
                raise CheckpointError("Checkpoint identity or schema mismatch")
            payloads = manifest["payloads"]
            if not isinstance(payloads, dict) or not REQUIRED_ROLES <= set(payloads) or len(payloads) > 256:
                raise CheckpointError("Missing required state roles")
            for role, descriptor in payloads.items():
                _name(role)
                if role in REQUIRED_ROLES:
                    if (set(descriptor) != {"member", "bytes", "sha256"} or descriptor["member"] != role
                            or not HASH.fullmatch(descriptor["sha256"])
                            or type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0):
                        raise CheckpointError("Invalid bundled payload descriptor")
                else:
                    self._validate_object(descriptor)
            self._validate_object(manifest["state_bundle"])
        except (TypeError, KeyError, json.JSONDecodeError) as error:
            raise CheckpointError("Malformed checkpoint manifest") from error
        return {"manifest": manifest, "commit": commit}

    def _validate_object(self, descriptor):
        if (set(descriptor) != {"object", "generation", "bytes", "sha256"}
                or not HASH.fullmatch(descriptor["sha256"])
                or descriptor["object"] != f"{self.prefix}/objects/{descriptor['sha256']}"
                or type(descriptor["generation"]) is not int or descriptor["generation"] <= 0
                or type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0):
            raise CheckpointError("Invalid payload identity/generation")

    def verify(self, snapshot, expected_identity, *, commit_generation=None):
        # Reuse exactly the extraction checks; files exist only on bounded local
        # scratch, not in memory. This verifies member hashes, not just tar bytes.
        with tempfile.TemporaryDirectory(prefix="gcs-verify-", dir=self.staging_dir) as directory:
            return self.restore(snapshot, expected_identity, Path(directory) / "state",
                                commit_generation=commit_generation)

    def _extract_bundle(self, archive_path, payloads, staging=None):
        seen = set()
        with tarfile.open(archive_path, "r|") as archive:
            for member in archive:
                if (member.name not in REQUIRED_ROLES or member.name in seen or not member.isfile()
                        or member.size != payloads[member.name]["bytes"]):
                    raise CheckpointError("Unsafe, duplicate or mismatched state archive member")
                seen.add(member.name)
                digest = hashlib.sha256()
                output = (staging / member.name).open("xb") if staging is not None else nullcontext(None)
                with archive.extractfile(member) as source, output as target:
                    while chunk := source.read(CHUNK_BYTES):
                        digest.update(chunk)
                        if target is not None:
                            target.write(chunk)
                    if target is not None:
                        target.flush()
                        os.fsync(target.fileno())
                if digest.hexdigest() != payloads[member.name]["sha256"]:
                    raise CheckpointError("Bundled state member checksum mismatch")
        if seen != REQUIRED_ROLES:
            raise CheckpointError("Incomplete state archive")

    @_preserve_python_rng
    def restore(self, snapshot, expected_identity, destination, *, commit_generation=None):
        """Generation-bound, one-pass download; expose destination only when valid.

        Have one restoring owner for destination. Local disk must fit this state
        snapshot; original model/candidate/cache files are separate dependencies.
        """
        destination = Path(destination)
        if destination.exists():
            raise CheckpointError("Restore destination already exists")
        receipt = self._manifest(snapshot, expected_identity, commit_generation)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".gcs-restore-", dir=destination.parent))
        try:
            bundle = receipt["manifest"]["state_bundle"]
            archive_path = staging / ".state.tar"
            with archive_path.open("wb") as stream:
                self._scan(bundle, stream, expected_digest=bundle["sha256"])
            self._extract_bundle(archive_path, receipt["manifest"]["payloads"], staging)
            archive_path.unlink()
            for role, entry in receipt["manifest"]["payloads"].items():
                if role in REQUIRED_ROLES:
                    continue
                with (staging / role).open("wb") as stream:
                    self._scan(entry, stream, expected_digest=entry["sha256"])
                    stream.flush()
                    os.fsync(stream.fileno())
            _sync_directory(staging)
            if destination.exists():
                raise CheckpointError("Restore destination already exists")
            os.rename(staging, destination)
            _sync_directory(destination.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return receipt
