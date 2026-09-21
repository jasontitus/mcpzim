"""Pack and immutably stage validated model/calibration inputs, without a VM.

Uses bounded concurrent resumable uploads, generation preconditions and CRC32C.
The final input manifest is published only after every object verifies. This
is artifact staging, not an optimizer checkpoint or readiness attestation.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import threading

import google_crc32c
from google.api_core.exceptions import PreconditionFailed
from google.cloud.storage.retry import DEFAULT_RETRY_IF_GENERATION_SPECIFIED

from gcs_checkpoints import make_client

CHUNK = 8 * 1024**2


def fingerprint(path):
    sha, crc, size = hashlib.sha256(), google_crc32c.Checksum(), 0
    with Path(path).open("rb") as stream:
        while data := stream.read(CHUNK):
            sha.update(data)
            crc.update(data)
            size += len(data)
    return {"bytes": size, "sha256": sha.hexdigest(), "crc32c": base64.b64encode(crc.digest()).decode()}


def atomic_json(path, value):
    pending = Path(str(path) + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def pack(root, files, output):
    """Deterministic regular-file archive; no symlink traversal or overwrite."""
    if output.exists():
        raise FileExistsError(output)
    pending = Path(str(output) + ".pending")
    try:
        with tarfile.open(pending, "w", format=tarfile.PAX_FORMAT) as archive:
            for relative in sorted(files):
                rel = Path(relative)
                if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                    raise ValueError("Unsafe archive member")
                source = root / rel
                if source.is_symlink() or not source.is_file() or root.resolve() not in source.resolve().parents:
                    raise ValueError("Only contained regular files may be archived")
                info = tarfile.TarInfo(rel.as_posix())
                info.size = source.stat().st_size
                info.mode, info.mtime, info.uid, info.gid = 0o600, 0, 0, 0
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        pending.rename(output)
    finally:
        pending.unlink(missing_ok=True)


def verify_archive(path, expected):
    """Bind every packed byte to the independently validated source inventory."""
    seen = set()
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected/nonregular/duplicate archive member")
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                while data := stream.read(CHUNK):
                    digest.update(data)
            if digest.hexdigest() != expected[member.name]:
                raise ValueError("Archive member checksum mismatch")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Missing archive member")


def upload(bucket, name, path, expected=None):
    before = fingerprint(path)
    if expected and any(before[key] != expected[key] for key in ("bytes", "sha256")):
        raise ValueError(f"Local input changed: {path.name}")
    blob = bucket.blob(name, chunk_size=CHUNK)
    blob.metadata = {"sha256": before["sha256"]}
    try:
        blob.upload_from_filename(str(path), if_generation_match=0, checksum="crc32c",
            crc32c_checksum_value=before["crc32c"], retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED, timeout=120)
    except PreconditionFailed:
        # A prior successful attempt may have lost its response; never overwrite.
        blob = bucket.blob(name)
    blob.reload()
    if int(blob.size) != before["bytes"] or blob.crc32c != before["crc32c"] or (blob.metadata or {}).get("sha256") != before["sha256"]:
        raise ValueError(f"Remote object conflict/corruption: {name}")
    return {"object": name, "generation": str(blob.generation), **before}


def stage(args):
    package = Path(args.package).resolve()
    manifest_path = package / "manifest.json"
    package_manifest = json.loads(manifest_path.read_text())
    validation = json.loads(Path(args.validation).read_text())
    identity = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if package_manifest.get("status") != "completed" or validation.get("status") != "validated" or validation.get("manifest_sha256") != identity:
        raise ValueError("Completed package with matching independent validation required")
    if package_manifest.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":
        raise ValueError("Unexpected reference revision")
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < package_manifest["exported_tensor_bytes"] + 24 * 1024**3:
        raise ValueError("Insufficient temporary archive capacity with 24GiB reserve")
    client = make_client("tiltastech-zimfo", use_gcloud=args.local_gcloud)
    bucket = client.bucket(args.bucket)
    prefix = f"inputs/{identity}"
    state = {"status": "uploading", "bucket": args.bucket, "prefix": prefix, "objects": {}}
    lock = threading.Lock()
    atomic_json(work / "status.json", state)

    def submit_one(name, path, expected=None):
        result = upload(bucket, name, path, expected)
        with lock:
            state["objects"][name] = result
            atomic_json(work / "status.json", state)
            print(f"Verified {len(state['objects'])} objects: {name}", flush=True)
        return result

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            model_files = package_manifest["model_artifacts"]["files"]
            for name, expected in sorted(model_files.items()):
                if Path(name).name != name:
                    raise ValueError("Unsafe model filename")
                futures.append(executor.submit(submit_one, f"models/{package_manifest['revision']}/{name}", Path(args.model_dir) / name, expected))
            archives = []
            groups = [("metadata", {"manifest.json": identity,
                **{entry["file"]: entry["sha256"] for entry in package_manifest["source_files"].values()}})]
            for sequence in package_manifest["sequences"]:
                directory = sequence["directory"]
                if Path(directory).name != directory:
                    raise ValueError("Unsafe sequence directory")
                seq_path = package / directory / "sequence.json"
                if hashlib.sha256(seq_path.read_bytes()).hexdigest() != sequence["manifest_sha256"]:
                    raise ValueError("Sequence manifest changed")
                seq = json.loads(seq_path.read_text())
                groups.append((directory, {f"{directory}/input.json": seq["input_sha256"],
                    f"{directory}/sequence.json": sequence["manifest_sha256"],
                    **{entry["file"]: entry["sha256"] for entry in seq["files"].values()}}))
            for name, expected_files in groups:
                archive = work / f"{name}.tar"
                if not archive.exists():
                    pack(package, expected_files, archive)
                verify_archive(archive, expected_files)
                object_name = f"{prefix}/archives/{archive.name}"
                archives.append(object_name)
                futures.append(executor.submit(submit_one, object_name, archive))
            results = [future.result() for future in as_completed(futures)]
        commit = {"schema_version": 1, "status": "staged", "package_manifest_sha256": identity,
                  "model": package_manifest["model"], "revision": package_manifest["revision"],
                  "bucket": args.bucket, "objects": sorted(results, key=lambda x: x["object"]),
                  "calibration_archives": archives,
                  "verification": "source SHA256; resumable CRC32C-validated upload; remote metadata CRC32C/size and generation; full consumer restore still required"}
        commit_path = work / "inputs-manifest.json"
        atomic_json(commit_path, commit)
        committed = upload(bucket, f"{prefix}/inputs-manifest.json", commit_path)
        state.update(status="staged", commit=committed)
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(work / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "validation", "model-dir", "work-dir", "bucket"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--local-gcloud", action="store_true")
    stage(parser.parse_args())
