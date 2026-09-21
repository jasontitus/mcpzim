"""Restore a committed GCS input package onto attached disk before GPU startup.

Generation-pinned reads, full SHA256 verification, regular-file extraction and
independent calibration validation. No cloud writes or VM operations.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from gcs_checkpoints import make_client
from checkpoints import _sync_directory
from validate_capture import validate_package

CHUNK = 64 * 1024**2


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(bucket, descriptor, target):
    target = Path(target)
    if type(descriptor.get("bytes")) is not int or descriptor["bytes"] < 0:
        raise ValueError("Invalid object size")
    generation = int(descriptor["generation"])
    if generation <= 0:
        raise ValueError("Invalid object generation")
    if target.exists():
        if target.stat().st_size != descriptor["bytes"] or digest(target) != descriptor["sha256"]:
            raise ValueError("Existing downloaded object differs")
        return
    blob = bucket.blob(descriptor["object"], generation=generation)
    pending = target.with_name(target.name + ".partial")
    written = 0
    checksum = hashlib.sha256()
    owned = False
    last_progress = time.monotonic()
    try:
        with pending.open("xb") as output:
            owned = True
            with blob.open("rb", chunk_size=CHUNK, if_generation_match=generation) as source:
                while data := source.read(CHUNK):
                    written += len(data)
                    if written > descriptor["bytes"]:
                        raise ValueError("Remote object exceeds declared size")
                    checksum.update(data)
                    output.write(data)
                    if time.monotonic() - last_progress >= 30:
                        print(json.dumps({"event": "input_download", "object": descriptor["object"],
                            "bytes": written, "total_bytes": descriptor["bytes"]}), flush=True)
                        last_progress = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        if written != descriptor["bytes"] or checksum.hexdigest() != descriptor["sha256"]:
            raise ValueError("Downloaded object checksum/size mismatch")
        os.link(pending, target)
    finally:
        if owned:
            pending.unlink(missing_ok=True)


def unpack(archive_path, destination, *, resume=False, expected=None):
    """Extract only ordinary contained files into a fresh single-owner tree."""
    seen = set()
    total = 0
    with tarfile.open(archive_path) as archive:
        for member in archive:
            rel = Path(member.name)
            if (not member.isfile() or rel.is_absolute() or ".." in rel.parts or not rel.parts
                    or member.name in seen or member.size < 0):
                raise ValueError("Unsafe or duplicate archive member")
            seen.add(member.name)
            if expected is not None and member.name not in expected:
                raise ValueError("Unexpected archive member")
            total += member.size
            if total > archive_path.stat().st_size:
                raise ValueError("Archive expansion exceeds uncompressed input")
            target = destination / rel
            for parent in (target, *target.parents):
                if parent.is_symlink(): raise ValueError("Symlink in archive destination")
            if destination.resolve() not in target.resolve().parents:
                raise ValueError("Archive destination escapes root")
            target.parent.mkdir(parents=True, exist_ok=True)
            if resume and target.exists() and target.stat().st_size == member.size and expected and digest(target) == expected[member.name]:
                continue
            pending = target.with_name(target.name + ".restore-partial") if resume else target
            if resume and pending.exists():
                if pending.is_symlink() or not pending.is_file(): raise ValueError("Unsafe extraction partial")
                pending.unlink()
            checksum = hashlib.sha256()
            with archive.extractfile(member) as source, pending.open("xb") as output:
                while data := source.read(CHUNK):
                    checksum.update(data); output.write(data)
                output.flush(); os.fsync(output.fileno())
            if expected is not None and checksum.hexdigest() != expected[member.name]:
                raise ValueError("Archive member checksum mismatch")
            if resume: os.replace(pending, target)
    if expected is not None and seen != set(expected):
        raise ValueError("Missing archive members")


def regular_tree(path):
    """Refuse links at every level before adopting or modifying any old tree."""
    for item in (path, *path.parents):
        if item.is_symlink(): raise ValueError("Symlink in staging path")
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            item = Path(root) / name
            if item.is_symlink() or not (item.is_file() or item.is_dir()):
                raise ValueError("Nonregular staging entry")


def archive_json(path, member_name, expected_sha):
    with tarfile.open(path) as archive:
        matches = [m for m in archive if m.name == member_name]
        if len(matches) != 1 or not matches[0].isfile() or not 0 <= matches[0].size <= 16 * 1024**2:
            raise ValueError("Missing/oversize trusted archive metadata")
        with archive.extractfile(matches[0]) as stream: data = stream.read(16 * 1024**2 + 1)
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Archive metadata hash mismatch")
    return json.loads(data)


def sequence_files(sequence, descriptor):
    directory = descriptor["directory"]
    if Path(directory).name != directory or directory in ("", ".", ".."):
        raise ValueError("Unsafe sequence directory")
    expected = {f"{directory}/sequence.json": descriptor["manifest_sha256"],
                f"{directory}/input.json": sequence["input_sha256"]}
    for entry in sequence["files"].values():
        name = entry["file"]; relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != directory or name in expected:
            raise ValueError("Unsafe/duplicate sequence member")
        expected[name] = entry["sha256"]
    return expected


def existing_members_complete(root, expected):
    return all((root / name).is_file() and digest(root / name) == checksum for name, checksum in expected.items())


def refuse_unknown(root, expected, *, partial_suffix=None):
    """Known incomplete files may be repaired; foreign files are never removed."""
    expected = set(expected)
    allowed_dirs = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."}
    for item in root.rglob("*"):
        relative = str(item.relative_to(root))
        if item.is_dir():
            if relative not in allowed_dirs: raise ValueError("Unknown staging directory: " + relative)
        elif relative not in expected and not (partial_suffix and relative.endswith(partial_suffix) and relative[:-len(partial_suffix)] in expected):
            raise ValueError("Unknown staging file: " + relative)


def ensure_download(bucket, obj, path, allow_partial):
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        if not allow_partial or partial.is_symlink() or not partial.is_file():
            raise ValueError("Unowned download partial")
        partial.unlink()  # Only this exact manifest-derived filename after writer stop.
    download(bucket, obj, path)


def prepare_resume(staging, root, manifest, manifest_sha, bucket, adopt_legacy):
    """Reconcile a stopped writer's tree against a trusted manifest chain."""
    regular_tree(staging)
    allowed_top = {"model", "calibration", "downloads", "FAILED", ".restore-owner.json", ".restore-owner.pending", ".restore-lock"}
    if any(p.name not in allowed_top for p in staging.iterdir()):
        raise ValueError("Unknown staging root entry")
    owner_path = staging / ".restore-owner.json"
    owner = {"schema_version": 1, "manifest_sha256": manifest_sha, "destination": str(root)}
    if owner_path.exists():
        if json.loads(owner_path.read_text()) != owner: raise ValueError("Staging belongs to another manifest/destination")
    elif not adopt_legacy:
        raise ValueError("Legacy staging requires explicit adoption after stopping its original writer")
    for name in ("model", "calibration", "downloads"):
        if not (staging / name).is_dir(): raise ValueError("Incomplete staging directory layout")
    objects = {obj["object"]: obj for obj in manifest["objects"]}
    archives = {Path(name).name: objects[name] for name in manifest["calibration_archives"]}
    if len(archives) != len(manifest["calibration_archives"]) or "metadata.tar" not in archives:
        raise ValueError("Ambiguous/missing metadata archive")
    paths = {name: staging / "downloads" / (hashlib.sha256(obj["object"].encode()).hexdigest() + ".tar") for name, obj in archives.items()}
    model_prefix = f"models/{manifest['revision']}/"
    models = {obj['object'][len(model_prefix):]: obj for obj in manifest['objects'] if obj['object'].startswith(model_prefix)}
    if any(Path(name).name != name or name in ('', '.', '..') for name in models): raise ValueError("Unsafe model filename")
    if set(objects) != {obj['object'] for obj in archives.values()} | {obj['object'] for obj in models.values()}:
        raise ValueError("Unexpected object in input package")
    refuse_unknown(staging / "model", models, partial_suffix=".partial")
    refuse_unknown(staging / "downloads", [p.name for p in paths.values()], partial_suffix=".partial")
    if shutil.disk_usage(staging).free < 2 * max(obj['bytes'] for obj in archives.values()) + 16 * 1024**3:
        raise ValueError("Insufficient resume inspection/extraction headroom")
    ensure_download(bucket, archives['metadata.tar'], paths['metadata.tar'], True)
    package = archive_json(paths['metadata.tar'], 'manifest.json', manifest['package_manifest_sha256'])
    if package.get('status') != 'completed' or package.get('revision') != manifest['revision']:
        raise ValueError("Wrong completed calibration metadata")
    metadata_files = {'manifest.json': manifest['package_manifest_sha256'],
                      **{entry['file']: entry['sha256'] for entry in package['source_files'].values()}}
    specs = {'metadata.tar': metadata_files}
    for descriptor in package['sequences']:
        directory = descriptor['directory']; name = directory + '.tar'
        if Path(directory).name != directory or directory in ('', '.', '..'):
            raise ValueError('Unsafe sequence directory')
        if name not in archives or name in specs: raise ValueError("Sequence archive coverage differs")
        existing = staging / 'calibration' / directory / 'sequence.json'
        if existing.is_file() and digest(existing) == descriptor['manifest_sha256']:
            sequence = json.loads(existing.read_text())
        else:
            ensure_download(bucket, archives[name], paths[name], True)
            sequence = archive_json(paths[name], directory + '/sequence.json', descriptor['manifest_sha256'])
        specs[name] = sequence_files(sequence, descriptor)
    if set(specs) != set(archives): raise ValueError("Extra committed calibration archive")
    all_files = {name for expected in specs.values() for name in expected}
    refuse_unknown(staging / 'calibration', all_files, partial_suffix='.restore-partial')
    # Only adopt after all preexisting paths have an exact trusted counterpart.
    pending_owner = staging / '.restore-owner.pending'
    pending_owner.write_text(json.dumps(owner, sort_keys=True) + '\n')
    os.replace(pending_owner, owner_path); _sync_directory(staging)
    complete = {name for name, expected in specs.items() if existing_members_complete(staging / 'calibration', expected)}
    complete_models = set()
    for name, obj in models.items():
        path = staging / 'model' / name
        if path.exists():
            if path.stat().st_size != obj['bytes'] or digest(path) != obj['sha256']:
                raise ValueError('Existing complete model object differs; preserving it for diagnosis')
            complete_models.add(name)
    remaining = sum(obj['bytes'] for name, obj in models.items() if name not in complete_models)
    remaining += 2 * sum(obj['bytes'] for name, obj in archives.items() if name not in complete)
    if shutil.disk_usage(staging).free < remaining + 16 * 1024**3:
        raise ValueError('Insufficient remaining restore disk space')
    print(json.dumps({'event':'input_resume_verified','completed_archives':len(complete),
                      'completed_model_files':len(complete_models),'remaining_bytes':remaining}),flush=True)
    for name, obj in archives.items():
        if name in complete:
            print('Reused verified archive members: ' + name, flush=True)
        else:
            ensure_download(bucket, obj, paths[name], True)
            unpack(paths[name], staging / 'calibration', resume=True, expected=specs[name])
            print('Restored archive: ' + name, flush=True)
        paths[name].unlink(missing_ok=True)
    for name, obj in models.items():
        ensure_download(bucket, obj, staging / 'model' / name, True)
        print(('Reused verified model: ' if name in complete_models else 'Restored model: ') + name, flush=True)


def restore(args):
    manifest_path = Path(args.manifest)
    if digest(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("Input manifest differs from trusted staging receipt")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "staged":
        raise ValueError("Committed staging manifest required")
    root = Path(args.destination).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    total = sum(obj["bytes"] for obj in manifest["objects"])
    resume_path = getattr(args, "resume_staging", None)
    if not resume_path and shutil.disk_usage(root.parent).free < total * 2 + 16 * 1024**3:
        raise ValueError("Insufficient restore disk space/headroom")
    names = [obj["object"] for obj in manifest["objects"]]
    archives = set(manifest["calibration_archives"])
    if len(names) != len(set(names)) or not archives <= set(names):
        raise ValueError("Duplicate or missing committed objects")
    bucket = make_client("tiltastech-zimfo", use_gcloud=args.local_gcloud).bucket(manifest["bucket"])
    if resume_path:
        staging = Path(resume_path).absolute()
        regular_tree(staging)
        if staging.parent != root.parent or not staging.name.startswith(".input-restore-") or not staging.is_dir():
            raise ValueError("Resume directory must be the exact existing sibling staging tree")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".input-restore-", dir=root.parent))
    lock = (staging / ".restore-lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if resume_path:
            prepare_resume(staging, root, manifest, args.expected_manifest_sha256, bucket,
                           getattr(args, "adopt_legacy_staging", False))
        else:
            for directory in ("model", "calibration", "downloads"):
                (staging / directory).mkdir()
            (staging / ".restore-owner.json").write_text(json.dumps({"schema_version": 1,
                "manifest_sha256": args.expected_manifest_sha256, "destination": str(root)}))
        model_prefix = f"models/{manifest['revision']}/"
        for obj in ([] if resume_path else manifest["objects"]):
            name = obj["object"]
            if name in archives:
                local = staging / "downloads" / (hashlib.sha256(name.encode()).hexdigest() + ".tar")
                download(bucket, obj, local)
                unpack(local, staging / "calibration")
                local.unlink()
            elif name.startswith(model_prefix) and Path(name[len(model_prefix):]).name == name[len(model_prefix):]:
                download(bucket, obj, staging / "model" / name[len(model_prefix):])
            else:
                raise ValueError("Unexpected object in committed input package")
            print(f"Restored {name}", flush=True)
        result = validate_package(staging / "calibration")
        if result["manifest_sha256"] != manifest["package_manifest_sha256"]:
            raise ValueError("Restored calibration identity mismatch")
        model_manifest = json.loads((staging / "calibration/manifest.json").read_text())["model_artifacts"]["files"]
        if {p.name for p in (staging / "model").iterdir()} != set(model_manifest):
            raise ValueError("Restored model file coverage mismatch")
        for name, expected in model_manifest.items():
            path = staging / "model" / name
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError("Restored model identity mismatch")
        result["input_commit_sha256"] = digest(manifest_path)
        with (staging / "restore-validation.json").open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        for path, _, _ in os.walk(staging, topdown=False):
            _sync_directory(Path(path))
        os.rename(staging, root)
        _sync_directory(root.parent)
    except BaseException:
        # Keep verified downloads for diagnosis, never expose a completed root.
        if staging.exists():
            (staging / "FAILED").write_text("Incomplete input restore; not usable for a GPU job\n")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True,
                        help="SHA256 from the trusted staging receipt, not the downloaded manifest itself")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--local-gcloud", action="store_true")
    parser.add_argument("--resume-staging")
    parser.add_argument("--adopt-legacy-staging", action="store_true")
    restore(parser.parse_args())
