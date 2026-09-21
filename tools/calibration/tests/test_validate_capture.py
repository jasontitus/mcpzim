import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

import validate_capture as vc


def write_json(path, data):
    path.write_text(json.dumps(data))


def save_tensors(path, tensors):
    header, body = {}, b""
    for name, (array, dtype) in tensors.items():
        data = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [len(body), len(body) + len(data)]}
        body += data
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


def update_sequence(root, sequence):
    path = root / "invocation-000000" / "sequence.json"
    write_json(path, sequence)
    manifest = vc.read_json(root / "manifest.json")
    manifest["sequences"][0]["manifest_sha256"] = vc.sha(path)
    manifest["exported_tensor_bytes"] = sum(info["bytes"] for info in sequence["files"].values())
    write_json(root / "manifest.json", manifest)


@pytest.fixture
def package(tmp_path, monkeypatch):
    # Exercise the exact production file validation with tiny tensors. Independent
    # architecture tests below check the unpatched real 497-projection inventory.
    monkeypatch.setattr(vc, "expected_projection_shapes", lambda: {"projection": [4, 2]})
    monkeypatch.setattr(vc, "HIDDEN_SIZE", 2)
    directory = tmp_path / "invocation-000000"
    directory.mkdir()
    record = {"ordinal": 0, "prompt": "test", "promptSHA256": hashlib.sha256(b"test").hexdigest(),
              "tokenIDs": [1, 2, 3], "conversationID": "calibration", "turn": 1}
    write_json(directory / "input.json", record)
    sequence = {"input_sha256": vc.sha(directory / "input.json"), "source_capture_sha256": vc.sha(directory / "input.json"),
                "conversation_id": "calibration", "turn": 1, "tokens": 3, "fresh_state": True, "files": {}}
    tensors = {"x": (np.zeros((2, 2), dtype=np.uint16), "BF16"),
               "row_indices": (np.array([0, 2], dtype=np.int32), "I32"),
               "sum_x": (np.zeros(2, dtype=np.float32), "F32"),
               "sum_x_squared": (np.zeros(2, dtype=np.float32), "F32")}
    for name in ("projection", "first_block_input", "final_normalized_hidden"):
        actual = tensors if name == "projection" else {"hidden": (np.zeros((1, 3, 2), dtype=np.uint16), "BF16")}
        path = directory / (hashlib.sha256(name.encode()).hexdigest()[:24] + ".safetensors")
        save_tensors(path, actual)
        sequence["files"][name] = {"file": str(path.relative_to(tmp_path)), "bytes": path.stat().st_size,
                                   "sha256": vc.sha(path), "total_rows": 3, "sample_rows": 2,
                                   "sampling": "one uniform row per contiguous stratum" if name == "projection" else "all rows",
                                   "moments": "uncentered all-row sums; diagonal only, not full X^T X",
                                   "tensors": {key: {"shape": list(array.shape), "dtype": vc.DTYPES[dtype][2]}
                                               for key, (array, dtype) in actual.items()}}
    manifest = {"schema_version": 1, "status": "completed", "model": "Qwen/Qwen3.8-27B", "revision": vc.REVISION,
                "model_artifacts": {"resolved_model_dir": "/fixture", "metadata_manifest_sha256": "a" * 64,
                                    "files": {name: {"bytes": 1, "sha256": "b" * 64} for name in vc.MODEL_FILES}},
                "runtime_provenance": {"python": "3.12", "mlx": "0.32.2", "mlx_lm": "0.31.3",
                                       "sanitizer": "fixture", "teacher_head": "original BF16",
                                       "collector_sha256": "c" * 64, "qwen_backend_sha256": "d" * 64},
                "source_invocation_count": 1, "source_invocation_sha256": [sequence["input_sha256"]],
                "weight_validation_sha256": "1" * 64, "capture_manifest_sha256": "2" * 64,
                "projections": {"projection": {"weight_shape": [4, 2], "dtype": "mlx.core.bfloat16"}},
                "rows_per_projection_per_invocation": 2,
                "sequences": [{"directory": directory.name, "tokens": 3, "manifest_sha256": "0" * 64}]}
    write_json(tmp_path / "capture-run.json", {"status": "completed", "invocationCount": "1"})
    write_json(tmp_path / "calibration-suite.json", {"conversations": [{"id": "calibration"}]})
    manifest["source_files"] = {role: {"file": name, "bytes": (tmp_path / name).stat().st_size,
                                      "sha256": vc.sha(tmp_path / name)}
                                for role, name in (("capture_manifest", "capture-run.json"),
                                                   ("calibration_suite", "calibration-suite.json"))}
    manifest["capture_manifest_sha256"] = vc.sha(tmp_path / "capture-run.json")
    write_json(tmp_path / "manifest.json", manifest)
    update_sequence(tmp_path, sequence)
    return tmp_path


def test_full_pinned_architecture_inventory():
    inventory = vc.expected_projection_shapes()
    assert len(inventory) == 497
    assert sum("linear_attn" in name for name in inventory) == 48 * 5
    assert sum("self_attn" in name for name in inventory) == 16 * 4
    assert sum("mlp" in name for name in inventory) == 64 * 3
    assert inventory["language_model.model.layers.3.self_attn.q_proj"] == [12288, 5120]
    assert inventory["language_model.model.layers.0.linear_attn.in_proj_qkv"] == [10240, 5120]
    assert inventory["language_model.lm_head"] == [248320, 5120]


def test_valid_package_scans_with_small_chunks(package, monkeypatch):
    monkeypatch.setattr(vc, "CHUNK_BYTES", 4)
    result = vc.validate_package(package)
    assert result["status"] == "validated"
    assert result["sequences"] == 1 and result["tokens"] == 3
    assert result["tensors"] == 6


@pytest.mark.parametrize("status", ["running", "failed", None])
def test_reject_unfinished(package, status):
    manifest = vc.read_json(package / "manifest.json")
    manifest["status"] = status
    write_json(package / "manifest.json", manifest)
    with pytest.raises(vc.CaptureValidationError, match="not completed"):
        vc.validate_package(package)


@pytest.mark.parametrize("change", ["missing_inventory", "wrong_shape", "quantized_dtype", "wrong_revision", "wrong_total"])
def test_reject_manifest_claims(package, change):
    path = package / "manifest.json"
    manifest = vc.read_json(path)
    if change == "missing_inventory":
        manifest["projections"] = {}
    elif change == "wrong_shape":
        manifest["projections"]["projection"]["weight_shape"] = [5, 2]
    elif change == "quantized_dtype":
        manifest["projections"]["projection"]["dtype"] = "mlx.core.uint32"
    elif change == "wrong_revision":
        manifest["revision"] = "a" * 40
    else:
        manifest["exported_tensor_bytes"] += 1
    write_json(path, manifest)
    with pytest.raises(vc.CaptureValidationError):
        vc.validate_package(package)


@pytest.mark.parametrize("change", ["missing", "corrupt", "sequence_checksum", "path_escape", "unreferenced"])
def test_reject_file_integrity(package, change):
    sequence = vc.read_json(package / "invocation-000000/sequence.json")
    info = sequence["files"]["projection"]
    path = package / info["file"]
    if change == "missing":
        path.unlink()
    elif change == "corrupt":
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif change == "sequence_checksum":
        with (package / "invocation-000000/sequence.json").open("a") as stream:
            stream.write(" ")
    elif change == "unreferenced":
        (path.parent / "pending.safetensors").write_bytes(b"garbage")
    else:
        info["file"] = "../outside.safetensors"
        update_sequence(package, sequence)
    with pytest.raises(vc.CaptureValidationError):
        vc.validate_package(package)


@pytest.mark.parametrize("change", ["nan", "inf_bf16", "duplicate_rows", "negative_row", "high_row", "shape", "dtype", "hidden_shape", "negative_moment"])
def test_reject_tensor_contents_even_with_valid_hashes(package, change):
    sequence = vc.read_json(package / "invocation-000000/sequence.json")
    tensors = {"x": (np.zeros((2, 2), dtype=np.uint16), "BF16"),
               "row_indices": (np.array([0, 2], dtype=np.int32), "I32"),
               "sum_x": (np.zeros(2, dtype=np.float32), "F32"),
               "sum_x_squared": (np.zeros(2, dtype=np.float32), "F32")}
    name = "projection"
    if change == "nan":
        tensors["sum_x"][0][0] = np.nan
    elif change == "inf_bf16":
        tensors["x"][0][0, 0] = 0x7F80
    elif change == "negative_moment":
        tensors["sum_x_squared"][0][0] = -1
    elif change in ("duplicate_rows", "negative_row", "high_row"):
        tensors["row_indices"][0][0] = {"duplicate_rows": 2, "negative_row": -1, "high_row": 3}[change]
    elif change == "shape":
        tensors["x"] = (np.zeros((2, 3), dtype=np.uint16), "BF16")
    elif change == "dtype":
        tensors["sum_x"] = (np.zeros(2, dtype=np.uint16), "BF16")
    elif change == "hidden_shape":
        name = "first_block_input"
        tensors = {"hidden": (np.zeros((1, 2, 2), dtype=np.uint16), "BF16")}
    info = sequence["files"][name]
    path = package / info["file"]
    save_tensors(path, tensors)
    info.update(sha256=vc.sha(path), bytes=path.stat().st_size)
    info["tensors"] = {key: {"shape": list(array.shape), "dtype": vc.DTYPES[dtype][2]}
                       for key, (array, dtype) in tensors.items()}
    update_sequence(package, sequence)
    with pytest.raises(vc.CaptureValidationError):
        vc.validate_package(package)


@pytest.mark.parametrize("change", ["ordinal", "tokens", "prompt", "conversation"])
def test_reject_wrong_input_with_updated_file_hash(package, change):
    path = package / "invocation-000000/input.json"
    record = vc.read_json(path)
    if change == "ordinal":
        record["ordinal"] = 1
    elif change == "tokens":
        record["tokenIDs"] = [1, -1, 3]
    elif change == "prompt":
        record["prompt"] = "changed"
    else:
        record["conversationID"] = "another"
    write_json(path, record)
    sequence = vc.read_json(path.parent / "sequence.json")
    sequence["input_sha256"] = sequence["source_capture_sha256"] = vc.sha(path)
    update_sequence(package, sequence)
    with pytest.raises(vc.CaptureValidationError):
        vc.validate_package(package)


@pytest.mark.parametrize("change", ["duplicate", "gap", "extra_directory"])
def test_reject_noncontiguous_invocations(package, change):
    path = package / "manifest.json"
    manifest = vc.read_json(path)
    if change == "duplicate":
        manifest["sequences"].append(manifest["sequences"][0])
    elif change == "gap":
        manifest["sequences"][0]["directory"] = "invocation-000001"
    else:
        (package / "invocation-000001").mkdir()
    write_json(path, manifest)
    with pytest.raises(vc.CaptureValidationError):
        vc.validate_package(package)


def test_json_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"status":"failed", "status":"completed"}')
    with pytest.raises(vc.CaptureValidationError, match="Duplicate"):
        vc.read_json(path)


@pytest.mark.parametrize("field", ["source_invocation_count", "source_invocation_sha256", "runtime_provenance", "model_artifacts", "schema_version", "source_files"])
def test_reject_missing_provenance(package, field):
    path = package / "manifest.json"
    value = vc.read_json(path)
    del value[field]
    write_json(path, value)
    with pytest.raises(vc.CaptureValidationError):
        vc.validate_package(package)


def test_reject_incomplete_original_capture_even_with_updated_hash(package):
    source = package / "capture-run.json"
    write_json(source, {"status": "running", "invocationCount": "1"})
    manifest = vc.read_json(package / "manifest.json")
    manifest["source_files"]["capture_manifest"].update(bytes=source.stat().st_size, sha256=vc.sha(source))
    manifest["capture_manifest_sha256"] = vc.sha(source)
    write_json(package / "manifest.json", manifest)
    with pytest.raises(vc.CaptureValidationError, match="incomplete"):
        vc.validate_package(package)
