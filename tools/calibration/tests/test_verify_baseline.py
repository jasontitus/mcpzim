import hashlib
import json
import struct
import pytest
from verify_baseline import REVISION, verify


def fixture(tmp_path, dtype="BF16"):
    name = "model-00001-of-00001.safetensors"
    tensors = {"weight": {"dtype": dtype, "shape": [1, 2], "data_offsets": [0, 4]}}
    header = json.dumps(tensors).encode()
    payload = struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00"
    (tmp_path / name).write_bytes(payload)
    (tmp_path / "config.json").write_text('{}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": name}}))
    hashes = {"model": "Qwen/Qwen3.8-27B", "revision": REVISION,
              "files": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}
    inventory = {"model": hashes["model"], "revision": REVISION, "tensors": tensors}
    return hashes, inventory, name


def test_original_hash_and_coverage_verified(tmp_path):
    hashes, inventory, _ = fixture(tmp_path)
    result = verify(tmp_path, hashes, inventory, tmp_path / "result.json")
    assert result["status"] == "validated"
    assert result["tensor_count"] == 1


@pytest.mark.parametrize("failure", ["corrupt", "missing", "wrong-revision", "quantized", "coverage"])
def test_bad_inputs_cannot_preserve_stale_success(tmp_path, failure):
    hashes, inventory, name = fixture(tmp_path)
    output = tmp_path / "result.json"
    output.write_text('{"status":"validated"}')
    if failure == "corrupt":
        payload = bytearray((tmp_path / name).read_bytes()); payload[-1] = 1
        (tmp_path / name).write_bytes(payload)
    elif failure == "missing":
        (tmp_path / name).unlink()
    elif failure == "wrong-revision":
        hashes["revision"] = "wrong"
    elif failure == "quantized":
        (tmp_path / "config.json").write_text('{"quantization_config":{}}')
    else:
        inventory["tensors"]["extra"] = inventory["tensors"]["weight"]
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(tmp_path, hashes, inventory, output)
    assert json.loads(output.read_text())["status"] == "failed"


def test_packed_integer_weights_rejected_even_when_hash_matches(tmp_path):
    hashes, inventory, _ = fixture(tmp_path, dtype="I16")
    with pytest.raises(ValueError, match="BF16"):
        verify(tmp_path, hashes, inventory, tmp_path / "result.json")
