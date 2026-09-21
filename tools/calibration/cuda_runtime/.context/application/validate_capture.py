"""CPU-only validator for a completed original-Qwen Mac activation package.

No MLX, CUDA or model loading. Safetensors bodies are scanned in bounded chunks;
the complete activation corpus and even a full hidden tensor are never loaded.
This establishes artifact integrity/coverage, not solver or model parity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HIDDEN_SIZE = 5120
VOCAB_SIZE = 248320
CHUNK_BYTES = 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
DTYPES = {"BF16": (2, "<u2", "mlx.core.bfloat16"),
          "F32": (4, "<f4", "mlx.core.float32"),
          "I32": (4, "<i4", "mlx.core.int32")}
MODEL_FILES = {f"model-{i:05d}-of-00018.safetensors" for i in range(1, 19)} | {
    "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "generation_config.json", "merges.txt", "vocab.json", "model.safetensors.index.json"}


class CaptureValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CaptureValidationError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    require(len(data) <= MAX_JSON_BYTES, "JSON exceeds bounded size")
    return json.loads(data, object_pairs_hook=_unique_object)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _checked_file(root, relative, digest, size=None):
    require(isinstance(relative, str), "Missing file name")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()), "Artifact escapes package")
    require(result.is_file() and not result.is_symlink(), f"Missing or unsafe file: {relative}")
    if size is not None:
        require(type(size) is int and size >= 0 and result.stat().st_size == size,
                f"File size mismatch: {relative}")
    require(_digest(digest) and sha(result) == digest, f"File checksum mismatch: {relative}")
    return result


def expected_projection_shapes():
    """Pinned text architecture; independent of the producer's claimed inventory."""
    result = {}
    for index in range(64):
        prefix = f"language_model.model.layers.{index}."
        shapes = {"mlp.gate_proj": [17408, 5120], "mlp.up_proj": [17408, 5120],
                  "mlp.down_proj": [5120, 17408]}
        if (index + 1) % 4:
            shapes.update({"linear_attn.in_proj_qkv": [10240, 5120],
                           "linear_attn.in_proj_z": [6144, 5120],
                           "linear_attn.in_proj_a": [48, 5120],
                           "linear_attn.in_proj_b": [48, 5120],
                           "linear_attn.out_proj": [5120, 6144]})
        else:
            shapes.update({"self_attn.q_proj": [12288, 5120],
                           "self_attn.k_proj": [1024, 5120],
                           "self_attn.v_proj": [1024, 5120],
                           "self_attn.o_proj": [5120, 6144]})
        result.update({prefix + name: shape for name, shape in shapes.items()})
    result["language_model.lm_head"] = [248320, 5120]
    return result


def validate_tensor_file(path, claimed, expected, total_rows=None):
    """Check exact header layout, scan finite values, and validate sampled indices."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Truncated safetensors header")
        header_bytes = struct.unpack("<Q", prefix)[0]
        require(2 <= header_bytes <= MAX_JSON_BYTES and 8 + header_bytes <= size,
                "Invalid safetensors header length")
        header = json.loads(stream.read(header_bytes), object_pairs_hook=_unique_object)
        header.pop("__metadata__", None)
        require(set(header) == set(expected) == set(claimed), "Tensor inventory mismatch")
        extents = []
        for name, (shape, allowed_dtypes) in expected.items():
            item = header[name]
            dtype = item["dtype"]
            require(dtype in allowed_dtypes and dtype in DTYPES, f"Unexpected tensor dtype: {name}")
            require(item["shape"] == shape and all(type(v) is int for v in item["shape"]),
                    f"Tensor shape mismatch: {name}")
            require(claimed[name] == {"shape": shape, "dtype": DTYPES[dtype][2]},
                    f"Claimed tensor metadata mismatch: {name}")
            offsets = item["data_offsets"]
            require(isinstance(offsets, list) and len(offsets) == 2
                    and all(type(v) is int for v in offsets), "Invalid tensor offsets")
            start, end = offsets
            require(0 <= start <= end and end - start == math.prod(shape) * DTYPES[dtype][0],
                    f"Tensor byte extent mismatch: {name}")
            extents.append((start, end, name, dtype))
        previous = 0
        for start, end, name, dtype in sorted(extents):
            require(start == previous, "Safetensors overlap or unaccounted bytes")
            previous = end
        require(8 + header_bytes + previous == size, "Safetensors length mismatch")
        for start, end, name, dtype in extents:
            stream.seek(8 + header_bytes + start)
            remaining = end - start
            indices = []
            while remaining:
                raw = stream.read(min(CHUNK_BYTES, remaining))
                require(raw and len(raw) % DTYPES[dtype][0] == 0, "Truncated tensor body")
                values = np.frombuffer(raw, dtype=DTYPES[dtype][1])
                if dtype == "BF16":
                    require(not np.any((values & 0x7F80) == 0x7F80), f"Nonfinite tensor: {name}")
                elif dtype == "F32":
                    require(bool(np.all(np.isfinite(values))), f"Nonfinite tensor: {name}")
                    if name == "sum_x_squared":
                        require(bool(np.all(values >= 0)), "Negative squared moments")
                if name == "row_indices":
                    indices.extend(values.tolist())
                remaining -= len(raw)
            if name == "row_indices":
                require(total_rows is not None and len(indices) == len(set(indices))
                        and all(0 <= value < total_rows for value in indices),
                        "Duplicate or out-of-bounds row index")
                count = len(indices)
                require(all(i * total_rows // count <= value < (i + 1) * total_rows // count
                            for i, value in enumerate(indices)), "Invalid stratified sampling")


def validate_package(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    require(manifest.get("status") == "completed", "Capture is not completed")
    require(manifest.get("schema_version") == 1, "Unsupported capture schema")
    require(manifest.get("model") == "Qwen/Qwen3.8-27B" and manifest.get("revision") == REVISION,
            "Wrong full-precision baseline identity")
    for key in ("weight_validation_sha256", "capture_manifest_sha256"):
        require(_digest(manifest.get(key)), f"Invalid provenance hash: {key}")
    source_files = manifest.get("source_files", {})
    require(set(source_files) == {"capture_manifest", "calibration_suite"}, "Missing original source files")
    source_documents = {}
    for role, filename in (("capture_manifest", "capture-run.json"), ("calibration_suite", "calibration-suite.json")):
        info = source_files[role]
        require(info["file"] == filename, "Wrong source filename")
        source_documents[role] = read_json(_checked_file(root, filename, info["sha256"], info["bytes"]))
    require(source_files["capture_manifest"]["sha256"] == manifest["capture_manifest_sha256"],
            "Original capture manifest changed")
    source_manifest = source_documents["capture_manifest"]
    require(source_manifest.get("status") == "completed", "Original invocation capture is incomplete")
    conversations = source_documents["calibration_suite"].get("conversations")
    require(isinstance(conversations, list) and len(conversations) > 0, "Missing calibration scenarios")
    conversation_ids = [entry["id"] for entry in conversations]
    require(len(set(conversation_ids)) == len(conversation_ids), "Duplicate calibration scenario")
    artifacts = manifest.get("model_artifacts", {})
    require(_digest(artifacts.get("metadata_manifest_sha256"))
            and isinstance(artifacts.get("resolved_model_dir"), str), "Missing model provenance")
    require(set(artifacts.get("files", {})) == MODEL_FILES, "Incomplete model artifact provenance")
    for descriptor in artifacts["files"].values():
        require(_digest(descriptor.get("sha256")) and type(descriptor.get("bytes")) is int
                and descriptor["bytes"] > 0, "Invalid model artifact descriptor")
    runtime = manifest.get("runtime_provenance", {})
    for key in ("python", "mlx", "mlx_lm", "sanitizer", "teacher_head"):
        require(isinstance(runtime.get(key), str) and bool(runtime[key]), "Incomplete runtime provenance")
    for key in ("collector_sha256", "qwen_backend_sha256"):
        require(_digest(runtime.get(key)), "Invalid runtime code hash")
    expected_shapes = expected_projection_shapes()
    inventory = manifest.get("projections", {})
    require(set(inventory) == set(expected_shapes), "Missing or unexpected projection inventory")
    for name, shape in expected_shapes.items():
        require(inventory[name] == {"weight_shape": shape, "dtype": "mlx.core.bfloat16"},
                f"Projection metadata mismatch: {name}")
    rows = manifest.get("rows_per_projection_per_invocation")
    require(type(rows) is int and 0 < rows <= 16384, "Invalid sampled row budget")
    sequences = manifest.get("sequences")
    require(isinstance(sequences, list) and 0 < len(sequences) <= 100000, "Invalid sequence list")
    source_hashes = manifest.get("source_invocation_sha256")
    # Swift capture metadata is a string dictionary, unlike the collector JSON.
    source_count = source_manifest.get("invocationCount")
    require(isinstance(source_count, str) and re.fullmatch(r"[1-9][0-9]*", source_count) is not None,
            "Invalid original invocation count")
    require(type(manifest.get("source_invocation_count")) is int
            and manifest["source_invocation_count"] == len(sequences)
            and int(source_count) == len(sequences)
            and isinstance(source_hashes, list) and len(source_hashes) == len(sequences)
            and all(_digest(digest) for digest in source_hashes), "Incomplete source invocation set")
    expected_directories = {f"invocation-{index:06d}" for index in range(len(sequences))}
    require({p.name for p in root.glob("invocation-*")} == expected_directories,
            "Missing or unreferenced invocation directories")
    total_bytes = total_tokens = tensor_count = 0
    for index, entry in enumerate(sequences):
        directory = f"invocation-{index:06d}"
        require(entry["directory"] == directory, "Noncontiguous or duplicate invocation")
        sequence_path = _checked_file(root, directory + "/sequence.json", entry["manifest_sha256"])
        sequence = read_json(sequence_path)
        tokens = entry["tokens"]
        require(type(tokens) is int and 0 < tokens <= 16384 and sequence["tokens"] == tokens,
                "Invalid or inconsistent sequence length")
        require(sequence.get("fresh_state") is True, "Sequence did not use fresh state")
        input_path = _checked_file(root, directory + "/input.json", sequence["input_sha256"])
        require(sequence["source_capture_sha256"] == sequence["input_sha256"], "Input copy changed")
        require(sequence["source_capture_sha256"] == source_hashes[index], "Wrong source invocation")
        invocation = read_json(input_path)
        token_ids = invocation["tokenIDs"]
        require(invocation["ordinal"] == index and type(invocation["ordinal"]) is int,
                "Invocation ordinal mismatch")
        require(len(token_ids) == tokens and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in token_ids),
                "Invalid token IDs or count")
        require(hashlib.sha256(invocation["prompt"].encode()).hexdigest() == invocation["promptSHA256"],
                "Prompt checksum mismatch")
        require(invocation["conversationID"] == sequence["conversation_id"] and invocation["turn"] == sequence["turn"],
                "Conversation provenance mismatch")
        require(invocation["conversationID"] in conversation_ids, "Invocation absent from calibration suite")
        files = sequence["files"]
        require(set(files) == set(expected_shapes) | {"first_block_input", "final_normalized_hidden"},
                "Missing or unexpected activation boundary")
        used_paths = set()
        for name, info in files.items():
            relative = info["file"]
            expected_name = hashlib.sha256(name.encode()).hexdigest()[:24] + ".safetensors"
            require(relative == directory + "/" + expected_name and relative not in used_paths,
                    "Wrong or reused tensor filename")
            used_paths.add(relative)
            path = _checked_file(root, relative, info["sha256"], info["bytes"])
            require(type(info["total_rows"]) is int and info["total_rows"] == tokens, "Incorrect activation row count")
            if name in expected_shapes:
                n = min(tokens, rows)
                width = expected_shapes[name][1]
                require(type(info["sample_rows"]) is int and info["sample_rows"] == n, "Incorrect sampled row count")
                require(info.get("sampling") == "one uniform row per contiguous stratum"
                        and info.get("moments") == "uncentered all-row sums; diagonal only, not full X^T X",
                        "Wrong sampling or moment semantics")
                expected = {"x": ([n, width], {"BF16"}),
                            "row_indices": ([n], {"I32"}), "sum_x": ([width], {"F32"}),
                            "sum_x_squared": ([width], {"F32"})}
            else:
                require(info.get("sampling") == "all rows", "Incomplete full hidden boundary")
                expected = {"hidden": ([1, tokens, HIDDEN_SIZE], {"BF16"})}
            validate_tensor_file(path, info["tensors"], expected, tokens)
            total_bytes += info["bytes"]
            tensor_count += len(expected)
        require({str(path.relative_to(root)) for path in (root / directory).glob("*.safetensors")} == used_paths,
                "Unreferenced tensor files")
        total_tokens += tokens
    require(manifest.get("exported_tensor_bytes") == total_bytes, "Total exported bytes mismatch")
    return {"status": "validated", "scope": "CPU artifact integrity and projection coverage only",
            "manifest_sha256": sha(root / "manifest.json"), "sequences": len(sequences),
            "tokens": total_tokens, "projections_per_sequence": len(expected_shapes),
            "tensors": tensor_count, "tensor_bytes": total_bytes,
            "limitations": ["Does not rerun tokenizer or model", "Model artifact descriptors are checked, original weight bytes are not reread",
                            "Does not validate GSQ/RCO numerical parity"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = validate_package(args.package)
    except (ValueError, OSError, KeyError, TypeError) as error:
        report = {"status": "failed", "error": str(error)}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "validated" else 1)
