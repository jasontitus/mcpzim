"""Prefill-only original-Qwen capture with bounded disk and MLX graph lifetime.

Exports exact replay tokens, first-block inputs, final normalized hidden states,
and stratified raw input rows plus all-row first/diagonal-second moments for each
linear projection. Sampled rows/diagonal moments DO NOT replace GSQ's nonlinear
block objective. The CUDA consumer regenerates block targets from replay inputs
and original weights, and reconstructs full-vocabulary teacher scores in chunks.
"""
import argparse
import hashlib
import json
import math
import importlib.metadata
import platform
import re
import subprocess
from pathlib import Path
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten
import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BENCHMARKS = Path(__file__).resolve().parents[2] / "docs/benchmarks/quantization-2026-09-19"


def expected_projection_shapes():
    result = {"language_model.lm_head": [248320, 5120]}
    mlp = {"gate_proj": [17408, 5120], "up_proj": [17408, 5120], "down_proj": [5120, 17408]}
    recurrent = {"in_proj_qkv": [10240, 5120], "in_proj_z": [6144, 5120],
                 "in_proj_a": [48, 5120], "in_proj_b": [48, 5120], "out_proj": [5120, 6144]}
    attention = {"q_proj": [12288, 5120], "k_proj": [1024, 5120],
                 "v_proj": [1024, 5120], "o_proj": [5120, 6144]}
    for index in range(64):
        prefix = f"language_model.model.layers.{index}"
        result.update({f"{prefix}.mlp.{name}": shape for name, shape in mlp.items()})
        kind, shapes = ("self_attn", attention) if index % 4 == 3 else ("linear_attn", recurrent)
        result.update({f"{prefix}.{kind}.{name}": shape for name, shape in shapes.items()})
    return result


def verify_projection_inventory(inventory):
    actual = {name: info["weight_shape"] for name, info in inventory.items()}
    if actual != expected_projection_shapes():
        raise ValueError("Projection names/shapes do not match pinned Qwen architecture")


def verify_model_artifacts(root, validation, metadata, expected_weights):
    """Stream hashes again immediately before load; never trust a detached success JSON."""
    root = Path(root).resolve()
    for document in (validation, metadata, expected_weights):
        if document.get("model") != MODEL or document.get("revision") != REVISION:
            raise ValueError("Wrong original model/revision")
    if validation.get("status") != "validated" or validation.get("precision") != "BF16":
        raise ValueError("Original BF16 validation required")
    if validation.get("verified_files") != expected_weights["files"]:
        raise ValueError("Weight validation does not match pinned expected files")
    required_metadata = {"config.json", "tokenizer.json", "tokenizer_config.json",
                         "chat_template.jinja", "generation_config.json", "merges.txt",
                         "vocab.json", "model.safetensors.index.json"}
    if set(metadata["files"]) != required_metadata:
        raise ValueError("Missing or unexpected configuration/tokenizer metadata")
    files = {**expected_weights["files"], **metadata["files"]}
    for name, expected in files.items():
        if Path(name).name != name:
            raise ValueError("Unsafe model artifact path")
        path = root / name
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Current model artifact hash/size mismatch: {name}")
    # mlx_lm loads globbed shards, so reject extra unverified weights too.
    if {p.name for p in root.glob("*.safetensors")} != set(expected_weights["files"]):
        raise ValueError("Unverified extra or missing model shards")
    return {"resolved_model_dir": str(root), "files": files}


def validate_source_capture(capture, manifest):
    paths = sorted(Path(capture).glob("invocation-*.json"))
    if manifest.get("status") != "completed" or not paths or len(paths) != int(manifest["invocationCount"]):
        raise ValueError("Incomplete or empty invocation capture")
    hashes = []
    for index, path in enumerate(paths):
        record = json.loads(path.read_text())
        if path.name != f"invocation-{index:06d}.json" or type(record.get("ordinal")) is not int or record["ordinal"] != index:
            raise ValueError("Noncontiguous or duplicate source invocation ordinal")
        if record.get("schemaVersion") != 1 or not isinstance(record.get("conversationID"), str) or not record["conversationID"]:
            raise ValueError("Invalid source invocation schema/context")
        if type(record.get("turn")) is not int or record["turn"] < 0:
            raise ValueError("Invalid source turn")
        tokens = record.get("tokenIDs")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < 248320 for t in tokens):
            raise ValueError("Invalid source tokens")
        if not isinstance(record.get("prompt"), str) or record.get("promptSHA256") != hashlib.sha256(record["prompt"].encode()).hexdigest():
            raise ValueError("Captured prompt hash mismatch")
        hashes.append(sha(path))
    return paths, hashes


def system_memory():
    """Conservative macOS available estimate: free + inactive + speculative pages."""
    output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+) bytes", output).group(1))
    counts = {name: int(value) for name, value in re.findall(r"^(Pages [^:]+):\s+(\d+)\.", output, re.M)}
    available = sum(counts.get(name, 0) for name in ("Pages free", "Pages inactive", "Pages speculative")) * page_size
    return available


class MemoryGuard:
    def __init__(self, limit, reserve, available=system_memory, active=mx.get_active_memory):
        if limit <= 0 or reserve < 0:
            raise ValueError("Invalid memory limits")
        self.limit, self.reserve, self.available, self.active = limit, reserve, available, active
        self._available_at, self._available_bytes = -math.inf, None

    def check(self, additional=0, refresh=False):
        if self.active() + additional > self.limit:
            raise RuntimeError("MLX memory budget/headroom would be exceeded")
        now = time.monotonic()
        if refresh or now - self._available_at >= 1.0:
            self._available_bytes, self._available_at = self.available(), now
        if self._available_bytes < self.reserve + additional:
            raise RuntimeError("System memory reserve would be consumed")


def estimate_export_bytes(paths, rows):
    """All invocations, BF16 boundaries/X, FP32 moments, int32 indices plus headers."""
    if rows <= 0:
        raise ValueError("Positive sample row count required")
    shapes = expected_projection_shapes()
    dimensions = sum(shape[1] for shape in shapes.values())
    total = 4 * 1024**2  # top-level manifest/provenance and one pending file
    for path in paths:
        count = len(json.loads(path.read_text())["tokenIDs"])
        if not 1 <= count <= 16384:
            raise ValueError("No empty or silently truncated sequences")
        sampled = min(count, rows)
        projections = dimensions * (sampled * 2 + 8) + len(shapes) * sampled * 4
        boundaries = count * 5120 * 2 * 2
        total += projections + boundaries + (len(shapes) + 2) * 4096
        total += path.stat().st_size + 1024**2  # input copy and per-sequence manifest
    return total


def check_export_budget(paths, rows, limit, free_bytes, reserve):
    estimate = estimate_export_bytes(paths, rows)
    if estimate > limit:
        raise RuntimeError(f"Complete export estimate {estimate} exceeds disk budget {limit}; do not truncate the corpus")
    if free_bytes < reserve + estimate:
        raise RuntimeError("Complete export would consume the disk reserve")
    return estimate


def runtime_provenance():
    from mlx_lm.models import qwen3_5
    return {"python": platform.python_version(), "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"), "collector_sha256": sha(__file__),
            "qwen_backend_sha256": sha(qwen3_5.__file__),
            "sanitizer": "qwen3_5 original HF import: prefix map, discard vision/MTP; conv axis2->1; selected RMSNorm weights +1 before inference",
            "teacher_head": "original BF16 head; CUDA parity and global-vocabulary normalization required"}


def sample_indices(count, limit, key):
    """One deterministic uniform sample per contiguous stratum; no padding."""
    if count <= 0 or limit <= 0:
        raise ValueError("Positive row counts required")
    n = min(count, limit)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    edges = np.arange(n + 1, dtype=np.int64) * count // n
    return np.array([rng.integers(edges[i], edges[i + 1]) for i in range(n)], dtype=np.int32)


class Sink:
    def __init__(self, root, rows=32, byte_limit=40 * 1024**3, free_reserve=24 * 1024**3, memory_guard=None):
        self.root = Path(root)
        self.root.mkdir(parents=False, exist_ok=False)
        self.rows, self.byte_limit, self.free_reserve = rows, byte_limit, free_reserve
        self.written = 0
        self.records = {}
        self.current = None
        self.memory_guard = memory_guard
        self.expected_shapes = None
        self.expected_dtype = None

    def begin(self, invocation):
        self.current = str(invocation)
        self.directory = self.root / self.current
        self.directory.mkdir()
        self.records = {}

    def save(self, name, tensors, metadata):
        if name in self.records:
            raise ValueError(f"Repeated projection within unchunked prefill: {name}")
        planned = sum(x.nbytes for x in tensors.values()) + 65536
        if self.written + planned > self.byte_limit:
            raise RuntimeError("Export budget exceeded; partial run remains incomplete")
        if shutil.disk_usage(self.root).free < self.free_reserve + planned:
            raise RuntimeError("Disk reserve would be consumed")
        filename = hashlib.sha256(name.encode()).hexdigest()[:24] + ".safetensors"
        path = self.directory / filename
        temporary = self.directory / ("pending-" + filename)
        mx.eval(*tensors.values())
        mx.save_safetensors(str(temporary), tensors)
        temporary.replace(path)
        size = path.stat().st_size
        self.written += size
        self.records[name] = {"file": str(path.relative_to(self.root)), "bytes": size,
                              "sha256": sha(path), "tensors": {
                                  k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                  for k, v in tensors.items()}, **metadata}

    def projection(self, name, x):
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("Collector requires a single unpadded complete sequence")
        if self.expected_dtype is not None and x.dtype != self.expected_dtype:
            raise ValueError(f"Unexpected projection activation dtype: {name}")
        if self.expected_shapes is not None and (name not in self.expected_shapes or x.shape[-1] != self.expected_shapes[name][1]):
            raise ValueError(f"Projection input width/name mismatch: {name}")
        if self.memory_guard:
            self.memory_guard.check(x.size * 8 + min(x.shape[1], self.rows) * x.shape[-1] * 4)
        flat = x.reshape(-1, x.shape[-1])
        indices = sample_indices(flat.shape[0], self.rows, self.current)
        selected = mx.array(indices)
        float_x = flat.astype(mx.float32)
        sampled = flat[selected]
        first = mx.sum(float_x, axis=0)
        diagonal = mx.sum(float_x * float_x, axis=0)
        finite = mx.all(mx.isfinite(float_x))
        moments_finite = mx.all(mx.isfinite(first)) & mx.all(mx.isfinite(diagonal))
        mx.eval(sampled, first, diagonal, finite, moments_finite)
        if not bool(finite.item()):
            raise ValueError(f"Nonfinite input: {name}")
        if not bool(moments_finite.item()):
            raise ValueError(f"Nonfinite moments/FP32 overflow: {name}")
        if self.memory_guard:
            self.memory_guard.check()
        self.save(name, {"x": sampled, "row_indices": selected,
                         "sum_x": first, "sum_x_squared": diagonal},
                  {"total_rows": flat.shape[0], "sample_rows": len(indices),
                   "sampling": "one uniform row per contiguous stratum",
                   "moments": "uncentered all-row sums; diagonal only, not full X^T X"})

    def full(self, name, x):
        if self.expected_dtype is not None and (x.dtype != self.expected_dtype or x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 5120):
            raise ValueError(f"Unexpected full boundary shape/dtype: {name}")
        if self.memory_guard:
            self.memory_guard.check(x.nbytes)
        mx.eval(x)
        if not bool(mx.all(mx.isfinite(x)).item()):
            raise ValueError(f"Nonfinite input: {name}")
        self.save(name, {"hidden": x}, {"total_rows": x.shape[1], "sampling": "all rows"})


class CaptureLinear(nn.Linear):
    def __init__(self, original, name, sink):
        nn.Module.__init__(self)
        self.weight = original.weight
        if "bias" in original:
            self.bias = original.bias
        object.__setattr__(self, "_capture_name", name)
        object.__setattr__(self, "_capture_sink", sink)

    def __call__(self, x):
        self._capture_sink.projection(self._capture_name, x)
        return super().__call__(x)


def install(model, sink):
    inventory, replacements = {}, []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            raise ValueError("Quantized model cannot supply reference activations")
        if isinstance(module, nn.Linear):
            inventory[name] = {"weight_shape": list(module.weight.shape),
                               "dtype": str(module.weight.dtype)}
            # Avoid allocating sequence_length x vocabulary logits. Capture
            # the complete final hidden state and sample the head boundary.
            if name != "language_model.lm_head":
                replacements.append((name, CaptureLinear(module, name, sink)))
    model.update_modules(tree_unflatten(replacements))
    return inventory


def collect(args):
    from mlx_lm import load
    capture = Path(args.capture_dir)
    source_manifest = json.loads((capture / "capture-run.json").read_text())
    validation = json.loads(Path(args.validation).read_text())
    if source_manifest.get("status") != "completed" or validation.get("status") != "validated":
        raise ValueError("Completed prompt capture and verified original weights required")
    if validation.get("model") != "Qwen/Qwen3.8-27B" or validation.get("precision") != "BF16":
        raise ValueError("Original BF16 Qwen required")
    paths, source_hashes = validate_source_capture(capture, source_manifest)
    metadata_path = Path(getattr(args, "metadata_manifest", BENCHMARKS / "baseline-metadata-manifest.json"))
    hashes_path = Path(getattr(args, "weight_hashes", BENCHMARKS / "original-weight-hashes.json"))
    disk_estimate = check_export_budget(paths, args.rows_per_call, int(args.max_export_gib * 1024**3),
        shutil.disk_usage(Path(args.output_dir).parent).free, int(args.min_free_gib * 1024**3))
    artifacts = verify_model_artifacts(args.model_dir, validation, json.loads(metadata_path.read_text()),
                                       json.loads(hashes_path.read_text()))
    artifacts["metadata_manifest_sha256"] = sha(metadata_path)
    guard = MemoryGuard(int(getattr(args, "max_memory_gib", 80) * 1024**3),
                        int(getattr(args, "memory_reserve_gib", 16) * 1024**3))
    guard.check(validation["weight_file_bytes"] + 4 * 1024**3)
    suite_path = Path(source_manifest["suitePath"])
    suite_bytes = suite_path.read_bytes()
    suite_data = json.loads(suite_bytes)
    if (suite_data.get("schema_version", suite_data.get("schemaVersion")) != 1
            or not suite_data.get("conversations")):
        raise ValueError("Invalid source calibration suite")
    source_manifest_bytes = (capture / "capture-run.json").read_bytes()
    if json.loads(source_manifest_bytes) != source_manifest:
        raise ValueError("Capture manifest changed during preflight")
    sink = Sink(args.output_dir, args.rows_per_call, int(args.max_export_gib * 1024**3),
                int(args.min_free_gib * 1024**3), memory_guard=guard)
    (sink.root / "capture-run.json").write_bytes(source_manifest_bytes)
    (sink.root / "calibration-suite.json").write_bytes(suite_bytes)
    source_files = {name: {"file": filename, "bytes": (sink.root / filename).stat().st_size,
                          "sha256": sha(sink.root / filename)}
                    for name, filename in (("capture_manifest", "capture-run.json"),
                                            ("calibration_suite", "calibration-suite.json"))}
    manifest = {"schema_version": 1,
                "source_invocation_count": len(paths), "source_invocation_sha256": source_hashes,
                "source_files": source_files, "status": "running",
                "model_artifacts": artifacts, "runtime_provenance": runtime_provenance(),
                "memory_policy": {"mlx_limit_bytes": guard.limit, "system_reserve_bytes": guard.reserve,
                    "allocator_limit_is_advisory": True, "checks": "system headroom before load/prefill and at most once/second during projections; active MLX budget at each projection"}, "scope": "original BF16 prefill; no generation",
                "model": validation["model"], "revision": validation["revision"],
                "weight_validation_sha256": sha(args.validation),
                "capture_manifest_sha256": sha(capture / "capture-run.json"),
                "rows_per_projection_per_invocation": args.rows_per_call,
                "estimated_complete_export_bytes": disk_estimate,
                "sequences": [], "limitations": [
                    "Raw projection X is sampled; first and diagonal-second moments cover all rows",
                    "No full Gram matrices; diagonal moments cannot replace full Hessians",
                    "GSQ must recompute nonlinear block targets and quantized-prefix inputs from replay data",
                    "RCO head reconstruction from final hidden requires CUDA/MLX numerical parity",
                    "All sequences use fresh recurrent/attention state, no padding and implicit positions 0..N-1"]}
    manifest_path = sink.root / "manifest.json"
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        mx.set_memory_limit(guard.limit)
        mx.set_cache_limit(512 * 1024**2)
        model, tokenizer = load(args.model_dir, lazy=False)
        guard.check()
        if any(v.dtype != mx.bfloat16 for _, v in tree_flatten(model.parameters())):
            raise ValueError("Unexpected parameter dtype in full-precision baseline")
        inventory = install(model, sink)
        if len(model.language_model.model.layers) != 64 or len(inventory) != 497:
            raise ValueError("Unexpected hybrid projection coverage")
        verify_projection_inventory(inventory)
        sink.expected_shapes = expected_projection_shapes()
        sink.expected_dtype = mx.bfloat16
        manifest["projections"] = inventory
        atomic_json(manifest_path, manifest)
        for index, path in enumerate(paths):
            if sha(path) != source_hashes[index]:
                raise ValueError("Source invocation changed after preflight validation")
            record = json.loads(path.read_text())
            if record["promptSHA256"] != hashlib.sha256(record["prompt"].encode()).hexdigest():
                raise ValueError("Captured prompt hash mismatch")
            tokens = tokenizer.encode(record["prompt"])
            if tokens != record["tokenIDs"]:
                raise ValueError("Original tokenizer does not reproduce captured token IDs")
            if not tokens or len(tokens) > 16384:
                raise ValueError("No empty or silently truncated sequences")
            # Conservative temporary workspace estimate, not an exact peak predictor.
            guard.check(max(4 * 1024**3, len(tokens) * 17408 * 4 * 12), refresh=True)
            sink.begin(f"invocation-{index:06d}")
            shutil.copyfile(path, sink.directory / "input.json")
            token_array = mx.array([tokens], dtype=mx.int32)
            body = model.language_model.model
            embeddings = body.embed_tokens(token_array)
            sink.full("first_block_input", embeddings)
            hidden = body(token_array, cache=None, input_embeddings=embeddings)
            sink.full("final_normalized_hidden", hidden)
            sink.projection("language_model.lm_head", hidden)
            if set(sink.records) != set(inventory) | {"first_block_input", "final_normalized_hidden"}:
                raise ValueError("Missing projection input coverage")
            sequence = {"source_capture_sha256": sha(path), "input_sha256": sha(sink.directory / "input.json"),
                        "conversation_id": record["conversationID"], "turn": record["turn"],
                        "tokens": len(tokens), "fresh_state": True, "files": sink.records}
            atomic_json(sink.directory / "sequence.json", sequence)
            manifest["sequences"].append({"directory": sink.current, "tokens": len(tokens),
                "manifest_sha256": sha(sink.directory / "sequence.json")})
            manifest["exported_tensor_bytes"] = sink.written
            manifest["elapsed_seconds"] = time.monotonic() - started
            manifest["mlx_peak_memory_bytes"] = mx.get_peak_memory()
            atomic_json(manifest_path, manifest)
            print(f"Captured {index+1}/{len(paths)}: {len(tokens)} tokens, {sink.written/1024**3:.2f} GiB written", flush=True)
            del token_array, embeddings, hidden
            mx.clear_cache()
        manifest["status"] = "completed"
        atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("capture-dir", "model-dir", "validation", "output-dir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--metadata-manifest", default=str(BENCHMARKS / "baseline-metadata-manifest.json"))
    parser.add_argument("--weight-hashes", default=str(BENCHMARKS / "original-weight-hashes.json"))
    parser.add_argument("--max-memory-gib", type=float, default=80)
    parser.add_argument("--memory-reserve-gib", type=float, default=16)
    parser.add_argument("--rows-per-call", type=int, default=32)
    parser.add_argument("--max-export-gib", type=float, default=40)
    parser.add_argument("--min-free-gib", type=float, default=24)
    collect(parser.parse_args())
