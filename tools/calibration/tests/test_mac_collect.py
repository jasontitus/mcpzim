import json
from pathlib import Path
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mac_collect import Sink, install, sample_indices


def test_sampling_is_reproducible_unique_and_bounded():
    indices = sample_indices(99, 12, "sample")
    assert np.array_equal(indices, sample_indices(99, 12, "sample"))
    assert len(set(indices.tolist())) == 12
    assert 0 <= indices.min() <= indices.max() < 99
    assert np.array_equal(sample_indices(4, 64, "sample"), np.arange(4))


def test_interception_preserves_hybrid_forward_and_captures_every_projection(tmp_path):
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    args = ModelArgs(model_type="qwen3_5", text_config={
        "model_type": "qwen3_5_text", "hidden_size": 64,
        "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 128, "full_attention_interval": 2,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 64, "linear_value_head_dim": 64,
        "rope_parameters": {"type": "default", "rope_theta": 10000,
                            "partial_rotary_factor": 1.0}})
    model = Model(args)
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    before = model(tokens)
    mx.eval(before)
    sink = Sink(tmp_path / "capture", rows=3, free_reserve=0)
    sink.begin("call0")
    inventory = install(model, sink)
    after = model(tokens)
    mx.eval(after)
    np.testing.assert_allclose(np.array(before), np.array(after), rtol=2e-5, atol=2e-5)
    assert set(sink.records) == set(inventory) - {"language_model.lm_head"}
    assert len(inventory) == 16  # linear attention 5 + full attention 4 + MLP 6 + head
    record = sink.records["language_model.model.layers.0.linear_attn.in_proj_a"]
    arrays = mx.load(str(sink.root / record["file"]))
    assert arrays["x"].shape == (3, 64)
    assert arrays["sum_x"].shape == (64,)
    assert record["total_rows"] == 5


def test_moments_use_all_rows_not_only_samples(tmp_path):
    sink = Sink(tmp_path / "capture", rows=2, free_reserve=0)
    sink.begin("call")
    x = mx.array(np.arange(24, dtype=np.float32).reshape(1, 8, 3))
    sink.projection("linear", x)
    arrays = mx.load(str(sink.root / sink.records["linear"]["file"]))
    np.testing.assert_allclose(np.array(arrays["sum_x"]), np.array(x).sum(axis=(0, 1)))
    np.testing.assert_allclose(np.array(arrays["sum_x_squared"]), (np.array(x)**2).sum(axis=(0, 1)))
    np.testing.assert_array_equal(np.array(arrays["x"]), np.array(x)[0, np.array(arrays["row_indices"])])


def test_budget_nonfinite_and_quantized_model_rejection(tmp_path):
    sink = Sink(tmp_path / "capture", byte_limit=1, free_reserve=0)
    sink.begin("call")
    with pytest.raises(RuntimeError, match="budget"):
        sink.projection("linear", mx.ones((1, 8, 3)))
    assert not list(sink.root.rglob("*.safetensors"))
    with pytest.raises(ValueError, match="Nonfinite"):
        sink.projection("linear", mx.array([[[float("nan")]]]))
    model = nn.Sequential(nn.Linear(64, 64))
    nn.quantize(model, bits=4)
    with pytest.raises(ValueError, match="Quantized"):
        install(model, sink)


def test_repeated_module_call_cannot_silently_replace_capture(tmp_path):
    sink = Sink(tmp_path / "capture", free_reserve=0)
    sink.begin("call")
    sink.projection("linear", mx.ones((1, 3, 4)))
    with pytest.raises(ValueError, match="Repeated"):
        sink.projection("linear", mx.ones((1, 3, 4)))


def test_finite_input_with_overflowed_moments_is_rejected(tmp_path):
    sink = Sink(tmp_path / "overflow", free_reserve=0)
    sink.begin("call")
    with pytest.raises(ValueError, match="moments/FP32 overflow"):
        sink.projection("linear", mx.full((1, 2, 3), 1e30, dtype=mx.float32))
    assert not sink.records


def test_expected_inventory_detects_same_count_wrong_shape_and_name():
    from mac_collect import expected_projection_shapes, verify_projection_inventory
    shapes = expected_projection_shapes()
    assert len(shapes) == 497
    inventory = {name: {"weight_shape": list(shape)} for name, shape in shapes.items()}
    verify_projection_inventory(inventory)
    inventory["language_model.lm_head"]["weight_shape"] = [248320, 4096]
    with pytest.raises(ValueError, match="names/shapes"):
        verify_projection_inventory(inventory)
    inventory["language_model.lm_head"]["weight_shape"] = [248320, 5120]
    inventory["wrong.head"] = inventory.pop("language_model.lm_head")
    with pytest.raises(ValueError, match="names/shapes"):
        verify_projection_inventory(inventory)


def test_projection_width_checked_before_capture(tmp_path):
    sink = Sink(tmp_path / "width", free_reserve=0)
    sink.begin("call")
    sink.expected_shapes = {"linear": [12, 5]}
    with pytest.raises(ValueError, match="width/name"):
        sink.projection("linear", mx.ones((1, 2, 4)))


def test_source_ordinals_and_schema_fail_closed(tmp_path):
    import hashlib
    from mac_collect import validate_source_capture
    record = {"schemaVersion": 1, "ordinal": 0, "conversationID": "case", "turn": 1,
              "prompt": "hello", "promptSHA256": hashlib.sha256(b"hello").hexdigest(), "tokenIDs": [1, 2]}
    path = tmp_path / "invocation-000000.json"
    path.write_text(json.dumps(record))
    paths, hashes = validate_source_capture(tmp_path, {"status": "completed", "invocationCount": "1"})
    assert paths == [path] and len(hashes[0]) == 64
    for key, value, error in (("ordinal", 1, "ordinal"), ("ordinal", True, "ordinal"),
                              ("tokenIDs", [True], "tokens"), ("turn", -1, "turn"),
                              ("schemaVersion", 2, "schema"), ("prompt", "tampered", "hash")):
        path.write_text(json.dumps({**record, key: value}))
        with pytest.raises(ValueError, match=error):
            validate_source_capture(tmp_path, {"status": "completed", "invocationCount": "1"})
    path.write_text(json.dumps(record))
    path.rename(tmp_path / "invocation-000001.json")
    with pytest.raises(ValueError, match="ordinal"):
        validate_source_capture(tmp_path, {"status": "completed", "invocationCount": "1"})


def test_provenance_rehashes_actual_files_and_rejects_extra_shards(tmp_path):
    from mac_collect import MODEL, REVISION, verify_model_artifacts, sha
    metadata_names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "generation_config.json", "merges.txt", "vocab.json", "model.safetensors.index.json")
    def entry(name):
        path = tmp_path / name
        path.write_bytes(b"original")
        return {"bytes": path.stat().st_size, "sha256": sha(path)}
    identity = {"model": MODEL, "revision": REVISION}
    weights = {**identity, "files": {"model-00001-of-00001.safetensors": entry("model-00001-of-00001.safetensors")}}
    metadata = {**identity, "files": {name: entry(name) for name in metadata_names}}
    validation = {**identity, "status": "validated", "precision": "BF16", "verified_files": weights["files"]}
    assert len(verify_model_artifacts(tmp_path, validation, metadata, weights)["files"]) == 9
    (tmp_path / "config.json").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash/size"):
        verify_model_artifacts(tmp_path, validation, metadata, weights)
    (tmp_path / "config.json").write_bytes(b"original")
    (tmp_path / "extra.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="extra or missing"):
        verify_model_artifacts(tmp_path, validation, metadata, weights)
    with pytest.raises(ValueError, match="revision"):
        verify_model_artifacts(tmp_path, {**validation, "revision": "wrong"}, metadata, weights)


def test_memory_guard_checks_active_budget_and_system_reserve():
    from mac_collect import MemoryGuard
    MemoryGuard(100, 20, available=lambda: 80, active=lambda: 10).check(30)
    with pytest.raises(RuntimeError, match="budget/headroom"):
        MemoryGuard(100, 20, available=lambda: 1000, active=lambda: 80).check(21)
    with pytest.raises(RuntimeError, match="reserve"):
        MemoryGuard(100, 20, available=lambda: 39, active=lambda: 0).check(20)


def test_disk_estimate_accounts_for_all_calls_and_boundaries_before_load(tmp_path):
    from mac_collect import estimate_export_bytes, check_export_budget, expected_projection_shapes
    paths = []
    for i, count in enumerate((2, 100)):
        path = tmp_path / f"invocation-{i:06d}.json"
        path.write_text(json.dumps({"tokenIDs": [1] * count}))
        paths.append(path)
    shapes = expected_projection_shapes()
    dims = sum(shape[1] for shape in shapes.values())
    assert dims == 3396608
    expected = 4 * 1024**2
    for path, count in zip(paths, (2, 100)):
        n = min(count, 32)
        expected += dims * (n * 2 + 8) + 497 * n * 4 + count * 5120 * 4
        expected += 499 * 4096 + path.stat().st_size + 1024**2
    assert estimate_export_bytes(paths, 32) == expected
    assert estimate_export_bytes(paths, 64) > expected
    assert check_export_budget(paths, 32, expected, expected + 20, 20) == expected
    with pytest.raises(RuntimeError, match="do not truncate"):
        check_export_budget(paths, 32, expected - 1, expected + 20, 20)
    with pytest.raises(RuntimeError, match="disk reserve"):
        check_export_budget(paths, 32, expected, expected + 19, 20)


def test_head_reconstruction_from_full_bf16_hidden_matches_native_logits(tmp_path):
    # Tiny original-style text model: full vocabulary here, no full model loaded.
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    model = Model(ModelArgs(model_type="qwen3_5", text_config={
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "vocab_size": 128, "full_attention_interval": 2,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 64, "linear_value_head_dim": 64,
        "rope_parameters": {"type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0}}))
    from mlx.utils import tree_map
    model.update(tree_map(lambda x: x.astype(mx.bfloat16), model.parameters()))
    model.eval()
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    native = model(tokens)
    body = model.language_model.model
    embeddings = body.embed_tokens(tokens)
    hidden = body(tokens, cache=None, input_embeddings=embeddings)
    sink = Sink(tmp_path / "head", free_reserve=0)
    sink.begin("call")
    sink.full("first_block_input", embeddings)
    sink.full("final_normalized_hidden", hidden)
    recovered = mx.load(str(sink.root / sink.records["final_normalized_hidden"]["file"]))["hidden"]
    reconstructed = model.language_model.lm_head(recovered)
    mx.eval(native, reconstructed)
    assert recovered.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(native.astype(mx.float32)), np.array(reconstructed.astype(mx.float32)))


def test_system_headroom_query_is_cached_but_active_memory_always_checked():
    from mac_collect import MemoryGuard
    calls = []
    active = [0]
    guard = MemoryGuard(100, 20, available=lambda: calls.append(1) or 80, active=lambda: active[0])
    guard.check(10)
    guard.check(10)
    assert len(calls) == 1
    guard.check(10, refresh=True)
    assert len(calls) == 2
    active[0] = 95
    with pytest.raises(RuntimeError, match="budget/headroom"):
        guard.check(10)
