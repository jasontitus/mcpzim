# GSQ Q1 candidates to Zimfo's Prism GGUF runtime

The packing implementation has **CPU-tested binary compatibility and a successful full Qwen3.8-27B runtime load and prefill**. The full-model fixture uses simple round-to-nearest initialization solely to verify export and runtime compatibility. It is not a learned GSQ result and establishes neither answer quality, cross-backend numerical parity, Metal performance nor phone viability.

## Binary and solver contract

The shipping XCFramework pins Prism llama.cpp commit `62061f91088281e65071cc38c5f69ee95c39f14e`. Its Q1_0 type is **41**, group size **128**, and block size **18 bytes**: a little-endian FP16 scale followed by sixteen sign bytes. Column `8*j+k` occupies bit `k` of byte `j`; one means positive. Each row is divided into contiguous groups, with no cross-row grouping. This is neither ternary nor the stock TQ1_0 format.

The solver's `zimfo-q1-v1` safetensors candidate contains:

- `codes`: UINT8 `[output_rows, input_columns / 8]`, positive signs packed little-bit-first.
- `scales`: BF16 `[output_rows, input_columns / 128]`, the **effective scales from the hard candidate**, not unrounded FP32 optimizer parameters.
- Metadata: `format=zimfo-q1-v1`, `group_size=128`, `bit_order=little`, `effective_scale_dtype=BF16`.

GSQ's hard rule is `sign_logits > 0`; a zero logit becomes negative. The bridge preserves that rule. It never recomputes scales using mean absolute weights. Signed scales remain signed, matching the runtime's `±d` interpretation.

`packing.q1.pack(codes, effective_scales)` returns packed UINT8 rows and measured packing-error statistics. FP16 scale overflow always fails. Rounding or underflow that changes the hard candidate fails by default and requires an explicit `allow_scale_loss` choice. When accepted, the report includes changed groups, maximum scale error, and summed squared error over all represented weights. These errors are additional to the original quantization error. Check scale representability during optimization to avoid discovering an unexportable candidate at the end.

## Qwen layout and protected tensors

Original Qwen HF tensor names map through the pinned converter to runtime names. The text deployment intentionally omits vision and MTP tensors. The exporter requires an explicit choice for **every two-dimensional text weight, including token embeddings and the output head**; it cannot silently retain large high-precision matrices while reporting a one-bit model budget.

Hybrid recurrent attention needs the converter's grouped-to-tiled value-head layout. The bridge permutes QKV value rows, Z rows, and A/B rows together with their learned scales. The output projection's columns are permuted in whole 128-column groups; a head configuration splitting those groups is rejected. Tests compare all five transformed projection families against the actual pinned converter.

Unquantized tensors use the pinned converter's transformations, including value-channel ordering for convolution, recurrent scalar ordering, `A_log -> -exp(A_log)`, `dt_bias` renaming, and the selected normalization `weight + 1` convention. One-dimensional parameters remain FP32. The converter performs these transforms after expanding source BF16 to FP32; MLX may apply its normalization adjustment at BF16 precision. Full-Qwen cross-backend parity must measure that difference rather than silently assume identical reference numerics.

## Serializer and exact budgets

`packing.gguf.write` writes GGUF v3 metadata, reversed storage dimensions, tensor descriptors, 32-byte alignment and streamed payloads. It validates payload sizes and SHA-256, parses the declared metadata count, rejects incompatible alignment, and publishes the final file exclusively after fsync. Failure cleanup only removes this writer's own partial file; a preexisting partial belongs to its original writer. Parent-directory fsync completes local publication. This is local durable publication, not GCS checkpointing.

`packing.export_qwen` uses the pinned converter to generate the real vocabulary/model metadata. It computes the final byte count from that metadata and all transformed tensor declarations **before reading candidate tensor bodies**. It checks an optional maximum GGUF size and disk space for staged payloads plus the final file. The final serialized size must match this preflight value.

A local layout calculation with the original Qwen metadata produced 851 runtime tensors and the following exact budgets for that metadata/name configuration:

| Layout | Serialized bytes | Difference from published Bonsai |
|---|---:|---:|
| All 498 eligible 2D text tensors Q1; other tensors FP32 | 3,803,453,760 | +1,280 |
| Same, but recurrent A/B projection weights FP32 | 3,894,507,840 | +91,055,360 (about 2.39%) |

The additive RCO cost manifest is `tools/calibration/packing/evidence/qwen-cost-manifest.json`: its fixed container/protected cost is 21,576,000 bytes; add each selected tensor’s padded `q1_bytes` or `bf16_bytes`. Tests verify mixed choices against the serializer. Regenerate via `cost_manifest` whenever metadata changes. The generated metadata-only GGUF was 10,943,650 bytes. The first layout has also been serialized and loaded successfully in the full-model compatibility experiment below. These remain **layout budgets**, not completed learned model files or quality results. Metadata changes can change the exact number; the exporter recomputes it for every plan. Both scenarios include one-bit embeddings and output head. Models retaining either at higher precision must use their actual resulting budget.

## Export invocation

The adapter accepts a JSON plan containing `model_dir`, `weight_hashes`, `tensor_inventory`, `metadata_manifest`, and `choices`. The three provenance documents are the pinned baseline files under `docs/benchmarks/quantization-2026-09-19/`. It rechecks current source shard hashes and original tensor coverage, then hashes config/tokenizer files. Source shards or converter code outside the pinned sets are rejected.

`choices` maps each original HF 2D tensor name to either `{"mode":"q1","candidate":"/path/to/candidate.safetensors","sha256":"..."}`, `{"mode":"bf16"}`, `{"mode":"f16"}`, or `{"mode":"f32"}`. `bf16` uses GGML type30 and preserves original BF16 bits for reference choices; `f16` is an explicit alternative that can add rounding. Other parameters automatically use the protected FP32 route. Optional plan fields are `allow_scale_loss` (default false), `max_gguf_bytes` and `disk_reserve_bytes` (default 10 GiB).

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python \
  -m packing.export_qwen --plan /path/to/export-plan.json \
  --prism-source /path/to/pinned-prism-checkout \
  --output-dir /path/to/new-export-directory
```

The production image must include the pinned Prism source checkout (or an archive with a verified `zimfo-prism-source.json` receipt), `gguf-py`, converter dependencies and this package. That image integration is separate from the locally tested adapter; do not assume the current cloud image already contains it. Staged Linux paths must regenerate the plan rather than reuse Mac absolute paths.

The output directory must be new. A running/failed/completed `report.json` records tensor source/candidate provenance, packing errors, explicit embedding/head strategies, the exact size budget, final SHA-256 and serialized byte accounting. Tensor payloads remain available for auditing; completed candidate generation and full-model runtime validation are separate gates.

## Verification completed and limitations

The local CPU tests exercise:

- Exact sign ordering, zero-logit behavior, negative scales and multiple groups/rows.
- An actual pinned GSQ `GumbelQuantizer1Bit.get_hard_weights()` BF16 candidate packing/unpacking with zero added error.
- Equality with the **shipped XCFramework's** `dequantize_row_q1_0`, not only a Python roundtrip.
- Group-preserving Qwen recurrent layout equality against the pinned converter, including normalization and recurrent A transforms.
- Safetensors dtype, metadata, dimensions and candidate hash validation.
- Serialization byte accounting, immutable output behavior, retained unrelated partial files, metadata count/alignment rejection and failed-hash cleanup.

An actual 20,736-byte synthetic one-layer Llama GGUF contains nine Q1 matrices (including embeddings and output head) and three FP32 norms. `runtime_smoke.c` linked against the shipped macOS XCFramework loaded it and ran three-token prefill on **CPU only**, yielding 32 finite logits. Explicit CPU devices and disabled KV/operation offload prevent an accidental Metal context. The fixture is not a quantized Qwen model and its random outputs carry no quality meaning.

Run the tests with `PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python -m pytest tools/calibration/packing/tests -q`. Set `ZIMFO_PRISM_SOURCE` and `ZIMFO_GSQ_SOURCE` for staged source locations. Integration tests explicitly skip when the pinned source checkout or shipped framework is unavailable; a skip is not compatibility evidence. `tiny_fixture.py` and `runtime_smoke.c` provide the reproducible full-container smoke. The full-model compatibility experiment below additionally exercises Qwen export and runtime load. Learned candidates still require their own export/load check, CPU/Metal numerical comparisons, inference speed measurements and phone testing after optimization.

## Full 27B compatibility experiment (2026-09-19)

The original pinned BF16 weights were streamed in bounded CPU row batches to create **498 RTN candidates**, including embeddings and the output head. No full 55 GB model, CUDA or Metal context was loaded during candidate creation. The candidate rule was positive iff `weight > 0`, with a group-128 FP32 mean-absolute scale rounded to BF16. Candidate creation took about 98 seconds and export about 103 seconds on this Mac. Source shard hashes, candidate hashes and the export plan were retained with the ignored experiment artifacts.

The exporter produced **3,803,453,760 bytes** containing all **851 runtime tensors**: 498 Q1 matrices and 353 protected FP32 tensors. The shipped Prism XCFramework loaded that actual full Qwen GGUF on CPU and completed three-token prefill, producing **248,320 finite logits**. This catches missing tensors, unsupported tensor shapes, hybrid attention layout incompatibility and container/load failures before renting a GPU. It does not compare logits to a full-precision oracle or measure answer quality.

The public evidence receipt is [`full-qwen-layout-proof.json`](../tools/calibration/packing/evidence/full-qwen-layout-proof.json). The ignored GGUF, logs, candidate provenance and export report are under `tools/calibration/runs/q1-layout-probe-20260919/`. The minimum 24 GiB free-disk reserve was preserved. The export command's outer timing wrapper could not access a sandboxed macOS timing counter; the exporter completed, and the subsequent independent runtime process exited successfully.

The fixture explicitly allowed FP16 scale rounding for this experiment: **171 groups** changed, with maximum scale error **2.9802322387695312e-08** and total squared represented-weight error **1.7990942069445737e-11**. These are packing errors, not total quantization errors. The production solver now constrains effective scales to the BF16/FP16 intersection during optimization. Consequently, these RTN candidate files must be canonicalized and rehashed before reuse under that newer initializer identity; they cannot be silently relabeled as exact production seeds. No fixture output replaces the calibration corpus, full-precision baseline or held-out comparison results.

## Adversarial review

Parent review found an ownership bug in failure cleanup: an exclusive-open failure could delete another writer's existing `.partial`. The writer now tracks ownership, preserves preexisting files, fsyncs the publication directory and has a regression test. Parent also requested explicit metadata alignment/count validation; both are enforced rather than assumed.

Independent review of solver component units confirmed full-vocabulary normalization is global across vocabulary chunks and flagged scalar-probability validation, candidate dtype/bit-order checks, immutable candidate reads, finite-sign validation and BF16 allocation-gradient parity for the solver owner. Independent review of GCS transport found no blocker under its documented frozen-source/synchronous/single-owner contract and recommended typed manifest payload validation plus hashing bytes as they enter a bundled tar to catch source mutation during bundling. Those reviews do not certify full-model optimization, which remains a separate gate.

A subsequent independent boundary-head check compared the chunked manual gradient with the actual pinned upstream GSQ custom autograd function on tiny tensors. Only its CUDA RNG access was redirected to CPU, retaining the upstream forward/backward arithmetic and identical chunk seeds. BF16 sign-logit and scale gradients matched exactly; loss differed by 1.19e-7. FP32 gradient differences were below 3e-7. This is local arithmetic evidence, not proof of full-model CUDA memory feasibility. Review also found that the streamed BF16 linear allocation gradient omitted the cast of its weight gradient back to BF16. The solver owner corrected it; independent dense-oracle regressions pass with one-row, partial and full-row chunks.


The final integration review caught two recovery defects before GPU execution: the GPU container initially exposed validated input files writable, and completed-block recovery could republish different archive metadata at the same optimizer cursor. Inputs now receive a nested read-only mount. Checkpoint names include a successful-publication ordinal, preserving immutable versions when phase artifacts change without another optimizer update. The reproducer showed identical candidate tensor contents but different tar file modes after extraction. Independent reruns passed **34 solver/orchestration/gradient tests**, including recovery at completed blocks, cold-host configuration recovery, missing-original-report recovery and the BF16 gradient regressions. Cold recovery now checks free space before downloading; it conservatively reserves six times committed payload bytes plus 10 GiB and retains prior evidence. Diagnostic smoke checkpoints are explicitly distinguished from production-stage recovery.

The launcher review also verified the one-hour infrastructure deadline, no dependency installation or image pulls during GPU startup, immutable runtime/source identities, retained-disk ownership checks and generation-pinned remote checkpoint checks. A real harmless subprocess test confirmed deadline expiry terminates and reaps the child. These checks support a bounded GPU feasibility attempt after the prepared image passes its CPU checks; they do not establish that the 27B workload fits CUDA memory or produces a useful quantized model.
