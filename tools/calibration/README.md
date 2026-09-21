# RCO feasibility probes

These are numerical implementation probes, not a calibration collector or a
working GSQ/RCO quantizer. They do not download a model or alter the app.

```sh
uv venv tools/calibration/.venv
uv pip install --python tools/calibration/.venv/bin/python -r tools/calibration/requirements-probe.txt
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python -m pytest tools/calibration/tests -q
```

`streamed_linear_probe.py` keeps frozen reference weights and candidate
differences on CPU, transfers row tiles, and implements their first-order
backward explicitly. The tests use independent dense PyTorch autograd as the
oracle and finite differences as a second check. They cover connected nonlinear
layers, soft and hard straight-through probabilities, input and allocation
gradients, an Adam update, uneven tiles, and zero-probability candidates.

The Metal check skips if the executing environment cannot access Metal. CPU
success does not establish CUDA compatibility, BF16 behavior, or a memory bound
for the complete model. The prototype holds all candidate storage in host RAM;
production requires a bounded disk-backed candidate cache, block recomputation,
hybrid attention coverage, exact chunked loss, and integration with RCO's budget
solver. It does not support higher-order gradients, trainable candidate weights,
or mutation of candidate storage during a forward/backward pair.

See [the research and feasibility plan](../../docs/QUANTIZATION_PLAN_2026-09-19.md).

## CUDA preflight and prepared GCP launch

Run locally to verify the runner without claiming GPU results:

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python tools/calibration/cuda_preflight.py --device cpu --profile tiny --output /tmp/zimfo-cpu-preflight.json
```

The CUDA target profile covers FP32/BF16 gradient parity and the actual chosen
model's MLP projection dimensions (5120 to 17408 and reverse) at 2048, 4096 and
7330 tokens, with seven candidate choices. It checks three repeated iterations,
PyTorch peak allocated/reserved memory under 20 GiB and live allocations after
release. It does not model complete attention/recurrent blocks, head loss or
whole-model storage. Memory success is limited to these projections. Full
device-driver memory peaks and full-model host-cache bounds remain future work.

Prepare a reviewed, allowlisted archive and launch command **without renting**:

```sh
tools/calibration/.venv/bin/python tools/calibration/gcp_preflight.py --output-dir tools/calibration/runs/my-prepared-run
```

**Paid execution is currently disabled, including `--execute`.** The historical
projection launcher does not meet the approved full-model readiness gate.
The retained launch/cleanup implementation has tests, but there is no bypass
flag to spend on the incomplete pipeline. Its guest script now requires an
already installed environment and never installs packages on the GPU host.
See the [GPU readiness runbook](../../docs/QUANTIZATION_GPU_READINESS.md) for the
replacement workflow and remaining launch gates.

The prepared synthetic run costs approximately $0.53 for a full hour at the
September 19 public Iowa rates, including 100 GiB disk and ephemeral IPv4,
before small result egress/tax. Recheck before launch. This is not a full
quantization estimate. No GPU has been launched as part of preparation.

## Actual app input capture

The headless discussion CLI now accepts `--capture-model-inputs NEW_DIRECTORY`
alongside `--report-json`. The parent directory must exist; the capture directory
must not. Capture stays disabled in ordinary app use. Example after rebuilding
`MCPZimEvalCLI`:

```sh
ios/build-eval/Build/Products/Debug/MCPZimEvalCLI --probe-discuss \
  --zim /path/to/wiki.zim --gguf /path/to/Bonsai-27B-Q1_0.gguf \
  --suite /path/to/calibration-development-suite.json \
  --capture-model-inputs tools/calibration/runs/capture-001 \
  --report-json tools/calibration/runs/capture-001-answers.json
```

Each invocation JSON contains the full rendered prompt (including reused prefix),
its UTF-8 SHA-256, exact originating token IDs, runtime/model identity, sampler,
stop strings, conversation ID and zero-based turn. Multiple calls per turn are
preserved; deterministic-only turns have no invocation. `capture-run.json` stays
`running` after interruption or failed writes; only `completed` permits further
validation. A completed capture does not imply correct answers or BF16 inputs.
Raw content remains local under ignored `runs/`; do not commit captures.

This is the first input-capture boundary, **not the calibration exporter**.
Source/model/tokenizer/template hashes and invocation-purpose annotations remain
required before freezing the calibration manifest. Never feed Bonsai token IDs
into Qwen without tokenizer validation. The current CLI MLX branch still assumes
Bonsai and needs a full-precision Qwen adapter before baseline measurements.

## Paired quality/size reports

`quality_frontier.py` consumes normalized, already-scored JSON runs:

```sh
PYTHONPATH=tools/calibration tools/calibration/.venv/bin/python \
  tools/calibration/quality_frontier.py --reference qwen-bf16-scored.json \
  --bonsai bonsai-scored.json --candidate qwen-q1-scored.json \
  --output comparison.json
```

The output must not exist. These are not raw `ProbeReport` files: the bridge from
Mac results and reviewed grading to this contract is still needed. No real score
or performance measurement has been generated by this module yet.

Contract version 1:

- Run: `schema_version: 1`, `status: completed`, `protocol_sha256`,
  `mode: model-only|app-conversation`, `model`, and nonempty `cases`.
- Model: `id`, immutable `revision`, `precision`, `runtime`, `machine`, measured
  integer `deployment_bytes` for all required files, `artifact_manifest_sha256`.
  Original Qwen reference requires `original_weights: true` and BF16/FP16.
  Candidates require matching `source_model` and `source_revision`.
- Case: unique `id`, `conversation_id`, `category`, `case_key_sha256`,
  `status: scored|failed`, rubric `score` in [0,1], and `critical_failures` list
  (explicitly empty when none). Failed executions score zero and remain paired.
  Model-only cases require `model_invoked: true`. Optional nonnegative metrics:
  `ttft_seconds`, `decode_tokens_per_second`, `peak_memory_bytes`.
- Protocol hash covers frozen split, suite, source snapshot, rubric, sampling and
  context policy. Case key hashes identical logical input/evidence for model-only
  comparison, or identical scenario/source context for app conversations whose
  later inputs can diverge. Pin each model's correct rendering separately.

Report scores weight conversations equally, then turns equally within each
conversation. Loss against Qwen and gain against Bonsai are percentage-point
changes. Per-category results and new critical failures remain visible even if
aggregate quality improves. Modest size increases are reported, not rejected.
No automatic promotion or statistical significance is inferred. Original-weight
attestation/hashes must be checked by the runner; this reporter cannot prove the
provenance of an arbitrary supplied JSON file. Optional performance fields list
coverage so missing timings cannot look like a complete benchmark.
