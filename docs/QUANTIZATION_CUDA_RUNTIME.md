# CUDA runtime preparation

September 19, 2026. This is a dependency and upstream-component integration
gate for the [GPU readiness runbook](QUANTIZATION_GPU_READINESS.md). It does
not establish full-model optimization feasibility or authorize GPU rental.

## Completed local evidence

The Linux/amd64 image built successfully on this Mac. Nine failure-focused
tests passed inside the image. Both CPU component reports passed, and a CUDA
request without a GPU failed as required. Source checks and `pip check` passed.
No cloud instance, paid build service, image push or GPU execution occurred.

- Local immutable image index:
  `sha256:1ea284684a05dab167b89d574006dc65ec9209ddbafed4d08af9db4bbac0911f`.
- Linux/amd64 platform manifest:
  `sha256:dff52692bbcfcb8124c544feaeda3d33acde9a7f4e7aec124aab3c03237eb40e`.
- [CPU component reports](../tools/calibration/cuda_runtime/evidence/cpu-checks.json).
- [CUDA-unavailable rejection](../tools/calibration/cuda_runtime/evidence/cuda-unavailable-rejection.json).
- [Image identity and configuration](../tools/calibration/cuda_runtime/evidence/image-inspect.json).

The image is locally available as `zimfo-quant-runtime:20260919`. After staging,
record the registry identity and verify it before launch. Build preparation
left approximately 96 GiB free on the Mac; model weights were not copied into
the image or its build context.

## Runtime choice and scope

The local Linux x86-64 build uses official PyTorch 2.11.0, CUDA 13.0, Python
3.12.3, pinned at base digest
`sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0`.
CUDA 13 is selected for the preferred Blackwell RTX PRO 6000 96 GB option;
the RCO reference CUDA 12.6 wheel is not used. The checks require the PyTorch
binary to contain native `sm_120` support. Driver compatibility and actual CUDA
execution remain hardware gates.

Public upstream sources, including their Apache 2.0 LICENSE files, are staged
from exact Git commits, never from a potentially modified working tree:

- [GSQ](https://github.com/IST-DASLab/GSQ/tree/03fc16484c369e3127225615d5e03e8d3a6043e3):
  `03fc16484c369e3127225615d5e03e8d3a6043e3`.
- [RCO](https://github.com/IST-DASLab/RCO/tree/9a1e09c07d468109cbe60a1b87d5036034a79d10):
  `9a1e09c07d468109cbe60a1b87d5036034a79d10`.

The optimization import set needs Transformers, Accelerate, safetensors,
compressed-tensors, datasets, Lion, wandb and basic utilities. It does not need
vLLM, Ray, inference serving, lm-eval, or their CUDA extension stack for the
checked path. W&B is disabled and Hugging Face is offline. The official base
contains Torch's compiled kernels; this build does not yet supply a custom GGUF
packing tool or specialized fused recurrent kernels. Slow unfused operation is
not evidence of acceptable full-model throughput.

`requirements.full.lock` records hash-pinned Linux resolution.
`base-cuda-constraints.txt` records CUDA/Torch packages supplied by the immutable
base. `finalize_lock.py` removes only those version-validated packages from the
install list; installation uses hashes, wheels only and `--no-deps` so it cannot
silently replace the CUDA stack. `pip check` must pass. A system-site-packages
virtual environment preserves the base and holds additional dependencies.

## Reproduce locally, before rental

Run from the repository root. Source staging refuses to overwrite an existing
directory; preserve or explicitly remove only the generated `.context/sources`
directory when intentionally rebuilding it.

```sh
python3 tools/calibration/cuda_runtime/prepare_sources.py \
  --gsq /tmp/zimfo-gsq-research --rco /tmp/zimfo-rco-research
docker buildx build --platform linux/amd64 --load \
  --tag zimfo-quant-runtime:20260919 tools/calibration/cuda_runtime
```

The build context is allowlisted to public source, lockfiles and check scripts.
It excludes model weights, captured app content, credentials and the rest of
the repository. No image has been pushed and no paid build service is used.
The build executes failure-focused tests and each upstream check in a separate
process. This avoids GSQ's `src.*` imports colliding with RCO's bare `common`,
`metrics`, and `search` modules. Production stage workers must preserve this
process isolation or introduce an explicitly tested namespace adapter.

CPU reports are in `/opt/build-evidence/gsq-cpu.json` and `rco-cpu.json` within a
successful image. They explicitly record `gpu_validated: false` and
`whole_model_validated: false`. CPU success cannot satisfy the paid-launch
readiness manifest. Record the final image ID now and its registry digest after
authorized staging; the mutable local tag is not a deployable identity.

On an authorized GPU later, run the already built image with read-only input
mounts, writable output only, and no network for component checks:

```sh
python upstream_check.py --stage gsq --device cuda --report /outputs/gsq-cuda.json
python upstream_check.py --stage rco --device cuda --report /outputs/rco-cuda.json
```

These CUDA checks use tiny tensors. They supplement, and never replace, the
required actual-Qwen-block and full-model RCO smoke. No package installation,
source checkout or dependency resolution belongs in the GPU startup path.

## Integration contract with Mac capture

The authoritative replay inputs are full, exact token IDs, original checkpoint
revision, template/tokenizer identity, sequence boundaries, attention masks and
position IDs where not derivable, and per-invocation provenance. Every invocation
starts with fresh hybrid recurrent state. Initial embeddings and final normalized
hidden states can avoid some recomputation after CUDA/MLX parity is established.
Compare final hidden states and logits on replay; recompute the reference on
CUDA if parity is inadequate.

Bounded sampled input activations are useful for diagnostics and initialization.
They do not substitute for nonlinear GSQ block reconstruction or the changing
quantized-prefix inputs. GSQ must regenerate full block inputs/outputs from the
original tokens/model as needed, one block at a time. Preserve batch/sequence
weights and sampled token indices; never silently treat sampled rows as all tokens.

For RCO, final reference hidden states plus the original output head allow
chunked full-vocabulary teacher targets. The upstream `compute_kl_loss` chunks
the *loss* but still materializes all model logits and transfers all reference
log probabilities to the GPU. A chunked-head integration is still required for
bounded memory. A top-k objective would change the experiment and cannot silently
replace full-vocabulary KL. Output-head quantization requires comparing against
the unquantized reference head, not the candidate's own head.

## Manual adversarial review

- **CPU pass mistaken for GPU readiness:** reports encode device and scope;
  GSQ's one-bit forward unconditionally uses CUDA RNG and is explicitly not run
  on CPU. CPU checks cover initialization/hard export and real import paths.
- **False algorithm integration:** the RCO check executes upstream interpolation,
  masked full-vocabulary KL, candidate gradients, manifold projection, Adam,
  retraction and moment transport. It compares loss/gradients to an independent
  small dense calculation. This proves component compatibility, not full solver
  integration, speed, memory, final packing, or quality.
- **Impossible budget silently accepted:** upstream retraction does not reject
  targets outside available costs. The wrapper rejects nonfinite/unattainable
  targets and invalid weights, with tests, before invoking it. Full serialized
  file costs and protected tensors still need the production budget adapter.
- **Dependency drift or CUDA replacement:** hash lock, immutable base, protected
  CUDA constraints, `pip check`, exact version/compiled-architecture checks.
- **Hidden/private build context:** public Git archives plus Docker allowlist;
  no model or app-data COPY. Source manifest rejects changed or unexpected files.
- **Source namespace collision:** stages run in distinct processes.
- **Missing deployment capabilities:** current image is preparation only. Actual
  Qwen GSQ/RCO runner, packing/export, checkpoint/resume integration and CUDA
  parity/memory checks remain launch gates. Do not relabel component success as
  complete readiness.

## Production solver revision (September 20)

The later `solver-20260920` image includes the actual hybrid Qwen consumer and
`python -m solver.job`, superseding the component-only image above. The job
runs initialization, GSQ smoke, RCO smoke, learned observed-token embeddings,
all hybrid blocks, learned full-vocabulary output head, then RCO allocation.
GSQ and RCO run in separate processes. It consumes the entire captured token
sequence with fresh recurrent state; teacher activations are recomputed from
original BF16 weights on CUDA, not reused from sampled or MLX activations.

The smoke uses the longest complete captured invocation, actual linear and full
attention blocks, full-model RCO backward, and a durable checkpoint restore
whose next stochastic optimizer update must match uninterrupted execution.
GSQ smoke block and Adam states are reused as content-bound warm starts. RCO
smoke states are diagnostic because subsequent GSQ changes its candidates.
Only hardware execution establishes peak memory and throughput. The learned
head keeps full optimizer state but offloads the frozen body before allocating
it; this is not a claim of constant-memory head optimizer state.

Q1 scales are projected onto the BF16-compute/FP16-storage grid after updates;
nonfinite or unrepresentable values fail closed. The initial head and unseen
embedding rows use RTN. Observed embedding rows and the complete output head
are learned before a final-quality recipe can proceed. The exact packing cost
manifest covers all498 selectable matrices, protected tensors, metadata and
alignment. The wrapper rejects budgets below the complete Q1 file size.

Checkpoints contain optimizer, scheduler, RNG, exact next cursor, frozen
configuration, immutable candidate/cache archives, boundary reports, costs and
warm starts. Core state is bundled, with a default120-second cadence and phase
boundaries. Cold recovery uses `solver.job --prepare-resume` to verify a pinned
GCS commit and reconstruct portable paths. It reads the original checkpoint
prefix and writes a new attempt prefix so relocated configuration never
collides with an immutable commit. The initial recovery intentionally verifies
and reads checkpoint data twice; it resumes one stage, not all later stages.

Local Linux tests now cover real tiny hybrid block optimization, the complete
embedding→block→head→RCO pipeline, CPU smoke restoration, cold RCO recovery
without original report files, completed-block and next-block-start recovery,
native SDPA versus eager parity, and independent BF16 gradient oracles. The
head oracle invokes the actual pinned upstream GSQ autograd function with CPU
RNG substitution; allocation gradients match dense BF16 cast boundaries.
These tests do not establish full27B CUDA feasibility or answer quality.

The job emits `checkpointed` only for committed progress and
`ready_for_packaging` only after the optimization phases; neither means the
model passed Zimfo evaluation. Explicit packing, runtime reload and held-out
Qwen/Bonsai comparison remain required. Runtime code, Google Cloud Storage,
and the pinned Prism converter are installed before the GPU starts.

Final prepared solver image (41 Linux CPU tests plus both CLIs passed):
`us-central1-docker.pkg.dev/tiltastech-zimfo/zimfo-quantization/runtime@sha256:0c95864edb0b19dd9c722c5e2cca1ca210fab693b2f142749424711dc2956ab9`.
The registry digest was independently inspected after push. Evidence and exact
application hashes are in `cuda_runtime/evidence/solver-build.json` and
`solver-source-manifest.json`. Two loader issues were caught before GPU use:
Accelerate's default dtype would promote BF16 weights to meta FP32, and rotary
buffers would remain on CPU. Loading now specifies BF16, then moves devices
without dtype conversion and asserts all parameter/buffer devices. Do not use
the superseded `solver-20260920` registry tag; immutable tags were preserved.
