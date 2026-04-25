# Fine-tune data generation

Scripts for generating supervised fine-tuning (SFT) data to LoRA-tune
Gemma 3 4B (or similar) on our specific tool-calling + response
patterns. Teacher is a local LM Studio server running any large
model (Qwen 3.6 27B, Gemma 3 12B, etc.); output is mlx_lm.lora-
compatible JSONL.

## Setup

1. **LM Studio** — load a 14B+ instruction-tuned model. Settings:
   - Local Server → on, port 1234
   - Developer → Parallel Requests: **8** (default is 1; without this
     async concurrency is a lie)
   - Keep Model Loaded: on
   - Prompt caching: automatic via llama.cpp prefix reuse — no UI
     setting needed, but the system prompt must be byte-identical
     across calls (this script ensures that)

2. **Python env**:
   ```sh
   cd tools/fine-tune
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python openai tqdm
   ```

## Usage

```sh
.venv/bin/python generate.py \
  --base-url http://localhost:1234/v1 \
  --model qwen3.6-27b \
  --n 5000 \
  --concurrency 8 \
  --out train.jsonl
```

Resumable: re-run the same command and it counts lines in `train.jsonl`
+ skips that many sampled queries.

## Output format

Each line is one training example as:
```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"bars in north beach"},
  {"role":"assistant","content":"```tool_call\n{...}\n```"},
  {"role":"user","content":"[TOOL_RESPONSE]\n{...}"},
  {"role":"assistant","content":"Found 25 bars near North Beach..."}
]}
```

Drop this straight into `mlx_lm.lora` as `--train-data train.jsonl`.

## Coverage

Seeds span these categories (weighted for training balance):

| Category | Weight | Example |
|---|---|---|
| exploration | 1.5 | "bars near me", "cool spots around Palo Alto" |
| learn_topic | 1.5 | "tell me about Rayleigh scattering" |
| discuss_aspect | 1.5 | follow-up patterns for deeper topic exploration |
| directions | 1.0 | "directions to the Ferry Building" |
| current_place | 1.0 | "where am I", "what's this neighborhood" |
| compare | 1.0 | "how is X different from Y" |
| narrate | 0.5 | "read me the article on X" |

Slot pools (PLACES, POIS, TOPICS, PEOPLE, ASPECTS, LANDMARKS,
COMPARE_PAIRS) each have 50-100+ entries so 5-10k generations see
healthy diversity without over-sampling any single value.

## Speed tips

- LM Studio Developer → Parallel Requests = 8 (biggest single win)
- Q4_K_M teacher model runs 2-4x faster than Q6/Q8 with minimal
  generation-quality loss for this pattern task
- `--max-tokens 512` (default) — most trajectories land ~400 tokens
- `--temperature 0.8` — high enough for diversity, low enough that
  JSON stays valid
- Pre-warm step runs automatically before the concurrent batch, so
  the first 8 async calls don't all re-compute the system prompt
- Expected throughput on M2 Max 64 GB with Qwen 3.6 27B Q4_K_M +
  parallel=8: ~4-8 trajectories/second post-warmup → 10k examples
  in ~30 min

## Training the LoRA (finetune.sh)

`finetune.sh` runs the full pipeline: LoRA train → fuse → HF→GGUF
convert → Q4_K_M quantize. Idempotent per step — each checks for its
output file and skips if present.

```sh
bash finetune.sh train_v3.jsonl

# override defaults via env
BATCH_SIZE=8 ITERS=500 LORA_LAYERS=16 MAX_SEQ_LEN=2048 \
    bash finetune.sh train_v3.jsonl

# watchdog-safe combo (system busy / display in heavy use)
BATCH_SIZE=2 MAX_SEQ_LEN=1024 OUT_DIR=ft-out-v7b \
    bash finetune.sh train_v7b.jsonl
```

**Note on `LORA_RANK`**: the env var is *not* wired through to
`mlx_lm lora` — that flag has no `--rank` CLI option, only a `-c
CONFIG` YAML file. Every run so far has trained at mlx-lm's default
**rank 8**, regardless of `LORA_RANK=...`. If you genuinely want a
different rank, write a YAML config with `lora_parameters.rank: N`
and pass `-c <config.yaml>` directly to `mlx_lm`.

Outputs land under `ft-out/` (gitignored):

| file | purpose |
|---|---|
| `ft-out/adapters/adapters.safetensors` | LoRA weights (+ per-100-iter checkpoints) |
| `ft-out/fused-hf/` | dequantized HF checkpoint with LoRA merged in |
| `ft-out/gemma3-4b-it-ft.f16.gguf` | reference F16 GGUF (~8.5 GB) |
| `ft-out/gemma3-4b-it-ft.Q4_K_M.gguf` | shipping quant (~2.7 GB) |

Eval path: add/adjust the `gemma3-4b-ft` `ModelSpec` in
`../llama-smoke/grid.py`, then run the A/B grid — see
`llama-smoke/GRID_RESULTS_FT_AB*.md` for prior runs. The authoritative
comparison is on the **Q4_K_M GGUF**, not the bf16 fused model;
quantization can wash out LoRA effects so measure what ships.

## Gotchas

- **MPI / mlx-lm**: MLX dlopens `libmpi.dylib` at startup and aborts
  with `Abort trap: 6` if it resolves to MPICH (it requires Open MPI).
  If `mpich` is installed via Homebrew (`/opt/homebrew/lib/libmpi.dylib`
  → MPICH), run `brew unlink mpich` before training — MLX then proceeds
  without MPI, which is fine for single-node. Reversible with
  `brew link mpich`.
- **LM Studio / other Metal workloads**: LoRA training at `BATCH_SIZE=8`
  peaks around ~68 GB of unified memory. If LM Studio is serving a
  model or another heavy Metal process is running, the training run
  can die at ~iter 20 with
  `[METAL] Command buffer execution failed: Discarded (victim of GPU
  error/recovery) ... InnocentVictim` — that's Metal speak for
  "another GPU task faulted and your process got killed as collateral."
  Quit LM Studio (and any other Metal-heavy tools) before training.
- **Val interval**: mlx-lm only runs validation every 200 iters by
  default — you won't have an overfitting signal before that. Train
  loss alone isn't enough to judge generalization.
- **macOS GPU watchdog (`ImpactingInteractivity`)**: when the display
  is actively in use (Kiwix browsing, Chrome rendering, etc.), Metal
  kills any single command buffer that runs longer than ~5 s to keep
  the system responsive. Failure looks like
  `[METAL] Command buffer execution failed: Impacting Interactivity
  (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)`
  followed by `Abort trap: 6`. Distinct from `InnocentVictim` — this
  is the kernel killing *you* for being too slow, not collateral
  damage. Mitigation: shrink the per-step compute with
  `BATCH_SIZE=2 MAX_SEQ_LEN=1024`. Smaller batch + half-length
  sequences cleared every watchdog kill we saw. A foreground display
  is required — running headless avoids it entirely.
- **Val loss does not predict eval-grid behaviour**. Across runs the
  *lowest* val loss has matched the *worst* scenario score. v6 had
  val 0.254 (best of any run) and 5/13 passes (worst); v4 had val
  0.269 and 8/13 passes (best). Treat val as a *training-health*
  signal (no overfitting / no divergence) and the A/B grid as the
  *behaviour* signal — they answer different questions. Always
  ship the model that wins the grid, not the one with the lowest
  val.

## Run log

All runs on Mac Studio M1 Ultra 128 GB, base model
`mlx-community/gemma-3-4b-it-bf16`, `LORA_LAYERS=16`, mlx-lm default
rank 8 (regardless of `LORA_RANK` setting — see note above), val
every 200 iters. Val batches was mlx-lm default (25) for the v3-v6
runs; v7a/v7b ran after the script was changed to default 5 so
those numbers are noisier. *Pass* column is the count of
passing scenarios on the 13-scenario A/B grid at Q4_K_M / KV
q8_0/q8_0 — see `../llama-smoke/GRID_RESULTS_FT_*.md`.

| run | data | rows | iters | batch | seq | val end | pass | wall | artifact |
|---|---|---|---|---|---|---|---|---|---|
| stock | — | — | — | — | — | — | 6/13 | — | upstream `gemma3-4b-it` Q4_K_M |
| v3 | `train_v3.jsonl` | 1577 | 500 | 8 | 2048 | 0.269 | 7/13 | ~100 min | `ft-out-v3/` |
| **v4** | `train_v4.jsonl` (+ chains_3090b) | 1685 | 500 | 8 | 2048 | 0.269 | **8/13** | ~60 min | `ft-out-v4/` ← **ship candidate** |
| v5-it200 | `train_v5.jsonl` (+ places_diverse) | 2115 | 200 | 8 | 2048 | 0.286 | 7/13 | crashed @ 270 | `ft-out-v5-iter200/` |
| v6-it200 | `train_v6.jsonl` (+ chains_3090c, places_diverse2) | 3013 | 200 | 4 | 2048 | 0.269 | 6/13 | crashed @ 430 | `ft-out-v6-iter200/` |
| v6-it400 | same | 3013 | 400 | 4 | 2048 | 0.254 | 5/13 | crashed @ 430 | `ft-out-v6-iter400/` |
| v7a | `train_v7a.jsonl` (v4 + chains_3090c) | 2079 | 500 | 2 | 1024 | 0.229 | 8/13 | ~25 min | `ft-out-v7a/` |
| v7b | `train_v7b.jsonl` (v4 + places_diverse2) | 2177 | 500 | 2 | 1024 | 0.230 | 7/13 | ~25 min | `ft-out-v7b/` |
| **v7c** | `train_v7c.jsonl` (v4 + places_diverse) | 2115 | 500 | 2 | 1024 | 0.255 | **10/13** | ~25 min | `ft-out-v7c/` ← **ship candidate** |

### Key observations

- **v7c is the current ship candidate** — 10/13, beating v4 by 2 and
  stock by 4. Adding `train_places_diverse.jsonl` (the original
  436-row diversity batch) to the v4 corpus picked up
  `sky_is_blue_chain` and `french_revolution_chain` without losing
  any of v4's wins.
- **The v7a/v7b/v7c bisection isolated each new batch's effect on
  top of v4**. See [`DATA_RECIPE.md`](DATA_RECIPE.md) for the full
  per-batch impact and what to generate next.
- **v6 collapsed despite the best val loss**. Adding all four data
  augmentations on top of v3 (chains_3090b, chains_3090c,
  places_diverse, places_diverse2 → 3013 rows) drove val to 0.254
  but cratered the grid to 5/13. The pieces that work alone (v7c at
  +2) cancel out and become net negative when stacked together —
  classic overlapping-data interaction effect.
- **batch-2 / seq-1024 is the watchdog-safe profile**. v7a/b/c ran
  cleanly while batch-4/seq-2048 got `ImpactingInteractivity`-killed
  early. Tradeoff: ~4× fewer samples seen at iter 500 → undertrained
  vs v4 (which ran batch 8). v7c hitting 10/13 anyway argues
  the data is the dominant factor, not the training depth.
