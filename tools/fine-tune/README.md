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
BATCH_SIZE=8 ITERS=500 LORA_RANK=16 LORA_LAYERS=16 \
    bash finetune.sh train_v3.jsonl
```

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

## Run log

| date | data | rows | iters | batch | rank / layers | val@end | train@end | peak mem | wall | artifact |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-04-24 | `train_v3.jsonl` | 1499 | 500 | 8 | 16 / 16 | 0.269 | 0.222 | 68.6 GB | ~100 min | `gemma3-4b-it-ft.Q4_K_M.gguf` |

2026-04-24 run — val trajectory 3.510 → 0.282 → 0.272 → 0.269 (iter
1 / 200 / 400 / 500). Tight train/val gap throughout (≤0.06), no
overfitting; curve decelerated cleanly by iter 50. 500 iters looked
well-calibrated — further training likely to invert val. Machine:
Mac Studio M1 Ultra 128 GB.
