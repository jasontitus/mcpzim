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
