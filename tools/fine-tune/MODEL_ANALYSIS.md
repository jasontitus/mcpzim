# Multi-base-model fine-tune analysis (2026-04-26)

What we've learned about training mcpzim-style assistants on
different base models, after a couple days of running v3–v7c data
recipes and the multi-base CUDA pipeline. Companion to
[`DATA_RECIPE.md`](DATA_RECIPE.md) (which covers data curation) —
this doc covers **base model + pipeline** choice.

## Bottom-line table

All entries trained on the v7c data recipe (1577 v3 + 102
chains_3090b + 436 places_diverse = 2115 rows), 500 iters, rank 16,
last-16-layer LoRA. Eval is the 13-scenario A/B grid at Q4_K_M /
KV q8_0/q8_0. See `../llama-smoke/GRID_RESULTS_FT_*.md` for raw
data. Sizes/peaks are at inference time.

| model | pipeline | passes | peak RAM | disk | wall avg | ship class |
|---|---|---|---|---|---|---|
| **gemma3-4b-v7c** | Mac mlx-lm | **10/13** | 3.2 GB | 2.7 GB | ~2 s | ✅ iPhone ship |
| **qwen3-8b-v7c** | pcgaming PEFT | **9/13** | 5.6 GB | 4.7 GB | ~14 s | iPhone-Pro / Mac |
| qwen3.5-4b-v7c | pcgaming PEFT | 7/13 | 3.0 GB | 2.5 GB | ~16 s | alt iPhone |
| qwen3-4b-v7c | pcgaming PEFT | 6/13 | 3.2 GB | 2.5 GB | ~3 s | alt iPhone |
| stock gemma3-4b-it | (no FT) | 6/13 | 2.8 GB | 2.7 GB | ~1 s | baseline |
| qwen3-1.7b-v7c | pcgaming PEFT | 5/13 | 1.7 GB | 1.0 GB | ~1 s | tiny |
| qwen3.5-9b-v7c | pcgaming PEFT | 4/13 | 5.8 GB | 5.3 GB | ~22 s | (looping in thinking mode) |
| gemma3-1b-v7c | pcgaming PEFT | 1/13 | 1.0 GB | 769 MB | ~2 s | regression vs stock |
| qwen3.6-27b-v7c | Mac mlx-lm | 0/13 | 16 GB | 15 GB | ~12 s | ⚠️ broken (see below) |
| qwen3.6-27b-v7c (iter 100) | pcgaming Unsloth | not graded yet | ~16 GB | 15 GB | ~80 s | ✅ coherent CoT, hits thinking-loop on grader |

`narrate_hp_garage`, `grav_waves_chain`, `wwi_vs_wwii_chain`,
`french_revolution_chain` were "stuck" (everyone failing). Of those,
**qwen3-8b is the first model to pass `wwi_vs_wwii_chain`**, and
gemma3-4b-v7c is the only one passing `french_revolution_chain`
consistently. So bigger base models do unlock specific stuck
scenarios — they're not just data-bound.

## Pipeline-by-pipeline

### Mac mlx-lm (`finetune.sh`)

Works cleanly for **gemma-3-4b-it-bf16** at v7c → 10/13. This is
our shipping model. Loss 3.51 → 0.255, tight train/val gap, healthy
training.

Doesn't work for:
- **gemma-3-4b-it via pcgaming PEFT** — see "pcgaming Gemma3
  multimodal broken" below. Mac mlx-lm is the only working path
  for this model.
- **mlx-community/Qwen3.6-27B-bf16** — training "completes" with a
  clean-looking loss (3.4 → 0.27) but the resulting bf16 / F16 GGUF
  / Q4_K_M GGUF all generate incoherent garbage tokens
  (`'*dx!nfrdb-related/'`, random Chinese fragments). 0/13.
  Confirmed not a chat-template or quantization issue — the trained
  weights themselves are corrupted. Likely an mlx-lm bug in
  Qwen 3.6 support (the family is brand new).

Tradeoff: Mac mlx-lm needs 64+ GB unified memory for 27B+ training
and is the only path for multimodal Gemma 3, but it's slow (~3 hours
for a 27B run on M1 Ultra) and we've now seen it silently produce
broken weights for at least one model family.

### pcgaming PEFT (`finetune_cuda.sh`)

Works cleanly for **all the Qwen text-only models** (Qwen3-1.7B,
Qwen3-4B, Qwen3.5-4B, Qwen3-8B, Qwen3.5-9B), and for Gemma 3 1B
text-only.

Doesn't work for:
- **Multimodal Gemma 3 (gemma-3-4b-it)**. The
  `Gemma3ForConditionalGeneration` extraction path produces a
  trained adapter that, after fuse + GGUF + Q4_K_M, generates
  **only newline tokens**. Train loss looks fine. Output is dead.
  Pipeline-issue not yet diagnosed — we use Mac mlx-lm instead for
  this model.

Tradeoff: pcgaming is fast (3090's CUDA inference on smaller models
is much quicker than Mac M1 Ultra on big ones), and the Qwen path
is reliable. Gemma 3 4B has to go to the Mac.

### Bottlenecks

- **Mac mlx-lm + 27B**: ~3 hour runs. Fragile to memory pressure
  (we hit OOM at bsz=2 without grad-checkpoint, fits at bsz=1 +
  grad-ckpt). Disk-heavy: each 27B run writes ~120 GB across
  fused-hf + F16 GGUF + Q4_K_M GGUF. Mac is at 95% disk usage.
- **pcgaming + bnb / CUDA 13 jitlink mismatch**: bitsandbytes
  installs against a CUDA 13 nvjitlink shared object, but the
  system has CUDA 12.0. Fix: add the venv's `nvidia/cu13/lib` to
  `LD_LIBRARY_PATH` (now baked into `finetune_cuda.sh`).
- **pcgaming + Gemma 3 multimodal**: the
  `Gemma3ForConditionalGeneration` → `Gemma3ForCausalLM` extraction
  path is brittle in current transformers (4.55+). Don't run.

## Architecture-specific findings

### Qwen 3.5 / 3.6: thinking mode hurts our eval

The Qwen 3.5 / 3.6 family default to a chat template that opens
the assistant turn with `<think>\n` and trains to emit a reasoning
block before answering. This **interacts badly with our eval**:
the model often loops on tool calls inside its thinking, never
emits a final user-facing response, and the eval grader marks ✗
because there's no `final_content`.

Concrete: qwen3.5-9b-v7c gets only 4/13, but its outputs are
**coherent** — calling tools (`near_named_place`, `near_places`,
`search`, `article_overview`), even diagnosing the fixture's
"San Francisco vs San Carlos" geocoding issue. It just never
concludes. This is not corruption.

Two fixes worth trying:
1. **Patch the chat template** to set `enable_thinking=false` so it
   emits an empty `<think>\n\n</think>\n\n` opener instead of an
   open `<think>\n`. Lets the model jump straight to the answer.
2. **Train data with explicit `<think>...</think>` blocks** showing
   short reasoning, so the model learns to actually close its
   thinking and produce output.

(1) is ~5 min — modify GGUF metadata or `chat_template.jinja`
before convert, re-quantize. (2) is a data-gen change.

### Gemma 3 4B (multimodal): Mac mlx-lm only

`google/gemma-3-4b-it` is the multimodal variant (vision tower +
text core in one checkpoint). For LoRA on the text path:

- **Mac mlx-lm**: works. Outputs the language-model subtree as
  `gemma3_text` GGUF and fine-tunes via mlx_lm.lora.
- **pcgaming PEFT**: extraction path is broken in current
  transformers. After training looks fine, output is newlines-only
  (0/13 in the smoke grid, vs the Mac path's 10/13).

For now, ship the Mac path. If gemma 3 4B becomes a bottleneck
(e.g., Mac is busy with 27B QLoRA experiments), the pcgaming
extraction path needs fixing — likely something in
`finetune_cuda.py`'s `_is_gemma3_multimodal` branch loading the
wrong layers via `Gemma3ForCausalLM`.

### Gemma 3 1B: too small

gemma3-1b is purely capacity-limited. v7c training drops it to
1/13 (worse than stock 6/13). The data is too rich for the model
to fit. Don't ship a 1B — anything under 2B regresses against the
stock Gemma 4B at this task.

### Qwen 3.6 27B: mlx-lm IS the bug, not the model

**2026-04-26 update — definitively answered.** Trained Qwen 3.6 27B
on pcgaming via Unsloth (4-bit base + LoRA on `Qwen/Qwen3.6-27B`).
Pulled the iter-100 adapter to Mac, fused against the upstream bf16
base, converted to Q4_K_M, and ran a one-shot eval. Output:

> "The user is asking for good restaurants in San Francisco. I
> tried to call near_named_place with 'San Francisco', but it
> failed with 'no fixture for place=san francisco'. This suggests
> the specific..."

Coherent CoT, valid tool calls, accurate diagnosis of fixture
errors. Same training step that produced `'*dx!nfrdb-related/'`
random tokens via mlx-lm now produces fluent reasoning via
Unsloth/PEFT. Conclusion: **mlx-lm's Qwen 3.6 implementation is
broken**; the underlying weights and our v7c data are fine.

Two follow-ups for the iter-100 model to actually score on the grid:

1. **Disable thinking mode at inference** — the embedded
   `chat_template.jinja` still sets `<think>\n` at the assistant
   turn. The model loops on tool calls inside its reasoning,
   never emits a `final_content`. Fix: patch the GGUF's chat
   template OR pass `enable_thinking=false` via
   llama-cpp-python's chat completion args. Until this lands, the
   13-scenario grid will mark Qwen 3.6 ✗ on most rows for the
   wrong reason.
2. **Don't fuse against `mlx-community/Qwen3.6-27B-bf16`** — its
   layer naming or quantization differs from upstream
   `Qwen/Qwen3.6-27B` enough that PEFT's `merge_and_unload()`
   shows MISSING weights for 64 layers' `up_proj`, plus
   `embed_tokens`, `norm`, and `lm_head`. Resulting fused-hf is
   randomly initialized and useless. Always fuse against the
   exact upstream HF repo the LoRA was trained against.

Open question for next round: **Unsloth gradient offload kicks in
at the iter-100 eval boundary on a 24 GB 3090** (VRAM hits 24.3/
24.5 GB during eval, Unsloth enables "smartly offload gradients to
save VRAM," and per-iter time blows up 34 s → 633 s — projected
67 h to finish 500 iters). Iter-100 is the realistic ceiling on
this hardware unless we drop effective batch from 4 to 2 (BATCH_SIZE=1
GRAD_ACCUM=2) to avoid the offload trip-wire. Worth retrying with
that smaller config to get a full iter-500 run.

## Recommendations for next round

Prioritised, highest-leverage first:

1. **Validate QLoRA pipeline on Qwen3.5-4B** (~30 min on pcgaming)
   to confirm 4-bit base + LoRA training produces a working model
   at our v7c data. Compare to standard-LoRA Qwen3.5-4B's 7/13.
   *In flight as of 2026-04-26 02:55.*

2. **If QLoRA validates: train Qwen 3.6 27B via QLoRA** on pcgaming
   (~3 hours). This is the alt path for that model and will
   answer whether mlx-lm is the source of the corruption.

3. **Patch Qwen 3.5/3.6 chat templates to disable thinking-mode**
   at inference. ~5 min per model. Could push qwen3.5-9b from 4/13
   to maybe 7-8/13 (recovering scenarios where it currently loops).

4. **Generate v8 data with `<think>...</think>` examples** to teach
   the Qwen family to wrap reasoning. Slower than (3) but more
   durable.

5. **Targeted gen for the 3 stuck scenarios**
   (`narrate_hp_garage`, `grav_waves_chain`,
   `french_revolution_chain` for non-gemma) — see DATA_RECIPE.md.

6. **Investigate the pcgaming Gemma 3 multimodal extraction bug**
   — would unlock fast-iteration training on the same 4B Gemma we
   ship today. Lower priority while Mac mlx-lm works.

## Open issues / known footguns

- **Mac disk at 95%** (412 GB free of 7.3 TB). Each 27B run
  consumes ~170 GB across HF cache + ft-out artifacts. Prune older
  ft-out-* dirs (v3..v7b are reproducible; only v7c, v7c-aligned
  Qwen variants, and ship candidates need to stay).
- **pcgaming bnb / CUDA 13 jitlink fix is venv-pathed**. If the
  venv is rebuilt from scratch, the LD_LIBRARY_PATH setup in
  `finetune_cuda.sh` may need the path updated.
- **`finetune_cuda.sh`'s Step 3.5** unconditionally tries to fetch
  `tokenizer.model` from HF. For Qwen models that don't ship one,
  this prints a benign "no tokenizer.model on hub for ..." warning.
  Not a bug, just noise.
- **Val loss is unreliable** as a model-quality signal across
  pipelines. Mac mlx-lm Qwen3.6-27B reports val 0.27 with garbage
  output; pcgaming Qwen3.5-9B reports val 0.43 and is healthy. Run
  a one-shot `eval.py` after every new training run before
  spending 10 min on the full grid.
