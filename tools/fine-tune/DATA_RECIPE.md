# Data recipe — what works, what doesn't, what to add

Empirical findings from the v3–v7c LoRA runs on Gemma 3 4B IT (bf16
base, mlx-lm default rank 8, fused → Q4_K_M GGUF, evaluated on the
13-scenario A/B grid in `../llama-smoke/grid.py` at KV q8_0/q8_0).
Numbers are scenario passes out of 13.

| | base | what it adds |
|---|---|---|
| stock | gemma-3-4b-it Q4_K_M | (no fine-tune) — 6/13 |
| **v7c (ship)** | v4 corpus + `train_places_diverse.jsonl` | **10/13** |
| v4 | v3 corpus + `train_chains_3090b.jsonl` | 8/13 |
| v3 | `train_v3.jsonl` | 7/13 |

## Per-batch impact (vs v4 baseline at 8/13)

| batch | rows | net Δ | what changes | recommendation |
|---|---|---|---|---|
| `train_places_diverse.jsonl` | 436 | **+2** | gains `sky_is_blue_chain` + `french_revolution_chain`; no losses | **GENERATE MORE LIKE THIS** |
| `train_chains_3090c.jsonl` | 400 | 0 | gains `french_revolution_chain`, loses `nearby_stories_pa` (washes) | neutral — keep but don't prioritise |
| `train_places_diverse2.jsonl` | 498 | −1 | loses `crispr_chain` for nothing | drop — actively hurting |
| `train_chains_3090b.jsonl` | 102 | (already in v4) | foundational chain data | keep |

**Don't blindly stack everything.** Combining all four on top of v3
(the v6 corpus, 3013 rows) collapsed the grid to **5/13** despite the
*lowest val loss of any run* (0.254). Pieces that work alone don't
necessarily compose — overlapping data adds appear to interact and
cancel each other.

## What "works" in `train_places_diverse.jsonl`

The 436 grounded examples it contains do something specific that
`places_diverse2` doesn't, and that the chain-style data can't reach:

1. **Diverse `near_places` `kinds`**: restaurants, cafes, parks,
   museums, hotels, gas_stations, hospitals, etc. — not just bars.
   This breaks the "I found N bars in San Carlos!" memorization
   we saw in v3.
2. **Varied result counts** — the assistant follow-up turn cites
   actual counts/names from the tool response rather than
   parroting a templated "I found 25 X in Y" phrase.
3. **Non-trivial topical generalization** — it picks up
   `sky_is_blue_chain` and `french_revolution_chain` even though
   those scenarios aren't directly modeled by `near_places` data.
   That suggests the underlying gain is "ground responses in tool
   output" as a transferable behaviour, not just memorising more
   templates.

## What scenarios still fail for everyone (stock + every FT)

These three never pass in any run:

- `narrate_hp_garage` — narrate-style; weight 0.5 in the existing
  generator (low). Worth raising and producing more of these.
- `grav_waves_chain` — gravitational-waves Wikipedia chain.
- `wwi_vs_wwii_chain` — comparison chain on WWI vs WWII.

Both `_chain` failures look like the model isn't grounding its
follow-up turns on the section content — it produces plausible-
sounding but uncited summaries that the eval rejects. The chain
data we already have (`train_chains_3090b/c.jsonl`) is structurally
the same pattern but for *different* topics. Two avenues:

- **Expand topical coverage of chain data** to include physics
  (gravitational waves, relativity), 20th-century military history
  (WWI/WWII comparisons), and similar broad-knowledge-comparison
  prompts. Current chain corpus seems biased toward
  biographical/single-topic exploration.
- **Tighten the assistant turn's grounding**: each follow-up
  should explicitly cite section content, not paraphrase from
  general knowledge.

## Recommendations for the next data-gen round (Windows / 3090)

Priority 1 — **more places_diverse-style data**:
- Run the same generator (`generate_places_diverse.py`) for another
  500–1000 examples *with the same v1 prompt* (the one that
  produced the 436-row batch in `train_places_diverse.jsonl`).
- Avoid changing the prompt template — the v2 prompt that produced
  `train_places_diverse2.jsonl` is what regressed `crispr_chain`.

Priority 2 — **narrate-style data**:
- The existing `generate.py` weights narrate at 0.5. Raise to 1.5
  for one batch of ~300 rows.
- Pattern: user asks "read me the article on X" / "narrate Y to
  me", model does `article_overview` + `get_article_section`
  calls, then produces a flowing narrative response that pulls
  specific facts from the section bodies.

Priority 3 — **comparison / chain data targeting WWI/WWII, physics
chains, broad-knowledge comparisons**:
- Extend `generate_chains.py` (or its successor) to include
  comparison topics (`Compare X and Y` patterns) and physics
  topics. Start with ~200 rows.

Avoid:
- More `places_diverse2`-style data with the v2 prompt — confirmed
  net negative on the grid.
- Adding all batches at once. Each new batch should be evaluated
  in isolation against v4 (the known-good baseline) before being
  combined. The v6 collapse showed that compatibility between
  batches is not transitive — A and B might both be neutral but
  A+B regress.

## Training profile that works on Mac (M1 Ultra 128 GB)

`BATCH_SIZE=2 MAX_SEQ_LEN=1024 ITERS=500` with `--grad-checkpoint`
clears the macOS GPU watchdog (see README "Gotchas"). Peak ~14 GB,
~25 min wall-clock end-to-end including fuse + GGUF + quant.

If the 3090 box can run mlx-lm equivalents (or alternate trainer)
with full batch and longer sequences, prefer that — undertrained
runs at batch 2 have noisier scenario outcomes.

## Artifact layout

`ft-out-*/` directories are gitignored (each is ~12 GB: bf16
fused-hf + f16 GGUF + Q4_K_M GGUF + adapter checkpoints). The
shipping artifact is just the **2.7 GB Q4_K_M GGUF**:

- v7c (ship): `ft-out-v7c/gemma-3-4b-it-ft.Q4_K_M.gguf`
- v4 (prior best): `ft-out-v4/gemma3-4b-it-ft.Q4_K_M.gguf`

Sync to Windows / phone via HF Hub, rsync, or scp — not git.
