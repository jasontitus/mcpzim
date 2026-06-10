# How we picked the on-device tool-calling model — evaluation history

*(Written 2026-06-10. Covers ~April–June 2026. Companion docs:
`ON_DEVICE_MODEL_REPORT_2026-04-23.md` (stock-model sweep),
`tools/llama-smoke/LFM25_MEMORY_PERF_FRONTIER.md` (quant frontier),
`GEMMA4_QAT_MTP.md` (Gemma 4 QAT/MTP research), `EXTENDED_CONTEXT_EVAL.md`,
and the `tools/llama-smoke/GRID_RESULTS_*.md` artifacts.)*

## The problem

Zimfo needs a local LLM that, on an **iPhone (17 Pro Max, hard 6,144 MB
process jetsam cap** even with the increased-memory entitlement), can:

- **drive a tool loop** over offline ZIM archives (Wikipedia + OSM/streetzim +
  mdwiki): pick the right tool from ~11–21 declarations, emit well-formed
  call syntax, ground its answers in tool results;
- hold a **multi-turn voice conversation** (TTS-latency-sensitive — prefill
  and decode speed are UX, not benchmarks);
- leave room for everything else in the process: Kokoro TTS (~300 MB),
  live map WebViews (300–500 MB each), ZIM readers, KV cache.

Practical budget for weights + cache: **~3.5–4.5 GB**. The goal crystallized
as: *the fastest, most accurate, most memory-efficient model — all three
axes, not any one.*

## Evaluation methodology (and how it evolved)

1. **9-scenario Python harness** (`tools/llm-smoke/`, mlx-lm/mlx-vlm on an
   M2 Max): tool-selection accuracy + prefill/decode tok/s + peak RSS at
   7k/20k/40k-token preambles. Fast cross-family screening. Caveat learned:
   it's a **floor**, not ground truth — template quirks under-score some
   families (Gemma 4 especially, §1b of the model report).
2. **13-scenario llama.cpp grid** (`tools/llama-smoke/grid.py`, GGUF,
   subprocess-isolated peak-RSS): the production harness once we settled on
   llama.cpp as the runtime. Adds multi-turn knowledge *chains* whose final
   turn must be answered **from cached context with no tool call** — the
   walking-companion pattern. This became the scorecard all decisions cite.
3. **Swift end-to-end harness** (`MCPZimEvalCLI` / `EvalHarness.swift`,
   18 scenarios): the real `ChatSession` + IntentRouter fast-paths +
   MCPToolAdapter against stub ZIMs — catches integration failures the
   prompt-level harnesses can't (and validates conversation-state features).
4. **On-device verification**: peak RSS via jetsam/crash tooling, tok/s on
   the phone, gist-based debug reports from real walks.

**Hard-won methodology lesson (2026-06-10):** at sampling temp 0.2–0.3, the
13-grid carries **±1 noise** — four borderline chain scenarios
(french_revolution, grav_waves, wwi, narrate) float between runs. Single-run
deltas of one scenario are not signal; use **majority-of-3** before believing
any ±1 change. (Discovered when a targeted FT "fixed" a scenario that the
same-day baseline no longer failed.)

## Phase 1 — stock-model sweep (April 2026, MLX runtime)

Screened the plausible 2–4B field (and references) on the 9-scenario harness.
Consolidated scorecard in `ON_DEVICE_MODEL_REPORT_2026-04-23.md`; highlights:

| Model | Acc /9 | Peak @7k/20k | Verdict |
|---|---|---|---|
| Qwen 3 4B Instruct (then-default) | 6/9 | 3.75/5.55 GB | OK accuracy; KV balloons at long ctx; 20% malformed-JSON rate on voice phrasings (fixed w/ repairJSON, still flaky) |
| Qwen 3.5 4B | 6/9-ish | — | **Disqualified: hybrid-cache bug** — MambaCache breaks partial-prefix reuse → full ~13 s prefill *every turn* (upstream mlx-swift-lm#157 unfixed) |
| Gemma 3 4B IT | **7/9** | 3.74/4.37 GB | Best stock pick: native RotatingKVCache keeps 20k ctx at ~4.4 GB |
| Gemma 4 E2B | 1/9 ⚠️ | 4.49 GB | **Hard cliff at ≥11 tool declarations** — emits `<eos>` as first token |
| Gemma 4 E4B | 5/9 ⚠️ | 6.18 GB | Degrades (not cliffs) but over budget at 20k |
| gemma-3n E2B/E4B | 5–6/9 | 3.7–4.9 GB | mid |
| Phi-4-mini (4/6-bit) | 4–6/9 | 3.7–8.6 GB | accuracy/memory never both OK |
| Phi-3.5, Nemotron variants | 0–3/9 | — | refuse our tool schemas; dropped |
| Gemma 3 12B QAT *(mac ref)* | **9/9** | 9.2 GB+ | proves the scenarios are solvable; 2× over phone cap |

Decisions out of phase 1: Gemma-class attention economics (sliding-window KV)
matter as much as raw accuracy; small stock models top out ~6–7/9 on our tool
surface; the 12B reference shows the ceiling is the *model*, not the eval.

## Phase 2 — fine-tune on our own tool surface (May 2026)

Two structural shifts:

- **Runtime → llama.cpp (GGUF)** for the shipping model: full
  `llama_context_params` control (q8_0 KV, flash-attn, iSWA `swa_full=false`),
  no MLX loader coupling, and the 13-grid became the scorecard.
- **Train the format in, don't prompt it in.** `tools/fine-tune/`: a teacher
  fleet (LM Studio boxes + a DS4 server on the LAN) generates synthetic
  tool-call trajectories *in the exact prompt format the app renders*
  (system+tools folded into the first user turn for Gemma; format mismatch
  was found to silently regress training). Single-turn + 2-turn chains +
  targeted hard-case rows; mlx-lm LoRA → fuse → GGUF → quantize.

**Gemma 3 4B FT (V7C)** was the first shipped fine-tune: **10/13 @ 3.18 GB**
— up from 7/13-equivalent stock, and proof that targeted data moves the
needle more than model swaps.

## Phase 3 — LFM2.5-8B-A1B: the MoE bet (late May 2026)

The reasoning: on a phone, **active parameters set speed; total parameters
set memory; quantization sets the exchange rate.** A hybrid-MoE with 8.3B
total / **1.5B active** (LiquidAI LFM2.5-8B-A1B) decodes like a 1.5B
(~46 tok/s on the iPhone, ~1.8× the Gemma FT) while carrying 8B-class
capacity. Two more properties made it ideal for *this* app:

- **KV is uniquely cheap**: only 6 attention layers (rest is recurrent
  shortconv) ≈ 3 KB/token at q8_0 — a 32k window costs ~100 MB (Gemma-class
  models pay several× more per token). Long replies and long conversations
  are nearly free.
- It quantizes well (see Phase 5).

Stock, it was *terrible* on our grid: **2/13** (`GRID_RESULTS_LFM25.md`) —
wrong tools, unparseable calls. After the same FT recipe (the v7 dataset:
~4.7k rows incl. 317 targeted hard-case rows; 800 iters; LFM2-specific
pipeline that unstacks MoE experts + restores the tokenizer for GGUF
conversion):

> **LFM2.5-8B-A1B FT v7-full Q3_K_M = 12/13 @ 4.16 GB**, ~46 tok/s on
> iPhone — beating the Gemma 3 4B FT (10/13) by two scenarios at +1 GB,
> and *undercutting LFM2.5's own Q4_K_M baseline* (5.16 GB) on memory.

That 2/13 → 12/13 swing is the central finding of the whole effort: **the
fine-tune, not the base model, is the moat.** A stock model that aces generic
benchmarks still fails *our* 11-tool, voice-phrased, multi-turn workload.

## Phase 4 — challengers since (June 2026): Gemma 4 QAT, MTP

Google's QAT (quantization-aware-trained) releases and MTP (multi-token
prediction) drafters prompted a full re-evaluation (deep-research +
measurements in `GEMMA4_QAT_MTP.md`):

| Candidate | Result | Why it lost |
|---|---|---|
| Gemma 4 12B QAT (`gemma4_unified`) | 8/9 (Python), **10.5 GB resident**, 12.6–15.5 GB peak | Full-multimodal weights alone ~1.7× the phone cap. Mac-only reference. |
| Gemma 4 E4B QAT | **9/9 stock** (Python; QAT lifts plain E4B's 5/9 → 9/9) but **8/13 on our grid** @ ~5.3 GB | Perfect on single-turn; **fails all 5 knowledge chains** (untuned); bigger + dense-slower than LFM; MLX loader can't parse the QAT layout (llama.cpp can). Only ever wins 1 of 3 axes even if fine-tuned. |
| MTP (draft-mtp, official drafter, Metal) | **−13% decode** on our 13-prompt workload (31% acceptance), measured on llama.cpp HEAD | Speculative overhead exceeds gain on Metal; community drafter GGUFs are fork-incompatible (convert the official assistant yourself). Parked. |
| llama.cpp b9434 → HEAD bump | HEAD ~13% *slower* prefill on LFM, decode wash | Keep the pin. |

Also ran one more FT iteration (**v8hist**: +224 history-chain rows targeting
the one failing scenario). It fixed that scenario but stably broke two others
(11/13 net) — shelved, and the ±1-noise lesson above came out of it.

## Phase 5 — quantization frontier (the last free win)

Plain PTQ frontier (`LFM25_MEMORY_PERF_FRONTIER.md`): Q4_K_M 11/13 @ 5.16 GB;
**Q3_K_M 12/13 @ 4.16 GB** (shipped); Q2_K collapses (3/13 @ 3.28 GB).

2026-06-10: filled the Q3→Q2 gap with an **importance matrix** computed on
2 MB of our own tool-call transcripts + **lattice i-quants**:

| Quant (imatrix) | /13 | Peak RSS | Decode (b9434) |
|---|---|---|---|
| IQ2_M (2.6G) | 5/13 | 2.97 GB | — (collapses) |
| IQ3_XXS (3.1G) | 11/13 | 3.43 GB | 127 t/s |
| **IQ3_XS (3.3G)** ★ | **12/13 ×3 runs** | **3.64 GB** | **136 t/s** |
| Q3_K_S (3.5G) | 9/13 | 3.83 GB | — (K-quants lose < ~3.5 bpw) |
| Q3_K_M (3.8G, prev ship) | 12/13 | 4.17 GB | 110 t/s |

**IQ3_XS strictly dominates the old ship**: same stable accuracy, −0.53 GB,
+24% decode (smaller weights win on a bandwidth-bound MoE; the feared i-quant
Metal penalty never materialized). True QAT was evaluated and *not* needed:
no tooling targets llama.cpp K-quant layouts, and calibrated PTQ captured
the available win. Sub-3-bpw would need an MoE-aware per-tensor mix
(`--tensor-type`: experts low, router/attention/embeddings high) or QAT.

## Where we landed (shipping config, 2026-06-10)

- **Model:** LFM2.5-8B-A1B, LoRA-fine-tuned on our tool corpus (v7-full)
- **Quant:** **IQ3_XS + imatrix** (3.3 GB file; HF
  `sliderforthewin/lfm2.5-8b-a1b-ft-GGUF`)
- **Runtime:** llama.cpp b9434 xcframework, q8_0/q8_0 KV, flash-attn,
  `swa_full=false`
- **Context:** **n_ctx 32,768** (~100 MB KV; the model trains to 131k) with
  **cross-turn KV prefix reuse** (two-tier: pure-append works on the hybrid;
  divergence falls back to full prefill — measured 99.3% reuse, 23-token
  prefills, turn 2 at 1.3 s vs turn 1's 3.5 s)
- **Scorecard:** 12/13 (single floating borderline chain), ~3.6 GB peak,
  136 t/s decode (M2 Max; ~46+ t/s-class on iPhone) — vs the 6,144 MB cap

## Lessons that generalize

1. **Fine-tuning on the exact tool surface beats model shopping.** 2/13 →
   12/13 (LFM); 7→10/13 (Gemma). Every untuned challenger lost the chains.
2. **MoE is the phone architecture.** Active params buy speed, total params
   amortize into quantized memory; dense 4B+ can't win all three axes.
3. **Match the training prompt format to inference byte-for-byte** or the FT
   silently regresses.
4. **Tool-count is a cliff, not a slope** for small models (Gemma 4 E2B dies
   at ≥11 declarations). Culling the tool surface is a model-quality lever.
5. **Calibrated quantization (imatrix on your own transcripts) ≈ poor-man's
   QAT** — and i-quants beat K-quants below ~3.5 bpw. Always bench speed too:
   the smaller quant was *faster*.
6. **Trust the harness only as far as its noise floor** — ±1 on a 13-grid at
   temp 0.2–0.3; majority-of-3 before believing scenario-level deltas.
7. **Python harness scores are floors** (template quirks); confirm in the
   Swift e2e harness and on-device before drawing model conclusions.
8. **Hybrid models constrain the runtime**: recurrent state can't be
   partially truncated (KV reuse needs the append-only path), and upstream
   loader support (mlx-swift-lm vs llama.cpp) becomes a selection criterion.
