# Gemma 4 QAT + MTP on-device — research findings (2026-06-09)

Deep-research pass (23 sources, 25 claims adversarially verified — 20 confirmed,
5 killed) on whether we can run **Gemma 4 E4B QAT** and/or **MTP** (multi-token
prediction / speculative decoding) on the iPhone, via our two runtimes:
**mlx-swift-lm** (MLX) and the **llama.cpp xcframework** (LlamaCppSwift, pinned
b9434). Time-sensitive — everything here is dated within days of 2026-06-09 and
is moving week-to-week; **re-check before committing to an architecture.**

## TL;DR

Both QAT and MTP exist and are officially Google-blessed (across llama.cpp,
Ollama, LM Studio, vLLM, SGLang, LiteRT-LM, MLX). **But the specific combo we
want — E4B QAT + MTP end-to-end on iOS — is NOT shippable today on either of our
runtimes. The gap is on the Apple-Silicon *client* side, not Google's release.**

## QAT scoring — the win is real and large

Same harness (Python native-template eval, 9 tool-call scenarios — the only one
that could run BOTH; QAT can't load in our Swift CLI, see below):

| Gemma 4 E4B (4-bit) | Tool-call score |
|---|---|
| Plain q4 (PTQ, `mlx-community/gemma-4-e4b-it-4bit`) | **5/9** |
| **QAT q4** (`mlx-community/gemma-4-e4b-it-qat-4bit`) | **9/9** |

QAT cleared the exact 4 scenarios plain q4 choked on (it removed the tool-count
cliff that made plain E4B emit empty output), landing E4B at **12B-class scores**
(12B QAT = 8/9, Gemma 3 12B = 9/9) at ~1/3 the size. Caveats: (1) the Python
native path under-scores (Gemma-4 template quirks), so read 5→9 as a strong
*relative* signal; (2) **no Swift-CLI QAT number exists** (loader gap) — for
reference plain e4b scored 13/18 in the Swift CLI, a different/larger set.

## QAT reality: it's 4 incompatible flavors, not one

Google ships the E4B QAT as **four distinct layouts**; a runtime must target the
right one:
1. **Unquantized QAT Q4_0** — BF16 master safetensors (the re-quant *source*).
2. **GGUF Q4_0** — ready for llama.cpp / LM Studio.
3. **Mobile-optimized `wNa8o8`** — Google's *purpose-built-for-phones* format
   (2-bit decode layers, compressed KV/vocab, static activations). **NOT GGUF;
   not loadable by llama.cpp or mlx-swift as-is.** The phone-targeted variant is
   loadable by *none* of our runtimes.
4. **Compressed-tensors w4a16** — for vLLM.

"MLX" in Google's docs means the Python/Apple-Silicon ecosystem generically — it
is **not** an endorsement of mlx-swift / mlx-swift-lm (1–2 tiers behind mlx-lm).

## The two iOS paths, precisely

- **MLX (mlx-swift-lm): blocked / unreleased.** The Gemma 4 MTP drafter class in
  Python mlx-lm (PR #1276) is **open/unmerged**, lands only the model class (no
  spec-decode loop), in no release. mlx-swift-lm is further behind. Separately,
  the **QAT load gap is real**: plain `gemma-4-e4b-it-4bit` fails in mlx-lm with
  "126 parameters not in model" (layers 24–41 ship redundant K/V tensors from
  KV-sharing) — same class as our mlx-swift-lm `keyNotFound … k_proj/weight` on
  the QAT 4-bit. **No landed fix confirmed**; mlx-swift-lm #282 unverified.
- **llama.cpp: generic MTP merged (~b9330, `--spec-type draft-mtp`; our b9434
  post-dates it), but Gemma-4 MTP is UNTESTED in-repo** (the PR's only test
  models are Qwen; "Gemma" appears nowhere). And the only **Apple-Silicon Metal
  benchmarks show MTP *regressing* decode 11–28%** (Qwen, b9330) — the ~1.7–2.2×
  speedup is **unproven on Metal** today. (Medium confidence: small-N early bug
  reports; the "MTP always degrades on Metal" generalization was refuted, so
  treat as "unproven," not "permanently slower." Unmeasured on Gemma.)

## MTP drafters

Official E4B drafter exists (`google/gemma-4-E4B-it-assistant`, Apache-2.0) and
QAT+MTP is co-designed (official `{model}-qat-q4_0-unquantized-assistant`). But
the official QAT drafters ship as **BF16 safetensors, not GGUF**; ready-to-run
GGUF MTP drafters are **community-only** and lossy to convert (naive Q4_0 drops
step-0 acceptance ~80% → ~35%).

## MEASURED on Apple Silicon Metal (2026-06-09, M2 Max, llama.cpp master 76da245)

We built llama.cpp HEAD (post-#24282) with Metal and ran the first known
Gemma-4-QAT + MTP measurement on Apple Silicon (llama-server `/completion`,
256 tok, seed 1, fa on, ngl 99):

| Run | Decode | Notes |
|---|---|---|
| E4B QAT Q4_0, no draft | **63–69 t/s** | `llama-bench tg128` = 68.98; server runs 63.1–68.6 |
| E4B QAT Q4_0 + MTP (`draft-mtp`, official drafter, n-max 3) | **67.3 t/s** | **draft acceptance 31%** (123/392) |

**Verdict: MTP works on Metal and does NOT regress — but the gain is ~+7%,**
nowhere near the 1.7–2.2× headline (the low 31% acceptance eats the benefit;
the PR author saw 48% on Snapdragon and called it "not amazing"). This refutes
both the hype AND the "MTP regresses on Metal" generalization from #23752 —
at least for Gemma 4 E4B on an M2 Max.

**Conversion gotcha (cost us a debugging loop):** community Gemma-4 MTP
drafter GGUFs (e.g. AtomicChat) are FORK-converted and do NOT load upstream —
wrong arch string (`gemma4_assistant` vs upstream `gemma4-assistant`) and an
hparams mismatch (`GGML_ASSERT(n_layer_nextn == n_layer_all)`). The working
recipe is the PR's: convert the OFFICIAL
`google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` (79M params) with the
SAME llama.cpp checkout's `convert_hf_to_gguf.py` → 164 MB F16 GGUF, arch
`gemma4-assistant` → loads + drafts cleanly.

Also confirmed: the official main QAT GGUF
(`google/gemma-4-E4B-it-qat-q4_0-gguf`, 4.79 GiB, arch `gemma4`) loads and
runs cleanly on Metal via llama.cpp — **the QAT quality win is reachable
through our llama.cpp path with no MLX loader gap.**

## FULL TEST vs our shipping model (2026-06-09, M2 Max)

The complete three-leg test, run the same way prior models were scored:

**1. Full 13-scenario grid** (`tools/llama-smoke/grid.py`, KV q8_0/q8_0 — the
exact config behind LFM2.5-FT's 12/13; results in
`tools/llama-smoke/GRID_RESULTS_GEMMA4_E4B_QAT.md`):

| Model | Score | Peak RSS | Notes |
|---|---|---|---|
| **LFM2.5-8B-A1B FT** *(shipping)* | **12/13** | **4.16 GB** | fine-tuned on our tools + chains |
| **Gemma 4 E4B QAT Q4_0** *(stock)* | **8/13** | ~5.3 GB | perfect on single-turn (8/8 incl. both compares); **all 5 fails are the multi-turn knowledge chains** |

Stock-vs-tuned read: QAT E4B never hit a tool-format cliff and aced every
single-turn scenario — impressive for an untuned model — but loses every deep
chain, exactly where our FT data targeted LFM. Quirk: it emits
`compare_articles` arguments as a bare JSON **list** (`["A","B"]`), which
crashed the harness until `dispatch_tool` learned to normalize it; the
**in-app Swift adapter would need the same tolerance** before QAT could ship.

**2. MTP enabled vs disabled** (same 13 scenario prompts, HEAD llama-server,
Metal; two degenerate instant-EOS prompts excluded):

| Config | Mean decode (11 prompts) | Acceptance |
|---|---|---|
| MTP OFF | **61.7 t/s** | — |
| MTP ON (`draft-mtp`, official drafter, n-max 3) | **53.6 t/s** | 31% (943/3060) |

**MTP = −13% on our actual workload, slower on every prompt.** The earlier
single-prompt +7% was variance; across the real prompts MTP regresses on
Metal, consistent with llama.cpp #23752. **Definitive skip** until acceptance
or the Metal draft path improves upstream.

**3. In-app shippability:** the **shipping b9434 xcframework loads + decodes
the QAT GGUF cleanly** (probe: `MCPZimEvalCLI --probe-llama --gguf <qat>` →
load OK, 64 chunks on MTL0). **QAT needs no llama.cpp bump**; only MTP would.

**Bottom line:** LFM2.5-FT stays the shipping model (12/13 @ 4.16 GB beats
stock 8/13 @ 5.3 GB). The live opportunity is **fine-tuning E4B QAT on our
chain data** — its stock single-turn perfection suggests real FT headroom —
but that requires a QAT-aware FT pipeline (BF16 QAT master → FT → re-quant)
and the adapter list-args tolerance first.

## Recommendation (updated with measurements)

- **Ship QAT via llama.cpp, skip MTP.** The official QAT Q4_0 GGUF loads and
  runs cleanly on Metal (~63–69 t/s on M2 Max) — the 9/9-class quality win is
  reachable through `LlamaCppProvider` with no MLX dependency. MTP measured at
  **~+7% on Metal** — not worth a second model + a fork-sensitive conversion
  pipeline + acceptance tuning. Revisit MTP if acceptance improves upstream.
- App caveat: our pinned xcframework (b9434) predates the MTP merge by ~134
  commits — fine for QAT-without-MTP (gemma4 arch support is older); an
  xcframework bump + a Swift draft/verify loop would be needed for in-app MTP.
- The MLX re-quant path (QAT BF16 master → `mlx_lm.convert` standard 4-bit) is
  still the MLX-side option, but llama.cpp now looks like the shorter route.

## Open questions (unanswered by the research)

1. Is our mlx-swift-lm `keyNotFound k_proj` tracked by #282, and is there a fix
   or a strict-load / tensor-stripping workaround? (needs a direct repo check)
2. Does LlamaCppSwift @ b9434 expose `--spec-type draft-mtp` on iOS, and what's
   the min tag that loads Gemma-4 QAT GGUF *and* runs Gemma-4 MTP (vs Qwen-only)?
3. Has anyone produced a Gemma-4 E4B MTP GGUF llama.cpp's draft-mtp accepts, with
   measured E4B acceptance / decode speedup on Metal specifically?
4. Is the mobile `wNa8o8` variant loadable by any iOS runtime (LiteRT-LM?)?

## Sources (primary)

- Google QAT blog: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- Google MTP blog: https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- Gemma docs (core + MTP): https://ai.google.dev/gemma/docs/core , https://ai.google.dev/gemma/docs/mtp/mtp
- E4B QAT flavors: https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers
- E4B MTP drafter: https://huggingface.co/google/gemma-4-E4B-it-assistant
- llama.cpp MTP PR: https://github.com/ggml-org/llama.cpp/pull/22673 ; Metal regression: https://github.com/ggml-org/llama.cpp/issues/23752
- mlx-lm Gemma-4 MTP PR (open): https://github.com/ml-explore/mlx-lm/pull/1276 ; QAT load gap: https://github.com/ml-explore/mlx-lm/issues/1242
