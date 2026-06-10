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

## Recommendation

- **MTP: park it.** Unmerged on MLX; untested + currently-regressing on
  llama.cpp/Metal. Re-check in a few weeks (fast-moving).
- **QAT quality is reachable now without waiting on upstream:** take the
  **unquantized QAT BF16 master** and re-quantize to a *standard* MLX 4-bit
  layout (`mlx_lm.convert`). Keeps most of QAT's quality but produces a checkpoint
  our mlx-swift loader already accepts (like plain e4b-it-4bit did) — sidesteps
  both the QAT-quant-layout gap and the wNa8o8 problem. **Not yet tried.**

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
