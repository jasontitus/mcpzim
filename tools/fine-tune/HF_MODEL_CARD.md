---
license: other
license_name: lfm-open-license
license_link: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
base_model: LiquidAI/LFM2.5-8B-A1B
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - tool-calling
  - function-calling
  - offline
  - ios
  - on-device
  - moe
  - imatrix
---

# LFM2.5-8B-A1B — offline tool-calling fine-tune (GGUF)

A LoRA fine-tune of [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
(8.3B-total / **1.5B-active** hybrid MoE) for **on-device, fully-offline
tool calling** over local knowledge archives. It is the shipping model of
[Zimfo / mcpzim](https://github.com/jasontitus/mcpzim) — an iPhone app that
answers questions from offline ZIM archives (Wikipedia, OpenStreetMap,
medical) by driving an 11-tool loop (article search/retrieval, nearby-places,
routing, comparisons) and grounding every answer in tool results.
**Nothing leaves the device.**

Why this base: on a phone, *active* parameters set decode speed, *total*
parameters set memory, and quantization sets the exchange rate. LFM2.5's
1.5B-active MoE decodes like a small model (~46 tok/s on an iPhone 17 Pro
Max) while carrying 8B-class capacity — and only 6 of its 24 layers are
attention (~6.9 KB/token KV at q8_0), so a 32k context costs just ~226 MB.

## Files

| File | Quant | Size | Notes |
|---|---|---|---|
| `lfm2.5-8b-a1b-ft.imx.IQ3_XS.gguf` | **IQ3_XS + imatrix** ★ | 3.56 GB | **Recommended.** Importance matrix computed on the app's own tool-call transcripts; strictly dominates Q3_K_M below. |
| `lfm2.5-8b-a1b-ft.Q3_K_M.gguf` | Q3_K_M | 4.11 GB | Previous shipping quant, kept for rollback/comparison. |

## Evaluation

13-scenario tool-calling grid (single-turn tool selection + multi-turn
knowledge chains whose final turn must be answered **from cached context
with no tool call**), run via llama.cpp with q8_0/q8_0 KV. Full methodology
and history: [MODEL_EVALUATION_HISTORY.md](https://github.com/jasontitus/mcpzim/blob/main/MODEL_EVALUATION_HISTORY.md).

| Model | Grid | Peak RSS | Decode (M2 Max) |
|---|---|---|---|
| **This FT, IQ3_XS imatrix** | **12/13** (stable ×3 runs) | **3.64 GB** | **136 t/s** |
| This FT, Q3_K_M | 12/13 | 4.17 GB | 110 t/s |
| Base LFM2.5-8B-A1B (stock, Q4_K_M) | **2/13** | 5.2 GB | — |
| Gemma 3 4B fine-tune (same corpus) | 10/13 | 3.18 GB | — |
| Gemma 4 E4B QAT (stock) | 8/13 | ~5.3 GB | — |

The stock-vs-FT gap (2/13 → 12/13) is the point of this artifact: the base
model is highly capable, but reliable tool calling on a *specific* tool
surface comes from training on it.

### Why IQ3_XS + imatrix

The importance matrix was computed with `llama-imatrix` over ~2 MB of the
app's real tool-call transcripts (the model's exact deployment
distribution), then used for lattice i-quantization. Versus plain Q3_K_M:
same accuracy, **−0.53 GB peak RSS, +24% decode** (the MoE is
memory-bandwidth-bound, so smaller weights decode faster on Metal).
Below ~3.5 bpw, imatrix i-quants beat K-quants outright in our testing
(Q3_K_S scored 9/13 despite being larger); IQ2_M collapses (5/13).

## Prompt format

ChatML turn markers (`<|im_start|>` / `<|im_end|>`; llama.cpp adds the
`<|startoftext|>` BOS automatically). The fine-tune corpus **folds the
system prompt + tool declarations into the first user turn**, and tool
responses come back as **user** turns:

````
<|im_start|>user
{system prose}

{tool declarations block}

{first user question}<|im_end|>
<|im_start|>assistant
```tool_call
{"function": "article_overview", "parameters": {"title": "Solar panel"}}
```<|im_end|>
<|im_start|>user
[TOOL_RESPONSE name=article_overview]
{"title": "Solar panel", "lead": "...", "available_sections": [...]}<|im_end|>
<|im_start|>assistant
Solar panels convert sunlight into electricity through photovoltaic cells...<|im_end|>
````

Tool calls are emitted as fenced ` ```tool_call ` blocks containing
`{"function": ..., "parameters": {...}}` JSON. The exact renderer/parser
(including the malformed-JSON repair passes that production needs) lives in
[`LFM25Template.swift`](https://github.com/jasontitus/mcpzim/blob/main/swift/Sources/MCPZimKit/LFM25Template.swift)
(turn markers) delegating to `Gemma3Template` (body/parse).

## Running it

```sh
# llama.cpp — the configuration the app ships:
llama-server -m lfm2.5-8b-a1b-ft.imx.IQ3_XS.gguf \
  -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q8_0
```

Notes from production use:
- The model trains to 131k context; the app runs **n_ctx=32768**
  (~226 MB KV at q8_0). Budget math:
  [CONTEXT_BUDGET.md](https://github.com/jasontitus/mcpzim/blob/main/CONTEXT_BUDGET.md).
- LFM2.5 is a **hybrid** (recurrent shortconv + 6 attention layers):
  llama.cpp cannot partially truncate its recurrent state — KV reuse across
  turns works for **append-only** prompts; on divergence, re-prefill from
  scratch.
- Long replies are cheap (KV is pre-allocated at n_ctx); the app floors
  reply budgets at 1024 tokens.

## Training

LoRA (rank 16, 8 layers, 800 iters, bsz 2, lr 1.5e-5, max-seq 1792) via
mlx-lm on ~4.7k synthetic tool-call trajectories generated by larger teacher
models against the app's exact tool schema and prompt rendering — single-turn
calls, 2-turn chains, and targeted hard-case rows. Fused, converted with
llama.cpp's `convert_hf_to_gguf.py` (the pipeline unstacks the MoE experts
and restores the upstream tokenizer), then quantized. Recipe:
[`tools/fine-tune/`](https://github.com/jasontitus/mcpzim/tree/main/tools/fine-tune).

## Intended use & limitations

- Built for **Zimfo's tool surface** (offline Wikipedia/OSM tools, JSON
  calls, the prompt format above). It is *not* tuned as a general chat
  assistant, and other tool schemas / prompt formats will underperform —
  the 2/13 stock score cuts both ways.
- The 13th grid scenario is a borderline multi-turn chain that floats at
  sampling temperature 0.2–0.3 (the eval's noise floor is ±1).
- Inherits the base model's license (LFM Open License — see the
  [base model](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)) and its
  knowledge cutoff/limitations.
