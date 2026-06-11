# Context budget on the shipping model (LFM2.5 FT · IQ3_XS) — 2026-06-10

How much conversation the app can actually hold, and what it costs. Shipping
config: `n_ctx = 32,768` with q8_0/q8_0 KV and cross-turn KV prefix reuse
(`LlamaCppProvider.contextTokens`, set in `ChatSession.init`).

## Why LFM2.5's context is unusually cheap

LFM2.5-8B-A1B is a hybrid: **only 6 of its 24 layers are attention** (8
KV-heads × 64 head-dim); the other 18 are recurrent shortconv whose state is
**O(1) in context length**. Exact KV cost (from the model config):

```
6 layers × 2 (K+V) × 8 kv-heads × 64 dim × 2 B  = 12 KB/token  (fp16)
                                       at q8_0  ≈ 6.9 KB/token
```

llama.cpp pre-allocates the KV buffer for the whole window at load, so n_ctx
is a constant resident cost, not a grows-with-use cost:

| n_ctx | q8_0 KV resident | Worst-case full re-prefill* |
|---|---|---|
| 8,192 *(old)* | ~57 MB | ~4 s |
| **32,768 (shipping)** | **~226 MB** | **~15 s** |
| 131,072 *(model max)* | ~906 MB | ~60 s |

\* at the measured 2,142 t/s prefill (M2 Max, b9434, IQ3_XS). iPhone prefill
is slower (not yet measured for IQ3_XS) — scale accordingly.

For comparison, a Gemma-3-class dense model pays several× more per token
(more attention layers), which is why the Gemma GGUF fallback providers stay
at the 8k default.

## Memory: does 32k fit the phone?

Hard process cap (iPhone 17 Pro Max, with entitlement): **6,144 MB**.

| Component | MB |
|---|---|
| Model peak RSS, IQ3_XS @ 8k ctx (measured, 13-grid) | ~3,640 |
| Extra KV for 32k (226 − 57) | ~170 |
| **Model-side total @ 32k** | **~3,810** |
| Headroom for Kokoro TTS (~300), live map WebView (300–500), ZIM readers, UI | **~2,330** |

Comfortable. Even 131k would fit (~4.5 GB model-side) — it's **latency**, not
memory, that caps the window (next section).

## What 32k means in conversation terms

Budget composition per the app's prompt assembly:

- System preamble + 11 tool declarations: **~1.5–2k tokens**
- Typical tool-loop turn (user msg + tool call + trimmed result + reply):
  **~0.4–2.5k tokens** (places queries cheap; article/discuss turns heavy —
  discuss grounds ~6 passages ≈ 2–2.5k)
- Reply headroom: up to 1,024 tokens (`replyTokensFloor`)

So ~29–30k of usable history ≈ **roughly 15–50 turns** depending on tool
weight (vs **4–8 turns** at the old 8k, which tool-heavy walks exhausted).
Overflow now throws a clear "prompt exceeds n_ctx" error instead of a
cryptic decode failure.

## Why per-turn latency stays flat: KV prefix reuse

Without reuse, every turn re-prefills the whole transcript — cost grows
linearly with conversation length (~15 s per turn at a full 32k on the Mac;
worse on phone). With the two-tier reuse shipped 2026-06-10:

- **Append turns** (the normal case — the transcript strictly extends):
  measured **~99.3% reuse, 23-token prefills** across tool-loop iterations
  and follow-up turns; turn 2 of a planning conversation answered in 1.3 s
  vs turn 1's 3.5 s.
- **Shape switches** (e.g. tool-loop ↔ discuss-mode grounded prompts, which
  are built fresh): no shared prefix → full re-prefill. This is the
  worst-case column above, and it's also why LFM (hybrid) can't use partial
  KV truncation — recurrent state only supports append or full wipe.

## Why not 131k today

1. **Worst-case re-prefill**: a shape switch deep in a 131k transcript would
   stall ~1 min on Mac, likely several on phone — unacceptable for voice.
2. ~900 MB of always-resident KV spends 40% of the app's headroom on a
   window we can't fill gracefully yet.

The path to bigger effective context isn't a bigger n_ctx — it's
**transcript compaction** (summarize/drop tool *responses* older than ~N
turns; the deterministic ConversationFocus already preserves their
semantics), which stretches any window ~3–5× and shrinks the worst-case
re-prefill at the same time. Scoped, not yet built.

## Pointers

- `LlamaCppProvider.contextTokens` + the KV-reuse implementation:
  `ios/MCPZimChat/Providers/LlamaCppProvider.swift` (debug:
  `MCPZIM_KV_DEBUG=1` prints `[kv] prompt/reuse/prefill` per generate)
- Quant decision: `tools/llama-smoke/LFM25_MEMORY_PERF_FRONTIER.md`
- Model selection arc: `MODEL_EVALUATION_HISTORY.md`
