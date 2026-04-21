# KV cache compression — options for on-device Gemma 4

This is a decision doc for how we shrink the KV cache on MLX Swift. See
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) for the broader memory-tuning context —
this file only covers the KV-cache axis.

## Why this matters

Baseline today: KV cache is FP16. On a ~2 500-token prompt (routing follow-up
turn after tool-result trimming) that's **~500 MB of KV alone**, per
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) §"Known remaining costs". On a 6 GB
iPhone with 2.6 GB of Gemma-4 weights already resident plus libzim cluster
caches plus the SZRG routing graph, that KV footprint is what drives us to
cap `articleCapKB` and `maxReplyTokens` defensively.

KV-cache quantization is the single biggest remaining memory lever. The rest
of this doc evaluates two concrete options.

## Option 1 — MLX built-in `QuantizedKVCache` (recommended; ships today)

### What it is

MLX already ships `QuantizedKVCache` in `mlx-swift`. Groupwise 4-bit
quantization of K and V (`kvBits: 4, kvGroupSize: 64`). No new kernels
required, no fork of MLX itself.

Claimed memory cut: ~4× vs FP16. On our 2 500-token prompt that's
~500 MB → ~125 MB.

### Why it's not enabled today

The hook is already wired up and commented out in
`ios/MCPZimChat/Providers/Gemma4Provider.swift:153-160`. The blocker is
upstream:

> Swift-Gemma4-Core 0.1.0's `Gemma4TextAttention` calls
> `cache.update(keys:values:)` unconditionally. MLX's `QuantizedKVCache`
> rejects that path with a runtime assertion
> (`mlx-swift/Source/MLXLMCommon/KVCache.swift:894`). The quantized cache
> wants `cache.updateQuantized(keys:values:)`.

This is tracked as watch-list item #4 in `OPTIMIZATIONS.md`.

### Enablement path

1. Fork `Swift-Gemma4-Core`. Teach `Gemma4TextAttention` to branch:
   ```swift
   if let qCache = cache as? QuantizedKVCache {
       (qKeys, qValues) = qCache.updateQuantized(keys: keys, values: values)
   } else {
       (keys, values) = cache.update(keys: keys, values: values)
   }
   ```
   And route the subsequent attention call through
   `MLX.quantizedScaledDotProductAttention` when the cache is quantized.
2. Pin `Package.swift` / `Package.resolved` to the fork (or a branch of it).
3. Flip `kvBits: 4, kvGroupSize: 64` in `Gemma4Provider.generate(...)`. Also
   consider `quantizedKVStart: 256` so the first ~256 tokens stay FP16 —
   mitigates the short-prompt TTFT regression (see tradeoffs below).
4. Gate on `DeviceProfile`: enable on ≤ 8 GB devices where memory is the
   binding constraint. Leave FP16 on Macs where the quality/speed trade
   isn't worth it.
5. File a PR upstream on Swift-Gemma4-Core so the fork can eventually be
   retired.

### Tradeoffs

**Memory (the win).** ~4× shrink on the KV tensors. Only KV — weights and
activations are unchanged. On a 6 GB iPhone this is the single biggest
remaining lever; on a 16+ GB Mac it's mostly a non-event.

**Quality (small but real).** 4-bit groupwise is usually called
"near-lossless". Published numbers on comparable schemes (KIVI, QuaRot,
KVQuant) land at <0.1 perplexity delta and <1% MMLU drop. Caveats:

- **Keys quantize worse than values.** K-channels have outlier features that
  RoPE (used by Gemma) amplifies. Some models regress more on K-side than
  others; Gemma-4-E2B-specific numbers aren't published anywhere we've
  found.
- **mcpzim-specific risk: tool-call robustness.** `ChatToolCallParser` parses
  `<tool_call>{…}</tool_call>` JSON out of the raw token stream. A stray
  space, a missed `<end_of_turn>`, or a mis-escaped quote near a decision
  boundary will make parsing flakier. The existing Gemma4 prompt-formatter
  byte-equality tests won't catch this; we'd want an added integration test
  that asserts tool-call parse success rate across a representative prompt
  corpus.

**Speed (ambiguous, context-dependent).**

- **Prefill / TTFT: slightly slower.** Quantizing prompt K/V costs compute
  that FP16 doesn't. Visible in the `stream opened in X s` log line.
- **Decode: usually faster at longer contexts** — attention is
  memory-bandwidth-bound and we just shrank the bandwidth 4×. At our
  2 500-token prompts the decode win is modest; below ~1k it can be a
  wash.
- MLX mitigates the short-context hit via `quantizedKVStart` — keep the
  first N tokens in FP16, only quantize the tail. Worth setting.

**Integration fragility (the real cost).** Forking `Swift-Gemma4-Core` is a
permanent maintenance tax:

- Rebase on every upstream release. `OPTIMIZATIONS.md` watch-list #3 (Gemma
  chat-template revisions) and #4 (this issue) both imply upstream churn.
- Our test suite already asserts byte-equality with
  `Gemma4PromptFormatter.userTurn(...)` from Swift-Gemma4-Core; any fork
  divergence beyond the attention path needs to stay out of scope or those
  tests break.
- The alternative is to land the fix upstream via PR — cheaper long-term,
  slower to ship.

**Operational gotchas.**

- **Memory-spike transition.** First `generate()` briefly holds both FP16
  and quantized paths during warmup. With `MLX.GPU.cacheLimit: 512 MB`
  (`Gemma4Provider.init`) this should be fine but needs a device check
  before shipping.
- **Prompt-caching interaction** (`OPTIMIZATIONS.md` watch-list #5). If we
  later adopt prefix reuse across turns, the quantized cache has to be
  saved/restored in its quantized form. Not a blocker; just a constraint.
- **Output determinism.** Token-stream snapshot tests would shift. The
  byte-equality test on `Gemma4PromptFormatter` is upstream of the cache
  and is unaffected.

### When to enable vs skip

| Situation | Enable? |
| --- | --- |
| 4 GB / 6 GB iPhone, routing / long-article workloads | **Yes** — the binding constraint is KV memory. |
| 8 GB iPhone Pro | Yes, via DeviceProfile. |
| 16+ GB Mac, developer workflow | Skip — quality/speed not worth a small memory win. |
| Tool-call regression observed in eval | Back off to FP16 until groupwise / `quantizedKVStart` retuned. |

## Option 2 — TurboQuant (watch-list, not yet)

### What it is

Google Research technique (ICLR 2026, published late 2025 / early 2026).
Data-oblivious KV quantization: random-rotate the K/V vectors, apply
Lloyd-Max scalar quantization per group, optionally add a 1-bit QJL
residual for error correction. Claims:

- 3-bit K + 3-bit V ≈ ~5–6× compression vs FP16
- "Near-lossless" quality without calibration data
- Up to 8× faster attention on H100 (via fused CUDA/Triton kernels)

### MLX landscape (as of April 2026)

| Project | Flavor | Integration | Notes |
| --- | --- | --- | --- |
| [SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM) | **MLX Swift** | Monolithic inference server, internal fork of `mlx` + `mlx-c` | MIT, 411★, Gemma 4 dense + MoE listed. ~3.5× KV compression ("~3.6 bits/coord"). **Not a consumable Swift Package.** |
| [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) | MLX Python | Drop-in `KVCache` + monkey-patch | Apache 2.0, fused Metal kernels, 4.6× at 98% FP16 speed. Kernels tied to Python bindings. |
| [helgklaizar/turboquant_mlx](https://github.com/helgklaizar/turboquant_mlx) | MLX Python | 1–3 bit + asymmetric K8/V3 | OpenAI-compat server. |
| [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) | MLX Python | Drop-in mlx-lm replacement | |
| [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) | MLX Python | Drop-in mlx-lm replacement | |
| [Incept5/gemma4-benchmark](https://github.com/Incept5/gemma4-benchmark) | MLX Python | Benchmark harness | Actual Gemma-4 TurboQuant numbers — see below. |

The key observation: **only SwiftLM is Swift-native, and it's a server
binary, not a library.** Its Metal kernels live inside an internal `mlx`
fork. Consuming it from `MCPZimChat` would mean either vendoring that fork
(significant maintenance commitment — you'd track two upstreams forever)
or extracting the Metal shaders into a standalone package (meaningful
engineering work). Every other impl is Python, unreachable from a Swift
iOS app.

### Benefit analysis for mcpzim specifically

Concrete numbers from `Incept5/gemma4-benchmark` (M5 Max, Gemma 4 E2B):

- **Below 16k context: negligible speed benefit.**
- 32–64k context: +5–10% tokens/sec.
- 128–256k context: +15–19% tokens/sec.

Our `ChatSession` caps replies at 256 tokens and trims tool results to
~4 KB / ~2 500 tokens (`OPTIMIZATIONS.md` line 23). **We never get near
16k.** The TurboQuant speed curve doesn't bite at our workload.

Where it does help:

- **Memory.** At our prompt sizes, ~500 MB KV at FP16 → ~125 MB at MLX 4-bit
  → ~110 MB at TurboQuant ~3.6 bits/coord. That's a ~15 MB marginal win
  over Option 1 — real but modest.
- **Headroom for raising `articleCapKB` / `maxReplyTokens`.** If we want
  to feed whole Wikipedia sections instead of 4 KB slices, TurboQuant
  pushes the break-even out. This is where its long-context curve becomes
  interesting — but it's a future-us problem, not a today-us problem.

### Trigger to revisit

Revisit TurboQuant when **either**:

1. `mlx-swift-lm` upstream ships a `TurboQuantKVCache` we can use without
   forking.
2. Someone (us or the community) extracts SwiftLM's Metal kernels into a
   standalone Swift Package with a clean `KVCache`-conforming API.
3. We raise `articleCapKB` / `maxReplyTokens` enough that the cache routinely
   crosses 16k tokens, at which point the Incept5 speed curve starts paying.

## Recommendation

1. **Ship Option 1 first.** Fork `Swift-Gemma4-Core`, patch the attention
   path, enable `kvBits: 4, kvGroupSize: 64, quantizedKVStart: 256`, gate
   on `DeviceProfile`. Captures ~90% of the memory win available from
   KV-cache quantization. Small diff, clear upstream PR path.
2. **Add a tool-call regression eval** as part of Option 1 — representative
   prompt corpus, assert parse success rate. Protects against the
   mcpzim-specific quality risk.
3. **Update `OPTIMIZATIONS.md` watch-list** to cross-reference this doc
   and add TurboQuant as a new entry alongside "MLX dcache" and
   "Lightning quants".
4. **Hold off on TurboQuant** until one of the triggers above fires.

## References

- [TurboQuant: Redefining AI efficiency with extreme compression — Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [SharpAI/SwiftLM — MLX Swift inference server with TurboQuant](https://github.com/SharpAI/SwiftLM)
- [arozanov/turboquant-mlx — fused Metal kernels, 4.6× at 98% FP16 speed](https://github.com/arozanov/turboquant-mlx)
- [Incept5/gemma4-benchmark — Gemma 4 × TurboQuant numbers on Apple Silicon](https://github.com/Incept5/gemma4-benchmark)
- [TurboQuant — Extreme KV Cache Quantization — llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [ml-explore/mlx-swift — `QuantizedKVCache`](https://github.com/ml-explore/mlx-swift)
