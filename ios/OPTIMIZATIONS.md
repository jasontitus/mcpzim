# MCPZimChat tuning notes

Running Gemma 4 + a 1 GB routing ZIM on-device is tight on RAM. This file
captures what we tuned, why, and what upstream features to watch for so we
can keep getting wins as the ecosystem moves.

## Why this matters

Baseline (naive setup) peaked at **45+ GB resident** on a 128 GB dev Mac —
fine there, fatal on a 6–8 GB iPhone. We're currently plateaued at **~4 GB
resident, ~6 GB transient peak** with 4-bit quantized KV cache (down from
~6 GB steady / ~8 GB peak before the cache quantization).

## Current choices

| Knob | Value | File | Why |
| --- | --- | --- | --- |
| Model | `mlx-community/gemma-4-e2b-it-4bit` | `Gemma4Provider.swift` | E2B ≈ 2 B active params, 4-bit weights ≈ 2.6 GB RAM. Smallest Gemma-4 instruction-tuned variant, best fit for an iPhone. |
| KV cache quantization | **disabled** (`kvBits: nil`) | `Gemma4Provider.swift` | Target was 4-bit for a ~4× cache memory win, but Swift-Gemma4-Core 0.1.0's attention layer calls the non-quantized `cache.update(keys:values:)` path which `QuantizedKVCache` rejects at runtime (KVCache.swift:894 assertion). Re-enable as soon as Gemma4SwiftCore teaches its attention layer about `updateQuantized` — see watch-list. |
| `MLX.GPU.cacheLimit` | 512 MB | `Gemma4Provider.init` | MLX's Metal buffer pool defaulted to *system memory limit* (~128 GB on M1 Ultra). Uncapped, each `generate()` stacked another ~2.5 GB of pooled tensors. Capping is the #1 reason we stopped seeing runaway growth. |
| `maxTokens` | 256 | `ChatSession.runGenerationLoop` | MLX pre-reserves KV space proportional to this. 256 is enough for all replies we've produced in practice (a 90-road Kaunas→Vilnius answer was ~350 tokens visible, well under the cap). |
| `temperature` / `topP` | 0.3 / 0.9 | `ChatSession.runGenerationLoop` | E2B at the default 0.7 hedged ("Would you like me to…?") instead of calling tools. Lower temperature = more decisive tool use. |
| Tool result trimming | polyline summarized, `turn_by_turn` ≤ 20, article text ≤ 4 KB | `ChatSession.trimForModel` | Untrimmed routing result was 55 KB → 30 868 tokens → 2 GB KV-cache spike. Trimmed is 3.4 KB / 2 490 tokens. 12× context reduction, 6× speed-up on the follow-up turn. |
| SZRG graph parse | zero-copy via `withUnsafeBytes`, `decodeGeoms: false` | `SZRGGraph.swift`, `ZimService.loadGraph` | Old parser used `Data.subdata(in:)` per polyline — millions of copies inflated a 226 MB graph to ~720 MB Swift struct heap. Zero-copy + skip-geoms drops the ZimService Δmem by ~600 MB and cuts parse time from 20 s → 5 s. A* only needs node positions + edge distances; precise polylines were only ever used for UI rendering (which doesn't exist yet). |
| `MLX.Memory.memoryLimit` | not set (uses MLX default) | — | Considering setting this as a hard jetsam-avoidance ceiling on iOS. Held off until we have device numbers. |
| Auto-load model on launch | enabled | `ChatSession.init` | Model is useless unsolved. `Gemma4Provider.load()` is idempotent so the one-shot init-time Task cannot double-load. |

## Known remaining costs

These are the next bars to push if we need more headroom:

- **Graph parse still allocates ~240 MB** of Swift arrays (nodes/edges/names). Further shrink options: pack lat/lon into `Int32` pairs and compute Doubles on demand; store names as a single contiguous byte buffer + offsets (avoid per-name `String` allocation). Estimated saving: 80–120 MB.
- **Scan-on-launch opens every ZIM in Documents**. For a user with lots of large ZIMs this multiplies the baseline. Next step: only "open" = memory-map metadata; defer full reader init until a tool actually needs it.
- **KV cache for routing follow-up turns**: at 4-bit it's ~500 MB at 2 500-token prompt. Further reduction options:
  - `maxKVSize` (rotating window) — OK for short chats, loses context on long ones.
  - Prompt caching: reuse the system turn + tool declarations across iters so only the delta gets re-encoded.
- **ZIM cluster decompression**: libzim keeps decompressed ZSTD clusters cached per archive. A single `get_article` call on a big ZIM can pin 50 MB+. Need a cluster-cache cap on the `ZimArchive` side.

## Watch-list — upstream features to adopt as they land

1. **MLX "dcache"** — dynamic / dense cache variant under discussion in mlx /
   mlx-lm. Would replace the current simple/rotating cache with automatic
   sparsification on long contexts. Track
   [ml-explore/mlx](https://github.com/ml-explore/mlx) issues tagged `kv
   cache`; fold in behind a feature flag once it ships in `mlx-swift-lm`.
2. **Lightning quants** — Apple's faster low-bit matmul kernels. As new
   quant schemes hit `mlx-community` (look for `*-lightning-*` or `*-q4_K_M`-style
   tags in model IDs), swap `verifiedModelId` in `Gemma4Provider.swift` and
   measure tokens/sec + footprint.
3. **Gemma-4 chat template revisions** — Google has already iterated on
   `chat_template.jinja` once (April 13); expect more. If `mlx-community`
   re-uploads `gemma-4-e2b-it-4bit` with a new `chat_template.jinja`, compare
   the byte-emission against our `Gemma4PromptTemplate.render(...)` and
   re-match. Our tests already assert byte-equality with
   `Gemma4PromptFormatter.userTurn(...)` from Swift-Gemma4-Core — keep that
   assertion live.
4. **Swift-Gemma4-Core > 0.1.0** — v0.1.0 was shipped on April 8, which
   predates Google's template update. When a newer tag appears, check:
   - Whether it exposes a multi-turn `Gemma4PromptFormatter.conversation(...)`
     that handles tool-declaration turns properly — if so, drop our
     hand-rolled template and delegate.
   - **Whether its `Gemma4TextAttention` now routes through
     `cache.updateQuantized(keys:values:)` when the cache is a
     `QuantizedKVCache`.** Today it calls plain `cache.update(keys:values:)`
     which the quantized cache rejects (runtime trap in KVCache.swift:894).
     Once fixed, re-enable `kvBits: 4, kvGroupSize: 64` in
     `Gemma4Provider.generate(...)` — gives a ~4× cut to the `stream opened`
     memory spike.
5. **mlx-swift-lm 2.32+** — track for: (a) cache-reset APIs that don't
   require re-loading the container; (b) native tool-call event parsing
   (the `.toolCall` case we currently ignore is a placeholder waiting for
   real wiring); (c) prompt-caching / prefix reuse across turns.
6. **CoreKiwix.xcframework > 14.2.0** — libzim has per-release perf
   improvements (cluster-cache tuning is on their roadmap). Refresh the
   vendored framework periodically.
7. **Kiwix-shared ZIMs via Files** — see README; need security-scoped
   bookmarks so the user can point at Downloads/ZIMs owned by Kiwix instead
   of duplicating into the app's Documents dir. Implementation pending.

## How to measure after any change

Send a routing query ("directions from Kaunas to Vilnius") and compare:

```
[ZimSvc] parse graph · <s>s · Δmem=+<MB> MB
[Tool]   tool route_from_places returned <N> bytes in <s>s · Δmem=+<MB> MB (trimmed for model: <N2> bytes)
[Gemma4] encoded <T> tokens in <s>s
[Gemma4] stream opened in <s>s — awaiting first token… · mem=<MB>  ← the big one
[Gemma4] generate() finished — <chunks> chunks, <s>s total · mem=<MB>
```

Key numbers to track run-over-run:
- `parse graph` Δmem — should stay ≤ 400 MB
- `stream opened` mem delta vs. the preceding line — should stay ≤ 1.5 GB
- `generate() finished` mem — should return to near the pre-stream value
