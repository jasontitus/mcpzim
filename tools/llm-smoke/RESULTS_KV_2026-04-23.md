# Gemma 3 KV-cache variants — bench results 2026-04-23

Testing whether we can save memory / speed up decode by overriding Gemma 3
4B IT's default KV cache layout (`StandardKVCache` on 1/6 global layers +
`RotatingKVCache(512)` on 5/6 sliding-window layers).

Harness: `tools/llm-smoke/bench_kv.py`.

## Variants tested

| name | implementation | intent |
|---|---|---|
| `default` | Gemma 3's built-in mix (StandardKV + Rotating-512) | baseline |
| `bounded_512` | `make_prompt_cache(model, max_kv_size=512)` — forces ALL layers to Rotating-512 | tighter cap on the 1/6 global layers |
| `q8_from_64` | `kv_bits=8, kv_group_size=64, quantized_kv_start=64` | int8 K/V |
| `q4_from_64` | `kv_bits=4, kv_group_size=64, quantized_kv_start=64` | int4 K/V |

## Headline results — 9-scenario eval, ~1.8k-token preamble

| variant | name-correct | full-pass | prefill tok/s | decode tok/s | peak MB |
|---|---|---|---|---|---|
| default | 7/9 | 7/9 | 345 | 54 | 3260 |
| **bounded_512** | **7/9** | **7/9** | **398** | **72** | **3260** |
| q8_from_64 | 0/9 | 0/9 | — `NYI` | — | — |
| q4_from_64 | 0/9 | 0/9 | — `NYI` | — | — |

### QuantizedKVCache is NYI against RotatingKVCache

```
RotatingKVCache Quantization NYI
```

Gemma 3 uses `RotatingKVCache` on 5 of 6 layers, so neither `kv_bits=8`
nor `kv_bits=4` works for it in Python mlx-lm today. Swift-side
`mlx-swift-lm` has the same class hierarchy; I'd expect the same block
until someone upstream implements `RotatingKVCache.maybeQuantize`. Worth
re-probing after each `mlx-swift-lm` bump; if it lands we'd get ~4x
compression on the global-layer fraction for free.

### bounded_512 is a free win at normal prompt sizes

Forcing the 1/6 global layers into a 512-token rotating window doesn't
lose accuracy on any of our 9 scenarios, and decode gets ~33% faster
(72 vs 54 tok/s). Prefill gains ~15%. Peak RSS unchanged at 1.8k-token
preamble size — cache memory is dwarfed by prefill scratch here.

## Long-preamble (7k-token) test

Repeated the bench with a `PREAMBLE * 5` preamble (~7k tokens) to try to
push the cache into the regime where the global-layer cap matters.

Result: **both variants tanked to 1/9**. The padding (same 4KB block five
times) confused Gemma 3 — repeated content is a bad long-preamble
substitute. Peak memory was 3846 MB for both — still dominated by
prefill scratch, not final cache size.

Conclusion: the long-preamble test as written isn't discriminating. To
make it useful we'd need a realistic diverse 7k-token preamble (e.g.
export the real `ChatSession.systemMessage` from the iOS app and feed it
here). Not blocking a recommendation.

## Recommendation

1. **Keep `bounded_512` on the shortlist.** If we wire Gemma 3 into
   `Gemma4Provider` (or a sibling `Gemma3Provider`), expose a provider
   flag that swaps the model's `newCache` to `RotatingKVCache(512,
   keep: 0)` on every layer instead of the global/sliding mix. Free
   +33% decode.

2. **Don't patch Gemma3Text for QuantizedKVCache yet** — even with the
   `cache as? QuantizedKVCacheProtocol` branch added, `RotatingKVCache`
   itself doesn't support quantisation in current mlx-swift-lm, so the
   change would hit the same NYI on the 5/6 sliding layers. Wait for
   upstream to land quantised rotating caches.

3. **Before we ship Gemma 3**, rerun this with a realistic 7k-token
   system preamble (not the PREAMBLE×5 padding). That's the regime
   where `bounded_512` might either clearly win on memory OR start
   losing tool accuracy because the global layers no longer retain
   enough of the tool-schema section. Both outcomes are informative.

4. **Memory headroom check**: `default` on Gemma 3 4B IT at 1.8k
   preamble peaks at 3.26 GB. Our app preamble is ~7k tokens which will
   add roughly 500 MB on top (1/6 global layers scale with prompt
   length; 5/6 cap at 512 regardless). So Gemma 3 4B at 7k-preamble on
   iPhone 17 Pro Max ≈ 3.8 GB peak, leaving ~2 GB under the 6 GB
   jetsam cap for WebView + Metal — tight but workable, similar to
   Qwen 3 4B today.

## Related tasks

- `task #43` — patching Gemma3Text.swift for `QuantizedKVCacheProtocol`
  is still pending but now lower-priority given the rotating-layer NYI.
- `task #40` — MCPZimKit template adapter for Gemma 3 is the other
  blocker before any on-device swap.
