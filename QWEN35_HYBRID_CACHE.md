# Qwen 3.5 hybrid-cache cannot reuse across turns — why every follow-up pays ~13 s prefill

## Symptom

On-device Qwen 3.5 4B turns run **full prefill every time**. With the
app's ~7,000-token system preamble that costs ~13 seconds of GPU prefill
before the first token streams — even when the conversation hasn't
changed and the LCP between the cached prompt and the new prompt covers
`cached.count`. The in-app log looks like:

```
cache miss: full prefill 7001 tokens (LCP=6983, cached.count=6983, prompt.count=7001)
```

`LCP == cached.count == 6983` — that's a textbook cache hit — yet the
code takes the full-prefill branch anyway.

## Why we force the full prefill

`ios/MCPZimChat/Providers/Gemma4Provider.swift` has this guard in
`generate()` around line 493:

```swift
let cacheIsHybrid = existing?.contains(where: { $0 is MambaCache }) ?? false
if common == cached.count, common > 0, let existing, !existing.isEmpty,
   !cacheIsHybrid
{
    // reuse path — fast
} else {
    // full-prefill path — slow
    kvCache = context.model.newCache(parameters: genParams)
    inputTokens = tokens32
    hit = false
}
```

Qwen 3.5 uses **hybrid attention**: `MambaCache` on half the layers
(state-space model layers) and `KVCacheSimple` on the other half
(attention layers). When we take the reuse path and feed the new tail
tokens into this hybrid cache, MLX trips on a shape mismatch:

```
broadcast_shapes (…128,256) vs (…129,256)
```

…and aborts the process (uncaught C++ exception → `std::terminate`).
The defensive guard disables reuse whenever a `MambaCache` is present,
paying the full 7k-token prefill to avoid the crash.

## Root cause (per upstream)

[mlx-swift-lm#157](https://github.com/ml-explore/mlx-swift-lm/issues/157)
— `davidkoski` identified this isn't a `KVCache` bug. The hybrid cache
itself is fine. The problem is **illegal per-evaluation mutable state
stored on the model class**:

- `Qwen35` stores `precomputedPositionIds` and `ropeDeltas` as `var` fields.
- `Qwen3VL` / `GlmOcr` do the same.
- `GPTOSS` caches `sinksActive` similarly.

Those fields are meant to be scratch for the current forward pass, but
on a second call (partial-prefix reuse) they leak across, producing the
128/129 off-by-one in the attention shape.

The fix upstream would move those fields into `LMOutput.State` and
switch the affected models to the state-aware
`callAsFunction(_ input:cache:state:)` overload. `alankessler` offered
(2026-03-27) to implement it; **no PR yet**.

## What we've checked

| Option | Status |
|--------|--------|
| Bump `mlx-swift-lm` past 3.31.3 | We're already on HEAD. `main` is 2 docs-only commits ahead. No fix landed. |
| Bump `mlx-swift` (MLX core) | Also at HEAD (0.31.3). The bug is not in mlx-swift. |
| Narrower Swift-side guard (e.g. only block when appending > N tokens) | Wouldn't help. Stale position IDs trip on any second call regardless of append size. |
| Manually patch `Qwen35.swift` in our vendored copy | Possible, but high-risk: requires matching the `LMOutput.State` flow and correctly resetting scratch fields without breaking first-call generation. Defer until the upstream PR lands so we can cherry-pick. |
| PR #155 (hybrid-cache serialization round-trip) | Already in 3.31.3 — fixes prompt-cache save/load for hybrid; doesn't address the runtime reuse bug. |

## Performance mitigation that is available

Upstream [mlx-swift-lm#225](https://github.com/ml-explore/mlx-swift-lm/issues/225)
("pipeline prefill chunks with asyncEval — 10x on GDN models") proposes
async chunked prefill specifically for GDN / hybrid-cache models. That
wouldn't fix cache reuse, but it could shrink the forced full-prefill
from ~13 s to ~3-4 s by overlapping compute across chunks. Out of scope
for now; noted here so future work can trace it.

## Practical guidance

- **Fast turns matter more than the 9/9 eval point** → ship Qwen 3 4B
  (non-hybrid, same 9/9 score on our matrix) as the default. Qwen 3.5
  stays in the picker for users who want its slightly better response
  wording on edge cases and don't mind the latency.
- **Watch for an mlx-swift-lm release tagged ≥ 3.32** — the refactor in
  #157 will land there.
- **Do not remove the `cacheIsHybrid` guard.** It's the only thing
  between us and a SIGABRT mid-generation.

## Related code

- `ios/MCPZimChat/Providers/Gemma4Provider.swift::generate` — the
  guard.
- `ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
  — where the refactor would land locally (if we choose to
  pre-patch ahead of upstream).
- `ios/LocalPackages/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift`
  — our other local patch, for the `QuantizedKVCacheProtocol` branch;
  documented in `HOW_TO_BUILD.md`.
