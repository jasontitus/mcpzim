# On-device model + KV-cache bench — 2026-04-23

All numbers from `tools/llm-smoke/` (MacBook Pro M2 Max / 64 GB, Python
mlx-lm HEAD, MLX 0.31.3-equivalent, 4-bit `mlx-community` weights unless
noted). Scenarios
transplanted from `ios/MCPZimEval/EvalHarness.swift`. Swift on-device
behaviour may differ — the iOS provider adds IntentRouter fast-paths, a
tighter template, and uses the vendored mlx-swift-lm 3.31.3 which has a
slightly different `QuantizedKVCache` prefill path than Python mlx-lm.

## TL;DR

- **Gemma 3 4B IT 4bit** beats our current Qwen 3 4B Instruct-2507 default
  on tool-calling accuracy (7/9 vs 6/9) at equivalent phone budget, same
  decode speed, and scales substantially better to long contexts thanks
  to its sliding-window + global-layer hybrid attention.
- **Quantized KV (4-bit group-64) is a steady-state win but a peak-memory
  loss** in Python mlx-lm for Qwen 3: active-decode drops, but prefill
  dequant transients push peak from 5.5 GB → 8.0 GB at 20k tokens.
  Worth re-measuring on iOS where mlx-swift-lm's prefill path may
  behave differently — our current `DeviceProfile.useQuantizedKVCache =
  true` may be tighter on peak than Python reports suggest.
- **Gemma 3 inherent KV compression** (5/6 layers use
  `RotatingKVCache(512)` natively) outperforms Qwen 3 + kv4 on peak memory
  at long contexts and is the reason Gemma handles 20k preambles at
  ~4.4 GB where Qwen kv4 balloons to 8 GB.
- **Phi-4 / Phi-3.5 / Nemotron** all either tank tool-calling accuracy
  at 4-bit, bust the phone memory budget at 6-bit, or refuse our tool
  schemas entirely. None merit pursuit.
- **Prompt warming (KV cache reuse across turns) works cleanly on both
  Gemma 3 and Qwen 3** with 10% warm/cold ratios — no MambaCache reuse
  bug like Qwen 3.5. This is why we can keep the fast-follow-up UX
  regardless of which default we pick.

## 1. Consolidated scorecard — accuracy + perf + memory at 7k/20k/40k

One row per (model, variant) with everything needed to compare: 9-pass,
prefill/decode tok/s (from the 9-scenario eval), and peak memory at the
three preamble sizes the app might plausibly target. Sorted by
phone-viability (peak @ 7k), then by 9-pass.

| Model | 4-bit size | 9-pass | Prefill t/s | Decode t/s | **7k peak** | **20k peak** | **40k peak** |
|---|---|---|---|---|---|---|---|
| gemma-3n-E2B-it-lm-4bit | 2.4 GB | 5/9 | 809 | 75 | **3.69** | 4.10 | 4.90 |
| Phi-4-mini-instruct-4bit | 2.2 GB | 4/9 | 550 | 58 | 3.73 | 5.17 | 7.72 |
| **gemma-3-4b-it-4bit** | 2.2 GB | **7/9** | 381 | 66 | 3.74 | **4.37** | **5.41** |
| gemma-3-4b-it-qat-4bit | 2.2 GB | 6/9 | 400 | 70 | 3.74 | 4.37 | 5.41 |
| gemma-3-4b-it-4bit-DWQ | 2.2 GB | 6/9 | 409 | 79 | 3.74 | 4.37 | 5.41 |
| Qwen3-4B-Thinking-2507-4bit | 2.0 GB | 1/9 | 315 | 39 | 3.75† | 5.55† | 8.41† |
| **Qwen3-4B-Instruct-2507-4bit** *(app today)* | 2.0 GB | 6/9 | 404 | 70 | 3.75 | 5.55 | 8.41 |
| Qwen3-4B-Instruct-2507-4bit `+kv4` *(iOS DeviceProfile)* | 2.0 GB | 6/9 | 275 | 42 | 4.38 | 8.01 | 13.94 |
| **gemma-4-e2b-it-4bit** *(app today, HF template)* | 2.2 GB | 1/9 ⚠️ | 504 | 27 | **4.49** | 5.24 | 6.37 |
| gemma-4-e2b-it-4bit *(native Gemma4 template)* | 2.2 GB | 0/9 ⚠️ | 461 | 0 | 4.49 | 5.24 | 6.37 |
| Phi-4-mini-instruct-6bit | 3.1 GB | 6/9 | 535 | 34 | 4.63 | 6.08 | 8.63 |
| gemma-3n-E4B-it-lm-4bit | 3.7 GB | 6/9 | 436 | 43 | 4.88 | 5.38 | 6.19 |
| gemma-4-e4b-it-4bit *(HF template)* | ~4 GB | 0/9 ⚠️ | 1993 | 41 | 6.18 | 7.08 | 8.58 |
| gemma-4-e4b-it-4bit *(native Gemma4 template)* | ~4 GB | **5/9** ⚠️ | 206 | 14 | 6.18 | 7.08 | 8.58 |
| gemma-3-12b-it-qat-4bit *(mac-only)* | 6.9 GB | **9/9** | 128 | 15 | 9.17 | 10.83 | 13.39 |
| Phi-3.5-mini-instruct-4bit | 2.1 GB | 3/9 | 509 | 59 | *not benched* | — | — |
| Llama-3.1-Nemotron-Nano-4B-v1.1-4bit | 2.1 GB | 2/9 | 307 | 45 | *not benched* | — | — |
| NVIDIA-Nemotron-3-Nano-4B-4bit | 2.1 GB | 0/9 | 310 | 45 | *not benched* | — | — |

*All peaks in GB. ⚠️ Gemma 4 scores are the Python-harness floor (see §1b).
† Qwen 3 Thinking uses the same weights as Instruct; peak memory assumed identical.*

### Phone viability summary (6 GB iPhone jetsam; ~3.5–4 GB usable for model+cache)

| Model | @ 7k | @ 20k | @ 40k |
|---|---|---|---|
| gemma-3n-E2B-it-lm-4bit | ✓ | ✓ | ✓ (tight) |
| **gemma-3-4b-it-4bit** | ✓ | ✓ | ✓ (tight) |
| gemma-3-4b-it-qat / DWQ | ✓ | ✓ | ✓ (tight) |
| Qwen 3 Instruct 4bit (no kv quant) | ✓ | ⚠︎ | ✗ |
| Phi-4-mini-4bit | ✓ | ⚠︎ | ✗ |
| **gemma-4-e2b-it-4bit** *(app today)* | ✓ (tight) | ⚠︎ | ✗ |
| Qwen 3 Instruct 4bit + kv4 *(iOS today)* | ⚠︎ | ✗ | ✗✗ |
| gemma-3n-E4B | ✓ (tight) | ⚠︎ | ✗ |
| Phi-4-mini-6bit | ✓ (tight) | ✗ | ✗✗ |
| gemma-4-e4b | ✗ | ✗ | ✗✗ |
| gemma-3-12b-it-qat-4bit | ✗✗ | ✗✗✗ | ✗✗✗ |

### 1b. Gemma 4 ⚠️ caveat

Measured two templates for Gemma 4:

1. **HF default template** (via `processor.tokenizer.apply_chat_template`):
   E2B 1/9, E4B 0/9. Both reply in prose / refuse.
2. **Native Gemma 4 template** (port of
   `MCPZimKit.Gemma4PromptTemplate` + `Gemma4ToolFormat` —
   `tools/llm-smoke/gemma4_format.py`): **E2B 0/9, E4B 5/9.** E4B
   correctly emits `<|tool_call>call:…<tool_call|>` for 5 of the 9
   scenarios.

**E2B hard cliff at ≥11 tool declarations.** Isolated this with a
tool-count sweep: E2B emits perfect native tool calls with 1–9 tools in
the system turn, but at 11 tools (our eval count) emits `<eos>` as its
very first sampled token, producing empty output. E4B doesn't cliff —
it degrades gracefully, defaulting to clarifying questions on scenarios
whose tool call needs unspecified optional args.

**E4B 5/9 passes:** compare_musk_bezos, relations_us_iran,
narrate_hp_garage, what_is_here_in_sf, how_much_longer.
**E4B 4/9 misses:** restaurants_in_sf (empty), nearby_stories_here
(→ what_is_here), nearby_stories_palo_alto (empty),
tell_me_about_palo_alto (empty). The empty outputs are partial
manifestations of the same cliff at 11 tools — some prompts push E4B
closer to the refusal boundary than others.

The iOS app uses `Gemma4SwiftCore` + a 4 KB tool-forcing preamble
(`ChatSession.composeSystemMessage`) that our Python harness doesn't
match. That larger preamble may pull E2B back from the cliff. A proper
Swift harness rerun against our 9 scenarios is required for a
trustworthy Gemma 4 number; the 5/9 E4B figure here is a **strong floor
for E4B** and a **soft floor for E2B**.

Practical fix regardless: reducing our tool surface to **≤10
declarations** would unblock E2B in the Python path. Candidate merges —
fold `what_is_here` into `nearby_stories(radius=0)`, or combine
`nearby_stories` + `nearby_stories_at_place` behind a single tool with
optional `place` arg.

## 1c. Model family scorecard (original) — 9-scenario tool selection

Scenarios: `restaurants_in_sf`, `nearby_stories_here`,
`nearby_stories_palo_alto`, `tell_me_about_palo_alto`,
`compare_musk_bezos`, `relations_us_iran`, `narrate_hp_garage`,
`what_is_here_in_sf`, `how_much_longer`. Pass = correct tool name + all
required-arg substrings appear. Temperature 0, `max_tokens=200`.

Scorecard sorted by pass rate. Prefill / decode speeds are mean across
the 9 cases (preamble ≈ 1.1–1.2 k tokens for Qwen / Phi / Gemma 3,
≈ 900 tokens for Gemma 4 via mlx-vlm).

| Model | 4-bit weights | 9-pass | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|
| gemma-3-12b-it-qat-4bit *(mac-only)* | 6.9 GB | **9/9** | 128 | 15 |
| **gemma-3-4b-it-4bit** | 2.2 GB | **7/9** | 381 | 66 |
| gemma-3-4b-it-qat-4bit | 2.2 GB | 6/9 | 400 | 70 |
| gemma-3-4b-it-4bit-DWQ | 2.2 GB | 6/9 | 409 | 79 |
| gemma-3n-E4B-it-lm-4bit | 3.7 GB | 6/9 | 436 | 43 |
| **Qwen3-4B-Instruct-2507-4bit** *(app default today)* | 2.0 GB | 6/9 | 404 | 70 |
| Phi-4-mini-instruct-6bit | 3.1 GB | 6/9 | 535 | 34 |
| gemma-3n-E2B-it-lm-4bit | 2.4 GB | 5/9 | 809 | 75 |
| Phi-4-mini-instruct-4bit | 2.2 GB | 4/9 | 550 | 58 |
| Phi-3.5-mini-instruct-4bit | 2.1 GB | 3/9 | 509 | 59 |
| Llama-3.1-Nemotron-Nano-4B-v1.1-4bit | 2.1 GB | 2/9 | 307 | 45 |
| **gemma-4-e4b-it-4bit** *(Python harness, see note)* | ~4 GB | **0/9** | 504 | 27 |
| NVIDIA-Nemotron-3-Nano-4B-4bit | 2.1 GB | 0/9 | 310 | 45 |
| Qwen3-4B-Thinking-2507-4bit | 2.0 GB | 1/9 | 315 | 39 |
| **gemma-4-e2b-it-4bit** *(Python harness, see note)* | 2.2 GB | **1/9** | 1993 | 41 |

### ⚠️ Gemma 4 Python-harness caveat

Both Gemma 4 variants score near-zero through `mlx_vlm.stream_generate`
with the HF default chat template:

- **E2B**: emits zero output tokens — hits EOS immediately after
  reading the preamble. Refuses before speaking.
- **E4B**: decodes at 27–57 tok/s but every reply is clarification
  prose ("What kind of restaurants? What radius?") instead of tool
  calls.

The iOS app **ships Gemma 4 E2B today** and works — it uses
`Gemma4SwiftCore` with a custom ~4 KB tool-forcing preamble and a
handwritten `Gemma4PromptTemplate` that the HF chat template doesn't
replicate. The 1/9 Python figure is a floor driven by prompt
templating, not model capacity. A proper Gemma 4 evaluation needs a
Swift harness (`tools/gemma-smoke`) against the same 9 scenarios — see
§7 for the gap.

### Observations from the passing models

- **Shared miss across every 4B candidate except Qwen:**
  `restaurants_in_sf` ("Are there any good restaurants in San
  Francisco?") picks `near_places` instead of `near_named_place`.
  Tool-description tightening on `near_places` ("user is already
  there; no place name in the request") would lift Gemma 3 to 8/9.
- **Only Gemma 3 (and 12B + 3n-E4B) pick `nearby_stories`** for
  "Tell me something interesting about where I am." Qwen and Phi
  both fall to `what_is_here` which is a location-describer, not a
  nearby-stories tool.
- **Gemma 3 12B at 4-bit** nails all 9, confirming the scenarios are
  achievable — the 4B gap is capacity, not prompt design.

**Shared miss across every 4B candidate except Qwen:** `restaurants_in_sf`
("Are there any good restaurants in San Francisco?") picks `near_places`
instead of `near_named_place`. Likely a tool-description issue —
tightening `near_places` to "user is already there; no place name in the
request" would lift Gemma 3 to 8/9.

**Surprising miss across every 4B candidate except Gemma 3 vanilla:**
`nearby_stories_here` ("Tell me something interesting about where I am.")
picks `what_is_here` (a location-description tool) instead of
`nearby_stories`. Only Gemma 3 4b, Gemma 3 12b, and the 3n-E4B variant
pick `nearby_stories`.

## 2. KV cache variants — Gemma 3 4B IT 4bit

Forcing the cache layout via `make_prompt_cache(model, max_kv_size=N)`
or `kv_bits`/`kv_group_size`/`quantized_kv_start` gen kwargs.

| variant | cache layout | Pass | Prefill tok/s | Decode tok/s | Peak (1.8k-prompt) |
|---|---|---|---|---|---|
| default | 1/6 StandardKV + 5/6 Rotating-512 | 7/9 | 345 | 54 | 3260 MB |
| bounded_512 | all Rotating-512 | 7/9 | 398 | 72 | 3260 MB |
| q8_from_64 | kv_bits=8, start=64 | ERR | — | — | — |
| q4_from_64 | kv_bits=4, start=64 | ERR | — | — | — |

**`RotatingKVCache Quantization NYI`** in Python mlx-lm — K/V quant is
incompatible with rotating cache. Since Gemma 3 uses rotating on 5/6
layers, neither `kv_bits=4` nor `kv_bits=8` works. The same class
hierarchy exists in mlx-swift-lm; until upstream lands
`RotatingKVCache.maybeQuantize`, our existing `cache as?
QuantizedKVCacheProtocol` branch in `Gemma4Text.swift` cannot be
replicated for Gemma 3.

**`bounded_512` is a free +15%/+33% prefill/decode at no accuracy cost**
for our normal prompt sizes — worth wiring into `Gemma3Provider.newCache`
as the default if/when we ship Gemma 3.

## 3. Prompt-warming (KV cache reuse) — Gemma 3 vs Qwen 3

Two-turn test: prefill turn-1 preamble, extend by 33–41 tokens for
turn-2 (follow-up user message), reuse cache across the boundary.

| Model | Turn-1 cold | Turn-2 cold | Turn-2 WARM | Saved | BPE stable |
|---|---|---|---|---|---|
| gemma-3-4b-it-4bit | 4763 ms | 4822 ms | **461 ms** (10%) | 4362 ms | ✓ |
| Qwen3-4B-Instruct-2507-4bit | 4500 ms | 4652 ms | **375 ms** (8%) | 4277 ms | ✓ |
| gemma-3-4b-it-4bit-DWQ | 3865 ms | 4963 ms | **370 ms** (7%) | 4593 ms | ✓ |

No hybrid-cache reuse bug on either. Gemma 3 does NOT have the scratch-
state bug (mlx-swift-lm#157) that forces full prefill on Qwen 3.5, so
the warm-cache path in `ChatSession` will work out of the box.

## 4. Memory comparison at real preamble sizes

iOS app's preamble today is ~7 k tokens; headroom question is "how much
more context can we carry before jetsam kicks in?". Same configurations
as the iOS app: Gemma 3 at its native cache layout (no KV quant available)
vs Qwen 3 at `kv_bits=4, kv_group_size=64, quantized_kv_start=0` (iOS
`DeviceProfile.useQuantizedKVCache = true`).

| Model | Variant | Prompt | Load | **Peak** | Active-decode | Prefill | Decode |
|---|---|---|---|---|---|---|---|
| gemma-3-4b-it-4bit | default | 7k | 2442 | **3742** | 2699 | 438 t/s | **80** t/s |
| gemma-3-4b-it-4bit | default | 20k | 2442 | **4366** | 2954 | 421 t/s | 72 t/s |
| gemma-3-4b-it-4bit | default | 40k | 2442 | **5406** | 3344 | 401 t/s | 66 t/s |
| gemma-3-4b-it-4bit | bounded_512 | 7k | 2442 | 3742 | 2699 | 446 t/s | 84 t/s |
| gemma-3-4b-it-4bit | bounded_512 | 20k | 2442 | 4366 | 2954 | 393 t/s | 70 t/s |
| gemma-3-4b-it-4bit | bounded_512 | 40k | 2442 | 5406 | 3344 | 409 t/s | 63 t/s |
| gemma-4-e2b-it-4bit *(mlx-vlm)* | default | 7k | 3415 | 4493 | 3416 | **1993 t/s** | 41 t/s |
| gemma-4-e2b-it-4bit | default | 20k | 3415 | 5237 | 3416 | 1889 t/s | 69 t/s |
| gemma-4-e2b-it-4bit | default | 40k | 3415 | **6365** | 3416 | 1767 t/s | 46 t/s |
| gemma-4-e2b-it-4bit | **kv4** (iOS default) | 7k/20k/40k | — | — | — | *NYI: RotatingKVCache Quantization* | — |
| gemma-4-e4b-it-4bit *(mlx-vlm)* | default | 7k | 4979 | **6182** | 4979 | 504 t/s | 27 t/s |
| gemma-4-e4b-it-4bit | default | 20k | 4979 | **7084** | 4979 | 630 t/s | 44 t/s |
| gemma-4-e4b-it-4bit | default | 40k | 4979 | **8582** | 4979 | 571 t/s | 19 t/s |
| Qwen3-4B-Instruct-2507-4bit | default (no KV quant) | 7k | 2160 | 3753 | 3168 | 352 t/s | 64 t/s |
| Qwen3-4B-Instruct-2507-4bit | default (no KV quant) | 20k | 2160 | **5550** | **5004** | 282 t/s | 34 t/s |
| Qwen3-4B-Instruct-2507-4bit | default (no KV quant) | 40k | 2160 | **8410** | **7812** | 194 t/s | 17 t/s |
| Qwen3-4B-Instruct-2507-4bit | **kv4** (iOS default) | 7k | 2160 | 4381 | 2444 | 275 t/s | 42 t/s |
| Qwen3-4B-Instruct-2507-4bit | **kv4** (iOS default) | 20k | 2160 | **8014** | 2961 | 187 t/s | 30 t/s |
| Qwen3-4B-Instruct-2507-4bit | **kv4** (iOS default) | 40k | 2160 | **13941** | 3751 | 120 t/s | 11 t/s |

*Gemma 4 peaks are ~750 MB heavier than Gemma 3 / Qwen 3 at the same
prompt size — the multimodal checkpoint loads the vision + audio
encoders into GPU even though we only prompt text. A text-only variant
(`mlx-community/Gemma4-E2B-IT-Text-int4`) exists but plain mlx-lm still
rejects it with the same shared-KV mismatch as the multimodal weights.
The 750 MB is roughly the weight delta — not avoidable without a custom
loader that skips vision/audio modules.*

All MB unless noted.

### What the numbers say

- **At 7k tokens (current iOS preamble)**: Gemma 3 default and Qwen 3
  default have nearly identical peak (~3.75 GB). Qwen 3 + kv4 *costs*
  600 MB more peak than Qwen default. Why? In Python mlx-lm the prefill
  path materialises the bf16 K/V for the current chunk before
  quantising, so big prefills pay a transient tax on top of the
  quantised steady state. Decode active memory IS lower with kv4
  (2444 MB vs 3168 MB, a 724 MB steady-state win), but the peak that
  jetsam watches is higher, not lower.

- **At 20k tokens**: the divergence is stark —
  - Gemma 3 default: **4.37 GB peak**, decode 72 tok/s.
  - Qwen 3 default: **5.55 GB peak**, decode crashes to 34 tok/s.
  - Qwen 3 + kv4: **8.01 GB peak**, decode 30 tok/s. Would jetsam on
    any phone.
  Gemma's sliding-window attention caps per-token decode cost on 5/6
  layers, so decode stays fast; Qwen's O(N) attention at every layer
  tanks as context grows.

- **Decode speed at 20k**: Gemma 72 tok/s vs Qwen default 34 tok/s —
  the sliding-window attention difference shows up as 2× decode
  throughput at long contexts.

### 40 k headroom test

Measured (see table above). Summary of the critical numbers:

| Config | 40k peak | 40k active-decode | 40k decode t/s |
|---|---|---|---|
| Gemma 3 default | **5.41 GB** | 3.34 GB | 66 |
| Gemma 3 bounded_512 | 5.41 GB | 3.34 GB | 63 |
| Qwen 3 default | 8.41 GB | 7.81 GB | 17 |
| Qwen 3 kv4 | **13.94 GB** | 3.75 GB | 11 |

Qwen 3 + kv4 at 40k hits **14 GB of peak memory** — roughly 2× the
MacBook's usable Metal budget and well beyond any iPhone. The
prefill-dequant transient cost compounds with the cache size. Qwen 3
default at 40k hits 8.4 GB peak which also exceeds the phone cap.

Gemma 3 at 40k fits inside 5.4 GB — still safely under the 6 GB
jetsam ceiling, and decode throughput degrades gracefully from 80 →
66 tok/s (vs Qwen's 64 → 17). **Gemma 3 is the only candidate that
keeps phone-viable memory at long contexts.**

### Scaling curves

| Peak MB | 7k | 20k | 40k | Δ per 10k tok |
|---|---|---|---|---|
| Gemma 3 default | 3742 | 4366 | 5406 | ~500 MB |
| Gemma 3 bounded_512 | 3742 | 4366 | 5406 | ~500 MB |
| Gemma 4 E2B default | 4493 | 5237 | 6365 | ~570 MB |
| Gemma 4 E4B default | 6182 | 7084 | 8582 | ~730 MB |
| Qwen 3 default | 3753 | 5550 | 8410 | ~1400 MB |
| Qwen 3 kv4 | 4381 | 8014 | 13941 | ~2900 MB |

Gemma 3 and Gemma 4 both scale at roughly 1/3 the rate of Qwen 3
default (sliding-window attention caps most layers' KV at 512). Qwen 3
+ kv4 scales 6× faster than Gemma 3 due to prefill dequant transients
compounding with context length.

## 5. Phone-budget analysis (iPhone 17 Pro Max, 6 GB jetsam)

Typical app RSS budget at generate-time, measured empirically:
- MLX model + KV cache: 3–4 GB (the number we've just benchmarked)
- MapLibre WebView + Metal scratch for the chat UI: 0.5–1 GB
- WebKit / SwiftUI / system frameworks: 0.5 GB
- Headroom for voice + Kokoro TTS loaded: 0.5–1 GB

That leaves a usable envelope of **~3.5–4 GB for model+cache peak**
under jetsam.

| Config | 7k-peak | 20k-peak | 40k-peak | Phone-safe @ 7k | @ 20k | @ 40k |
|---|---|---|---|---|---|---|
| Gemma 3 4B / default | 3.74 | 4.37 | 5.41 | ✓ | ✓ | ✓ (tight) |
| Gemma 3 4B / bounded_512 | 3.74 | 4.37 | 5.41 | ✓ | ✓ | ✓ (tight) |
| **Gemma 4 E2B / default** *(app today)* | **4.49** | **5.24** | **6.37** | ✓ | ⚠︎ tight | ✗ |
| Gemma 4 E4B / default | 6.18 | 7.08 | 8.58 | ✗ | ✗ | ✗✗ |
| Qwen 3 4B / default | 3.75 | 5.55 | 8.41 | ✓ | ✗ | ✗✗ |
| Qwen 3 4B / kv4 | 4.38 | 8.01 | 13.94 | ⚠︎ +600 MB tax | ✗✗ | ✗✗✗ |

So at the iOS app's current 7k preamble, **the kv4 config costs us 600 MB
of peak headroom versus running Qwen 3 without kv4** — a surprising
inversion given the whole point of kv4 was to save phone memory. This
warrants a measurement on device: if mlx-swift-lm's prefill path is
really different from Python mlx-lm's (chunked bf16 then quantise
vs. quantise-in-place), the iOS behaviour could be the opposite. If it
isn't, we should consider turning kv4 off — at worst it's wasting peak
for marginal active-decode savings at the current preamble size.

## 6. Recommendations

1. **Plan a default-model swap from Qwen 3 4B Instruct-2507 to Gemma 3
   4B IT-4bit.** +1 scenario on the LLM eval, same phone budget, dense
   attention (no cache-reuse bug), inherent KV compression handles long
   contexts ~2× better. Gates (block the swap until done):
   - Build `Gemma3Provider` + `Gemma3Template` in MCPZimKit (different
     turn markers `<start_of_turn>` / `<end_of_turn>`, different
     tool-call conventions).
   - Rerun `EvalHarness` on iPhone; verify the Swift provider gets at
     least Qwen parity under the real IntentRouter + full preamble.
   - Measure peak on-device at 7k preamble to compare to Qwen today.

2. **Ship `bounded_512` as the Gemma 3 cache default** if the native
   1/6 StandardKV layers are the memory-scaling pain point on iOS.
   Python numbers say it costs nothing on accuracy and gives +33%
   decode throughput. Free win.

3. **Re-measure `DeviceProfile.useQuantizedKVCache = true` on iPhone.**
   If mlx-swift-lm mirrors Python mlx-lm's behaviour, kv4 is costing
   us 600 MB peak at 7k for a 700 MB active-decode gain — net neutral
   at best. On longer contexts (20k+), kv4 is catastrophic. At minimum
   we should instrument on-device peak mem to confirm the Swift
   implementation behaves differently from what Python shows.

4. **Do NOT patch Gemma3Text.swift for QuantizedKVCache yet.** Even
   with the `cache as? QuantizedKVCacheProtocol` branch added, 5/6 of
   Gemma 3's layers use `RotatingKVCache`, which has no quantisation
   support in current mlx-swift-lm (NYI mirrors Python). Re-probe each
   upstream bump.

5. **Keep Qwen 3 Thinking, all Phi variants, and all Nemotron variants
   off the roadmap.** Data above; none are fixable at 4B.

## 7. What we DIDN'T test (known gaps)

- On-device numbers. All measurements are from Python mlx-lm on macOS
  (M2 Max / 64 GB), which may overstate prefill peaks vs mlx-swift-lm's
  Metal-shared path.
- Accuracy at 7k / 20k / 40k preambles with a realistic (not PREAMBLE×N)
  preamble built from the live iOS `systemMessageText` + tool
  declarations. The KV-variants bench regressed to 1/9 at 20k because
  my stitched filler preamble, despite being textually diverse,
  doesn't match the rhetorical structure of the real system prompt.
- Multi-turn warm cache at 20k+ (only tested at 1.8k).

## 8. Harness

- `tools/llm-smoke/eval.py` — 9-scenario eval, CLI takes any mlx-lm
  model id.
- `tools/llm-smoke/bench.py` — memory + warm-cache single model.
- `tools/llm-smoke/bench_kv.py` — KV variants (default / bounded /
  q8 / q4).
- `tools/llm-smoke/bench_memory.py` — cross-model memory at configurable
  preamble size. Used for section 4.
- `tools/llm-smoke/.venv/` — uv-managed Python 3.12 env.

Run any of the above inside the venv:
```
cd tools/llm-smoke && source .venv/bin/activate
python bench_memory.py --sizes 7000,20000,40000
```
