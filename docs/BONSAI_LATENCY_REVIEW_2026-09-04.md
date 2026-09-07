# Bonsai latency: recent MTPLX and llama.cpp work

Follow-up: [physical iPhone tests and implementation, September 5](BONSAI_PHONE_LATENCY_2026-09-05.md).

Reviewed September 4, 2026 (Pacific). This is a source and artifact audit,
not a new inference benchmark. Keep Bonsai Q1 as the phone baseline. The
highest-priority experiments are preserving warm state across more turn
shapes and comparing recent Metal changes against the pinned runtime.

## What the app actually runs

- The binary package pins Prism `62061f9` / `prism-b9591`; it calls the C API
  directly, not `llama-server`.
- `LlamaCppProvider` offloads all layers, enables flash attention, and uses
  Q4_0 K/V for phone Bonsai. Batch and microbatch sizes are both 512.
- The allocated context is 16,384 tokens; grounded discussion intentionally
  rebuilds its evidence window above 6,144 prompt tokens.
- Exact prompt appends reuse cached state. If a prompt diverges, the provider
  tries partial removal; if the hybrid state cannot rewind, it clears the
  sequence and prefills the entire prompt.
- Grounded discussion retains raw answer text and appends deduplicated evidence,
  usually at most one new section window. Generic tool turns also retain raw
  tool emissions. Static-prefix preparation exists and grounded turns can
  preempt it. Final-token-only logits and throttled UI updates already exist.

I read the header and tensor directory of the installed
`Bonsai-27B-Q1_0.gguf` directly: 3,803,452,480 bytes, 851 tensors,
`general.architecture=qwen35`, 64 blocks, embedding width 5,120,
24 query heads / 4 KV heads, and recurrent state size 128. There are no
`nextn`, `mtp`, `dspark`, or `blk.64` tensor names. This artifact does not
contain the native prediction head needed to turn on MTPLX's MTP algorithm.
The “65/65 layers offloaded” runtime message is not evidence of an MTP block.

Relevant local sources: `ios/MCPZimChat/Providers/LlamaCppProvider.swift`,
`ios/MCPZimChat/Chat/ChatSession.swift`, and
`ios/LocalPackages/llama.cpp-swift/Package.swift`.

## Existing phone measurements establish the opportunity

The final July 14 iPhone 17 Pro Max capture in
[the evaluation report](BONSAI_27B_IPHONE_EVALUATION.md) records:

| Turn | Prompt / reused tokens | First token | Total generation |
| --- | ---: | ---: | ---: |
| Putin overview | 696 / 0 | 8.390 s | 20.177 s |
| Parents | 857 / 816 | 0.770 s | 6.562 s |
| School | 1,129 / 911 | 2.941 s | 7.664 s |

These are historical observations, not current performance promises. They show
why the number of newly evaluated tokens matters: an ordinary warm turn can
already start below a second, while new evidence and full resets cost seconds.
The report also notes that Metal work can finish after `llama_decode` returns;
prefill-call time alone understates the wait for usable logits.

## What transfers from MTPLX

**Prompt fidelity and scheduling transfer well.** MTPLX 2.10 reports reducing
a mid-session first-token wait from 1.8–2.2 seconds to 0.11 seconds by preserving
reasoning history and reducing new prefill to about 20 tokens. This is the same
class of optimization as our exact raw transcript retention, including hidden
ChatML markers. It does not require new weights.
[MTPLX 2.10 release](https://github.com/youssofal/MTPLX/releases/tag/v2.10.0)

MTPLX 2.11 addresses tool-call reserialization, trailing request instructions,
unnecessary SSD restores, and background work delaying warm requests. It also
retains valid boundary snapshots and protects the active session. These are
useful design precedents for our transitions between tools and grounded answers.
Its GPU-residency keepalive is a separate macOS result on much larger models;
measure pause-dependent latency and phone energy before adopting it.
[MTPLX 2.11 release](https://github.com/youssofal/MTPLX/releases/tag/v2.11.1)

**Its kernels require selective porting.** Compiled MLX verify graphs are not
usable directly by a GGML C API provider. Flash-Next's sparse attention, MoE,
and n-gram embedding optimizations target operators absent from this Bonsai
artifact. Its low-bit matrix kernels also need the exact weight layout; a
kernel optimized for Q4/Q8 is not automatically an efficient Q1_g128 kernel.

**Speculation is a later experiment.** The installed GGUF lacks native MTP
weights. Prism supplies a separate target-trained DSpark drafter, but documents
its speedup on CUDA and says Apple Silicon verification does not yet amortize
the draft cost. Its Q4_1 file is 1.79 GB; shared embeddings/head mean file size
is not equal to incremental live memory. Extra state and buffers must also be
measured. Do not assume a generic Qwen draft head has good acceptance against
the binary target.
[Bonsai model card](https://huggingface.co/prism-ml/Bonsai-27B-gguf#speculative-decoding-dspark)

Prompt-copy speculation needs no second model and could be tested on answers
that quote retrieved evidence. The benefit for short, newly phrased Wikipedia
answers is unknown. Hybrid rollback and exact sampling verification are required
before that can safely become a shipping path.

## Recent llama.cpp changes worth a controlled comparison

Checked public Metal commit history from August 10 through September 4.

| Change | Date merged | Relevance to this app |
| --- | --- | --- |
| [Quantized KV → F16 attention scratch, #27390](https://github.com/ggml-org/llama.cpp/pull/27390), with [large-batch gate #27438](https://github.com/ggml-org/llama.cpp/pull/27438) | Aug 20 | Q4 KV is our actual configuration. Compare warm suffix batches as well as cold prefill; include the follow-up gate and measure scratch memory. Published results vary by model and context. |
| [Per-device flash-attention vector tuning, #26570](https://github.com/ggml-org/llama.cpp/pull/26570) | Aug 24 | Tune the A19 Pro for Bonsai's head geometry and Q4 KV; results from M-series or A18 tables are not phone measurements. Only 16 of Bonsai's 64 blocks use full attention. |
| [Metal 4 language-version fix, #27461](https://github.com/ggml-org/llama.cpp/pull/27461) | Sep 1 | Fixes tensor probes silently disabling acceleration on M5/A19 and mismatched external metallib dispatch. Our July phone capture already reports active tensor support, so this is an engagement/correctness check, not a demonstrated new speedup. |
| [Autorelease-pool fixes, #27883](https://github.com/ggml-org/llama.cpp/pull/27883) | Sep 1 | Include in sustained-session memory tests; lower resource leakage can preserve headroom, but no Bonsai throughput claim is established. |
| [XCFramework metallib support, #28163](https://github.com/ggml-org/llama.cpp/pull/28163) | Sep 1 | Potential startup improvement. Verify the compiled library includes the intended tensor kernels; startup compilation is separate from warm-turn prefill. |

Two similarly named improvements are not direct Bonsai optimizations:
[Mamba-2 chunked SSD MMA, #26647](https://github.com/ggml-org/llama.cpp/pull/26647)
and [SSM scan rollback, #26623](https://github.com/ggml-org/llama.cpp/pull/26623)
target Mamba/SSM-scan paths. Bonsai uses Qwen's gated-delta architecture;
“recurrent” alone does not establish operator compatibility. Here SSD means
state-space duality, not a disk cache.

Upgrading a framework also does not install `llama-server`'s checkpoint policy
into our Swift provider. That policy needs explicit integration. Keep the Mac
ternary model out of an unqualified binary swap: its existing group-128 Q2
GGUF differs from mainline's group-64 Q2 format.

## Recommended implementation order

1. **Record a current multi-turn baseline with cause-level timing.** Separate
   retrieval, model-lock wait, tokenization, reused/new tokens, final-logit wait,
   visible text, and first audio. Add cache-miss reasons: topic/source switch,
   raw answer scrub, retry, cancelled prefill, and rolling-window rebuild.
   Existing provider logs expose much of this, but the end-to-end timeline and
   explicit reset attribution need joining.
2. **Prototype one bounded in-memory recovery checkpoint.** Capture at a valid
   prompt boundary before generation, then restore only if that exact token
   prefix matches and the required attention KV still exists. This may recover
   from altered or cancelled replies without recomputing the evidence prefix.
   It will not rescue a new topic whose system prompt differs from the start.
   A shared immutable grounding preamble is a separate experiment for that case.
3. **A/B the recent runtime in isolation with the same Q1 file.** Compare against
   `62061f9` on the phone before changing the package pin. Use 64/128/256/512
   fresh-token suffixes at 1K/3K/6K histories, plus cold prompts. Vary microbatch
   size only as a separate experiment. Check answers as well as timings.
4. **Investigate remaining cold latency.** Profile passage preparation and the
   common grounding prefix. Move per-turn instructions to the suffix where
   semantically equivalent; keep quality tests for attribution and history.
   Prewarming must be cancellable and must not replace an active conversation's
   useful state with a generic tool prefix.
5. **Try speculation only after the above.** Require a measured improvement in
   total answer time, acceptable peak memory, unchanged target distribution,
   and no sustained thermal regression.

The bundled C header already exposes `llama_state_seq_*_ext` with
`PARTIAL_ONLY` and `ON_DEVICE`. The latter's contract invalidates previous
on-device snapshots for the same sequence when another is captured: it is
not a free multi-entry session bank. Recurrent-only restores also depend on
the matching attention state remaining valid. Benchmark save/restore and
correctness before choosing a storage representation.

Avoid per-turn disk snapshots as the first solution. Our historical phone
experiment saved a 184.5 MB state in 7.686 seconds, even though immediate restore
took 0.047 seconds. An in-memory/device checkpoint has a different cost profile
that has not yet been measured here.

For acceptance, interleave baseline/candidate runs, hold weights, sampler,
archive, prompt, output length, and thermal conditions constant, and include
10–20 turns plus a speech session. Report median and tail latency, cold versus
warm, peak memory including mapped weights, and interruption recovery. The
existing sub-gigabyte process-footprint log must not be read as the entire
memory cost of a 3.8 GB mapped model.

No inference behavior or runtime dependency changed in this audit.
