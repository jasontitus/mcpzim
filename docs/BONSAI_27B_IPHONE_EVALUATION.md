# Bonsai 27B evaluation and iPhone integration

Status: implemented, Mac-validated, installed, and load-validated on an
iPhone 17 Pro Max on 2026-07-14. The final signed build is installed and the
3,803,452,480-byte model is in its cache. The exact final artifact passed the
three-turn on-device conversation, cancellation, and liveness checks described
below.

## Decision

Ship `prism-ml/Bonsai-27B-gguf` / `Bonsai-27B-Q1_0.gguf` as the phone
operating point, using Prism ML's pinned llama.cpp fork, Metal offload, a
16,384-token allocated context, and Q4_0 K/V caches. Keep normal grounded
conversation on the 6,144-token rolling budget: 16K is overflow/safety
capacity, not the target prompt length.

Do not fine-tune Bonsai yet. The tests show that the 1-bit model retained the
content and follow-up quality needed for the target conversations. The
observed misses were orchestration defects: one routing error, one oversized
retrieval payload, and two evaluation-rubric false negatives. Those are now
fixed outside the model, where the behavior is deterministic and testable.

The 7.2 GB ternary build remains the quality reference for Mac evaluation. It
does not fit the phone's practical app-memory budget.

## Why GGUF for MCPZimChat

Prism's official phone demonstration uses MLX Swift, but MCPZimChat already
has an in-process llama.cpp provider. Prism reports a 5.2 GB peak at 4K for
the 1-bit GGUF, versus 5.9 GB for the 1-bit MLX pack. The roughly 700 MB of
headroom matters because the app also holds ZIM indexes, retrieved text, UI,
and runtime state.

This is an experimental iOS path and must stay behind physical-device
memory/liveness testing. The runtime is pinned to Prism commit `62061f9`
(`prism-b9591`), the exact build used for the measured phone results. Stock
llama.cpp now supports Bonsai Q1, but it has not yet beaten this pin in a
controlled app benchmark, and the app's Mac ternary GGUF still uses Prism's
fork-only group-128 Q2 format.

Official model/runtime references:

- [Bonsai 27B GGUF](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- [Bonsai 27B MLX 1-bit](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit)
- [Prism llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp)
- [Prism mlx-swift fork](https://github.com/PrismML-Eng/mlx-swift/tree/prism)

## Mac evidence

The expanded conversational suite contains 16 multi-turn scenarios. It
includes the requested chains verbatim in intent:

- Vladimir Putin -> parents -> school
- Battle of the Alamo -> deaths -> combatants
- how gravitational waves are created, followed by examples and detection
- comparative history, science, nearby-place, and StreetZIM follow-ups

The ternary Q2_0 model passed 16/16 with retrieval evidence enforced. The
raw 1-bit Q1_0 run passed 12/16. Adjudication found:

- two correct `narrate_article` answers that the rubric incorrectly required
  to use `read_article`;
- one 4,228-token prompt exceeding the 4,096-token phone context;
- one WWI/WWII casualty follow-up routed to nearby stories even though the
  generated answer content was correct.

The rubric now accepts either evidence-bearing article tool. Model-facing
place payloads are capped and stripped of map/debug fields. Comparative
casualty, combatant, and cause follow-ups are handled by the deterministic
intent router; interpretive comparison questions remain with the model.

The exact Swift `LlamaCppProvider` was then built and probed with the 1-bit
file. It recognized 498 Q1_0 tensors, offloaded 65/65 layers to Apple Metal,
created a 72 MiB Q4_0 K/V cache at 4K, and successfully generated the expected
native `near_named_place` tool call. This directly rules out the earlier
CPU-only runtime configuration on Mac.

## iPhone 17 Pro Max evidence

The signed Debug build was installed on Jazzman 17 and the model was copied
in eight independently verified parts. The provider rejected an earlier
truncated cache entry, verified the parts totaled exactly 3,803,452,480 bytes,
assembled them to a staging file, atomically renamed the completed GGUF, and
removed the parts.

A console-attached launch reported:

- device `MTL0 (Apple A19 Pro GPU)`, tensor support enabled;
- 498 Q1_0 tensors in a 3.53 GiB / 1.13 BPW model;
- 65/65 model layers offloaded to Metal;
- 3,616.77 MiB Metal-mapped model buffer and 170.51 MiB CPU-mapped tail;
- initially a 4,096-token context with flash attention, raised first to 8,192
  and then to 16,384 after physical-device capacity experiments;
- initially a 72 MiB Metal K/V buffer (Q4_0 keys and values), 144 MiB at 8K
  and approximately 288 MiB at 16K;
- 149.62 MiB Metal recurrent-state buffer and 509.28 MiB Metal compute buffer.

The persisted app log reached `Loaded Bonsai 27B (1-bit · Metal)` while the
full Wikipedia ZIM and California StreetZIM were open. Repeated launches
remained alive without an immediate jetsam or GPU restart. A longer thermal
and multi-turn soak is still required before release qualification.

The first real on-device conversation also passed its discourse check:

- `Tell me about Putin` retrieved the local article and answered with the
  correct offices and terms.
- `What about his parents?` retained the Putin referent, retrieved the Early
  life/Family evidence, and correctly described his father, mother, and
  grandfather.

Persisted timestamps put whole-turn completion at roughly 34.5 seconds and
43.5 seconds respectively. Those initial figures included retrieval, prompt
prefill, and full answer generation. Stage-level instrumentation was added and
rerun on 2026-07-14 with the phone at nominal thermal state:

| turn | prompt | prefill call | TTFT | output | generation total | steady decode |
|---|---:|---:|---:|---:|---:|---:|
| Tell me about gravitational wave | 1,890 tok | 18.591 s (101.7 tok/s) | 23.229 s | 111 tok | 37.097 s | 7.93 tok/s |
| How are they created? | 1,731 tok | 24.457 s (70.8 tok/s) | 28.170 s | 15 tok | 30.177 s | 6.98 tok/s |

The first turn's offline-Wikipedia retrieval took about 0.30 seconds. The
follow-up spent about 2.08 seconds on corpus search/reranking before generation.
Its prompt had a 54-token common prefix, but Bonsai's hybrid recurrent state
could not be partially truncated, so llama.cpp performed a full cache reset
and re-prefilled all 1,731 tokens. `llama_decode` returns before the final
logits read fully synchronizes the Metal work, so TTFT—not the raw prefill-call
duration—is the correct user-visible prefill measure.

This confirms that A19 Pro's tensor path is accelerating batched prefill, but
prompt prefill and final-logit synchronization still dominate latency. The
next performance work should reduce grounded passage tokens and make the
discussion transcript append-only enough to retain the hybrid cache. Decode
is the secondary bottleneck at about 7–8 tokens/s.

### Prefix-cache, 8K, and 16K-context experiments (2026-07-14)

The grounded Wikipedia path now retains an append-only, passage-deduplicated
transcript. Qwen's hidden empty reasoning marker is also reproduced on rebuilt
assistant turns; omitting those four tokens had silently forced Bonsai's hybrid
recurrent cache to reset even when the visible transcript was append-only.

On the M1 Ultra with a synthetic Wikipedia-shaped prompt:

- cold: 1,114-token prefill in 3.492 s; TTFT 3.874 s;
- exact follow-up append: reused 1,162/1,194 tokens, prefilling only 32 in
  0.099 s; TTFT 0.287 s;
- standalone follow-up: cache reset, 1,115-token prefill in 3.474 s; TTFT
  3.854 s.

On iPhone 17 Pro Max with real Wikipedia retrieval ("Tell me about
gravitational wave" → "How are they created?") the follow-up reused 2,010
tokens. Prefill improved from 24.457 s to 7.243 s, TTFT from 28.170 s to
14.349 s, and total generation from 30.177 s to 16.281 s. That capture still
appended three newly ranked sections; warm turns are now capped at the best
unseen section.

The final three-turn target conversation ("Tell me about Vladimir Putin" →
"What about his parents?" → "Where did he go to school?") then completed
without a cache reset:

| turn | prompt/reuse | TTFT | total | result |
|---|---:|---:|---:|---|
| overview | 2,064 / cold | 28.121 s | 43.332 s | correct overview |
| parents | 2,625 / 2,185 reused | 7.691 s | 12.295 s | correct father and mother |
| school | 3,081 / 2,659 reused | 8.190 s | 15.751 s | correct schools and university |

The final footprint was 813 MB. Thermal state reached `serious` on the third
turn and steady decode fell to 6.61 tok/s; sustained decode/thermal behavior is
now the main latency bottleneck in this short run, rather than repeated full
prompt prefill.

Bonsai now uses an 8,192-token Q4 KV window on the target phone. The model-ready
logged footprint increased by only about 71 MB (matching the expected extra 4K
KV), peaked at 932 MB during retrieval, and remained alive through two
45-second verification runs. An exact post-template guard reserves reply space
and trims complete old exchanges, so oversized prompts fail readably instead
of reaching llama.cpp's n_ctx error.

llama.cpp session files are explicit context snapshots, not an automatic SSD
cache. At 8K on the phone, a 1,980-token state produced a 184.5 MB file: save
took 7.686 s and immediate restore 0.047 s with an exact token match. This is
not suitable per query; it remains an opt-in DEBUG benchmark and is a possible
future cold-resume optimization after lifecycle and flash-write policy work.

The final device build allocates a 16,384-token Q4 KV window. The provider
reported `n_ctx=16384`, flash attention enabled, and a 914 MB model-ready
logged footprint. That is about 145 MB above the verified 8K build, matching
the expected additional 8K of Q4 KV. The same three-turn Putin regression
remained fully grounded and completed without a cache reset:

| turn | prompt/reuse | TTFT | total | result |
|---|---:|---:|---:|---|
| overview | 2,064 / cold | 28.124 s | 37.641 s | correct overview |
| parents | 2,581 / 2,141 reused | 7.491 s | 12.123 s | correct father and mother |
| school | 3,036 / 2,614 reused | 10.244 s | 17.481 s | correct schools and university |

The final turn reached 959 MB logged footprint, raised one iOS memory warning,
completed normally, and the process remained alive after 95 seconds. This is
why the 6K grounded-chat ceiling remains in place even though the allocated
window is larger.

A synthetic 9,044-token prompt then proved that the extra capacity is usable,
not merely allocated: the phone returned `OK` without an `n_ctx` error or
termination. From a thermally preheated `serious` state, cold prefill took
144.736 s at 62.5 tok/s, TTFT was 152.045 s, total generation was 152.552 s,
and the peak logged footprint was 981 MB. Prompts of that size are therefore
supported as a recovery margin but are far too slow for routine conversation.

Detailed raw results:

- [`GRID_RESULTS_BONSAI_TERNARY_27B_MAC.md`](../tools/llama-smoke/GRID_RESULTS_BONSAI_TERNARY_27B_MAC.md)
- [`GRID_RESULTS_BONSAI_1BIT_27B_MAC.md`](../tools/llama-smoke/GRID_RESULTS_BONSAI_1BIT_27B_MAC.md)

### Conversational retrieval and interaction pass (2026-07-14)

The grounded path now chooses an evidence depth from the question type and
extracts sentence-aligned windows around the relevant fact instead of placing
the start of every selected section into the prompt. Named sections are added
to the append-only cache only when needed; the anchor lead is canonicalized so
the warm lead cannot consume the one-new-passage follow-up allowance.

On iPhone 17 Pro Max, the same real-ZIM Putin chain improved as follows:

| turn | prompt/reuse | TTFT | total | peak logged footprint | result |
|---|---:|---:|---:|---:|---|
| overview | 649 / cold | 8.235 s | 18.916 s | 959 MB | correct overview |
| parents | 793 / 752 reused | 0.941 s | 4.546 s | 963 MB | correct father and mother |
| school | 1,092 / 828 reused | 4.111 s | 8.766 s | 955 MB | correct school and university |

This reduces the cold overview prompt by 69% versus the earlier 2,064-token
capture and the third-turn prompt by 65% versus 3,036 tokens. The parents
follow-up needed no new passage because its evidence was already present;
school appended one 887-character Education window. The model remained at
nominal thermal state for all three turns and the process stayed alive.

The exact final contextual-chip artifact was then reinstalled and benchmarked
on 2026-07-14. It loaded Bonsai plus the 16K Q4 context in 2.58 seconds with
Wikipedia and California StreetZIM open, and completed the target chain:

| turn | prompt/reuse | TTFT | total | result |
|---|---:|---:|---:|---|
| overview | 696 / cold | 8.390 s | 20.177 s | correct offices, terms, birth |
| parents | 857 / 816 reused | 0.770 s | 6.562 s | correct father and mother |
| school | 1,129 / 911 reused | 2.941 s | 7.664 s | correct high school and university |

Thermal state remained nominal. Generation footprint stayed about 936–963 MB;
the transient Wikipedia retrieval peak was 988 MB. The process remained alive
through the 45-second crash/jetsam watch. The user then tapped contextual chips
for career, early life, and family; each retained the Putin referent, selected
the corresponding grounded section, and generated a new set of relevant chips.

A five-second cancellation benchmark over "Tell me about the French
Revolution" also passed. The UI/session recorded the stop at 5.04 seconds, did
not decode or publish an assistant answer, and remained alive through a
20-second watch. The Metal command already in flight drained at the next
prefill-batch boundary about 1.4 seconds later. Cancellation is therefore
responsive at the product layer but cannot preempt the middle of one submitted
GPU batch; smaller adaptive prefill batches are a possible future refinement.

Every grounded answer now carries visible Wikipedia article/section chips (or
StreetZIM provenance for map/place answers), making library evidence distinct
from model preknowledge. Generation has a real stop path checked between
prefill batches and decode tokens; the voice screen can interrupt speech or
thinking and immediately return to listening.

Suggestion chips are derived from real headings on the pinned article, store a
complete natural-language prompt, suppress the facet just asked, and prefer
subject-specific facets. Biography, battle, and science examples include
"Where did Vladimir Putin go to school?", "Who were the combatants?", "How
many people died?", and "How was it first detected?" Low-signal table-of-
contents residue is omitted instead of padding the row with a generic chip.

The final real-ZIM Mac regressions also caught two cases that unit fixtures did
not: nested event headings could leak an unrelated NATO chip, and "How were
they first detected?" could rank detector design above historical observation.
Event chips now exclude nested biography/foreign-policy residue. First-
detection questions prefer History, Discovery, Observation, or First detection
headings. The resulting gravitational-wave follow-up retrieved "LIGO and Virgo
observations" and correctly described the 14 September 2015 GW150914 signal.

### Smaller-model comparison

The phone-sized LFM2.5 8B-A1B IQ3_XS build (3.30 GiB) was run through the same
real Wikipedia path on the M1 Ultra with Metal offload and a 32K context. It
kept the Alamo referent, answered the combatants follow-up with both Santa
Anna's Mexican troops and the Texian defenders, and reused 1,583/1,740 prompt
tokens; the follow-up completed in 2.2 seconds. In the gravitational-wave
thread it answered the overview in 5.6 seconds and the corrected 2015 direct-
detection follow-up in 5.2 seconds.

The smaller model is viable as a fallback, but it was less robust: one Alamo
overview stopped mid-sentence, and before the deterministic retrieval fixes it
was more likely to follow the wrong section or topic. Keep Bonsai Q1 as the
primary iPhone model because it gives the stronger conversational quality at a
similar compressed file size. Retain the LFM build as the speed/size control,
not as the default. The orchestration changes are shared by both providers.

## Fine-tuning gate

Current decision: no Bonsai fine-tune.

Reconsider only after collecting at least 100 real failed turns from the
phone where all of the following are true:

1. retrieval returned sufficient evidence;
2. deterministic reference/routing logic chose the right subject and tool;
3. the prompt fit the 16K safety window and the grounded 6K rolling budget;
4. the remaining failure is the model's answer, synthesis, or native tool
   choice rather than app orchestration.

Evaluate the resulting error slice against the stock 1-bit model. Fine-tune
only if a native tool-role SFT improves the held-out error slice by at least
10 percentage points without regressing the core 16-scenario suite.

Do not reuse the old fenced-JSON corpus as-is; Bonsai should be trained with
the same native tool-role transcript format used at inference.

## QVAC Fabric compatibility

QVAC Fabric is not currently a drop-in Bonsai training path. Its published
fine-tuning architectures list Qwen 3, Gemma 3, and BitNet, while Bonsai's
GGUF identifies as `qwen35`. Its documented quantization choices also do not
include an explicitly supported Bonsai Q1 export target.

If the fine-tuning gate is crossed, first ask QVAC to confirm all three items
before spending a training run:

- Qwen 3.5/3.6 hybrid recurrent-attention support;
- export or quantization into Prism's Bonsai Q1 format;
- preservation of native tool-role/chat-template behavior.

QVAC references:

- [Fine-tuning documentation](https://docs.qvac.tether.io/ai-capabilities/fine-tuning/)
- [Fabric](https://qvac.tether.io/dev/fabric/)

## Remaining physical-device gate

Completed: final-size verification, Q1/Metal/16K/Q4 runtime validation,
append-only conversational prefix reuse, a real prompt beyond 8K, and
short-term liveness with Wikipedia plus StreetZIM open.

Before treating the provider as release-qualified, complete:

1. run the Alamo and gravitational-wave conversations end to end on the 16K
   build (the Putin chain is complete);
2. repeat the longest place chain and confirm normal prompt compaction stays
   below the 6,144-token rolling budget;
3. measure peak Metal memory and complete a 10-minute thermal loop. TTFT and
   decode rate now have a nominal-thermal baseline above, while the memory
   warning on the third 16K turn remains the release guardrail.


The official MLX 1-bit result (~11 tokens/s on iPhone 17 Pro Max) is the
comparison floor. If experimental GGUF is unstable or materially slower,
keep the same product/orchestration work and switch only the provider to
Prism's MLX Swift path.
