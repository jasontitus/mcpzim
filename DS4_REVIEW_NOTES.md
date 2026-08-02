# Batch 1 notes (KokoroSwift MLX perf)

Reviewed 20 files (Package.swift, 10 Albert files, 9 BuildingBlocks files) for performance only.
Hot path context: Albert/CustomAlbert runs once per utterance (cold); the decoder/generator
(Decoder.swift, Generator.swift) run per audio frame and use the BuildingBlocks conv/norm modules —
so ConvWeighted/AdaIN1d/AdainResBlk1d/AdaLayerNorm/InstanceNorm1d findings are anchored to the hot
per-frame path.

Findings written to .ds4-sweep-perf/batch-1.md:
- ConvWeighted.swift:100 weightNorm recomputed per call (medium)
- AdainResBlk1d.swift:95-134 swappedAxes layout churn (low)
- AdaLayerNorm.swift:25-26 double reduction (low)
- InstanceNorm1d.swift:68-70 double reduction (low)

Albert files marked clean: they are cold (per-utterance), standard attention, and init-time
element-by-element weight copies are one-time. Package.swift is config only.

## batch-2 (KokoroSwift MLX TTS) — 2026-08-01
Reviewed 20 files under mlx-performance-review + performance-review.
Top findings: KokoroTTS.createAlignmentTarget per-frame `.item()` loop (critical) and per-phoneme `.item()` map (high); TimestampPredictor per-token `.item()`/slice-sum (high); LSTM per-token xProj slice + backward insert(at:0) (med/low); DurationEncoder/TextEncoder redundant xPad zeros+copy + per-layer re-mask (med/low); Tokenizer per-char String alloc (low).
Dismissed: MLXSTFT/Generator/ProsodyPredictor/Decoder loops are batch==1 or config-bounded (numUpsamples/numKernels/depth); SineGen/SourceModule per-utterance one-time syncs; KokoroConfig one-time cached load.

## batch-6 (MLX LLM models M-Z) — 2026-08-01
Reviewed 20 MLX Swift LM model files under mlx-performance-review + performance-review.
Findings written to .ds4-sweep-perf/batch-6.md:
- NanoChat.swift:17 hand-composed functionalRMSNorm instead of fused MLXFast.rmsNorm (medium)
- Phi.swift:66 per-layer queries.asType(.float32)+back dtype round-trip in decode (medium)
- NemotronH.swift:82 per-call MLXArray.ones([groupSize]) identity weight in Mamba decode (low)
Dismissed: sanitize()/dequant loops are one-time cold load; MoE gates are vectorized whole-array ops
(argPartition/takeAlong) with no host syncs; Qwen35/Qwen3Next conv-state concatenated is bounded by
convKernelSize (kernel-1 rows kept in cache); gatedDeltaUpdate/ssmUpdate prefer Metal kernels for
seqLen==1 decode (host branch on known dim, not a sync); getLlama4AttentionScale builds one [L] tensor
per forward, not per token; scalar MLXArray(invScale)/MLXArray(cap) creations are constant-factor.
Cross-file: checked GatedDelta.swift/SSM.swift reachability — the per-token sequential fallbacks live
in those files, not this batch.

## batch-7 (MLX LLM common adapters/chat/sampling) — 2026-08-01
Reviewed 20 files under mlx-performance-review + performance-review.
Findings written to .ds4-sweep-perf/batch-7.md:
- DoRA+Layers.swift:22-24 DoRA forward recomputes full adapted-weight matmul + row norm per call (plus QDoRA per-call dequantizedWeight) (medium)
- ChatSession.swift:264 output += chunk O(n^2) string accumulation over tokens (medium)
- Evaluate.swift:444-448 FrequencyPenalty per-token vocab-size histogram alloc (low); :1889 `(batch ?? [])+[element]` O(n^2) collect reducer (low)
Dismissed: SSM segsum l×l repeated is algorithm-inherent Mamba; TokenIterator .item() per token is the asyncEval pipelined token handoff (inherent); TokenRing.append bounded by repetitionContextSize; LoRA `scale*z` fp32 scalar promotion is constant-factor on small d_out tensors; InterpolationUtils Metal kernels are inherent bicubic support-region; config/registry/adapter-cold files clean.
## batch-8 (MLXLMCommon) — 2026-08-01
Reviewed 20 files under mlx-performance-review + performance-review. Findings in batch-8.md: Tokenizer NaiveStreamingDetokenizer O(n^2) re-decode (med); Gemma clipResidual fp32 round-trip (med) + rmsNorm 1+weight recomputed (low); ProportionalRoPE split/concat churn (med); YarnRoPE/SuScaledRoPE per-token copy+scale (low); RotatingKVCache rolled mask per-token (low). Dismissed: KVCacheSimple step-preallocated growth is O(1) amortized; SwitchLinear swappedAxes is a cheap view; registries/config/load/factory are cold path; tool parsers bounded by single tool call.

## Batch 10 (perf: MLXVLM Models Gemma4/GlmOcr/Idefics3)
- Read all 3 files fully. Applied mlx-performance-review + performance-review checklists.
- Sync-point census: Gemma4 1489 (.item() pooler max) and 1706 (.item() image-token count) are one-time per-prefill scalar syncs — dismissed per skill FP.
- gemma4MaskedScatter:60 — `mask.flattened().asArray(Bool.self)` on broadcast [B,L,hidden] mask → host materialization O(L*hidden) per image prefill. Reported medium.
- Gemma4 multi-dim RoPE loop (118) and patchPositions host loop (1569) and processor token-expansion (1856) — reported low.
- GlmOcr applyMrope per-chunk index (105) — low; getRopeIndex host per-token enumeration (941) — medium; vision per-frame SDPA loop (509) — low.
- Idefics3 prepareInputsForMultimodal per-row slicing (715) — medium; Encoder outputHiddenStates discarded (534/664) — low.
- KV-cache: all 3 use Standard/RotatingKVCache or attentionWithCacheUpdate — no per-token KV concat, dismissed.
- Dismissed (no unbounded growth): gemma4BuildLayerTypes, sanitize funcs (one-time weight load), GlmOcr autoregressive per-token posIds build (seqLen=1), Gemma4 intermediates array (bounded by layers).

## batch-12 (MLXVLM Models + factory) — 2026-08-01
Reviewed: Qwen2VL, Qwen35, Qwen35MoE, Qwen3VL, QwenVL, SmolVLM2, VLMModel, VLMModelFactory, Package.swift.
- Qwen35.swift:100-119 gatedDeltaOps — sequential T scan, per-step subscripts, no eval, ys accumulation → O(T) graph growth on prefill (linear layers ~3/4). HIGH.
- Qwen35.swift:366-378 applyInterleavedMRope — per-dim subscript+stack per attention layer per decode token. MED.
- Qwen3VL.swift:924-938 same applyInterleavedMRope per-dim loop. MED.
- Qwen3VL.swift:1328-1510 getRopeIndex — .item()/asArray host syncs + firstIndex scans per image, once per prompt. LOW.
- Qwen3VL.swift:542-550 vision Attention rebuilds dense seq^2 mask per block + host loop over grids. LOW.
- SmolVLM2.swift:114-121,134-141 — String += in prompt-build loops (bounded frames/tiles); hoist repeated-token string. LOW.
- Clean: Qwen2VL (mrope is single split/concat per layer, KV-cache decode correct), Qwen35MoE (init-time sanitize), QwenVL (mergeInputIdsWithImageFeatures host sync is one-time per prompt), VLMModel (protocol), VLMModelFactory (cold-path config/registry), Package.swift (manifest).

## batch-16 (perf, ios/MCPZimChat Providers+Views) 2026-08-01
Read all 9 listed files in full + cross-file ChatSession.swift (partial, 6166 lines), LogArchive.swift, ModelProvider.swift. Ran formatter/metadata greps + dispatch/task greps.
Findings (8):
- LlamaCppProvider.swift:1127 buffered += piece + 1142 contains -> O(n^2) decode loop (medium)
- LlamaCppProvider.swift:1117 pieceBuf + String(decoding:.map) per-token allocs (low)
- ChatView.swift:229 showThinkingIndicator calls displayText; MessageRow:490 calls again -> double regex pipeline per 10Hz push (medium)
- ChatView.swift:473 HeroMediaView no isLatestAssistant guard -> WKWebView per article trace in history (medium)
- DebugPane.swift:21 DateFormatter per row over <=500 debugEntries (medium)
- PastLogsView.swift:70 DateFormatter+ByteCountFormatter per row; 74/78 re-fetch metadata already batched in allFiles() (medium)
- ModelProvider.swift:195 default formatTranscript += over turns (low, fallback only)
- HeroMediaView.swift:129 full-article read to scan 64KB (low)
Dismissed: stateStream continuations leak (single subscription at ChatSession:1885, no growth), TraceKindCache trim (bounded 1024), displayText regexes (static/precompiled -> not regex-recompile finding), LibraryView registryCapabilities contains (bounded 4-5), MockProvider chunked/Task.sleep (dev-only).

## batch-17 (perf, ios/MCPZimChat Views + Voice) 2026-08-01
Read all 11 listed files in full (PlacesWebView 1313, RouteWebView 1092, TTSService 652, RootView 337, ZimURLSchemeHandler 182, KokoroDownloader 166, KokoroAssets 107, VoiceChatView 150, SpeechRecognizerService 355, ObjCExceptionWrapper .h/.m).
Findings (4):
- PlacesWebView.swift:78 payload + 205 resolveSpec re-parse trace.rawResult JSON on every body eval (GPS ticks) — medium
- RouteWebView.swift:70-109 routeEndpoints/turnByTurn/resolveSpec each re-parse rawResult per body eval + downsample/geoJSON rebuild — medium
- KokoroDownloader.swift:116 didWriteData Task{@MainActor}+@Observable state write per progress callback — low
- TTSService.swift:42 normalized samples.map allocates copy per chunk + 2nd PCM copy — low
Dismissed:
- RootView.swift launchQuestions/rawContext busy-wait + output += (DEBUG-only harness, bounded) — per FP (dev-only, maxTokens 8).
- ZimURLSchemeHandler readQueue concurrent queue + full-entry reads: bounded by ZIM entries, off-main, documented; lookup O(n) over library bounded by user ZIMs.
- KokoroAssets isDownloaded/currentBytesOnDisk attributesOfItem per file: fixed 2 files, bounded.
- SpeechRecognizerService sharedLegacyRecognizer reuse is the correct shared-client pattern.
- TTSService prepForTTS ~20 replacingOccurrences passes + while-contains whitespace collapse: bounded by 1200-char cap per utterance, constant-factor.

## batch-18 (perf, ios Voice + Eval/Test CLI) 2026-08-01
Read all 5 listed files in full (VoiceChatController 1179, ConversationalEvalTests 558, GemmaToolEmissionTests 306, ZimfoIntentsTests 192, EvalCLI 118). Cross-file: SpeechRecognizerService, MemoryStats (mach task_info syscall), StreamingSpeechPolicy, ChatSession.debug.
Findings (1):
- VoiceChatController.swift:775 sanitizeForSpeech recomputed every 75ms generation poll over entire growing assistant text even when unchanged (medium).
Dismissed:
- handleCapturedAudio Task{@MainActor} per 100ms buffer (~10/s) — bounded, needed for meter; not a finding.
- streamAssistantReply MemoryStats.physFootprintMB()/availableMemoryMB() — only inside new-text branch, bounded per generation event; not per-poll.
- takeSpeakablePrefix Array(text) — only on new fragment.
- Test/CLI files (ConversationalEval/GemmaToolEmission/ZimfoIntents/EvalCLI): test harness + CLI arg parse, bounded inputs (maxTokens 64-96, debugEntries cap 20k, per-turn scans), not hot paths — clean.
- output += chunk in GemmaToolEmissionTests: tiny 64-96 token outputs, bounded — per FP (small bounded).
