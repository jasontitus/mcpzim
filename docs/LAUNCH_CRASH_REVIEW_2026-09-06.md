# Launch crash review — September 6, 2026

## Finding and repair

The background model downloader introduced in `d6793e2f6` also added a complete
SHA-256 pass over an already cached GGUF on each process launch. Its
`FileHandle.read(upToCount:)` loop lacked an inner autorelease pool. The read
buffers could survive until the worker's outer pool drained, retaining memory
proportional to the entire file **before llama.cpp opened the model**.

A controlled macOS reproduction using a 512 MiB sparse file and an outer
worker-level autorelease pool measured:

| Routine | Peak process RSS | Digest |
| --- | ---: | --- |
| Previous loop | 545,423,360 bytes | `9acca8e8c22201155389f65abbf6bc9723edc7384ead80503839f49dcc56d767` |
| Per-chunk pool | 9,666,560 bytes | Same |
| Fixed shipping routine, actual 3,803,452,480-byte Bonsai | 13,058,048 bytes | `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0` |

The fix drains temporary buffers after each 1 MiB read. Checksum verification,
background transfers, and Bonsai availability are retained. This is a
reproduced memory defect, not a claim that a new phone launch has been observed.
The phone was unavailable to devicectl during this review.

## Phone evidence

The two user attachments (September 5 at 11:50 and 11:55, build 1) reach model
ready, restore the prefix cache, and finish their speech playback. Their
unclean-session banner alone is not a crash diagnosis.

Newer opt-in diagnostic logs were retrieved from Firebase Storage under device
`5393c74a`, including September 6 15:13, 18:19, 18:40, and 19:07. In build
`20260907010803`, 18:19:12 enters `ensureGGUF` with cached Bonsai; 18:19:14 reports
a memory warning at 2704 MB. There is no successful cache-verification or model
open message. Similar failures begin with build `20260906214904`.

Build `20260907015458` switches to uncached LFM. At 19:19 it still reports a
setup stall and later a 2703.7 MB memory warning. This does not support treating
a smaller model or prewarm deferral as a sufficient fix. The shared checksum
routine is also used at download completion.

Memory samples are process-wide, so a delta logged around opening a ZIM cannot
be attributed entirely to that ZIM while the hash worker runs concurrently.
Likewise the startup-fix commit comments attributing ~1.8 GB to the reranker
were hypotheses, not isolated measurements. No OS jetsam report was available
here; the shared logs and memory reproduction strongly support memory
exhaustion, but do not prove every reported termination had the same cause.

## Change review and preservation

The two-day history comprises the large 123-file consolidation `d6793e2f6`
and startup commits `2ac263eda`, `45c041edc`, and `9ed9e11a6`. Review focused on
runtime changes and launch reachability; benchmark JSON, generated reports,
and other evidence files do not execute at launch.

| Area | Assessment / disposition |
| --- | --- |
| Model downloader / cache checks | Reproduced unbounded hashing memory; fixed shared routine. |
| Checksum memoization | A verdict was keyed by path/size/mtime but omitted the expected digest. Fixed key; added good-then-bad digest regression. |
| Restored model destinations | Persisted filesystem paths used `URL(string:)`. Fixed to `URL(fileURLWithPath:)`; tested a restored path containing spaces. |
| Model selection / memory guards | Keep current saved-model selection and restored Bonsai eligibility. No further speculative capacity changes. |
| Startup prewarm edits | Deferral remains; it is not the checksum fix. Initialization and launch still have separate warmup paths; these warrant cleanup but are not necessary to repair the reproduced allocation bug. |
| Conversation grounding, article disambiguation, map radius / empty results | Retain source-bound answers and Carmel Valley repairs; exercised by the full package regression suite. These execute on queries rather than the failing pre-model phase. |
| llama.cpp prefix cache and performance changes | Retain Q4 KV, batching, and persisted prefix restoration; failed traces stop before successful model load and prefix restore. |
| Speech / playback changes | Retain both engines and playback-completion fixes. Attached successful logs demonstrate both speech paths after model load. No speech retuning included. |
| LocalSwarm and ZIM sharing | Retain existing integration; no sharing mutation in this repair. |

Adversarial review checked that the fix still consumes every file byte,
preserves the pinned digest, propagates read failures, keeps memory bounded
under an outer autorelease pool, and cannot reuse a success for a different
expected digest. The same function serves both cache checks and downloaded
files. It does not bypass integrity checking or modify model weights.

## Validation and remaining limits

- Signed Mac test host: 10 model-download tests passed.
- Repeatable memory regression: `python3 tools/model-download/check-hash-memory.py`.
  It compiles the actual shipping function, measures peak RSS, and fails above
  64 MiB for the 512 MiB fixture. An optional file argument tests real weights.
- Actual Bonsai checksum and memory verified as above.
- Phone screen-lock, force-quit, and physical launch validation remain pending;
  TestFlight is the delivery path while the user is remote.
- The download manager still deserves a separate concurrency audit: task
  restoration is asynchronous, and per-model completion storage has only one
  waiter slot. This repair does not claim those interruption cases are proven.

Release acceptance and full-suite results are appended after verification.

Validation update: the full MCPZimKit suite passed 613 tests with no failures.
The signed Mac app passed the signature gate and remained running for over a
minute after a normal launch with its saved Ternary Bonsai model mapped. This
uses a different model/device than the phone; it is not evidence of phone
memory headroom. macOS privacy protection prevented reading that app's live
container log, so no stronger ready-state claim is made from that launch.

## Release acceptance

Zimfo **1.0 (20260907023920)** is in internal TestFlight testing. The canonical
upload script exited successfully, Xcode reported `Upload succeeded` and
`EXPORT SUCCEEDED`, and the script reported `upload submitted`. The exact
exported IPA passed the distribution signature gate for `com.tiltastech.zimfo`,
team `A6G8H8NGAM`. App Store Connect reported `IN_BETA_TESTING`; the script
verified its assignment to `InternalTesters`.

Local release evidence: `/private/tmp/zimfo-launch-repair-testflight.log` and
`/private/tmp/zimfo-launch-distribution-signature.log`. Archive retained at
`ios/build-testflight/1.0-20260907023920/Zimfo.xcarchive`.
