# Field issues from device logs — 2026-08-03

Source sessions: `eval/corpus/raw/5393c74a/2026-08-03_21-34-40.log` (k1 kart /
go-cart) and the war-of-1812 session (`2026-08-03_23-14-36.log` on-device;
copy at `/tmp/war1812.log` — re-pull via devicectl or logpipe if lost).
Replay each transcript with MCPZimEvalCLI --probe-discuss as the acceptance
test when fixing.

## P0 — grounded KV cache diverging every turn (likely regression)
War-of-1812 session: EVERY grounded turn logs `reused=0 ·
mode=reset-after-divergence` with LCP collapsing to 5–377 tokens, ~6 s full
re-prefill per turn even mid-discussion (append should be warm). Suspect the
2026-08-03 `rawAssistantText` change (A2, commit 496c7a14^..) interacting
with the grounded prompt cache: the cache stores `cacheAnswer` (raw buffer)
but the transcript rebuild may now feed scrubbed vs raw text inconsistently.
Verify on Mac: 3-turn probe-discuss, watch `[Perf] reused=`. Bisect against
352a4599 if unclear.

## P1 — silent dead turn on fast-path locate miss
"Where is k1 kart" → `locate` failed ("could not resolve") → NO reply
rendered at all; user reset the conversation 19 s later. Fix: the locate
fast-path miss must (a) fall back to name search ("K1 Speed" would hit),
(b) always render an honest miss reply + suggestions, mirroring the
article-miss did-you-mean chain.

## P1 — near_places unmapped kind returns instant zero
"nearest go cart place" → kinds=["go cart place"] not in chip vocabulary →
scan guard (correct) → 0 hits in 2 ms, no fallback. Fix: kind-synonym map
("go cart"/"go-kart"/"karting" → karting category; general mechanism), and
unmapped kinds fall back to NAME search over places before returning empty.

## P1 — wrong-article binding + lost facet in war-of-1812 session
1. "Tell me about the war of 1812 what were the what were the causes?" →
   title kept the trailing question clause + dictation stutter → search
   rescue → "1812 Louisiana hurricane". Attribution correctly flagged it
   (0.50) and the model declined. Fix: strip trailing interrogative clauses
   ("what were the causes") and dedupe stuttered n-grams before title
   dispatch.
2. "How many people died on each side in the war?" → "the war" bound to a
   capital-punishment article (user: "Not the one about capital
   punishment"). Resolver should prefer the discussion's pinned subject for
   "the war/the X" anaphora before any fresh search.
3. After the correction, the re-grounded turn answered start/end DATES, not
   the casualty question — the correction replaced the question so the
   facet (deaths) was lost. Fix: corrections ("not the one about X") should
   re-run the PREVIOUS question with the corrected binding, not answer the
   correction text itself.

## P2 — verify-only
- Disneyland map tile AbortErrors (code=20, z7–z14): user reports the map
  looked fine — consistent with MapLibre canceling superseded tiles during
  the initial camera jump. Confirm benign, then drop these lines to a
  counter to keep logs readable.
- mcp-deploy-verify.sh still reads backgrounded/suspended as "app died";
  teach it to check the session log tail for a clean "backgrounded" line
  before declaring failure (two false alarms so far).

## Standing context
Attribution chips (496c7a14) are live and already catching bad retrieval —
keep the `[Attrib]` line in any refactor. Firebase key rotation is pending
at next TestFlight cycle (see memory + gitignore note).
