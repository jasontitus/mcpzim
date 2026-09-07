# Conversation relevance and latency — September 5, 2026

The user's Putin conversation retrieved the right ZIM sections but refused
early life, career, family, and NATO follow-ups. After rejecting Family, its
retry returned an incidental family mention from Wealth. The school answer
was the only successful follow-up. The repair keeps complete source excerpts
and does not invoke answer generation or change Bonsai weights/llama.cpp.

## Findings and repairs

| Finding | Repair | Adversarial check |
|---|---|---|
| Friendly card wording loses the identity of the section that generated it. | Cards carry article, archive, and exact heading. A selected card reads that section and nested children directly, without semantic indexing. | Reject different article/archive, missing heading, and modified questions. Stop at the next peer/ancestor heading. |
| Individual sentences must repeat question words such as “early life,” “family,” and “career develop.” | Include heading context and normalize conversational framing. Prefer the requested section over incidental body-word matches. | Heading context cannot cover an added secret password, lunar expedition, or unsupported date. Every emitted sentence still occurs in its attributed ZIM passage. |
| Every Foreign policy heading creates a West/NATO question. | Use foreign-policy wording for that heading; NATO wording requires a NATO heading. | Synthetic foreign-policy-only article never offers NATO. |
| Broad search aliases can accept foreign-relations prose for NATO or a father for a mother. | Separate these evidence matches from retrieval aliases. | Both adversarial tests failed before repair and pass afterward. |
| A literal mother/father gate rejects a clause naming both parents, then accepts an incidental cousin/mother mention on retry. | Retain complete collective parent-identification clauses, without assigning names to roles. Identity questions reject incidental mentions. | The expanded phone replay exposed this regression; unit tests cover the actual parent clause, cousin/care distractors, and parent words in article titles. |
| Retry exclusions accumulate across questions, preventing later retries of previously read sections. | Carry exclusions forward only within the current question's retry. | Review first-attempt reset and retry union at the source-selection boundary. |

A section choice first tries focused selection within its scoped source. If
the friendly question does not match the body's wording, it falls back to
reading that section in source order. This fallback is available only for
validated section navigation; free-form unknown facts do not receive it.
No inferred/paraphrased claims enter UI, attribution, speech, or history.

## Shared debug-log retrieval

“Share debug logs for analysis” uploads to Firebase Storage. Cloud access was
verified with the existing authenticated `gcloud` CLI, including retrieval of
the original failing session. No phone is required to download uploaded logs:

```sh
gcloud storage ls --long 'gs://tiltastech-zimfo.firebasestorage.app/debug-logs/**'
gcloud storage cp 'gs://tiltastech-zimfo.firebasestorage.app/debug-logs/<device>/<session>.log' /private/tmp/zimfo-session.log
```

The uploader excludes the actively written session. A finished session is
eligible on the next launch/background upload pass; the reported session
became available after relaunch. The existing `tools/logpipe/ingest.sh` can
ingest the full corpus. Raw logs contain conversation and location data and
remain outside tracked files; the benchmark contains only test questions,
timings, and source-section names. Consent and upload behavior are unchanged.

## Verification

- Full MCPZimKit suite: **588 tests passed**.
- Signed phone build, install, and silent replay against the actual
  `wikipedia_en_all_maxi_2025-10.zim`, with Bonsai 27B Q1 loaded.
- First replay: all five reported follow-ups return relevant source excerpts;
  both invented-fact questions return missing evidence. All factual sentences
  log exact source attribution. Follow-ups take 4–100 ms from `[User]` to
  `[Assistant]`, with 2–13 ms in excerpt selection/render bookkeeping.
- Final reviewed replay: **14/14 checks pass** — 11 relevant source answers
  and three missing-fact refusals. The source answers have exact attribution
  and no transient refusals. Three reworded free-form questions work without
  selecting a card. A topic switch to Einstein returns the complete parent
  identity clause for the mother question; secret-password questions fail
  closed on both topics.
- Final iOS Debug and macOS Release builds pass their signature gates. The
  Mac app is compiled/verified only; runtime checks use the physical phone.

| Reported turn | Original first reply (ms) | Reviewed first reply (ms) |
|---|---:|---:|
| Putin overview | 293 | 277 |
| Early life | 1052 (refusal) | 26 |
| School | 76 | 4 |
| Career | 72 (refusal) | 8 |
| Family | 67 (refusal, then Wealth excerpt) | 5 |
| West and NATO | 90 (refusal) | 97 |

The three free-form rewordings take 61–67 ms. The first free-form follow-up
on the new Einstein article still prepares a semantic index and takes 1029 ms;
the direct-section optimization does not remove preparation for every route.
Machine-readable results, including each reply's timing and source attribution,
are in [the benchmark](benchmarks/conversation-relevance-2026-09-05.json).

The installed iOS code is `MCPZimChat.debug.dylib`, UUID
`41F3B6D3-5E17-30DD-83AA-AEE6A8564502`, SHA-256
`a7695fd3ef499bc46c77e14ccfe768b74be6b6040f14072e233cc93483e99afd`.
The app was restored to a normal launch with the autorun environment removed.
The installed process then passed the 45-second device health watch.
A system Jetsam snapshot appeared during that watch. It lists MCPZimChat as
active, without a termination reason; the app remained alive at every check.

These are individual on-device observations, not a statistically controlled
10× inference benchmark. Timing starts at submission after normal startup;
it excludes model loading and does not measure the next display refresh.
No audio or microphone use, TestFlight upload, or remote publication occurs.
