# Current-article Siri work — September 13–14, 2026

## User scope and overnight operation

Expose only the article actively being discussed or previewed. Keep archive search in Zimfo. No bulk Spotlight index, reading-history index, whole-ZIM transfer, chat-transcript export, TestFlight upload, or OS upgrade. User authorized overnight iteration using the Mac client, tests and adversarial reviews. Phone can remain locked.

An hourly thread follow-up was created for eight runs: `zimfo-single-article-siri-overnight`. Continue useful independent work, not repeated unchanged polling. Update this document so subsequent runs can resume without repeating completed work. Current host macOS is **26.6.2**; Xcode 27 RC is installed separately. This host cannot verify macOS 27 Siri AI behavior.

## Research and design

Apple documents `AppEntity` + `Transferable` + `NSUserActivity.appEntityIdentifier` as an onscreen-content route without an assistant schema. Use a separate current-document entity rather than adding full-text export to the durable article entities already returned by quick-answer actions. The latter could allow old Shortcuts outputs to request arbitrary full articles after the viewing context ends.

Sources:

- [Apple App Intents updates: schema-free onscreen content](https://developer.apple.com/documentation/Updates/AppIntents)
- [NSUserActivity appEntityIdentifier: association and clearing](https://developer.apple.com/documentation/foundation/nsuseractivity/appentityidentifier)
- [WWDC26 advanced App Intents: single-document activities and indexing alternatives](https://developer.apple.com/videos/play/wwdc2026/343/)
- [WWDC25 App Intents: transferable onscreen content](https://developer.apple.com/videos/play/wwdc2025/275/)

`SiriArticleReference` captures exact archive, path, title, and library fingerprint from successful tool results. Discussion source bundles can grow through retrieval; only their anchor document is exposed, with a visible “Discussing” label. The source must have an exact resolved path; never search again by title to reconstruct one.

`ZimfoCurrentArticleEntity` has an opaque in-memory ID. Its query resolves only the current valid lease, and suggests nothing. It conforms to `Transferable` using export-only plain text. `SiriCurrentArticleContext` rechecks visibility eligibility, ID, exact source, fingerprint, cancellation, and payload size before/after asynchronous retrieval. Replacing a document or clearing its owning view invalidates prior requests. A different window's disappearance must not clear the newer owner's context.

The view activity disables search, public indexing, prediction, and Handoff. No `IndexedEntity`, `CSSearchableIndex`, full text in activity userInfo, webpage fallback, or historical donation. Clearing association prevents future retrieval; it cannot recall text Siri already received or erase Apple's conversation history.

`OfflineKnowledge.document` exports source attribution and complete parsed sections, with an 8 MiB HTML guard, at most 250 sections, and a **128 KiB UTF-8 transfer limit**. Exceeding the limit refuses the transfer rather than silently truncating an article. This bounds transfer/parsing entry, not libzim's internal decompression allocation. Article text is untrusted document data, never used as agent/system instructions.

Scene inactivity alone does not revoke context, because invoking Siri can make the app inactive. Background, hidden navigation, setup/search/voice presentation, generation, source changes, and disappearance suppress the chat context. Search article previews publish only after successful source validation.

## Implementation in progress

- New `ios/MCPZimChat/AppIntents/SiriCurrentArticle.swift`, registered in the existing Xcode project targets alongside the intent implementation.
- `swift/Sources/MCPZimKit/OfflineKnowledge.swift`: bounded single-document export.
- `ChatSession.swift`: exact source capture from discussion/overview/source-directive results.
- `ChatView.swift` and `RootView.swift`: visible source label, current-document view activity and suppression.
- `ZimfoIntentsTests.swift`: lifecycle, race, ownership, eligibility/fingerprint, cancellation, bounds and activity flags.
- `OfflineKnowledgeTests.swift`: complete cleaned document, Unicode size rejection and invalid article identity.

Build/test logs:

- `/tmp/zimfo-current-article-core-tests.log`
- `/tmp/zimfo-current-article-mac-build.log`

Core selection passed **58 tests**. Mac test-host build and iOS 27 signed build both compiled successfully. iOS signature and metadata gates passed, including registration of the current entity, plain-text export-only transfer, no unrelated domain schema, and the activity type. Damaged metadata with a missing entity or missing activity registration was correctly rejected.

The real Mac integration test **passed**: the actual `ZimfoCurrentArticleEntity.exported(as: .plainText)` returned **89,009 bytes**, containing the selected Putin source and birthplace evidence; a subsequent export after revocation correctly failed, and the query no longer resolved the ID. The measured transfer after title resolution/cache warmup was about 5 ms; this is not a cold end-to-end search benchmark. Logs: `/tmp/zimfo-current-article-mac-tests.log`.

The first hosted suite failed its two actual-window tests because the host could not acquire window focus. An activation wait also failed. Computer Use confirmed **the Mac is locked and cannot automatically unlock**. Do not ask the sleeping user to unlock it, disable the lock, or claim interactive UI validation. Actual-window tests now explicitly skip if focus is unavailable; independent notification-handler tests exercise A→empty B→article B→A revocation/restoration without desktop access. Final hosted suite: **24 selected tests, 22 passed, 2 skipped, zero failures**. The two skips are the actual interactive window tests. Deterministic window-notification coverage, all context/race tests, the real Transferable test, and existing article-identity/diagnostics tests passed. Combined with 58 core passes, this is **80 passing tests and two explicit UI skips**. The final bounded adversarial review found no remaining blockers in the reviewed context/focus contracts. Before any runnable Mac launch, verify signature using the repository gate. Preserve all unrelated working-tree edits.

## Adversarial iterations

1. Initial review required exact path capture, a separate current-only entity, no merged RAG sources, guards after every await, and clear visibility lifetimes. These are implemented.
2. Implementation review identified Mac multiwindow focus and preview view reuse. Added actual key-window observation and article-ID-bound preview tasks with state reset/cancellation checks.
3. Follow-up found an empty/settings window could leave another window’s article available. Context now separately tracks the published NSWindow and revokes when a different window becomes key, even without any new article. Comparing window identity avoids same-window chat/preview notification-order races. Tests include actual NSWindow key changes and switching to an empty window.
4. The raw-size guard cannot prevent libzim's initial allocation/decompression; documented explicitly. No claim of a hard process-memory limit.

### First overnight follow-up — 23:46 PDT

An independent adversarial review of `ChatSession` wiring found stale-source paths beyond the entity lifecycle itself. Fixed these before building the next phone artifact:

- Explicit offline handoffs revoke the prior Siri source before validating the new one, including stale/failed handoffs.
- Direct article choices, discussion entry, narration, article reading and route actions revoke the old source before awaiting a tool. A failed replacement cannot restore it. Reading currently suppresses the prior discussion document; it does not publish the narrated article as a new current entity.
- Same-topic overviews compare exact source identity. Selecting the same title from another archive/path replaces the source and its discussion state.
- Explicit subject substitutions revoke the historical anchor. A second review found the caller's copied discussion state could restore the source after failed exploration; both the caller copy and stored state now clear it before that fallback. The final bounded review confirmed this path is closed and found no introduced blockers in the changed lifetime code.

Added six `SiriDiscussionWiringTests` using the production chat router, a scripted provider and tiny fixture archives. They cover exact resolved paths, discussion switching/reset, identical titles in different archives, failed explicit handoffs, failed article cards, reading-button failures and spoken reading. All six passed without loading a model or requesting GPS.

Added three `CurrentDocumentBoundaryTests`: oversized raw HTML must never request parsed sections; cancellation during a suspended raw load prevents parsing; cancellation during a suspended section load prevents export even if the service ignores cancellation and returns normally. The core selection passed **61 tests**.

Final hosted selection: **29 executed, 28 passed, one skipped, zero failures**. The empty-window interactive test was excluded; the actual-key-window test explicitly skipped because focus was unavailable. Both interactive checks remain pending. The notification test, existing entity/lifecycle tests, real-archive Transferable test and all six new chat tests passed. Combined result: **89 passing tests, two desktop-focus checks pending**. Future headless runs should exclude the exact names `testActualMacWindowKeyChangesRestoreCorrectArticle` and `testFocusingWindowWithoutArticleRevokesPreviousWindow` rather than retrying focus acquisition.

Both signed builds succeeded. Mac signature gate passed before hosted execution. iPhone build **202609140645** passed the signature and Siri metadata gates; it has **not been installed**. This supersedes the previous `202609140600` artifact at the same path.

Logs for this pass:

- `/tmp/zimfo-siri-wiring-core-tests.log`
- `/tmp/zimfo-siri-wiring-mac-build.log`
- `/tmp/zimfo-siri-wiring-mac-signature.log`
- `/tmp/zimfo-siri-wiring-mac-tests.log`
- `/tmp/zimfo-siri-wiring-ios-build.log`
- `/tmp/zimfo-siri-wiring-ios-signature.log`

The suspended-I/O tests bound export behavior, not libzim allocation. This pass did not measure sustained memory growth or verify Siri AI conversation routing.

### Second overnight follow-up — 00:40 PDT

Added and passed a real-library repeated-transfer check: **100 measured exports** after five warmups, each byte-identical to the **89,009-byte** baseline. The current entity ID remained stable, then clearing the context made both another export and ID resolution fail. The loop took **0.288 seconds**, including an intentional 1 ms yield per iteration. Process footprint samples every 20 iterations were **318.689, 318.798, 318.814, 318.954, 318.954, 318.970 MiB**: a **0.281 MiB** increase. The test keeps one baseline and scopes each temporary result to a helper. These are warm, process-level observations; they do not prove leak freedom, bound cold decompression, or establish phone memory behavior.

Added a direct regression for the copied discussion-state bug from the first follow-up. A fixture article mentions another person; an exact section card establishes a dated question. A spoken subject substitution then encounters the normal unavailable-model guard and falls back to evidence in the original discussion. The test requires the old Siri source to stay revoked and checks the fallback trace, excluding normal topic-exit or new-overview routes. An independent adversarial review confirmed the test exercises the intended path without production test seams.

The first test run exposed an unrelated fixture assumption: a freeform “When did she study in Paris?” did not return the expected date with the model unloaded. The actual revocation/fallback assertions passed. The setup now uses the existing validated section-card flow to establish that question independently of lexical answer scoring. The final **seven chat wiring tests passed, zero failures/skips**. The repeated-export test passed separately. Aggregate coverage now has **91 passing tests**, with the same two desktop-focus checks pending. No production code changed this pass, so the signed iPhone artifact remains build `202609140645`; no iPhone rebuild or installation was needed.

Artifacts:

- `/tmp/zimfo-siri-repeat-mac-build.log`, `/tmp/zimfo-siri-repeat-mac-signature.log`
- `/tmp/zimfo-siri-repeat-mac-tests.log` (resource measurement and initial fixture assertion failure)
- `/tmp/zimfo-siri-followup-mac-build.log`, `/tmp/zimfo-siri-followup-mac-signature.log`
- `/tmp/zimfo-siri-followup-mac-tests.log` (final seven passing chat tests)

The remaining acceptance work requires an unlocked desktop or the user's upgraded phone. No further independent implementation/test gaps are currently identified. Later heartbeats should avoid repeating completed tests or posting unchanged status; resume only for new evidence, user input, or an actionable gap.

### Unlocked Mac UI check — 01:12 PDT

After the user confirmed the Mac was unlocked, Computer Use successfully drove the signed app. Verified the following in the actual SwiftUI interface:

- Discussing Vladimir Putin completed from the downloaded archive in about 1.7 seconds and displayed **Discussing: Vladimir Putin**.
- Past Logs showed `current_article_context published` for this live discussion (`2026-09-14_01-07-33_8C2A3F55.log`, publication at `08:08:15Z`). This verifies live publication, not a Siri export request.
- Settings and Past Logs hid the discussion UI; returning restored the Putin source label.
- Switching to Albert Einstein removed the source label during preparation, then displayed **Discussing: Albert Einstein** after about 1.6 seconds.
- New Conversation cleared the label and returned to the empty trailhead. Only this newly created test conversation was cleared.

The two hosted AppKit window tests still skipped. Added diagnostic detail and confirmed the specific failure: `active=false, hidden=false, policy=0, canBecomeKey=true, visible=true, keyWindow=nil`. Thus the XCTest-launched process did not activate, despite a regular activation policy and eligible visible windows. This must no longer be reported as evidence that the desktop is locked. Direct app interaction worked. The actual multiwindow automated assertions remain unverified; the notification-handler tests and live single-window transitions are separate passing evidence. No production code or phone artifact changed.

Logs: `/tmp/zimfo-siri-unlocked-tests.log` and `/tmp/zimfo-siri-focus-diagnostic-tests.log`; signed test build/gate logs use `/tmp/zimfo-siri-focus-diagnostic-*.log`. Aggregate remains **91 passing tests**, two hosted focus tests skipped. Do not repeatedly rerun them without addressing test-host activation. The next product acceptance step is Siri AI on the updated phone, using the morning checklist below.

## Next overnight pass

The implementation and three review/test passes are complete. Continue only with useful independent work; do not repeatedly rerun the passing suite without a change or unresolved concern.

1. ChatSession discussion/source switching/reset and failed-exploration copied-state regression coverage is complete. The latter exercises unavailable-model failure, not malformed model-selection output.
2. Raw-HTML rejection before parsing and cancellation during both suspended load stages now have passing service-boundary tests. No further boundary reruns needed without changes.
3. The bounded real-archive repeated-export/resource check is complete. Do not rerun it without a new concern; retain the warm-process measurement limitations above.
4. The user unlocked the Mac and live single-window SwiftUI checks passed (above). Hosted multiwindow checks still cannot activate the XCTest process. Investigate that specific harness limitation before rerunning; do not ask the user to unlock again or change security settings. Siri AI and actual export through Siri remain device acceptance work.
5. iOS 27 build `202609140645`, signature and metadata gates passed after the shared source-lifetime fixes. It is now installed on the phone (see deployment below). No additional iOS build is necessary unless shared code changes again.
6. After any changes, run focused tests, signature gate before runnable app execution, adversarial review, and update results here. No TestFlight upload or OS upgrades.

## iOS 27 deployment — September 14

The user completed the phone update and Xcode's required system-wide device-support installation. Read-only checks confirmed CoreDevice **642.16**, completed Xcode first-launch setup, and the connected phone running **iOS 27.0 (24A437)**. The earlier unavailable-device state was resolved. The app signature and Siri metadata gates were rechecked before deployment.

Installed and launched **Zimfo 1.0, build 202609140645**, verified independently using the phone's installed-app listing. The startup log identifies this same build, both local archives, successful Bonsai model/context loading and prefix-cache restoration. No app rebuild, archive indexing, or phone-library changes were needed.

The original deployment monitor reported the app absent at its 40-second sample. A subsequent live query found Zimfo running as PID **1193**, and the user confirmed the app stayed open without interaction. The contemporaneous memory-pressure report lists Zimfo as active/frontmost with **no termination reason**; `largestProcess=MCPZimChat` alone is not evidence that it was killed. The initial report is an apparent false negative, not a confirmed app crash.

Found and fixed a monitoring defect: failed `devicectl` process queries were previously converted to an empty process list. They now report **unknown**, while a successful query without the app remains a failure; an unavailable final query returns **inconclusive (exit 3)**. Four mocked command-boundary regression tests passed, plus a bounded independent adversarial review with no blockers. The corrected on-device watch **passed all nine samples through 45 seconds**, without reinstalling or relaunching the app. This measures sampled process state, not uninterrupted survival between samples.

Artifacts: `/tmp/zimfo-siri27-phone-deploy.log`, `/tmp/zimfo-ios27-startup.log`, `/tmp/zimfo-siri27-phone-watch.log`; memory reports in `/tmp/mcpzim-crash-wifi/JetsamEvent-2026-09-14-120858.ips` and `JetsamEvent-2026-09-14-120951.ips`. No transcript or full crash report is committed here.

The first live test failed: after discussing Vladimir Putin, “Where did he go to school?” produced Siri's contact-card school-address response. The next section records the evidence and diagnostic changes.

## iOS 27 contact-routing failure and activity diagnostics — September 14

On build `202609140645`, the user requested “Tell me about Vladimir Putin” in Zimfo. The local overview completed and `current_article_context published` appeared at **19:13:28Z**. The user reports Siri answered the school follow-up with “it looks like you don't have a school address in your contact card”. The pulled session log contains **no `current_article_query` or `current_article_export`**. This trial does not establish successful Siri document access; a prepared in-memory source is not proof that SwiftUI configured its `NSUserActivity` or that Siri consumed it.

The same log shows app backgrounding at **19:13:50Z**, followed by another publication at **19:13:56Z**. The user clarified that Siri showed a line with the contact-card reply over the Zimfo UI, rather than the user switching away. The existing log cannot explain the relationship between that overlay and the recorded transition. Existing code deliberately retains a source during scene inactivity and revokes it during actual backgrounding. Do not weaken that lifetime based on this log alone.

Added diagnostic events in `SiriCurrentArticle.swift`: activity configuration now records `entity_associated` or `configuration_rejected`; context revocation records a fixed internal reason; visibility refresh records scene phase and Boolean visibility/source flags. These events contain no article titles, text, archive paths, or questions. `entity_associated` means the app assigned the entity to the activity, not that Siri requested or accepted it. The change does not alter source lifetime, indexing, or retrieval behavior.

An independent adversarial review found no lifecycle or privacy regression. Signed iOS and Mac builds succeeded, both signature gates passed, and the iOS Siri metadata gate passed. **15 focused tests passed** (nine current-article lifecycle tests and six diagnostics tests); the same two hosted actual-window-focus checks remain excluded for their documented activation limitation. These reruns are a subset of the previously reported coverage, not 15 additional unique tests.

Apple's [September 14 Siri AI release announcement](https://www.apple.com/newsroom/2026/09/siri-ai-a-profoundly-more-capable-and-personal-assistant-is-here/) says rollout begins today in beta, including onscreen awareness and conversational follow-ups on supported devices with Apple Intelligence enabled. An installed iOS 27 version alone does not confirm this phone's Siri AI readiness. Apple's [contextual-cues documentation](https://developer.apple.com/documentation/AppIntents/providing-contextual-cues-to-apple-intelligence-and-siri) supports associating a single onscreen entity through `NSUserActivity`; there is no basis here for archive-wide indexing or impersonating an unrelated domain schema.

Next targeted acceptance: keep a completed Putin discussion visible and ask **“Summarize this article”**. Correlate activity association, scene/revocation events, query, and export. This distinguishes article access from pronoun routing; a plausible answer without a source export still does not prove use of Zimfo's archive.

Diagnostic artifact: build `202609141925`, installed and launched on the phone; the deployment monitor passed all nine samples through 45 seconds. This confirms sampled startup survival, not a Siri routing fix. Build/test/deployment logs: `/tmp/zimfo-siri-activity-ios-build.log`, `/tmp/zimfo-siri-activity-mac-build.log`, `/tmp/zimfo-siri-activity-mac-tests.log`, `/tmp/zimfo-siri-activity-phone-deploy.log`. Original phone evidence: `/tmp/zimfo-ios27-siri-followup.log`; the full conversation log is not committed.

### Diagnostic phone reproduction

The installed-app listing independently confirms build `202609141925`. After a new Putin overview, the phone logged active/visible source publication and **`current_article_activity entity_associated` at 19:23:35Z**. The user repeated the school question and received the contact-card response, then asked “Summarize this article” and received “to do that I'll need to use ChatGPT”. The subsequently pulled log has no query, export, configuration rejection, or context-revocation event after that association. Thus the SwiftUI activity-configuration callback now has direct phone evidence; actual Siri content consumption still does not.

Apple's current iOS 27 user guide documents a **separate opt-in: Settings → Siri → Try Siri AI (Beta)** ([Turn on Siri AI](https://support.apple.com/en-euro/guide/iphone/aside/glos65nenlkk/27/ios/27)). The next user check is whether that opt-in has been completed, followed by the same current-article test. Do not infer its state solely from iOS 27, Apple Intelligence availability, or a ChatGPT suggestion. The ChatGPT prompt is not evidence of local/offline success, and no approval to send screen content to ChatGPT has been assumed.

Logs: `/tmp/zimfo-siri-activity-phone-repro.log`, `/tmp/zimfo-siri-activity-phone-summary.log`; installed version evidence `/tmp/zimfo-siri-activity-installed-apps.json`.

## Morning handoff

- The overnight heartbeat was paused at 07:42 PDT on September 14 after independent work completed. Later unchanged checks did not rerun tests or change the implementation. Resume work with device-test results or a concrete new gap.
- Signed Mac artifact: `ios/build-mac-bonsai/Build/Products/Release/MCPZimChatMac.app` (test-host build, signature verified).
- Signed iPhone artifact: `ios/build-siri27/Build/Products/Debug-iphoneos/MCPZimChat.app`, diagnostic build `202609141925`, signature/metadata verified, **installed on iOS 27.0**, launch/45-second watch passed.
- Mac OS remains 26.6.2. Siri AI conversation behavior requires iOS/macOS 27 acceptance.
- The phone OS update and app installation are complete. Test the current-article rows in [phone acceptance](SIRI_PHONE_ACCEPTANCE.md).
- Open/discuss a document in Zimfo before asking Siri about “this article” or “he”. Pure Siri questions from the Home Screen do not create an onscreen document context in this implementation.
- Watch `current_article_query` and `current_article_export` events; a plausible answer without an export may be Siri's own knowledge rather than Zimfo's source.
- There is no archive-wide Spotlight indexing, no saved reading-history index, no transcript export, and no claim that source text already delivered to Siri can be recalled.
