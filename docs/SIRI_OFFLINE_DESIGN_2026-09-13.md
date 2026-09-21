# Zimfo, Siri AI, and offline questions

Research and implementation review: 13 September 2026. The working tree already contained unrelated conversation, voice, and model changes; this work preserves them. Build and device-validation results are recorded at the end.

## What Apple actually supports

Apple announced conversational Siri AI, personal context, onscreen awareness, and conversation history for the 27 releases. Its June announcement lists device, language, and regional restrictions. These are system features, not a promise that every third-party action gains unrestricted conversation or offline speech recognition. [Apple announcement](https://www.apple.com/newsroom/2026/06/apple-introduces-siri-ai-a-profoundly-more-capable-and-personal-assistant/)

Xcode 27 RC, build 27A266a, was released September 9. Apple opened submissions built with the new SDKs. We downloaded the RC from the signed-in Apple Developer account, verified the Apple-signed XIP, and installed it beside Xcode 26.6 at `/Applications/Xcode-27-RC.app`. Its compiler is Swift 6.4 and it includes iOS 27.0. The global `xcode-select` selection remains unchanged; use `DEVELOPER_DIR=/Applications/Xcode-27-RC.app/Contents/Developer` for this toolchain. [Release](https://developer.apple.com/news/releases/?id=09092026h), [submission announcement](https://developer.apple.com/news/?id=k1mtkt1k)

App Intents remains the integration foundation. App Schemas describe recognized content and actions; entity queries resolve references, and supported indexed entities let Siri answer questions about content. Returning an entity allows another action to take that entity as input. This is a stronger contract than trying to retain Siri's private conversation history. [WWDC26: App Schemas](https://developer.apple.com/videos/play/wwdc2026/240/)

The new `.system.searchInApp` schema opens an app's search results and works independently of its other domains. `IntentValueQuery` is a route for supported structured queries over content too large to index. Indexed content requires update/delete/reindex handling. Onscreen entity annotations associate what is visible with queryable identities. These are distinct capabilities; a search-in-app intent is not a background question-answering provider. [WWDC26: advanced integration](https://developer.apple.com/videos/play/wwdc2026/343/), [search schema](https://developer.apple.com/documentation/appintents/appschema/systemintent/searchinapp)

A critical restriction: use schemas only when their meaning matches the app. Apple's current `reader.document` reference lists **Shortcuts**, while `maps.place` and `files.file` list **Siri and Shortcuts**. Do not label an encyclopedia as messages/notes, claim an arbitrary custom Q&A intent has schema-level natural-language routing, or assume `reader.document` enables Siri Q&A. The baseline uses App Shortcuts and explicit article entities. The iOS 27 implementation adds the genuine system search schema. [Domain guidance](https://developer.apple.com/documentation/appintents/making-actions-and-content-discoverable-by-apple-intelligence), [reader document](https://developer.apple.com/documentation/appintents/appschema/readerentity/document), [map place](https://developer.apple.com/documentation/appintents/appschema/mapsentity/place)

## User experience implemented

- Say **“Ask an offline question with Zimfo”**, pause for **“What would you like to know?”**, then speak the question. “Zimfo encyclopedia” is an alternate registered trigger. A recognizable named Wikipedia topic resolves locally. If it cannot identify the topic, Siri asks for an article title. A fuzzy match, including a single suggestion, is a choice to confirm rather than an answer.
- **“Look something up with Zimfo”** remains available. Its old first-search-hit/raw-HTML behavior is replaced by parsed, source-bound prose and `requestDisambiguation` choices.
- The question action returns an **Offline article** entity with title, archive, and a short source excerpt. In Shortcuts, connect that result to the Article parameter of another question action. “What about her family?” uses that explicit article. Bare pronouns without an article never borrow the app's most recent conversation.
- **“Zimfo what's nearby”** or **“What's around a place with Zimfo”** queries StreetZim. Category parameters cover cafés, restaurants, pharmacies, hospitals, hotels, and museums. Ambiguous named locations use `requestDisambiguation`, including locality/category/coordinates. The selected coordinates are used directly; we do not geocode the selected name again.
- **“Continue a question in Zimfo”** stages a question and, if supplied, its exact source. The app exposes the pending question, preserves an existing draft, and lets the user add it to the composer. Sending starts the established article discussion pipeline with the original archive and path. The source remains visible and can be cleared.
- On iOS 27, the system search schema opens a local result list with article previews and a question-to-conversation action. Selecting a result revalidates its archive before displaying prose.

This supports explicit repeatable question/answer turns. Whether Siri carries the returned entity across spontaneous voice-only follow-ups is a device acceptance test, not an API guarantee. Longer unrestricted discussion runs in Zimfo's existing conversation UI.

## Implementation map

Paths below are relative to the repository root. The core is reusable without App Intents; the app layer handles Siri interaction, library identity, and foreground UI.

| Component | Responsibility |
|---|---|
| `swift/Sources/MCPZimKit/OfflineKnowledge.swift` | Validate input, infer explicit topics, resolve exact articles or bounded choices, extract source sentences, refuse live/unsupported/oversized answers |
| `ios/MCPZimChat/AppIntents/ZimfoRunner.swift` | Open enabled archives off the main actor, cache by deterministic library fingerprint, reject filename collisions, format nearby results |
| `ios/MCPZimChat/AppIntents/ZimfoIntents.swift` | App Shortcuts, article entity/query, `requestValue` and `requestDisambiguation`, authentication, returned entity, structured handoff, iOS 27 search schema |
| `ios/MCPZimChat/App/MCPZimChatApp.swift` | Construct the session without background model loading |
| `ios/MCPZimChat/Views/RootView.swift` | Foreground initialization, value-based Settings navigation, search results and validated article preview |
| `ios/MCPZimChat/Views/ChatView.swift` | Pending question banner, draft preservation, visible source and Clear source, guarded submission |
| `ios/MCPZimChat/Chat/ChatSession.swift` | Start model/location work only on foreground launch; dispatch a submitted handoff through the existing discussion pipeline using the exact archive/path |
| `ios/scripts/verify-siri-metadata.py` | Verify actual iOS 27 build output contains the expected actions, authentication/foreground behavior, article entity, and search schema |

The question flow is: validate question → load enabled library → resolve named subject or request article → confirm fuzzy candidates → reload after Siri clarification → validate article ID and current archive → extract evidence → recheck library generation → return spoken text plus article entity. A supplied Article parameter follows the same validation path and rejects a question naming a different subject.

The handoff transports `{question, source: {zim, path, title}, libraryVersion}` in memory. It is not a serialized Siri transcript. The app preserves an existing composer draft and requires explicit submission. The exact source is validated again when submitted; archive replacement does not silently switch editions.

The search schema is guarded by Swift compiler 6.4 and iOS/macOS 27 availability. Existing custom shortcuts compile with Xcode 26.6. Authentication is required for content actions; `.system.searchInApp` specifically requires local-device authentication. Entity identifiers are base64-encoded sorted JSON, not secret credentials or cryptographic proof: live source validation is what makes them usable.

## Boundaries and offline operation

`OfflineKnowledge` is a pure Swift service wrapper over `ZimService`. It calls no network client, model, microphone, or generic MCP dispatcher. It uses the existing title resolver, parsed article sections, relevance selection, and `SourceBoundAnswer`. A Wikipedia archive is not mandatory when StreetZim already bundles the requested Wikipedia article.

Answers identify downloaded source material and use complete source sentences. No evidence means abstention. This proves source provenance, not perfect relevance or that an encyclopedia claim is true. Live-condition questions receive an archive limitation response. StreetZim place results explicitly say they reflect downloaded coverage, and do not claim live opening status or absence of places outside that coverage.

Input is limited to 500 characters. Candidate lists, answer scoring, and output are bounded. The answer path rejects oversized material rather than cutting a sentence before a qualification or negation. Title resolution and entity validation can parse article content before that byte check, so this is not a general parsing or memory limit. Cancellation is checked at service boundaries. These are **not hard execution or decompression deadlines**: synchronous libzim reads and cold opening of a large enabled library can still exceed Siri's budget. A future dedicated reader worker process or reader-level interrupt/byte API would be needed for a reliable hard cutoff.

The raw HTML budget is 8 MiB; the separate scoring budget remains 500,000 parsed UTF-8 bytes and 250 sections, with a 900-character hard speech limit. Raw markup is not a reliable proxy for prose size. These thresholds were separated after a real Einstein question hit the original combined `tooLarge` failure. Optional numeric diagnostics report each size before its guard and check cancellation after the callback.

The app is constructed with `autoLoadOnInit: false`. Library/GPS/model prewarming starts only after a foreground activation. A session-owned model-start guard prevents view recreation from starting a second load; initialization continues through transient permission/Siri overlays. Background App Intents use `ZimfoRunner` instead of constructing a conversational pipeline.

Downloaded content and local retrieval work without internet. Siri activation, dictation, language support, routing, and Siri AI synthesis remain Apple's separate system services and must be tested in airplane mode. Shortcuts' typed parameters are the fallback if voice invocation needs connectivity. For offline in-app conversation, download both content and the chosen model beforehand. This integration does not add a cloud-answer fallback.

## Identity and privacy

Article identity contains a deterministic library fingerprint, archive name, exact ZIM path, and title. Rehydration uses sorted-key JSON. Fingerprints include enabled/disabled state, documents metadata, and external bookmark/file metadata. Changed/removed archives invalidate old entities. Two distinct enabled archives with the same filename fail closed because the underlying service addresses archives by name.

Every invocation re-reads the source and checks current library identity. The entity's display title and returned excerpt are not authority for a later question. A supplied article conflicting with a named subject is rejected; a user-confirmed correction is applied to the question's subject while preserving its factual facet.

Content-bearing question and nearby actions require authentication. No transcript, location history, or implicit last-topic state is persisted for this feature. No bulk Spotlight index or behavioral donations are created. Siri and Shortcuts receive requested text and may retain it according to system settings; “local retrieval” does not mean text returned to Siri can never be processed by Apple. Handoff is an explicit action, with bounded in-memory draft state.

## Adversarial reviews and resulting changes

An independent reviewer challenged the initial architecture and implementation while coding. Findings and responses:

| Attack or failure | Response |
|---|---|
| Misspelled topic produces a plausible unrelated first hit | Explicit `requestDisambiguation`, even for one fuzzy candidate |
| Mercury disambiguation page becomes a factual answer | Detect disambiguation prose and filter unreadable/disambiguation choices |
| “How does it work?” becomes an article named “it” | Reject pronoun subject phrases before topic inference |
| Confirm Marz → Mars but evidence still searches for Marz | Canonicalize only the confirmed subject; preserve question facet |
| Re-encoded JSON changes entity ID | Sorted keys and identifier round-trip tests |
| Saved excerpt survives archive disable/replacement | Revalidate generation and read exact source again |
| Two archives have the same filename | Fail with an actionable rename/disable message |
| Handoff switches archive editions | Carry exact source and fingerprint through composer submission |
| A previous source silently constrains a new typed question | Visible source label, Clear source, clear on empty draft |
| Model load/jetsam on background Siri cold launch | Foreground-gated initialization and session-owned model guard |
| Handoff disappears behind Voice/Settings | Dismiss voice presentation and return navigation to chat |
| Setup/generation drops the submitted draft | Stage first; composer checks setup, generation, and model switching before clearing |
| Negation after speech cutoff changes meaning | Whole-sentence output or explicit oversized-answer refusal |
| Out-of-coverage or old map facts treated as live truth | Coverage wording, archive attribution, and live-data abstention |

## Device acceptance before release

Use the [step-by-step phone checklist](SIRI_PHONE_ACCEPTANCE.md) after the September 14 phone update. Updating the phone OS does not install this working-tree build of Zimfo.

Run on an iOS 27 Siri AI device and a supported older iPhone. Test both typed Shortcuts and Siri voice: exact article, typo, Mercury, empty library, StreetZim-only bundled article, source removed while Siri is clarifying, permission denied, fresh GPS unavailable, category filtering, and current-hours questions. Repeat a cold launch with airplane mode enabled and model assets absent. Verify no model download or prewarm starts for a background answer.

Chain two question actions using the returned article; then change the Article parameter, rename/remove the archive, and try again. Stage a handoff while typing, generating, viewing Settings, and in voice mode. Verify the draft and exact source survive until Send, and a second handoff does not overwrite the first. Exercise iOS 27 system search and inspect the exported App Intents metadata.

Full semantic indexing and `.maps.place`/`IntentValueQuery` adoption are subsequent work, after device tests establish supported query contracts and an explicit content-indexing preference. The current implementation does not claim these features.

## Automated test coverage

Tests use generated in-memory articles and do not need real Wikipedia downloads, models, or GPS. The app-level tests run inside a signed macOS host; they validate app identity handling, not Siri's user interface.

| Coverage | Tests / gate |
|---|---|
| Exact local prose, family follow-up facet, absent evidence, StreetZim bundled article | `OfflineKnowledgeTests`, `SourceBoundAnswerTests`, `BundledArticleTests` |
| Single fuzzy suggestion needs confirmation; duplicate, unreadable, and disambiguation results filtered; no matches | `OfflineKnowledgeTests` resolution cases |
| Corrected spelling preserves factual facet and unrelated words | `testConfirmedCorrectionPreservesQuestionFacetAndOtherWords` |
| Same article title in two archives answers from the selected edition | `testAnswerUsesSelectedArchiveWhenTitlesCollide` |
| Invalid input, live-data refusal, byte limit, whole-sentence speech limit | `OfflineKnowledgeTests` boundary cases |
| Cancellation reaches production answer and resolution entry points | Both cancellation cases in `OfflineKnowledgeTests` |
| Stable serialized IDs; stale generations, unknown archive, missing/oversized paths and changed/disambiguation titles rejected; display/excerpt metadata ignored | `SiriArticleIdentityTests` |
| Exact source and generation preserved in structured handoff | `testHandoffRetainsExactSourceAndGeneration` |
| Exported action registration, authentication, foreground/background flags, new search schema | `verify-siri-metadata.py` against the built iPhone app |

These tests do not simulate Siri's clarification UI, a real archive changing during a suspended Siri prompt, GPS permissions, UI sheet/navigation lifecycle, or foreground/background memory use. The phone checklist covers those integration gaps. Cancellation tests establish cooperative entry checks, not interruption of synchronous libzim decompression. A handoff structure test does not establish that all composer gestures work on a phone.

## iOS 26.2 field report and routing correction

On September 13, the user reported that “ask zimfo tell me about Albert Einstein” produced a location query, and a later attempt reported no contact named Zimfo. Siri also asked to enable Zimfo answers. The saved app log contained only ordinary startup/background entries. This did **not** establish which App Intent, if any, executed: those entry points had no logger.

The initial documentation and suggested “Ask Zimfo” trigger were insufficient. Classic App Shortcuts do not accept arbitrary String question text appended to a trigger phrase. Use a distinct registered phrase, wait for the parameter prompt, and then speak the question. The new build removes bare “Ask Zimfo” and keeps question phrases separate from location actions. This changes what the app advertises; it cannot guarantee Siri's recognition or clear Siri's private routing state. [Apple: parameterized phrase limitations](https://developer.apple.com/videos/play/wwdc2022/10170/), [Apple: App Shortcuts matching and parameters](https://developer.apple.com/videos/play/wwdc2023/10102/)

`SiriDiagnostics` and `SiriInvocation` in `LogArchive.swift` now log every action's execution and every article query separately from `ChatSession`. Each invocation has an ID, action name, build/OS, begin/end, elapsed time, and failure/cancellation classification. Prompt boundaries use `awaiting_*` events; exceptions at those boundaries are recorded as `parameter_exit`, because App Intents can use exceptions for prompting/resumption. They are not automatically treated as retrieval failures. The question parameter is optional in metadata but explicitly requested inside `perform()` when absent, so the app can log before asking for it. Question length and candidate count may be recorded; raw questions, titles, coordinates, answers, and localized error messages are not recorded by this logger.

Events are written synchronously into the existing rolling Past Logs archive and mirrored into a bounded 200-entry live ring. Debug pane Copy/Report includes this ring. Logging does not construct a chat session or start a model. Rapid launches use distinct log filenames so one launch cannot truncate another's log within the same second. Existing optional log-upload settings still govern whether archived diagnostics leave the device.

Interpretation: `action=NearbyPlaceIntent begin` establishes a location action ran; `action=AskOfflineQuestionIntent begin` establishes the question action ran; `article_query_*` alone establishes only discovery/resolution. If no invocation appears, Siri may have stopped at routing, permission, or parameter handling outside app code. Past Logs remains the source for earlier processes; the live ring only contains the current process's events.

The pre-existing “previous session ended uncleanly” warning is a heuristic based on the last log line. A normally completed background Siri action can be the last writer before iOS ends the process; that warning alone is not evidence of a crash. Inspect the invocation's end event and device termination report before drawing that conclusion.

The next phone check should use the exact two-turn phrase, then run the question directly in Shortcuts, then intentionally run a nearby action. Compare action names in logs. The unsupported combined utterance remains a negative/exploratory test, not a supported acceptance condition.

Correction validation: signed iOS build `202609132200` passed compilation, signature verification, and the metadata gate. The gate now checks distinct question phrases, removal of bare “Ask Zimfo”, and question prompting inside `perform()`. Three damaged metadata variants were rejected (ambiguous trigger, question phrase assigned to nearby, prompting moved outside execution). All **11** selected signed Mac app tests passed: six identity tests and five diagnostics tests. The latter cover durable logs without a ChatSession, rapid archive reopening, error-content redaction, parameter/cancellation outcomes, bounded live logs, and a real invalid-question invocation that logs before library/GPS access. Logs: `/tmp/zimfo-siri-routing-ios-build.log`, `/tmp/zimfo-siri-routing-mac-tests.log`. The prior 40 core tests remain applicable; no core retrieval code changed in this correction. Corrected voice behavior is not yet verified.

The corrected build was installed on the paired iPhone on September 13. Launch succeeded after unlocking, and the repository deployment script confirmed it stayed alive throughout its 45-second check. This validates installation and launch, not Siri recognition.

## Einstein size-failure follow-up

The next device logs **did** show `AskOfflineQuestionIntent`, successful question prompting, two loaded archives, and an article selection. Both attempts then failed with `OfflineKnowledge.Failure` code 3 (`tooLarge`). The diagnostic OS version was **26.6.2**. Siri presented that thrown localized error as “something went wrong”. The shared error code alone could not distinguish raw HTML, parsed-content, or spoken-length limits.

The follow-up separates the HTML budget from the prose budget, measures all three, and returns recognized question-action failures as an explicit dialog with **no article output**. Cancellation and system parameter-control errors still propagate. Successful actions retain the article entity for chaining; callers must now allow an absent article on an explained failure. No selected source or cached excerpt is returned as a successful answer when validation fails.

A DEBUG-only, opt-in `MCPZIM_SIRI_EINSTEIN_SMOKE=1` launch invokes the real question action with the fixed public subject “Tell me about Albert Einstein”. It runs once in the normal foreground launch sequence, logs sizes and whether an attributed answer returned, and does not edit the conversation or use Siri speech. This probe is for on-device retrieval validation, not proof that Siri speaks the result correctly.

Build `202609132225` was installed and the direct-intent probe passed on the phone. Einstein measured **763,730 HTML bytes**, **103,161 parsed bytes**, **33 sections**, and **202 answer characters** (247 including source attribution). The question action returned in **573 ms**; the whole probe took **577 ms**. This confirms that the old 500,000-byte raw-HTML guard was the blocker. The run returned an attributed answer, not a no-evidence fallback. Log: `/tmp/zimfo-siri-einstein-smoke.log`.

Validation for this correction: **43 core tests passed**, including markup-heavy Wikipedia, >8 MiB rejection, parsed-prose/section limits, and cancellation after numeric diagnostics; **12 signed Mac app tests passed**, including an explained invalid-input result with no article and preserving system control-flow errors. Signed iPhone build and metadata gates passed. Logs: `/tmp/zimfo-siri-size-tests.log`, `/tmp/zimfo-siri-size-mac-tests.log`, `/tmp/zimfo-siri-size-ios-build.log`. The new Siri voice presentation of the successful result remains a user-device acceptance test.

## Birthplace question follow-up

The user reported that “Tell me where Vladimir Putin was born” asked which offline article to use. The encyclopedia topic extractor lacked both the direct “where was … born” form and its embedded “tell me where … was born” form. The local question action now recognizes these, their birthdate equivalents, polite wrappers, and “where is … born”, while preserving the original overview router. Pronouns without an explicit source, relatives, and multiple people do not silently select an article. Fuzzy matches still use `requestDisambiguation`; exact matches do not need it.

Topic recognition alone was insufficient: a keyword match on “born” could return a date-only lead or a relative's birthplace. `EvidenceQuestion` now requires a named assertion connecting the article subject to a place. It rejects date-only and relative evidence, unsupported relative-subject requests, and pronoun-only source assertions whose antecedent might be another person. A selected biography's parenthetical title qualifier is removed for prose-name matching, while full article identity remains mandatory. This is a conservative English extraction rule, not general natural-language understanding; unsupported sentence syntax can produce an explicit no-evidence answer.

Adversarial review identified and regression tests cover: a parent mentioned in the same sentence as the subject's birthplace, a spouse introduced before a pronoun birthplace assertion, and a biography titled “John Smith (explorer)”. Existing overview, authorship, source identity, and correction behavior remain covered. Diagnostics now distinguish `topic_inferred`, `topic_not_inferred`, `exact_article_match`, and candidate counts without logging question text.

A DEBUG-only opt-in launch with `MCPZIM_SIRI_PUTIN_BIRTHPLACE_SMOKE=1` calls the real question action with the fixed public question, checks that the returned entity is Vladimir Putin, and checks its source excerpt for a birthplace assertion mentioning Leningrad or Saint Petersburg. The first physical run, build `202609132255`, passed: automatic exact article selection, 1,232,662 HTML bytes, 98,134 parsed bytes, 39 sections, 199 answer characters, and 659 ms action time. Log: `/tmp/zimfo-siri-putin-birthplace-smoke.log`. This validates local retrieval on the user's iOS 26.6.2 phone, not Siri speech recognition. A separate voice attempt in that same log still needed a source choice; raw questions are not retained. The user subsequently reported asking where Vladimir Putin was born, choosing Vladimir Putin when prompted, and receiving the answer. Their spontaneous “where did he go to school” follow-up produced Siri’s contact-card school-address response. This is a failed conversational acceptance case: the current action does not establish an ongoing Zimfo Siri session. Explicit Article chaining in Shortcuts and in-app continuation remain the implemented paths; repeating a free-standing pronoun to Siri is not supported by the current integration.

The final regression selection passed **55 tests** (`OfflineKnowledgeTests`, `SourceBoundAnswerTests`, `BundledArticleTests`, `EvidenceQuestionTests`). Log: `/tmp/zimfo-siri-birthplace-tests.log`. Final reviewed build `202609132305` passed signed iOS compilation, signature verification, and Siri metadata verification, and was installed on the phone. Its direct-intent probe again automatically selected the exact article and verified the birthplace excerpt, taking **498 ms** for the action and 500 ms for the full probe. Log: `/tmp/zimfo-siri-putin-birthplace-final.log`. The deployment script reported success through its 45-second watch. A new system jetsam report listed a suspended Zimfo process with reason `idle-exit`; the watch alone must not be interpreted as proof against background process termination. The final bounded adversarial review found no remaining blockers in these corrections. Siri recognition of the newly supported phrasing still needs a fresh voice trial.

## Validation results

- Xcode 27 RC signed generic iPhone Debug build: **passed**, including App Intents schema validation. Signature gate verified `com.tiltastech.zimfo`, team `A6G8H8NGAM`. Artifact: `ios/build-siri27/Build/Products/Debug-iphoneos/MCPZimChat.app`; build log: `/tmp/zimfo-siri27-ios-build.log`.
- Exported `Metadata.appintents/extract.actionsdata`: verified question and handoff actions, article entity, authentication policies, and `SystemSearchInAppIntent` schema registration. The schema validator requires local-device authentication for the search action; the implementation satisfies that requirement.
- The reusable metadata gate passes on the signed iPhone artifact. Four temporary damaged variants were rejected: missing search schema, missing article entity, unauthenticated question action, and search incorrectly marked as background. The signed artifact itself was not modified.
- Core regression selection: **40 tests passed**, including 15 `OfflineKnowledgeTests`, 19 `SourceBoundAnswerTests`, and 6 `BundledArticleTests`. The follow-up strengthened two existing tests and added five cases. Log: `/tmp/zimfo-siri-swift-tests-followup.log`.
- App-level entity/handoff identity tests: **6 passed** in the signed macOS host, built with Xcode 26.6. Verified stable identifiers, rejection of stale/missing/oversized sources and changed/disambiguation titles, distrust of display metadata, and exact-source handoff. Signature gate verified `org.mcpzim.MCPZimChatMac`, team `A6G8H8NGAM`, before execution. Log: `/tmp/zimfo-siri-mac-tests-followup.log`. The initial Release test build exposed an existing unrelated TTS test referencing a Debug-only helper. The focused run excludes `TTSNormalizationTests.swift`; this is not a full application test-suite pass.
- Adversarial review: initial design and repeated implementation reviews completed. Final navigation and parsing-limit documentation corrections are incorporated.
- September 13 phone installation: signature and Siri metadata gates passed, then `mcp-deploy-verify.sh` installed the existing iOS 27 SDK build on the paired iPhone 17 Pro Max (user reports iOS 26.2). Launch succeeded and the app remained alive throughout the 45-second check. Two system jetsam reports appeared; both listed Zimfo as active with no termination reason. This is an installation/foreground-launch smoke test, not validation of Siri behavior.
- iOS 27 voice follow-ups, physical-device cold launch, and airplane-mode invocation remain **unverified**. No TestFlight upload was performed.

The iOS 27 SDK and Metal toolchain are installed and the signed device build works. Xcode's first-launch simulator components still need administrator setup: CoreSimulator reports an older installed version. Simulator execution is not part of the validation above.

Reproduce the iPhone build from the repository root:

```sh
DEVELOPER_DIR=/Applications/Xcode-27-RC.app/Contents/Developer \
  xcodebuild -project ios/MCPZimChat.xcodeproj -scheme MCPZimChat \
  -configuration Debug -destination 'generic/platform=iOS' \
  -derivedDataPath ios/build-siri27 \
  -clonedSourcePackagesDirPath ios/build-bonsai/SourcePackages \
  DEVELOPMENT_TEAM=A6G8H8NGAM CODE_SIGN_STYLE=Automatic \
  -allowProvisioningUpdates build
ios/scripts/verify-app-signature.sh \
  ios/build-siri27/Build/Products/Debug-iphoneos/MCPZimChat.app \
  com.tiltastech.zimfo
```

Run core regression tests with:

```sh
cd swift
swift test --filter 'OfflineKnowledgeTests|SourceBoundAnswerTests|BundledArticleTests|EvidenceQuestionTests'
```

Check the iOS 27 artifact's integration metadata (read-only, no app launch):

```sh
python3 ios/scripts/verify-siri-metadata.py \
  ios/build-siri27/Build/Products/Debug-iphoneos/MCPZimChat.app
```

For the focused app tests, use the signed macOS workflow in [SIGNED_APP_BUILDS.md](SIGNED_APP_BUILDS.md) with scheme `MCPZimChatMacTests`, `ENABLE_TESTABILITY=YES`, and `-only-testing:MCPZimChatMacTests/SiriArticleIdentityTests`. Run `build-for-testing`, pass the signature gate, then run `test-without-building` with the same configuration and DerivedData path. Until the unrelated Release TTS test is repaired, also pass `EXCLUDED_SOURCE_FILE_NAMES=TTSNormalizationTests.swift` for this focused run. Do not launch an unsigned/ad-hoc test host.


## Single current-document integration

The subsequent [overnight current-article work](SIRI_CURRENT_ARTICLE_WORK.md) adds a separate current-only document entity with plain-text transfer and onscreen activity context. It does not index the archive or change the general article entity into a full-text exporter. Follow that work log for current build/test status and the distinction between local contract validation and Siri AI acceptance.
