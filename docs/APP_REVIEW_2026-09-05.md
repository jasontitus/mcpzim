# Zimfo app review — September 5, 2026

Scope: first-party iOS/macOS app, MCPZimKit, provider integrations, archive
import/download/sharing, embedded web content, maps/routing, voice lifecycle,
diagnostics, App Intents, and signed deployment scripts. This extends the
[earlier change review](ADVERSARIAL_REVIEW_2026-09-05.md). Vendored engines are
reviewed at their integration boundaries; this is not an audit of every line
of libzim, llama.cpp, MLX, WebKit, or Firebase.

No microphone or audio playback is used. No distribution upload is requested.

The subsequent user-reported conversation failures, shared-log retrieval,
source-selection repairs, latency measurements, and adversarial checks are
recorded in [conversation relevance review](CONVERSATION_RELEVANCE_2026-09-05.md).
The subsequent coffee-to-museum location failure and its repairs are in
[place continuity review](PLACE_CONTINUITY_2026-09-05.md).
The Napoleon map-widget extraction and delayed speech repairs are in the
[Napoleon speech review](NAPOLEON_SPEECH_2026-09-05.md).
Current speech-engine research and isolated Mac memory/latency measurements
are in the [on-device speech review](ON_DEVICE_TTS_REVIEW_2026-09-05.md).

## Finding register

The following issues were recorded from code inspection and runtime/build checks before repairs.
All 19 findings are repaired. Verification and remaining coverage limits are recorded below.

| ID | Severity | Reproducer / consequence | Initial location |
|---|---|---|---|
| S1 | P1 | Model prose can invent facts or reverse negation despite a lexical “supported” badge. Generic/native tool paths and generated map/reduce notes bypass grounding. | ChatSession, AnswerAttribution |
| D1 | P1 | Download a 404/HTML response under an existing `.zim` filename: the old archive is deleted before the new file is validated or moved. | ZimDownloadManager.SessionDelegateShim |
| D2 | P2 | Cancel or pause/resume a download while a progress/completion callback is queued: task labels re-adopt retired tasks and can recreate canceled rows or overwrite new state/resume data. | ZimDownloadManager.itemID, pause, delegate |
| W1 | P1 | A local article can contain a remote image, script, iframe, or fetch. Navigation delegates block navigation but do not block subresource requests. | ZimURLSchemeHandler, web view configurations |
| C1 | P2 | Stop while waiting for GPS: `try? sleep` repeats immediately on cancellation until the deadline. A stream that ends normally on cancellation can also leave an empty reply. | ChatSession.awaitLocationIfAny, generation loop |
| C2 | P1 | Programmatic/macOS-menu model switch can unload a provider while a turn still owns it; rapid selections can overlap loads. The UI guard does not enforce the session API contract. | ChatSession.select |
| L1 | P2 | Refresh replaces the library with Documents entries, dropping external bookmarks from the live library. Repeated imports open duplicate readers/scopes before deduplicating; successful scopes are never released. | ChatSession.openReaders/addReaders/openEach, LibzimReader, ZimfoRunner |
| L2 | P2 | Disabled archives become enabled after relaunch and are still used by Siri/App Intents. Enablement exists only in the live ChatSession array. | LibraryEntry, setEnabled, ZimfoRunner |
| L3 | P2 | A sibling path such as `Documents-extra/file.zim` passes a string-prefix sandbox check and can be treated as deletable app-owned content. | LibraryEntry.isInSandboxDocuments, persistBookmarks |
| P1 | P2 | Concurrent diagnostics uploads are not deduplicated; disabling sharing does not cancel uploads already in progress. A failed Firebase configuration still allows SDK calls afterward. | DiagnosticsUploader, AppTelemetry |
| K1 | P2 | Legacy debug token migration deletes the plaintext token even if Keychain storage fails; setter deletes an existing Keychain item before knowing replacement can succeed. Getter warning is a setter invocation, not infinite recursion. | DebugReportConfig |
| V1 | P1 | Stop/dismiss voice while microphone or speech permission is pending; the old async start resumes afterward and can reopen capture. | VoiceChatController.start |
| D3 | P2 | Nearby import silently deletes a colliding file based only on equal length; different archives can have the same byte count. Voice replacement also deletes the old asset before moving the new one. | ZimSwarmController |
| D4 | P2 | Voice destination helper accepts `kokoro_mlx/../../…`; upstream transport validation currently limits reachability, but the app boundary itself is unsafe. | ZimSwarmController.voiceModelDestination |
| C3 | P2 | Phone reproduction: after Einstein, “Show me cafes in Palo Alto” is answered against Einstein. The imperative is not recognized; the host also rejects question-shaped map intents as knowledge questions. | IntentRouter, ChatSession.intentLeavesDiscussion |
| R1 | P2 | Stopping a route does not interrupt A* exploration; direct map-button tasks are not registered with the session cancellation owner. | Router, ChatSession map actions |
| W2 | P1 | Page-world scripts can post Website/Call messages directly to the native bridge, bypassing offline browser resource restrictions. A programmatic link activation also is not proof of a human click. | PlacesWebCoordinator, navigation delegates |
| S2 | P2 | HTML document titles and h1 headings are concatenated into the lead sentence (“Duet Duet A duet…”). Model-markup display cleanup can also alter legitimate source text. | ArticleSections.parse, ChatView |
| B1 | P1 | The Mac Release Crashlytics phase uses shell command substitution for Xcode environment variables; both commands fail and the plist path becomes `//GoogleService-Info-Mac.plist`, failing the build. | project.yml, generated project.pbxproj |

## Verification baseline

Full MCPZimKit suite before these app-wide repairs: **571 tests passed**.
Final suite: **580 tests passed**. All seven silent phone probes and the
seven-question real-ZIM smoke passed. The loaded Wikipedia ZIM itself says
duet performers take turns. That is a source error, not demonstrated model
invention. Exact extraction establishes provenance, not truth or currency.

## Review coverage and limits

Conversation routing, source selection, citations/history/speech boundary,
tool parsing, cancellation, and cache recovery are inspected together. Routing
and binary graph parsers are covered by the full kit tests. Download callback
ownership, file installation, archive scope lifetime, browser resource access,
telemetry consent, native tool tracing, and model switching receive targeted
adversarial checks. Voice is inspected and compiled only because audio is
prohibited. Phone tests use typed DEBUG autorun; no Siri invocation or real
multi-gigabyte download is started. Long-running background transfer survival,
live voice acoustics, and all alternate model weights remain separate runtime
coverage limits; compilation is not reported as a runtime pass.

## Repairs and adversarial disposition

| IDs | Final repair | Evidence / limits |
|---|---|---|
| S1, S2 | Exact source excerpts or missing evidence; current-turn tool provenance; no model drafts in UI/history; native private history reset; repeated HTML titles removed. | Full unit suite plus hostile streaming/no-tool/tool/cancel phone probes and real parent/password questions. Native FM runtime remains untested. |
| D1 | Validate HTTP status, ZIM header, and libzim open before staging; commit by atomic rename; reopen the installed archive in the live library. | Unit tests reject HTML, truncated headers, bad responses, and failed replacement while preserving the original. No large network download started. |
| D2 | Retire task IDs, reject stale restore/progress/completion callbacks, serialize pause/resume handoff, and accept resume data only for its current pause attempt. | Phone replay of canceled-task progress, restored snapshot, and completion: no revived row, no keep-awake ownership, original bytes preserved, staging removed. |
| D3, D4 | Preserve colliding received files under a unique name instead of deduplicating by size; atomic voice replacement; reject traversal components. | Source review and traversal tests. A repeated transfer can now occupy additional storage; no byte-identity claim is made without verification. |
| W1 | CSP on ZIM responses and generated media HTML blocks Internet subresources, retaining local scripts, styles, fonts, blobs, and workers. | Phone: local fetch succeeds; remote fetch/image/script blocked, three policy violations recorded. Real California map loaded config, local tile/font resources, and drew 25 pins. |
| W2 | Native actions live in an isolated WKContentWorld; only trusted clicks forward button/anchor payloads. Page-world action messages and external navigation are denied. | Phone: page cannot see handler, synthetic click cannot dispatch, app-only isolated-world positive control can dispatch. No external link, phone call, or share action was actually opened. Physical click usability is a remaining manual check. |
| C1, C2, R1 | Cancellation checkpoints after streams/tool awaits; GPS waits stop; map-button tasks share cancellation ownership; A* checks cancellation; model switching waits for the owned turn and serializes loads. | Cancellation probe ends as “Stopped.”; routing unit regressions reject canceled/invalid-node work. Alternate-model switching under real memory pressure remains untested. |
| C3 | Recognize explicit POI imperatives; real map intents leave the article pin even when phrased as questions. | Phone: Einstein → cafes in Palo Alto → what is around here, with actual StreetZIM tool results. Named cafe lookup took 0.284 seconds; this is tool time, not a general TTFT benchmark. |
| L1–L3 | Refresh retains external archives; deduplicate before opening; reader owns/relinquishes its security scope; open archives off the main actor; persist disabled identities for chat and App Intents; use canonical directory boundaries. | Path, symlink, and container-relocation tests. Library rebuild cancels active work, clears old source anchors, and uses a revision check against stale rebuild completion. Actual external-provider revocation and Siri invocation remain untested. |
| P1 | Main-actor upload ownership prevents duplicates and stale callbacks; opt-out cancels unfinished uploads; SDK calls require successful Firebase configuration. Privacy copy reflects opt-in sharing. | Build and source review. No diagnostics consent was changed and no test upload was sent. Already transferred bytes cannot be recalled by canceling an upload. |
| K1 | Update Keychain item in place, add only if absent; delete legacy defaults only after successful storage/removal. | Build and source review. No real PAT or Keychain item was read/modified during testing. |
| V1 | A startup-attempt identity is checked after both permission awaits and before starting capture; Stop invalidates it. | iOS compilation and adversarial interleaving review. No microphone/playback test, per the user's instruction. |
| B1 | Use quoted shell environment expansion for the built resources path in both the project generator input and generated project; retain Xcode build-setting syntax in input-file declarations. | The original signed Mac Release build reproduced the failure. The repaired Release build and repository signature gate passed; no Mac app launch was performed. |

The final review also challenged the repairs themselves. The new symlink
regression initially failed because Foundation did not resolve a parent link
when the leaf was absent; canonicalization was corrected and the full suite
rerun. Source rendering was changed to bypass both model-markup scrubbing and
Markdown interpretation. Comparison citations retain each article's archive.
The first real-content phone smoke caught the cafe/Einstein topic pin bug;
the final smoke demonstrates its correction. Browser checks include a
positive control so “blocked everything” cannot masquerade as success.

This was an adversarial self-review of the entire working change set,
including the earlier conversation/latency changes, not an independent
reviewer. The earlier cache benchmark results remain in the companion
reports; no inference engine or model weights were replaced during this pass.

## Reproducible evidence

- `swift test --package-path swift`: 580 passed, zero failures.
- Signed iPhone Debug build and signature gate: passed; installed on the connected phone.
- Ordinary phone launch: process alive throughout the final 45-second watch.
- Signed Mac Release build and signature gate: passed. The trust check required access to macOS trust services outside the filesystem sandbox; no signing policy was weakened.
- Device binary UUID: `D557BB03-02C1-3FBD-9E7D-41DD0BFCE3E5`.
- [Sanitized phone checks](benchmarks/app-review-2026-09-05.json).
- Local detailed evidence (not committed; includes private device logs):
  `/private/tmp/zimfo-full-app-tests.log`,
  `/private/tmp/zimfo-app-review-build.log`,
  `/private/tmp/zimfo-app-review-final-phone.log`, and
  `/private/tmp/zimfo-app-review-map-phone.log`,
  `/private/tmp/zimfo-app-review-health-watch.log`,
  `/private/tmp/zimfo-app-review-mac-retry.log`, and
  `/private/tmp/zimfo-app-review-mac-signature.log`.
- Phone probe flag: `MCPZIM_APP_REVIEW_PROBE=1` (DEBUG only). It does not invoke
  audio. Ordinary launches have neither this flag nor question autorun flags.
- `git diff --check`: passed. No TestFlight upload or commit performed.

Offline browser enforcement follows the [W3C CSP specification](https://www.w3.org/TR/CSP/).
Native-action separation uses Apple's [WKContentWorld](https://developer.apple.com/documentation/webkit/wkcontentworld)
and [world-scoped message handlers](https://developer.apple.com/documentation/webkit/wkusercontentcontroller/addscriptmessagehandler(_:contentworld:name:)).

## Subsequent LocalSwarm integration review

The requested sibling dependency refresh found and repaired four additional
engine integration issues, including a physical-phone download failure. See
the [LocalSwarm update report](LOCALSWARM_UPDATE_2026-09-05.md) for findings,
patches, regression proof, and silent transfer results. The original 19-item
register and evidence above describe the preceding review pass.
