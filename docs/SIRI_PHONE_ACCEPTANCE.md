# Siri phone acceptance — iOS 27

Prepared September 13, 2026 for testing after the phone update on September 14. All boxes below start unverified. See [research and implementation](SIRI_OFFLINE_DESIGN_2026-09-13.md) for API sources, limits, and automated validation.

## Prepare

- [ ] Record phone model, exact iOS build, Siri language/region, Siri AI availability, and Zimfo build number.
- [ ] Install the Siri implementation build. Updating iOS alone does not install these Zimfo changes. Follow [signed app builds](SIGNED_APP_BUILDS.md); the current development artifact is `ios/build-siri27/Build/Products/Debug-iphoneos/MCPZimChat.app`.
- [ ] Enable a Wikipedia archive with Albert Einstein, Mars, and Mercury (or choose equivalent available topics). Enable a StreetZim map covering a familiar place. Record archive filenames/editions.
- [ ] For longer in-app discussion, download the chosen local model. Quick question actions should work without a model.
- [ ] Open Shortcuts, find Zimfo's **Ask an offline question**, **Continue a question in Zimfo**, and nearby actions. First test typed parameters to separate app behavior from Siri speech/routing behavior.

## First pass: quick answers and clarification

| Done | Action | Expected result |
|---|---|---|
| [ ] | Say “Ask an offline question with Zimfo”, **wait for Siri's prompt**, then “Tell me about Albert Einstein” | Short answer attributed to the downloaded article; no raw HTML; no need to enter chat |
| [ ] | After the Siri question prompt, say “Tell me where Vladimir Putin was born”; also try “Where was Vladimir Putin born?” and “Where is Vladimir Putin born?” | Automatically selects the exact available biography and returns its birthplace sentence, rather than a date-only lead or an unnecessary article prompt |
| [ ] | With an explicit biography Article, ask where the person’s father was born | No substitution of the biography subject’s own birthplace; unsupported relation gives an explained no-evidence result |
| [ ] | Ask “Tell me about Marz” using typed input | Confirm Mars if suggested, even if it is the only fuzzy match; answer only after choosing |
| [ ] | Ask about Mercury | A choice of specific meanings, or a request for a more specific article; never a factual answer from a disambiguation page |
| [ ] | Cancel a clarification | No answer or conversation draft appears from the cancelled action |
| [ ] | Ask a subject absent from all enabled archives | No guessed answer; request a different article or explain missing content |
| [ ] | Ask about a secret password or another fact absent from the selected article | Say the downloaded source does not contain an answer |
| [ ] | Ask for current hours / “Is it open right now?” | Explain that downloaded material cannot verify current conditions |

## Explicit follow-ups and in-app continuation

Create a two-action shortcut: first **Ask an offline question** with “Tell me about Albert Einstein”; second **Ask an offline question** with “What about his family?”. Set the second action's **Article** to the first action's returned **Offline article** variable, rather than copying its text.

- [ ] The second answer uses Einstein's article and the family facet.
- [ ] Run the family question without an Article parameter. It asks which article to use rather than borrowing the last in-app conversation.
- [ ] Supply Einstein's Article but ask about Mars. It rejects the conflicting subject rather than silently switching sources.
- [ ] Separately try a spontaneous spoken follow-up. Record whether Siri supplies context. This is an Apple integration experiment; explicit Shortcuts chaining is the supported baseline.
- [ ] Chain the returned Article into **Continue a question in Zimfo**. The question and source appear in the app. **Use in message**, then **Send**, starts discussion using that source.
- [ ] Repeat while a draft exists, while generating, from Settings, and from voice mode. Existing text survives; Settings/voice do not hide the new handoff; generation/setup does not discard it.
- [ ] Send a second handoff while one is pending. It is rejected rather than overwriting the first.
- [ ] Use **Clear source**, or empty the composer, then type a new topic. The old article no longer constrains it.

## Library changes and offline behavior

- [ ] While Siri is asking for clarification, disable the source archive in Zimfo, then answer Siri. It asks for a current source rather than serving the stale one. Re-enable the archive afterward.
- [ ] Save an Article output, then rename/replace/remove its archive. Reusing the output fails with an actionable source error. Use a disposable archive for replacement; do not delete the only copy of downloaded content.
- [ ] Disable Wikipedia while leaving a StreetZim archive with a known bundled Wikipedia article enabled. That article still answers locally.
- [ ] Disable all relevant archives. Questions explain how to enable/download content. Re-enable them afterward.
- [ ] Enable airplane mode and explicitly turn Wi-Fi off. Record Bluetooth state. Quit and reopen through typed Shortcuts; repeat the exact question and chained follow-up.
- [ ] Repeat through Siri voice. Record separately whether Siri activates, understands the words, routes the action, and returns its answer. A failure before action execution is not evidence that archive retrieval needs internet.
- [ ] If practical, repeat on an installation with no model assets. Background quick answers must not prompt a model download or show model startup. Avoid deleting the main installation's model just for this check.
- [ ] Repeat the in-app continuation offline with model assets available.

## StreetZim and system search

- [ ] Use **What's around a place** with a known place and category café; results match the category and state downloaded coverage.
- [ ] Use an ambiguous place name; confirm the choice with locality/coordinates. Results must be near the chosen place, not a second resolution of its name.
- [ ] Use **What's around me** with location allowed, then denied. Denied/stale location gives an actionable message; a named-place query remains available.
- [ ] Ask outside map coverage. No results must not imply that no places exist there.
- [ ] Ask Siri to search Zimfo for a topic. On iOS 27, verify a real result list opens, its preview shows local prose, and **Use in conversation** carries the selected source. Record the utterance Siri actually recognizes.
- [ ] Invoke content actions from the lock screen. Protected article/place content must wait for the required authentication.

## Record failures

The September 13 attempt on iOS 26.2 did not pass: a combined “ask zimfo tell me about Albert Einstein” utterance produced location/contact handling. The correction removes the ambiguous bare trigger and adds intent logging; it still needs a fresh voice test. After installing the corrected build, find `[Siri] action=...` records (with an invocation ID) in the debug pane or Settings → Past Logs. Run the question directly in Shortcuts and a nearby action separately to establish which action names appear. A `parameter_exit` can be Siri's prompt/resume flow, not an archive failure.

Follow-up logs identified the actual phone OS as **26.6.2** and confirmed that the two-turn voice request reached the question action. It then hit the old HTML-size guard. Build `202609132225` fixes that limit: a direct-action Einstein test on the phone returned an attributed answer in 573 ms. Retest the same two-turn Siri request to confirm the system speaks the result; that final voice presentation is still unverified.

Build `202609132305` is installed and its direct-intent phone probe verified automatic Putin article selection and a birthplace excerpt in 498 ms. Retest the spoken birthplace question on this build; this probe did not exercise Siri speech recognition.

The user also confirmed receiving a birthplace answer after manually choosing Vladimir Putin. A spontaneous “where did he go to school” follow-up instead produced Siri’s contact-card school-address response. Mark spontaneous conversational follow-up as **failed on this iOS 26.6.2 trial**, not merely awaiting testing. The explicit Article-chaining checks above are separate and still require a device run.

For each failure record: checklist row, exact utterance/typed input, expected and actual result, whether Siri reached Zimfo, selected article/place, archive edition, connection state, warm/cold launch, elapsed time, and screenshot or relevant app log. Do not mark spontaneous voice follow-ups or offline Siri invocation as supported until their device checks pass.


## Current article experiment after the OS update

See [overnight current-article work](SIRI_CURRENT_ARTICLE_WORK.md) for build and validation status. This feature exposes one active source document through onscreen context, without indexing the archive.

September 14: verified the phone on **iOS 27.0 (24A437)** and installed **Zimfo build 202609140645**. App launch and installed version are confirmed. The initial monitor's apparent exit report was not corroborated by the running process or memory report; its query-error handling has been corrected. The first school follow-up **failed**: Siri returned the contact-card school-address response. Zimfo logged an in-memory source publication, but no current-article query or export. Backgrounding and later republication also occurred; their cause is not established. The diagnostic build adds activity-association and context-revocation events to distinguish these stages.

The user clarified that Siri displayed the contact-card response in a line over the Zimfo UI. Diagnostic build **202609141925** is now installed; launch and the 45-second process watch passed. This build adds logging without changing the source lifetime or claiming to fix Siri routing.

On that diagnostic build, the school question again produced the Contacts response. “Summarize this article” instead prompted to use ChatGPT. Phone logs confirm `entity_associated` at 19:23:35Z, with no query/export or subsequent revocation in the retrieved log. Activity attachment is verified; Siri document consumption remains unverified.

- [ ] Verify the separate **Settings → Siri → Try Siri AI (Beta)** opt-in documented in Apple's [iOS 27 guide](https://support.apple.com/en-euro/guide/iphone/aside/glos65nenlkk/27/ios/27), then repeat the current-article test. OS version alone does not establish that this is enabled.

- [ ] On the diagnostic build, keep the completed Putin discussion visible and ask **“Summarize this article”**. Correlate `current_article_activity entity_associated`, visibility/revocation events, `current_article_query`, and `current_article_export`. Activity association alone is not proof Siri consumed the source.
- [ ] Open/discuss Vladimir Putin in Zimfo. Verify the visible “Discussing” article title. Invoke Siri and ask “Where did he go to school?” Record whether `current_article_query` and `current_article_export` appear in Past Logs.
- [ ] Ask a second question about that same document. Distinguish source-backed Siri text export from Siri's own knowledge/web answer; a plausible answer alone is insufficient.
- [ ] Switch to a different article, then repeat a pronoun question. Siri must use the new document, ask for clarification, or fail; never silently export the old one.
- [ ] If two enabled archives contain the same title, explicitly choose the other archive's article card. The exported document must identify that archive and its exact selected article.
- [ ] While discussing one article, choose an unavailable article or send a stale explicit source handoff. The error must not make the previous article available as the current document again.
- [ ] Read another article or request directions. The old discussion document must no longer export. Narration currently clears this context without publishing a replacement document.
- [ ] Reset the conversation, open Settings, or close the preview, then repeat. Prior current-document IDs must not export source text.
- [ ] Repeat with airplane mode and Wi-Fi off; record Siri activation/routing and local content export separately.
- [ ] Try an article beyond the text-transfer limit. It must refuse whole-document export rather than silently provide a truncated document as complete.

The school follow-up failed in the first iOS 27 trial; the remaining checks are unverified. Siri AI readiness on this phone also needs confirmation independently of its OS version. The Mac host is still on 26.6.2; automated export/lifecycle tests are not proof of version-27 conversational behavior.
