# Zimfo App Privacy disclosure

Last audited: July 16, 2026

This document maps the current Zimfo Apple-platform code to the questions in
App Store Connect. It is an engineering audit, not legal advice. Recheck the
answers whenever telemetry, accounts, advertising, cloud AI, or support-report
features change.

## Recommended release configuration

The developer-only GitHub Gist debug-report uploader is now compiled only
under `#if DEBUG`. These answers apply only to a newly archived Release build
containing that change; an older uploaded build must be replaced or disclosed
as described in **Resolved source blocker: debug reports** below.

In App Store Connect, answer:

> **Do you or your third-party partners collect data from this app?**
>
> **Yes, we collect data from this app.**

Do not answer “No.” The app initializes Google Analytics for Firebase,
Firebase Crashlytics, and (on iOS) Firebase Performance Monitoring at launch.
Analytics collection is explicitly forced on in
[`AppTelemetry.configure()`](../ios/MCPZimChat/Chat/AppTelemetry.swift).

### Copy-ready App Store Connect selections

| App Store Connect data type | Collected? | Linked to the user? | Used for tracking? | Purposes to select | Why Zimfo must disclose it |
| --- | --- | --- | --- | --- | --- |
| **Location → Coarse Location** | **Yes** | **Yes** | **No** | **Analytics; App Functionality** | Google Analytics derives general geography from masked IP addresses. Firebase Performance Monitoring also uses the request IP to provide country-level performance segmentation. Because Firebase associates events with app-instance, installation, session, or device identifiers, use Apple's conservative **linked** answer. This is distinct from Zimfo's GPS location, which remains on device. |
| **Identifiers → Device ID** | **Yes** | **Yes** | **No** | **Analytics; App Functionality** | Analytics creates an app-instance ID and, when IDFA is unavailable, collects Apple's vendor identifier (IDFV). Crashlytics and Performance use Crashlytics/Firebase installation IDs to associate reports and compute installation-level metrics. Zimfo has no named account system, does not set an Analytics user ID, does not request ATT, and does not link an advertising SDK, but Apple treats linkage to a device or other identifying details as linked to the user. |
| **Usage Data → Product Interaction** | **Yes** | **Yes** | **No** | **Analytics** | Analytics automatically records app lifecycle/session interaction. Zimfo also sends categorical `query_started` and `query_completed` events: query class, route/tool category, model ID, ZIM kind/variant/count, duration, response length, and cancellation status. No prompt or article text is sent. Firebase Sessions also records foreground/background timing to group performance events by session. These records are associated with Firebase/app-instance identifiers. |
| **Diagnostics → Crash Data** | **Yes** | **Yes** | **No** | **App Functionality** | Crashlytics collects crash stack traces, exception/application state, device model, OS, app version, and related crash information so crashes can be fixed. Reports are associated with Crashlytics/Firebase installation identifiers. |
| **Diagnostics → Performance Data** | **Yes** | **Yes** | **No** | **Analytics; App Functionality** | On iOS, Firebase Performance collects launch, foreground/background, screen-rendering and HTTP request timing plus CPU/memory and device/app attributes, associated with installation/session identifiers. Zimfo adds categorical query traces and timing/count metrics. Performance Monitoring is not linked into the macOS target, but App Store Connect answers are app-level and must include the most data-collecting platform. |
| **Diagnostics → Other Diagnostic Data** | **Yes** | **Yes** | **No** | **Analytics; App Functionality** | Crashlytics receives Zimfo's bounded custom keys/logs for the last query class, route, ZIM-kind mix, and duration. Firebase Installations, Sessions, and Google Data Transport also collect SDK/app-quality metadata associated with installation/session identifiers, such as SDK state, network type, and dropped-event counts. |

For every listed type, answer **No** to “Is this data used for tracking?”
Zimfo has no ad SDK, the Firebase plists have advertising disabled, and the
project does not use App Tracking Transparency, IDFA, Google Ads linking, a
data broker, or cross-company advertising measurement. Revisit this answer if
any of those facts change.

For every listed type, use the conservative answer **linked to the user:
Yes**. Apple defines linkage broadly enough to include association with a
device or other identifying details, not only a named account. Firebase uses
app-instance, installation, session, and device identifiers to associate these
records. The app has no named user accounts and does not set an Analytics user
ID, but that is not enough to justify answering “not linked.” Do not enable
Google Signals, Analytics user IDs, Google Ads linking, or other
identity/data-sharing features without re-auditing both “linked” and
“tracking.”

## Data types not collected by the release configuration

Do **not** select the following types based on the current normal app flow:

| Data type | Why it is not collected |
| --- | --- |
| **Precise Location** | Core Location coordinates are used locally for nearby-place lookup and offline routing. `AppTelemetry` has no location parameter and no coordinates are sent to Firebase. Firebase's IP-derived country/region is disclosed separately as **Coarse Location**. |
| **Search History** | User query text, search terms, article titles, and conversation text are not accepted by `AppTelemetry`. Only a bounded query class such as `factoid`, `topical`, or `navigational` is sent. |
| **Other User Content** | Prompts, model answers, article excerpts, and tool results stay on device in ordinary use. This answer is no longer correct if the GitHub Gist debug reporter ships; see below. |
| **Audio Data** | Microphone audio is consumed by on-device speech recognition and is not included in telemetry or uploaded. The current legacy recognizer sets `requiresOnDeviceRecognition = true`. Voice transcripts become local chat input and are not sent to Firebase. |
| **Browsing History** | Zimfo does not send the articles or sections read to Firebase. Its categorical ZIM-kind telemetry is product interaction, not browsing history. |
| **Files and Documents / Other Data** | Local ZIM contents and filenames are not sent. Telemetry derives only bounded categories: `wikipedia`, `mdwiki`, or `streetzim`; Wikipedia variant (`nopic`, `maxi`, `mini`, or `unknown`); and a file count. |
| **Contact Info, Contacts, Health & Fitness, Financial Info, Purchases, Sensitive Info, Photos or Videos** | The audited build neither requests nor transmits these types. |

“Processed only on device” is the important distinction. Apple says on-device
data is not “collected” for App Privacy answers. The privacy policy should say
that **user content and GPS location stay on device**, not that **the app
collects no data** or that **nothing ever leaves the device**.

## What the app actually sends

The app-defined Firebase payload is deliberately categorical. It includes:

- query type (`factoid`, `topical`, `navigational`, and similar bounded values);
- model identifier;
- loaded-library mix, Wikipedia variant, ZIM count, and whether a StreetZIM is
  present;
- route and primary tool category;
- ZIM-kind mix used by tools;
- total, first-response, and tool time; tool count; response character count;
  cancellation status;
- categorical Crashlytics keys and logs derived from the same fields.

It does not accept raw query text, article titles, article paths, coordinates,
ZIM filenames, file contents, audio, or transcripts. This protection is
visible in the API and comments in
[`AppTelemetry.swift`](../ios/MCPZimChat/Chat/AppTelemetry.swift).

Third-party SDK collection is broader than the custom payload:

- Google Analytics automatically assigns an app-instance identifier, collects
  the IDFV when IDFA is unavailable, measures lifecycle/session events, and
  derives approximate geography from masked IP addresses.
- Crashlytics collects crash/application/device state and installation IDs.
- Firebase Performance Monitoring collects application, rendering, network,
  CPU/memory, device, OS, and country-level performance data. It monitors HTTP
  URLs without query parameters and aggregates their paths into patterns; this
  can reveal which model-download or content-host endpoint is being used, but
  it does not see offline `zim://` reads or local ZIM file contents.

## Resolved source blocker: debug reports

The debug tooling can store a GitHub PAT in `UserDefaults` and upload a JSON
report as a secret GitHub Gist. The JSON contains the full conversation, tool
arguments, tool-result previews, and debug log, which can contain raw queries,
article/ZIM filenames, and precise GPS coordinates. The source itself says:

> `personal-dev only; don't ship the app publicly with this in place.`

Relevant code:

- [`DebugReport.swift`](../ios/MCPZimChat/Chat/DebugReport.swift)
- [`DebugPane.swift`](../ios/MCPZimChat/Views/DebugPane.swift)
- [`LibraryView.swift`](../ios/MCPZimChat/Views/LibraryView.swift)

**Status:** the PAT field, Report action, and Gist network transport have been
gated under `#if DEBUG` in the working tree. Local log viewing and copying
remain available. Archive and upload a fresh Release build before open beta,
then verify that the Release binary does not contain `api.github.com/gists`.

If the Gist uploader ships, the table above is insufficient. Conservatively
add at least:

| Additional data type | Linked? | Tracking? | Purpose |
| --- | --- | --- | --- |
| **User Content → Customer Support** | **Yes** (stored under the user's GitHub account) | **No** | **App Functionality** |
| **User Content → Other User Content** | **Yes** | **No** | **App Functionality** |
| **Location → Precise Location** | **Yes** | **No** | **App Functionality** |

The privacy policy would also need to explain exactly what the Report button
uploads, that GitHub stores it, its retention/deletion path, and that chat
content or logs may contain sensitive information. The existing UI does not
show the GitHub account name or a data preview immediately before upload, so
do not assume Apple's narrow “optional disclosure” exception applies.

## Privacy-policy and implementation risks

1. **Never claim “Zimfo collects no data.”** Firebase collection is enabled at
   launch. A safe headline is: “Your questions, voice, GPS coordinates, and
   offline library stay on your device. Zimfo collects limited pseudonymous
   analytics and diagnostics through Firebase.”
2. **Qualify the location statement.** Precise GPS stays local, while Firebase
   derives coarse country/region information from network IP addresses.
3. **Analytics is forced on with no in-app consent or withdrawal control.**
   `Analytics.setAnalyticsCollectionEnabled(true)` overrides the legacy plist
   flag, and there is no privacy/telemetry toggle. Apple's Review Guideline
   5.1.1 says apps collecting user or usage data must secure consent and offer
   an understandable way to withdraw it. Before public review, add a first-run
   analytics choice and a persistent Settings toggle, or obtain counsel that
   the proposed flow is sufficient. Crash reporting should have a separately
   documented control if it remains automatic.
4. **A privacy policy must be reachable inside the app.** Apple requires the
   App Store Connect privacy-policy URL and an easily accessible in-app link.
   Confirm the final build has the in-app link before review.
5. **Describe retention and deletion honestly.** Firebase documents 90-day
   retention before deletion begins for Crashlytics identifiers/reports;
   Performance retains IP-associated events for 30 days and installation-
   associated/de-identified performance data for 60 days before deletion
   begins. Analytics retention depends on the Firebase/GA4 property setting.
6. **Do not enable optional Google features silently.** Google Signals, user
   IDs, AdSupport/IDFA, Google Ads linking, or broader Google data-sharing can
   expand the linkage and require changing the current “no tracking” answers.
7. **Firebase Performance sees outbound URL paths.** Query parameters are
   excluded and local ZIM access is not HTTP, but model download paths and
   other outbound endpoints can be aggregated in the Firebase console. The
   policy should not promise that Firebase sees only the hand-authored query
   trace fields.

## Official references

- Apple, [App Privacy Details](https://developer.apple.com/app-store/app-privacy-details/)
  — definitions of collection, linked data, tracking, data types, and purposes;
  also confirms that data processed only on device is not collected.
- Apple, [Manage app privacy](https://developer.apple.com/help/app-store-connect/manage-app-information/manage-app-privacy)
  — App Store Connect workflow and requirement to include third-party SDKs.
- Apple, [App Review Guidelines, section 5.1.1](https://developer.apple.com/app-store/review/guidelines/#privacy)
  — privacy-policy, consent, withdrawal, data minimization, and permission
  requirements.
- Firebase, [Prepare for Apple's App Store data disclosure requirements](https://firebase.google.com/docs/ios/app-store-data-collection)
  — current SDK-by-SDK collection for Analytics, Crashlytics, Performance,
  Sessions, Installations, and Google Data Transport.
- Google Analytics, [Prepare for Apple's App Store data-disclosure requirements](https://support.google.com/analytics/answer/10285841)
  — app-instance IDs, lifecycle events, masked-IP geography, and optional
  advertising/identity features.
- Firebase Help, [Data collection](https://support.google.com/firebase/answer/6318039)
  — Analytics app-instance ID and the iOS IDFA/IDFV fallback behavior.
- Firebase, [Privacy and Security](https://firebase.google.com/support/privacy)
  — Crashlytics and Performance identifiers and retention periods.
- Firebase, [Performance Monitoring](https://firebase.google.com/docs/perf-mon)
  and [HTTP/S network traces](https://firebase.google.com/docs/perf-mon/network-traces)
  — automatic performance collection and URL-pattern handling.
