# Zimfo TestFlight open beta submission

This is the copy-and-paste submission sheet for the iOS app record whose bundle
ID is `com.tiltastech.zimfo`. It separates the information required for an
external TestFlight beta from metadata that is only required for a public App
Store release.

Before submitting, replace every value surrounded by `[SQUARE BRACKETS]` and
verify that all three public URLs return HTTP 200 without authentication.

## Public URLs

Use these values in both the TestFlight localization and, later, the App Store
version metadata:

| Field | Value |
| --- | --- |
| Marketing URL | `https://tiltastech-zimfo.web.app/` |
| Privacy Policy URL | `https://tiltastech-zimfo.web.app/privacy` |
| Support URL | `https://tiltastech-zimfo.web.app/support` |

The Marketing URL is visible to TestFlight testers and should explain Zimfo,
its device/storage requirements, and how offline setup works. The Privacy URL
must state the app's actual Firebase Analytics, Crashlytics, and Performance
Monitoring practices. The Support URL must contain a working support email and
real contact information; Apple says a release Support URL must lead to actual
contact information for app problems, feedback, and feature requests. See
[Beta App Localization fields](https://developer.apple.com/documentation/appstoreconnectapi/betaapplocalization/attributes-data.dictionary)
and [Platform version information](https://developer.apple.com/help/app-store-connect/reference/app-information/platform-version-information).

Recommended additional routes:

- `https://tiltastech-zimfo.web.app/beta` — beta installation and first-run
  guide; add the TestFlight public link after Apple approves the build.
- `https://tiltastech-zimfo.web.app/licenses` — Wikipedia, OpenStreetMap,
  model, TTS, framework, and other third-party notices.
- `https://tiltastech-zimfo.web.app/accessibility` — accessibility support and
  contact path. This is useful for the later App Store accessibility
  declaration, but is not required for the first external beta.

## TestFlight > Test Information (English, U.S.)

Apple requires a beta description and feedback email for external testing;
the marketing and privacy URLs are also visible in TestFlight. This beta
metadata may differ from the later App Store metadata. See Apple's
[Provide test information](https://developer.apple.com/help/app-store-connect/test-a-beta-version/provide-test-information/)
and [Beta App Localizations](https://developer.apple.com/documentation/appstoreconnectapi/beta-app-localizations).

### Beta App Description

```text
Zimfo is a private, offline AI guide to Wikipedia and StreetZIM maps. Download an on-device language model, add Wikipedia and regional map ZIM files, then ask questions, discuss articles, find nearby places, and plan routes without a network connection. Voice input, speech playback, retrieval, and AI inference run on the device. This beta focuses on natural conversation, source-grounded answers, hands-free use, offline setup, and performance on supported iPhones and iPads.
```

### Feedback Email

```text
support@tiltastech.com
```

Create and test that mailbox or forwarding address before entering it. Apple
uses it as the TestFlight invitation reply-to address as well as the address
testers can use from TestFlight.

### Marketing URL

```text
https://tiltastech-zimfo.web.app/
```

### Privacy Policy URL

```text
https://tiltastech-zimfo.web.app/privacy
```

### Invitation Experience

Leave **App Information** off until an approved App Store version has useful
screenshots and a category. It does not help the first beta because Apple only
pulls this material from the latest version already Ready for Distribution.

## Build > What to Test

```text
Please focus on first-run setup and real multi-turn conversation.

1. Download the recommended on-device model and verify that an interrupted download resumes.
2. In Library > Offline Setup, download and add a no-picture Wikipedia ZIM. If desired, add a state, region, or country StreetZIM from streetzim.web.app.
3. Ask factual and explanatory questions, start an article discussion with “Let’s talk about Mongolia,” and try follow-up questions that depend on the selected topic.
4. Explicitly choose a source, for example: “Use the Wikipedia article on Santa Rosa, California and tell me about the 1906 earthquake.”
5. With a StreetZIM loaded and Location allowed, try nearby-place search and offline routing.
6. Try both typed chat and hands-free voice conversation, including stopping or interrupting speech.

Please report slow time to first response, incorrect article or section selection, topic drift, microphone or playback problems, unusually low volume, excessive heat or memory pressure, crashes, and confusing setup. Prompts, transcripts, microphone audio, exact article titles, precise location, and ZIM filenames are not uploaded by Zimfo. Privacy-safe query categories, library types, timing, performance diagnostics, and crash data are sent through Firebase to improve the beta.

Initial setup downloads several gigabytes. Use Wi-Fi, external power, and at least 8 GB of free device storage. The complete English Wikipedia option is much larger; Popular English or Simple English is recommended for phone testing.
```

## TestFlight App Review Information

Apple requires beta-review contact information, reviewer notes, and demo
credentials only when sign-in is required. This contact information is not the
same as the public App Store contact metadata. See
[TestFlight test information](https://developer.apple.com/help/glossary/testflight-test-information/)
and [Beta App Review Detail](https://developer.apple.com/documentation/appstoreconnectapi/beta-app-review-detail).

### Contact Information

```text
First name: [REVIEW CONTACT FIRST NAME]
Last name: [REVIEW CONTACT LAST NAME]
Phone: [REVIEW CONTACT PHONE WITH COUNTRY CODE]
Email: support@tiltastech.com
```

Use a person who can answer Apple during the review window. Do not use a phone
number or mailbox that is unattended.

### Sign-in Required

```text
No
```

Zimfo has no account and no login. Do not enter a demo username or password.

### Review Notes

```text
Zimfo is an offline Wikipedia and mapping assistant. No account or sign-in is required. The app does not include advertising, purchases, subscriptions, or user-to-user communication.

FIRST-RUN REVIEW STEPS
1. Connect the test device to Wi-Fi and allow several gigabytes of free storage.
2. Launch Zimfo. The default Bonsai 27B on-device model downloads approximately 3.8 GB on first use. The download is resumable; please keep the app open until it completes.
3. Open Library > Offline Setup. Choose Simple English Wikipedia (approximately 937 MB), wait for the Safari download to finish, return to Zimfo, tap “Add downloaded library or map,” and select the .zim file from Files.
4. Type “Tell me about Mongolia,” or ask another Wikipedia question. Microphone, Speech Recognition, and Location permissions are optional for typed Wikipedia questions.
5. To test voice, allow Microphone and Speech Recognition, then tap the microphone. To test nearby places or routes, also allow Location and add an appropriate StreetZIM from https://streetzim.web.app/.

The GGUF model weights, TTS assets, and ZIM archives downloaded by the app are data, not executable code. Inference and searches run in the shipped native app runtime. User prompts, transcripts, microphone audio, exact article titles, precise location, and ZIM filenames remain on device. Firebase receives bounded query categories, model/library classes, timing metrics, and crash/performance diagnostics; it does not receive query text.

If the large first-run download prevents review, please contact support@tiltastech.com and we will provide assistance immediately.
```

The several-gigabyte cold start is a review risk. Before submission, perform a
clean-install run on a non-development device and confirm that the exact steps
above work. A future small reviewer/demo ZIM would reduce review time, but the
review notes must never claim one exists until it is actually public and tested.

## Encryption / export compliance

The iOS app currently contains:

```xml
<key>ITSAppUsesNonExemptEncryption</key>
<false/>
```

Therefore, if App Store Connect asks whether the build uses **non-exempt
encryption**, answer:

```text
No
```

The intended declaration is that Zimfo does not implement or ship non-exempt
cryptography; ordinary HTTPS and Apple operating-system encryption are exempt.
No export-compliance document should be required under that declaration. Do
not change this answer if proprietary cryptography, a VPN, or other non-exempt
encryption is later added. Apple requires every developer to make the actual
determination; see [Overview of export compliance](https://developer.apple.com/help/app-store-connect/manage-app-information/overview-of-export-compliance/)
and [Determine and upload app encryption documentation](https://developer.apple.com/help/app-store-connect/manage-app-information/determine-and-upload-app-encryption-documentation).

## Content rights

For **Does your app contain, show, or access third-party content?**, select:

```text
Yes, and I have the necessary rights or permission to use it.
```

Zimfo accesses user-downloaded Wikipedia/Kiwix and OpenStreetMap/StreetZIM
content and downloads third-party language-model and speech-model weights. The
selection is only defensible after all applicable licenses and attribution
requirements have been checked and honored. Before external review:

- publish the third-party notices at `/licenses`;
- expose attribution/notices from inside the app;
- verify the exact licenses for every selectable model and TTS voice/model;
- retain Wikipedia CC BY-SA/GFDL and OpenStreetMap ODbL attribution as
  applicable; and
- confirm that redistribution or app-directed downloading is permitted.

Apple requires apps that access third-party content to have the necessary
rights or other legal permission. See [App information: Content Rights](https://developer.apple.com/help/app-store-connect/reference/app-information/app-information/).
This is a release gate, not a box to check speculatively.

## Age rating recommendation

Do not mark Zimfo as **Made for Kids**. It has third-party analytics and can
retrieve unfiltered encyclopedia material on war, crime, drugs, sexuality,
medical treatment, and other mature subjects.

Recommended questionnaire approach:

- Parental controls: **No**
- Age assurance: **No**
- Unrestricted web access: **No** — Zimfo opens specific download links in
  Safari; it is not an embedded general web browser.
- User-generated content: **No** — private prompts are not distributed.
- Social media: **No**
- Messaging and chat: **No** — the user talks to an on-device model, not to
  another user.
- Advertising: **No**
- Gambling, contests, and loot boxes: **None/No**
- Disclose encyclopedia content descriptors honestly. At minimum, consider
  **Infrequent** for profanity, horror/fear, alcohol/tobacco/drug references,
  medical/treatment information, mature/suggestive themes, sexual content,
  realistic violence, and weapons.
- For the open beta, use **Override to Higher Age Rating: 16+** (shown as the
  corresponding older-system rating where applicable) until content filtering
  and a broader content audit justify a lower rating.

Apple's definitions make clear that messaging means communication between
users, UGC means broad distribution of user-created content, and unrestricted
web access means free browsing inside the app. See [Age rating categories and
definitions](https://developer.apple.com/help/app-store-connect/reference/app-information/age-ratings-values-and-definitions/)
and [Set an app age rating](https://developer.apple.com/help/app-store-connect/manage-app-information/set-an-app-age-rating).
The final frequency answers must reflect the archives and model behavior that
actually ship; this recommendation is not a substitute for that audit.

## App privacy answers

Do **not** select “No, we do not collect data from this app.” Firebase
Analytics, Crashlytics, and Performance Monitoring are enabled. Apple requires
the declaration to cover third-party SDKs as well as first-party code. See
[Manage app privacy](https://developer.apple.com/help/app-store-connect/manage-app-information/manage-app-privacy)
and [App Privacy Details](https://developer.apple.com/app-store/app-privacy-details/).

Based on the current `AppTelemetry.swift`, use this conservative declaration
for the beta build, then re-check it against the Firebase SDK privacy manifests
embedded in the exact archive:

| Data type | Collected | Linked to user/device | Tracking | Purpose |
| --- | --- | --- | --- | --- |
| Location > Coarse Location | Yes | Yes | No | Analytics; App Functionality |
| Identifiers > Device ID | Yes | Yes | No | Analytics; App Functionality |
| Usage Data > Product Interaction | Yes | Yes | No | Analytics |
| Usage Data > Other Usage Data | Yes | Yes | No | Analytics |
| Diagnostics > Crash Data | Yes | Yes | No | App Functionality; Analytics |
| Diagnostics > Performance Data | Yes | Yes | No | Analytics; App Functionality |
| Diagnostics > Other Diagnostic Data | Yes | Yes | No | Analytics; App Functionality |

Do not declare these as collected by Zimfo under the current implementation:

- precise or coarse location;
- audio data or speech transcripts;
- search history, browsing history, or query text;
- other user content or exact article titles;
- ZIM filenames; or
- contact, health, financial, or purchase information.

“Tracking” remains **No** because the app does not link its telemetry with
third-party data for advertising/advertising measurement and does not share it
with a data broker. If Firebase settings, SDK behavior, event parameters, or
data uses change, update both App Store Connect and `/privacy` before shipping.

## External group and public-link process

Apple's current flow requires an internal group before an external group. See
[Invite external testers](https://developer.apple.com/help/app-store-connect/test-a-beta-version/invite-external-testers)
and the [TestFlight overview](https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/).

1. In **TestFlight**, create an internal group named `Zimfo Internal` and add
   at least the account holder/developer as an internal tester.
2. Install the processed build through TestFlight and complete a clean-device
   smoke test, including the model download, ZIM import, typed query,
   microphone, background/foreground, and relaunch.
3. Create an external group named `Zimfo Open Beta`.
4. Add the build, paste **What to Test**, and complete Test Information and
   Beta App Review Information.
5. Confirm export compliance, content rights, and the three live site URLs.
6. Submit the build for TestFlight App Review. Select **Automatically notify
   testers** only if existing external testers should receive it immediately
   upon approval.
7. After approval, select the external group and choose **Create Public Link**.
8. Choose **Filter by Criteria**. Start with iPhone/iPad devices actually
   verified to run the model and **iOS/iPadOS 18.0 or later**. A supported OS
   filter alone is not a substitute for verifying device memory.
9. Set the initial public-link tester limit to **100**. Increase it gradually
   after crash-free sessions, download completion, memory pressure, and support
   volume are understood. The maximum is 10,000 external testers per app.
10. Copy the approved public link to `/beta`, then share that page rather than
    scattering a raw TestFlight link. The public link can be disabled at any
    time and may be reshared by anyone.

### Limits to remember

- A TestFlight build is testable for **90 days**.
- The app can have up to **10,000 external testers** and **100 internal
  App Store Connect users**.
- The first external build requires a full TestFlight App Review; later builds
  for the same version may not require one.
- Only **one build of each version** can be in TestFlight App Review at a time.
- Apple permits up to **six TestFlight App Review build submissions in a
  24-hour period**.
- Builds uploaded as **TestFlight Internal Only** cannot be used for the public
  external group.

These limits and the public-link controls are documented in
[Invite external testers](https://developer.apple.com/help/app-store-connect/test-a-beta-version/invite-external-testers)
and [TestFlight overview](https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/).

## What is needed now versus App Store release

### Needed now for an external/open TestFlight beta

- [ ] Current Apple Developer agreements accepted and app record matches
  `com.tiltastech.zimfo`.
- [ ] Build uploaded normally (not TestFlight Internal Only), processed, and
  free of blocking compliance warnings.
- [ ] Build created with Xcode 26 or later and an iOS 26 SDK, as required for
  uploads since April 28, 2026. See [Upcoming Requirements](https://developer.apple.com/news/upcoming-requirements/).
- [ ] Clean-install smoke test completed through TestFlight.
- [ ] Marketing, privacy, and support sites publicly available.
- [ ] Beta App Description, Feedback Email, Marketing URL, and Privacy URL.
- [ ] Per-build What to Test text.
- [ ] Beta App Review contact, notes, and Sign-in Required = No.
- [ ] Export compliance answered; `ITSAppUsesNonExemptEncryption=false`
  present in the archived app.
- [ ] Content licenses/attribution verified and accessible.
- [ ] Privacy disclosure matches the exact Firebase-enabled archive.
- [ ] Updated age-rating questionnaire completed; not Made for Kids.
- [ ] Internal group created, then external group created and first build
  submitted for beta review.
- [ ] Public link created only after approval, initially capped at 100 testers.
- [ ] Crashlytics dSYMs uploaded for the release archive and a non-fatal/test
  event verified in Firebase.

### Needed only for a full public App Store release

- [ ] App Store version screenshots for every required device class. Screens
  must show the real app in use, not just a splash screen.
- [ ] App Store description, subtitle, keywords, promotional text, Support URL,
  copyright, and release setting.
- [ ] App Privacy responses published, not merely saved.
- [ ] Price (Free unless the business plan changes), countries/regions, and
  availability configured.
- [ ] Digital Services Act trader status declared. Apple says TestFlight-only
  distribution is not acting as a trader on the App Store; App Store release
  in the EU requires the appropriate status and, for traders, verified public
  contact details. See [DSA trader requirements](https://developer.apple.com/help/app-store-connect/manage-compliance-information/manage-european-union-digital-services-act-trader-requirements).
- [ ] Accessibility declarations tested per device family and published only
  for features the app actually satisfies.
- [ ] Full App Review contact/notes and the final release build selected.
- [ ] Final content, safety, license, model-output, and medical-information
  review completed.

Suggested future App Store metadata:

```text
Name: Zimfo
Subtitle: Offline AI knowledge & maps
Primary category: Reference
Secondary category: Navigation
Keywords: offline,wikipedia,encyclopedia,AI,maps,navigation,knowledge,voice,ZIM,travel
Copyright: 2026 [LEGAL OWNER NAME]
Release setting: Manual
```

Suggested promotional text:

```text
Talk with an on-device AI using offline Wikipedia and StreetZIM maps—ask questions, explore articles, find places, and plan routes without a connection.
```

Suggested App Store description:

```text
Zimfo turns offline knowledge and maps into a natural conversation.

Choose an on-device language model, add a no-picture Wikipedia archive and an optional StreetZIM for your state, region, or country, then ask questions without relying on a cloud AI service. Explore a topic through follow-up questions, explicitly choose a Wikipedia source, find nearby places, and plan routes from downloaded map data.

VOICE CONVERSATION
Talk hands-free and hear spoken responses. Speech recognition, retrieval, AI inference, and speech generation run on your device.

GROUNDED OFFLINE KNOWLEDGE
Answers can use the Wikipedia articles and sections in your own ZIM library, with visible source information that helps distinguish retrieved evidence from general model knowledge.

OFFLINE PLACES AND ROUTING
Add a StreetZIM for the area you need to search nearby places, view offline maps, and calculate routes without a network connection.

PRIVATE BY DESIGN
Prompts, transcripts, microphone audio, exact article titles, precise location, and ZIM filenames stay on your device. Zimfo uses privacy-safe analytics and diagnostics to understand feature performance and crashes; see the privacy policy for details.

INITIAL DOWNLOADS REQUIRED
The AI model and offline libraries are not included in the app download. Initial setup requires Wi-Fi, several gigabytes of free storage, and time to download the model and the Wikipedia or map archives you choose. Once setup is complete, core knowledge and map features work offline.

Zimfo can make mistakes. Verify important medical, legal, financial, safety, and navigation information with authoritative sources.
```

Apple requires screenshots, description, keywords, Support URL, copyright, and
other platform-version fields for an App Store release; the field definitions
and limits are in [Platform version information](https://developer.apple.com/help/app-store-connect/reference/app-information/platform-version-information).
