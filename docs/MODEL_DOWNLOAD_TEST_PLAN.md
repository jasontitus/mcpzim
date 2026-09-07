# Model download & bootstrap — manual device test plan

The model-download path runs through the same background `URLSession`
downloader as the ZIM catalog (`ZimDownloadManager`), so it inherits
suspension/termination survival, resume-data, pause/resume/retry, and
staged size+SHA-256 verification. These are the manual on-device checks the
automated suite can't cover. Run against a signed TestFlight build on a
real iPhone (no simulator) — the Metal/llama.cpp and iOS process-cap
behavior only manifests on device.

Devices: a **6 GB base iPhone** (snug) and an **8 GB+ Pro iPhone** (balanced),
plus a second phone that will share (the "friend" device).

---

## A. Fresh install — offline friend bootstrap (no internet)

The headline path: a brand-new phone with **no network** gets Wikipedia,
maps, and an AI model from a friend over peer-to-peer Wi-Fi.

1. **Friend device**: Settings → Library → Nearby Sharing → **Share my library**
   ON. Confirm the toggle says **"Include chat models (N GB)"** with the total
   size of every downloaded model. (The friend must have at least one model
   fully downloaded, e.g. LFM2.5 or Gemma 3 4B FT.)
2. **Fresh install**: put the recipient phone in **Airplane Mode**. First
   launch → Offline Setup → **Copy from a friend nearby**.
3. Confirm the swarm discovers the friend and transfers ZIMs + model + voice.
4. After the transfer:
   - The library shows the received Wikipedia/maps.
   - A model is adopted and auto-selected **only if it fits this phone**.
     - On a snug phone with a friend's LFM2.5 shared, it should **not** crash;
       if a smaller Gemma was also shared it should auto-select that.
     - On a snug phone where the only shared model is LFM2.5, the app should
       show a clear "needs ~3700 MB but this device can't hold it" error,
       **not** a crash.
5. Type a question. It should answer entirely offline.
6. **Voice**: the mic button should use the shared Kokoro/Supertonic voice
   (installed from the share), no download.

**Pass:** no internet used, no crash, model + content answer offline.

---

## B. Model download interruptions (on the device downloading the model)

Use the default-capacity model (on a snug phone that's Gemma 3 4B FT). Fresh
install, download a model, then interrupt it in each of these ways. The
download must recover without a crash.

1. **Airplane mode mid-download** — start a model download, toggle Airplane
   Mode. The row should flip to **"Waiting for network…"**. Re-enable Wi-Fi;
   the download must resume on its own. No stall-failure after the
   5-minute timeout.
2. **Switch apps** — start a download, home-screen to another app for ~2 min,
   return. The download must have continued (background session) and show
   progress, not restart from 0.
3. **Screen lock** — lock the screen, wait ~5 min, unlock. Download must have
   continued.
4. **Relaunch the app** (background completion) — start a download, swipe the
   app away, relaunch. The download must resume from where it left off (not
   restart), and on completion the model loads.
5. **Force-quit cancels transfer (documented limitation)** — start a
   download, force-quit the app, reopen. The download is cancelled by iOS;
   the app must **recover** by finding the partial state and offering Resume,
   and never claim "uninterrupted downloading."
6. **Expired/download-link failure** — with a model that has a bad/expired
   URL (or by blocking the CDN), the download must surface `.failed` with a
   Retry action, not hang at 0%.
7. **Insufficient storage** — with the volume nearly full, start a model
   download. It must fail **before** starting with "needs N GB but only M GB
   free," and must not delete or disturb an already-working model.

**Pass:** every interruption recovers cleanly; no crash; a working model is
never lost.

---

## C. Crash-at-100% regression

The original bug: the default model download reached 100% and the app
crashed (load OOM right after the download finished).

1. On a **6 GB phone**, fresh install (or reset model selection). Confirm the
   app **defaults to Gemma 3 4B FT** (not LFM2.5) and that the download +
   load complete without crashing.
2. Open the model picker. Confirm **Gemma 3 4B FT is marked "Recommended"**,
   LFM2.5 is visible, and **Bonsai 27B is flagged "Too large for this
   device."**
3. Manually pick **Bonsai 27B**. It must **not** crash — it should show the
   "needs ~5500 MB but this device can't hold it" error.
4. Manually pick **LFM2.5** on the 6 GB phone. Same — clear error, no crash.
5. On an **8 GB+ phone**, confirm the default is **LFM2.5** and it loads
   without crashing.

**Pass:** the default fits the phone; a too-big manual pick fails gracefully
instead of aborting.

---

## D. Model sharing (ZIM-like)

1. On the friend device with 2+ models downloaded, confirm Nearby Sharing
   includes **every** downloaded model (toggle shows the summed size).
2. Receive on a second phone and confirm **both** models are adopted into
   the cache (check the model picker — both appear without a download).
3. Confirm a shared model is **re-shareable** onward (seed chain A → B → C).

**Pass:** all downloaded models transfer; recipient gets a working model with
zero internet.

---

## E. Notes

- `GemmaToolEmissionTests` fails on the Mac with a llama.cpp Metal cleanup
  abort (`ggml_metal_rsets_free` / `GGML_ASSERT([rsets->data count] == 0)`)
  during test-process teardown — a test-harness issue, not the on-device
  download/load path. It doesn't affect the iOS app.
- The automated coverage (Mac XCTest) pins the TaskLabel round-trip, SHA-256
  helper, cache-validity gate, storage-gate message, capacity-aware default,
  and model-adoption. It runs via:
  ```sh
  xcodebuild -project ios/MCPZimChat.xcodeproj -scheme MCPZimChatMacTests \
    -destination 'platform=macOS' -derivedDataPath ios/build-mac-bonsai \
    test -only-testing:MCPZimChatMacTests/ModelDownloadTests \
         -only-testing:MCPZimChatMacTests/ModelSharingTests
  ```
