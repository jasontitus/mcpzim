# iOS app — status & hand-off

**What this doc is.** The iOS app was scaffolded from a Linux host (no Xcode,
no device), so everything compiles *in principle* but has not been exercised
on Apple hardware. This file lists exactly what's done, what's guessed, and
the fastest path to a running build on your Mac.

Git state: branch `claude/mcp-zim-server-1HOTI`, tip commit `b0e6d27`.
All iOS code lives under `ios/`; the Swift library it depends on lives under
`swift/`.

## Status by component

| File / area | Status | Notes |
| --- | --- | --- |
| `ios/project.yml` (XcodeGen spec) | ✅ written, ⚠️ not run | Needs `brew install xcodegen && xcodegen generate` on your Mac. First generate may surface `project.yml` typos I couldn't catch. |
| `ios/MCPZimChat/App/MCPZimChatApp.swift` | ✅ | SwiftUI `@main` entry. Trivial. |
| `ios/MCPZimChat/Info.plist` | ✅ | Includes `UIFileSharingEnabled` + `LSSupportsOpeningDocumentsInPlace` so iTunes file-sharing works for dropping ZIMs in. |
| `ios/MCPZimChat/Providers/ModelProvider.swift` | ✅ | Protocol + `ModelLoadState` enum. Stable. |
| `ios/MCPZimChat/Providers/MockProvider.swift` | ✅ | Canned responses; emits `<tool_call>` blocks for a couple of prompts so the tool loop is exercisable without weights. |
| `ios/MCPZimChat/Providers/Gemma4Provider.swift` | ⚠️ guessed API | **Known guesswork** — uses `LLMModelFactory.shared.loadContainer(...)`, `container.generate(text:parameters:)`, and a `.chunk/.info/.toolCall` switch I inferred from `MLXLMCommon`. Real API may differ. See "Likely rough edges" below. |
| `ios/MCPZimChat/Libzim/LibzimReader.swift` | ⚠️ bridge missing | Swift side is written; assumes an Obj-C++ class `ZimArchive` with a specific method surface. **The bridge itself (`LibzimBridge.mm` + header) is NOT written** — you need to author it after vendoring `CoreKiwix.xcframework`. |
| `ios/MCPZimChat/Chat/ChatSession.swift` | ✅ logic, ⚠️ prompt format | Tool loop is solid: scans the stream for `<tool_call>…</tool_call>`, dispatches through `MCPToolAdapter`, appends a synthetic tool turn, resumes. **Uses generic `<\|user\|>/<\|assistant\|>` tokens**, not Gemma 4's native `<start_of_turn>` template — quality will be degraded until we swap in `Gemma4PromptFormatter`. |
| `ios/MCPZimChat/Chat/Message.swift` | ✅ | Plain value types. |
| `ios/MCPZimChat/Views/*.swift` | ✅ written, ⚠️ not previewed | ChatView + MessageRow + ToolCallRow + ModelPickerView + LibraryView. Xcode SwiftUI Previews are included but haven't been run. |
| `swift/` (MCPZimKit) | ✅ | Pure-Swift SZRG v2 parser, A*, geocoder, ZimReader protocol, MCPToolAdapter. Tests written but not run (no Swift on the Linux host). |
| Tests | 🚫 none on iOS target yet | Suggest a first pass of unit tests on `ChatSession.extractToolCall(in:)` and a Preview-friendly fixture for `ToolCallRow`. |

## Likely rough edges (things to expect on first build)

1. **MLXLMCommon API drift.** `Gemma4Provider.generate(prompt:parameters:)`
   calls what I believe is `await container.generate(text:parameters:)` and
   pattern-matches `GenerationEvent` cases. The actual names in
   `mlx-swift-lm` 2.30.x may be different. The `#if canImport(Gemma4SwiftCore)
   && canImport(MLXLMCommon)` guard means the non-Gemma build path still
   works, so fix this *after* getting the Mock model running end-to-end.
   Quick triage: compile errors will point at lines ~59–80 of
   `Gemma4Provider.swift`; check `Sources/Gemma4Verify/main.swift` in the
   Swift-Gemma4-Core repo for a canonical call site.

2. **Gemma 4 chat template.** `ChatSession.runGenerationLoop()` concatenates
   turns with `<|user|>`, `<|assistant|>`, `<|tool|>` markers. Gemma 4's
   native template is `<start_of_turn>user…<end_of_turn>`. Two ways to fix:
   - Add `formatTranscript(_ messages: [(role: String, text: String)]) -> String`
     to `ModelProvider` with a generic default; override in
     `Gemma4Provider` using `Gemma4PromptFormatter.userTurn(_:)`.
   - Or inject a provider-supplied system prompt + only call
     `Gemma4PromptFormatter.userTurn` for the final user message and keep the
     tool-response turns as additional user turns prefixed with
     `<tool_response>...`.

3. **LibzimBridge is not written.** `LibzimReader.swift` (the
   `#if canImport(CoreKiwix)` branch) refers to a type `ZimArchive` with
   these methods, but I didn't write the Obj-C++ implementation:
   ```
   - (instancetype)initWithFileURL:(NSURL *)url error:(NSError **)error;
   - (BOOL)hasEntry:(NSString *)path;
   - (NSString * _Nullable)metadataValue:(NSString *)key;
   - (ZimEntryBridge * _Nullable)readEntryAtPath:(NSString *)path;
   - (ZimEntryBridge * _Nullable)readMainPage;
   - (NSArray<ZimSearchHitBridge *> *)searchFulltext:(NSString *)query limit:(int32_t)limit;
   - (NSArray<ZimSearchHitBridge *> *)suggestTitles:(NSString *)query limit:(int32_t)limit;
   @property (readonly) BOOL hasFulltextIndex;
   @property (readonly) BOOL hasTitleIndex;
   @property (readonly) int32_t articleCount;
   ```
   The Kiwix iOS app's `Model/Utilities/` directory
   (github.com/kiwix/kiwix-apple) has a ready-to-crib reference — look at
   how it calls `zim::Archive::getMetadata` from Obj-C++.

4. **Document picker semantics.** `LibraryView` uses `.fileImporter(...)`
   which in recent iOS returns security-scoped URLs, *not* copies. To
   reliably re-open a ZIM across launches you'll want to either:
   - Call `startAccessingSecurityScopedResource()` + copy the bytes into
     `Documents/`, or
   - Hold onto a bookmark (`URL.bookmarkData(...)`) and resolve it on
     each launch.
   The app currently assumes files already live in Documents; the importer
   is a convenience that will work on simulator but may fail silently on
   device without the above. Pick one strategy and wire it.

5. **`@ObservationIgnored` on `stateObservationTask`.** Used to keep the
   Task handle out of the `@Observable` graph. If you're on Xcode < 15.3 the
   attribute may not be available; either upgrade Xcode or store the Task
   in a separate non-observable holder.

6. **Selection picker keying.** `ModelPickerView` uses `\.id` as the picker
   key on `any ModelProvider`. Swift's existential picker support can be
   fussy about `Hashable` conformance on `any`; if the Picker complains,
   switch to keying by the `String` id directly (already done for the
   `Picker(selection: ...)` binding — just double-check the iteration).

## First-build checklist (run on your Mac)

```sh
# 0. One-time tools.
brew install xcodegen

# 1. Generate the Xcode project from project.yml.
cd ~/path/to/mcpzim/ios
xcodegen generate
open MCPZimChat.xcodeproj

# 2. First build target: SIMULATOR with the Mock model.
#    No xcframework, no Gemma. This validates that the scaffolding, SwiftUI,
#    and MCPZimKit wiring compile cleanly.
#    Expected outcome: app launches, model menu shows "Mock (scripted)" and
#    "Gemma 4 4B (4-bit) — not linked", tapping Mock → Load model → "ready",
#    sending a message gets a canned reply. The library view is empty and
#    no tool calls fire (no ZIMs).

# 3. Add your development team in Signing & Capabilities. Deploy to a
#    physical iPhone 14 Pro or newer — Gemma 4 won't load on older RAM.

# 4. Drop ZIMs into Documents (use Finder → device → Files → MCPZimChat).
#    Tap Library → Refresh. Expected outcome: LibzimReader throws
#    `.notLinked` for each file (because the xcframework isn't in yet);
#    that's expected.

# 5. Vendor CoreKiwix.xcframework.
#    Easiest: grab the latest Kiwix-apple release asset and drop into
#    ios/MCPZimChat/Frameworks/. Then:
#       - In project.yml, uncomment the `framework:` dependency line.
#       - Author LibzimBridge.mm (~150 lines; see item 3 above).
#       - Re-run xcodegen generate.
#    Expected outcome: library view shows your ZIMs with the correct kind
#    badge (Wikipedia / mdwiki / streetzim).

# 6. Add Swift-Gemma4-Core.
#    In Xcode: File → Add Package Dependencies →
#       https://github.com/yejingyang8963-byte/Swift-gemma4-core.git
#    version 0.1.0+. Add product Gemma4SwiftCore to the MCPZimChat target.
#    Fix any MLXLMCommon API drift in Gemma4Provider.swift (item 1 above).
#    Expected outcome: model menu shows "Gemma 4 4B (4-bit)" (no suffix);
#    tapping Load triggers the first ~1.5 GB weight download from HF Hub.

# 7. Fix the Gemma chat template (item 2 above) so answers read cleanly.

# 8. End-to-end smoke test:
#    - Load a streetzim ZIM for your area.
#    - Load Gemma 4.
#    - Ask: "Plan a driving route from A to B" (names should be in
#      the streetzim search index — check via the "What's in my library?"
#      prompt first).
#    - Expect a tool-trace chip showing `route_from_places` with the
#      resolved origin/destination and a coalesced road-segment list.
```

## What I'd do next, in order

1. Get Mock model working in the Simulator (steps 1–2). Smallest
   satisfying feedback loop.
2. Write `LibzimBridge.mm` against CoreKiwix.xcframework. This is the
   biggest gap and blocks everything ZIM-related.
3. Unit-test `ChatSession.extractToolCall(in:)` with `XCTest` — low-effort
   coverage for the most subtle part of the tool loop.
4. Fix `Gemma4Provider` API + the chat template; load Gemma on-device;
   measure the Instruments memory graph at steady state.
5. Add streaming polyline rendering — the tool result already contains a
   `polyline: [[lat, lon], ...]` array that `MapKit` can draw natively in
   a second SwiftUI view (nice-to-have, not required).
6. Consider a macOS Catalyst target; the same scaffolding should compile
   for macOS with only the `.iOS(.v17)` deployment info changed.

## Things not in the repo that you'll want

- `CoreKiwix.xcframework` — a binary, 50+ MB. Don't check it in. Either
  add `ios/MCPZimChat/Frameworks/*.xcframework` to `.gitignore` or keep
  the framework in a sibling directory and point `project.yml` at it.
- Gemma 4 weights — downloaded at runtime by `mlx-swift-lm`'s HF Hub
  client into the app sandbox. Nothing to check in.
- Your Apple Developer Team ID — set it in `project.yml`
  (`DEVELOPMENT_TEAM`) or in Xcode's Signing tab after each regenerate.

## Where the interesting code is

When you're reading the app on your Mac, start here:

- `ios/MCPZimChat/Chat/ChatSession.swift:runGenerationLoop()` — the
  tool-aware generation loop, about 60 lines.
- `swift/Sources/MCPZimKit/MCPToolAdapter.swift:dispatch(tool:args:)` —
  where incoming JSON args turn into `ZimService` calls.
- `swift/Sources/MCPZimKit/Router.swift:aStar(graph:origin:goal:)` — A*.
- `swift/Sources/MCPZimKit/SZRGGraph.swift:parse(_:)` — the SZRG v2
  parser.

If those four compile and pass their tests on your Mac, the rest is
integration plumbing.
