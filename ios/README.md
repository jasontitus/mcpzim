# MCPZim Chat (iOS)

A minimal SwiftUI chat app that:

- Runs a local LLM on-device (Gemma 4 via [Swift-Gemma4-Core](https://github.com/yejingyang8963-byte/Swift-gemma4-core);
  the `ModelProvider` protocol accepts any other Swift-native model as an
  alternative).
- Opens a folder of ZIM files (Wikipedia, mdwiki, streetzim) through
  `CoreKiwix.xcframework` and exposes them as tools via
  [`swift/MCPZimKit`](../swift).
- Detects `<tool_call>` blocks in the model's output stream, runs the tool
  in-process through `MCPToolAdapter`, and feeds the JSON result back to the
  model on the next decoding pass — so *"give me a driving route from Boston
  Common to Fenway Park"* triggers a real A* run over the loaded streetzim
  graph and the answer is rendered as turn-by-turn directions with a
  collapsible tool trace.

```
┌────────────┐   draft    ┌───────────────────┐  <tool_call>   ┌─────────────────┐
│ ChatView   ├──────────▶ │ ChatSession       ├──────────────▶ │ MCPToolAdapter  │
│ (SwiftUI)  │ ◀──────────│ (tool-aware loop) │ ◀──────────────│ ZimService      │
└────────────┘  tokens    └──────┬────────────┘  JSON result   └─────┬───────────┘
                                 │                                   │
                                 ▼                                   ▼
                          ModelProvider                        ZimReader(s)
                          (Gemma 4 / Mock)                     (LibzimReader)
```

## Prerequisites

- macOS 14+ with Xcode 15.3+.
- An Apple Developer account (free is fine for sideloading to your own
  device; App Store distribution obviously needs a paid account).
- An iPhone with at least 6 GB RAM for Gemma 4 4-bit (iPhone 14 Pro / 15 /
  15 Pro / 16, or an M-series iPad). Simulator works for UI development
  with the `MockProvider`.
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`.

## One-time setup

```sh
cd ios
xcodegen generate          # produces MCPZimChat.xcodeproj from project.yml
open MCPZimChat.xcodeproj
```

### 1. Vendor CoreKiwix.xcframework

Drop `CoreKiwix.xcframework` into `ios/MCPZimChat/Frameworks/`. Two ways to
get it:

- **Easy**: download the latest release artifact from
  [kiwix/kiwix-apple](https://github.com/kiwix/kiwix-apple) (their Brewfile
  points at the URL their CI publishes).
- **From source**: `pip install kiwix-build && kiwix-build --target libkiwix
  --config apple_all_static` — outputs an xcframework under
  `BUILD_apple_all_static/INSTALL/`.

Then open `ios/project.yml`, uncomment the `framework:` dependency line, and
re-run `xcodegen generate`.

### 2. Add Swift-Gemma4-Core

In Xcode: *File > Add Package Dependencies*, URL
`https://github.com/yejingyang8963-byte/Swift-gemma4-core.git`, version
`0.1.0+`. Then add product `Gemma4SwiftCore` to the `MCPZimChat` target.

(`Gemma4Provider.swift` detects the package via `#if canImport(Gemma4SwiftCore)`
so the app still builds — with a "not linked" model entry — before you add it.)

### 3. Sign

Open the project, select the `MCPZimChat` target → *Signing & Capabilities*
→ set your Team. (Or set `DEVELOPMENT_TEAM` in `project.yml` and regenerate.)

## Adding ZIM files

The app enumerates `.zim` files in its own Documents folder. To get files in:

1. Plug in your device, open Finder → *[device name]* → *Files* → drop
   the ZIMs into `MCPZimChat`.
2. Or use the **Add ZIM** button inside the app to pick a file from
   Files.app (via the document picker — the picker copies the file into the
   app sandbox).

The app scans Documents automatically on launch.

## Running

- First launch, pick **Gemma 4 4B (4-bit)** from the model menu and tap
  *Load model*. The first load downloads ~1.5 GB of weights from HuggingFace
  (this is the HF Hub download built into `mlx-swift-lm`; subsequent launches
  are instant).
- Ask things like:
  - *"What's in my library?"* → calls `list_libraries`.
  - *"What is aspirin used for?"* → `search` + `get_article` against mdwiki.
  - *"Plan a driving route from Boston Common to Fenway Park"* → `geocode` +
    `plan_driving_route` against the loaded streetzim ZIM.

Tool calls show up as collapsible chips under the assistant's message so you
can see exactly what the model asked and what came back.

## Mock model for UI work

Select **Mock (scripted)** from the model menu to develop layout / tool-call
rendering without loading Gemma. The mock emits canned `<tool_call>` blocks
for a couple of prompts so the tool loop still exercises end-to-end.

## Troubleshooting

- **"CoreKiwix.xcframework is not linked"** — you haven't dropped the
  xcframework into `Frameworks/` yet. The app still runs but can't open
  `.zim` files.
- **Memory warnings when loading Gemma + a large streetzim graph** —
  streetzim graphs for a whole US state can be 100–200 MB. Gemma 4 takes
  ~400 MB steady-state. That's fine on a 6 GB+ device; on older hardware,
  either use a smaller streetzim extract or switch to a Gemma 4 smaller
  variant.
- **Sandboxed filesystem** — the document picker imports files into the
  app's own Documents dir rather than referring to the user-selected
  location. This is intentional — once imported, the ZIM is addressable by
  a stable URL the app can re-open across launches.

## Extending

Add another model by conforming to `ModelProvider` (see
`Providers/ModelProvider.swift`). Drop an instance into
`ChatSession.init()`'s `self.models = [...]` list. The picker and the chat
loop work against the protocol, no view changes needed.
