// swift-tools-version: 5.9
//
// Headless smoke-test harness for the Gemma 4 + MCPZimKit stack.
//
// Purpose: iterate on prompt templating, tokenization, and ZIM tool dispatch
// *without* rebuilding the iOS/Mac app and reloading the UI every time.
// Run with:
//
//     cd tools/gemma-smoke
//     swift run GemmaSmoke "tell me about baseball"
//
// The executable loads the shared mlx-community Gemma 4 E2B 4-bit weights
// (downloaded once into ~/.cache/huggingface on first run), renders our
// `Gemma4PromptTemplate` over the user message, streams generation through
// the same stop-marker filter the app uses, and prints everything to stdout
// so you can inspect the raw model output verbatim.

import PackageDescription

let package = Package(
    name: "GemmaSmoke",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../swift"),
        .package(url: "https://github.com/yejingyang8963-byte/Swift-gemma4-core.git", from: "0.1.0"),
    ],
    targets: [
        .executableTarget(
            name: "GemmaSmoke",
            dependencies: [
                .product(name: "MCPZimKit", package: "swift"),
                .product(name: "Gemma4SwiftCore", package: "Swift-gemma4-core"),
            ],
            path: "Sources/GemmaSmoke"
        ),
    ]
)
