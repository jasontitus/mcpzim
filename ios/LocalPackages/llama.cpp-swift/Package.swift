// swift-tools-version:5.9
//
// Wrapper SPM package over the upstream llama.cpp XCFramework.
//
// Vendored from:
//   https://github.com/ggml-org/llama.cpp/releases/download/b8911/llama-b8911-xcframework.zip
//
// The XCFramework ships iOS device + iOS sim + macOS + visionOS (device+sim)
// + tvOS (device+sim) slices with Metal embedded via
// GGML_METAL_EMBED_LIBRARY=ON. See build-xcframework.sh upstream for the
// exact cmake flags. We don't modify the framework — just expose it as a
// local binaryTarget so the rest of the project can `import llama` and
// hit the C API directly.
//
// UPGRADE PROCEDURE (weekly-ish):
//   1. Check https://github.com/ggml-org/llama.cpp/releases for the
//      newest `bXXXX` tag that ships a `llama-bXXXX-xcframework.zip`.
//   2. `curl -L -o /tmp/x.zip <URL> && unzip /tmp/x.zip -d /tmp/`
//   3. `rm -rf ios/LocalPackages/llama.cpp-swift/llama.xcframework`
//   4. `mv /tmp/build-apple/llama.xcframework ios/LocalPackages/llama.cpp-swift/`
//   5. Bump the pinned tag in this comment and in HOW_TO_BUILD.md.
//   6. Rebuild the app — any newly-renamed C symbols will surface as
//      compile errors against our `LlamaCppProvider.swift` wrapper.
//
// Why not bump to each master commit: releases are tagged roughly
// daily (the GH Actions CI auto-tags on green). Using a release rather
// than a raw commit means we're on a ggml-org-validated binary, not a
// "last minute" master push.

import PackageDescription

let package = Package(
    name: "LlamaCppSwift",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        // Re-export the `llama` C module so callers say `import llama`.
        .library(name: "LlamaCppSwift", targets: ["LlamaCppSwift"]),
    ],
    targets: [
        // The actual binary. XCFramework contains a `llama.framework`
        // per platform slice and an `Info.plist` selecting the right
        // slice at build time. No Swift or Obj-C code of our own here
        // — we re-export the framework's `llama` module via the
        // umbrella below.
        .binaryTarget(
            name: "llama",
            path: "llama.xcframework"
        ),
        // Thin Swift umbrella so the rest of the project imports
        // `LlamaCppSwift` and the framework's `import llama` is
        // transitively available. Also a natural place to put any
        // Swift-side helpers that wrap raw C pointers (we keep those
        // in the app target for now — this stays empty).
        .target(
            name: "LlamaCppSwift",
            dependencies: ["llama"],
            path: "Sources/LlamaCppSwift"
        ),
    ]
)
