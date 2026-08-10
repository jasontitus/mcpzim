// swift-tools-version:5.9
//
// Wrapper SPM package over the upstream llama.cpp XCFramework.
//
// Vendored from:
//   PrismML-Eng/llama.cpp branch `prism`, commit 62061f9
//   (`prism-b9591`). This build supplies the tested Q1_0_g128 Metal path for
//   Bonsai 27B and the fork-only Q2_0_g128 path used by the Mac ternary model.
//   Stock llama.cpp now supports Bonsai Q1, but its Q2 format is group-64 and
//   is not interchangeable with Prism's existing group-128 GGUF.
//
// The XCFramework ships iOS device + iOS sim + macOS + visionOS (device+sim)
// + tvOS (device+sim) slices with Metal embedded via
// GGML_METAL_EMBED_LIBRARY=ON. See build-xcframework.sh upstream for the
// exact cmake flags. We don't modify the framework — just expose it as a
// local binaryTarget so the rest of the project can `import llama` and
// hit the C API directly.
//
// UPGRADE PROCEDURE (weekly-ish):
//   1. Check out the desired PrismML-Eng/llama.cpp `prism` commit.
//   2. Run its `build-xcframework.sh` with Xcode's Metal toolchain installed.
//   3. `rm -rf ios/LocalPackages/llama.cpp-swift/llama.xcframework`
//   4. Copy `build-apple/llama.xcframework` here.
//   5. Bump the pinned commit in this comment and in HOW_TO_BUILD.md.
//   6. Rebuild the app — any newly-renamed C symbols will surface as
//      compile errors against our `LlamaCppProvider.swift` wrapper.
//
// Keep the exact commit pinned until a controlled Q1 benchmark and Q2 model
// migration are complete. An arbitrary upstream XCFramework is not compatible
// with the existing Prism group-128 ternary GGUF.

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
