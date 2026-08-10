# LlamaCppSwift — vendored llama.cpp XCFramework wrapper

This SPM package wraps a `llama.xcframework` built from the
[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), branch
`prism`, commit `62061f9` (`prism-b9591`). This exact provenance is important:
the Mac ternary Bonsai model uses Prism's group-128 Q2 format, which is not
interchangeable with stock llama.cpp's group-64 Q2 format.

Stock llama.cpp now supports the Bonsai Q1 format. Moving the phone model to
stock remains a benchmarkable option, but do not replace this single framework
until the Mac ternary model has a compatible migration plan.

The xcframework itself (~562 MB with ios device + ios sim + macOS +
visionOS device/sim + tvOS device/sim slices) is **not checked into
git** — restore it from the pinned Prism build or rebuild that exact commit.

## Restore after a fresh clone

Check out Prism commit `62061f9`, run its `build-xcframework.sh`, and copy
`build-apple/llama.xcframework` into this directory. Do not substitute a stock
release archive: it will not load the current ternary Bonsai GGUF.

Verify:

```sh
ls llama.xcframework/Info.plist   # should exist
```

Then `xcodegen generate` in `ios/` will pick up the package and the
build target (`MCPZimChat`) can `import llama`.

## Upgrading

See `Package.swift`'s header comment for the canonical upgrade procedure.
Benchmark the phone Q1 model before changing runtimes, and verify the Mac Q2
model format before replacing the framework. Update the pinned revision here
and in `Package.swift` together.
