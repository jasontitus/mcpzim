# LlamaCppSwift — vendored llama.cpp XCFramework wrapper

This SPM package wraps the prebuilt `llama.xcframework` from
[ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).
The xcframework itself (~562 MB with ios device + ios sim + macOS +
visionOS device/sim + tvOS device/sim slices) is **not checked into
git** — you download it once per clone.

## Restore after a fresh clone

```sh
cd ios/LocalPackages/llama.cpp-swift
TAG=b8911   # keep in sync with the comment in Package.swift
curl -L -o /tmp/llama.zip \
  "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-xcframework.zip"
rm -rf llama.xcframework
unzip -q /tmp/llama.zip -d /tmp/llama-zip
mv /tmp/llama-zip/build-apple/llama.xcframework .
rm -rf /tmp/llama.zip /tmp/llama-zip
```

Verify:

```sh
ls llama.xcframework/Info.plist   # should exist
```

Then `xcodegen generate` in `ios/` will pick up the package and the
build target (`MCPZimChat`) can `import llama`.

## Upgrading

See `Package.swift`'s header comment — the canonical upgrade procedure
also lives there (check releases for new `bXXXX` tag, run the same
curl + mv, bump the tag in `Package.swift` + this README + HOW_TO_BUILD.md).
