// Re-exports the `llama` C module so callers can
// `import LlamaCppSwift` and pick up the full C API.
//
// Anything heavier (tokenizer helpers, sampling chain wrappers,
// streaming adapters) lives in the app target in
// `LlamaCppProvider.swift` — keeping this package surface minimal
// means upgrading the pinned XCFramework is a drop-in replace, not
// a Swift-code migration.
@_exported import llama
