# MCPZimKit (Swift)

Pure-Swift companion to the Python `mcpzim` server, aimed at **iOS hosts** —
Swift-Gemma4-Core, MLX-Swift apps, or any Swift code that wants to make a
group of ZIM files available as tools to a local LLM.

```
┌─ your iOS app ─────────────────────────────────────────────────────┐
│                                                                    │
│   Swift-Gemma4-Core ──► Gemma4ToolLoop ──► MCPToolAdapter          │
│                                               │                    │
│                                               ▼                    │
│                                         DefaultZimService ──►┐     │
│                                                              │     │
│        ┌─────────────── ZimReader protocol ──────────────────┘     │
│        │                                                           │
│   CoreKiwix.xcframework (libzim + libkiwix — Obj-C++ bridge)       │
│        │                                                           │
│        └──► *.zim on disk (Wikipedia, mdwiki, streetzim)           │
└────────────────────────────────────────────────────────────────────┘
```

## What's in the box

| Module | Responsibility |
| --- | --- |
| `SZRGGraph` | Byte-for-byte parser for streetzim's `routing-data/graph.bin` (SZRG v2). Zigzag-varint polylines, CSR adjacency, deduped name table. |
| `Router` | A* shortest driving-time search. Same cost (`dist_m / (speed_kmh/3.6)`) and heuristic (`haversine / (100/3.6)`) as the JS viewer, so results match on-device. Outputs distance, duration, polyline, and turn-by-turn road segments. |
| `Geocoder` | Resolves a free-text query to lat/lon using streetzim's prefix-chunked JSON index. Prefix normalization exactly matches `_queryPlaces`. |
| `ZimReader` | Protocol the host app implements, typically by wrapping `CoreKiwix.xcframework` (Kiwix's shipped iOS binary) in an Obj-C++ shim. Also handles type classification (wikipedia / mdwiki / streetzim / generic). |
| `ZimService` + `DefaultZimService` | Clean in-process Swift API. Actor-based, safe to share across concurrent tool calls. |
| `MCPToolAdapter` | Transport-agnostic MCP tool registry. Exposes `tools: [MCPTool]` (schemas are baked JSON) and `dispatch(tool:args:)`. Register each tool with the official `modelcontextprotocol/swift-sdk` (for LAN/stdio transports), OpenAI-style function calling, or your own tool loop. |

## Adding libzim

MCPZimKit deliberately does **not** depend on libzim directly — it's Swift
Package Manager, and libzim is C++. The idiomatic path:

1. Either grab `CoreKiwix.xcframework` from the Kiwix project
   ([kiwix/kiwix-apple](https://github.com/kiwix/kiwix-apple) documents the
   download URL and signing; `brew bundle` fetches it on macOS) or build a
   slimmer libzim-only xcframework via
   [kiwix/kiwix-build](https://github.com/kiwix/kiwix-build) (`--config
   apple_all_static`).
2. In your app target, add the xcframework and a small Obj-C++ module that
   exposes `libzim::Archive` through a thin Swift-facing wrapper.
3. Conform that wrapper to `ZimReader`:
   ```swift
   final class LibzimReader: ZimReader {
       let archive: CppZimArchive  // your Obj-C++ bridge type
       var metadata: ZimMetadata { ... }
       var kind: ZimKind { ... }
       func read(path: String) throws -> ZimEntry? { ... }
       func readMainPage() throws -> ZimEntry? { ... }
       func searchFullText(query: String, limit: Int) throws -> [ZimSearchHit] { ... }
   }
   ```

## Using it with Swift-Gemma4-Core

Swift-Gemma4-Core has no tool-calling of its own; the generation loop emits an
`AsyncStream<Generation>` that your code has to scan for tool-call syntax.
`Examples/Gemma4Integration/Gemma4ToolLoop.swift` is a minimal pattern — it
generates a tool list preamble, detects `<tool_call>...</tool_call>` blocks,
dispatches them via `MCPToolAdapter`, and re-feeds the result to Gemma.

## Running MCP over the wire

For clients that speak MCP JSON-RPC (Claude Desktop, Claude Code, any
HTTP/stdio MCP host on the LAN), plug `MCPToolAdapter` into the official
[`modelcontextprotocol/swift-sdk`](https://github.com/modelcontextprotocol/swift-sdk):

```swift
let registry = await adapter.registry
let server = try MCP.Server()
for tool in registry.tools {
    server.addTool(
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchemaJSON
    ) { args in
        try await adapter.dispatch(tool: tool.name, args: args)
    }
}
try await server.run(transport: .stdio)  // or .streamableHttp, etc.
```

(Exact SDK method names may drift across versions — the pattern is
`addTool(name:description:schema:handler:)`. Check the SDK's README.)

## Tests

```sh
cd swift
swift test
```

The test suite builds small SZRG v2 fixtures in memory via `encodeGraphV2`, so
you don't need a real ZIM on disk. These mirror the Python tests in
`../tests/test_routing.py` and `../tests/test_geocode.py` line-for-line, so if
both pass you know the Swift and Python implementations agree.
