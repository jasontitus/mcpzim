// SPDX-License-Identifier: MIT
//
// Example: wire MCPZimKit into a Swift-Gemma4-Core generation loop.
//
// Swift-Gemma4-Core (github.com/yejingyang8963-byte/Swift-gemma4-core) ships
// only the inference engine — no built-in tool-calling. This file shows the
// minimum glue an iOS app needs so a local Gemma 4 model can answer questions
// like "give me a driving route from Boston Common to Fenway Park" using the
// ZIM-backed tools we expose.
//
// The pattern is:
//   1. Pre-prompt Gemma with the registry of available MCPZim tools.
//   2. Stream the model's output.
//   3. Detect tool-call blocks (`<tool_call>{"name":..., "arguments":...}</tool_call>`
//      is a common Gemma 4 convention — adjust to whatever template you use).
//   4. Run the tool in-process via MCPZimKit's adapter.
//   5. Inject the result back into the transcript (Swift-Gemma4-Core v0.1
//      ships only `Gemma4PromptFormatter.userTurn(_:)`, so format the tool
//      response as a user turn prefixed with `<tool_response>...`) and resume
//      generation.
//
// This file is intentionally *illustrative* — it compiles against MCPZimKit
// but does not import Swift-Gemma4-Core, because this package avoids a hard
// dep on MLX/Gemma4 for reachability on Linux CI. Drop it into an iOS target
// that depends on both `MCPZimKit` and `Gemma4SwiftCore`.

import Foundation
import MCPZimKit

public struct Gemma4ToolLoop {
    let adapter: MCPToolAdapter

    public init(adapter: MCPToolAdapter) {
        self.adapter = adapter
    }

    /// Render a system preamble enumerating the tools Gemma can call. The
    /// exact prompt template depends on the Gemma 4 variant in use (Gemma 4
    /// supports a native `<start_of_turn>tool` role as well as inline
    /// `<tool_call>...</tool_call>` blocks). Adapt the format to match
    /// `Gemma4PromptFormatter` in your copy of Swift-Gemma4-Core.
    public func systemPromptToolsBlock() async -> String {
        let registry = await adapter.registry
        var lines: [String] = [
            "You have access to the following tools. To call one, emit a",
            "<tool_call>{\"name\":\"TOOL\",\"arguments\":{...}}</tool_call>",
            "block. Wait for the <tool_response> message before continuing.",
            "",
            "Tools:",
        ]
        for tool in registry.tools {
            let schema = String(data: tool.inputSchemaJSON, encoding: .utf8) ?? "{}"
            lines.append("- \(tool.name): \(tool.description)")
            lines.append("  schema: \(schema)")
        }
        return lines.joined(separator: "\n")
    }

    /// Extract the first tool call from a streamed Gemma 4 output chunk.
    /// Returns nil if no complete `<tool_call>...</tool_call>` block is
    /// present yet.
    public func extractToolCall(from buffer: String) -> (name: String, args: [String: Any], range: Range<String.Index>)? {
        guard let start = buffer.range(of: "<tool_call>"),
              let end = buffer.range(of: "</tool_call>", range: start.upperBound..<buffer.endIndex)
        else { return nil }
        let jsonRange = start.upperBound..<end.lowerBound
        let json = String(buffer[jsonRange])
        guard let data = json.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = parsed["name"] as? String
        else { return nil }
        let args = (parsed["arguments"] as? [String: Any]) ?? [:]
        return (name, args, start.lowerBound..<end.upperBound)
    }

    /// Dispatch a detected tool call and format the response for re-injection
    /// into the Gemma 4 conversation.
    public func runToolCall(name: String, args: [String: Any]) async -> String {
        do {
            let result = try await adapter.dispatch(tool: name, args: args)
            let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
            let pretty = String(data: data, encoding: .utf8) ?? "{}"
            return "<tool_response name=\"\(name)\">\(pretty)</tool_response>"
        } catch {
            return "<tool_response name=\"\(name)\" error=\"true\">\(error)</tool_response>"
        }
    }
}

/*
 Sketch of the full generation loop (pseudo-code; requires Swift-Gemma4-Core):

 let service = DefaultZimService(readers: openedReaders())
 let adapter = await MCPToolAdapter.from(service: service)
 let loop = Gemma4ToolLoop(adapter: adapter)

 var transcript = await loop.systemPromptToolsBlock()
 transcript += Gemma4PromptFormatter.userTurn("Route from Boston Common to Fenway Park")

 while true {
     var buffer = ""
     for await chunk in container.generate(text: transcript) {
         guard case .chunk(let s) = chunk else { continue }
         buffer += s
         if let call = loop.extractToolCall(from: buffer) {
             let response = await loop.runToolCall(name: call.name, args: call.args)
             transcript = String(buffer[..<call.range.upperBound]) + response
             break  // restart generation with the tool result appended.
         }
     }
     // No tool call detected => Gemma has produced its final answer.
     if loop.extractToolCall(from: buffer) == nil { print(buffer); break }
 }
 */
