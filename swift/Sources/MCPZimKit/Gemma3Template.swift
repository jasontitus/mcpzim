// SPDX-License-Identifier: MIT
//
// `ModelTemplate` for Gemma 3 family (gemma-3-4b-it / gemma-3-12b-it /
// gemma-3-27b-it variants). Gemma 3 wasn't tuned with a native tool-call
// protocol the way Gemma 4 was, so we inline a Qwen-style JSON-in-tags
// convention in the system turn and parse the same way. Turn markers use
// Gemma's canonical `<start_of_turn>` / `<end_of_turn>` pair.
//
//   <start_of_turn>user
//   <system prose>
//
//   # Tools
//   <inlined JSON tool schemas>
//
//   <user text><end_of_turn>
//   <start_of_turn>model
//   <tool_call>
//   {"name":"near_named_place","arguments":{"place":"San Francisco","kinds":["restaurant"]}}
//   </tool_call><end_of_turn>
//   <start_of_turn>user
//   <tool_response>
//   {"name":"near_named_place","content":{…}}
//   </tool_response>
//   <end_of_turn>
//   <start_of_turn>model
//   <synthesised reply>
//   <end_of_turn>
//
// Notes:
//
//   * Gemma 3's tokenizer registers `<start_of_turn>` and `<end_of_turn>` as
//     single special tokens, so passing this string through `Tokenizer.encode`
//     (the path our provider uses) encodes them as single IDs rather than BPE
//     byte sequences. The app's `Gemma4Provider` goes through that path.
//   * Gemma 3 only knows `user` and `model` roles — there's no dedicated
//     system turn, so we fold any system preamble (plus tool declarations)
//     into the first user turn's body. That matches what
//     `tokenizer.apply_chat_template` does for this family.
//   * `<tool_call>` / `</tool_call>` are emitted as plain text bytes; the
//     model learns to produce them from the in-system demonstration. Parsing
//     is the same streaming walk `QwenChatMLTemplate` uses.

import Foundation

public struct Gemma3Template: ModelTemplate {

    public init() {}

    /// Gemma 3's tokenizer emits `<bos>` automatically on first token —
    /// leave empty and let the special token ordering handle it. Matches
    /// what `Gemma3TextModel.newCache` expects.
    public var bos: String { "" }

    /// Single turn-close marker. The model's chat tune trained on exactly
    /// this string; if we stop on anything else we risk truncating a real
    /// assistant emission mid-stream.
    public var stopMarkers: [String] { ["<end_of_turn>"] }

    public var logCategory: String { "Gemma3" }

    public func formatSystemTurn(
        systemMessage: String, tools: [ModelToolDeclaration]
    ) -> String {
        // Gemma 3 has no `system` role — return a pseudo-system block that
        // `renderTranscript` folds into the first user turn. We still emit
        // it as a discrete string so callers that build a transcript
        // manually get something useful, but in-app the normal path is
        // `renderTranscript`.
        var out = ""
        if !systemMessage.isEmpty {
            out += systemMessage
            if !systemMessage.hasSuffix("\n") { out += "\n" }
        }
        if !tools.isEmpty {
            if !out.isEmpty { out += "\n" }
            out += "# Tools\n\n"
            out += "You may call one or more functions to assist with the user query.\n\n"
            out += "Emit each call as:\n"
            out += "<tool_call>\n"
            out += "{\"name\": <function-name>, \"arguments\": <args-as-JSON-object>}\n"
            out += "</tool_call>\n\n"
            out += "Here are the available tools:\n"
            out += "<tools>\n"
            for t in tools {
                out += Self.toolJSONLine(t) + "\n"
            }
            out += "</tools>"
        }
        return out
    }

    public func renderTranscript(
        systemPreamble: String,
        tools: [ModelToolDeclaration],
        turns: [ChatTurn]
    ) -> String {
        let sysBody = formatSystemTurn(systemMessage: systemPreamble, tools: tools)
        var out = bos
        var pendingSys = sysBody

        for t in turns {
            switch t.role {
            case .system:
                // If upstream already emitted a .system turn with tool
                // prose, concatenate onto the pending sys block. The
                // renderer injects it into the next user turn.
                if pendingSys.isEmpty {
                    pendingSys = t.text
                } else {
                    pendingSys += "\n\n" + t.text
                }
            case .user:
                var body = t.text
                if !pendingSys.isEmpty {
                    body = pendingSys + "\n\n" + body
                    pendingSys = ""
                }
                out += "<start_of_turn>user\n\(body)<end_of_turn>\n"
            case .assistant:
                out += "<start_of_turn>model\n\(t.text)<end_of_turn>\n"
            case .tool:
                // Gemma 3 convention: tool responses come back as user
                // turns whose body is a `<tool_response>…</tool_response>`
                // block. The body text is already rendered by
                // `formatToolResponse`.
                out += "<start_of_turn>user\n\(t.text)<end_of_turn>\n"
            }
        }

        // If no turns were provided but we had a system preamble, still
        // surface it as an empty user turn so the model has context when
        // the caller supplies the real user prompt later.
        if !pendingSys.isEmpty {
            out += "<start_of_turn>user\n\(pendingSys)<end_of_turn>\n"
        }

        // Open the assistant turn for generation.
        out += "<start_of_turn>model\n"
        return out
    }

    public func firstToolCall(in buffer: String) -> ToolCallMatch? {
        firstToolCall(in: buffer, allowImplicitClose: false)
    }

    public func firstToolCallAfterClip(in buffer: String) -> ToolCallMatch? {
        firstToolCall(in: buffer, allowImplicitClose: true)
    }

    /// Streaming-safe (`allowImplicitClose=false`) vs end-of-stream rescue
    /// (`true`). Gemma 3 doesn't have a single canonical tool-call
    /// wrapper — it emits tool calls in several formats the same in-system
    /// demonstration tries to steer, observed in the wild on
    /// `mlx-community/gemma-3-text-4b-it-4bit` through the Swift
    /// EvalHarness (2026-04-23):
    ///
    ///   1. `<tool_call>{…}</tool_call>`                  (what our
    ///      system turn demonstrates — canonical)
    ///   2. ```` ```tool_call\n{…}\n``` ```` (markdown fence with
    ///      `tool_call` info string)
    ///   3. ```` ```json\n{…}\n``` ```` (markdown fence with
    ///      `json` info string)
    ///   4. ```` ```\n{…}\n``` ```` (unfenced markdown fallback)
    ///
    /// Arg-shape variants we've seen inside the body:
    ///   - `{"name":…,"arguments":{…}}`          (canonical)
    ///   - `{"function":{"name":…,"arguments":…}}` (OpenAI-ish)
    ///   - `{"function":{"name":…,"parameters":{"properties":…}}}` (Python-schema-ish)
    ///   - `{"tool_call":{"name":…,"arguments":…}}`         (wrapped)
    ///   - `{"name":…,"parameters":{…}}`           (`parameters` alias)
    ///
    /// We try each wrapper in order and each arg-shape inside. Unknown
    /// wrappers fall through to nil.
    public func firstToolCall(
        in buffer: String, allowImplicitClose: Bool
    ) -> ToolCallMatch? {
        // 1. Canonical `<tool_call>…</tool_call>` wrapper.
        if let m = findCall(in: buffer, open: "<tool_call>", close: "</tool_call>",
                            allowImplicitClose: allowImplicitClose) {
            return m
        }
        // 2/3. Markdown code-fence wrappers. ``` ```tool_call / ```json ```
        for fenceTag in ["```tool_call", "```json", "```tool", "```"] {
            if let m = findCall(in: buffer, open: fenceTag, close: "```",
                                allowImplicitClose: allowImplicitClose) {
                return m
            }
        }
        return nil
    }

    private func findCall(
        in buffer: String, open: String, close: String,
        allowImplicitClose: Bool
    ) -> ToolCallMatch? {
        guard let openMarker = buffer.range(of: open) else { return nil }
        let searchStart = openMarker.upperBound
        let closeRange: Range<String.Index>
        if let r = buffer.range(of: close, range: searchStart..<buffer.endIndex) {
            closeRange = r
        } else if allowImplicitClose {
            if let r = buffer.range(of: "</tool", range: searchStart..<buffer.endIndex) {
                closeRange = r
            } else {
                closeRange = buffer.endIndex..<buffer.endIndex
            }
        } else {
            return nil
        }
        let body = String(buffer[openMarker.upperBound..<closeRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let obj = Self.decodeObject(body)
                ?? Self.decodeObject(Self.repairJSON(body))
        else { return nil }

        if let (name, args) = Self.extractNameArgs(obj) {
            return ToolCallMatch(
                range: openMarker.lowerBound..<closeRange.upperBound,
                name: name, arguments: args
            )
        }
        return nil
    }

    /// Pull `(name, arguments)` out of whichever of the five observed
    /// JSON shapes the model emitted. Returns nil if no shape matches.
    private static func extractNameArgs(_ obj: [String: Any])
        -> (name: String, arguments: [String: Any])?
    {
        // (a) canonical `{"name":…,"arguments":…}`
        if let name = obj["name"] as? String,
           let args = obj["arguments"] as? [String: Any]
        {
            return (name, args)
        }
        // (b) `{"name":…,"parameters":{…}}` (parameters alias used by
        // Gemma 3 when it follows Python-style schema naming).
        if let name = obj["name"] as? String,
           let params = obj["parameters"] as? [String: Any]
        {
            if let nestedProps = params["properties"] as? [String: Any] {
                return (name, nestedProps)
            }
            return (name, params)
        }
        // (c/d) `{"function":{"name":…,"arguments":…}}` /
        //       `{"function":{"name":…,"parameters":…}}`
        if let inner = obj["function"] as? [String: Any],
           let name = inner["name"] as? String
        {
            if let args = inner["arguments"] as? [String: Any] {
                return (name, args)
            }
            if let params = inner["parameters"] as? [String: Any] {
                if let nestedProps = params["properties"] as? [String: Any] {
                    return (name, nestedProps)
                }
                return (name, params)
            }
            return (name, [:])
        }
        // (e) `{"tool_call":{"name":…,"arguments":…}}`
        if let wrapped = obj["tool_call"] as? [String: Any],
           let (name, args) = extractNameArgs(wrapped)
        {
            return (name, args)
        }
        return nil
    }

    public func formatToolCall(name: String, arguments: [String: Any]) -> String {
        let body: [String: Any] = ["name": name, "arguments": arguments]
        let data = (try? JSONSerialization.data(
            withJSONObject: body, options: [.sortedKeys]
        )) ?? Data()
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return "<tool_call>\n\(json)\n</tool_call>"
    }

    public func formatToolResponse(name: String, payload: [String: Any]) -> String {
        let wrapped: [String: Any] = ["name": name, "content": payload]
        let data = (try? JSONSerialization.data(
            withJSONObject: wrapped, options: [.sortedKeys]
        )) ?? Data()
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return "<tool_response>\n\(json)\n</tool_response>"
    }

    // MARK: - Private helpers (kept local — shared with QwenChatMLTemplate
    //         but re-declared so both templates stay decoupled)

    private static func decodeObject(_ s: String) -> [String: Any]? {
        guard let data = s.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    /// Same best-effort repair passes as `QwenChatMLTemplate.repairJSON` —
    /// Gemma 3 hasn't exhibited the same quirks in the wild (it wasn't
    /// trained on native tool calls) but the repairs are cheap + safe:
    /// double commas, trailing commas, whitespace-wedged strings, doubled
    /// opening quotes, and brace-balance clip recovery.
    private static func repairJSON(_ s: String) -> String {
        var out = s
        while out.contains(",,") {
            out = out.replacingOccurrences(of: ",,", with: ",")
        }
        out = out.replacingOccurrences(
            of: #",\s*([}\]])"#, with: "$1", options: .regularExpression
        )
        out = out.replacingOccurrences(
            of: #""\s+"(?=[^"\s,\]\}])"#, with: "\"", options: .regularExpression
        )
        out = out.replacingOccurrences(
            of: #"([{,])\s*""(?=[A-Za-z_])"#, with: "$1\"", options: .regularExpression
        )
        let opens = out.filter { $0 == "{" }.count
        let closes = out.filter { $0 == "}" }.count
        if opens > closes { out += String(repeating: "}", count: opens - closes) }
        return out
    }

    private static func toolJSONLine(_ t: ModelToolDeclaration) -> String {
        var props: [String: [String: Any]] = [:]
        var required: [String] = []
        for p in t.parameters {
            var pd: [String: Any] = [
                "type": p.type.jsonSchemaTypeName,
                "description": p.description ?? "",
            ]
            if let ev = p.enumValues, !ev.isEmpty { pd["enum"] = ev }
            if p.nullable { pd["nullable"] = true }
            if p.type == .array { pd["items"] = ["type": "string"] }
            props[p.name] = pd
            if p.required { required.append(p.name) }
        }
        let parameters: [String: Any] = [
            "type": "object",
            "properties": props,
            "required": required,
        ]
        let function: [String: Any] = [
            "name": t.name,
            "description": t.description,
            "parameters": parameters,
        ]
        let outer: [String: Any] = [
            "type": "function",
            "function": function,
        ]
        let data = (try? JSONSerialization.data(
            withJSONObject: outer, options: [.sortedKeys]
        )) ?? Data()
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}

private extension ModelToolDeclaration.Parameter.ParamType {
    var jsonSchemaTypeName: String {
        switch self {
        case .string:  return "string"
        case .integer: return "integer"
        case .number:  return "number"
        case .boolean: return "boolean"
        case .array:   return "array"
        case .object:  return "object"
        }
    }
}
