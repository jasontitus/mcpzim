// SPDX-License-Identifier: MIT
//
// `ModelTemplate` for Qwen-family instruction-tuned models (Qwen 2 / 2.5
// / 3 / 3.5, including MoE and "Text" variants). All use ChatML:
//
//   <|im_start|>system
//   <system prose>
//   # Tools
//   <tools>
//   {"type":"function","function":{"name":"…","description":"…","parameters":{…}}}
//   …
//   </tools>
//   <|im_end|>
//   <|im_start|>user
//   <user text>
//   <|im_end|>
//   <|im_start|>assistant
//   <tool_call>
//   {"name":"near_named_place","arguments":{"place":"San Francisco","kinds":["restaurant"]}}
//   </tool_call><|im_end|>
//   <|im_start|>user
//   <tool_response>
//   {"name":"near_named_place","content":{…canned…}}
//   </tool_response>
//   <|im_end|>
//   <|im_start|>assistant
//   <synthesised reply>
//   <|im_end|>
//
// Unlike Gemma 4's custom mini-format, Qwen tool calls + tool responses
// are plain JSON. The parser here walks the streaming buffer for a
// `<tool_call>…</tool_call>` block and uses JSONSerialization to decode.

import Foundation

public struct QwenChatMLTemplate: ModelTemplate {

    public init() {}

    /// Qwen tokenizers auto-insert `<|im_start|>` via `add_generation_prompt`
    /// in their chat_template.jinja, but we're rendering by hand — leave
    /// the BOS empty and let the first `<|im_start|>system` marker anchor
    /// the prompt. The tokenizer's `<|endoftext|>` / `<|im_start|>` are
    /// single special tokens so the whole string encodes cleanly.
    public var bos: String { "" }

    /// ChatML uses `<|im_end|>` as the universal turn-close marker.
    public var stopMarkers: [String] { ["<|im_end|>"] }

    public func formatSystemTurn(
        systemMessage: String, tools: [ModelToolDeclaration]
    ) -> String {
        var out = "<|im_start|>system\n"
        out += systemMessage
        if !tools.isEmpty {
            out += "\n\n# Tools\n\n"
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
        out += "<|im_end|>\n"
        return out
    }

    public func renderTranscript(
        systemPreamble: String,
        tools: [ModelToolDeclaration],
        turns: [ChatTurn]
    ) -> String {
        var out = bos
        if !systemPreamble.isEmpty || !tools.isEmpty {
            out += formatSystemTurn(systemMessage: systemPreamble, tools: tools)
        }
        for t in turns {
            // Tool turns come pre-formatted (we rendered the
            // `<tool_response>{…}</tool_response>` body ourselves);
            // treat them as user-role content in the ChatML transcript
            // so the model sees them before generating the next
            // assistant turn.
            let role: String
            switch t.role {
            case .user:       role = "user"
            case .assistant:  role = "assistant"
            case .system:     role = "system"
            case .tool:       role = "user"   // ChatML convention
            }
            out += "<|im_start|>\(role)\n\(t.text)<|im_end|>\n"
        }
        // Open the assistant turn so generation picks up in assistant mode.
        out += "<|im_start|>assistant\n"
        return out
    }

    public func firstToolCall(in buffer: String) -> ToolCallMatch? {
        guard let openMarker = buffer.range(of: "<tool_call>") else { return nil }
        guard let closeMarker = buffer.range(
            of: "</tool_call>", range: openMarker.upperBound..<buffer.endIndex
        ) else { return nil }
        let bodyStart = openMarker.upperBound
        let body = String(buffer[bodyStart..<closeMarker.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = body.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        guard let name = obj["name"] as? String,
              let args = obj["arguments"] as? [String: Any]
        else { return nil }
        return ToolCallMatch(
            range: openMarker.lowerBound..<closeMarker.upperBound,
            name: name, arguments: args
        )
    }

    /// Qwen 3's reasoning mode wraps its scratchpad in `<think>…</think>`
    /// before the user-facing answer. Strip any CLOSED span — leave
    /// open-but-unclosed spans alone so the UI doesn't flash half a
    /// reasoning block while streaming.
    public func stripReasoning(_ text: String) -> String {
        guard text.contains("<think>") else { return text }
        var out = text
        // Remove every <think>…</think> span (non-greedy, spans lines).
        while let openRange = out.range(of: "<think>"),
              let closeRange = out.range(
                of: "</think>",
                range: openRange.upperBound..<out.endIndex
              )
        {
            // Swallow a single trailing whitespace / newline after the
            // close so we don't leave a blank line where the block was.
            var after = closeRange.upperBound
            while after < out.endIndex, out[after].isWhitespace {
                after = out.index(after, offsetBy: 1)
            }
            out.removeSubrange(openRange.lowerBound..<after)
        }
        return out
    }

    public func formatToolResponse(name: String, payload: [String: Any]) -> String {
        // Tool-response turn body. ChatSession wraps this in a `tool`
        // ChatTurn; `renderTranscript` emits it as a user-role ChatML
        // block (Qwen convention). Encoding as JSON matches the tool's
        // declared schema output and is what the model's fine-tune
        // expects.
        let wrapped: [String: Any] = [
            "name": name,
            "content": payload,
        ]
        let data = (try? JSONSerialization.data(
            withJSONObject: wrapped, options: [.sortedKeys]
        )) ?? Data()
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return "<tool_response>\n\(json)\n</tool_response>"
    }

    // MARK: - Tool declaration → Qwen JSON line

    /// One line of `<tools>` block — `{"type":"function","function":{…}}`.
    private static func toolJSONLine(_ t: ModelToolDeclaration) -> String {
        var props: [String: [String: Any]] = [:]
        var required: [String] = []
        for p in t.parameters {
            var pd: [String: Any] = [
                "type": p.type.jsonSchemaType,
                "description": p.description ?? ""
            ]
            if let ev = p.enumValues, !ev.isEmpty { pd["enum"] = ev }
            if p.nullable { pd["nullable"] = true }
            if p.type == .array {
                pd["items"] = ["type": "string"]  // we only ship string arrays today
            }
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
    /// Lowercase JSON-schema type name ("string" / "integer" / …).
    var jsonSchemaType: String {
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
