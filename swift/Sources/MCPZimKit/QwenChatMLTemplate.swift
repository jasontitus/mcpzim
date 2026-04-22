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

    public var logCategory: String { "Qwen" }

    public func formatSystemTurn(
        systemMessage: String, tools: [ModelToolDeclaration]
    ) -> String {
        var out = "<|im_start|>system\n"
        out += systemMessage
        // `/no_think` is Qwen 3's built-in soft switch that disables
        // the `<think>…</think>` reasoning scratchpad for this turn (and,
        // when in the system prompt, every subsequent turn). Without it
        // Qwen happily spends 300+ tokens deliberating "should I call
        // nearby_places or something else… or maybe…" before emitting
        // the actual tool call — burning KV cache (~hundreds of MB at
        // 6k-token prompts) and adding multi-second latency per turn.
        // We strip any CLOSED `<think>…</think>` in `stripReasoning`
        // anyway, but the tokens still paid cache memory before the
        // strip. Disabling at the source is cheaper than cleaning up
        // after. Keep as the final marker before the close tag so a
        // per-turn `/think` directive in a user message can still flip
        // it back on if we ever need richer reasoning (e.g. math).
        out += "\n\n/no_think"
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
        // On Qwen 3.5 (and later) the model's own chat template injects
        // `<think>\n\n</think>\n\n` right after the assistant open when
        // `enable_thinking=False` — an empty reasoning block that tells
        // the model "I've finished thinking, now answer". We mirror that
        // here because Qwen 3.5 ignores Qwen 3's `/no_think` soft
        // directive (which we still leave in the system turn as a
        // belt-and-braces for Qwen 3). Qwen 3 accepts the empty block
        // as well — it reads as a finished reasoning span and skips
        // its own `<think>…</think>`. Net effect across both families:
        // no reasoning scratchpad in the generated output, cutting
        // ~hundreds of tokens of KV per turn.
        out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return out
    }

    public func firstToolCall(in buffer: String) -> ToolCallMatch? {
        return firstToolCall(in: buffer, allowImplicitClose: false)
    }

    public func firstToolCallAfterClip(in buffer: String) -> ToolCallMatch? {
        return firstToolCall(in: buffer, allowImplicitClose: true)
    }

    /// Internal impl with an extra `allowImplicitClose` toggle. Streaming
    /// callers pass `false` (default) — they need strict framing so we
    /// don't fire a tool call the model is still in the middle of
    /// writing. The generation-loop post-stream cleanup pass calls with
    /// `true` to rescue Qwen 3.5 emissions that got clipped by the
    /// `<|im_end|>` stop marker mid-closer (observed: body ends with
    /// `</tool` or even bare JSON, no `</tool_call>` at all).
    public func firstToolCall(
        in buffer: String, allowImplicitClose: Bool
    ) -> ToolCallMatch? {
        guard let openMarker = buffer.range(of: "<tool_call>") else { return nil }
        let searchStart = openMarker.upperBound
        let closeRange: Range<String.Index>
        if let r = buffer.range(of: "</tool_call>", range: searchStart..<buffer.endIndex) {
            closeRange = r
        } else if allowImplicitClose {
            // Partial-close fallbacks — only safe AFTER the model has
            // stopped generating, otherwise we'd match whatever partial
            // JSON is mid-stream.
            if let r = buffer.range(of: "</tool", range: searchStart..<buffer.endIndex) {
                closeRange = r
            } else {
                // No closer at all — treat the entire tail as body.
                closeRange = buffer.endIndex..<buffer.endIndex
            }
        } else {
            return nil
        }
        let bodyStart = openMarker.upperBound
        let body = String(buffer[bodyStart..<closeRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // Some Qwen variants (Qwen 3.5 especially) emit JSON shapes
        // that don't quite match our prompt's "{name, arguments}"
        // template. Observed in the wild:
        //   {"name": "foo", "arguments": {...}}           ← canonical
        //   {"function": "foo", "arguments": {...}}       ← Qwen 3.5 flat
        //   {"function": {"name": "foo", "arguments": {...}}} ← nested
        //   {"function": {"arguments": {...}}}            ← broken (no name)
        // Try canonical first, then fall back to each alternate shape.
        // Tolerate a trailing or interior comma and a missing closing
        // brace from clipped Qwen output by attempting a one-shot
        // brace-balance repair before the second decode pass.
        guard let obj = Self.decodeObject(body) ?? Self.decodeObject(Self.repairJSON(body))
        else { return nil }

        // 1) canonical {"name": …, "arguments": …}
        if let name = obj["name"] as? String,
           let args = obj["arguments"] as? [String: Any]
        {
            return ToolCallMatch(
                range: openMarker.lowerBound..<closeRange.upperBound,
                name: name, arguments: args
            )
        }
        // 2) Qwen 3.5 flat: {"function": "name", "arguments": {…}}
        if let name = obj["function"] as? String,
           let args = obj["arguments"] as? [String: Any]
        {
            return ToolCallMatch(
                range: openMarker.lowerBound..<closeRange.upperBound,
                name: name, arguments: args
            )
        }
        // 3) Qwen 3.5 nested: {"function": {"name": …, "arguments": …}}
        if let inner = obj["function"] as? [String: Any],
           let name = inner["name"] as? String,
           let args = inner["arguments"] as? [String: Any]
        {
            return ToolCallMatch(
                range: openMarker.lowerBound..<closeRange.upperBound,
                name: name, arguments: args
            )
        }
        // 4) Qwen 3.5 OpenAI-ish:
        //    {"type": "function", "arguments": {"name": "X", "arguments": {…}}}
        //    The `name` + real `arguments` are nested one level deeper
        //    inside the outer `arguments`. Observed in the wild on
        //    `principled-intelligence/Qwen3.5-4B-text-only` bf16 for
        //    the `restaurants_in_sf` scenario.
        if obj["type"] as? String == "function",
           let inner = obj["arguments"] as? [String: Any],
           let name = inner["name"] as? String,
           let args = inner["arguments"] as? [String: Any]
        {
            return ToolCallMatch(
                range: openMarker.lowerBound..<closeRange.upperBound,
                name: name, arguments: args
            )
        }
        return nil
    }

    private static func decodeObject(_ s: String) -> [String: Any]? {
        guard let data = s.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    /// Best-effort repair for Qwen 3.5 tool-call JSON quirks seen on
    /// device. Applied only after a strict decode has already failed,
    /// so the wins are pure — no risk of masking a good parse.
    ///
    /// Observed emissions (real captures, all on Qwen 3.5 4-bit):
    ///   1) `{"name": "foo",,"arguments": {...,,...}}`
    ///      — stray double commas after strings. We collapse `,,` to
    ///        `,`.
    ///   2) `{"titles": ["A", "B",]}` / `{"x": 1,}`
    ///      — trailing comma before `]` or `}`. Strict JSON rejects
    ///        it; JavaScript accepts it, and Qwen leaks JS habits.
    ///   3) `["North Korea", " "South Korea"]`
    ///      — an extra whitespace-only string injected before a real
    ///        string. We drop the empty-whitespace string so the
    ///        array parses as the two values the model meant.
    ///   4) `{"function": {"name": "foo", "arguments": {…}` (clipped)
    ///      — brace-balance repair, same as before.
    ///
    /// Cheap, ordered, idempotent. If more than one missing closing
    /// brace is needed (deep clip), the caller still bails gracefully.
    private static func repairJSON(_ s: String) -> String {
        var out = s
        // (1) Double commas → single. Iterate to collapse `,,,` too.
        while out.contains(",,") {
            out = out.replacingOccurrences(of: ",,", with: ",")
        }
        // (2) Trailing commas inside `{…}` / `[…]`.
        out = out.replacingOccurrences(
            of: #",\s*([}\]])"#,
            with: "$1",
            options: .regularExpression
        )
        // (3) Whitespace-only string wedged before an unquoted
        //     bareword: `" "South Korea"` → `"South Korea"`. The
        //     model split one string into two by emitting an extra
        //     `" "` after the opening quote, so the rescue is to
        //     collapse the `"…whitespace…"` into a single `"`,
        //     re-joining the closing quote of the bogus slot with
        //     the bareword that follows. Lookahead gates the repair
        //     on the next char being a non-delimiter — intentional
        //     whitespace values like `{"a": " "}` (followed by `}`)
        //     or `["x", " ", "y"]` (followed by `,`) are left alone.
        out = out.replacingOccurrences(
            of: #""\s+"(?=[^"\s,\]\}])"#,
            with: "\"",
            options: .regularExpression
        )
        // (4) Brace-balance clip repair (original behaviour).
        let opens = out.filter { $0 == "{" }.count
        let closes = out.filter { $0 == "}" }.count
        if opens > closes {
            out += String(repeating: "}", count: opens - closes)
        }
        return out
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

    public func formatToolCall(name: String, arguments: [String: Any]) -> String {
        // ChatML tool-call emission — the `<tool_call>…</tool_call>`
        // block wrapped around a JSON object. Matches what
        // `firstToolCall(in:)` parses back. No leading `<|im_start|>`
        // here — the caller injects this as the body of an assistant
        // turn, and the transcript renderer wraps it.
        let body: [String: Any] = [
            "name": name,
            "arguments": arguments,
        ]
        let data = (try? JSONSerialization.data(
            withJSONObject: body, options: [.sortedKeys]
        )) ?? Data()
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return "<tool_call>\n\(json)\n</tool_call>"
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
