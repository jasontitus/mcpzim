// SPDX-License-Identifier: MIT
//
// Gemma 4's native tool-calling protocol uses a custom mini-format (NOT JSON)
// inside `<|tool_call>…<tool_call|>` and `<|tool_response>…<tool_response|>`
// blocks. The rules (extracted from the April-2026 Google template):
//
//   - string   → `<|"|>VALUE<|"|>`
//   - number   → decimal literal (no quotes)
//   - bool     → `true` / `false`
//   - null     → `null`
//   - object   → `{k1:v1,k2:v2,…}` (keys unquoted, values formatted recursively)
//   - array    → `[v1,v2,…]`
//
// Keys are sorted so output is stable (mirrors Python's `dictsort`).
//
// This file also emits Gemma-style tool *declarations* that go inside the
// `<|tool>…<tool|>` blocks of a system turn, derived from JSON-Schema-like
// parameter descriptors used by MCPZim's tool registry.

import Foundation

public enum Gemma4ToolFormat {

    // MARK: - Value formatting

    /// Gemma-format any JSON-decodable value. `NSNull` → "null".
    public static func formatValue(_ value: Any) -> String {
        if value is NSNull { return "null" }
        if let s = value as? String { return "<|\"|>\(s)<|\"|>" }
        if let b = value as? Bool { return b ? "true" : "false" }
        if let n = value as? NSNumber {
            // NSNumber's boolean-ness is already handled above via `as? Bool`,
            // but NSNumber carrying a Bool can still slip through on some
            // bridges — check one more time so booleans never render as 0/1.
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                return n.boolValue ? "true" : "false"
            }
            return n.stringValue
        }
        if let arr = value as? [Any] {
            return "[" + arr.map(formatValue).joined(separator: ",") + "]"
        }
        if let dict = value as? [String: Any] {
            let pairs = dict.keys.sorted().map { k in "\(k):\(formatValue(dict[k]!))" }
            return "{" + pairs.joined(separator: ",") + "}"
        }
        // Fallback: string-describe and quote.
        return "<|\"|>\(String(describing: value))<|\"|>"
    }

    // MARK: - Tool call emission (used by tests / future first-party agent)

    /// Format a tool call emitted by the model. The model usually produces
    /// this — we mainly use it in tests to round-trip.
    public static func formatToolCall(name: String, arguments: [String: Any]) -> String {
        let body = arguments.keys.sorted().map { k in
            "\(k):\(formatValue(arguments[k]!))"
        }.joined(separator: ",")
        return "<|tool_call>call:\(name){\(body)}<tool_call|>"
    }

    // MARK: - Tool response emission

    /// Format a tool's JSON response into Gemma's `<|tool_response>…<tool_response|>`
    /// block. Accepts dicts (keyed body) or scalars (wrapped as `{value:…}`).
    public static func formatToolResponse(name: String, payload: Any) -> String {
        let body: String
        if let dict = payload as? [String: Any] {
            body = dict.keys.sorted().map { k in
                "\(k):\(formatValue(dict[k]!))"
            }.joined(separator: ",")
        } else {
            body = "value:\(formatValue(payload))"
        }
        return "<|tool_response>response:\(name){\(body)}<tool_response|>"
    }

    // MARK: - Tool declaration (system turn)

    /// Minimal shape of a Gemma-4 tool declaration. `MCPZim`'s registry has
    /// richer JSON-schema data but for the end-to-end bring-up we stick to
    /// what Gemma actually reads: name, description, and parameter schemas
    /// (each with type, description, optional enum, optional nullable).
    public struct ToolDeclaration: Sendable {
        public struct Parameter: Sendable {
            public enum ParamType: String, Sendable {
                case string = "STRING"
                case integer = "INTEGER"
                case number = "NUMBER"
                case boolean = "BOOLEAN"
                case array = "ARRAY"
                case object = "OBJECT"
            }
            public let name: String
            public let type: ParamType
            public let description: String?
            public let required: Bool
            public let enumValues: [String]?
            public let nullable: Bool
            public init(
                name: String,
                type: ParamType,
                description: String? = nil,
                required: Bool = false,
                enumValues: [String]? = nil,
                nullable: Bool = false
            ) {
                self.name = name
                self.type = type
                self.description = description
                self.required = required
                self.enumValues = enumValues
                self.nullable = nullable
            }
        }
        public let name: String
        public let description: String
        public let parameters: [Parameter]

        public init(name: String, description: String, parameters: [Parameter]) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    /// Emit a single `<|tool>declaration:NAME{…}<tool|>` block exactly as
    /// Google's chat_template.jinja does. Properties are keyed by name,
    /// required params listed separately, no trailing newlines.
    public static func formatToolDeclaration(_ decl: ToolDeclaration) -> String {
        var body = "declaration:\(decl.name){"
        body += "description:<|\"|>\(decl.description)<|\"|>"

        if !decl.parameters.isEmpty {
            body += ",parameters:{"
            // properties:{KEY:{type:"T",description:"D",enum:[…],nullable:true},…}
            body += "properties:{"
            body += decl.parameters.map { p -> String in
                var attrs: [String] = []
                if let desc = p.description {
                    attrs.append("description:<|\"|>\(desc)<|\"|>")
                }
                if let ev = p.enumValues, !ev.isEmpty {
                    let list = ev.map { "<|\"|>\($0)<|\"|>" }.joined(separator: ",")
                    attrs.append("enum:[\(list)]")
                }
                if p.nullable {
                    attrs.append("nullable:true")
                }
                attrs.append("type:<|\"|>\(p.type.rawValue)<|\"|>")
                return "\(p.name):{\(attrs.joined(separator: ","))}"
            }.joined(separator: ",")
            body += "}"

            // required:["a","b"]
            let requiredNames = decl.parameters.filter { $0.required }.map(\.name)
            if !requiredNames.isEmpty {
                body += ",required:[" + requiredNames.map { "<|\"|>\($0)<|\"|>" }.joined(separator: ",") + "]"
            }
            body += ",type:<|\"|>OBJECT<|\"|>}"
        }

        body += "}"
        return "<|tool>\(body)<tool|>"
    }

    /// Emit the full `<|turn>system\n…<turn|>\n` preamble for a set of
    /// tools (plus an optional free-form system message). Returns an empty
    /// string if both are empty. Tool declarations are concatenated inline,
    /// matching the exact byte sequence the Google template produces.
    public static func formatSystemTurn(systemMessage: String, tools: [ToolDeclaration]) -> String {
        if systemMessage.isEmpty && tools.isEmpty { return "" }
        var body = "<|turn>system\n"
        if !systemMessage.isEmpty {
            body += systemMessage
            if !systemMessage.hasSuffix("\n") { body += "\n" }
        }
        for t in tools {
            body += formatToolDeclaration(t)
        }
        body += "<turn|>\n"
        return body
    }
}
