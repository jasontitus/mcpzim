// SPDX-License-Identifier: MIT
//
// Model-family abstraction. Different on-device LLMs drive tool-calling
// through incompatible mini-languages: Gemma 4 emits a custom
// `<|tool_call>call:NAME{k:v,…}<tool_call|>` block inside `<|turn>` /
// `<turn|>` chat markers; Qwen (2.x / 2.5 / 3) uses ChatML
// `<|im_start|>` / `<|im_end|>` turns with JSON tool calls wrapped in
// `<tool_call>{…}</tool_call>`; Llama 3 again different. The app needs
// one code path that drives *any* of them; everything model-specific
// sits behind this protocol.
//
// The HOST (ChatSession + MCPToolAdapter) owns:
//   - the semantic tool registry (names, descriptions, JSON schemas)
//   - the preamble prose
//   - the tool dispatch logic
//
// The TEMPLATE owns:
//   - turn markers + BOS / stop tokens
//   - how tool declarations appear in the system turn
//   - how the model's output tool calls are parsed back into Swift
//   - how a tool response turn is rendered for the next generation
//
// v1 lives in-process. Gemma4Template is a thin adapter over the
// existing `Gemma4PromptTemplate` / `Gemma4ToolFormat` /
// `Gemma4ToolCallParser`; a future `QwenChatMLTemplate` will implement
// the same protocol with ChatML + JSON tool calls.

import Foundation

/// Generic tool declaration. The host builds one per tool it exposes;
/// the template renders them in the model's native format.
public struct ModelToolDeclaration: Sendable {
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
            name: String, type: ParamType,
            description: String? = nil, required: Bool = false,
            enumValues: [String]? = nil, nullable: Bool = false
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

/// One parsed tool call from the model's streamed output.
public struct ToolCallMatch {
    /// Range in the scanned buffer the call occupies. Used to splice
    /// the call out and continue streaming.
    public let range: Range<String.Index>
    public let name: String
    public let arguments: [String: Any]
    public init(range: Range<String.Index>, name: String, arguments: [String: Any]) {
        self.range = range
        self.name = name
        self.arguments = arguments
    }
}

/// Per-model-family rendering + parsing for tool-using chat turns.
/// See file header for the responsibility split between host / template.
public protocol ModelTemplate: Sendable {

    /// Prompt prefix. Gemma emits `<bos>\n`. ChatML emits empty / BOS token.
    var bos: String { get }

    /// Turn-close substrings the streaming decoder should halt on.
    /// Gemma: `["<turn|>", "<|turn>"]` (both conventional + misorder).
    /// Qwen ChatML: `["<|im_end|>"]`.
    var stopMarkers: [String] { get }

    /// Render the system turn + tool declarations. Output is a complete
    /// chat turn the model will consume, including any turn close marker.
    func formatSystemTurn(systemMessage: String, tools: [ModelToolDeclaration]) -> String

    /// Render the full transcript: system preamble + tool declarations
    /// (if `tools` non-empty) + all chat turns + the assistant-open tag
    /// so generation starts in assistant mode.
    func renderTranscript(
        systemPreamble: String,
        tools: [ModelToolDeclaration],
        turns: [ChatTurn]
    ) -> String

    /// Scan a streaming output buffer for the first complete tool call.
    /// Returns nil while still buffering; returns `ToolCallMatch` with
    /// the byte range + parsed call once we have a full block.
    func firstToolCall(in buffer: String) -> ToolCallMatch?

    /// Post-stream variant — called once after generation finishes
    /// without the streaming parser having matched a tool call. Allows
    /// the template to be lenient about partial closers (Qwen 3.5
    /// sometimes gets clipped by `<|im_end|>` mid-`</tool_call>`,
    /// leaving `</tool` or bare JSON). Default impl delegates to the
    /// strict streaming version.
    func firstToolCallAfterClip(in buffer: String) -> ToolCallMatch?

    /// Render an assistant tool-call emission in the model's native
    /// wire format — inverse of `firstToolCall`. Used by the fast-path
    /// dispatcher when it wants to inject a synthetic round-trip
    /// (classifier-chosen tool + adapter-dispatched result) into the
    /// transcript so the next generate() pass summarises the tool's
    /// output instead of re-emitting the tool call. Keeping this
    /// template-owned means the emission matches byte-for-byte what
    /// the model would have produced — no LCP drift at the KV cache.
    func formatToolCall(name: String, arguments: [String: Any]) -> String

    /// Render a tool-response turn to feed back to the model. Ends with
    /// the assistant-open tag so the next `generate(...)` resumes in
    /// assistant mode.
    func formatToolResponse(name: String, payload: [String: Any]) -> String

    /// Remove any reasoning / scratchpad text the model emits before
    /// its user-facing answer (Qwen 3's `<think>…</think>`, DeepSeek's
    /// `<reasoning>…</reasoning>`, etc.). Returns the cleaned assistant
    /// text. Default: pass-through (most families don't have a
    /// reasoning mode). Called against the final assistant message
    /// before it lands in the chat bubble and against streaming
    /// buffers (so only *closed* reasoning spans are stripped — any
    /// open `<think>` without a close stays visible to avoid flashing
    /// mid-stream).
    func stripReasoning(_ text: String) -> String

    /// Human-friendly log category for the model family (e.g. `"Gemma4"`,
    /// `"Qwen"`, `"Llama3"`). Used by the host's debug pane to tag lines
    /// with the *actual* family being used — helpful now that one
    /// provider class hosts multiple families via the template slot.
    var logCategory: String { get }
}

public extension ModelTemplate {
    func stripReasoning(_ text: String) -> String { text }
    var logCategory: String { "LLM" }
    func firstToolCallAfterClip(in buffer: String) -> ToolCallMatch? {
        firstToolCall(in: buffer)
    }
}

// MARK: - Gemma 4 implementation

/// `ModelTemplate` for Gemma 4 / Gemma 4 Text. Delegates to the existing
/// `Gemma4PromptTemplate` / `Gemma4ToolFormat` / `Gemma4ToolCallParser`
/// so behaviour is byte-for-byte identical to the pre-abstraction code.
public struct Gemma4Template: ModelTemplate {

    public init() {}

    public var bos: String { Gemma4PromptTemplate.bos }

    public var stopMarkers: [String] { ["<turn|>", "<|turn>"] }

    public var logCategory: String { "Gemma4" }

    public func formatSystemTurn(
        systemMessage: String, tools: [ModelToolDeclaration]
    ) -> String {
        Gemma4ToolFormat.formatSystemTurn(
            systemMessage: systemMessage,
            tools: tools.map { $0.asGemma4 }
        )
    }

    public func renderTranscript(
        systemPreamble: String,
        tools: [ModelToolDeclaration],
        turns: [ChatTurn]
    ) -> String {
        if tools.isEmpty {
            return Gemma4PromptTemplate.render(systemPreamble: systemPreamble, turns: turns)
        }
        return Gemma4PromptTemplate.render(
            systemMessage: systemPreamble,
            tools: tools.map { $0.asGemma4 },
            turns: turns
        )
    }

    public func firstToolCall(in buffer: String) -> ToolCallMatch? {
        guard let m = Gemma4ToolCallParser.firstCall(in: buffer) else { return nil }
        return ToolCallMatch(range: m.range, name: m.name, arguments: m.arguments)
    }

    public func formatToolCall(name: String, arguments: [String: Any]) -> String {
        Gemma4ToolFormat.formatToolCall(name: name, arguments: arguments)
    }

    public func formatToolResponse(name: String, payload: [String: Any]) -> String {
        Gemma4ToolFormat.formatToolResponse(name: name, payload: payload)
    }
}

// MARK: - Generic → Gemma-specific conversion helper

extension ModelToolDeclaration {
    /// Convert to `Gemma4ToolFormat.ToolDeclaration` for
    /// `Gemma4Template`'s delegating implementation.
    var asGemma4: Gemma4ToolFormat.ToolDeclaration {
        let params: [Gemma4ToolFormat.ToolDeclaration.Parameter] =
            parameters.map { p in
                let t: Gemma4ToolFormat.ToolDeclaration.Parameter.ParamType = {
                    switch p.type {
                    case .string:  return .string
                    case .integer: return .integer
                    case .number:  return .number
                    case .boolean: return .boolean
                    case .array:   return .array
                    case .object:  return .object
                    }
                }()
                return .init(
                    name: p.name, type: t,
                    description: p.description,
                    required: p.required,
                    enumValues: p.enumValues,
                    nullable: p.nullable
                )
            }
        return .init(name: name, description: description, parameters: params)
    }
}
