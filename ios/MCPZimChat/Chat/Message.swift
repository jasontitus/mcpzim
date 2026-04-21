// SPDX-License-Identifier: MIT

import Foundation
import MCPZimKit

public struct ToolRoundTripEntry: Equatable, Sendable {
    public let assistantEmission: String
    public let toolResponseTurn: String
    public init(assistantEmission: String, toolResponseTurn: String) {
        self.assistantEmission = assistantEmission
        self.toolResponseTurn = toolResponseTurn
    }
}

public struct ChatMessage: Identifiable, Equatable, Sendable {
    public enum Role: String, Sendable, Equatable {
        case user, assistant, tool, system
    }

    public let id: UUID
    public let role: Role
    public var text: String
    /// When the assistant's turn contained tool calls, we stash a compact
    /// summary here so the UI can render a collapsible "Tools used" chip.
    public var toolCalls: [ToolCallTrace]
    /// Set on the assistant placeholder when the user's question was
    /// submitted; the UI renders `finishedAt − startedAt` as a
    /// friendly "⌛ 4.2 s" subtitle under the reply bubble.
    public var startedAt: Date?
    /// Set when the whole generation + tool loop + final streaming
    /// completes.
    public var finishedAt: Date?
    /// For assistant messages that involved one or more tool calls,
    /// this carries the **exact** intermediate round-trip text so a
    /// follow-up turn can rebuild the full prompt byte-for-byte and
    /// the KV cache's LCP match survives. Each entry captures:
    ///   • `assistantEmission` — the `<|tool_call>call:X{...}<tool_call|>`
    ///     text the sampler emitted (plus any trailing chars the
    ///     final sampler token decoded to).
    ///   • `toolResponseTurn` — the formatted `<|tool_response>…<|tool_response|>`
    ///     body the ChatSession fed back to the model.
    /// Empty for pure-reply assistant turns and for non-assistant roles.
    public var toolRoundTrips: [ToolRoundTripEntry] = []

    public init(id: UUID = UUID(), role: Role, text: String = "",
                toolCalls: [ToolCallTrace] = [],
                startedAt: Date? = nil, finishedAt: Date? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.toolCalls = toolCalls
        self.startedAt = startedAt
        self.finishedAt = finishedAt
    }

    public var elapsed: TimeInterval? {
        guard let s = startedAt, let f = finishedAt else { return nil }
        return max(0, f.timeIntervalSince(s))
    }
}

extension ChatMessage.Role {
    var asChatTurnRole: ChatTurn.Role {
        switch self {
        case .user:      return .user
        case .assistant: return .assistant
        case .tool:      return .tool
        case .system:    return .system
        }
    }
}

public struct ToolCallTrace: Identifiable, Equatable, Sendable {
    public let id = UUID()
    public let name: String
    public let arguments: String    // JSON-encoded
    /// JSON the model actually sees — polylines summarized, long text
    /// truncated, etc. Used by the debug/tool-trace UI.
    public let result: String
    /// JSON of the full untrimmed tool output. For routing tools, this
    /// carries the complete `polyline`/`roads`/`turn_by_turn` — needed by
    /// `RouteMapView` to draw an actual map without a re-dispatch.
    public let rawResult: String
    public let error: String?

    public init(
        name: String,
        arguments: String,
        result: String,
        rawResult: String? = nil,
        error: String? = nil
    ) {
        self.name = name
        self.arguments = arguments
        self.result = result
        self.rawResult = rawResult ?? result
        self.error = error
    }

    public var succeeded: Bool { error == nil }
}
