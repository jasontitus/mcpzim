// SPDX-License-Identifier: MIT

import Foundation
import MCPZimKit

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

    public init(id: UUID = UUID(), role: Role, text: String = "", toolCalls: [ToolCallTrace] = []) {
        self.id = id
        self.role = role
        self.text = text
        self.toolCalls = toolCalls
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
