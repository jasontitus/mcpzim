// SPDX-License-Identifier: MIT

import Foundation

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

public struct ToolCallTrace: Identifiable, Equatable, Sendable {
    public let id = UUID()
    public let name: String
    public let arguments: String    // JSON-encoded
    public let result: String       // JSON-encoded
    public let error: String?

    public var succeeded: Bool { error == nil }
}
