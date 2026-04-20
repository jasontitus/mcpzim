// SPDX-License-Identifier: MIT
//
// Pure-data turn type shared by the iOS app, by any future macOS/CLI host,
// and by the chat-template renderers (e.g. `Gemma4PromptTemplate`).

import Foundation

public struct ChatTurn: Sendable, Equatable {
    public enum Role: String, Sendable, Equatable {
        case user, assistant, tool, system
    }
    public let role: Role
    public let text: String
    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}
