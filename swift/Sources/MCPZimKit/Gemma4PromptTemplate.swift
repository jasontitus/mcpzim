// SPDX-License-Identifier: MIT
//
// Renders a `[ChatTurn]` transcript into Gemma 4's native chat template.
// Gemma 4 uses the `<|turn>role\n...<turn|>\n` protocol (note: pipe
// placement differs between open and close — *not* `<start_of_turn>` /
// `<end_of_turn>`, which is Gemma 2/3). The tokenizer registers `<bos>`,
// `<|turn>`, and `<turn|>` as special tokens, so as long as we pass this
// string through `Tokenizer.encode(text:)` (the path `Gemma4PromptFormatter`
// uses internally) they'll tokenize to single IDs rather than BPE byte
// sequences.
//
// Gemma 4 recognises only two first-class roles — `user` and `model`:
//   - system turns (including the caller-supplied tool preamble) are folded
//     into the next user turn, since the model has no dedicated system
//     token.
//   - tool turns are emitted as user turns whose body is a
//     `<tool_response>…</tool_response>` block, matching the convention
//     used by tool-use fine-tunes.
//
// The output always ends on `<|turn>model\n` so callers can stream it into
// `ModelContainer.generate(…)` and have the model continue the current
// assistant reply.

import Foundation

public enum Gemma4PromptTemplate {
    public static let bos = "<bos>"
    public static let turnOpen = "<|turn>"
    public static let turnClose = "<turn|>\n"

    public static func userTurnOpen(_ role: String = "user") -> String {
        "\(turnOpen)\(role)\n"
    }

    /// Simple overload with no tool declarations — fold a string preamble
    /// into the first user turn (legacy behavior for the initial bring-up).
    public static func render(systemPreamble: String, turns: [ChatTurn]) -> String {
        renderImpl(systemMessage: systemPreamble, tools: [], foldSystemIntoFirstUser: true, turns: turns)
    }

    /// Full overload. Renders a proper `<|turn>system\n…<turn|>` block when
    /// `tools` or `systemMessage` is non-empty — matches Gemma 4's native
    /// tool-calling protocol.
    public static func render(
        systemMessage: String,
        tools: [Gemma4ToolFormat.ToolDeclaration],
        turns: [ChatTurn]
    ) -> String {
        renderImpl(systemMessage: systemMessage, tools: tools, foldSystemIntoFirstUser: false, turns: turns)
    }

    private static func renderImpl(
        systemMessage: String,
        tools: [Gemma4ToolFormat.ToolDeclaration],
        foldSystemIntoFirstUser: Bool,
        turns: [ChatTurn]
    ) -> String {
        var out = bos
        // Tool declarations always go in their own system turn — Gemma 4's
        // trained protocol expects them there, not folded into a user turn.
        if !tools.isEmpty || (!foldSystemIntoFirstUser && !systemMessage.isEmpty) {
            out += Gemma4ToolFormat.formatSystemTurn(
                systemMessage: foldSystemIntoFirstUser ? "" : systemMessage,
                tools: tools
            )
        }
        var pendingUserPrefix = foldSystemIntoFirstUser ? systemMessage : ""

        func flushUser(_ body: String) {
            var text = body
            if !pendingUserPrefix.isEmpty {
                text = pendingUserPrefix + "\n" + text
                pendingUserPrefix = ""
            }
            out += userTurnOpen("user") + text + turnClose
        }

        for turn in turns {
            switch turn.role {
            case .system:
                if foldSystemIntoFirstUser {
                    pendingUserPrefix += pendingUserPrefix.isEmpty ? turn.text : "\n" + turn.text
                } else {
                    // Treat as an extra mid-conversation system turn.
                    out += Gemma4ToolFormat.formatSystemTurn(systemMessage: turn.text, tools: [])
                }
            case .user:
                flushUser(turn.text)
            case .tool:
                // Tool responses come back as Gemma's user-side
                // `<|tool_response>…<tool_response|>` block — already formatted
                // by the caller via `Gemma4ToolFormat.formatToolResponse(…)`.
                flushUser(turn.text)
            case .assistant:
                out += userTurnOpen("model") + turn.text + turnClose
            }
        }

        if !pendingUserPrefix.isEmpty {
            out += userTurnOpen("user") + pendingUserPrefix + turnClose
        }

        // Open the next model turn so the caller's `generate()` continues it.
        out += userTurnOpen("model")
        return out
    }
}
