// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class Gemma4PromptTemplateTests: XCTestCase {
    func testFoldsSystemPreambleIntoFirstUserTurn() {
        let prompt = Gemma4PromptTemplate.render(
            systemPreamble: "You are helpful.",
            turns: [ChatTurn(role: .user, text: "Hi")]
        )
        // Body is followed immediately by <turn|> (no intervening newline) to
        // match Gemma4PromptFormatter's canonical output.
        XCTAssertEqual(prompt, "<bos><|turn>user\nYou are helpful.\nHi<turn|>\n<|turn>model\n")
    }

    func testRendersAssistantAsModelRole() {
        let prompt = Gemma4PromptTemplate.render(
            systemPreamble: "",
            turns: [
                ChatTurn(role: .user, text: "Hi"),
                ChatTurn(role: .assistant, text: "Hello!"),
                ChatTurn(role: .user, text: "Bye"),
            ]
        )
        XCTAssertEqual(
            prompt,
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n"
        )
    }

    func testMatchesGemma4PromptFormatterReferenceForSingleUserTurn() {
        // `Gemma4PromptFormatter.userTurn(msg)` returns:
        //   "<bos><|turn>user\n{msg}<turn|>\n<|turn>model\n"
        // We must produce the same string, byte-for-byte, so `container.encode`
        // tokenizes us identically to a call site that used the reference.
        let message = "tell me about baseball"
        let ours = Gemma4PromptTemplate.render(
            systemPreamble: "",
            turns: [ChatTurn(role: .user, text: message)]
        )
        let reference = "<bos><|turn>user\n\(message)<turn|>\n<|turn>model\n"
        XCTAssertEqual(ours, reference)
    }

    func testToolTurnWrapsInToolResponseBlock() {
        // Caller pre-formats the tool payload via `Gemma4ToolFormat.formatToolResponse`.
        let toolResp = Gemma4ToolFormat.formatToolResponse(name: "search", payload: ["hits": [] as [Any]])
        let prompt = Gemma4PromptTemplate.render(
            systemPreamble: "",
            turns: [
                ChatTurn(role: .user, text: "search"),
                ChatTurn(role: .assistant, text: "<|tool_call>call:search{}<tool_call|>"),
                ChatTurn(role: .tool, text: toolResp),
            ]
        )
        XCTAssertTrue(prompt.contains("<|turn>user\n<|tool_response>response:search"))
        XCTAssertTrue(prompt.hasSuffix("<|turn>model\n"))
    }

    func testSystemTurnOverloadEmitsToolDecls() {
        let decl = Gemma4ToolFormat.ToolDeclaration(
            name: "list_libraries",
            description: "List loaded ZIMs.",
            parameters: []
        )
        let prompt = Gemma4PromptTemplate.render(
            systemMessage: "You are helpful.",
            tools: [decl],
            turns: [ChatTurn(role: .user, text: "hi")]
        )
        XCTAssertTrue(prompt.contains("<|turn>system\nYou are helpful.\n<|tool>declaration:list_libraries{"))
        XCTAssertTrue(prompt.contains("<|turn>user\nhi<turn|>"))
    }

    func testTrailingSystemAttachesToBareUserTurn() {
        let prompt = Gemma4PromptTemplate.render(
            systemPreamble: "",
            turns: [ChatTurn(role: .system, text: "Follow safety rules.")]
        )
        XCTAssertEqual(prompt, "<bos><|turn>user\nFollow safety rules.<turn|>\n<|turn>model\n")
    }

    func testStartsWithBos() {
        let prompt = Gemma4PromptTemplate.render(systemPreamble: "preamble", turns: [])
        XCTAssertTrue(prompt.hasPrefix("<bos>"))
    }

    func testEndsOnOpenModelTurn() {
        let prompt = Gemma4PromptTemplate.render(systemPreamble: "preamble", turns: [])
        XCTAssertTrue(prompt.hasSuffix("<|turn>model\n"))
    }

    func testConcatenatesMultipleSystemTurns() {
        let prompt = Gemma4PromptTemplate.render(
            systemPreamble: "line 1",
            turns: [
                ChatTurn(role: .system, text: "line 2"),
                ChatTurn(role: .user, text: "ok"),
            ]
        )
        XCTAssertTrue(prompt.contains("line 1\nline 2\nok"))
    }
}
