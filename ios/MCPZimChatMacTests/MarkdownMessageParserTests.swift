// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimChatMac

final class MarkdownMessageParserTests: XCTestCase {
    func testParsesCommonModelReplyStructure() {
        let source = """
        ## What is studied?

        Particle physics includes **matter** and `force carriers`.

        - Fermions
          - Quarks
        1. Build an accelerator
        2. Record the collision

        > Evidence should remain grounded.

        ```swift
        let answer = 42
        ```
        """

        XCTAssertEqual(MarkdownMessageParser.parse(source), [
            .heading(level: 2, text: "What is studied?"),
            .paragraph("Particle physics includes **matter** and `force carriers`."),
            .unorderedItem(depth: 0, text: "Fermions"),
            .unorderedItem(depth: 1, text: "Quarks"),
            .orderedItem(depth: 0, number: "1", text: "Build an accelerator"),
            .orderedItem(depth: 0, number: "2", text: "Record the collision"),
            .quote("Evidence should remain grounded."),
            .code(language: "swift", text: "let answer = 42"),
        ])
    }

    func testParsesMarkdownTableWithoutShowingSeparatorRow() {
        let source = """
        | Runtime | Use |
        |:--|---:|
        | llama.cpp | Bonsai |
        | MLX | Kokoro |
        """

        XCTAssertEqual(MarkdownMessageParser.parse(source), [
            .table([
                ["Runtime", "Use"],
                ["llama.cpp", "Bonsai"],
                ["MLX", "Kokoro"],
            ]),
        ])
    }

    func testKeepsUnclosedFenceReadableDuringStreaming() {
        XCTAssertEqual(MarkdownMessageParser.parse("```json\n{\"ready\": true}"), [
            .code(language: "json", text: "{\"ready\": true}"),
        ])
    }

    func testPlainTextRemainsPlainParagraph() {
        XCTAssertEqual(MarkdownMessageParser.parse("A normal answer."), [
            .paragraph("A normal answer."),
        ])
    }
}
