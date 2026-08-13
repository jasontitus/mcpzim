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

    func testCarriageReturnsStillFoldIntoLineBreaks() {
        // `parse` now skips the \r\n normalisation copies unless a CR is
        // actually present (PI review 2026-08-13, perf #2); pin the branch
        // that still has to do the work.
        XCTAssertEqual(MarkdownMessageParser.parse("# Title\r\n\r\nBody.\rMore."), [
            .heading(level: 1, text: "Title"),
            .paragraph("Body.\nMore."),
        ])
    }
}

/// Perf #2 of the 2026-08-13 review replaced the whole-buffer strip behind
/// `MessageRow.displayText` with one that runs the ICU passes over only the
/// head of the buffer that can still contain a sentinel. The saving is real
/// but the equivalence is an argument about the patterns, not something the
/// compiler checks — so pin the bounded path against the unbounded pipeline
/// it was derived from. Any divergence here is a visible regression in a
/// settled reply. (Lives in this file because it is the chat renderer's
/// only test bundle entry point.)
final class AssistantMarkupStripperTests: XCTestCase {
    /// What `displayText` produced before the bounded head existed: the
    /// full pipeline over the whole buffer, then the trim.
    private func reference(_ raw: String) -> String {
        AssistantMarkupStripper.stripMarkup(raw).text
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func assertBounded(
        _ raw: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(
            AssistantMarkupStripper.displayText(raw), reference(raw),
            "bounded strip diverged on \(String(reflecting: raw))",
            file: file, line: line)
    }

    private let narration = String(
        repeating: "The collider accelerates protons to near light speed. ",
        count: 200)

    func testBoundedStripMatchesFullPipelineOnStreamingShapes() {
        let cases = [
            "",
            "A normal answer.",
            "  padded answer  \n",
            // The shape the bounded head exists for: closed tool-call
            // markup, then tens of KB of prose with no sentinel in it.
            "<|tool_call>call:get_article{title:<|\"|>LHC<|\"|>}<tool_call|>" + narration,
            "<tool_call>{\"name\":\"near_places\"}</tool_call>\n\n" + narration,
            "<|tool_response>{\"ok\":true}<tool_response|>" + narration,
            "<think>weighing the options</think>" + narration,
            "scratchpad reasoning</think>" + narration,
            // Truncating shapes — the tail the fast path held back has to
            // disappear along with everything from the opener.
            narration + "<|tool_call>call:search{query:",
            narration + "<think",
            narration + "<tool",
            // Sentinels straddling / abutting the window boundary.
            "abc<to" + narration,
            "abc<|tool_response>x<tool_response|>ol_call|>" + narration,
            "<think>a</think>b<to" + narration,
            narration + "<",
            "<",
            "<|\"",
            // Prose that legitimately contains an angle bracket.
            "For n < 10 the sum converges. " + narration,
            narration + " and 5 < 6 wraps up.",
            // Multi-byte content either side of the split point.
            "<|tool_call>call:x{}<tool_call|>café ☕️ 日本語 " + narration,
            "🇯🇵🇯🇵🇯🇵<tool_call>y</tool_call>🇯🇵🇯🇵🇯🇵" + narration,
        ]
        for raw in cases { assertBounded(raw) }
    }

    func testBoundedStripMatchesFullPipelineUnderFuzz() {
        let tokens = [
            "<", ">", "|", "/", "\"", " ", "\n", "think", "tool", "_call",
            "_response", "<think>", "</think>", "<|tool_call>", "<tool_call|>",
            "<tool_call>", "</tool_call>", "<|tool_response>",
            "<tool_response|>", "<|\"|>", "<to", "<|to", "prose ", "abc",
        ]
        // Deterministic LCG so a failure is reproducible from the seed.
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next(_ bound: Int) -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((seed >> 33) % UInt64(bound))
        }
        for _ in 0..<600 {
            var raw = ""
            for _ in 0...next(20) { raw += tokens[next(tokens.count)] }
            // Half the cases get a long sentinel-free tail — the region the
            // bounded head is allowed to skip.
            if next(2) == 0 { raw += narration }
            for _ in 0...next(6) { raw += tokens[next(tokens.count)] }
            assertBounded(raw)
        }
    }
}
