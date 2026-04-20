// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class Gemma4ToolFormatTests: XCTestCase {
    // MARK: - Value formatting

    func testStringFormatsInSpecialQuotes() {
        XCTAssertEqual(Gemma4ToolFormat.formatValue("hello"), "<|\"|>hello<|\"|>")
    }

    func testBoolsRenderAsKeywords() {
        XCTAssertEqual(Gemma4ToolFormat.formatValue(true), "true")
        XCTAssertEqual(Gemma4ToolFormat.formatValue(false), "false")
    }

    func testNullRendersAsKeyword() {
        XCTAssertEqual(Gemma4ToolFormat.formatValue(NSNull()), "null")
    }

    func testIntegersAndFloats() {
        XCTAssertEqual(Gemma4ToolFormat.formatValue(42), "42")
        XCTAssertEqual(Gemma4ToolFormat.formatValue(3.14), "3.14")
    }

    func testNestedObjectKeysSorted() {
        let v: [String: Any] = ["zed": 1, "alpha": "one", "mid": [true, 2]]
        XCTAssertEqual(
            Gemma4ToolFormat.formatValue(v),
            "{alpha:<|\"|>one<|\"|>,mid:[true,2],zed:1}"
        )
    }

    // MARK: - Tool-call emission

    func testToolCallFormat() {
        let out = Gemma4ToolFormat.formatToolCall(
            name: "search",
            arguments: ["query": "Mount Everest", "limit": 3]
        )
        XCTAssertEqual(
            out,
            "<|tool_call>call:search{limit:3,query:<|\"|>Mount Everest<|\"|>}<tool_call|>"
        )
    }

    // MARK: - Tool-response emission

    func testToolResponseWrapsDict() {
        let out = Gemma4ToolFormat.formatToolResponse(
            name: "list_libraries",
            payload: ["count": 1, "zims": ["wikipedia_en_100_mini.zim"]]
        )
        XCTAssertEqual(
            out,
            "<|tool_response>response:list_libraries{count:1,zims:[<|\"|>wikipedia_en_100_mini.zim<|\"|>]}<tool_response|>"
        )
    }

    func testToolResponseWrapsScalarAsValue() {
        let out = Gemma4ToolFormat.formatToolResponse(name: "health", payload: "ok")
        XCTAssertEqual(out, "<|tool_response>response:health{value:<|\"|>ok<|\"|>}<tool_response|>")
    }

    // MARK: - Declaration emission

    func testToolDeclarationSerializesParameters() {
        let decl = Gemma4ToolFormat.ToolDeclaration(
            name: "search",
            description: "Full-text search across the loaded ZIMs.",
            parameters: [
                .init(name: "query", type: .string, description: "what to search for", required: true),
                .init(name: "limit", type: .integer, required: false),
                .init(name: "kind", type: .string, required: false,
                      enumValues: ["wikipedia", "mdwiki", "streetzim"]),
            ]
        )
        let out = Gemma4ToolFormat.formatToolDeclaration(decl)
        XCTAssertTrue(out.hasPrefix("<|tool>declaration:search{"))
        XCTAssertTrue(out.hasSuffix("<tool|>"))
        XCTAssertTrue(out.contains("description:<|\"|>Full-text search across the loaded ZIMs.<|\"|>"))
        XCTAssertTrue(out.contains("parameters:{properties:{"))
        XCTAssertTrue(out.contains("query:{description:<|\"|>what to search for<|\"|>,type:<|\"|>STRING<|\"|>}"))
        XCTAssertTrue(out.contains("enum:[<|\"|>wikipedia<|\"|>,<|\"|>mdwiki<|\"|>,<|\"|>streetzim<|\"|>]"))
        XCTAssertTrue(out.contains("required:[<|\"|>query<|\"|>]"))
        XCTAssertTrue(out.contains("type:<|\"|>OBJECT<|\"|>}"))
    }

    func testToolDeclarationNoParameters() {
        let decl = Gemma4ToolFormat.ToolDeclaration(
            name: "list_libraries",
            description: "List loaded ZIMs.",
            parameters: []
        )
        let out = Gemma4ToolFormat.formatToolDeclaration(decl)
        XCTAssertEqual(
            out,
            "<|tool>declaration:list_libraries{description:<|\"|>List loaded ZIMs.<|\"|>}<tool|>"
        )
    }

    // MARK: - Round-trip with the parser

    func testRoundTripSimpleCall() throws {
        let call = Gemma4ToolFormat.formatToolCall(
            name: "search",
            arguments: ["query": "kangaroo"]
        )
        let buffer = "Sure, let me search. \(call) And then…"
        let match = try XCTUnwrap(Gemma4ToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "search")
        XCTAssertEqual(match.arguments["query"] as? String, "kangaroo")
    }

    func testRoundTripMixedTypes() throws {
        let call = Gemma4ToolFormat.formatToolCall(
            name: "plan_route",
            arguments: [
                "from": "A",
                "to": "B",
                "distance": 12.5,
                "via": ["X", "Y"],
                "cached": true,
                "zim": NSNull(),
            ]
        )
        let match = try XCTUnwrap(Gemma4ToolCallParser.firstCall(in: call))
        XCTAssertEqual(match.name, "plan_route")
        XCTAssertEqual(match.arguments["from"] as? String, "A")
        XCTAssertEqual(match.arguments["to"] as? String, "B")
        XCTAssertEqual(match.arguments["distance"] as? Double, 12.5)
        XCTAssertEqual(match.arguments["via"] as? [String], ["X", "Y"])
        XCTAssertEqual(match.arguments["cached"] as? Bool, true)
        XCTAssertTrue(match.arguments["zim"] is NSNull)
    }

    func testParserReturnsNilOnMalformedCall() {
        XCTAssertNil(Gemma4ToolCallParser.firstCall(in: "<|tool_call>call:missing brace<tool_call|>"))
        XCTAssertNil(Gemma4ToolCallParser.firstCall(in: "<|tool_call>not a call format<tool_call|>"))
    }

    func testParserToleratesParenWrapAndArrow() throws {
        // E2B sometimes echoes the declaration shape when emitting a call.
        // The parser should still recover name and empty arguments.
        let weird = "<|tool_call>call:list_libraries({}) -> OBJECT<tool_call|>"
        let match = try XCTUnwrap(Gemma4ToolCallParser.firstCall(in: weird))
        XCTAssertEqual(match.name, "list_libraries")
        XCTAssertTrue(match.arguments.isEmpty)
    }

    func testParserToleratesParenWrapWithArgs() throws {
        let weird = "<|tool_call>call:search({query:<|\"|>kangaroo<|\"|>}) -> OBJECT<tool_call|>"
        let match = try XCTUnwrap(Gemma4ToolCallParser.firstCall(in: weird))
        XCTAssertEqual(match.name, "search")
        XCTAssertEqual(match.arguments["query"] as? String, "kangaroo")
    }

    func testParserReturnsNilWhenCloseMissing() {
        XCTAssertNil(Gemma4ToolCallParser.firstCall(in: "<|tool_call>call:search{q:<|\"|>kan"))
    }

    // MARK: - System turn (preamble)

    func testSystemTurnWrapsTools() {
        let decl = Gemma4ToolFormat.ToolDeclaration(
            name: "list_libraries",
            description: "List loaded ZIMs.",
            parameters: []
        )
        let turn = Gemma4ToolFormat.formatSystemTurn(systemMessage: "You are helpful.", tools: [decl])
        XCTAssertTrue(turn.hasPrefix("<|turn>system\nYou are helpful.\n"))
        XCTAssertTrue(turn.hasSuffix("<turn|>\n"))
        XCTAssertTrue(turn.contains("<|tool>declaration:list_libraries{"))
    }

    func testSystemTurnEmptyWhenNothing() {
        XCTAssertTrue(Gemma4ToolFormat.formatSystemTurn(systemMessage: "", tools: []).isEmpty)
    }
}
