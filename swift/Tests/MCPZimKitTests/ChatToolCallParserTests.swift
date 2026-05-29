// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class ChatToolCallParserTests: XCTestCase {
    func testFindsWellFormedCall() throws {
        let buffer = #"sure thing <tool_call>{"name":"search","arguments":{"q":"kangaroo"}}</tool_call> and here's the plan"#
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "search")
        XCTAssertEqual(match.arguments["q"] as? String, "kangaroo")
        XCTAssertEqual(String(buffer[match.range]),
                       #"<tool_call>{"name":"search","arguments":{"q":"kangaroo"}}</tool_call>"#)
    }

    func testMissingArgumentsDefaultsToEmpty() throws {
        let buffer = #"<tool_call>{"name":"list_libraries"}</tool_call>"#
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "list_libraries")
        XCTAssertTrue(match.arguments.isEmpty)
    }

    func testReturnsNilWhenCloseTagIsMissing() {
        let buffer = #"<tool_call>{"name":"search","arguments":{"q":"ka"#
        XCTAssertNil(ChatToolCallParser.firstCall(in: buffer))
    }

    func testReturnsNilOnMalformedJson() {
        let buffer = #"<tool_call>{name: search}</tool_call>"#
        XCTAssertNil(ChatToolCallParser.firstCall(in: buffer))
    }

    func testReturnsNilWhenNameFieldMissing() {
        let buffer = #"<tool_call>{"arguments":{"q":"x"}}</tool_call>"#
        XCTAssertNil(ChatToolCallParser.firstCall(in: buffer))
    }

    func testReturnsFirstCallWhenMultiplePresent() throws {
        let buffer = #"<tool_call>{"name":"a"}</tool_call>noise<tool_call>{"name":"b"}</tool_call>"#
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "a")
    }

    func testRangeSlicesCleanly() throws {
        let buffer = #"prefix <tool_call>{"name":"geocode","arguments":{"q":"Seattle"}}</tool_call> suffix"#
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        let pre = String(buffer[..<match.range.lowerBound])
        let post = String(buffer[match.range.upperBound...])
        XCTAssertEqual(pre, "prefix ")
        XCTAssertEqual(post, " suffix")
    }

    func testHandlesNestedJsonArguments() throws {
        let buffer = #"<tool_call>{"name":"route_from_places","arguments":{"origin":"A","destination":"B","zim":null}}</tool_call>"#
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "route_from_places")
        XCTAssertEqual(match.arguments["origin"] as? String, "A")
        XCTAssertEqual(match.arguments["destination"] as? String, "B")
        XCTAssertTrue(match.arguments["zim"] is NSNull)
    }

    // Apple Foundation Models emits a variant wrapper in its response
    // stream: `<|tool_call|{JSON}>`. The parser should accept it and
    // produce the same Match shape as the canonical form so ChatSession
    // doesn't need a provider-specific code path.
    func testParsesAppleFMVariant() throws {
        let buffer = #"<|tool_call|{"name":"near_named_place","arguments":{"place":"Adams Morgan"}}>"#
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "near_named_place")
        XCTAssertEqual(match.arguments["place"] as? String, "Adams Morgan")
        // Range should consume the trailing '>' so the wrapper fully
        // disappears from the visible transcript.
        XCTAssertEqual(String(buffer[match.range]), buffer)
    }

    // A partial Apple-FM wrapper (opener received, JSON not yet
    // balanced) must return nil so the streaming loop keeps collecting
    // tokens rather than dispatching a malformed call.
    func testReturnsNilOnPartialAppleFMWrapper() {
        let buffer = #"<|tool_call|{"name":"near_named_place","arguments":{"place":"Adams"#
        XCTAssertNil(ChatToolCallParser.firstCall(in: buffer))
    }

    // FT'd Gemma 3 4B (gists 80daf913, 3f39d873) sometimes emits
    // `"arguments={...}` (literal `=` token) where it should have emitted
    // `"arguments":{...}`. The strict JSON path can't recover, but the
    // repair fallback rewrites `{"arguments=` → `{"arguments":` and
    // parses cleanly.
    func testParsesGemma3KeyEqualsGarble() throws {
        let buffer = """
        <tool_call>
        {"arguments={"title": "Einstein's prediction of gravity waves", "max_sections": 1}, "name": "article_overview"}
        </tool_call>
        """
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "article_overview")
        XCTAssertEqual(match.arguments["title"] as? String,
                       "Einstein's prediction of gravity waves")
        XCTAssertEqual(match.arguments["max_sections"] as? Int, 1)
        // Range should swallow the entire <tool_call>...</tool_call> block
        // so the user-visible transcript doesn't show the raw garble.
        XCTAssertEqual(String(buffer[match.range]), buffer)
    }

    // Repair must NOT corrupt `=` characters that legitimately appear
    // inside string values (e.g. URL query strings). Anchor the regex
    // to JSON-key positions only.
    func testRepairLeavesValueEqualsAlone() throws {
        let buffer = #"<tool_call>{"name":"fetch","arguments":{"url":"https://x.com/a?q=1&b=2"}}</tool_call>"#
        let match = try XCTUnwrap(ChatToolCallParser.firstCall(in: buffer))
        XCTAssertEqual(match.name, "fetch")
        XCTAssertEqual(match.arguments["url"] as? String,
                       "https://x.com/a?q=1&b=2")
    }
}
