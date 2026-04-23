// SPDX-License-Identifier: MIT
//
// `firstToolCall` (strict, streaming) + `firstToolCallAfterClip`
// (tolerant, post-stream) behaviour on the Qwen ChatML template.
// Covers the clipped-tool-call rescue that saves Qwen 3.5 from the
// "model emits <|im_end|> mid-</tool_call> and we hang on the reply"
// failure mode.

import Foundation
import XCTest

@testable import MCPZimKit

final class QwenClippedToolCallTests: XCTestCase {

    private let tpl = QwenChatMLTemplate()

    // MARK: - Strict streaming parser (canonical)

    func testCanonicalCloseMatches() {
        let buffer = """
        <tool_call>
        {"name": "near_named_place", "arguments": {"place": "SF", "kinds": ["bar"]}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.name, "near_named_place")
        XCTAssertEqual(m?.arguments["place"] as? String, "SF")
    }

    func testStreamingReturnsNilWhenCloseMarkerMissing() {
        // Classic partial stream: <tool_call> open, JSON mostly there,
        // no </tool_call> yet. Streaming parser MUST NOT fire.
        let buffer = """
        <tool_call>
        {"name": "near_named_place", "arguments": {"place": "SF"
        """
        XCTAssertNil(tpl.firstToolCall(in: buffer))
    }

    func testStreamingReturnsNilWhenNoToolCallOpener() {
        let m = tpl.firstToolCall(in: "just a plain text reply.")
        XCTAssertNil(m)
    }

    // MARK: - Clipped parser (post-stream)

    func testClippedPartialCloserMatches() {
        // Real repro from the device: model emitted `</tool` but was
        // clipped by <|im_end|> before completing the closer.
        let buffer = """
        <tool_call>
        {"name": "near_named_place", "arguments": {"place": "North Beach", "kinds": ["restaurant"]}
        </tool
        """
        let m = tpl.firstToolCallAfterClip(in: buffer)
        XCTAssertNotNil(m, "should rescue clipped tool call")
        XCTAssertEqual(m?.name, "near_named_place")
        XCTAssertEqual(m?.arguments["place"] as? String, "North Beach")
    }

    func testClippedNoCloserButBalancedJSONMatches() {
        // Model stopped right after the outer `}` — no closer at
        // all, but the JSON is complete.
        let buffer = """
        <tool_call>
        {"name": "route_from_places", "arguments": {"origin": "my location", "destination": "SF"}}
        """
        let m = tpl.firstToolCallAfterClip(in: buffer)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.name, "route_from_places")
    }

    func testClippedMissingOuterBraceIsRepaired() {
        // Exactly the reported failure: missing the final `}` on the
        // outer object. `repairJSON` balances it before decoding.
        let buffer = """
        <tool_call>
        {"name": "near_named_place", "arguments": {"place": "North Beach"}
        """
        let m = tpl.firstToolCallAfterClip(in: buffer)
        XCTAssertNotNil(m, "should repair missing trailing brace")
        XCTAssertEqual(m?.name, "near_named_place")
    }

    func testClippedReturnsNilOnTotalJunk() {
        // `<tool_call>` present but the body is unparseable even
        // after brace balancing. Don't falsely dispatch.
        let buffer = """
        <tool_call>
        this is not JSON at all {{{
        """
        XCTAssertNil(tpl.firstToolCallAfterClip(in: buffer))
    }

    func testClippedStillReturnsNilWhenNoOpener() {
        XCTAssertNil(tpl.firstToolCallAfterClip(in: "nothing to see here"))
    }

    // MARK: - Alternate JSON shapes Qwen 3.5 sometimes emits

    func testQwenFlatFunctionShape() {
        let buffer = """
        <tool_call>
        {"function": "near_named_place", "arguments": {"place": "SF"}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m, "flat {function: <name>, arguments: {…}} shape")
        XCTAssertEqual(m?.name, "near_named_place")
    }

    func testQwenNestedFunctionShape() {
        let buffer = """
        <tool_call>
        {"function": {"name": "near_named_place", "arguments": {"place": "SF"}}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m, "nested {function: {name, arguments}} shape")
        XCTAssertEqual(m?.name, "near_named_place")
    }

    func testQwenTypeFunctionShape() {
        // OpenAI-ish nested form observed on text-only-finetuned 4B.
        let buffer = """
        <tool_call>
        {"type": "function", "arguments": {"name": "near_named_place", "arguments": {"place": "SF"}}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m, "type:function/arguments/(name, arguments) shape")
        XCTAssertEqual(m?.name, "near_named_place")
    }

    // MARK: - Malformed JSON repair (observed on-device)
    //
    // All three of these are real captures from a Qwen 3.5 4B 4-bit
    // session — see dropped-request.log in the repo root. Each caused
    // a silent turn drop before the repair pass was added.

    func testRepairsDoubleCommasInToolCallBody() {
        // "Give me directions to San Francisco" — Qwen emitted two
        // stray commas that killed the strict decode.
        let buffer = """
        <tool_call>
        {"name": "route_from_places",,"arguments": {"destination": "San Francisco",,"origin": "my location"}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m, "double-comma body should parse via repair")
        XCTAssertEqual(m?.name, "route_from_places")
        XCTAssertEqual(m?.arguments["destination"] as? String, "San Francisco")
        XCTAssertEqual(m?.arguments["origin"] as? String, "my location")
    }

    func testRepairsTrailingCommaBeforeCloser() {
        // Trailing comma is valid JS, not JSON. Qwen leaks JS habits.
        let buffer = """
        <tool_call>
        {"name": "near_named_place", "arguments": {"place": "SF", "kinds": ["bar",]}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.name, "near_named_place")
        XCTAssertEqual(m?.arguments["kinds"] as? [String], ["bar"])
    }

    func testRepairsWhitespaceOnlyStringBeforeBareword() {
        // "Compare North Korea and South Korea" — Qwen wedged an
        // extra `" "` between the array comma and the next string
        // opening quote, leaving the second title unquoted in a way
        // that parses as a whitespace-only string followed by an
        // unterminated bareword.
        let buffer = """
        <tool_call>
        {"name": "compare_articles","arguments": {"titles": ["North Korea", " "South Korea"]}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m, "should recover both titles after splice repair")
        XCTAssertEqual(m?.name, "compare_articles")
        XCTAssertEqual(m?.arguments["titles"] as? [String],
                       ["North Korea", "South Korea"])
    }

    func testRepairsCombinedDoubleCommaAndSpliceFromLog() {
        // Verbatim capture of the `compare_articles` emission that
        // dropped on device (dropped-request.log:498). Both ,, and
        // the `" "`-wedge are present; the repair has to handle them
        // together.
        let buffer = """
        <tool_call>
        {"name": "compare_articles",,"arguments": {"titles": ["North Korea", " "South Korea"]}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.name, "compare_articles")
        XCTAssertEqual(m?.arguments["titles"] as? [String],
                       ["North Korea", "South Korea"])
    }

    func testRepairsDoubledOpeningQuoteOnFirstKey() {
        // Verbatim capture from the 2026-04-22 debug gist
        // 59d19a5f…. Qwen emitted a spurious `""` after the
        // nested object's opening `{`. Without the repair the
        // strict JSONSerialization decode fails and the tool call
        // silently drops.
        let buffer = """
        <tool_call>
        {"name":"near_named_place","arguments":{""place":"north beach","kinds":["bar"],"radius_km":1}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m, "doubled opening quote should be repaired")
        XCTAssertEqual(m?.name, "near_named_place")
        let args = m?.arguments ?? [:]
        XCTAssertEqual(args["place"] as? String, "north beach")
        XCTAssertEqual(args["kinds"] as? [String], ["bar"])
    }

    func testRepairsDoubledOpeningQuoteCombinedWithWhitespaceWedge() {
        // Second real capture. Same emission shape plus the
        // whitespace-only-string wedge before subsequent keys —
        // the pattern from "Museums near north beach". Both the
        // `{""` and the `, " "key` forms need to be repaired
        // in the same pass.
        let buffer = """
        <tool_call>
        {"name":"near_named_place","arguments":{""place":"North Beach", " "kinds":["museum"], " "radius_km":1}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.name, "near_named_place")
        let args = m?.arguments ?? [:]
        XCTAssertEqual(args["place"] as? String, "North Beach")
        XCTAssertEqual(args["kinds"] as? [String], ["museum"])
    }

    func testRepairPreservesLegitimateEmptyStringValue() {
        // Guard: the doubled-quote rule only fires when the
        // doubled pair is followed by an IDENTIFIER char — an
        // intentional empty string value (followed by `,`, `]`,
        // or `}`) must survive untouched.
        let buffer = """
        <tool_call>
        {"name":"search","arguments":{"query":"","kind":""}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.arguments["query"] as? String, "")
        XCTAssertEqual(m?.arguments["kind"] as? String, "")
    }

    func testRepairLeavesLegitimateWhitespaceStringAlone() {
        // Guard for the splice repair: a whitespace-only string that
        // IS followed by a valid JSON delimiter is real and must
        // survive.
        let buffer = """
        <tool_call>
        {"name": "search", "arguments": {"sep": " ", "q": "hello"}}
        </tool_call>
        """
        let m = tpl.firstToolCall(in: buffer)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.arguments["sep"] as? String, " ",
                       "an intentional whitespace value must not be clobbered")
    }

    // MARK: - formatToolCall round-trip
    //
    // The fast-path injector needs to emit a synthetic tool-call in
    // the model's native wire format so iter 0 of runGenerationLoop
    // sees a completed round-trip and generates prose (not another
    // tool_call). The contract: what we format, we must be able to
    // parse back.

    func testFormatToolCallRoundTripsThroughFirstToolCall() {
        let name = "compare_articles"
        let args: [String: Any] = [
            "titles": ["North Korea", "South Korea"]
        ]
        let emission = tpl.formatToolCall(name: name, arguments: args)
        // The emission must be wrapped in <tool_call>...</tool_call>
        // so the transcript renderer doesn't accidentally double-wrap
        // and the streaming parser finds it on round-trip.
        XCTAssertTrue(emission.hasPrefix("<tool_call>"),
                      "missing <tool_call> open marker: \(emission)")
        XCTAssertTrue(emission.contains("</tool_call>"),
                      "missing </tool_call> close marker: \(emission)")
        let m = tpl.firstToolCall(in: emission)
        XCTAssertNotNil(m, "emission must be parseable by firstToolCall")
        XCTAssertEqual(m?.name, "compare_articles")
        XCTAssertEqual(m?.arguments["titles"] as? [String],
                       ["North Korea", "South Korea"])
    }

    func testFormatToolCallHandlesEmptyArgs() {
        let emission = tpl.formatToolCall(
            name: "what_is_here", arguments: [:]
        )
        let m = tpl.firstToolCall(in: emission)
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.name, "what_is_here")
        XCTAssertTrue((m?.arguments ?? [:]).isEmpty)
    }

    // MARK: - Reasoning strip

    func testStripReasoningRemovesClosedThinkSpan() {
        let raw = "<think>Let me figure this out…</think>\n\nThe answer is 42."
        XCTAssertEqual(tpl.stripReasoning(raw), "The answer is 42.")
    }

    func testStripReasoningLeavesOpenThinkInPlace() {
        // Mid-stream open-but-not-closed <think> must survive so the
        // UI doesn't flash half a reasoning block while streaming.
        let raw = "<think>Still reasoning…"
        XCTAssertEqual(tpl.stripReasoning(raw), raw)
    }
}
