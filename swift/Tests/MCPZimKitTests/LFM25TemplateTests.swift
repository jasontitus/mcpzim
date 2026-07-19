// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class LFM25TemplateTests: XCTestCase {
    func testStripReasoningHandlesOpeningMarkerInjectedByChatTemplate() {
        let answer = "The main innovations included TensorFlow and GNMT."
        let raw = "\(answer)\n</think>\n\(answer)"

        XCTAssertEqual(LFM25Template().stripReasoning(raw), answer)
    }

    func testStripReasoningRemovesOrdinaryClosedReasoningSpan() {
        let raw = "<think>Private scratchpad.</think>\nThe visible answer."

        XCTAssertEqual(
            LFM25Template().stripReasoning(raw),
            "The visible answer.")
    }
}
