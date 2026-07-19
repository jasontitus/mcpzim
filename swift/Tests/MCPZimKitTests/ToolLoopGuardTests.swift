// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimKit

final class ToolLoopGuardTests: XCTestCase {
    func testExactDuplicateStops() {
        var guardrail = ToolLoopGuard()
        XCTAssertEqual(guardrail.evaluate(
            toolName: "search", canonicalArguments: #"{"query":"San Francisco founded"}"#),
                       .allow)
        XCTAssertEqual(guardrail.evaluate(
            toolName: "search", canonicalArguments: #"{"query":"San Francisco founded"}"#),
                       .stop("duplicate search call suppressed"))
    }

    func testSearchBudgetStopsThirdDistinctQuery() {
        var guardrail = ToolLoopGuard()
        XCTAssertEqual(guardrail.evaluate(toolName: "search", canonicalArguments: "a"), .allow)
        XCTAssertEqual(guardrail.evaluate(toolName: "search", canonicalArguments: "b"), .allow)
        XCTAssertEqual(guardrail.evaluate(toolName: "search", canonicalArguments: "c"),
                       .stop("search call budget exhausted (2 per turn)"))
    }

    func testOtherToolsOnlyStopOnExactDuplicate() {
        var guardrail = ToolLoopGuard()
        for i in 0..<4 {
            XCTAssertEqual(guardrail.evaluate(
                toolName: "get_article_section", canonicalArguments: "\(i)"), .allow)
        }
    }
}
