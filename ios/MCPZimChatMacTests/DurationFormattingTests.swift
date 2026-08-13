// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimChatMac

/// Route replies echo this string to the user verbatim, so a formatting
/// slip is user-visible. Rounding minutes inside the hour remainder let
/// the carry escape the hour: 7199 s rendered "1h 60m" and 3599 s
/// rendered "60m" (2026-08-13 review).
@MainActor
final class DurationFormattingTests: XCTestCase {

    func testHourBoundaryCarriesInsteadOfRenderingSixtyMinutes() {
        XCTAssertEqual(ChatSession.formatDuration(seconds: 7199), "2h")
        XCTAssertEqual(ChatSession.formatDuration(seconds: 3599), "1h")
        XCTAssertEqual(ChatSession.formatDuration(seconds: 3630), "1h 1m")
    }

    func testOrdinaryDurationsUnchanged() {
        XCTAssertEqual(ChatSession.formatDuration(seconds: 0), "0m")
        XCTAssertEqual(ChatSession.formatDuration(seconds: 90), "2m")
        XCTAssertEqual(ChatSession.formatDuration(seconds: 600), "10m")
        XCTAssertEqual(ChatSession.formatDuration(seconds: 3600), "1h")
        XCTAssertEqual(ChatSession.formatDuration(seconds: 5400), "1h 30m")
    }

    func testNegativeInputClampsToZero() {
        XCTAssertEqual(ChatSession.formatDuration(seconds: -42), "0m")
    }
}
