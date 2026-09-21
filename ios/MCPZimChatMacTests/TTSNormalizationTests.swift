// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimChatMac

@MainActor
final class VisibleErrorLoggingTests: XCTestCase {
    func testDisplayedErrorsSurviveDismissalAndRepeatedOccurrences() {
        let session = ChatSession(autoLoadOnInit: false)
        let message = "Could not continue.\nPlease try again."
        session.lastError = message
        session.lastError = message // same outstanding alert is not a new event
        session.lastError = nil
        session.lastError = message // a later occurrence must still be logged
        session.lastError = nil
        session.libraryError = "Archive unavailable"
        session.libraryError = nil
        session.modelState = .failed("Model unavailable")
        session.modelState = .notLoaded
        let errors = session.debugEntries.filter { $0.category == "UIError" }
        XCTAssertEqual(errors.count, 4)
        XCTAssertEqual(errors.filter { $0.message.contains("Chat: " + message) }.count, 2)
        XCTAssertTrue(errors.contains { $0.message.contains("Library: Archive unavailable") })
        XCTAssertTrue(errors.contains { $0.message.contains("Model: Model unavailable") })
    }
}

final class TTSNormalizationTests: XCTestCase {
    func testHonorificAndSaintRemainConversational() {
        let normalized = KokoroTTSService.prepForTTS(
            "Dr. Vladimir Putin once worked in St. Petersburg."
        )

        XCTAssertEqual(
            normalized,
            "Dr. Vladimir Putin once worked in St. Petersburg."
        )
    }

    func testDriveAndStreetStillExpandInRouteContext() {
        let normalized = KokoroTTSService.prepForTTS(
            "Continue on Hamilton Dr. for 0.2 mi, then turn onto Main St."
        )

        XCTAssertEqual(
            normalized,
            "Continue on Hamilton Drive for 0 point 2 miles, then turn onto Main Street"
        )
    }
}

final class AutomaticVoicePolicyTests: XCTestCase {
    func testKokoroRequiresInstalledAssetsAndMeasuredHeadroom() {
        XCTAssertTrue(TTSFactory.prefersKokoroAutomatically(assetsInstalled: true,
            availableMemoryMB: 3500, thermallyConstrained: false))
        for memory in [0.0, -1, 3499, Double.nan, Double.infinity] {
            XCTAssertFalse(TTSFactory.prefersKokoroAutomatically(assetsInstalled: true,
                availableMemoryMB: memory, thermallyConstrained: false))
        }
        XCTAssertFalse(TTSFactory.prefersKokoroAutomatically(assetsInstalled: false,
            availableMemoryMB: 8000, thermallyConstrained: false))
        XCTAssertFalse(TTSFactory.prefersKokoroAutomatically(assetsInstalled: true,
            availableMemoryMB: 8000, thermallyConstrained: true))
    }
}

final class SupertonicRuntimeTests: XCTestCase {
    func testRealRuntimeWarmupAndReuseWithoutPlayback() async throws {
        guard ProcessInfo.processInfo.environment["ZIMFO_TEST_SUPERTONIC_RUNTIME"] == "1" else {
            throw XCTSkip("Opt-in real Core ML model test")
        }
        guard Supertonic3Assets.readyForSilentWarmup else { throw XCTSkip("Cached models required") }
        let start = Date()
        try await Supertonic3TTSService.prewarmRuntime()
        let warmed = Date()
        let first = Supertonic3TTSService()
        let text = "Arnold Alois Schwarzenegger is an Austrian and American actor and former politician."
        let one = try await first.synthesizeWithoutPlayback(text)
        let second = Supertonic3TTSService()
        let beforeSecond = Date()
        try await second.prepareForConversation()
        let two = try await second.synthesizeWithoutPlayback(text)
        XCTAssertGreaterThan(one.sampleCount, 0)
        XCTAssertGreaterThan(two.sampleCount, 0)
        print("SUPERTONIC-REUSE warm=\(warmed.timeIntervalSince(start)) first=\(beforeSecond.timeIntervalSince(warmed)) second=\(Date().timeIntervalSince(beforeSecond))")
    }
}
