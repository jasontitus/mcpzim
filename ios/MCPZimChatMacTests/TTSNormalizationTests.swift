// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimChatMac

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
