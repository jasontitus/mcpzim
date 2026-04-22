// SPDX-License-Identifier: MIT
//
// `MCPToolAdapter.sanitizeZim(_:loadedZimNames:)` strips a
// hallucinated `zim` filename from a tool-args dict so the underlying
// service falls back to "search all loaded ZIMs" instead of returning
// an "unknown ZIM" error to the model.
//
// Real on-device capture (Qwen 3 4B 4-bit): the model emitted
//   compare_articles({"titles":[...], "zim":"wikipediapedia_en_all maxi 2025-10.zim"})
// against an actually-loaded ZIM
//   wikipedia_en_all_maxi_2025-10.zim
// — duplicated "pedia" + space instead of underscore. Without the
// sanitiser the tool errored, iter 1 mangled the entity names while
// summarising the error, and the user got nothing.

import Foundation
import XCTest

@testable import MCPZimKit

final class SanitizeZimArgTests: XCTestCase {

    func testNoZimArgIsPassThrough() {
        let args: [String: Any] = ["title": "Palo Alto"]
        let out = MCPToolAdapter.sanitizeZim(args, loadedZimNames: ["a.zim"])
        XCTAssertEqual(out["title"] as? String, "Palo Alto")
        XCTAssertNil(out["zim"])
    }

    func testEmptyZimArgIsPassThrough() {
        // Empty string is treated as "not specified" — leave it alone
        // so the dispatcher sees the same shape it always sees.
        let args: [String: Any] = ["title": "X", "zim": ""]
        let out = MCPToolAdapter.sanitizeZim(args, loadedZimNames: ["a.zim"])
        XCTAssertEqual(out["zim"] as? String, "")
    }

    func testExactMatchIsPreserved() {
        let args: [String: Any] = ["zim": "wikipedia_en_all_maxi_2025-10.zim"]
        let out = MCPToolAdapter.sanitizeZim(
            args, loadedZimNames: ["wikipedia_en_all_maxi_2025-10.zim",
                                   "osm-silicon-valley.zim"]
        )
        XCTAssertEqual(out["zim"] as? String,
                       "wikipedia_en_all_maxi_2025-10.zim")
    }

    func testCaseInsensitiveMatchIsPreserved() {
        // Models occasionally lowercase ZIM filenames in their
        // emissions even when the on-disk name has caps. Don't punish
        // that — match case-insensitively.
        let args: [String: Any] = ["zim": "WIKIPEDIA_EN_ALL_MAXI_2025-10.ZIM"]
        let out = MCPToolAdapter.sanitizeZim(
            args, loadedZimNames: ["wikipedia_en_all_maxi_2025-10.zim"]
        )
        XCTAssertEqual(out["zim"] as? String,
                       "WIKIPEDIA_EN_ALL_MAXI_2025-10.ZIM",
                       "case-insensitive match should keep the original arg")
    }

    func testHallucinatedZimIsRemoved() {
        // Verbatim repro of the dropped-request log capture.
        let args: [String: Any] = [
            "titles": ["South Korea", "North Korea"],
            "zim": "wikipediapedia_en_all maxi 2025-10.zim",
        ]
        let out = MCPToolAdapter.sanitizeZim(
            args, loadedZimNames: ["wikipedia_en_all_maxi_2025-10.zim"]
        )
        XCTAssertNil(out["zim"], "hallucinated zim should be stripped")
        XCTAssertEqual((out["titles"] as? [String])?.count, 2,
                       "other args must survive the strip")
    }

    func testCloseFilenameNotSubstituted() {
        // We never substitute — a "close" miss could be a different
        // ZIM (en vs es, full vs nopic). Strip, don't repair.
        let args: [String: Any] = ["zim": "wikipedia_es_all_maxi_2025-10.zim"]
        let out = MCPToolAdapter.sanitizeZim(
            args, loadedZimNames: ["wikipedia_en_all_maxi_2025-10.zim"]
        )
        XCTAssertNil(out["zim"], "near-miss should drop, not substitute")
    }

    func testEmptyLoadedListStripsAnyZim() {
        let args: [String: Any] = ["zim": "anything.zim"]
        let out = MCPToolAdapter.sanitizeZim(args, loadedZimNames: [])
        XCTAssertNil(out["zim"])
    }
}
