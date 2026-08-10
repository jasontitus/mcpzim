// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimChatMac

final class OfflineContentCatalogTests: XCTestCase {
    func testWikipediaCatalogChoosesNewestNoPictureEdition() {
        let html = """
        <a href="wikipedia_en_all_nopic_2026-03.zim">wikipedia_en_all_nopic_2026-03.zim</a> 2026-03-28 00:38 48G
        <a href="wikipedia_en_top_nopic_2026-03.zim">wikipedia_en_top_nopic_2026-03.zim</a> 2026-03-18 12:16 2.0G
        <a href="wikipedia_en_simple_all_nopic_2026-05.zim">wikipedia_en_simple_all_nopic_2026-05.zim</a> 2026-05-10 22:45 937M
        <a href="wikipedia_en_all_nopic_2026-06.zim">wikipedia_en_all_nopic_2026-06.zim</a> 2026-06-26 09:02 49G
        <a href="wikipedia_en_top_nopic_2026-06.zim">wikipedia_en_top_nopic_2026-06.zim</a> 2026-06-16 13:20 2.1G
        """

        let choices = WikipediaArchiveCatalog.parse(html: html)

        XCTAssertEqual(choices.map(\.edition), [.top, .simple, .complete])
        XCTAssertEqual(choices[0].filename, "wikipedia_en_top_nopic_2026-06.zim")
        XCTAssertEqual(choices[0].sizeLabel, "2.1 GB")
        XCTAssertEqual(choices[1].sizeLabel, "937 MB")
        XCTAssertEqual(choices[2].filename, "wikipedia_en_all_nopic_2026-06.zim")
    }

    func testWikipediaCatalogIgnoresPicturesAndOtherLanguages() {
        let html = """
        <a href="wikipedia_en_all_maxi_2026-06.zim">wikipedia_en_all_maxi_2026-06.zim</a> 2026-06-26 09:02 120G
        <a href="wikipedia_es_all_nopic_2026-06.zim">wikipedia_es_all_nopic_2026-06.zim</a> 2026-06-26 09:02 12G
        """

        XCTAssertTrue(WikipediaArchiveCatalog.parse(html: html).isEmpty)
    }
}
