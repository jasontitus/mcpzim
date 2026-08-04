// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimChatMac

/// Pins the two catalog-page formats the in-app downloader depends on, so a
/// silent format change upstream fails loudly here instead of quietly
/// stranding the app on its baked-in fallback lists.
final class CatalogParsingTests: XCTestCase {

    // MARK: Kiwix directory index (Wikipedia)

    private let kiwixIndexSample = """
        <a href="wikipedia_en_all_maxi_2025-08.zim">wikipedia_en_all_maxi_2025-08.zim</a>             2025-08-24 16:47  111G
        <a href="wikipedia_en_all_maxi_2026-02.zim">wikipedia_en_all_maxi_2026-02.zim</a>             2026-02-26 02:08  115G
        <a href="wikipedia_en_all_mini_2026-06.zim">wikipedia_en_all_mini_2026-06.zim</a>             2026-06-18 13:38   12G
        <a href="wikipedia_en_all_nopic_2026-03.zim">wikipedia_en_all_nopic_2026-03.zim</a>            2026-03-28 00:38   48G
        <a href="wikipedia_en_all_nopic_2026-06.zim">wikipedia_en_all_nopic_2026-06.zim</a>            2026-06-26 09:02   49G
        <a href="wikipedia_en_simple_all_maxi_2026-05.zim">wikipedia_en_simple_all_maxi_2026-05.zim</a>      2026-05-10 20:40  3.2G
        <a href="wikipedia_en_simple_all_nopic_2026-05.zim">wikipedia_en_simple_all_nopic_2026-05.zim</a>     2026-05-10 22:45  937M
        <a href="wikipedia_en_top_maxi_2026-06.zim">wikipedia_en_top_maxi_2026-06.zim</a>             2026-06-16 22:10  7.8G
        <a href="wikipedia_en_top_nopic_2026-03.zim">wikipedia_en_top_nopic_2026-03.zim</a>            2026-03-18 12:16  2.1G
        <a href="wikipedia_en_top_nopic_2026-06.zim">wikipedia_en_top_nopic_2026-06.zim</a>            2026-06-16 13:20  2.1G
        """

    func testWikipediaParserFindsAllSixEditionFlavorPairs() {
        let items = WikipediaZimCatalog.parse(html: kiwixIndexSample)
        XCTAssertEqual(items.count, 6, "3 editions × 2 flavors (nopic + maxi)")
        XCTAssertEqual(Set(items.map(\.id)),
                       ["wikipedia.top.nopic", "wikipedia.simple.nopic",
                        "wikipedia.complete.nopic", "wikipedia.top.maxi",
                        "wikipedia.simple.maxi", "wikipedia.complete.maxi"])
    }

    func testWikipediaParserPicksNewestDateAndExpandsSize() throws {
        let items = WikipediaZimCatalog.parse(html: kiwixIndexSample)
        let topNopic = try XCTUnwrap(items.first { $0.id == "wikipedia.top.nopic" })
        XCTAssertTrue(topNopic.url.absoluteString.hasSuffix("wikipedia_en_top_nopic_2026-06.zim"),
                      "must pick 2026-06 over 2026-03, got \(topNopic.url)")
        XCTAssertEqual(topNopic.sizeLabel, "2.1 GB")
        XCTAssertTrue(topNopic.recommended)

        let allMaxi = try XCTUnwrap(items.first { $0.id == "wikipedia.complete.maxi" })
        XCTAssertTrue(allMaxi.url.absoluteString.hasSuffix("wikipedia_en_all_maxi_2026-02.zim"))
        XCTAssertEqual(allMaxi.sizeLabel, "115 GB")
        XCTAssertFalse(allMaxi.recommended)
        XCTAssertEqual(allMaxi.kind, .wikipedia(images: true))
    }

    func testWikipediaParserIgnoresMiniFlavor() {
        let items = WikipediaZimCatalog.parse(html: kiwixIndexSample)
        XCTAssertFalse(items.contains { $0.url.absoluteString.contains("_mini_") })
    }

    // MARK: StreetZIM landing page (maps)

    private let streetZimSample = """
        <div class="map-card">
          <div class="map-card-head">
            <div class="map-card-title">Europe</div>
            <div class="map-card-size">62.1 GB</div>
          </div>
          <p class="map-card-desc">United Kingdom, Ireland, France &mdash; and more.</p>
          <div class="map-card-links">
            <a class="btn btn-primary" href="https://archive.org/download/streetzim-europe/osm-europe-2026-05-06.zim" data-track="download" data-region="europe" data-title="Europe">Download</a>
          </div>
        </div>
        <h3 class="maps-tier-header">States, cities &amp; islands</h3>
        <div class="map-card">
          <div class="map-card-head">
            <div class="map-card-title">Korea &amp; Mongolia</div>
            <div class="map-card-size">5.2 GB</div>
          </div>
          <p class="map-card-desc">The Korean Peninsula and Mongolia &mdash; Seoul, Busan.</p>
          <div class="map-card-links">
            <a class="btn btn-primary" href="https://archive.org/download/streetzim-korea-mongolia/osm-korea-mongolia-2026-05-11c.zim" data-track="download" data-region="korea-mongolia" data-title="Korea &amp; Mongolia">Download</a>
          </div>
        </div>
        """

    func testStreetZimParserExtractsCardsTiersAndEntities() throws {
        let items = StreetZimCatalog.parse(html: streetZimSample)
        XCTAssertEqual(items.count, 2)

        let europe = try XCTUnwrap(items.first { $0.id == "map.europe" })
        XCTAssertEqual(europe.title, "Europe")
        XCTAssertEqual(europe.sizeLabel, "62.1 GB")
        XCTAssertEqual(europe.tier, "Continents & regions",
                       "cards before the first tier header get the default tier")
        XCTAssertEqual(europe.url.absoluteString,
                       "https://archive.org/download/streetzim-europe/osm-europe-2026-05-06.zim")
        XCTAssertTrue(europe.detail.contains("—"), "&mdash; must be decoded")

        let korea = try XCTUnwrap(items.first { $0.id == "map.korea-mongolia" })
        XCTAssertEqual(korea.title, "Korea & Mongolia", "&amp; must be decoded")
        XCTAssertEqual(korea.tier, "States, cities & islands")
        XCTAssertEqual(korea.kind, .map)
    }

    func testStreetZimLiveFallbackIsPlausible() {
        // The baked fallback must always be usable on its own: dozens of
        // regions, every URL an archive.org .zim, sizes populated.
        XCTAssertGreaterThanOrEqual(StreetZimCatalog.fallback.count, 40)
        for item in StreetZimCatalog.fallback {
            XCTAssertTrue(item.url.absoluteString.hasPrefix("https://archive.org/download/streetzim-"),
                          "unexpected URL \(item.url)")
            XCTAssertTrue(item.url.absoluteString.hasSuffix(".zim"))
            XCTAssertGreaterThan(item.sizeBytes, 0, "\(item.id) has no size")
            XCTAssertNotNil(item.tier)
        }
        XCTAssertGreaterThanOrEqual(WikipediaZimCatalog.fallback.count, 6)
    }
}
