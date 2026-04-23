// SPDX-License-Identifier: MIT
//
// Defence-in-depth on the `w` field that streetzim writes into
// search-data records. Contract: `w` is a Wikipedia tag like
// `"en:HP_Garage"`. A pre-fa6208b streetzim build polluted `w`
// with website URLs from the Overture places enrichment. Our
// guard (`DefaultZimService.sanitizedWikiTag`) drops URL-shaped
// values at ingest so stale ZIMs degrade to "no wiki tag"
// instead of misfeeding `articleByTitle` and false-positiving
// `near_places(hasWiki: true)`.

import Foundation
import XCTest

@testable import MCPZimKit

final class SanitizedWikiTagTests: XCTestCase {

    func testNilAndEmptyPassThroughAsNil() {
        XCTAssertNil(DefaultZimService.sanitizedWikiTag(nil))
        XCTAssertNil(DefaultZimService.sanitizedWikiTag(""))
    }

    func testLanguagePrefixedTagsPassThrough() {
        XCTAssertEqual(
            DefaultZimService.sanitizedWikiTag("en:HP_Garage"),
            "en:HP_Garage")
        XCTAssertEqual(
            DefaultZimService.sanitizedWikiTag("fr:Tour_Eiffel"),
            "fr:Tour_Eiffel")
        XCTAssertEqual(
            DefaultZimService.sanitizedWikiTag("en:Palo Alto"),
            "en:Palo Alto")
    }

    func testBareTitlesPassThrough() {
        // Some older records omit the language prefix; we keep them
        // so the fallback resolution paths still see them.
        XCTAssertEqual(
            DefaultZimService.sanitizedWikiTag("HP_Garage"),
            "HP_Garage")
        XCTAssertEqual(
            DefaultZimService.sanitizedWikiTag("HP Garage"),
            "HP Garage")
    }

    func testUrlLikeValuesAreStripped() {
        // The actual pre-fa6208b collision: POI website URLs wrote
        // into `w` alongside (or instead of) the Wikipedia tag.
        XCTAssertNil(
            DefaultZimService.sanitizedWikiTag("https://www.hpgarage.com"))
        XCTAssertNil(
            DefaultZimService.sanitizedWikiTag("http://example.com"))
        XCTAssertNil(
            DefaultZimService.sanitizedWikiTag(
                "https://en.wikipedia.org/wiki/HP_Garage"))
    }

    func testUrlDetectionIsContainsBased() {
        // We don't require the URL to START with a scheme — anything
        // with a `://` is URL-shaped and can't be a Wikipedia tag. A
        // wiki tag would never legitimately contain `://`.
        XCTAssertNil(
            DefaultZimService.sanitizedWikiTag(" https://foo.com"))
        XCTAssertNil(
            DefaultZimService.sanitizedWikiTag("garbage://prefix"))
    }

    func testValuesWithColonButNoSchemePassThrough() {
        // `:` alone is the language-prefix separator; only `://`
        // marks a URL. An edge case like `en:Some:Weird_Title` stays
        // intact — `articleByTitle`'s language-prefix stripper
        // handles the first colon and keeps the rest as the title.
        XCTAssertEqual(
            DefaultZimService.sanitizedWikiTag("en:Some:Weird_Title"),
            "en:Some:Weird_Title")
    }
}
