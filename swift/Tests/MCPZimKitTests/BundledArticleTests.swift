// SPDX-License-Identifier: MIT
//
// (b) `articleByTitle` must resolve an article a streetzim bundled inline
// at `wiki-article/<Title>` (option B), so `narrate_article` works fully
// offline when no Wikipedia ZIM is loaded — kiwix can't deep-link across
// ZIMs, so the streetzim has to carry its own copy. A real Wikipedia ZIM,
// when present, still wins.

import Foundation
import XCTest
@testable import MCPZimKit

final class BundledArticleTests: XCTestCase {

    /// In-memory reader of a given `ZimKind` serving a path→HTML map.
    private final class MapReader: ZimReader, @unchecked Sendable {
        let store: [String: Data]
        let _kind: ZimKind
        init(kind: ZimKind, _ pages: [String: String]) {
            _kind = kind
            store = pages.reduce(into: [:]) { $0[$1.key] = Data($1.value.utf8) }
        }
        var metadata: ZimMetadata { ZimMetadata(name: "m") }
        var kind: ZimKind { _kind }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { false }
        var hasRoutingData: Bool { _kind == .streetzim }
        func read(path: String) throws -> ZimEntry? {
            store[path].map { ZimEntry(path: path, title: path, mimetype: "text/html", content: $0) }
        }
        func readMainPage() throws -> ZimEntry? { nil }
    }

    private let articleHTML = """
    <h1>Camarillo Ranch House</h1>
    <p>The <b>Camarillo Ranch House</b> is a Queen Anne Victorian house in \
    Camarillo, California, built in 1892 by Adolfo Camarillo.</p>
    <h2>History</h2><p>It was added to the National Register in 1996.</p>
    """

    func testResolvesStreetzimBundledArticleWithNoWikipediaZim() async throws {
        // Only a streetzim, carrying the article at wiki-article/<Title>.
        let sz = MapReader(kind: .streetzim,
                           ["wiki-article/Camarillo_Ranch_House": articleHTML])
        let svc = DefaultZimService(readers: [(name: "osm-ca", reader: sz)])

        // Pass the OSM-tag form ("en:...") with spaces — the resolver
        // strips the lang prefix and tries underscore/case variants.
        let hit = try await svc.articleByTitle(
            title: "en:Camarillo Ranch House", zim: nil, section: "lead")
        XCTAssertEqual(hit.zim, "osm-ca")
        XCTAssertEqual(hit.path, "wiki-article/Camarillo_Ranch_House")
        XCTAssertTrue(hit.section.text.contains("Queen Anne Victorian"),
                      "lead text parsed from the bundled article: \(hit.section.text)")
    }

    func testWikipediaZimWinsOverBundledCopy() async throws {
        let wiki = MapReader(kind: .wikipedia,
                             ["A/Camarillo_Ranch_House": "<p>Full Wikipedia article body.</p>"])
        let sz = MapReader(kind: .streetzim,
                           ["wiki-article/Camarillo_Ranch_House": articleHTML])
        // streetzim listed first to prove priority is by kind, not order.
        let svc = DefaultZimService(readers: [(name: "osm-ca", reader: sz),
                                              (name: "wiki", reader: wiki)])
        let hit = try await svc.articleByTitle(
            title: "Camarillo Ranch House", zim: nil, section: "lead")
        XCTAssertEqual(hit.zim, "wiki", "the complete Wikipedia ZIM wins")
        XCTAssertTrue(hit.section.text.contains("Full Wikipedia article body"))
    }

    func testThrowsWhenNeitherHasIt() async throws {
        let sz = MapReader(kind: .streetzim, ["wiki-article/Somewhere_Else": "<p>x</p>"])
        let svc = DefaultZimService(readers: [(name: "osm-ca", reader: sz)])
        do {
            _ = try await svc.articleByTitle(
                title: "Camarillo Ranch House", zim: nil, section: "lead")
            XCTFail("expected notFound")
        } catch ZimServiceError.notFound {
            // expected
        }
    }
}
