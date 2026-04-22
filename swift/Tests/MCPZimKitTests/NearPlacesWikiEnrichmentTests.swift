// SPDX-License-Identifier: MIT
//
// End-to-end check that `MCPToolAdapter.dispatch("near_places" | "near_named_place", …)`
// actually produces result rows enriched with `excerpt` / `wiki_path` /
// `wiki_title` when the underlying service returns places carrying a
// `wiki` OSM tag. A unit test of `parsePlacesJSON` in isolation isn't
// enough — it asserts "given enriched JSON, the iOS side extracts it
// right", but doesn't verify the upstream pipeline actually builds
// that enriched JSON.

import Foundation
import XCTest

@testable import MCPZimKit

final class NearPlacesWikiEnrichmentTests: XCTestCase {

    // MARK: - Fixture builders

    /// A small near_places result: two museums, one with a wiki tag,
    /// one without. Matches the real OSM shape where only a subset
    /// of POIs in the returned radius carry a Wikipedia cross-ref.
    private func fixtureWithOneWikiTaggedMuseum() -> StubZimService.Fixture {
        var fix = StubZimService.Fixture()

        let cantor = Place(
            name: "Cantor Arts Center",
            kind: "poi",
            lat: 37.432,
            lon: -122.170,
            subtype: "museum",
            wiki: "en:Cantor_Arts_Center",
            wikidata: "Q5035378"
        )
        let noWiki = Place(
            name: "The Foster Museum",
            kind: "poi",
            lat: 37.410,
            lon: -122.145,
            subtype: "museum"
        )

        let result = NearPlacesResult(
            totalInRadius: 2,
            breakdown: ["museum": 2],
            results: [
                (place: cantor, distanceMeters: 1850),
                (place: noWiki, distanceMeters: 3400),
            ]
        )

        // near_places keyed on (lat, lon, kinds).
        let npKey = StubZimService.keyNearPlaces(
            lat: 37.441, lon: -122.155, kinds: ["museum"]
        )
        fix.nearPlaces[npKey] = .init(result: result)

        // articleByTitle keyed on the wiki tag + section. Only the
        // Cantor article is stocked; "The Foster Museum" has no
        // wiki tag so it won't be queried.
        let cantorArticleKey = StubZimService.keyArticleByTitle(
            title: "en:Cantor_Arts_Center", section: "lead"
        )
        fix.articleByTitle[cantorArticleKey] = .init(
            zim: "wikipedia_en_all_maxi_2025-10.zim",
            path: "A/Cantor_Arts_Center",
            title: "Cantor Arts Center",
            section: ArticleSection(
                title: "lead",
                level: 0,
                text: "The Iris & B. Gerald Cantor Center for Visual Arts "
                    + "(Cantor Arts Center) is an art museum on the campus "
                    + "of Stanford University in Stanford, California."
            )
        )
        return fix
    }

    /// Same as above but as a nearNamedPlace fixture — the dispatcher
    /// path for `near_named_place` is a separate branch.
    private func namedPlaceFixture(_ base: StubZimService.Fixture) -> StubZimService.Fixture {
        var fix = base
        let resolved = Place(
            name: "Palo Alto",
            kind: "place",
            lat: 37.441,
            lon: -122.155,
            subtype: "city"
        )
        // Pull the existing near_places entry and mirror it into
        // nearNamedPlace keyed on ("palo alto", ["museum"]).
        let npKey = StubZimService.keyNearPlaces(
            lat: 37.441, lon: -122.155, kinds: ["museum"]
        )
        guard let r = base.nearPlaces[npKey]?.result else {
            return fix
        }
        let key = StubZimService.keyNearNamedPlace(
            place: "Palo Alto", kinds: ["museum"]
        )
        fix.nearNamedPlace[key] = .init(resolved: resolved, result: r)
        return fix
    }

    // MARK: - near_places dispatch

    func testNearPlacesEnrichesWikiTaggedResultsWithExcerpt() async throws {
        let svc = StubZimService(fixture: fixtureWithOneWikiTaggedMuseum())
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)

        let result = try await adapter.dispatch(tool: "near_places", args: [
            "lat": 37.441,
            "lon": -122.155,
            "kinds": ["museum"],
            "radius_km": 5.0,
        ])

        let rows = try XCTUnwrap(result["results"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 2)

        let cantorRow = try XCTUnwrap(rows.first { ($0["name"] as? String) == "Cantor Arts Center" })
        XCTAssertEqual(cantorRow["wiki_path"] as? String, "A/Cantor_Arts_Center",
                       "tagged row should carry a wiki_path")
        XCTAssertEqual(cantorRow["wiki_title"] as? String, "Cantor Arts Center")
        let excerpt = try XCTUnwrap(cantorRow["excerpt"] as? String)
        XCTAssertTrue(excerpt.contains("Cantor"), "excerpt should include the article lead")

        let fosterRow = try XCTUnwrap(rows.first { ($0["name"] as? String) == "The Foster Museum" })
        XCTAssertNil(fosterRow["excerpt"],
                     "rows without a wiki tag should stay un-enriched")
        XCTAssertNil(fosterRow["wiki_path"])
    }

    // MARK: - near_named_place dispatch

    func testNearNamedPlaceEnrichesWikiTaggedResultsWithExcerpt() async throws {
        let base = fixtureWithOneWikiTaggedMuseum()
        let svc = StubZimService(fixture: namedPlaceFixture(base))
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)

        let result = try await adapter.dispatch(tool: "near_named_place", args: [
            "place": "Palo Alto",
            "kinds": ["museum"],
            "radius_km": 5.0,
        ])

        let rows = try XCTUnwrap(result["results"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 2)

        let cantorRow = try XCTUnwrap(rows.first { ($0["name"] as? String) == "Cantor Arts Center" })
        XCTAssertEqual(cantorRow["wiki_path"] as? String, "A/Cantor_Arts_Center")
        XCTAssertNotNil(cantorRow["excerpt"])
    }

    // MARK: - Name-search fallback when streetzim didn't tag

    /// Mirrors the real failure mode observed on
    /// `osm-silicon-valley-flagtest2.zim`: every museum comes back
    /// with `wiki == nil` because the streetzim build-time
    /// (name, lat, lon)-join didn't match. The iOS side should
    /// fall back to a Wikipedia name-search and still enrich the
    /// row.
    private func fixtureNoWikiTagsButArticlesExist() -> StubZimService.Fixture {
        var fix = StubZimService.Fixture()

        // Three real museums near Palo Alto — none have `wiki` set.
        let museums: [Place] = [
            Place(name: "Palo Alto History Museum",
                  kind: "poi", lat: 37.4419, lon: -122.1441, subtype: "museum"),
            Place(name: "Museum of American Heritage",
                  kind: "poi", lat: 37.4428, lon: -122.1455, subtype: "museum"),
            Place(name: "Palo Alto Junior Museum and Zoo",
                  kind: "poi", lat: 37.4324, lon: -122.1410, subtype: "museum"),
        ]
        let result = NearPlacesResult(
            totalInRadius: 3,
            breakdown: ["museum": 3],
            results: museums.map { ($0, 500.0) }
        )
        let key = StubZimService.keyNearPlaces(
            lat: 37.441, lon: -122.155, kinds: ["museum"]
        )
        fix.nearPlaces[key] = .init(result: result)

        // Wikipedia search returns a close-match top hit for two of the
        // three — one gets an unrelated hit (should be rejected).
        fix.search[StubZimService.keySearch(query: "Palo Alto History Museum")] = [
            SearchHitResult(
                zim: "wikipedia_en_all_maxi_2025-10.zim",
                kind: .wikipedia,
                path: "A/Palo_Alto_History_Museum",
                title: "Palo Alto History Museum",
                snippet: ""
            )
        ]
        fix.search[StubZimService.keySearch(query: "Museum of American Heritage")] = [
            SearchHitResult(
                zim: "wikipedia_en_all_maxi_2025-10.zim",
                kind: .wikipedia,
                path: "A/Museum_of_American_Heritage",
                title: "Museum of American Heritage",
                snippet: ""
            )
        ]
        // Junior Museum and Zoo — search returns a totally unrelated
        // "Zoo" article; should be rejected by the title-match guard.
        fix.search[StubZimService.keySearch(query: "Palo Alto Junior Museum and Zoo")] = [
            SearchHitResult(
                zim: "wikipedia_en_all_maxi_2025-10.zim",
                kind: .wikipedia,
                path: "A/Zoo",
                title: "Zoo",
                snippet: ""
            )
        ]

        // Lead-paragraph stubs for the two matches.
        fix.articleByTitle[StubZimService.keyArticleByTitle(
            title: "Palo_Alto_History_Museum", section: "lead"
        )] = .init(
            zim: "wikipedia_en_all_maxi_2025-10.zim",
            path: "A/Palo_Alto_History_Museum",
            title: "Palo Alto History Museum",
            section: ArticleSection(
                title: "lead", level: 0,
                text: "The Palo Alto History Museum is a local history museum located in downtown Palo Alto, California."
            )
        )
        fix.articleByTitle[StubZimService.keyArticleByTitle(
            title: "Museum_of_American_Heritage", section: "lead"
        )] = .init(
            zim: "wikipedia_en_all_maxi_2025-10.zim",
            path: "A/Museum_of_American_Heritage",
            title: "Museum of American Heritage",
            section: ArticleSection(
                title: "lead", level: 0,
                text: "The Museum of American Heritage (MOAH) is a non-profit museum in Palo Alto, California."
            )
        )

        return fix
    }

    func testFallbackSearchFindsArticlesForUntaggedMuseums() async throws {
        let svc = StubZimService(fixture: fixtureNoWikiTagsButArticlesExist())
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)

        let result = try await adapter.dispatch(tool: "near_places", args: [
            "lat": 37.441,
            "lon": -122.155,
            "kinds": ["museum"],
            "radius_km": 5.0,
        ])

        let rows = try XCTUnwrap(result["results"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 3)

        // The two museums whose Wikipedia titles close-match their
        // place names should end up enriched.
        let pahmOpt: [String: Any]? = rows.first {
            ($0["name"] as? String) == "Palo Alto History Museum"
        }
        let pahm = try XCTUnwrap(pahmOpt)
        XCTAssertEqual(pahm["wiki_path"] as? String, "A/Palo_Alto_History_Museum",
                       "fallback name-search should pick up untagged rows")
        XCTAssertEqual(pahm["wiki_title"] as? String, "Palo Alto History Museum")
        let excerpt = try XCTUnwrap(pahm["excerpt"] as? String)
        XCTAssertTrue(excerpt.contains("Palo Alto"))

        let moahOpt: [String: Any]? = rows.first {
            ($0["name"] as? String) == "Museum of American Heritage"
        }
        let moah = try XCTUnwrap(moahOpt)
        XCTAssertEqual(moah["wiki_path"] as? String, "A/Museum_of_American_Heritage")

        // The zoo got a bad search hit — the title-match guard must
        // reject it rather than stapling an unrelated article on.
        let zooOpt: [String: Any]? = rows.first {
            ($0["name"] as? String) == "Palo Alto Junior Museum and Zoo"
        }
        let zoo = try XCTUnwrap(zooOpt)
        XCTAssertNil(zoo["wiki_path"], "unrelated search hits must be rejected")
    }

    func testOsmTagStillWinsOverSearchFallback() async throws {
        // A place with BOTH a wiki tag AND a searchable name — the
        // OSM-tag path is faster + more canonical, so it should be
        // used and the search fallback must not duplicate work.
        let base = fixtureWithOneWikiTaggedMuseum()
        let svc = StubZimService(fixture: base)
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)
        let result = try await adapter.dispatch(tool: "near_places", args: [
            "lat": 37.441,
            "lon": -122.155,
            "kinds": ["museum"],
            "radius_km": 5.0,
        ])
        let rows = try XCTUnwrap(result["results"] as? [[String: Any]])
        let cantor = try XCTUnwrap(rows.first { ($0["name"] as? String) == "Cantor Arts Center" })
        XCTAssertEqual(cantor["wiki_path"] as? String, "A/Cantor_Arts_Center",
                       "OSM-tagged path resolves first; fallback shouldn't overwrite it")
    }

    // MARK: - parsePlacesJSON consumes the enriched output

    func testEndToEndResultRouteFromDispatchThroughParse() async throws {
        let svc = StubZimService(fixture: fixtureWithOneWikiTaggedMuseum())
        let adapter = await MCPToolAdapter(service: svc, hasStreetzim: true)
        let result = try await adapter.dispatch(tool: "near_places", args: [
            "lat": 37.441,
            "lon": -122.155,
            "kinds": ["museum"],
            "radius_km": 5.0,
        ])
        let rawJSON = try String(
            data: JSONSerialization.data(withJSONObject: result, options: []),
            encoding: .utf8
        ) ?? "{}"

        let payload: PlacesPayload = parsePlacesJSON(rawResult: rawJSON)
        let cantorOpt: PlacesPayload.Place? = payload.places
            .first { $0.label == "Cantor Arts Center" }
        let cantor = try XCTUnwrap(cantorOpt)
        XCTAssertEqual(cantor.wikiPath, "A/Cantor_Arts_Center")
        XCTAssertEqual(cantor.wikiTitle, "Cantor Arts Center")
        XCTAssertTrue(cantor.description.contains("Cantor"))

        let fosterOpt: PlacesPayload.Place? = payload.places
            .first { $0.label == "The Foster Museum" }
        let foster = try XCTUnwrap(fosterOpt)
        XCTAssertNil(foster.wikiPath)
    }
}
