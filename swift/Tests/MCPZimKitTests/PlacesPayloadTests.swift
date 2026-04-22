// SPDX-License-Identifier: MIT
//
// Unit tests for `parsePlacesJSON` — exercises the `results` and
// `stories` shapes that the four places-returning tools
// (`near_named_place`, `near_places`, `nearby_stories`,
// `nearby_stories_at_place`) emit. Each test feeds a canned JSON
// payload and asserts on the extracted places + origin + radius.

import Foundation
import XCTest

@testable import MCPZimKit

final class PlacesPayloadTests: XCTestCase {

    // MARK: - `results` shape (near_places / near_named_place)

    func testParsesNearNamedPlaceResults() throws {
        let json = """
        {
            "radius_km": 1.5,
            "total_in_radius": 76,
            "results": [
                {"name": "The Saloon", "type": "amenity", "subtype": "bar",
                 "lat": 37.80012, "lon": -122.40876, "distance_m": 184},
                {"name": "Vesuvio Cafe", "type": "amenity", "subtype": "bar",
                 "lat": 37.79742, "lon": -122.40688, "distance_m": 523}
            ],
            "origin": {"lat": 37.44122, "lon": -122.15527},
            "resolved": {"name": "North Beach", "lat": 37.80117, "lon": -122.40900}
        }
        """

        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.count, 2)
        XCTAssertEqual(p.places[0].label, "The Saloon")
        XCTAssertEqual(p.places[0].lat, 37.80012, accuracy: 1e-5)
        XCTAssertEqual(p.places[0].lon, -122.40876, accuracy: 1e-5)
        XCTAssertEqual(p.places[0].description, "bar · 184 m")

        XCTAssertEqual(p.places[1].label, "Vesuvio Cafe")
        XCTAssertEqual(p.places[1].description, "bar · 523 m")

        // resolved wins over origin as the coverage centre.
        XCTAssertEqual(p.origin?.lat ?? 0, 37.80117, accuracy: 1e-5)
        XCTAssertEqual(p.origin?.lon ?? 0, -122.40900, accuracy: 1e-5)
        XCTAssertEqual(p.radiusKm, 1.5)
    }

    func testFallsBackToOriginWhenNoResolved() throws {
        let json = """
        {
            "radius_km": 2,
            "results": [
                {"name": "X", "type": "poi", "lat": 1.0, "lon": 2.0, "distance_m": 50}
            ],
            "origin": {"lat": 37.44, "lon": -122.15}
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.origin?.lat ?? 0, 37.44, accuracy: 1e-5)
        XCTAssertEqual(p.origin?.lon ?? 0, -122.15, accuracy: 1e-5)
        XCTAssertEqual(p.radiusKm, 2)
    }

    func testDistanceInKilometersWhenOver1000m() throws {
        let json = """
        {
            "results": [
                {"name": "Far One", "type": "restaurant",
                 "lat": 37.8, "lon": -122.4, "distance_m": 2400}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.first?.description, "restaurant · 2.4 km")
    }

    func testSkipsResultsMissingLatLon() throws {
        let json = """
        {
            "results": [
                {"name": "Has Coords", "type": "poi", "lat": 10.0, "lon": 20.0},
                {"name": "No Coords", "type": "poi"},
                {"name": "Half Coords", "type": "poi", "lat": 10.0}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.count, 1)
        XCTAssertEqual(p.places.first?.label, "Has Coords")
    }

    func testMissingNameBecomesPlaceholder() throws {
        let json = """
        {"results": [{"type": "poi", "lat": 10.0, "lon": 20.0}]}
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.first?.label, "(unnamed)")
    }

    // MARK: - `stories` shape (nearby_stories / nearby_stories_at_place)

    func testParsesNearbyStoriesResults() throws {
        let json = """
        {
            "radius_km": 3,
            "count": 2,
            "stories": [
                {"place_name": "HP Garage", "wiki_title": "HP Garage",
                 "wiki_tag": "en:HP_Garage",
                 "lat": 37.44453, "lon": -122.15269,
                 "distance_m": 750,
                 "excerpt": "The HP Garage is a private museum where Hewlett-Packard was founded."},
                {"place_name": "Palo Alto", "wiki_title": "Palo Alto",
                 "wiki_tag": "en:Palo_Alto",
                 "lat": 37.44188, "lon": -122.14302,
                 "distance_m": 150,
                 "excerpt": "Palo Alto is a charter city in the northwestern corner of Santa Clara County."}
            ],
            "origin": {"lat": 37.44, "lon": -122.15}
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.count, 2)
        XCTAssertEqual(p.places[0].label, "HP Garage")
        XCTAssertEqual(p.places[0].lat, 37.44453, accuracy: 1e-5)
        XCTAssertTrue(p.places[0].description.contains("private museum"))
        XCTAssertEqual(p.radiusKm, 3)
    }

    func testStoriesExcerptTruncatesAt140Chars() throws {
        let longExcerpt = String(repeating: "lorem ipsum dolor sit amet, ", count: 10)
        XCTAssertGreaterThan(longExcerpt.count, 140)
        let json = """
        {
            "stories": [
                {"place_name": "X", "lat": 1, "lon": 2, "excerpt": "\(longExcerpt)"}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        let desc = p.places.first?.description ?? ""
        XCTAssertTrue(desc.hasSuffix("…"))
        XCTAssertLessThanOrEqual(desc.count, 141)  // 140 + the ellipsis
    }

    func testStoriesShortExcerptKeptVerbatim() throws {
        let json = """
        {
            "stories": [
                {"place_name": "X", "lat": 1, "lon": 2,
                 "excerpt": "A short preview."}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.first?.description, "A short preview.")
    }

    // MARK: - Edge cases

    func testEmptyPayloadWhenJSONInvalid() {
        let p = parsePlacesJSON(rawResult: "not json")
        XCTAssertTrue(p.places.isEmpty)
        XCTAssertNil(p.origin)
        XCTAssertNil(p.radiusKm)
    }

    func testEmptyPayloadWhenNoResultsOrStories() {
        let p = parsePlacesJSON(rawResult: "{\"radius_km\": 5}")
        XCTAssertTrue(p.places.isEmpty)
        XCTAssertEqual(p.radiusKm, 5)
    }

    func testCombinedResultsAndStoriesAppend() throws {
        // If a caller ever routes both shapes into the same payload
        // (not currently done in practice but cheap to handle), we
        // union them rather than dropping one.
        let json = """
        {
            "results": [
                {"name": "R1", "type": "poi", "lat": 1.0, "lon": 2.0}
            ],
            "stories": [
                {"place_name": "S1", "lat": 3.0, "lon": 4.0, "excerpt": "x"}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.count, 2)
        XCTAssertEqual(Set(p.places.map(\.label)), ["R1", "S1"])
    }

    // MARK: - Enriched near_places results (excerpt + wiki_path)

    func testResultsExcerptWinsOverKindDistance() {
        let json = """
        {
            "results": [
                {"name": "HP Garage", "type": "poi", "subtype": "museum",
                 "lat": 37.44453, "lon": -122.15269, "distance_m": 750,
                 "excerpt": "The HP Garage is a private museum where "
                   + "Hewlett-Packard was founded.",
                 "wiki_path": "A/HP_Garage",
                 "wiki_title": "HP Garage"}
            ]
        }
        """
        // (Swift doesn't concat string literals inside JSON; sanitise
        // by hand to keep the input a single literal below.)
        let p = parsePlacesJSON(rawResult: """
        {
            "results": [
                {"name": "HP Garage", "type": "poi", "subtype": "museum",
                 "lat": 37.44453, "lon": -122.15269, "distance_m": 750,
                 "excerpt": "The HP Garage is a private museum where Hewlett-Packard was founded.",
                 "wiki_path": "A/HP_Garage",
                 "wiki_title": "HP Garage"}
            ]
        }
        """)
        XCTAssertEqual(p.places.count, 1)
        let pl = p.places[0]
        XCTAssertTrue(pl.description.contains("private museum"),
                      "excerpt should replace kind·distance when present")
        XCTAssertFalse(pl.description.contains("750 m"),
                       "distance should be suppressed when excerpt is present")
        XCTAssertEqual(pl.wikiPath, "A/HP_Garage")
        XCTAssertEqual(pl.wikiTitle, "HP Garage")
        _ = json   // silence unused
    }

    func testResultsExcerptTruncatesAt140Chars() {
        let longExcerpt = String(repeating: "A long excerpt segment. ", count: 20)
        let json = """
        {
            "results": [
                {"name": "X", "type": "poi", "lat": 1.0, "lon": 2.0,
                 "excerpt": "\(longExcerpt)",
                 "wiki_path": "A/X"}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        let desc = p.places.first?.description ?? ""
        XCTAssertTrue(desc.hasSuffix("…"))
        XCTAssertLessThanOrEqual(desc.count, 141)
    }

    func testResultsWithoutExcerptFallBackToKindDistance() {
        let json = """
        {
            "results": [
                {"name": "NoWiki", "type": "amenity", "subtype": "bar",
                 "lat": 1.0, "lon": 2.0, "distance_m": 200}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertEqual(p.places.first?.description, "bar · 200 m")
        XCTAssertNil(p.places.first?.wikiPath)
    }

    func testStoriesPathExposedAsWikiPath() {
        let json = """
        {
            "stories": [
                {"place_name": "HP Garage", "wiki_title": "HP Garage",
                 "path": "A/HP_Garage",
                 "lat": 37.44, "lon": -122.15,
                 "excerpt": "The HP Garage is a private museum."}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        let pl = p.places.first
        XCTAssertEqual(pl?.wikiPath, "A/HP_Garage")
        XCTAssertEqual(pl?.wikiTitle, "HP Garage")
    }

    func testEmptyWikiFieldsMapToNil() {
        let json = """
        {
            "results": [
                {"name": "X", "type": "poi", "lat": 1.0, "lon": 2.0,
                 "wiki_path": "", "wiki_title": ""}
            ]
        }
        """
        let p = parsePlacesJSON(rawResult: json)
        XCTAssertNil(p.places.first?.wikiPath)
        XCTAssertNil(p.places.first?.wikiTitle)
    }

    func testPlacesToolNamesCoversEveryKnownFamily() {
        XCTAssertTrue(placesToolNames.contains("near_named_place"))
        XCTAssertTrue(placesToolNames.contains("near_places"))
        XCTAssertTrue(placesToolNames.contains("nearby_stories"))
        XCTAssertTrue(placesToolNames.contains("nearby_stories_at_place"))
        // Non-places tools must NOT be in the set.
        XCTAssertFalse(placesToolNames.contains("what_is_here"))
        XCTAssertFalse(placesToolNames.contains("plan_driving_route"))
        XCTAssertFalse(placesToolNames.contains("search"))
    }
}
