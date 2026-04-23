// SPDX-License-Identifier: MIT
//
// Pure-function parser for the geo-returning tool families
// (`near_named_place`, `near_places`, `nearby_stories`,
// `nearby_stories_at_place`). Lives in MCPZimKit (not ios/) so it can
// be exercised by `swift test` without linking any UI — the iOS
// `PlacesWebView` is a thin adapter over this type.

import Foundation

/// Subset of the tools whose raw JSON result is expected to carry a
/// geocoded `results` or `stories` array that `parsePlacesJSON` can
/// extract. Host code uses this to decide whether to render a map
/// preview for a given tool trace.
public let placesToolNames: Set<String> = [
    "near_named_place", "near_places",
    "nearby_stories", "nearby_stories_at_place",
]

public struct PlacesPayload: Equatable, Sendable {
    public struct Place: Equatable, Hashable, Sendable {
        public let lat: Double
        public let lon: Double
        public let label: String
        public let description: String
        /// ZIM path to the Wikipedia article covering this place —
        /// set when the tool's result carries a `wiki_path` field
        /// (populated by `fetchWikiExcerpts`). When present, the
        /// iOS popup + list surface a "Read article" button that
        /// dispatches `get_article_section` and renders the article
        /// inline in chat.
        public let wikiPath: String?
        /// Human-readable Wikipedia title (`HP Garage`), distinct
        /// from the raw `wiki` OSM tag (`en:HP_Garage`).
        public let wikiTitle: String?
        /// Overture-places enrichment fields. Populated when the
        /// streetzim was built with `--overture-places`; each is
        /// independent (a row may have a phone but no website, or
        /// vice versa). Rendered as compact chips in the pin popup
        /// and list row so the user can tap to call / open a site
        /// without leaving the map surface.
        public let website: String?
        public let phone: String?
        public let brand: String?

        public init(
            lat: Double, lon: Double,
            label: String, description: String,
            wikiPath: String? = nil, wikiTitle: String? = nil,
            website: String? = nil, phone: String? = nil,
            brand: String? = nil
        ) {
            self.lat = lat
            self.lon = lon
            self.label = label
            self.description = description
            self.wikiPath = wikiPath
            self.wikiTitle = wikiTitle
            self.website = website
            self.phone = phone
            self.brand = brand
        }
    }

    public let places: [Place]
    /// Coverage centre — prefer `resolved.{lat,lon}` (geocoded target
    /// for `*_at_place` / `_named_place` variants), fall back to
    /// `origin` (user coords for plain `near_places` /
    /// `nearby_stories`). Used as the centre of the map's coverage
    /// radius ring.
    public let origin: (lat: Double, lon: Double)?
    /// The `radius_km` arg the tool was invoked with, when present.
    public let radiusKm: Double?

    public init(
        places: [Place],
        origin: (lat: Double, lon: Double)?,
        radiusKm: Double?
    ) {
        self.places = places
        self.origin = origin
        self.radiusKm = radiusKm
    }

    // Equatable explicit since (Double, Double)? tuple doesn't auto-conform.
    public static func == (a: PlacesPayload, b: PlacesPayload) -> Bool {
        a.places == b.places
            && a.origin?.lat == b.origin?.lat
            && a.origin?.lon == b.origin?.lon
            && a.radiusKm == b.radiusKm
    }
}

/// Parse the raw JSON output of a places-returning tool into a
/// `PlacesPayload`. Handles both `results` (`near_places` /
/// `near_named_place`) and `stories` (`nearby_stories[_at_place]`)
/// shapes. Returns an empty payload if the JSON is malformed, the
/// tool isn't a places-returning family, or no lat/lon entries are
/// present.
public func parsePlacesJSON(rawResult: String) -> PlacesPayload {
    guard let data = rawResult.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
        return PlacesPayload(places: [], origin: nil, radiusKm: nil)
    }

    var places: [PlacesPayload.Place] = []

    if let results = json["results"] as? [[String: Any]] {
        for r in results {
            guard let lat = (r["lat"] as? NSNumber)?.doubleValue,
                  let lon = (r["lon"] as? NSNumber)?.doubleValue
            else { continue }
            let name = (r["name"] as? String) ?? "(unnamed)"
            let subtype = (r["subtype"] as? String) ?? ""
            let kind = (r["type"] as? String) ?? ""
            let distanceM = (r["distance_m"] as? NSNumber)?.intValue
            let excerpt = (r["excerpt"] as? String) ?? ""
            let wikiPath = (r["wiki_path"] as? String)?.nilIfEmpty
            let wikiTitle = (r["wiki_title"] as? String)?.nilIfEmpty
            // Overture-places enrichment. `ws` is the website
            // (renamed from `w` — `w` stays reserved for Wikipedia
            // tags). `p` is a phone, `brand` is the primary brand
            // name. All optional — older streetzims without
            // Overture ingestion just leave them nil.
            let website = (r["ws"] as? String)?.nilIfEmpty
            let phone = (r["p"] as? String)?.nilIfEmpty
            let brand = (r["brand"] as? String)?.nilIfEmpty
            // Description priority: Wikipedia excerpt (when the tool
            // enriched this row) > kind · distance. The popup + list
            // both read this field verbatim.
            let description: String = {
                if !excerpt.isEmpty {
                    return excerpt.count > 140
                        ? String(excerpt.prefix(140)) + "…"
                        : excerpt
                }
                var parts: [String] = []
                if !subtype.isEmpty { parts.append(subtype) }
                else if !kind.isEmpty { parts.append(kind) }
                if let d = distanceM { parts.append(formatDistanceMeters(d)) }
                return parts.joined(separator: " · ")
            }()
            places.append(.init(
                lat: lat, lon: lon,
                label: name, description: description,
                wikiPath: wikiPath, wikiTitle: wikiTitle,
                website: website, phone: phone, brand: brand
            ))
        }
    }
    if let stories = json["stories"] as? [[String: Any]] {
        for s in stories {
            guard let lat = (s["lat"] as? NSNumber)?.doubleValue,
                  let lon = (s["lon"] as? NSNumber)?.doubleValue
            else { continue }
            let place = (s["place_name"] as? String) ?? "(unnamed)"
            let excerpt = (s["excerpt"] as? String) ?? ""
            let preview = excerpt.count > 140
                ? String(excerpt.prefix(140)) + "…"
                : excerpt
            let wikiPath = (s["path"] as? String)?.nilIfEmpty
            let wikiTitle = (s["wiki_title"] as? String)?.nilIfEmpty
            places.append(.init(
                lat: lat, lon: lon,
                label: place, description: preview,
                wikiPath: wikiPath, wikiTitle: wikiTitle
            ))
        }
    }

    var origin: (lat: Double, lon: Double)? = nil
    if let resolved = json["resolved"] as? [String: Any],
       let lat = (resolved["lat"] as? NSNumber)?.doubleValue,
       let lon = (resolved["lon"] as? NSNumber)?.doubleValue
    {
        origin = (lat, lon)
    } else if let o = json["origin"] as? [String: Any],
              let lat = (o["lat"] as? NSNumber)?.doubleValue,
              let lon = (o["lon"] as? NSNumber)?.doubleValue
    {
        origin = (lat, lon)
    }

    let radius = (json["radius_km"] as? NSNumber)?.doubleValue

    return PlacesPayload(places: places, origin: origin, radiusKm: radius)
}

private func formatDistanceMeters(_ m: Int) -> String {
    if m < 1000 { return "\(m) m" }
    let km = Double(m) / 1000.0
    return String(format: "%.1f km", km)
}

private extension String {
    /// Collapses empty strings to `nil` so `?.nilIfEmpty` is an
    /// ergonomic way to skip missing fields in the JSON decode path.
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
