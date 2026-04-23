// SPDX-License-Identifier: MIT
//
// Streetzim prefix-chunked JSON geocoder. Mirrors the viewer's `_queryPlaces`
// normalization so lookups return exactly what the on-device UI would.

import Foundation

public struct Place: Equatable, Sendable {
    public let name: String
    public let kind: String   // "addr" | "place" | "poi" | ...
    public let lat: Double
    public let lon: Double
    public let subtype: String
    public let location: String
    /// OSM `wikipedia=` tag (e.g. `"en:Lincoln_Memorial"`). Present on
    /// streetzims built with generator commit a485ce3 or newer.
    public let wiki: String?
    /// OSM `wikidata=` tag (e.g. `"Q162458"`).
    public let wikidata: String?
    /// Overture-places enrichment fields (streetzim `--overture-places`).
    /// Each is independent — a record may carry a phone with no website,
    /// or a brand with neither. Surfaced as `ws` / `p` / `brand` in the
    /// tool JSON so the iOS popup can render call/website chips.
    public let website: String?
    public let phone: String?
    public let brand: String?

    public init(
        name: String, kind: String, lat: Double, lon: Double,
        subtype: String = "", location: String = "",
        wiki: String? = nil, wikidata: String? = nil,
        website: String? = nil, phone: String? = nil, brand: String? = nil
    ) {
        self.name = name
        self.kind = kind
        self.lat = lat
        self.lon = lon
        self.subtype = subtype
        self.location = location
        self.wiki = wiki
        self.wikidata = wikidata
        self.website = website
        self.phone = phone
        self.brand = brand
    }
}

public enum Geocoder {
    /// Normalize a free-text query into the 2-character prefix used to pick a
    /// `search-data/{prefix}.json` chunk. Matches the JS viewer exactly:
    /// lowercase, first two codepoints, spaces → `_`, non-alphanumeric → `_`,
    /// right-padded with `_` if shorter than 2.
    public static func normalizePrefix(_ query: String) -> String {
        let lowered = query.lowercased()
        var prefix = String(lowered.prefix(2))
        prefix = prefix.replacingOccurrences(of: " ", with: "_")
        prefix = String(prefix.map { ch -> Character in
            let ok = (ch == "_") || ch.isASCII && (ch.isLetter || ch.isNumber)
            return ok ? ch : "_"
        })
        while prefix.count < 2 { prefix.append("_") }
        return prefix
    }

    /// Filter a decoded `search-data/{prefix}.json` chunk for records whose
    /// name contains `query` (case-insensitive). Score = earliest match index
    /// then shortest name, matching the Python implementation.
    public static func rank(records: [[String: Any]], query: String, limit: Int, kinds: Set<String>? = nil) -> [Place] {
        let q = query.lowercased()
        if q.isEmpty { return [] }
        var scored: [(offset: Int, length: Int, place: Place)] = []
        for rec in records {
            guard let name = rec["n"] as? String, !name.isEmpty else { continue }
            let kind = (rec["t"] as? String) ?? ""
            if let wanted = kinds, !wanted.contains(kind) { continue }
            let lower = name.lowercased()
            guard let range = lower.range(of: q) else { continue }
            let place = Place(
                name: name,
                kind: kind,
                lat: (rec["a"] as? Double) ?? 0,
                lon: (rec["o"] as? Double) ?? 0,
                subtype: (rec["s"] as? String) ?? "",
                location: (rec["l"] as? String) ?? "",
                wiki: rec["w"] as? String,
                wikidata: rec["q"] as? String,
                website: Self.nonEmpty(rec["ws"] as? String),
                phone: Self.nonEmpty(rec["p"] as? String),
                brand: Self.nonEmpty(rec["brand"] as? String)
            )
            scored.append((
                offset: lower.distance(from: lower.startIndex, to: range.lowerBound),
                length: name.count,
                place: place
            ))
        }
        scored.sort { a, b in
            if a.offset != b.offset { return a.offset < b.offset }
            return a.length < b.length
        }
        return scored.prefix(limit).map { $0.place }
    }

    /// Collapse an empty string to nil so optional Place fields stay nil
    /// when the record didn't carry the tag.
    static func nonEmpty(_ s: String?) -> String? {
        guard let s, !s.isEmpty else { return nil }
        return s
    }
}
