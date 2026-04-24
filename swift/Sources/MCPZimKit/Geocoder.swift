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
    /// Normalize a free-text query into the chunk key used to pick a
    /// `search-data/{prefix}.json` chunk. Must match the writer
    /// `_prefix_key` in streetzim/create_osm_zim.py and the JS viewer
    /// ``keyFor`` — Latin-leading queries get a 2-char ASCII-alnum
    /// prefix, non-ASCII leading queries bucket into
    /// ``"u" + lowercase hex of the first codepoint``.
    ///
    /// Before this split, all non-ASCII names collapsed into a single
    /// ``__.json`` chunk — 350 MB on Japan, 230 MB on Iran — which
    /// crashed Kiwix Desktop on "find". The non-ASCII bucket now spreads
    /// across one chunk per distinct leading codepoint.
    public static func normalizePrefix(_ query: String) -> String {
        let lowered = query.lowercased().replacingOccurrences(of: " ", with: "_")
        guard let c0 = lowered.first else { return "__" }
        // Non-ASCII → codepoint hex bucket (matches writer _prefix_key).
        if !c0.isASCII, let scalar = c0.unicodeScalars.first {
            return "u" + String(scalar.value, radix: 16)
        }
        // ASCII path: alnum or '_' kept; anything else → '_'.
        func asciiNorm(_ ch: Character) -> Character {
            if ch == "_" { return "_" }
            if ch.isASCII && (ch.isLetter || ch.isNumber) { return ch }
            return "_"
        }
        var prefix = String(asciiNorm(c0))
        if lowered.count >= 2 {
            let c1 = lowered[lowered.index(lowered.startIndex, offsetBy: 1)]
            // 2nd char non-ASCII → collapse to '_' (bucket keyed by c0).
            prefix.append(c1.isASCII ? asciiNorm(c1) : "_")
        } else {
            prefix.append("_")
        }
        return prefix
    }

    /// Sub-bucket hash used when a search-data chunk was split into
    /// `{prefix}-{hex}` sub-files during repackage. Must match the
    /// writer (`cloud/repackage_zim._sub_bucket_for_name`) and JS
    /// (`resources/viewer/index.html` `subBucketFor`). FNV-1a 32-bit
    /// over the UTF-8 bytes; result mod ``nBuckets``.
    public static func subBucketFor(name: String, nBuckets: Int = 16) -> Int {
        var h: UInt32 = 0x811C9DC5
        for b in name.utf8 {
            h ^= UInt32(b)
            h = h &* 0x01000193
        }
        // Take the modulo in UInt32 space before converting, so we never
        // rely on `Int(h)` being non-negative (it is on 64-bit Swift
        // today, but would break if the hash is ever widened to UInt64).
        precondition(nBuckets > 0, "nBuckets must be > 0")
        return Int(h % UInt32(nBuckets))
    }

    /// Expand a query prefix through the manifest's ``sub_chunks``
    /// dictionary. Returns a list of actual chunk prefixes to fetch.
    /// When the query's prefix wasn't hot-split, returns a single-item
    /// list with that prefix unchanged; when it WAS split, returns the
    /// sub-bucket list the writer emitted (so callers fetch every
    /// sub-file to cover all possible hits for that query).
    public static func expandPrefix(_ prefix: String,
                                     manifest: [String: Any]) -> [String] {
        if let subs = (manifest["sub_chunks"] as? [String: [String]])?[prefix] {
            return subs
        }
        return [prefix]
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
