// SPDX-License-Identifier: MIT
//
// Apple Foundation Models native-tool bridge.
//
// Prototype scope: just `near_named_place`. This exists so we can
// A/B the two Apple-FM integration strategies:
//
//   1. Plain-text preamble + text-based `<|tool_call|{json}>` round-trip
//      through ChatSession's tool loop (current production path for
//      non-Gemma providers).
//   2. Apple's structured `Tool` + `@Generable` protocol where the
//      framework dispatches the tool function directly and keeps a
//      single warmed-up session across iterations.
//
// Strategy 2 should be faster because we pay the 10-15 s session
// warmup once per user turn instead of once per tool iteration, and
// the framework constrains decoding to a valid schema (so the model
// doesn't hallucinate a non-existent tool or argument name).

import Foundation
import MCPZimKit

#if canImport(FoundationModels)
import FoundationModels

@available(macOS 26.0, iOS 19.0, *)
public struct NearNamedPlaceNativeTool: Tool {
    public let service: any ZimService

    public var name: String { "near_named_place" }

    public var description: String {
        "Find points of interest near a named place. Returns the "
            + "resolved place, a category breakdown (by_category), a "
            + "grand total_in_radius, and the top-N ranked results by "
            + "distance. FIRST CALL for any \"what's in X\" / "
            + "\"what's around X\" question: pass ONLY `place` — leave "
            + "`kinds` empty to get the full breakdown. Only set "
            + "`kinds` for a follow-up drill-in (e.g. \"list the "
            + "bars\" → kinds=[\"bar\"])."
    }

    public init(service: any ZimService) { self.service = service }

    @Generable
    public struct Arguments {
        @Guide(description: "Free-text place name, e.g. 'Adams Morgan' or 'Dupont Circle'.")
        public var place: String

        @Guide(description: "Search radius in kilometres. Use 0.5 for 'walking distance', 1-2 for 'neighbourhood'. Default: 1.0.")
        public var radiusKm: Double

        @Guide(description: "Max number of ranked results to return. Default: 10.")
        public var limit: Int

        @Guide(description: "OSM category filter. IF the user's question names a category ('what BARS are in X', 'any CAFES near X', 'COLLEGES around X'), set this to that category (e.g. ['bar'], ['cafe'], ['college']). If the user only asks generically ('what's in X', 'what's around X'), leave it unset to get a by_category breakdown. Never pass ['addr'] — that returns street addresses, not places.")
        public var kinds: [String]?
    }

    public func call(arguments: Arguments) async throws -> String {
        let combined = try await service.nearNamedPlace(
            place: arguments.place,
            radiusKm: arguments.radiusKm > 0 ? arguments.radiusKm : 1.0,
            limit: arguments.limit > 0 ? arguments.limit : 10,
            kinds: (arguments.kinds?.isEmpty == false) ? arguments.kinds : nil,
            zim: nil
        )
        // Apple's framework only asks for a textual tool output here;
        // unlike the text-loop path we don't need to preserve a full
        // JSON envelope. Emit a compact, structured-ish transcript so
        // the model can summarise without having to juggle fields.
        var lines: [String] = []
        let resolved = combined.resolved
        lines.append("resolved: \(resolved.name) (\(resolved.kind)) @ \(resolved.lat), \(resolved.lon)")
        lines.append("total_in_radius: \(combined.result.totalInRadius)")
        let topCats = combined.result.breakdown
            .sorted { a, b in a.value != b.value ? a.value > b.value : a.key < b.key }
        if !topCats.isEmpty {
            let summary = topCats.prefix(12)
                .map { "\($0.value) \($0.key)" }
                .joined(separator: ", ")
            lines.append("by_category (top 12): \(summary)")
        }
        for (i, pair) in combined.result.results.prefix(10).enumerated() {
            let meters = Int(pair.distanceMeters.rounded())
            let subtype = pair.place.subtype.isEmpty ? pair.place.kind : pair.place.subtype
            lines.append("\(i + 1). \(pair.place.name) [\(subtype)] · \(meters) m")
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Remaining conversational-surface tools
//
// Descriptions deliberately mirror the MCP ones in MCPToolAdapter so
// the model's behaviour is the same regardless of which provider the
// user picked. `@Guide` text stays terse because every character is
// prefilled every turn; verbose prose would cost TTFT.

@available(macOS 26.0, iOS 19.0, *)
public struct RouteFromPlacesNativeTool: Tool {
    public let service: any ZimService
    public var name: String { "route_from_places" }
    public var description: String {
        "Plan a driving route between two free-text place names. "
            + "Returns distance, duration, and a numbered turn-by-turn "
            + "list. Use for ANY \"how do I get from X to Y\" question "
            + "instead of asking for lat/lons."
    }
    public init(service: any ZimService) { self.service = service }

    @Generable
    public struct Arguments {
        @Guide(description: "Origin place name, e.g. 'Adams Morgan' or 'Union Station'.")
        public var origin: String
        @Guide(description: "Destination place name, e.g. 'Dupont Circle'.")
        public var destination: String
        @Guide(description: "Optional streetzim filename to restrict to.")
        public var zim: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        let result = try await service.routeFromPlaces(
            origin: arguments.origin,
            destination: arguments.destination,
            zim: arguments.zim
        )
        var lines: [String] = []
        lines.append("origin: \(result.resolved.origin.name) (\(result.resolved.origin.lat), \(result.resolved.origin.lon))")
        lines.append("destination: \(result.resolved.destination.name) (\(result.resolved.destination.lat), \(result.resolved.destination.lon))")
        lines.append(String(format: "distance: %.1f km", result.route.distanceKilometers))
        lines.append(String(format: "duration: %.0f minutes", result.route.durationMinutes))
        if let zim = result.zimUsed {
            lines.append("zim: \(zim)")
        }
        if !result.route.turnByTurn.isEmpty {
            lines.append("turn_by_turn:")
            for (i, step) in result.route.turnByTurn.prefix(20).enumerated() {
                lines.append("\(i + 1). \(step)")
            }
            if result.route.turnByTurn.count > 20 {
                lines.append("… (\(result.route.turnByTurn.count - 20) more)")
            }
        }
        return lines.joined(separator: "\n")
    }
}

@available(macOS 26.0, iOS 19.0, *)
public struct SearchNativeTool: Tool {
    public let service: any ZimService
    public var name: String { "search" }
    public var description: String {
        "Keyword full-text search across every loaded ZIM (encyclopedic "
            + "lookups — Wikipedia-style). Pass SHORT KEYWORDS (article "
            + "titles, proper nouns), not sentences or questions. For "
            + "location questions (\"what's in X\", \"around X\") use "
            + "`near_named_place` instead. Returns paths + titles; read "
            + "bodies with `get_article`."
    }
    public init(service: any ZimService) { self.service = service }

    @Generable
    public struct Arguments {
        @Guide(description: "Short keyword query, e.g. 'Aspirin', 'Marie Curie'. Not a sentence.")
        public var query: String
        @Guide(description: "Max results to return. Default: 10.")
        public var limit: Int
        @Guide(description: "Optional kind filter: 'wikipedia', 'mdwiki', 'streetzim', 'generic'.")
        public var kind: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        let kind = arguments.kind.flatMap { ZimKind(rawValue: $0) }
        let limit = arguments.limit > 0 ? arguments.limit : 10
        let hits = try await service.search(query: arguments.query, limit: limit, kind: kind)
        if hits.isEmpty { return "no hits for \"\(arguments.query)\"" }
        var lines: [String] = ["\(hits.count) hit(s) for \"\(arguments.query)\":"]
        for (i, hit) in hits.prefix(limit).enumerated() {
            lines.append("\(i + 1). [\(hit.kind.rawValue)] \(hit.title) — path=\(hit.path), zim=\(hit.zim)")
        }
        return lines.joined(separator: "\n")
    }
}

@available(macOS 26.0, iOS 19.0, *)
public struct GetArticleNativeTool: Tool {
    public let service: any ZimService
    public var name: String { "get_article" }
    public var description: String {
        "Fetch a single ZIM entry by path (from a prior `search` hit) "
            + "and return its plain-text body. Feed the `path` field "
            + "verbatim — don't guess or reformat it."
    }
    public init(service: any ZimService) { self.service = service }

    @Generable
    public struct Arguments {
        @Guide(description: "ZIM entry path from a search result, e.g. 'A/Aspirin'. Use exactly what search returned.")
        public var path: String
        @Guide(description: "Optional ZIM filename to restrict to.")
        public var zim: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        let art = try await service.article(path: arguments.path, zim: arguments.zim)
        // Cap the body so a full Wikipedia article doesn't blow the
        // model's context or TTFT on the next turn. ~6 K chars is
        // enough for the model to summarise.
        let cap = 6000
        let body = art.text.count > cap
            ? String(art.text.prefix(cap)) + "\n… (truncated)"
            : art.text
        return "title: \(art.title)\nzim: \(art.zim)\npath: \(art.path)\n\n\(body)"
    }
}

@available(macOS 26.0, iOS 19.0, *)
public struct GetMainPageNativeTool: Tool {
    public let service: any ZimService
    public var name: String { "get_main_page" }
    public var description: String {
        "Fetch the main/home page of one or every loaded ZIM."
    }
    public init(service: any ZimService) { self.service = service }

    @Generable
    public struct Arguments {
        @Guide(description: "Optional ZIM filename. Omit for all.")
        public var zim: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        let pages = try await service.mainPage(zim: arguments.zim)
        if pages.isEmpty { return "no main pages available" }
        return pages.map { p -> String in
            let body = p.text.count > 2000
                ? String(p.text.prefix(2000)) + "… (truncated)"
                : p.text
            return "# \(p.zim) — \(p.title)\n\(body)"
        }.joined(separator: "\n\n")
    }
}

@available(macOS 26.0, iOS 19.0, *)
public struct ListLibrariesNativeTool: Tool {
    public let service: any ZimService
    public var name: String { "list_libraries" }
    public var description: String {
        "Inventory the ZIM archives available, with kinds and "
            + "capabilities. Call this first when you're unsure which "
            + "ZIM to consult."
    }
    public init(service: any ZimService) { self.service = service }

    @Generable
    public struct Arguments { /* no parameters */ }

    public func call(arguments: Arguments) async throws -> String {
        let inv = try await service.inventory()
        var lines: [String] = [
            "capabilities: \(inv.capabilities.joined(separator: ", "))",
            "zims (\(inv.zims.count)):",
        ]
        for z in inv.zims {
            lines.append("- \(z.name) [\(z.kind.rawValue)] routing=\(z.hasRoutingData) title=\"\(z.metadata.title)\"")
        }
        return lines.joined(separator: "\n")
    }
}

@available(macOS 26.0, iOS 19.0, *)
public struct ZimInfoNativeTool: Tool {
    public let service: any ZimService
    public var name: String { "zim_info" }
    public var description: String {
        "Return the `streetzim-meta.json` descriptor for loaded "
            + "streetzims — bbox, routing/satellite/wiki-crossref flags, "
            + "feature counts. Use this to check which region a zim "
            + "covers before calling routing or near_named_place."
    }
    public init(service: any ZimService) { self.service = service }

    @Generable
    public struct Arguments {
        @Guide(description: "Optional streetzim filename to restrict to. Omit for all.")
        public var zim: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        let rows = try await service.zimInfo(zim: arguments.zim)
        if rows.isEmpty { return "no streetzim-meta available" }
        // Serialise compactly — the dictionaries are small.
        let data = (try? JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys]))
            ?? Data()
        return String(data: data, encoding: .utf8) ?? "(encoding failed)"
    }
}

#endif
