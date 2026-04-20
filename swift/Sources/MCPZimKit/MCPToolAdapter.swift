// SPDX-License-Identifier: MIT
//
// Transport-agnostic MCP tool adapter.
//
// Exposes the seven MCPZim tools — four core plus three streetzim-only — as
// JSON schemas plus a single async JSON dispatch entrypoint. Host apps plug
// this into whatever transport they use:
//
//   * OpenAI-style function calling (just feed `toolList` into the model call).
//   * The official Swift MCP SDK (register each tool with `Server.addTool`,
//     forwarding the args dict to `dispatch(tool:args:)`).
//   * Direct in-process Swift calls (skip this file entirely and use
//     `ZimService` directly).
//
// Keeping this transport-agnostic means MCPZimKit has zero external deps and
// upgrades to the Swift MCP SDK don't ripple into this package's API.

import Foundation

public struct MCPTool: Sendable {
    public let name: String
    public let description: String
    /// JSON Schema for the tool's input, encoded as `Data` (UTF-8 JSON).
    /// Transport wrappers that want a `[String: Any]` can `JSONSerialization`
    /// decode this on the way out.
    public let inputSchemaJSON: Data
}

public struct MCPToolRegistry: Sendable {
    public let tools: [MCPTool]
}

/// Which slice of the tool catalogue to publish. The full surface exposes
/// raw-coordinate tools (`near_places`, `geocode`, `plan_driving_route`)
/// which are convenient for programmatic callers but a persistent footgun
/// for an LLM — the model routinely passes text where coordinates go, or
/// defaults lat/lon to 0. The conversational surface hides those and
/// keeps only tools that take free-text names.
public enum ToolSurface: Sendable {
    case conversational
    case full
}

/// Optional post-processor for `search` tool hits. Takes the user's
/// query text + the BM25-ranked hit list and returns a possibly-
/// reordered (and possibly-shortened) list. Hosted outside MCPZimKit
/// so downstream callers can plug in platform-specific rerankers
/// (e.g. `NLContextualEmbedding` on Apple platforms) without pulling
/// those dependencies into this package.
public typealias HitReranker = @Sendable (_ query: String, _ hits: [SearchHitResult]) async -> [SearchHitResult]

public actor MCPToolAdapter {
    private let service: any ZimService
    private let hasStreetzim: Bool
    private let surface: ToolSurface
    /// Category slugs advertised by the loaded streetzim(s), cached at
    /// construction time. Surfaced into the `kinds` parameter schema so
    /// MCP clients see the exact OSM vocabulary their data supports.
    private let categoryVocabulary: [String]

    /// Installed by the host if a semantic reranker is available. When
    /// set, we over-fetch BM25 candidates and let the reranker choose
    /// the top K — the model then sees the semantically-closest hits
    /// first, not just the keyword-densest.
    private var hitReranker: HitReranker?

    public init(service: any ZimService, hasStreetzim: Bool, surface: ToolSurface = .full, categoryVocabulary: [String] = []) {
        self.service = service
        self.hasStreetzim = hasStreetzim
        self.surface = surface
        self.categoryVocabulary = categoryVocabulary
    }

    /// Install (or clear) a semantic reranker. Passing `nil` drops
    /// back to plain BM25 order.
    public func installHitReranker(_ reranker: HitReranker?) {
        self.hitReranker = reranker
    }

    public static func from(service: any ZimService, surface: ToolSurface = .full) async -> MCPToolAdapter {
        let inventory = (try? await service.inventory()) ?? InventoryResult(zims: [], capabilities: [])
        let hasStreetzim = inventory.zims.contains { $0.kind == .streetzim && $0.hasRoutingData }
        let vocab: [String]
        if let concrete = service as? DefaultZimService {
            vocab = await concrete.categoryVocabulary()
        } else {
            vocab = []
        }
        return MCPToolAdapter(service: service, hasStreetzim: hasStreetzim, surface: surface, categoryVocabulary: vocab)
    }

    public var registry: MCPToolRegistry {
        var tools: [MCPTool] = [
            MCPTool(
                name: "list_libraries",
                description:
                    "Inventory the ZIM archives this server can read. Returns kinds, metadata, "
                    + "and the aggregate capabilities exposed by this tool set. Call this first.",
                inputSchemaJSON: Self.emptyObjectSchema
            ),
            MCPTool(
                name: "search",
                description:
                    "Keyword full-text search across every loaded ZIM, intended "
                    + "for ENCYCLOPEDIC lookups (Wikipedia-style). This is plain "
                    + "FTS — pass SHORT KEYWORDS (article titles, proper nouns), "
                    + "not full sentences or questions. "
                    + "Good: `query=\"Aspirin\"`, `query=\"Marie Curie\"`. "
                    + "Bad: `query=\"what is aspirin used for\"`. "
                    + "For LOCATION questions (\"what's in X\", \"near X\", "
                    + "\"around X\") prefer `near_named_place` instead — it "
                    + "returns coordinates and a category breakdown. Returns "
                    + "paths, titles, AND a ~200-char snippet from each hit's "
                    + "lead paragraph. READ THE SNIPPETS before picking — the "
                    + "top BM25 match isn't always the topically-closest one "
                    + "(e.g. a query about quantum computing's impact on "
                    + "encryption will surface \"Crypto-shredding\" because "
                    + "of keyword overlap, but the snippet makes clear it's "
                    + "about key destruction, not quantum threats). Prefer "
                    + "a lower-ranked hit if its snippet is a better topic "
                    + "match for the user's question.",
                inputSchemaJSON: Self.searchSchema
            ),
            MCPTool(
                name: "get_article",
                description: "Fetch a single ZIM entry by path, as plain text. "
                    + "For long Wikipedia articles (most of them) prefer "
                    + "`list_article_sections` to see the outline, then pull "
                    + "only the relevant parts with `get_article_section` — "
                    + "that keeps the prompt small and the reply grounded.",
                inputSchemaJSON: Self.articleSchema
            ),
            MCPTool(
                name: "list_article_sections",
                description: "Return the ordered section outline for an article "
                    + "(lead + each `<h2>` / `<h3>` heading). Use this to decide "
                    + "which sections to fetch for the user's question instead "
                    + "of loading the full body.",
                inputSchemaJSON: Self.articleSchema
            ),
            MCPTool(
                name: "get_article_section",
                description: "Fetch just one section of an article by name — "
                    + "use after `list_article_sections` so you're not dragging "
                    + "in references, navigation, or unrelated sections. Pass "
                    + "the section title exactly as `list_article_sections` "
                    + "returned it (case-insensitive, prefix-tolerant). "
                    + "Pass `section: \"lead\"` (or omit) to get just the "
                    + "introduction.",
                inputSchemaJSON: Self.articleSectionSchema
            ),
            MCPTool(
                name: "get_main_page",
                description: "Fetch the main/home page of one or every loaded ZIM.",
                inputSchemaJSON: Self.mainPageSchema
            ),
            MCPTool(
                name: "zim_info",
                description:
                    "Return the `streetzim-meta.json` descriptor for loaded streetzim ZIMs. "
                    + "Use this to check which region a streetzim covers (via `bbox`), "
                    + "whether it has routing / satellite / Wikipedia cross-refs, and the "
                    + "feature counts. Skipped silently for ZIMs built before the meta "
                    + "shipped (generator commit a485ce3).",
                inputSchemaJSON: Self.zimInfoSchema
            ),
        ]
        if hasStreetzim {
            // Name-based streetzim tools are safe for every surface.
            tools.append(contentsOf: [
                MCPTool(
                    name: "near_named_place",
                    description:
                        Self.nearPlacesProtocolBlurb
                        + " Takes a free-text place name and handles the geocode "
                        + "step internally — use it for \"what's around <place>\" "
                        + "style questions.",
                    inputSchemaJSON: Self.nearNamedPlaceSchema(vocabulary: categoryVocabulary)
                ),
                MCPTool(
                    name: "route_from_places",
                    description:
                        "Plan a driving route between two free-text place names. Convenience "
                        + "wrapper over geocode + plan_driving_route.",
                    inputSchemaJSON: Self.routeFromPlacesSchema
                ),
            ])
            // Raw-coordinate tools: powerful for programmatic callers but
            // a footgun for LLMs (model passes text where coords go, or
            // defaults to 0/0). Hidden on the conversational surface.
            if surface == .full {
                tools.append(contentsOf: [
                    MCPTool(
                        name: "plan_driving_route",
                        description:
                            "Compute a driving route between two lat/lon points using streetzim's "
                            + "routing graph. Returns distance, duration, a polyline, and a list of "
                            + "named road segments ready for turn-by-turn display.",
                        inputSchemaJSON: Self.routeSchema
                    ),
                    MCPTool(
                        name: "geocode",
                        description: "Resolve a place or address string to coordinates.",
                        inputSchemaJSON: Self.geocodeSchema
                    ),
                    MCPTool(
                        name: "near_places",
                        description:
                            Self.nearPlacesProtocolBlurb
                            + " Takes numeric lat/lon; for a named place use "
                            + "`near_named_place` instead.",
                        inputSchemaJSON: Self.nearPlacesSchema(vocabulary: categoryVocabulary)
                    ),
                ])
            }
        }
        return MCPToolRegistry(tools: tools)
    }

    /// Dispatch a single tool call. `args` is the JSON object decoded from the
    /// host transport. Returns a JSON-encodable dictionary.
    public func dispatch(tool: String, args: [String: Any]) async throws -> [String: Any] {
        switch tool {
        case "list_libraries":
            let inv = try await service.inventory()
            return Self.encodeInventory(inv)
        case "search":
            let query = (args["query"] as? String) ?? ""
            let requestedLimit = (args["limit"] as? Int) ?? 10
            let kindString = args["kind"] as? String
            let kind = kindString.flatMap { ZimKind(rawValue: $0) }
            // Over-fetch BM25 candidates when a reranker is wired up
            // so it has a broader pool to re-order.
            let fetchLimit = (hitReranker == nil)
                ? requestedLimit
                : max(requestedLimit * 2, 20)
            var hits = try await service.search(query: query, limit: fetchLimit, kind: kind)
            if let rerank = hitReranker {
                hits = await rerank(query, hits)
            }
            let top = Array(hits.prefix(requestedLimit))
            return ["query": query, "count": top.count, "hits": top.map(Self.encodeHit)]
        case "get_article":
            let path = (args["path"] as? String) ?? ""
            let zim = args["zim"] as? String
            let art = try await service.article(path: path, zim: zim)
            return Self.encodeArticle(art)
        case "list_article_sections":
            let path = (args["path"] as? String) ?? ""
            let zim = args["zim"] as? String
            let parsed = try await service.articleSections(path: path, zim: zim)
            return [
                "zim": parsed.zim,
                "path": path,
                "title": parsed.title,
                "count": parsed.sections.count,
                "sections": parsed.sections.map { s -> [String: Any] in
                    [
                        "title": s.title.isEmpty ? "lead" : s.title,
                        "level": s.level,
                        "bytes": s.bytes,
                    ]
                },
            ]
        case "get_article_section":
            let path = (args["path"] as? String) ?? ""
            let section = (args["section"] as? String) ?? ""
            let zim = args["zim"] as? String
            let hit = try await service.articleSection(path: path, section: section, zim: zim)
            return [
                "zim": hit.zim,
                "path": path,
                "title": hit.title,
                "section": hit.section.title.isEmpty ? "lead" : hit.section.title,
                "level": hit.section.level,
                "bytes": hit.section.bytes,
                "text": hit.section.text,
            ]
        case "get_main_page":
            let zim = args["zim"] as? String
            let pages = try await service.mainPage(zim: zim)
            return ["pages": pages.map(Self.encodeArticle)]
        case "zim_info":
            let zim = args["zim"] as? String
            let rows = try await service.zimInfo(zim: zim)
            return ["count": rows.count, "streetzims": rows]
        case "plan_driving_route":
            let req = RouteRequest(
                originLat: (args["origin_lat"] as? Double) ?? 0,
                originLon: (args["origin_lon"] as? Double) ?? 0,
                destLat: (args["dest_lat"] as? Double) ?? 0,
                destLon: (args["dest_lon"] as? Double) ?? 0,
                zim: args["zim"] as? String
            )
            let route = try await service.planDrivingRoute(req)
            return Self.encodeRoute(route)
        case "geocode":
            let places = try await service.geocode(
                query: (args["query"] as? String) ?? "",
                limit: (args["limit"] as? Int) ?? 5,
                zim: args["zim"] as? String,
                kinds: args["kinds"] as? [String]
            )
            return ["count": places.count, "results": places.map(Self.encodePlace)]
        case "near_named_place":
            let place = (args["place"] as? String) ?? ""
            let radius = (args["radius_km"] as? Double) ?? 1.0
            let limit = (args["limit"] as? Int) ?? 25
            let combined = try await service.nearNamedPlace(
                place: place,
                radiusKm: radius,
                limit: limit,
                kinds: args["kinds"] as? [String],
                zim: args["zim"] as? String
            )
            return Self.encodeNearPlaces(
                query: place,
                resolved: combined.resolved,
                radius: radius,
                result: combined.result
            )
        case "near_places":
            // Validate coords up front. Without this, a model that passes
            // a place name (forgetting this tool needs lat/lon) sends us
            // `(0, 0)` by default — which silently scans a streetzim for
            // "nothing near the Gulf of Guinea". Returning a tool error
            // tells the model what to call instead.
            guard let lat = args["lat"] as? Double,
                  let lon = args["lon"] as? Double,
                  !(lat == 0 && lon == 0)
            else {
                return [
                    "error": "near_places requires numeric `lat` and `lon` "
                        + "arguments. For a free-text place name, call "
                        + "`near_named_place` with `place`, which geocodes "
                        + "internally. (Received args: "
                        + (args.keys.sorted().joined(separator: ", ")) + ")",
                ]
            }
            let radius = (args["radius_km"] as? Double) ?? 1.0
            let limit = (args["limit"] as? Int) ?? 25
            let result = try await service.nearPlaces(
                lat: lat, lon: lon, radiusKm: radius,
                limit: limit, kinds: args["kinds"] as? [String],
                zim: args["zim"] as? String
            )
            return Self.encodeNearPlaces(
                origin: ["lat": lat, "lon": lon],
                radius: radius,
                result: result
            )
        case "route_from_places":
            let result = try await service.routeFromPlaces(
                origin: (args["origin"] as? String) ?? "",
                destination: (args["destination"] as? String) ?? "",
                zim: args["zim"] as? String
            )
            var body = Self.encodeRoute(result.route)
            body["origin_resolved"] = Self.encodePlace(result.resolved.origin)
            body["destination_resolved"] = Self.encodePlace(result.resolved.destination)
            if let zim = result.zimUsed { body["zim"] = zim }
            return body
        default:
            throw ZimServiceError.notFound(tool)
        }
    }

    // MARK: - Encoders

    private static func encodeInventory(_ inv: InventoryResult) -> [String: Any] {
        let zims = inv.zims.map { z -> [String: Any] in
            [
                "name": z.name,
                "kind": z.kind.rawValue,
                "title": z.metadata.title,
                "description": z.metadata.description,
                "language": z.metadata.language,
                "creator": z.metadata.creator,
                "publisher": z.metadata.publisher,
                "date": z.metadata.date,
                "tags": z.metadata.tags,
                "article_count": z.metadata.articleCount,
                "has_routing": z.hasRoutingData,
            ]
        }
        return [
            "zims": zims,
            "count": inv.zims.count,
            "capabilities": inv.capabilities,
        ]
    }

    private static func encodeArticle(_ a: ArticleResult) -> [String: Any] {
        [
            "zim": a.zim, "path": a.path, "title": a.title,
            "mimetype": a.mimetype, "text": a.text, "bytes": a.bytes,
        ]
    }

    private static func encodeHit(_ h: SearchHitResult) -> [String: Any] {
        [
            "zim": h.zim, "kind": h.kind.rawValue, "path": h.path,
            "title": h.title, "snippet": h.snippet,
        ]
    }

    /// Shared encoder for `near_places` / `near_named_place` results.
    /// Emits the headline counts + category breakdown *first* so a small
    /// model can answer "20 cafes, 10 bars …" without having to scan the
    /// full `results` list.
    private static func encodeNearPlaces(
        query: String? = nil,
        origin: [String: Double]? = nil,
        resolved: Place? = nil,
        radius: Double,
        result: NearPlacesResult
    ) -> [String: Any] {
        // Breakdown sorted descending so the model sees the biggest
        // buckets first — encourages "N cafes, M bars, …" phrasing.
        let sortedBreakdown = result.breakdown
            .sorted { a, b in
                if a.value != b.value { return a.value > b.value }
                return a.key < b.key
            }
            .map { (kv) -> [String: Any] in
                ["category": kv.key, "count": kv.value]
            }
        // Detect buckets that have counts > 0 in by_category but no
        // member present in `results` (because `results` is top-N nearest
        // across the entire radius). The model routinely reports "3
        // neighbourhoods" and then names only the one that happened to
        // land in top-N; this field tells it exactly which buckets it
        // must re-query to enumerate.
        let resultKinds = Set(result.results.flatMap { r -> [String] in
            var keys: [String] = []
            if !r.place.subtype.isEmpty { keys.append(r.place.subtype.lowercased()) }
            if !r.place.kind.isEmpty { keys.append(r.place.kind.lowercased()) }
            return keys
        })
        let underrepresented = result.breakdown
            .filter { !resultKinds.contains($0.key.lowercased()) }
            .map(\.key).sorted()
        var out: [String: Any] = [
            "radius_km": radius,
            // Grand total across every category that fell in the radius.
            // `by_category` sums to this.
            "total_in_radius": result.totalInRadius,
            // Each row: {"category": "<osm_type>", "count": N}. Note that
            // "poi" is a generic OSM catchall (anything tagged
            // amenity/tourism/… that didn't fit a finer bucket) — treat it
            // as one bucket among many, NOT as "points of interest" in the
            // user-facing sense.
            "by_category": sortedBreakdown,
            // How many entries are in `results` below (capped at `limit`).
            // NOT the same as `total_in_radius`.
            "results_shown": result.results.count,
            "results": result.results.map { pair -> [String: Any] in
                var r = encodePlace(pair.place)
                r["distance_m"] = Int(pair.distanceMeters.rounded())
                return r
            },
            // Embed the usage contract in every response so the model sees
            // it on every turn, not just at tool-list setup time.
            "usage_note":
                "`results` holds the top-N nearest places across ALL categories. "
                + "A bucket in `by_category` may have members NOT present in "
                + "`results`. To list items in a specific bucket, call this "
                + "tool again with `kinds=[\"<category>\"]`. NEVER invent "
                + "names for buckets whose members are not in `results`.",
        ]
        if !underrepresented.isEmpty {
            out["categories_needing_drill_in"] = underrepresented
        }
        if let query { out["query"] = query }
        if let origin { out["origin"] = origin }
        if let resolved { out["resolved"] = encodePlace(resolved) }
        return out
    }

    private static func encodePlace(_ p: Place) -> [String: Any] {
        var out: [String: Any] = [
            "name": p.name, "type": p.kind, "lat": p.lat, "lon": p.lon,
        ]
        if !p.subtype.isEmpty { out["subtype"] = p.subtype }
        // `location` means two different things in streetzim records:
        //   • `addr` kind → a real street address; always useful.
        //   • everything else → an admin-boundary label assigned by the
        //     generator. We've seen DC landmarks mislabelled as "Silver
        //     Spring, Maryland" etc., so we suppress it here to keep the
        //     model from confidently repeating bad data. Once the
        //     streetzim generator fixes its admin lookup this can come
        //     back unconditionally.
        if !p.location.isEmpty, p.kind.lowercased() == "addr" {
            out["location"] = p.location
        }
        // Forward streetzim cross-refs so the model can chain to the
        // Wikipedia ZIM via `get_article(path: "A/" + w.split(":",1)[1])`.
        if let w = p.wiki, !w.isEmpty { out["wikipedia"] = w }
        if let q = p.wikidata, !q.isEmpty { out["wikidata"] = q }
        return out
    }

    private static func encodeRoute(_ r: Route) -> [String: Any] {
        [
            "origin": ["lat": r.origin.lat, "lon": r.origin.lon],
            "destination": ["lat": r.destination.lat, "lon": r.destination.lon],
            "origin_node": r.originNode,
            "destination_node": r.destinationNode,
            "distance_m": r.distanceMeters,
            "distance_km": r.distanceKilometers,
            "duration_s": r.durationSeconds,
            "duration_min": r.durationMinutes,
            "roads": r.roads.map { [
                "name": $0.name.isEmpty ? "(unnamed road)" : $0.name,
                "distance_m": $0.distanceMeters,
                "duration_s": $0.durationSeconds,
            ] },
            "polyline": r.polyline.map { [$0.lat, $0.lon] },
            "turn_by_turn": r.turnByTurn,
        ]
    }

    // MARK: - Schemas (stored as baked JSON so the tool list has zero cost)

    private static let emptyObjectSchema: Data =
        #"{"type":"object","properties":{}}"#.data(using: .utf8)!

    private static let searchSchema: Data = #"""
    {"type":"object","required":["query"],"properties":{
        "query":{"type":"string"},
        "limit":{"type":"integer","default":10,"minimum":1,"maximum":50},
        "kind":{"type":"string","enum":["wikipedia","mdwiki","streetzim","generic"]}
    }}
    """#.data(using: .utf8)!

    private static let articleSchema: Data = #"""
    {"type":"object","required":["path"],"properties":{
        "path":{"type":"string","description":"ZIM entry path (e.g. A/Aspirin)"},
        "zim":{"type":"string","description":"Optional: restrict to this ZIM filename."}
    }}
    """#.data(using: .utf8)!

    private static let articleSectionSchema: Data = #"""
    {"type":"object","required":["path","section"],"properties":{
        "path":{"type":"string","description":"ZIM entry path (e.g. A/Aspirin)."},
        "section":{"type":"string","description":"Section title from `list_article_sections`. Use 'lead' (or omit-equivalent) for the intro paragraph."},
        "zim":{"type":"string","description":"Optional: restrict to this ZIM filename."}
    }}
    """#.data(using: .utf8)!

    private static let zimInfoSchema: Data = #"""
    {"type":"object","properties":{
        "zim":{"type":"string","description":"Optional: restrict to this streetzim filename. Omit for all."}
    }}
    """#.data(using: .utf8)!

    private static let mainPageSchema: Data = #"""
    {"type":"object","properties":{
        "zim":{"type":"string"}
    }}
    """#.data(using: .utf8)!

    private static let routeSchema: Data = #"""
    {"type":"object","required":["origin_lat","origin_lon","dest_lat","dest_lon"],"properties":{
        "origin_lat":{"type":"number"},
        "origin_lon":{"type":"number"},
        "dest_lat":{"type":"number"},
        "dest_lon":{"type":"number"},
        "zim":{"type":"string"}
    }}
    """#.data(using: .utf8)!

    private static let geocodeSchema: Data = #"""
    {"type":"object","required":["query"],"properties":{
        "query":{"type":"string"},
        "limit":{"type":"integer","default":5},
        "zim":{"type":"string"},
        "kinds":{"type":"array","items":{"type":"string"},"description":"Optional type filter: addr, place, poi, ..."}
    }}
    """#.data(using: .utf8)!

    /// Shared description text for `near_places` / `near_named_place`.
    /// Documents the two-step protocol (summary call, then optional drill
    /// into a single bucket) so every MCP client sees the same contract
    /// regardless of which tool they reach for.
    private static let nearPlacesProtocolBlurb: String =
        "Find points of interest near a location. Returns a ranked `results` list "
        + "(top-N nearest), a `by_category` summary of every bucket within the radius, "
        + "a grand `total_in_radius`, and `results_shown` (= results.count). "
        + "PROTOCOL: call first with NO `kinds` filter to get the full breakdown; "
        + "to list items in one bucket, call again with `kinds=[\"<name>\"]` "
        + "using a value that appeared as `by_category[i].category`. "
        + "Do not pass `kinds=[]`."

    /// Build the JSON schema for `near_named_place`. Injects the live
    /// OSM category vocabulary from the loaded streetzim(s) into the
    /// `kinds` parameter so clients see the exact allowed slugs.
    private static func nearNamedPlaceSchema(vocabulary: [String]) -> Data {
        schemaJSON(
            required: ["place"],
            properties: [
                ("place", ["type": "string", "description": "Free-text place name."]),
                ("radius_km", ["type": "number", "default": 1.0,
                               "description": "Search radius. 0.5 ≈ walking distance, 2–5 ≈ neighborhood."]),
                ("limit", ["type": "integer", "default": 10]),
                ("kinds", kindsSchema(vocabulary: vocabulary)),
                ("zim", ["type": "string",
                         "description": "Specific streetzim filename, else try them all."]),
            ]
        )
    }

    private static func nearPlacesSchema(vocabulary: [String]) -> Data {
        schemaJSON(
            required: ["lat", "lon"],
            properties: [
                ("lat", ["type": "number", "description": "Center latitude (get from geocode)."]),
                ("lon", ["type": "number", "description": "Center longitude (get from geocode)."]),
                ("radius_km", ["type": "number", "default": 1.0,
                               "description": "Search radius in km. 0.5 ≈ walking distance, 2–5 ≈ neighborhood."]),
                ("limit", ["type": "integer", "default": 10,
                           "description": "Max results, sorted nearest first."]),
                ("kinds", kindsSchema(vocabulary: vocabulary)),
                ("zim", ["type": "string",
                         "description": "Specific streetzim filename, else try them all."]),
            ]
        )
    }

    private static func kindsSchema(vocabulary: [String]) -> [String: Any] {
        var items: [String: Any] = ["type": "string"]
        var desc = "OSM category name(s) to filter by. Omit the whole "
            + "`kinds` parameter for the full by_category breakdown; never "
            + "pass an empty array. When drilling in, use a name exactly as "
            + "it appeared in a prior `by_category[i].category`."
        if !vocabulary.isEmpty {
            items["enum"] = vocabulary
            desc += " Known top-level OSM categories in the loaded "
                + "streetzim(s): " + vocabulary.joined(separator: ", ")
                + ". Finer-grained subtypes from a prior response's "
                + "`by_category` are also valid (e.g. cafe, bar, restaurant)."
        }
        return [
            "type": "array",
            "items": items,
            "description": desc,
        ]
    }

    private static func schemaJSON(required: [String], properties: [(String, [String: Any])]) -> Data {
        var props: [String: Any] = [:]
        for (k, v) in properties { props[k] = v }
        let root: [String: Any] = [
            "type": "object",
            "required": required,
            "properties": props,
        ]
        return (try? JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])) ?? Data()
    }

    private static let routeFromPlacesSchema: Data = #"""
    {"type":"object","required":["origin","destination"],"properties":{
        "origin":{"type":"string"},
        "destination":{"type":"string"},
        "zim":{"type":"string"}
    }}
    """#.data(using: .utf8)!
}
