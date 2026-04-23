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
import os

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

/// Snapshot of the host's "current location" at tool-dispatch time. Supplied
/// by the host via `HostStateProvider` so tools like `what_is_here` don't
/// need iOS frameworks in the pure-Swift MCPZimKit layer.
public struct LocationSnapshot: Sendable {
    public let lat: Double
    public let lon: Double
    public init(lat: Double, lon: Double) {
        self.lat = lat; self.lon = lon
    }
}

/// Snapshot of an active driving route. Mirrors the minimum `ActiveRoute`
/// shape the host tracks — enough to answer "how much longer?" without
/// re-running the routing graph. Host converts its own state into this
/// shape inside `HostStateProvider`.
public struct RouteSnapshot: Sendable {
    public struct Coordinate: Sendable {
        public let lat: Double
        public let lon: Double
        public init(lat: Double, lon: Double) {
            self.lat = lat; self.lon = lon
        }
    }
    public let origin: Coordinate
    public let destination: Coordinate
    public let originName: String
    public let destinationName: String
    public let totalDistanceMeters: Double
    public let totalDurationSeconds: Double
    /// Polyline vertices; parallel to `cumulativeDistanceMeters`.
    public let polyline: [Coordinate]
    /// Cumulative path distance up to each polyline index.
    /// `.last == totalDistanceMeters`.
    public let cumulativeDistanceMeters: [Double]
    public let turnByTurn: [String]

    public init(
        origin: Coordinate, destination: Coordinate,
        originName: String, destinationName: String,
        totalDistanceMeters: Double, totalDurationSeconds: Double,
        polyline: [Coordinate],
        cumulativeDistanceMeters: [Double],
        turnByTurn: [String]
    ) {
        self.origin = origin
        self.destination = destination
        self.originName = originName
        self.destinationName = destinationName
        self.totalDistanceMeters = totalDistanceMeters
        self.totalDurationSeconds = totalDurationSeconds
        self.polyline = polyline
        self.cumulativeDistanceMeters = cumulativeDistanceMeters
        self.turnByTurn = turnByTurn
    }
}

extension RouteSnapshot {
    /// Snap `current` to the nearest polyline vertex, then derive how much
    /// of the route is left. Same math the iOS host's `RouteProgress` runs;
    /// duplicated here so MCPZimKit doesn't depend on the iOS-side type.
    public func remaining(at current: Coordinate)
        -> (remainingMeters: Double, remainingSeconds: Double, fractionDone: Double)
    {
        guard polyline.count >= 2 else {
            return (totalDistanceMeters, totalDurationSeconds, 0)
        }
        var bestIdx = 0
        var bestD = Double.infinity
        for (i, p) in polyline.enumerated() {
            let d = Self.haversineMetersApprox(current.lat, current.lon, p.lat, p.lon)
            if d < bestD { bestD = d; bestIdx = i }
        }
        let covered = bestIdx < cumulativeDistanceMeters.count
            ? cumulativeDistanceMeters[bestIdx] : 0
        let remainingMeters = max(0, totalDistanceMeters - covered)
        let fraction = totalDistanceMeters > 0
            ? 1.0 - (remainingMeters / totalDistanceMeters) : 0
        let remainingSeconds = totalDurationSeconds * (1.0 - fraction)
        return (remainingMeters, remainingSeconds, fraction)
    }

    static func haversineMetersApprox(
        _ lat1: Double, _ lon1: Double, _ lat2: Double, _ lon2: Double
    ) -> Double {
        let R = 6_371_000.0
        let rlat1 = lat1 * .pi / 180
        let rlat2 = lat2 * .pi / 180
        let dlat = (lat2 - lat1) * .pi / 180
        let dlon = (lon2 - lon1) * .pi / 180
        let a = sin(dlat / 2) * sin(dlat / 2)
            + cos(rlat1) * cos(rlat2) * sin(dlon / 2) * sin(dlon / 2)
        return 2 * R * asin(min(1.0, sqrt(a)))
    }
}

/// Snapshot the host feeds in at dispatch time. Either field may be nil —
/// tools that need a missing piece (e.g. `route_status` with no active
/// route) return a clear error instead of failing.
public struct HostStateSnapshot: Sendable {
    public let activeRoute: RouteSnapshot?
    public let currentLocation: LocationSnapshot?
    public init(activeRoute: RouteSnapshot?, currentLocation: LocationSnapshot?) {
        self.activeRoute = activeRoute
        self.currentLocation = currentLocation
    }
}

/// Host-supplied closure for pulling "current world state" into a tool
/// call. When nil, `route_status` / `what_is_here` are still registered
/// but reject with a clear error so the model learns to stop calling
/// them. Mark sendable since actors dispatch it on arbitrary executors.
public typealias HostStateProvider = @Sendable () async -> HostStateSnapshot

/// Optional post-processor for `search` tool hits. Takes the user's
/// query text + the BM25-ranked hit list and returns a possibly-
/// reordered (and possibly-shortened) list. Hosted outside MCPZimKit
/// so downstream callers can plug in platform-specific rerankers
/// (e.g. `NLContextualEmbedding` on Apple platforms) without pulling
/// those dependencies into this package.
public typealias HitReranker = @Sendable (_ query: String, _ hits: [SearchHitResult]) async -> [SearchHitResult]

public actor MCPToolAdapter {
    /// Routed to `os.Logger` so diagnostics surface via
    /// `idevicesyslog` (plain `print` goes to stderr and is dropped
    /// by the device side). Host streamers see
    /// `Tool.MCPToolAdapter.…` when filtering.
    static let toolLog = Logger(
        subsystem: "org.mcpzim.MCPZimChat",
        category: "MCPToolAdapter"
    )

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

    /// Optional closure the host installs so location-aware tools
    /// (`route_status`, `what_is_here`) can read the latest active
    /// route + GPS fix at dispatch time. Nil means the tools still
    /// register (so model behaviour doesn't branch on host identity)
    /// but reject with a clear error.
    private var hostStateProvider: HostStateProvider?

    public init(
        service: any ZimService,
        hasStreetzim: Bool,
        surface: ToolSurface = .full,
        categoryVocabulary: [String] = [],
        hostStateProvider: HostStateProvider? = nil
    ) {
        self.service = service
        self.hasStreetzim = hasStreetzim
        self.surface = surface
        self.categoryVocabulary = categoryVocabulary
        self.hostStateProvider = hostStateProvider
    }

    /// Install (or clear) a semantic reranker. Passing `nil` drops
    /// back to plain BM25 order.
    public func installHitReranker(_ reranker: HitReranker?) {
        self.hitReranker = reranker
    }

    /// Install (or clear) the host-state provider. Called post-construction
    /// because the host usually builds the adapter before its route /
    /// location state actors are wired up.
    public func installHostStateProvider(_ provider: HostStateProvider?) {
        self.hostStateProvider = provider
    }

    public static func from(
        service: any ZimService,
        surface: ToolSurface = .full,
        hostStateProvider: HostStateProvider? = nil
    ) async -> MCPToolAdapter {
        let inventory = (try? await service.inventory()) ?? InventoryResult(zims: [], capabilities: [])
        let hasStreetzim = inventory.zims.contains { $0.kind == .streetzim && $0.hasRoutingData }
        let vocab: [String]
        if let concrete = service as? DefaultZimService {
            vocab = await concrete.categoryVocabulary()
        } else {
            vocab = []
        }
        return MCPToolAdapter(
            service: service,
            hasStreetzim: hasStreetzim,
            surface: surface,
            categoryVocabulary: vocab,
            hostStateProvider: hostStateProvider
        )
    }

    public var registry: MCPToolRegistry {
        // Phase-B tool cull (2026-04-23): the model's declared menu is now
        // 10 tools (down from 21). Pruned tools kept their dispatch
        // handlers, so existing transcripts / retained conversations that
        // still reference them continue to work — they're just no longer
        // declared, so the model doesn't see them in the system turn.
        // Memory win: ~1-2 k fewer tokens in the preamble; peak RSS drops
        // ~500 MB on our 3-turn eval scenarios. See ON_DEVICE_MODEL_REPORT
        // and EXTENDED_CONTEXT_EVAL.md for the Phase A data that drove this.
        //
        // Dropped (dispatch-only from here):
        //   list_libraries, get_article, list_article_sections,
        //   get_article_by_title, get_main_page, zim_info,
        //   near_named_place, nearby_stories_at_place,
        //   plan_driving_route, show_map, geocode
        //
        // Merged behaviour:
        //   - near_places now accepts optional `place` (free-text). When
        //     set, the adapter geocodes internally — the model doesn't need
        //     a separate `near_named_place` tool.
        //   - nearby_stories same treatment for the Wikipedia-linked variant.
        var tools: [MCPTool] = [
            MCPTool(
                name: "search",
                description:
                    "Keyword full-text search across every loaded ZIM — use "
                    + "ONLY when you don't already know the article's title. "
                    + "If the user named the entity (\"tell me about Palo "
                    + "Alto\", \"read the HP Garage article\") call "
                    + "`article_overview` / `narrate_article` DIRECTLY with "
                    + "`title` set to the entity — don't waste a round-trip "
                    + "on search first. Pass SHORT KEYWORDS (article titles, "
                    + "proper nouns), not full sentences. "
                    + "Good: `query=\"Aspirin\"`, `query=\"Marie Curie\"`. "
                    + "Bad: `query=\"what is aspirin used for\"`. "
                    + "For LOCATION questions (\"what's in X\", \"near X\", "
                    + "\"around X\") prefer `near_places` with a `place` arg. "
                    + "Returns paths, titles, AND a ~200-char snippet from "
                    + "each hit's lead paragraph. READ THE SNIPPETS before "
                    + "picking — the top BM25 match isn't always the "
                    + "topically-closest one.",
                inputSchemaJSON: Self.searchSchema
            ),
            MCPTool(
                name: "get_article_section",
                description: "Fetch an article's content. Pass `section` to get "
                    + "just one named section, or omit / pass `section: \"lead\"` "
                    + "to get the introduction. Use after deciding from a "
                    + "search hit or `article_overview` response which "
                    + "section you want. Pass `title` (e.g. \"Palo Alto\") "
                    + "OR `path` (e.g. \"A/Palo_Alto\") — either works.",
                inputSchemaJSON: Self.articleSectionSchema
            ),
            MCPTool(
                name: "article_overview",
                description:
                    "THE tool for SINGLE-ENTITY encyclopedic queries — "
                    + "one subject, one article: \"tell me about X\", "
                    + "\"what is X\", \"give me an overview of X\", \"who "
                    + "is X\". Pass `title` DIRECTLY with the entity name — "
                    + "DO NOT `search` first or use `get_article_by_title`. "
                    + "Returns the lead plus a few major narrative sections "
                    + "(skips boilerplate References / See also / External "
                    + "links, prefers History / Overview / Background / "
                    + "Description / Geography / Culture / Economy). "
                    + "**NEVER call this for questions mentioning TWO or "
                    + "more entities** (\"A vs B\", \"how have A and B\", "
                    + "\"relations between X and Y\", \"compare X with Y\", "
                    + "\"difference between X and Y\") — those ALWAYS go to "
                    + "`compare_articles` in a single call with "
                    + "`titles: [A, B]`. NOT for \"where am I\" (use "
                    + "`what_is_here`). NOT for \"read me the article "
                    + "aloud\" (use `narrate_article`).",
                inputSchemaJSON: Self.articleOverviewSchema
            ),
            MCPTool(
                name: "compare_articles",
                description:
                    "THE tool for ANY two-entity question: comparisons "
                    + "(\"how is X different from Y\", \"compare X and Y\"), "
                    + "relations (\"how have A and B gotten along\", "
                    + "\"relations between A and B\"), diffs, contrasts. "
                    + "Pass `titles: [A, B]` (2–4 entities). Internally "
                    + "probes for a dedicated Wikipedia relations article "
                    + "(\"A–B relations\", \"Foreign relations of A\") "
                    + "when exactly 2 titles are given — for e.g. "
                    + "countries / organizations that have one — and "
                    + "returns that. Otherwise pulls each entity's lead + "
                    + "top narrative sections in a side-by-side payload. "
                    + "Pass `section` to align on a named topic "
                    + "(e.g. `section=\"History\"`). Use this instead of "
                    + "calling `article_overview` twice.",
                inputSchemaJSON: Self.compareArticlesSchema
            ),
            MCPTool(
                name: "narrate_article",
                description:
                    "THE tool for \"read me the article\" / \"read aloud\" / "
                    + "\"recite the article about X\" / \"read the full piece\". "
                    + "Pass `title` directly. Returns the full article body "
                    + "cleaned for narration (headings announced as sentences, "
                    + "citation markers stripped, boilerplate sections dropped). "
                    + "The host streams this text directly to TTS — NO "
                    + "follow-up model generation. Use this, NOT "
                    + "`get_article_by_title`, when the user wants to HEAR "
                    + "the article. Supports any loaded Wikipedia-family ZIM.",
                inputSchemaJSON: Self.narrateArticleSchema
            ),
        ]
        if hasStreetzim {
            tools.append(contentsOf: [
                MCPTool(
                    name: "route_from_places",
                    description:
                        "Plan a driving route between two free-text place names "
                        + "(or \"my location\" as the origin). Returns distance, "
                        + "duration, polyline, and turn-by-turn instructions. "
                        + "The host renders the map automatically — no need for "
                        + "a separate \"show the map\" call.",
                    inputSchemaJSON: Self.routeFromPlacesSchema
                ),
                MCPTool(
                    name: "near_places",
                    description:
                        Self.nearPlacesProtocolBlurb
                        + " Pass `place` (free-text) for \"what's around "
                        + "<place>\" — the tool geocodes internally. Omit "
                        + "`place` and pass explicit `lat`/`lon` (or just "
                        + "rely on the user's current coordinates from the "
                        + "system preamble) for \"what's around me\" / "
                        + "\"nearest ___\" queries.",
                    inputSchemaJSON: Self.nearPlacesSchema(vocabulary: categoryVocabulary)
                ),
                // Composite "tell me something interesting" tool — wraps
                // near_places(has_wiki=true) + parallel lead-paragraph
                // fetches so the model gets story-ready excerpts on one
                // call instead of stitching together N article fetches.
                MCPTool(
                    name: "nearby_stories",
                    description:
                        "Return 3–5 story-ready excerpts (~1–2 paragraphs "
                        + "each) from Wikipedia articles for places near a "
                        + "location. The right tool for \"tell me something "
                        + "interesting about where I am\" / \"stories about "
                        + "this neighborhood\" / \"history of downtown "
                        + "Portland\". Pass `place` (free-text, e.g. "
                        + "\"Palo Alto\") for a named location; omit "
                        + "`place` to anchor on the user's current GPS. "
                        + "Optionally filter with `kinds` (e.g. "
                        + "`kinds=[\"museum\"]` for \"interesting museums\"). "
                        + "Each excerpt is substantive enough to narrate "
                        + "directly; read them out, don't summarize.",
                    inputSchemaJSON: Self.nearbyStoriesSchema(vocabulary: categoryVocabulary)
                ),
                MCPTool(
                    name: "what_is_here",
                    description:
                        "Answer ONLY these exact questions: \"where am I?\" / "
                        + "\"what neighborhood is this?\" / \"what city am "
                        + "I in?\". Reverse-geocodes the USER'S CURRENT GPS "
                        + "to the nearest named place (admin / neighborhood "
                        + "/ city) and pulls its Wikipedia lead if there "
                        + "is one. DO NOT use for \"tell me about <named "
                        + "place>\" — that's `article_overview`.",
                    inputSchemaJSON: Self.whatIsHereSchema
                ),
                MCPTool(
                    name: "route_status",
                    description:
                        "Check progress on the currently-active driving route. "
                        + "Use for \"how much longer?\" / \"what's my next "
                        + "turn?\" / \"am I there yet?\". Takes no arguments "
                        + "— reads the last planned route (via "
                        + "`route_from_places`) and the host's live GPS "
                        + "fix, returns remaining distance, ETA, progress "
                        + "%, and the next leg. Errors if no route is "
                        + "active.",
                    inputSchemaJSON: Self.emptyObjectSchema
                ),
            ])
        }
        return MCPToolRegistry(tools: tools)
    }

    /// Dispatch a single tool call. `args` is the JSON object decoded from the
    /// host transport. Returns a JSON-encodable dictionary.
    public func dispatch(tool: String, args: [String: Any]) async throws -> [String: Any] {
        let args = try await sanitizeOptionalZimArg(args)
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
            // Auto-fallback when a `kind` filter returns nothing (or
            // only a trickle) — small models routinely set
            // `kind="mdwiki"` on unrelated queries because the system
            // preamble mentioned mdwiki for medical questions. Don't
            // punish the user for that guess.
            if kind != nil, hits.count < 3 {
                let wider = try await service.search(query: query, limit: fetchLimit, kind: nil)
                var merged = hits
                var seen = Set(hits.map { "\($0.zim)\t\($0.path)" })
                for h in wider {
                    let key = "\(h.zim)\t\(h.path)"
                    if seen.contains(key) { continue }
                    seen.insert(key)
                    merged.append(h)
                }
                hits = merged
            }
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
        case "get_article_by_title":
            let title = (args["title"] as? String) ?? ""
            let sectionArg = args["section"] as? String
            let zim = args["zim"] as? String
            let hit = try await service.articleByTitle(
                title: title, zim: zim, section: sectionArg
            )
            return [
                "zim": hit.zim,
                "path": hit.path,
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
        case "show_map":
            let place = (args["place"] as? String) ?? ""
            let zim = args["zim"] as? String
            // Prefer place-kinds (city/town/village/POI) over addr
            // (which tends to match a bare "Burbank" to a Silicon
            // Valley street called "Burbank Drive"). Fall back to an
            // all-kinds search if the first pass comes up empty.
            var hits = try await service.geocode(
                query: place, limit: 1, zim: zim,
                kinds: ["place", "poi"]
            )
            if hits.isEmpty {
                hits = try await service.geocode(
                    query: place, limit: 1, zim: zim, kinds: nil
                )
            }
            guard let hit = hits.first else { throw ZimServiceError.noMatch(place) }
            // Return a single-point "polyline" so the existing
            // streetzim-viewer wiring in the host app renders a map
            // centred at `(lat, lon)` with our blue dot overlay. No
            // route line, no turns — just a map of the place. The
            // host falls back to whichever streetzim is enabled if
            // `zim` is blank.
            return [
                "zim": zim ?? "",
                "place": hit.name,
                "lat": hit.lat,
                "lon": hit.lon,
                "polyline": [[hit.lat, hit.lon]],
            ] as [String: Any]
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
            let excerpts = await fetchWikiExcerpts(for: combined.result)
            return Self.encodeNearPlaces(
                query: place,
                resolved: combined.resolved,
                radius: radius,
                result: combined.result,
                excerpts: excerpts
            )
        case "near_places":
            // Phase-B merge (2026-04-23): `near_named_place` was folded
            // into `near_places`. If the model passed a free-text
            // `place`, route through the named-place dispatch; that
            // code path geocodes, runs the coord scan, and returns the
            // richer `resolved` block. Otherwise fall through to the
            // coord path as before.
            if let place = args["place"] as? String,
               !place.trimmingCharacters(in: .whitespaces).isEmpty
            {
                var forwarded = args
                forwarded["place"] = place
                return try await dispatch(tool: "near_named_place", args: forwarded)
            }
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
                    "error": "near_places needs a center — either `place` "
                        + "(free-text, geocoded internally) or numeric "
                        + "`lat`/`lon`. For \"what's around me\" queries, "
                        + "use the user's current coords from the system "
                        + "preamble. (Received args: "
                        + (args.keys.sorted().joined(separator: ", ")) + ")",
                ]
            }
            let radius = (args["radius_km"] as? Double) ?? 1.0
            let limit = (args["limit"] as? Int) ?? 25
            let result = try await service.nearPlaces(
                lat: lat, lon: lon, radiusKm: radius,
                limit: limit, kinds: args["kinds"] as? [String],
                zim: args["zim"] as? String,
                hasWiki: (args["has_wiki"] as? Bool) ?? false
            )
            let excerpts = await fetchWikiExcerpts(for: result)
            return Self.encodeNearPlaces(
                origin: ["lat": lat, "lon": lon],
                radius: radius,
                result: result,
                excerpts: excerpts
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
        case "article_overview":
            return try await dispatchArticleOverview(args: args)
        case "compare_articles":
            return try await dispatchCompareArticles(args: args)
        case "article_relationship":
            // Retired 2026-04-21: folded into `compare_articles`. Kept
            // the dispatch case so any model trained on the old schema
            // (or an old ChatSession transcript replay) still reaches
            // the right code — we just funnel the args through the
            // compare handler, which probes for a dedicated relations
            // article before falling back to side-by-side comparison.
            var forwarded = args
            if let a = args["a"] as? String, let b = args["b"] as? String {
                forwarded["titles"] = [a, b]
            }
            return try await dispatchCompareArticles(args: forwarded)
        case "narrate_article":
            return try await dispatchNarrateArticle(args: args)
        case "nearby_stories":
            // Phase-B merge: if `place` is set, route through the
            // named-place dispatch; otherwise use the GPS-centred path.
            if let place = args["place"] as? String,
               !place.trimmingCharacters(in: .whitespaces).isEmpty
            {
                return try await dispatchNearbyStoriesAtPlace(args: args)
            }
            return try await dispatchNearbyStories(args: args)
        case "nearby_stories_at_place":
            return try await dispatchNearbyStoriesAtPlace(args: args)
        case "what_is_here":
            return try await dispatchWhatIsHere(args: args)
        case "route_status":
            return await dispatchRouteStatus()
        default:
            throw ZimServiceError.notFound(tool)
        }
    }

    // MARK: - Defensive arg sanitization

    /// Strip a hallucinated `zim` filename from `args` so an unknown
    /// value falls back to "search all loaded ZIMs" instead of
    /// returning an "unknown ZIM" error to the model.
    ///
    /// Real on-device capture (Qwen 3 4B 4-bit, dropped-request.log
    /// follow-up): the model called `compare_articles` with
    /// `zim="wikipediapedia_en_all maxi 2025-10.zim"` against a
    /// real ZIM `wikipedia_en_all_maxi_2025-10.zim` — duplicated
    /// "pedia", space instead of underscore. The exact-match check
    /// downstream rejected it, the tool errored, and iter 1's prose
    /// summary mangled the entity names while explaining why
    /// nothing was found. Dropping the unknown arg lets the same
    /// tool succeed by searching every loaded ZIM.
    ///
    /// We try exact match, then case-insensitive match. Anything
    /// else gets removed — never substituted, since "close"
    /// filenames could legitimately belong to different ZIMs (en
    /// vs es, full vs nopic, …) and the wrong substitution would
    /// be worse than no pin.
    ///
    /// Cost is one `service.inventory()` call per dispatch — but
    /// only when a `zim` arg is actually present. Most calls
    /// (which omit `zim`) short-circuit before the await.
    private func sanitizeOptionalZimArg(
        _ args: [String: Any]
    ) async throws -> [String: Any] {
        guard let z = args["zim"] as? String, !z.isEmpty else {
            return args
        }
        let inv = try await service.inventory()
        return Self.sanitizeZim(args, loadedZimNames: inv.zims.map(\.name))
    }

    /// Pure logic split out so tests can exercise it without
    /// having to mock an async `inventory()` call. Same contract:
    /// returns `args` untouched when `zim` is absent or matches a
    /// loaded ZIM (exact, then case-insensitive); returns a copy
    /// with the `zim` key removed otherwise.
    static func sanitizeZim(
        _ args: [String: Any], loadedZimNames: [String]
    ) -> [String: Any] {
        guard let z = args["zim"] as? String, !z.isEmpty else {
            return args
        }
        if loadedZimNames.contains(z) { return args }
        let zLower = z.lowercased()
        if loadedZimNames.contains(where: { $0.lowercased() == zLower }) {
            return args
        }
        var copy = args
        copy.removeValue(forKey: "zim")
        return copy
    }

    // MARK: - Dispatch helpers for composite tools

    private func dispatchArticleOverview(args: [String: Any]) async throws -> [String: Any] {
        let title = (args["title"] as? String) ?? ""
        guard !title.isEmpty else {
            return ["error": "article_overview requires a non-empty `title`."]
        }
        let zim = args["zim"] as? String
        let maxSections = max(1, min(10, (args["max_sections"] as? Int) ?? 5))
        let resolved = try await ArticleHeuristics.sectionsByTitle(
            service: service, title: title, zim: zim
        )
        let picked = ArticleHeuristics.pickOverview(
            sections: resolved.sections, maxSections: maxSections
        )
        return [
            "zim": resolved.zim,
            "path": resolved.path,
            "title": resolved.title,
            "section_count": picked.count,
            "sections": picked.map { s -> [String: Any] in
                [
                    "title": s.title.isEmpty ? "lead" : s.title,
                    "level": s.level,
                    "bytes": s.bytes,
                    "text": s.text,
                ]
            },
        ]
    }

    private func dispatchCompareArticles(args: [String: Any]) async throws -> [String: Any] {
        let titles = (args["titles"] as? [String]) ?? []
        let trimmed = titles.compactMap { t -> String? in
            let s = t.trimmingCharacters(in: .whitespacesAndNewlines)
            return s.isEmpty ? nil : s
        }
        guard trimmed.count >= 2 else {
            return [
                "error": "compare_articles needs at least two non-empty `titles`. "
                    + "Received: \(titles)",
            ]
        }
        guard trimmed.count <= 4 else {
            return [
                "error": "compare_articles caps at 4 titles to keep the prompt "
                    + "budget reasonable. Received \(trimmed.count).",
            ]
        }
        let section = (args["section"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let zim = args["zim"] as? String

        // Special-case: if the caller gave us exactly two titles and
        // didn't pin a specific section, first probe for a dedicated
        // Wikipedia relations article ("A–B relations", "Foreign
        // relations of A", etc.). For country/organization pairs that
        // have one, the relations article is a much richer answer than
        // stitching two independent ledes together. Only triggers when
        // the probe actually hits — otherwise we fall straight through
        // to the original side-by-side path, so comparisons like
        // "Elon Musk vs Jeff Bezos" (no relations article exists) don't
        // pay a wasted round-trip.
        if trimmed.count == 2, (section == nil || section?.isEmpty == true) {
            if let relations = try? await probeRelationsArticle(
                a: trimmed[0], b: trimmed[1], zim: zim
            ) { return relations }
        }

        let svc = service
        let articles: [[String: Any]] = try await withThrowingTaskGroup(of: (Int, [String: Any]).self) { group in
            for (i, title) in trimmed.enumerated() {
                group.addTask { [svc] in
                    do {
                        let row = try await Self.fetchCompareEntry(
                            service: svc, title: title,
                            section: section, zim: zim
                        )
                        return (i, row)
                    } catch {
                        return (i, [
                            "title": title,
                            "error": "Could not fetch: \(error)",
                        ])
                    }
                }
            }
            var out: [(Int, [String: Any])] = []
            for try await row in group { out.append(row) }
            out.sort { $0.0 < $1.0 }
            return out.map(\.1)
        }

        return [
            "requested": trimmed,
            "section_mode": section == nil ? "overview" : "aligned",
            "aligned_section": section ?? "",
            "count": articles.count,
            "articles": articles,
        ]
    }

    /// Shared cleaner for any wiki article we're about to dress up
    /// as a place's "Wikipedia preview." Returns nil when the
    /// article is a disambiguation page or the cleaned lead has no
    /// real content — caller should then skip the wiki enrichment
    /// entirely rather than show a misleading header.
    ///
    /// Rejection cases seen on device (2026-04-22 screenshot):
    ///   * "Oak Grove" → `Oak_Grove_(disambiguation)` / "may refer
    ///     to:" → rendered as the place's preview, nonsensical.
    ///   * "Wine Cellar" → generic `Wine_cellar` concept article →
    ///     rendered as the place's preview, also nonsensical.
    ///   * OSM-tagged places like "Palo Alto Junior Museum and Zoo"
    ///     → lead opens with the title repeated three times before
    ///     the first sentence → list row shows the triple-title
    ///     instead of the actual description.
    ///
    /// `stripLeadingTitleRepetition` handles the third; the disambig
    /// check handles the first; calling code gets nil for the
    /// second if the resolved title doesn't match the place name
    /// (tightened check in `fetchWikiExcerptsByNameSearch`).
    static func cleanedWikiExcerpt(
        title: String, leadText: String
    ) -> String? {
        if ArticleHeuristics.isDisambiguationArticle(
            title: title, leadText: leadText
        ) { return nil }
        let stripped = ArticleHeuristics.stripLeadingTitleRepetition(
            ArticleHeuristics.stripCitations(leadText), title: title
        )
        let excerpt = ArticleHeuristics.trimToSentence(
            stripped,
            maxChars: ArticleHeuristics.defaultStoryExcerptChars
        )
        return excerpt.isEmpty ? nil : excerpt
    }

    private static func fetchCompareEntry(
        service: any ZimService, title: String,
        section: String?, zim: String?
    ) async throws -> [String: Any] {
        if let namedSection = section, !namedSection.isEmpty {
            let hit = try await service.articleByTitle(
                title: title, zim: zim, section: namedSection
            )
            return [
                "title": hit.title,
                "zim": hit.zim,
                "path": hit.path,
                "sections": [[
                    "title": hit.section.title.isEmpty ? "lead" : hit.section.title,
                    "text": hit.section.text,
                    "bytes": hit.section.bytes,
                ]],
            ] as [String: Any]
        }
        let resolved = try await ArticleHeuristics.sectionsByTitle(
            service: service, title: title, zim: zim
        )
        // For comparison we want each article's lead + 2 biggest narrative
        // sections — tighter than article_overview's default so the side-
        // by-side payload stays under the model's prompt budget.
        let picked = ArticleHeuristics.pickOverview(
            sections: resolved.sections, maxSections: 3
        )
        return [
            "title": resolved.title,
            "zim": resolved.zim,
            "path": resolved.path,
            "sections": picked.map { s -> [String: Any] in
                [
                    "title": s.title.isEmpty ? "lead" : s.title,
                    "text": s.text,
                    "bytes": s.bytes,
                ]
            },
        ] as [String: Any]
    }

    /// Shared helper used by `compare_articles` when given exactly
    /// two titles: probe for a dedicated Wikipedia relations article
    /// ("A–B relations", "Foreign relations of A", …) and return its
    /// lead + counterpart-mentioning sections. Returns nil (via
    /// throwing) if no such article exists, which the caller treats
    /// as the signal to fall back to side-by-side comparison.
    private func probeRelationsArticle(
        a: String, b: String, zim: String?
    ) async throws -> [String: Any]? {
        let candidates = ArticleHeuristics.relationshipCandidates(a: a, b: b)
        let aKey = a.lowercased()
        let bKey = b.lowercased()
        for candidate in candidates {
            do {
                let resolved = try await ArticleHeuristics.sectionsByTitle(
                    service: service, title: candidate, zim: zim
                )
                // Verify the resolved article actually names BOTH
                // subjects — the title-index fuzzy-match can drift
                // into related-but-wrong articles (real capture:
                // compare("London","Paris") → `Foreign relations of
                // London` → resolved as `Foreign_relations_of_the
                // _Mayor_of_London`, which is a mayor-office article,
                // not a London/Paris relations piece). Skip anything
                // that doesn't mention both subjects in the title.
                let titleLower = resolved.title.lowercased()
                let mentionsA = titleLower.contains(aKey)
                    || aKey.split(separator: " ").contains { titleLower.contains($0) }
                let mentionsB = titleLower.contains(bKey)
                    || bKey.split(separator: " ").contains { titleLower.contains($0) }
                guard mentionsA && mentionsB else { continue }
                let counterpart = candidate.lowercased().contains(aKey) ? b : a
                let picked = ArticleHeuristics.sectionsMentioning(
                    counterpart, in: resolved.sections, maxExtra: 3
                )
                return [
                    "strategy": "dedicated_relations_article",
                    "resolved_title": resolved.title,
                    "zim": resolved.zim,
                    "path": resolved.path,
                    "requested": [a, b],
                    "sections": picked.map { s -> [String: Any] in
                        [
                            "title": s.title.isEmpty ? "lead" : s.title,
                            "text": s.text,
                            "bytes": s.bytes,
                        ]
                    },
                ]
            } catch {
                continue
            }
        }
        return nil
    }

    private func dispatchNarrateArticle(args: [String: Any]) async throws -> [String: Any] {
        let title = (args["title"] as? String) ?? ""
        guard !title.isEmpty else {
            return ["error": "narrate_article requires a non-empty `title`."]
        }
        let zim = args["zim"] as? String
        let resolved = try await ArticleHeuristics.sectionsByTitle(
            service: service, title: title, zim: zim
        )
        let body = ArticleHeuristics.formatForNarration(
            title: resolved.title, sections: resolved.sections
        )
        return [
            // Sentinel the host uses to short-circuit the normal "feed
            // tool result back to model" loop and emit `text` directly
            // as the assistant reply. See ChatSession for the check.
            "pass_through": true,
            "title": resolved.title,
            "zim": resolved.zim,
            "path": resolved.path,
            "section_count": resolved.sections.count,
            "bytes": body.utf8.count,
            "text": body,
        ]
    }

    private func dispatchNearbyStories(args: [String: Any]) async throws -> [String: Any] {
        guard let lat = args["lat"] as? Double,
              let lon = args["lon"] as? Double,
              !(lat == 0 && lon == 0)
        else {
            return [
                "error": "nearby_stories requires numeric `lat` and `lon`. For a "
                    + "named place, call `nearby_stories_at_place(place=…)` "
                    + "instead.",
            ]
        }
        let radius = (args["radius_km"] as? Double) ?? 2.0
        let maxStories = max(1, min(10, (args["max_stories"] as? Int) ?? 4))
        let kinds = args["kinds"] as? [String]
        let zim = args["zim"] as? String
        return try await buildNearbyStories(
            lat: lat, lon: lon, radius: radius,
            maxStories: maxStories, kinds: kinds, zim: zim,
            origin: ["lat": lat, "lon": lon],
            resolved: nil
        )
    }

    private func dispatchNearbyStoriesAtPlace(args: [String: Any]) async throws -> [String: Any] {
        let place = (args["place"] as? String) ?? ""
        guard !place.isEmpty else {
            return ["error": "nearby_stories_at_place requires a non-empty `place`."]
        }
        let radius = (args["radius_km"] as? Double) ?? 2.0
        let maxStories = max(1, min(10, (args["max_stories"] as? Int) ?? 4))
        let kinds = args["kinds"] as? [String]
        let zim = args["zim"] as? String
        // Reuse the streetzim geocode path used by near_named_place — fans
        // out across loaded streetzims if the pinned one misses.
        let hits = try await service.geocode(
            query: place, limit: 1, zim: zim, kinds: nil
        )
        guard let first = hits.first else {
            throw ZimServiceError.noMatch(place)
        }
        return try await buildNearbyStories(
            lat: first.lat, lon: first.lon, radius: radius,
            maxStories: maxStories, kinds: kinds, zim: zim,
            origin: nil,
            resolved: first
        )
    }

    /// Shared body for both story tools: pull POIs with a Wikipedia link
    /// in range, fetch the lead paragraph in parallel, trim each to ~800
    /// chars on a sentence boundary. Overfetches nearby POIs to survive
    /// wiki titles that don't resolve in the loaded ZIMs.
    private func buildNearbyStories(
        lat: Double, lon: Double, radius: Double,
        maxStories: Int, kinds: [String]?, zim: String?,
        origin: [String: Double]?, resolved: Place?
    ) async throws -> [String: Any] {
        let overfetch = max(maxStories * 3, maxStories + 3)
        let nearby = try await service.nearPlaces(
            lat: lat, lon: lon,
            radiusKm: radius,
            limit: overfetch,
            kinds: kinds,
            zim: zim,
            hasWiki: true
        )
        let candidates: [(place: Place, distanceMeters: Double)] = nearby.results
            .filter { pair in
                let w = pair.place.wiki ?? ""
                return !w.isEmpty
            }

        if candidates.isEmpty {
            var out: [String: Any] = [
                "radius_km": radius,
                "count": 0,
                "stories": [] as [[String: Any]],
                "note": "No wiki-linked places found in range. Try widening "
                    + "`radius_km` or dropping the `kinds` filter.",
            ]
            if let origin { out["origin"] = origin }
            if let resolved { out["resolved"] = Self.encodePlace(resolved) }
            return out
        }

        let svc = service
        let excerpts: [(Int, [String: Any])] = await withTaskGroup(
            of: (Int, [String: Any]?).self
        ) { group in
            for (i, pair) in candidates.prefix(maxStories * 2).enumerated() {
                let place = pair.place
                let distance = pair.distanceMeters
                let wiki = place.wiki ?? ""
                group.addTask { [svc] in
                    do {
                        // Wiki tag form "en:HP_Garage" is stripped of prefix
                        // inside articleByTitle; we pass the raw tag so the
                        // lookup stays identical to how get_article_by_title
                        // handles near_places' `wikipedia` field today.
                        let hit = try await svc.articleByTitle(
                            title: wiki, zim: nil, section: "lead"
                        )
                        guard let excerpt = Self.cleanedWikiExcerpt(
                            title: hit.title, leadText: hit.section.text
                        ) else { return (i, nil) }
                        return (i, [
                            "place_name": place.name,
                            "wiki_title": hit.title,
                            "wiki_tag": wiki,
                            "zim": hit.zim,
                            "path": hit.path,
                            "lat": place.lat,
                            "lon": place.lon,
                            "distance_m": Int(distance.rounded()),
                            "excerpt": excerpt,
                            "has_more_sections": true,
                        ])
                    } catch {
                        return (i, nil)
                    }
                }
            }
            var collected: [(Int, [String: Any])] = []
            for await row in group {
                if let payload = row.1 {
                    collected.append((row.0, payload))
                }
            }
            return collected
        }

        let ordered = excerpts
            .sorted { $0.0 < $1.0 }
            .prefix(maxStories)
            .map(\.1)
        var out: [String: Any] = [
            "radius_km": radius,
            "count": ordered.count,
            "stories": Array(ordered),
        ]
        if ordered.isEmpty {
            out["note"] = "Found \(candidates.count) wiki-linked places but "
                + "none of their articles resolved in the loaded Wikipedia "
                + "ZIMs. Try a different region or check that the matching "
                + "Wikipedia ZIM is loaded."
        } else if ordered.count < maxStories && candidates.count > ordered.count {
            out["note"] = "Some wiki-linked places had articles that didn't "
                + "resolve in the loaded Wikipedia ZIMs — returned the ones "
                + "that did."
        }
        if let origin { out["origin"] = origin }
        if let resolved { out["resolved"] = Self.encodePlace(resolved) }
        return out
    }

    private func dispatchWhatIsHere(args: [String: Any]) async throws -> [String: Any] {
        var lat = args["lat"] as? Double
        var lon = args["lon"] as? Double
        if lat == nil || lon == nil {
            if let provider = hostStateProvider {
                let snap = await provider()
                if let loc = snap.currentLocation {
                    lat = loc.lat
                    lon = loc.lon
                }
            }
        }
        guard let resolvedLat = lat, let resolvedLon = lon,
              !(resolvedLat == 0 && resolvedLon == 0)
        else {
            return [
                "error": "what_is_here needs either explicit `lat`+`lon` or a "
                    + "host-supplied GPS fix. Neither is available.",
            ]
        }
        // Reverse-geocode approximation: nearest admin-named place within
        // 1.5 km. No separate reverse-geocode primitive exists today; this
        // leans on the same proximity index as near_places.
        let nearest = try await service.nearPlaces(
            lat: resolvedLat, lon: resolvedLon,
            radiusKm: 1.5, limit: 5,
            kinds: ["place"],
            zim: args["zim"] as? String,
            hasWiki: false
        )
        guard let first = nearest.results.first else {
            return [
                "error": "No named place (neighborhood / city) within 1.5 km "
                    + "of (\(resolvedLat), \(resolvedLon)). This streetzim may "
                    + "not cover the current location.",
                "lat": resolvedLat,
                "lon": resolvedLon,
            ]
        }
        var out: [String: Any] = [
            "lat": resolvedLat,
            "lon": resolvedLon,
            "nearest_named_place": first.place.name,
            "admin_area": first.place.subtype.isEmpty ? first.place.kind : first.place.subtype,
            "distance_m": Int(first.distanceMeters.rounded()),
            "place_lat": first.place.lat,
            "place_lon": first.place.lon,
        ]
        if let wiki = first.place.wiki, !wiki.isEmpty {
            out["wikipedia"] = wiki
            if let hit = try? await service.articleByTitle(
                title: wiki, zim: nil, section: "lead"
            ),
               let summary = Self.cleanedWikiExcerpt(
                title: hit.title, leadText: hit.section.text
               )
            {
                out["wiki_title"] = hit.title
                out["wiki_zim"] = hit.zim
                out["wiki_summary"] = summary
            }
        }
        return out
    }

    private func dispatchRouteStatus() async -> [String: Any] {
        guard let provider = hostStateProvider else {
            return [
                "error": "route_status is only available when the host wires a "
                    + "route-state provider. No active-route plumbing in this "
                    + "environment.",
            ]
        }
        let snap = await provider()
        guard let route = snap.activeRoute else {
            return [
                "error": "No active driving route. Plan one first with "
                    + "`route_from_places` or `plan_driving_route`.",
            ]
        }
        guard let loc = snap.currentLocation else {
            return [
                "destination_name": route.destinationName,
                "total_km": route.totalDistanceMeters / 1000.0,
                "total_minutes": route.totalDurationSeconds / 60.0,
                "progress_pct": 0,
                "total_steps": route.turnByTurn.count,
                "note": "GPS fix not yet available — reporting full-route totals.",
            ]
        }
        let progress = route.remaining(at: .init(lat: loc.lat, lon: loc.lon))
        let pct = Int((progress.fractionDone * 100).rounded())
        // Approximate which turn is next by slotting progress into the
        // turn list. The `turnByTurn` array isn't index-aligned with the
        // polyline (it's one-per-road-segment), so this is a rough
        // heuristic — fine for "what's my next turn" voice prompts.
        var nextStep: String = ""
        if !route.turnByTurn.isEmpty {
            let idx = min(
                route.turnByTurn.count - 1,
                Int(progress.fractionDone * Double(route.turnByTurn.count))
            )
            nextStep = route.turnByTurn[idx]
        }
        return [
            "destination_name": route.destinationName,
            "remaining_km": progress.remainingMeters / 1000.0,
            "remaining_minutes": progress.remainingSeconds / 60.0,
            "progress_pct": pct,
            "next_step": nextStep,
            "total_steps": route.turnByTurn.count,
        ]
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

    /// Fetch Wikipedia lead paragraphs in parallel for every result
    /// that carries a `wiki` OSM tag (e.g. `"en:HP_Garage"`). Gives
    /// the chat-bubble map popup + the `List` sheet something richer
    /// than kind + distance to show when the user taps a pin; also
    /// means the skip-model-reply fast path can present real content
    /// without a second model turn.
    ///
    /// Returns a dict keyed by the result row's index so
    /// `encodeNearPlaces` can look up the matching excerpt without
    /// relying on positional pairing. Failures (wiki tag present but
    /// the article isn't in any loaded ZIM) drop silently — the row
    /// just doesn't get enriched.
    func fetchWikiExcerpts(
        for result: NearPlacesResult
    ) async -> [Int: [String: Any]] {
        // Overall cap so a result set of 50+ hits doesn't kick off 50
        // parallel ZIM reads — a handful of leads is enough to dress
        // up the top pins the user will actually see on the map.
        let cap = 10
        var candidates: [(idx: Int, wiki: String)] = []
        candidates.reserveCapacity(min(cap, result.results.count))
        for (idx, pair) in result.results.enumerated() {
            let w = pair.place.wiki ?? ""
            if !w.isEmpty {
                candidates.append((idx, w))
                if candidates.count >= cap { break }
            }
        }
        // Diagnostic log — `print()` to stderr doesn't reach
        // `idevicesyslog`, so route through `os.Logger` which the
        // host's streamer captures. Tells us whether "no wiki on pins"
        // is because the streetzim records didn't carry the tag vs
        // the article not resolving against a loaded Wikipedia ZIM.
        let totalWikiTagged = result.results.filter { !($0.place.wiki?.isEmpty ?? true) }.count
        let sampleNames = result.results.prefix(3)
            .map { $0.place.name.isEmpty ? "(unnamed)" : $0.place.name }
            .joined(separator: ", ")
        Self.toolLog.notice(
            "fetchWikiExcerpts: \(totalWikiTagged, privacy: .public) of \(result.results.count, privacy: .public) results carry a wiki tag; sample: \(sampleNames, privacy: .public); fetching \(candidates.count, privacy: .public)"
        )
        // Even when the streetzim contributed no tagged candidates
        // we still want the name-search fallback to run — that's the
        // whole point of the fallback. So instead of an early
        // `return [:]`, fall through to the task-group with zero
        // primary tasks + let the fallback pass do the heavy lifting.
        let svc = service
        return await withTaskGroup(
            of: (Int, [String: Any]?).self,
            returning: [Int: [String: Any]].self
        ) { group in
            for (idx, wikiTag) in candidates {
                group.addTask { [svc] in
                    do {
                        let hit = try await svc.articleByTitle(
                            title: wikiTag, zim: nil, section: "lead"
                        )
                        guard let excerpt = Self.cleanedWikiExcerpt(
                            title: hit.title, leadText: hit.section.text
                        ) else { return (idx, nil) }
                        return (idx, [
                            "excerpt": excerpt,
                            "wiki_title": hit.title,
                            "wiki_path": hit.path,
                            "zim": hit.zim,
                        ])
                    } catch {
                        return (idx, nil)
                    }
                }
            }
            var out: [Int: [String: Any]] = [:]
            for await (idx, payload) in group {
                if let payload { out[idx] = payload }
            }
            Self.toolLog.notice(
                "fetchWikiExcerpts: enriched \(out.count, privacy: .public) of \(candidates.count, privacy: .public) via OSM wiki-tag"
            )
            // Name-search fallback DISABLED (2026-04-22).
            //
            // The fallback tried to enrich rows OSM didn't wiki-tag
            // by running a Wikipedia search on the place name and
            // attaching the top hit's lead. Two real-device captures
            // show why this is almost always wrong for short,
            // common-noun POI names:
            //
            //   * "Taverna" (local Greek restaurant) → Wikipedia's
            //     `Taverna` article, which is about the scientific
            //     workflow software of the same name. The user sees
            //     a software description under a restaurant row.
            //   * "Masterworks" (local art-related POI) →
            //     Wikipedia's generic `Masterworks` concept article
            //     (a glossary-style entry on notable artworks).
            //
            // Even with exact-normalised-title matching + disambig
            // rejection, a single-word POI name that's ALSO a
            // common-noun Wikipedia entry will pass every check the
            // resolver runs. The article exists, the title matches,
            // it isn't a disambig page — but it isn't ABOUT this
            // place. The OSM wiki-tag primary path is an explicit
            // link from the mapper who tagged the node; the
            // fallback is a guess at string similarity.
            //
            // Trust the OSM tag, skip the guess. Leaving
            // `fetchWikiExcerptsByNameSearch` in place (not deleted)
            // so a future, stricter variant can re-enable it — the
            // current problem is the activation gate, not the
            // tightening we've already done.
            return out
        }
    }

    /// Secondary enrichment — searches the loaded Wikipedia ZIM by
    /// place name for rows the streetzim didn't tag. Only accepts a
    /// hit when the returned title *closely* matches the place's
    /// name (loose-equal after punctuation strip), so a generic
    /// "restaurant" query doesn't latch onto the restaurant
    /// Wikipedia article.
    func fetchWikiExcerptsByNameSearch(
        for result: NearPlacesResult,
        excluding alreadyEnriched: Set<Int>,
        cap: Int
    ) async -> [Int: [String: Any]] {
        var candidates: [(idx: Int, name: String)] = []
        for (idx, pair) in result.results.enumerated() {
            if alreadyEnriched.contains(idx) { continue }
            let name = pair.place.name
            if name.count < 4 { continue }  // skip 1–3 char POIs
            candidates.append((idx, name))
            if candidates.count >= cap { break }
        }
        if candidates.isEmpty { return [:] }
        let svc = service
        return await withTaskGroup(
            of: (Int, [String: Any]?).self,
            returning: [Int: [String: Any]].self
        ) { group in
            for (idx, name) in candidates {
                group.addTask { [svc] in
                    do {
                        let hits = try await svc.search(
                            query: name, limit: 3, kind: .wikipedia
                        )
                        guard let hit = hits.first else { return (idx, nil) }
                        // Tight title match. The previous "a contains
                        // b OR b contains a" loosening let too many
                        // unrelated concept pages through — e.g.
                        // "Wine Cellar" → `Wine_cellar` (generic
                        // concept article about wine cellars in
                        // general, not this specific place). Demand
                        // a normalised-equal title; anything else is
                        // a fuzzy match we can't trust.
                        let a = Self.normaliseTitle(name)
                        let b = Self.normaliseTitle(hit.title)
                        if a != b { return (idx, nil) }
                        let article = try await svc.articleByTitle(
                            title: hit.path.hasPrefix("A/")
                                ? String(hit.path.dropFirst(2))
                                : hit.title,
                            zim: hit.zim, section: "lead"
                        )
                        // Still run through the disambig / title-rep
                        // cleaner — an exact normalised title match
                        // can still land on a disambiguation page for
                        // short generic names (real capture: "Oak
                        // Grove" → `Oak_Grove` → "may refer to:").
                        guard let excerpt = Self.cleanedWikiExcerpt(
                            title: article.title,
                            leadText: article.section.text
                        ) else { return (idx, nil) }
                        return (idx, [
                            "excerpt":     excerpt,
                            "wiki_title":  article.title,
                            "wiki_path":   article.path,
                            "zim":         article.zim,
                        ])
                    } catch {
                        return (idx, nil)
                    }
                }
            }
            var out: [Int: [String: Any]] = [:]
            for await (idx, payload) in group {
                if let payload { out[idx] = payload }
            }
            return out
        }
    }

    /// Loose-equality normaliser for comparing OSM place names to
    /// Wikipedia article titles. Drops ampersands / periods /
    /// redundant spaces and lowercases; `&` → `and` so "Iris & B.
    /// Gerald Cantor Center" matches "Iris and B. Gerald Cantor
    /// Center" when Wikipedia uses the spelled-out form.
    private static func normaliseTitle(_ s: String) -> String {
        let lowered = s.lowercased()
            .replacingOccurrences(of: "&", with: "and")
            .replacingOccurrences(of: ".", with: "")
            .replacingOccurrences(of: ",", with: "")
            .replacingOccurrences(of: "'", with: "")
        // Collapse whitespace runs to single spaces then trim.
        let parts = lowered.split(whereSeparator: { $0.isWhitespace })
        return parts.joined(separator: " ")
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
        result: NearPlacesResult,
        excerpts: [Int: [String: Any]] = [:]
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
            "results": result.results.enumerated().map { idx, pair -> [String: Any] in
                var r = encodePlace(pair.place)
                r["distance_m"] = Int(pair.distanceMeters.rounded())
                // Inject Wikipedia lead + path when `fetchWikiExcerpts`
                // resolved the place's `wiki` tag. Popup rendering on
                // the iOS side reads `excerpt` / `wiki_path` to show
                // the summary + "Read article" button.
                if let enrich = excerpts[idx] {
                    if let excerpt = enrich["excerpt"] as? String, !excerpt.isEmpty {
                        r["excerpt"] = excerpt
                    }
                    if let path = enrich["wiki_path"] as? String, !path.isEmpty {
                        r["wiki_path"] = path
                    }
                    if let title = enrich["wiki_title"] as? String, !title.isEmpty {
                        r["wiki_title"] = title
                    }
                }
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
        // Overture enrichment — same short keys the iOS parser reads
        // (`parsePlacesJSON` looks for `ws`/`p`/`brand`). Only emitted
        // when the record had them; keeps older streetzim payloads
        // indistinguishable on the wire.
        if let ws = p.website { out["ws"] = ws }
        if let ph = p.phone { out["p"] = ph }
        if let br = p.brand { out["brand"] = br }
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

    private static let articleByTitleSchema: Data = #"""
    {"type":"object","required":["title"],"properties":{
        "title":{"type":"string","description":"Article title. Accepts either a bare title (\"HP Garage\") or the language-prefixed form returned by `near_places` in the `wikipedia` field (\"en:HP Garage\")."},
        "section":{"type":"string","default":"lead","description":"Section to return. Default 'lead' gives a one-paragraph summary suitable for rattling off several POIs in a row."},
        "zim":{"type":"string","description":"Optional: pin to a specific ZIM filename (otherwise we search every loaded Wikipedia ZIM)."}
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
        // Post-cull: the model only sees one `near_places` declaration.
        // Passing `place` routes internally through the geocode path
        // (what was previously `near_named_place`); passing explicit
        // `lat`/`lon` skips geocode and goes straight to the coord
        // path. Either form is valid, hence no required fields here.
        schemaJSON(
            required: [],
            properties: [
                ("place", ["type": "string",
                           "description": "Free-text place name (\"Palo Alto\", \"downtown Portland\"). When set, the tool geocodes this and uses it as the center — use for \"what's around <place>\" / \"restaurants in San Francisco\". Omit to search around `lat`/`lon` (or, if those are omitted too, the user's current GPS from the system preamble)."]),
                ("lat", ["type": "number", "description": "Center latitude. Use with `lon` for explicit coords; omit when `place` is set or the user's current GPS is the implicit center."]),
                ("lon", ["type": "number", "description": "Center longitude (pair with `lat`)."]),
                ("radius_km", ["type": "number", "default": 1.0,
                               "description": "Search radius in km. 0.5 ≈ walking distance, 2–5 ≈ neighborhood."]),
                ("limit", ["type": "integer", "default": 10,
                           "description": "Max results, sorted nearest first."]),
                ("kinds", kindsSchema(vocabulary: vocabulary)),
                ("zim", ["type": "string",
                         "description": "Specific streetzim filename, else try them all."]),
                ("has_wiki", ["type": "boolean", "default": false,
                              "description": "When true, only return places that have an associated Wikipedia / Wikidata article. Use for \"what's interesting around here\" and similar queries, or to filter hundreds of results (\"bars in Seattle\") down to notable ones."]),
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

    private static let showMapSchema: Data = #"""
    {"type":"object","required":["place"],"properties":{
        "place":{"type":"string","description":"Free-text place name to center the map on."},
        "zim":{"type":"string","description":"Optional streetzim filename to geocode against."}
    }}
    """#.data(using: .utf8)!

    private static let articleOverviewSchema: Data = #"""
    {"type":"object","required":["title"],"properties":{
        "title":{"type":"string","description":"Article title. Accepts bare title or the OSM language-prefixed form (\"en:HP Garage\")."},
        "max_sections":{"type":"integer","default":5,"minimum":1,"maximum":10,"description":"Max sections to return (always includes lead)."},
        "zim":{"type":"string","description":"Optional: pin to a specific Wikipedia ZIM filename."}
    }}
    """#.data(using: .utf8)!

    private static let compareArticlesSchema: Data = #"""
    {"type":"object","required":["titles"],"properties":{
        "titles":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":4,"description":"2–4 article titles to compare side-by-side."},
        "section":{"type":"string","description":"Optional: align on this named section from each article. Omit for each article's lead + top sections."},
        "zim":{"type":"string","description":"Optional: pin to a specific Wikipedia ZIM filename."}
    }}
    """#.data(using: .utf8)!

    private static let articleRelationshipSchema: Data = #"""
    {"type":"object","required":["a","b"],"properties":{
        "a":{"type":"string","description":"First entity name (country, organization, person, …)."},
        "b":{"type":"string","description":"Second entity name."},
        "zim":{"type":"string","description":"Optional: pin to a specific Wikipedia ZIM filename."}
    }}
    """#.data(using: .utf8)!

    private static let narrateArticleSchema: Data = #"""
    {"type":"object","required":["title"],"properties":{
        "title":{"type":"string","description":"Article title. Accepts bare title or the OSM language-prefixed form (\"en:HP Garage\")."},
        "zim":{"type":"string","description":"Optional: pin to a specific Wikipedia ZIM filename."}
    }}
    """#.data(using: .utf8)!

    private static let whatIsHereSchema: Data = #"""
    {"type":"object","properties":{
        "lat":{"type":"number","description":"Optional: latitude. Omit to use the host's current GPS fix."},
        "lon":{"type":"number","description":"Optional: longitude. Omit to use the host's current GPS fix."},
        "zim":{"type":"string","description":"Optional: restrict to a specific streetzim filename."}
    }}
    """#.data(using: .utf8)!

    private static func nearbyStoriesSchema(vocabulary: [String]) -> Data {
        // Post-cull: merged `nearby_stories_at_place` in. Pass `place`
        // for a named location; omit it to anchor on the user's GPS
        // (implicit) or explicit `lat`/`lon`.
        schemaJSON(
            required: [],
            properties: [
                ("place", ["type": "string",
                           "description": "Free-text place name — geocoded internally. Use for \"interesting stories from <place>\" / \"history of downtown Portland\". Omit to anchor on the user's GPS / explicit coords."]),
                ("lat", ["type": "number",
                         "description": "Center latitude. Use with `lon` for explicit coords; omit when `place` is set or the user's current GPS is the implicit center."]),
                ("lon", ["type": "number",
                         "description": "Center longitude."]),
                ("radius_km", ["type": "number", "default": 2.0,
                               "description": "Search radius. 1 ≈ walking, 2–5 ≈ neighborhood, 10 ≈ whole city."]),
                ("max_stories", ["type": "integer", "default": 4, "minimum": 1, "maximum": 10,
                                 "description": "How many story excerpts to return."]),
                ("kinds", kindsSchema(vocabulary: vocabulary)),
                ("zim", ["type": "string",
                         "description": "Optional streetzim filename."]),
            ]
        )
    }

    private static func nearbyStoriesAtPlaceSchema(vocabulary: [String]) -> Data {
        schemaJSON(
            required: ["place"],
            properties: [
                ("place", ["type": "string",
                           "description": "Free-text place name — geocoded internally."]),
                ("radius_km", ["type": "number", "default": 2.0,
                               "description": "Search radius around the resolved place."]),
                ("max_stories", ["type": "integer", "default": 4, "minimum": 1, "maximum": 10]),
                ("kinds", kindsSchema(vocabulary: vocabulary)),
                ("zim", ["type": "string",
                         "description": "Optional streetzim filename."]),
            ]
        )
    }
}
