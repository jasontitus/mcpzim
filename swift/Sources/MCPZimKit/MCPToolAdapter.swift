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

public actor MCPToolAdapter {
    private let service: any ZimService
    private let hasStreetzim: Bool

    public init(service: any ZimService, hasStreetzim: Bool) {
        self.service = service
        self.hasStreetzim = hasStreetzim
    }

    public static func from(service: any ZimService) async -> MCPToolAdapter {
        let inventory = (try? await service.inventory()) ?? InventoryResult(zims: [], capabilities: [])
        let hasStreetzim = inventory.zims.contains { $0.kind == .streetzim && $0.hasRoutingData }
        return MCPToolAdapter(service: service, hasStreetzim: hasStreetzim)
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
                    "Full-text search across every loaded ZIM. Returns paths, titles and "
                    + "snippets the agent can feed into get_article.",
                inputSchemaJSON: Self.searchSchema
            ),
            MCPTool(
                name: "get_article",
                description: "Fetch a single ZIM entry by path, as plain text.",
                inputSchemaJSON: Self.articleSchema
            ),
            MCPTool(
                name: "get_main_page",
                description: "Fetch the main/home page of one or every loaded ZIM.",
                inputSchemaJSON: Self.mainPageSchema
            ),
        ]
        if hasStreetzim {
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
                    name: "route_from_places",
                    description:
                        "Plan a driving route between two free-text place names. Convenience "
                        + "wrapper over geocode + plan_driving_route.",
                    inputSchemaJSON: Self.routeFromPlacesSchema
                ),
            ])
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
            let limit = (args["limit"] as? Int) ?? 10
            let kindString = args["kind"] as? String
            let kind = kindString.flatMap { ZimKind(rawValue: $0) }
            let hits = try await service.search(query: query, limit: limit, kind: kind)
            return ["query": query, "count": hits.count, "hits": hits.map(Self.encodeHit)]
        case "get_article":
            let path = (args["path"] as? String) ?? ""
            let zim = args["zim"] as? String
            let art = try await service.article(path: path, zim: zim)
            return Self.encodeArticle(art)
        case "get_main_page":
            let zim = args["zim"] as? String
            let pages = try await service.mainPage(zim: zim)
            return ["pages": pages.map(Self.encodeArticle)]
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
        case "route_from_places":
            let result = try await service.routeFromPlaces(
                origin: (args["origin"] as? String) ?? "",
                destination: (args["destination"] as? String) ?? "",
                zim: args["zim"] as? String
            )
            var body = Self.encodeRoute(result.route)
            body["origin_resolved"] = Self.encodePlace(result.resolved.origin)
            body["destination_resolved"] = Self.encodePlace(result.resolved.destination)
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

    private static func encodePlace(_ p: Place) -> [String: Any] {
        var out: [String: Any] = [
            "name": p.name, "type": p.kind, "lat": p.lat, "lon": p.lon,
        ]
        if !p.subtype.isEmpty { out["subtype"] = p.subtype }
        if !p.location.isEmpty { out["location"] = p.location }
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

    private static let routeFromPlacesSchema: Data = #"""
    {"type":"object","required":["origin","destination"],"properties":{
        "origin":{"type":"string"},
        "destination":{"type":"string"},
        "zim":{"type":"string"}
    }}
    """#.data(using: .utf8)!
}
