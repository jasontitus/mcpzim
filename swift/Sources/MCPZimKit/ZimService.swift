// SPDX-License-Identifier: MIT
//
// In-process Swift service API. Apps that host a local LLM (Swift-Gemma4-Core,
// MLX-Swift-based apps, or anything else) can register a `ZimService` as the
// tool backend directly, without paying JSON-RPC encode/decode cost per call.
// For LAN / cross-process scenarios, wrap this with MCPZimServerKit instead.

import Foundation

public struct ArticleResult: Sendable {
    public let zim: String
    public let path: String
    public let title: String
    public let mimetype: String
    public let text: String
    public let bytes: Int
}

public struct SearchHitResult: Sendable {
    public let zim: String
    public let kind: ZimKind
    public let path: String
    public let title: String
    public let snippet: String
}

public struct InventoryEntry: Sendable {
    public let name: String
    public let kind: ZimKind
    public let metadata: ZimMetadata
    public let hasRoutingData: Bool
}

public struct InventoryResult: Sendable {
    public let zims: [InventoryEntry]
    public let capabilities: [String]
}

public struct RouteRequest: Sendable {
    public let originLat: Double
    public let originLon: Double
    public let destLat: Double
    public let destLon: Double
    public let zim: String?
    public init(originLat: Double, originLon: Double, destLat: Double, destLon: Double, zim: String? = nil) {
        self.originLat = originLat
        self.originLon = originLon
        self.destLat = destLat
        self.destLon = destLon
        self.zim = zim
    }
}

public enum ZimServiceError: Error, CustomStringConvertible {
    case unknownZim(String)
    case noStreetzim
    case noMatch(String)
    case notFound(String)
    case noRoute

    public var description: String {
        switch self {
        case .unknownZim(let n): return "unknown zim \(n)"
        case .noStreetzim: return "no streetzim ZIM with routing data is loaded"
        case .noMatch(let q): return "could not resolve \(q)"
        case .notFound(let p): return "not found: \(p)"
        case .noRoute: return "no route found"
        }
    }
}

/// Host-facing service. Concrete implementations live in-app; MCPZimServerKit
/// adapts this interface to JSON-RPC.
public protocol ZimService: Sendable {
    func inventory() async throws -> InventoryResult
    func search(query: String, limit: Int, kind: ZimKind?) async throws -> [SearchHitResult]
    func article(path: String, zim: String?) async throws -> ArticleResult
    func mainPage(zim: String?) async throws -> [ArticleResult]

    // Streetzim-only. Implementations may throw `.noStreetzim` if unavailable;
    // MCPZimServerKit uses those throws to decide whether to register the tool.
    func planDrivingRoute(_ req: RouteRequest) async throws -> Route
    func geocode(query: String, limit: Int, zim: String?, kinds: [String]?) async throws -> [Place]
    func routeFromPlaces(origin: String, destination: String, zim: String?) async throws -> (resolved: (origin: Place, destination: Place), route: Route)
}

/// Default pure-Swift implementation. Constructed from a list of `ZimReader`s
/// supplied by the host app — the reader implementations are where libzim
/// (via CoreKiwix.xcframework or a pure-Swift port) actually lives.
public actor DefaultZimService: ZimService {
    private let readers: [(name: String, reader: ZimReader)]
    private var graphs: [String: SZRGGraph] = [:]
    private var chunks: [String: [String: [[String: Any]]]] = [:]
    private var manifests: [String: [String: Int]] = [:]

    public init(readers: [(name: String, reader: ZimReader)]) {
        self.readers = readers
    }

    private var streetzimReaders: [(name: String, reader: ZimReader)] {
        readers.filter { $0.reader.kind == .streetzim && $0.reader.hasRoutingData }
    }

    public func inventory() -> InventoryResult {
        let entries = readers.map { pair in
            InventoryEntry(
                name: pair.name,
                kind: pair.reader.kind,
                metadata: pair.reader.metadata,
                hasRoutingData: pair.reader.hasRoutingData
            )
        }
        var caps: Set<String> = []
        if !readers.isEmpty {
            caps.formUnion(["search", "get_article", "list_libraries"])
        }
        if readers.contains(where: { [.wikipedia, .mdwiki, .generic].contains($0.reader.kind) }) {
            caps.insert("encyclopedia")
        }
        if readers.contains(where: { $0.reader.kind == .mdwiki }) { caps.insert("medical") }
        if readers.contains(where: { $0.reader.kind == .wikipedia }) { caps.insert("general_knowledge") }
        if !streetzimReaders.isEmpty {
            caps.formUnion(["plan_route", "geocode", "maps"])
        }
        return InventoryResult(zims: entries, capabilities: caps.sorted())
    }

    public func search(query: String, limit: Int, kind: ZimKind?) async throws -> [SearchHitResult] {
        var results: [SearchHitResult] = []
        for pair in readers {
            if let wanted = kind, pair.reader.kind != wanted { continue }
            let hits = (try? pair.reader.search(query: query, limit: limit)) ?? []
            for h in hits {
                results.append(SearchHitResult(
                    zim: pair.name,
                    kind: pair.reader.kind,
                    path: h.path,
                    title: h.title,
                    snippet: "" // host-side HTML→text is optional; omitted here.
                ))
            }
        }
        return results
    }

    public func article(path: String, zim: String?) async throws -> ArticleResult {
        let targets = readers.filter { zim == nil || $0.name == zim }
        if targets.isEmpty { throw ZimServiceError.unknownZim(zim ?? "<any>") }
        for pair in targets {
            if let entry = try? pair.reader.read(path: path), let entry {
                return ArticleResult(
                    zim: pair.name,
                    path: entry.path,
                    title: entry.title,
                    mimetype: entry.mimetype,
                    text: String(data: entry.content, encoding: .utf8) ?? "",
                    bytes: entry.content.count
                )
            }
        }
        throw ZimServiceError.notFound(path)
    }

    public func mainPage(zim: String?) async throws -> [ArticleResult] {
        var out: [ArticleResult] = []
        for pair in readers {
            if let zim, pair.name != zim { continue }
            if let entry = try? pair.reader.readMainPage(), let entry {
                out.append(ArticleResult(
                    zim: pair.name,
                    path: entry.path,
                    title: entry.title,
                    mimetype: entry.mimetype,
                    text: String(data: entry.content, encoding: .utf8) ?? "",
                    bytes: entry.content.count
                ))
            }
        }
        return out
    }

    public func planDrivingRoute(_ req: RouteRequest) async throws -> Route {
        guard let pair = try pickStreetzim(req.zim) else { throw ZimServiceError.noStreetzim }
        let graph = try loadGraph(pair: pair)
        let origin = graph.nearestNode(lat: req.originLat, lon: req.originLon)
        let goal = graph.nearestNode(lat: req.destLat, lon: req.destLon)
        guard origin >= 0, goal >= 0, let route = aStar(graph: graph, origin: origin, goal: goal) else {
            throw ZimServiceError.noRoute
        }
        return route
    }

    public func geocode(query: String, limit: Int, zim: String?, kinds: [String]?) async throws -> [Place] {
        guard let pair = try pickStreetzim(zim) else { throw ZimServiceError.noStreetzim }
        let prefix = Geocoder.normalizePrefix(query)
        let manifest = try loadManifest(pair: pair)
        if !manifest.isEmpty && manifest[prefix] == nil { return [] }
        let records = try loadChunk(pair: pair, prefix: prefix)
        return Geocoder.rank(records: records, query: query, limit: limit, kinds: kinds.map(Set.init))
    }

    public func routeFromPlaces(origin: String, destination: String, zim: String?) async throws -> (resolved: (origin: Place, destination: Place), route: Route) {
        let oHits = try await geocode(query: origin, limit: 1, zim: zim, kinds: nil)
        let dHits = try await geocode(query: destination, limit: 1, zim: zim, kinds: nil)
        guard let o = oHits.first else { throw ZimServiceError.noMatch(origin) }
        guard let d = dHits.first else { throw ZimServiceError.noMatch(destination) }
        let route = try await planDrivingRoute(RouteRequest(
            originLat: o.lat, originLon: o.lon,
            destLat: d.lat, destLon: d.lon,
            zim: zim
        ))
        return ((origin: o, destination: d), route: route)
    }

    // MARK: - Internal loaders

    private func pickStreetzim(_ zim: String?) throws -> (name: String, reader: ZimReader)? {
        let candidates = streetzimReaders
        if candidates.isEmpty { return nil }
        guard let name = zim else { return candidates.first }
        guard let match = candidates.first(where: { $0.name == name }) else {
            throw ZimServiceError.unknownZim(name)
        }
        return match
    }

    private func loadGraph(pair: (name: String, reader: ZimReader)) throws -> SZRGGraph {
        if let cached = graphs[pair.name] { return cached }
        guard let entry = try pair.reader.read(path: "routing-data/graph.bin"), let entry else {
            throw ZimServiceError.noStreetzim
        }
        let g = try SZRGGraph.parse(entry.content)
        graphs[pair.name] = g
        return g
    }

    private func loadManifest(pair: (name: String, reader: ZimReader)) throws -> [String: Int] {
        if let cached = manifests[pair.name] { return cached }
        guard let entry = try pair.reader.read(path: "search-data/manifest.json"), let entry else {
            manifests[pair.name] = [:]
            return [:]
        }
        let parsed = (try? JSONSerialization.jsonObject(with: entry.content)) as? [String: Any]
        let chunks = (parsed?["chunks"] as? [String: Int]) ?? [:]
        manifests[pair.name] = chunks
        return chunks
    }

    private func loadChunk(pair: (name: String, reader: ZimReader), prefix: String) throws -> [[String: Any]] {
        let key = "\(pair.name)|\(prefix)"
        if let cached = chunks[pair.name]?[prefix] { return cached }
        guard let entry = try pair.reader.read(path: "search-data/\(prefix).json"), let entry else {
            return []
        }
        let parsed = (try? JSONSerialization.jsonObject(with: entry.content)) as? [[String: Any]] ?? []
        var byPrefix = chunks[pair.name] ?? [:]
        byPrefix[prefix] = parsed
        chunks[pair.name] = byPrefix
        _ = key
        return parsed
    }
}
