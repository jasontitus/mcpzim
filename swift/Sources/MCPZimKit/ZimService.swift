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

/// Result of a `nearPlaces` scan. Carries the top-N nearest records AND
/// the overall subtype breakdown within the search radius, so the UI can
/// surface a summary like "20 cafes, 10 bars, 5 attractions" — and then
/// let the user drill into any of those categories — without re-scanning.
public struct NearPlacesResult: Sendable {
    public let totalInRadius: Int
    /// subtype (preferred) or kind (fallback) → count of records that
    /// fell inside the radius. Sorted descending is the caller's job.
    public let breakdown: [String: Int]
    public let results: [(place: Place, distanceMeters: Double)]
    public init(totalInRadius: Int, breakdown: [String: Int], results: [(place: Place, distanceMeters: Double)]) {
        self.totalInRadius = totalInRadius
        self.breakdown = breakdown
        self.results = results
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
    func articleSections(path: String, zim: String?) async throws -> (zim: String, title: String, sections: [ArticleSection])
    func articleSection(path: String, section: String, zim: String?) async throws -> (zim: String, title: String, section: ArticleSection)
    func mainPage(zim: String?) async throws -> [ArticleResult]

    // Streetzim-only. Implementations may throw `.noStreetzim` if unavailable;
    // MCPZimServerKit uses those throws to decide whether to register the tool.
    func planDrivingRoute(_ req: RouteRequest) async throws -> Route
    func geocode(query: String, limit: Int, zim: String?, kinds: [String]?) async throws -> [Place]
    func nearPlaces(lat: Double, lon: Double, radiusKm: Double, limit: Int, kinds: [String]?, zim: String?) async throws -> NearPlacesResult
    func nearNamedPlace(place: String, radiusKm: Double, limit: Int, kinds: [String]?, zim: String?) async throws -> (resolved: Place, result: NearPlacesResult)
    func zimInfo(zim: String?) async throws -> [[String: Any]]
    func routeFromPlaces(origin: String, destination: String, zim: String?) async throws -> (resolved: (origin: Place, destination: Place), route: Route, zimUsed: String?)
}

/// Default pure-Swift implementation. Constructed from a list of `ZimReader`s
/// supplied by the host app — the reader implementations are where libzim
/// (via CoreKiwix.xcframework or a pure-Swift port) actually lives.
public actor DefaultZimService: ZimService {
    private let readers: [(name: String, reader: ZimReader)]
    private var graphs: [String: SZRGGraph] = [:]
    private var chunks: [String: [String: [[String: Any]]]] = [:]
    private var manifests: [String: [String: Int]] = [:]
    /// Cached streetzim bbox (minLat, minLon, maxLat, maxLon), loaded
    /// lazily from `streetzim-meta.json`. `nil` entry means "tried and
    /// the file wasn't there" — older streetzims don't ship the meta.
    private var bboxes: [String: (minLat: Double, minLon: Double, maxLat: Double, maxLon: Double)?] = [:]

    /// Optional log sink the host sets to surface slow-step progress in the
    /// UI debug pane. Thread-safe on the actor.
    public var logger: (@Sendable (String) -> Void)?

    public init(readers: [(name: String, reader: ZimReader)]) {
        self.readers = readers
    }

    public func setLogger(_ logger: (@Sendable (String) -> Void)?) {
        self.logger = logger
    }

    private func log(_ msg: String) {
        logger?(msg)
    }

    private func timed<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
        let t0 = Date()
        let memBefore = Double(MemoryStats.physFootprintBytes()) / 1_048_576
        let result = try block()
        let dt = Date().timeIntervalSince(t0)
        let memAfter = Double(MemoryStats.physFootprintBytes()) / 1_048_576
        log(String(format: "%@ · %.2fs · Δmem=%+.1f MB", label, dt, memAfter - memBefore))
        return result
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
            if let entry = try? pair.reader.read(path: path) {
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

    /// Parse an article into ordered sections and return their
    /// titles (ready to be shown to the user or to the model before
    /// asking it to pick sections to read).
    public func articleSections(path: String, zim: String?) async throws -> (zim: String, title: String, sections: [ArticleSection]) {
        let article = try await article(path: path, zim: zim)
        let sections = ArticleSections.parse(html: article.text)
        return (article.zim, article.title, sections)
    }

    /// Fetch just one section of an article. Keeps prompts tiny and
    /// KV-cache allocation predictable regardless of how large the
    /// underlying article is.
    public func articleSection(path: String, section: String, zim: String?) async throws -> (zim: String, title: String, section: ArticleSection) {
        let parsed = try await articleSections(path: path, zim: zim)
        guard let hit = ArticleSections.find(section, in: parsed.sections) else {
            throw ZimServiceError.notFound("section \"\(section)\" in \(path)")
        }
        return (parsed.zim, parsed.title, hit)
    }

    public func mainPage(zim: String?) async throws -> [ArticleResult] {
        var out: [ArticleResult] = []
        for pair in readers {
            if let zim, pair.name != zim { continue }
            if let entry = try? pair.reader.readMainPage() {
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
        try await geocodeResolved(query: query, limit: limit, zim: zim, kinds: kinds).map(\.place)
    }

    /// Same as `geocode` but also tells the caller which streetzim produced
    /// each hit. `nearNamedPlace` uses this so the follow-on `nearPlaces`
    /// call runs against the zim that actually matched — otherwise a nil
    /// `zim` arg makes `pickStreetzim` fall back to `candidates.first`,
    /// which is almost never the right one.
    private func geocodeResolved(query: String, limit: Int, zim: String?, kinds: [String]?) async throws
        -> [(place: Place, zim: String)]
    {
        // If a valid streetzim is pinned, use only it; otherwise try every
        // loaded streetzim until one returns results. Models routinely
        // guess the wrong zim (`wikipedia_en_*.zim`) — we recover by
        // fanning out instead of resolving to an empty answer.
        let candidates: [(name: String, reader: ZimReader)]
        if let zim, let match = streetzimReaders.first(where: { $0.name == zim }) {
            candidates = [match]
        } else {
            if let zim, !zim.isEmpty {
                log("geocode: '\(zim)' is not a loaded streetzim; fanning out across \(streetzimReaders.count) streetzim(s)")
            }
            candidates = streetzimReaders
        }
        guard !candidates.isEmpty else { throw ZimServiceError.noStreetzim }

        let prefix = Geocoder.normalizePrefix(query)
        let filterSet = kinds.map(Set.init)
        for pair in candidates {
            let manifest = try loadManifest(pair: pair)
            if !manifest.isEmpty && manifest[prefix] == nil { continue }
            let records = try loadChunk(pair: pair, prefix: prefix)
            let ranked = Geocoder.rank(records: records, query: query, limit: limit, kinds: filterSet)
            if !ranked.isEmpty { return ranked.map { ($0, pair.name) } }
        }
        return []
    }

    /// Find places within `radiusKm` of `(lat, lon)`. Scans the streetzim's
    /// prefix-chunked place index in full on first call (results are cached
    /// per-ZIM inside the existing `chunks` map). Returns up to `limit`
    /// nearest matches, optionally filtered by OSM-style kinds (e.g.
    /// `["amenity"]` or subtype strings like `["restaurant", "cafe"]`).
    public func nearPlaces(
        lat: Double, lon: Double,
        radiusKm: Double,
        limit: Int,
        kinds: [String]?,
        zim: String?
    ) async throws -> NearPlacesResult {
        guard let pair = try pickStreetzim(zim, containing: (lat: lat, lon: lon)) else {
            throw ZimServiceError.noStreetzim
        }
        // Bbox guard: if the zim advertises coverage and the query point
        // falls outside, bail out before loading any chunk. This is the
        // difference between a 4-second / 1 GB no-op scan of a country-
        // scale ZIM and a clean empty answer.
        if let bbox = loadBBox(pair: pair), !bboxContains(bbox, lat: lat, lon: lon) {
            log("nearPlaces: (\(lat), \(lon)) is outside \(pair.name) bbox — returning empty")
            return NearPlacesResult(totalInRadius: 0, breakdown: [:], results: [])
        }
        // Default filter: `poi` + `place` when the caller didn't pin a
        // specific set. "What's around X" almost never means "addresses"
        // or "street names" (which swamp the real answers in OSM data),
        // so we opt those out by default. Explicit `kinds: ["addr"]`
        // still works when the caller actually wants addresses.
        let effectiveKinds: Set<String>
        if let kinds, !kinds.isEmpty {
            effectiveKinds = Set(kinds.map { $0.lowercased() })
        } else {
            effectiveKinds = ["poi", "place"]
        }
        let radiusM = radiusKm * 1000
        var hits: [(Place, Double)] = []

        // Preferred fast path: streetzim-generator commit a485ce3+ ships
        // `category-index/<slug>.json` which pre-groups records by OSM
        // top-level type. Reading a handful of these files beats scanning
        // every prefix chunk — saves ~1 GB resident and seconds of latency
        // on country-scale ZIMs.
        //
        // `kinds` may arrive at *either* abstraction level:
        //   • top-level OSM keys (`amenity`, `tourism`, …) — match one slug.
        //   • subtypes (`cafe`, `restaurant`, …) — don't match any slug
        //     directly, so we load a broad set of likely-POI categories and
        //     let scanRecords filter down by subtype.
        if let catManifest = loadCategoryManifest(pair: pair),
           let categories = catManifest["categories"] as? [String: Any]
        {
            let available = Set(categories.keys.map { $0.lowercased() })
            let directHits = effectiveKinds.intersection(available)
            let slugs: Set<String>
            let applyKindFilter: Bool
            if !directHits.isEmpty {
                slugs = directHits
                applyKindFilter = false
            } else {
                // Load the POI-ish slice of the category manifest and
                // filter by subtype in-memory. If the manifest doesn't
                // expose any of these, we'll fall through to the prefix
                // scan below.
                let poiish: Set<String> = ["amenity", "tourism", "shop", "leisure", "historic", "poi", "place"]
                slugs = poiish.intersection(available)
                applyKindFilter = true
            }
            if !slugs.isEmpty {
                let joined = slugs.sorted().joined(separator: ",")
                log("nearPlaces via category-index: \(joined) in \(pair.name)")
                for slug in slugs.sorted() {
                    guard let recs = loadCategoryChunk(pair: pair, slug: slug) else { continue }
                    scanRecords(recs, filter: effectiveKinds, applyKindFilter: applyKindFilter,
                                centerLat: lat, centerLon: lon,
                                radiusMeters: radiusM, hits: &hits)
                }
                return summarize(hits: hits, limit: limit)
            }
        }

        // Fallback: scan the prefix-chunked search-data. Works with older
        // streetzims that predate the category index.
        let manifest = try loadManifest(pair: pair)
        log("nearPlaces full scan: \(manifest.count) chunk(s) in \(pair.name)")
        let prefixes = manifest.isEmpty ? [] : Array(manifest.keys)
        for prefix in prefixes {
            let records = try loadChunk(pair: pair, prefix: prefix)
            scanRecords(records, filter: effectiveKinds, applyKindFilter: true,
                        centerLat: lat, centerLon: lon,
                        radiusMeters: radiusM, hits: &hits)
        }
        return summarize(hits: hits, limit: limit)
    }

    /// Compute breakdown-by-subtype + top-N-by-distance from an
    /// in-radius hit list. Subtype is preferred (e.g. "cafe", "bar");
    /// if a record is subtype-less we fall back to its kind.
    private func summarize(hits: [(Place, Double)], limit: Int) -> NearPlacesResult {
        var breakdown: [String: Int] = [:]
        for pair in hits {
            let key = pair.0.subtype.isEmpty ? pair.0.kind : pair.0.subtype
            breakdown[key, default: 0] += 1
        }
        let sorted = hits.sorted { $0.1 < $1.1 }
        let top = Array(sorted.prefix(max(1, limit)))
        return NearPlacesResult(
            totalInRadius: hits.count,
            breakdown: breakdown,
            results: top
        )
    }

    private func scanRecords(
        _ records: [[String: Any]],
        filter: Set<String>, applyKindFilter: Bool,
        centerLat: Double, centerLon: Double,
        radiusMeters: Double,
        hits: inout [(Place, Double)]
    ) {
        for rec in records {
            guard let rlat = (rec["a"] as? Double) ?? (rec["lat"] as? Double),
                  let rlon = (rec["o"] as? Double) ?? (rec["lon"] as? Double)
            else { continue }
            let d = haversineMeters(centerLat, centerLon, rlat, rlon)
            guard d <= radiusMeters else { continue }
            if applyKindFilter {
                let kind = ((rec["t"] as? String) ?? (rec["type"] as? String) ?? "").lowercased()
                let subtype = ((rec["s"] as? String) ?? (rec["subtype"] as? String) ?? "").lowercased()
                if !filter.contains(kind) && !filter.contains(subtype) { continue }
            }
            let p = Place(
                name: (rec["n"] as? String) ?? (rec["name"] as? String) ?? "",
                kind: (rec["t"] as? String) ?? (rec["type"] as? String) ?? "",
                lat: rlat, lon: rlon,
                subtype: (rec["s"] as? String) ?? (rec["subtype"] as? String) ?? "",
                location: (rec["l"] as? String) ?? (rec["location"] as? String) ?? "",
                wiki: rec["w"] as? String,
                wikidata: rec["q"] as? String
            )
            hits.append((p, d))
        }
    }

    // MARK: - Category-index helpers (streetzim ≥ a485ce3)

    private func loadCategoryManifest(pair: (name: String, reader: ZimReader)) -> [String: Any]? {
        let cacheKey = "__cat_manifest__\(pair.name)"
        if let cached = manifests[cacheKey] {
            // Stored as a sentinel non-empty-only manifest just to cache
            // "yes we tried once"; convert back by re-fetching contents.
            _ = cached // unused, we re-read below; keeping the flag is cheap.
        }
        guard let entry = try? pair.reader.read(path: "category-index/manifest.json"),
              let json = try? JSONSerialization.jsonObject(with: entry.content) as? [String: Any]
        else { return nil }
        return json
    }

    private func loadCategoryChunk(pair: (name: String, reader: ZimReader), slug: String) -> [[String: Any]]? {
        let cacheKey = "__cat__:\(slug)"
        if let cached = chunks[pair.name]?[cacheKey] { return cached }
        guard let entry = try? pair.reader.read(path: "category-index/\(slug).json"),
              let decoded = (try? JSONSerialization.jsonObject(with: entry.content)) as? [[String: Any]]
        else { return nil }
        var byPrefix = chunks[pair.name] ?? [:]
        byPrefix[cacheKey] = decoded
        chunks[pair.name] = byPrefix
        return decoded
    }

    /// The union of category slugs exposed by every loaded streetzim —
    /// drawn from `category-index/manifest.json` when present. Used by
    /// tool adapters to pin the `kinds` parameter to a known vocabulary
    /// so the model doesn't invent slugs or pass `kinds=[]`.
    public func categoryVocabulary() -> [String] {
        var set: Set<String> = []
        for pair in streetzimReaders {
            if let manifest = loadCategoryManifest(pair: pair),
               let categories = manifest["categories"] as? [String: Any]
            {
                for key in categories.keys { set.insert(key.lowercased()) }
            }
        }
        return set.sorted()
    }

    /// Return the streetzim `streetzim-meta.json` block (if present) for
    /// each loaded streetzim — or for just the named one. Streetzims
    /// built before generator commit a485ce3 don't ship this file; those
    /// entries are omitted rather than failing the call.
    public func zimInfo(zim: String?) async throws -> [[String: Any]] {
        let targets = readers.filter { zim == nil || $0.name == zim }
        var out: [[String: Any]] = []
        for pair in targets {
            guard let entry = try? pair.reader.read(path: "streetzim-meta.json"),
                  let json = try? JSONSerialization.jsonObject(with: entry.content) as? [String: Any]
            else { continue }
            var row = json
            row["zim"] = pair.name
            out.append(row)
        }
        return out
    }

    /// One-shot "what's near <text place>" convenience: geocode the
    /// place then nearPlaces at its lat/lon. The model can now call this
    /// with a single utterance-sized argument and get a useful answer
    /// without needing to chain tools itself.
    public func nearNamedPlace(
        place: String, radiusKm: Double, limit: Int,
        kinds: [String]?, zim: String?
    ) async throws -> (resolved: Place, result: NearPlacesResult) {
        let hits = try await geocodeResolved(query: place, limit: 1, zim: zim, kinds: nil)
        guard let first = hits.first else { throw ZimServiceError.noMatch(place) }
        // Pin `nearPlaces` to the zim that resolved the name. Without this,
        // a nil/stale `zim` would send the follow-up scan against the wrong
        // streetzim and — on older ZIMs without a category-index — load
        // every prefix chunk (gigabytes, seconds).
        let result = try await nearPlaces(
            lat: first.place.lat, lon: first.place.lon,
            radiusKm: radiusKm, limit: limit,
            kinds: kinds, zim: first.zim
        )
        return (first.place, result)
    }

    public func routeFromPlaces(origin: String, destination: String, zim: String?) async throws
        -> (resolved: (origin: Place, destination: Place), route: Route, zimUsed: String?)
    {
        let streetzimNames = Set(streetzimReaders.map(\.name))
        let resolvedPreference: String? = {
            guard let zim, !zim.isEmpty else { return nil }
            if streetzimNames.contains(zim) { return zim }
            log("ignoring zim='\(zim)' (no matching streetzim loaded); trying all")
            return nil
        }()

        if let zim = resolvedPreference {
            return try await resolveAndRoute(origin: origin, destination: destination, zim: zim)
        }
        let candidates = streetzimReaders
        guard !candidates.isEmpty else { throw ZimServiceError.noStreetzim }
        var lastError: Error?
        for pair in candidates {
            do {
                log("trying streetzim \(pair.name) for \(origin) → \(destination)")
                return try await resolveAndRoute(origin: origin, destination: destination, zim: pair.name)
            } catch {
                lastError = error
                continue
            }
        }
        throw lastError ?? ZimServiceError.noMatch("\(origin) / \(destination)")
    }

    private func resolveAndRoute(origin: String, destination: String, zim: String) async throws
        -> (resolved: (origin: Place, destination: Place), route: Route, zimUsed: String?)
    {
        let oHits = try await geocode(query: origin, limit: 1, zim: zim, kinds: nil)
        let dHits = try await geocode(query: destination, limit: 1, zim: zim, kinds: nil)
        guard let o = oHits.first else { throw ZimServiceError.noMatch(origin) }
        guard let d = dHits.first else { throw ZimServiceError.noMatch(destination) }
        let route = try await planDrivingRoute(RouteRequest(
            originLat: o.lat, originLon: o.lon,
            destLat: d.lat, destLon: d.lon,
            zim: zim
        ))
        return ((origin: o, destination: d), route: route, zimUsed: zim)
    }

    // MARK: - Internal loaders

    private func pickStreetzim(_ zim: String?) throws -> (name: String, reader: ZimReader)? {
        try pickStreetzim(zim, containing: nil)
    }

    /// Like `pickStreetzim(_:)` but prefers a streetzim whose bbox
    /// contains `coord` when no valid zim name was passed. This avoids
    /// running a bare-lat/lon query against the alphabetically-first zim
    /// when a better-fitting one is loaded.
    private func pickStreetzim(
        _ zim: String?,
        containing coord: (lat: Double, lon: Double)?
    ) throws -> (name: String, reader: ZimReader)? {
        let candidates = streetzimReaders
        if candidates.isEmpty { return nil }
        if let name = zim, !name.isEmpty {
            if let match = candidates.first(where: { $0.name == name }) {
                return match
            }
            log("ignoring zim='\(name)' for streetzim tool (no matching streetzim loaded)")
        }
        if let coord {
            if let match = candidates.first(where: { pair in
                guard let bbox = loadBBox(pair: pair) else { return false }
                return bboxContains(bbox, lat: coord.lat, lon: coord.lon)
            }) {
                return match
            }
        }
        return candidates.first
    }

    /// True when (lat, lon) falls within the inclusive bbox. Does not
    /// handle antimeridian crossing — streetzim bboxes don't today.
    private func bboxContains(
        _ box: (minLat: Double, minLon: Double, maxLat: Double, maxLon: Double),
        lat: Double, lon: Double
    ) -> Bool {
        lat >= box.minLat && lat <= box.maxLat &&
            lon >= box.minLon && lon <= box.maxLon
    }

    private func loadBBox(pair: (name: String, reader: ZimReader))
        -> (minLat: Double, minLon: Double, maxLat: Double, maxLon: Double)?
    {
        if let cached = bboxes[pair.name] { return cached }
        guard let entry = try? pair.reader.read(path: "streetzim-meta.json"),
              let json = try? JSONSerialization.jsonObject(with: entry.content) as? [String: Any],
              let bb = json["bbox"] as? [String: Any],
              let minLat = (bb["minLat"] as? Double) ?? (bb["min_lat"] as? Double),
              let minLon = (bb["minLon"] as? Double) ?? (bb["min_lon"] as? Double),
              let maxLat = (bb["maxLat"] as? Double) ?? (bb["max_lat"] as? Double),
              let maxLon = (bb["maxLon"] as? Double) ?? (bb["max_lon"] as? Double)
        else {
            bboxes[pair.name] = .some(nil)
            return nil
        }
        let box = (minLat: minLat, minLon: minLon, maxLat: maxLat, maxLon: maxLon)
        bboxes[pair.name] = .some(box)
        return box
    }

    private func loadGraph(pair: (name: String, reader: ZimReader)) throws -> SZRGGraph {
        if let cached = graphs[pair.name] { return cached }
        log("loading routing-data/graph.bin from \(pair.name)…")
        let entry = try timed("read graph.bin") {
            try pair.reader.read(path: "routing-data/graph.bin")
        }
        guard let entry else { throw ZimServiceError.noStreetzim }
        log(String(format: "graph.bin = %.1f MB, parsing (skip geoms)…", Double(entry.content.count) / 1_048_576))
        // Skip per-edge polyline decoding — A* only reads node positions
        // and edge distances. Saves ~600 MB on country-scale graphs. Any
        // client that wants precise polylines can reparse with decodeGeoms=true.
        let g = try timed("parse graph") { try SZRGGraph.parse(entry.content, decodeGeoms: false) }
        graphs[pair.name] = g
        return g
    }

    private func loadManifest(pair: (name: String, reader: ZimReader)) throws -> [String: Int] {
        if let cached = manifests[pair.name] { return cached }
        log("loading search-data/manifest.json from \(pair.name)…")
        guard let entry = try pair.reader.read(path: "search-data/manifest.json") else {
            manifests[pair.name] = [:]
            return [:]
        }
        let parsed = (try? JSONSerialization.jsonObject(with: entry.content)) as? [String: Any]
        let chunks = (parsed?["chunks"] as? [String: Int]) ?? [:]
        manifests[pair.name] = chunks
        return chunks
    }

    private func loadChunk(pair: (name: String, reader: ZimReader), prefix: String) throws -> [[String: Any]] {
        if let cached = chunks[pair.name]?[prefix] { return cached }
        log("loading search-data/\(prefix).json from \(pair.name)…")
        guard let entry = try pair.reader.read(path: "search-data/\(prefix).json") else {
            return []
        }
        let parsed = (try? JSONSerialization.jsonObject(with: entry.content)) as? [[String: Any]] ?? []
        var byPrefix = chunks[pair.name] ?? [:]
        byPrefix[prefix] = parsed
        chunks[pair.name] = byPrefix
        return parsed
    }
}
