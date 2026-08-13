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
    public init(
        zim: String, kind: ZimKind, path: String,
        title: String, snippet: String
    ) {
        self.zim = zim
        self.kind = kind
        self.path = path
        self.title = title
        self.snippet = snippet
    }
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

/// Cheap lat/lon rejection window for a radius scan, derived once so the
/// per-record test is two subtractions instead of haversine's six
/// transcendentals.
///
/// It must never reject a record `haversineMeters` would have accepted, so
/// every bound is derived from that exact function (sphere R = 6 371 000 m,
/// `d = 2R·asin(√a)`, `a = sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2)`):
///
/// * Latitude — `sin²(Δφ/2) ≤ a ≤ sin²(d/2R)`, and both half-angles live in
///   [0, π/2] where sin is increasing, so `|Δφ| ≤ d/R` exactly. That is
///   `d / 111 194.9` degrees; dividing by 111 000 instead can only widen the
///   window.
/// * Longitude — `cos φ₁ cos φ₂ sin²(Δλ/2) ≤ a`, and once the latitude gate
///   has passed, both latitudes are within `Δφmax` of the centre, so
///   `√(cos φ₁ cos φ₂) ≥ cos(|centreLat| + Δφmax)`. Hence
///   `|sin(Δλ/2)| ≤ sin(d/2R) / cos(|centreLat| + Δφmax)`. When that ratio
///   reaches 1 — a window touching a pole, or a radius near half the globe —
///   longitude is unconstrained and the gate turns itself off.
///
/// Both gates fail open: a NaN/absurd radius or coordinate compares false and
/// falls through to the haversine, which decides as it always did.
struct RadiusBoundingBox {
    private let centerLat: Double
    private let centerLon: Double
    private let maxLatDelta: Double
    private let maxLonDelta: Double

    init(centerLat: Double, centerLon: Double, radiusMeters: Double) {
        var latDelta = Double.infinity
        var lonDelta = Double.infinity
        // 111 000 < the sphere's 111 194.9 m per degree of latitude, so this
        // over-estimates Δφmax rather than under-estimating it.
        let degrees = radiusMeters / 111_000.0
        if degrees.isFinite, degrees >= 0, degrees < 180 {
            latDelta = degrees
            let worstLat = min(90.0, abs(centerLat) + degrees)
            let cosWorst = cos(worstLat * .pi / 180)
            let halfChord = sin(radiusMeters / (2 * 6_371_000.0))
            // `degrees < 180` keeps radiusMeters/(2R) inside sin's increasing
            // branch, so halfChord is a genuine bound and not a fold-back.
            if cosWorst > 1e-9, halfChord.isFinite, halfChord >= 0 {
                let ratio = halfChord / cosWorst
                if ratio < 1 {
                    // A hair of slack absorbs libm rounding in sin/asin — the
                    // window may only ever be too wide.
                    lonDelta = 2 * asin(ratio) * 180 / .pi * 1.000_001
                }
            }
        }
        self.centerLat = centerLat
        self.centerLon = centerLon
        self.maxLatDelta = latDelta
        self.maxLonDelta = lonDelta
    }

    @inline(__always)
    func mayBeWithin(lat: Double, lon: Double) -> Bool {
        if abs(lat - centerLat) > maxLatDelta { return false }
        // Antimeridian: two points a degree apart across ±180 differ by ~359
        // in raw longitude, and rejecting them would drop real in-radius hits.
        var deltaLon = abs(lon - centerLon)
        if deltaLon > 180 { deltaLon = 360 - deltaLon }
        return !(deltaLon > maxLonDelta)
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
    func articleByTitle(title: String, zim: String?, section: String?) async throws -> (zim: String, path: String, title: String, section: ArticleSection)
    func mainPage(zim: String?) async throws -> [ArticleResult]

    // Streetzim-only. Implementations may throw `.noStreetzim` if unavailable;
    // MCPZimServerKit uses those throws to decide whether to register the tool.
    func planDrivingRoute(_ req: RouteRequest) async throws -> Route
    func geocode(query: String, limit: Int, zim: String?, kinds: [String]?) async throws -> [Place]
    func nearPlaces(lat: Double, lon: Double, radiusKm: Double, limit: Int, kinds: [String]?, zim: String?, hasWiki: Bool) async throws -> NearPlacesResult
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
    private var spatialGraphs: [String: SpatialGraph] = [:]
    private var chunks: [String: [String: [[String: Any]]]] = [:]
    /// LRU bookkeeping for `chunks` (most-recently-used last) plus the total
    /// record count currently pinned. The cache used to be unbounded — every
    /// prefix a session's geocode/nearPlaces calls ever touched stayed
    /// resident forever, the exact monotonic growth observed jetsamming the
    /// app at 5.4 GB RSS. Eviction keeps repeat-query speed while bounding
    /// the pin to one full-scan's worth of records.
    private var chunkLRU: [(zim: String, prefix: String)] = []
    private var cachedChunkRecords = 0
    /// Parsed fan-out leaf shards, budgeted separately from `chunks` —
    /// see `loadLeafChunk`.
    private struct LeafKey: Hashable {
        let zim: String
        let leaf: String
    }
    private var leafChunks: [LeafKey: (records: [[String: Any]], bytes: Int)] = [:]
    private var leafLRU: [LeafKey] = []
    private var cachedLeafBytes = 0
    private var manifests: [String: [String: Int]] = [:]
    /// zim → parsed category-index/manifest.json.
    private var categoryManifests: [String: [String: Any]] = [:]
    /// zim → prefix → fan-out leaf chunk names (`manifest.sub_chunks`).
    private var subChunkMaps: [String: [String: [String]]] = [:]
    /// Cached streetzim bbox (minLat, minLon, maxLat, maxLon), loaded
    /// lazily from `streetzim-meta.json`. `nil` entry means "tried and
    /// the file wasn't there" — older streetzims don't ship the meta.
    private var bboxes: [String: (minLat: Double, minLon: Double, maxLat: Double, maxLon: Double)?] = [:]
    /// Last few article bodies + parsed sections — see `ArticleCache`.
    private let articleCache = ArticleCache()
    /// Search-snippet LRU: one rendered lead line per (zim, path,
    /// maxChars). Search over-fetches candidates (limit×2 per variant)
    /// and the title / FTS / kind-fallback passes overlap heavily
    /// within a turn, so the same lead was stripped many times per
    /// query. Values are ≤ maxChars characters — the whole cache is
    /// bytes, not MB.
    private var snippetCache: [String: String] = [:]
    private var snippetLRU: [String] = []
    private static let maxCachedSnippets = 64

    /// Optional log sink the host sets to surface slow-step progress in the
    /// UI debug pane. Thread-safe on the actor.
    public var logger: (@Sendable (String) -> Void)?

    public init(readers: [(name: String, reader: ZimReader)]) {
        self.readers = readers
    }

    public func setLogger(_ logger: (@Sendable (String) -> Void)?) {
        self.logger = logger
    }

    /// Pre-load the routing graph + search manifest for every loaded
    /// streetzim, off the hot path. The first "directions to X" query
    /// otherwise pays a ~1.2 s graph.bin read+parse — call this at
    /// app start so that cost lands while the user is still reading
    /// the empty-state. Safe to call multiple times; each graph load
    /// is cached after the first hit.
    public func prewarmStreetzims() async {
        for pair in streetzimReaders {
            _ = try? loadManifest(pair: pair)
            _ = try? loadGraph(pair: pair)
        }
    }

    private func log(_ msg: String) {
        logger?(msg)
    }

    /// Defence-in-depth guard on the streetzim `w` record field.
    ///
    /// Contract: `w` is a Wikipedia tag (`en:HP_Garage`,
    /// `fr:Tour_Eiffel`), never a URL. A pre-fa6208b bug in
    /// streetzim's Overture-places enrichment wrote POI website URLs
    /// into `w` for any POI that had no OSM wiki tag. Every
    /// downstream consumer (`articleByTitle`, `near_places(hasWiki)`,
    /// the `"wikipedia"` field we forward to the LLM) treats that
    /// value as a title-shaped tag — a URL there mismatches the tag
    /// parser, makes `has_wiki` queries false-positive the record,
    /// and ultimately drops the excerpt silently.
    ///
    /// streetzim has been fixed (field renamed to `ws`), but an
    /// older ZIM file left on a user's device would still carry the
    /// collision. Strip any value that contains `://` at ingest so
    /// those stale records degrade gracefully to "no wiki tag" instead
    /// of misbehaving. Everything that's not URL-shaped — plain
    /// titles, language-prefixed tags, underscored or spaced —
    /// passes through unchanged.
    static func sanitizedWikiTag(_ raw: String?) -> String? {
        guard let raw, !raw.isEmpty else { return nil }
        if raw.contains("://") { return nil }
        return raw
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
        // Natural-language queries ("origin of pizza", "why is plasma
        // important") score poorly on libzim's bare BM25 — we see
        // `Pizza Hut` and Wikipedia admin pages at the top instead of
        // `History of pizza` / `Plasma (physics)`. Fix by running
        // several passes and merging:
        //   1. Title-suggest on the keyword core.
        //   2. FTS on each query variant (reformulated phrasings).
        // Title hits go first so the semantic reranker (applied
        // downstream in ChatSession) has better candidates to pick
        // from. Wikipedia-namespace noise pages are filtered out.
        let variants = Self.queryVariants(of: query)
        let keywordQuery = Self.keywordCore(of: query)
        var results: [SearchHitResult] = []
        var seen = Set<String>()
        let overfetch = max(limit * 2, 10)
        for pair in readers {
            if results.count >= limit { break }
            if let wanted = kind, pair.reader.kind != wanted { continue }
            let titleHits = (try? pair.reader.searchTitles(
                query: keywordQuery, limit: overfetch)) ?? []
            for h in titleHits {
                if results.count >= limit { break }
                if Self.isNoisePath(h.path) { continue }
                let key = "\(pair.name)\t\(h.path)"
                if seen.contains(key) { continue }
                seen.insert(key)
                let snippet = leadSnippet(from: pair.reader, zim: pair.name, path: h.path, maxChars: 220)
                results.append(SearchHitResult(
                    zim: pair.name, kind: pair.reader.kind,
                    path: h.path, title: h.title, snippet: snippet
                ))
            }
            for variant in variants {
                if results.count >= limit { break }
                let ftsHits = (try? pair.reader.search(query: variant, limit: overfetch)) ?? []
                for h in ftsHits {
                    if results.count >= limit { break }
                    if Self.isNoisePath(h.path) { continue }
                    let key = "\(pair.name)\t\(h.path)"
                    if seen.contains(key) { continue }
                    seen.insert(key)
                    let snippet = leadSnippet(from: pair.reader, zim: pair.name, path: h.path, maxChars: 220)
                    results.append(SearchHitResult(
                        zim: pair.name, kind: pair.reader.kind,
                        path: h.path, title: h.title, snippet: snippet
                    ))
                }
            }
        }
        return results
    }

    /// Generate a couple of reformulated variants for natural-language
    /// queries so we can union the BM25 result sets. Empirically:
    ///   "origin of pizza" → ["origin of pizza", "pizza history", "pizza"]
    ///   "why is plasma important" → ["why is plasma important", "plasma", "plasma physics"]
    /// Keep the original first so the reranker still has the raw
    /// keyword signal.
    private static func queryVariants(of q: String) -> [String] {
        var out: [String] = [q]
        let lower = q.lowercased()
        func push(_ s: String) {
            let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty, !out.contains(t) { out.append(t) }
        }
        // "origin(s) of X" / "history of X" → "X history"
        if let m = lower.range(of: #"^(?:origin|origins|history)\s+of\s+(?:the\s+)?(.+)$"#,
                               options: .regularExpression) {
            let tail = String(lower[m]).replacingOccurrences(
                of: #"^(?:origin|origins|history)\s+of\s+(?:the\s+)?"#,
                with: "", options: .regularExpression
            )
            push(tail + " history")
            push(tail)
        }
        // "why is X (important|useful|significant)" / "how does X work" / "what is X"
        if let m = lower.range(of: #"^(?:why\s+is|how\s+does|what\s+is)\s+(.+?)(?:\s+(?:important|useful|significant|work|used\s+for))?$"#,
                               options: .regularExpression) {
            let inner = String(lower[m])
                .replacingOccurrences(of: #"^(?:why\s+is|how\s+does|what\s+is)\s+"#,
                                      with: "", options: .regularExpression)
                .replacingOccurrences(of: #"\s+(?:important|useful|significant|work|used\s+for)$"#,
                                      with: "", options: .regularExpression)
            push(inner)
        }
        // Always include a bare keyword-core fallback.
        let core = Self.keywordCore(of: q)
        if core != q { push(core) }
        return out
    }

    /// Strip Wikipedia's namespace pages (AfD, reference desk, etc.)
    /// — the search index happily returns them and they push real
    /// encyclopedic articles off the top.
    private static func isNoisePath(_ path: String) -> Bool {
        if path.hasPrefix("Wikipedia:") { return true }
        if path.hasPrefix("Wikipedia%3A") { return true }
        if path.hasPrefix("User:") { return true }
        if path.hasPrefix("Talk:") { return true }
        if path.hasPrefix("Help:") { return true }
        if path.hasPrefix("Portal:") { return true }
        if path.hasPrefix("Category:") { return true }
        if path.hasPrefix("Template:") { return true }
        if path.hasPrefix("File:") { return true }
        if path.hasPrefix("Special:") { return true }
        return false
    }

    /// Strip stopwords + common question prefixes so natural-language
    /// queries survive the title index. "origin of pizza" → "pizza
    /// origin"; "why is plasma important" → "plasma important"; "what
    /// is aspirin used for" → "aspirin used". Order-preserving.
    private static func keywordCore(of q: String) -> String {
        let stop: Set<String> = [
            "the", "a", "an", "and", "or", "of", "is", "are", "was",
            "were", "be", "been", "to", "for", "in", "on", "at",
            "with", "about", "why", "what", "how", "when", "where",
            "which", "who", "does", "do", "did", "can", "could",
            "would", "should", "me", "my", "i", "you", "your",
            "its", "it", "as", "by", "this", "that", "these", "those",
            "tell", "give", "show",
        ]
        let lowered = q.lowercased()
        let tokens = lowered.split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        let kept = tokens.filter { !stop.contains(String($0)) }
        let core = kept.joined(separator: " ").trimmingCharacters(in: .whitespaces)
        return core.isEmpty ? q : core
    }

    /// Grab the opening of an article body and collapse it to a
    /// single plain-text line. Used to populate search snippets —
    /// keeps the model from picking a tangentially-named hit.
    private func leadSnippet(from reader: ZimReader, zim: String, path: String, maxChars: Int) -> String {
        let cacheKey = "\(zim)\t\(path)\t\(maxChars)"
        if let cached = snippetCache[cacheKey] {
            touchSnippet(cacheKey)
            return cached
        }
        let snippet = renderLeadSnippet(from: reader, path: path, maxChars: maxChars)
        cacheSnippet(cacheKey, snippet)
        return snippet
    }

    private func renderLeadSnippet(from reader: ZimReader, path: String, maxChars: Int) -> String {
        guard let entry = try? reader.read(path: path) else { return "" }
        // Lead-only fast path: the lead lives before the first <h2>/<h3>,
        // so strip just that prefix instead of parsing (and strip-HTMLing)
        // EVERY section of the body — this runs per candidate hit on the
        // search hot path (overfetch ≈ limit×2 per variant).
        var lead = Self.leadPrefixHTML(of: entry.content)
            .map(ArticleSections.stripHTML) ?? ""
        if lead.isEmpty {
            // No prose before the first heading (or undecodable prefix):
            // fall back to the bounded full parse, whose first section is
            // then the first named one — matching the pre-fast-path output.
            guard let html = String(data: entry.content, encoding: .utf8)
            else { return "" }
            let cap = 64 * 1024
            let head = html.count > cap ? String(html.prefix(cap)) : html
            lead = ArticleSections.parse(html: head).first?.text ?? ""
        }
        if lead.isEmpty { return "" }
        let singleLine = lead
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
        if singleLine.count <= maxChars { return singleLine }
        return String(singleLine.prefix(maxChars)) + "…"
    }

    /// Raw-UTF-8 scan for the first `<h2`/`<h3`, returning the decoded
    /// prefix before it (capped — a `maxChars`-sized snippet never needs
    /// more). Nil when the prefix isn't valid UTF-8.
    static func leadPrefixHTML(of content: Data, capBytes: Int = 64 * 1024) -> String? {
        content.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> String? in
            let bytes = raw.bindMemory(to: UInt8.self)
            let limit = min(bytes.count, capBytes)
            var cut = limit
            if limit >= 3 {
                let lt = UInt8(ascii: "<")
                for i in 0..<(limit - 2) where bytes[i] == lt {
                    let tag = bytes[i + 1] | 0x20   // ASCII lowercase
                    guard tag == UInt8(ascii: "h"),
                          bytes[i + 2] == UInt8(ascii: "2")
                              || bytes[i + 2] == UInt8(ascii: "3")
                    else { continue }
                    cut = i
                    break
                }
            }
            // Never split a multi-byte UTF-8 sequence at the byte cap.
            while cut > 0, cut < bytes.count, bytes[cut] & 0xC0 == 0x80 {
                cut -= 1
            }
            guard cut > 0 else { return "" }
            return String(bytes: bytes[0..<cut], encoding: .utf8)
        }
    }

    private func touchSnippet(_ key: String) {
        guard let idx = snippetLRU.firstIndex(of: key),
              idx != snippetLRU.count - 1
        else { return }
        snippetLRU.append(snippetLRU.remove(at: idx))
    }

    private func cacheSnippet(_ key: String, _ snippet: String) {
        if snippetCache.updateValue(snippet, forKey: key) == nil {
            snippetLRU.append(key)
            while snippetLRU.count > Self.maxCachedSnippets {
                snippetCache.removeValue(forKey: snippetLRU.removeFirst())
            }
        } else {
            touchSnippet(key)
        }
    }

    public func article(path: String, zim: String?) async throws -> ArticleResult {
        let targets = readers.filter { zim == nil || $0.name == zim }
        if targets.isEmpty { throw ZimServiceError.unknownZim(zim ?? "<any>") }
        for pair in targets {
            if let cached = await cachedArticle(pair: pair, path: path) {
                return ArticleResult(
                    zim: pair.name,
                    path: cached.path,
                    title: cached.title,
                    mimetype: cached.mimetype,
                    text: cached.html,
                    bytes: cached.bytes
                )
            }
        }
        throw ZimServiceError.notFound(path)
    }

    /// Cache-through read of one article body; nil when `path` isn't
    /// in this reader.
    private func cachedArticle(
        pair: (name: String, reader: ZimReader), path: String
    ) async -> ArticleCache.Entry? {
        if let hit = await articleCache.entry(zim: pair.name, path: path) {
            return hit
        }
        guard let entry = try? pair.reader.read(path: path) else { return nil }
        let stored = ArticleCache.Entry(
            path: entry.path,
            title: entry.title,
            mimetype: entry.mimetype,
            html: String(data: entry.content, encoding: .utf8) ?? "",
            bytes: entry.content.count,
            sections: nil
        )
        await articleCache.store(zim: pair.name, path: path, entry: stored)
        return stored
    }

    /// Parsed sections for an already-fetched body, reusing (and
    /// back-filling) the article cache — so a fetch+parse pair like
    /// `sectionsByTitle` costs one read and one parse instead of two
    /// of each.
    private func cachedSections(zim: String, path: String, html: String) async -> [ArticleSection] {
        if let cached = await articleCache.entry(zim: zim, path: path)?.sections {
            return cached
        }
        let sections = ArticleSections.parse(html: html)
        await articleCache.setSections(sections, zim: zim, path: path)
        return sections
    }

    /// Parse an article into ordered sections and return their
    /// titles (ready to be shown to the user or to the model before
    /// asking it to pick sections to read).
    public func articleSections(path: String, zim: String?) async throws -> (zim: String, title: String, sections: [ArticleSection]) {
        let article = try await article(path: path, zim: zim)
        let sections = await cachedSections(zim: article.zim, path: path, html: article.text)
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

    /// Look up a Wikipedia-family article by title. Accepts both a
    /// bare title ("HP Garage") and the OSM-style wiki tag
    /// ("en:HP Garage") that the streetzim stores on each POI.
    /// Searches the ZIM's title index (libzim `suggestTitles`) which
    /// handles redirects and approximate matches. Default `section`
    /// is "lead" — a reasonable summary.
    /// Capitalise the first letter of each space-separated word,
    /// preserving the case of all non-initial letters. Used by
    /// `articleByTitle` to build Title Case direct-path candidates
    /// without running through `String.capitalized` (which lowercases
    /// interior letters and breaks acronyms like "iPhone" → "Iphone").
    static func wordCapitalize(_ s: String) -> String {
        s.split(separator: " ", omittingEmptySubsequences: false)
            .map { w -> String in
                guard let first = w.first else { return String(w) }
                return String(first).uppercased() + w.dropFirst()
            }
            .joined(separator: " ")
    }

    private static let wikipediaLanguagePrefixes: Set<String> = [
        "ar", "cs", "da", "de", "el", "en", "es", "fi", "fr", "he",
        "hi", "hu", "id", "it", "ja", "ko", "nl", "no", "pl", "pt",
        "ro", "ru", "sv", "th", "tr", "uk", "vi", "zh",
    ]

    public func articleByTitle(title: String, zim: String?, section: String? = "lead")
        async throws -> (zim: String, path: String, title: String, section: ArticleSection)
    {
        // Strip language prefix if present (e.g. "en:HP Garage" → "HP Garage").
        let cleanedTitle: String = {
            if let r = title.range(of: ":"), r.lowerBound != title.startIndex {
                let prefix = String(title[..<r.lowerBound])
                // Strip only a known lowercase Wikipedia language tag. A
                // generic 2–3 letter rule corrupts legitimate titles such as
                // "TV: The Movie" and "US: The Book".
                if Self.wikipediaLanguagePrefixes.contains(prefix) {
                    return String(title[r.upperBound...])
                }
            }
            return title
        }()
        // Wikipedia titles in OSM use underscores for spaces; libzim
        // suggest accepts either, but let's also prepare a spaced form.
        let withSpaces = cleanedTitle.replacingOccurrences(of: "_", with: " ")

        // Candidate readers — Wikipedia-family for the full article set,
        // plus any streetzim that bundled the article inline. A streetzim
        // copy matters OFFLINE: kiwix can't deep-link across ZIMs, so a
        // streetzim that wants narratable articles carries its own at
        // `wiki-article/<Title>` (streetzim --bundle-wiki-articles, option B).
        let candidates: [(name: String, reader: ZimReader)] = readers.filter { pair in
            guard pair.reader.kind == .wikipedia || pair.reader.kind == .mdwiki else { return false }
            if let zim, pair.name != zim { return false }
            return true
        }
        let bundled: [(name: String, reader: ZimReader)] = readers.filter { pair in
            guard pair.reader.kind == .streetzim else { return false }
            if let zim, pair.name != zim { return false }
            return true
        }
        guard !candidates.isEmpty || !bundled.isEmpty else {
            throw ZimServiceError.notFound("no Wikipedia or streetzim article ZIM loaded")
        }

        // Wikipedia ZIMs store articles at predictable paths derived
        // from the title — usually `A/Title_With_Underscores` (classic
        // Kiwix layout) or just `Title_With_Underscores`. Since the
        // OSM `wikipedia=` tag is the actual Wikipedia article title,
        // we can go directly from tag → path without an index lookup.
        // Two orders of magnitude faster than `searchTitles` and
        // guaranteed to hit the exact article (no fuzzy-match drift).
        // Fall back to `searchTitles` only if every direct-path
        // variant misses.
        let underscored = withSpaces.replacingOccurrences(of: " ", with: "_")
        // Wikipedia canonicalises article paths in Title Case
        // (`A/North_Korea`, not `A/north_korea`). Voice input and the
        // lowercase fast-path titles both miss the raw-case paths, and
        // the `searchTitles` suggester fallback below is slower + has
        // been observed to miss on big ZIMs. Try Title Case variants
        // up-front so "north korea" still resolves without paying the
        // suggester round-trip.
        let titleCased = Self.wordCapitalize(withSpaces)
        let titleCasedUnderscored = titleCased.replacingOccurrences(of: " ", with: "_")
        let titleForms = [underscored, withSpaces, titleCasedUnderscored, titleCased]
        // Wikipedia/mdwiki store at `A/<Title>` (classic Kiwix) or bare
        // `<Title>`; a streetzim option-B bundle stores at
        // `wiki-article/<Title>`.
        var wikiPaths: [String] = []
        for t in titleForms { wikiPaths.append("A/\(t)"); wikiPaths.append(t) }
        let bundlePaths = titleForms.map { "wiki-article/\($0)" }

        // Probe Wikipedia/mdwiki first (complete + current), then any
        // streetzim-bundled copy. Identical parse/return path for both.
        let probes = candidates.map { ($0, wikiPaths) } + bundled.map { ($0, bundlePaths) }
        for (pair, pathSet) in probes {
            for candidate in pathSet {
                if let entry = await cachedArticle(pair: pair, path: candidate) {
                    let sections = await cachedSections(
                        zim: pair.name, path: candidate, html: entry.html)
                    let wantSection = section ?? "lead"
                    let found = ArticleSections.find(wantSection, in: sections)
                        ?? sections.first
                    guard let sec = found else {
                        throw ZimServiceError.notFound("no sections in \(candidate)")
                    }
                    return (pair.name, candidate, entry.title, sec)
                }
            }
        }
        // Fallback: fuzzy title suggest. Handles redirects, title
        // drift, minor casing differences — slower but catches the
        // cases the direct path missed.
        for pair in candidates {
            if let hit = (try? pair.reader.searchTitles(query: withSpaces, limit: 1))?.first {
                let parsed = try await articleSections(path: hit.path, zim: pair.name)
                let wantSection = section ?? "lead"
                let found = ArticleSections.find(wantSection, in: parsed.sections)
                    ?? parsed.sections.first
                guard let sec = found else {
                    throw ZimServiceError.notFound("no sections in \(hit.path)")
                }
                return (parsed.zim, hit.path, parsed.title, sec)
            }
        }
        throw ZimServiceError.notFound("title \"\(cleanedTitle)\" not found in any Wikipedia ZIM")
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
        // Spatial (SZCI/SZRC) ZIMs — built with --spatial-chunk-scale, the
        // only layout whose routing graph fits a large country in mobile RAM
        // — have no monolithic graph.bin. Route via the async cell-based A*
        // (mirrors the streetzim JS viewer). `cachedSpatialGraph` returns nil
        // for ordinary monolithic ZIMs, which fall through to the sync path.
        if let spatial = try cachedSpatialGraph(for: pair) {
            let index = spatial.index
            let origin = nearestNodeSpatial(index: index, lat: req.originLat, lon: req.originLon)
            let goal = nearestNodeSpatial(index: index, lat: req.destLat, lon: req.destLon)
            guard origin >= 0, goal >= 0,
                  let route = await routeSpatial(graph: spatial, index: index,
                                                 origin: origin, goal: goal)
            else { throw ZimServiceError.noRoute }
            return route
        }
        let graph = try loadGraph(pair: pair)
        let origin = graph.nearestNode(lat: req.originLat, lon: req.originLon)
        let goal = graph.nearestNode(lat: req.destLat, lon: req.destLon)
        guard origin >= 0, goal >= 0, let route = aStar(graph: graph, origin: origin, goal: goal) else {
            throw ZimServiceError.noRoute
        }
        return route
    }

    /// Memoized `loadSpatialGraph` — parses the SZCI index once per ZIM
    /// (it's tens of MB) and reuses the actor (its cell cache warms across
    /// routes). Returns nil for monolithic ZIMs.
    private func cachedSpatialGraph(for pair: (name: String, reader: ZimReader)) throws -> SpatialGraph? {
        if let cached = spatialGraphs[pair.name] { return cached }
        guard let g = try loadSpatialGraph(zimName: pair.name) else { return nil }
        spatialGraphs[pair.name] = g
        return g
    }

    public func geocode(query: String, limit: Int, zim: String?, kinds: [String]?) async throws -> [Place] {
        // Literal "lat,lon" (from "my location" substitution, or a user
        // pasting coords) short-circuits the streetzim index: return a
        // synthetic Place so route_from_places / near_places can proceed
        // without needing a matching POI in the ZIM.
        if let p = Self.parseLatLon(query) {
            return [p]
        }
        return try await geocodeResolved(query: query, limit: limit, zim: zim, kinds: kinds).map(\.place)
    }

    /// Parse strings like "37.44121,-122.15530" or "37.44,-122.15" into
    /// a synthetic `Place`. Accepts an optional space after the comma.
    /// Returns nil unless both halves are valid decimal degrees.
    static func parseLatLon(_ s: String) -> Place? {
        let trimmed = s.trimmingCharacters(in: .whitespaces)
        let parts = trimmed.split(separator: ",", maxSplits: 1, omittingEmptySubsequences: true)
        guard parts.count == 2,
              let lat = Double(parts[0].trimmingCharacters(in: .whitespaces)),
              let lon = Double(parts[1].trimmingCharacters(in: .whitespaces)),
              (-90...90).contains(lat), (-180...180).contains(lon)
        else { return nil }
        return Place(
            name: String(format: "%.5f, %.5f", lat, lon),
            kind: "here", lat: lat, lon: lon
        )
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

        // Try progressively looser variants of the query. Models often
        // add a city/state disambiguator ("Union Square, San Francisco")
        // that the streetzim's prefix index doesn't carry, but the
        // primary name ("Union Square") resolves cleanly.
        let filterSet = kinds.map(Set.init)
        for attempt in Self.geocodeVariants(of: query) {
            let prefix = Geocoder.normalizePrefix(attempt)
            for pair in candidates {
                let manifest = try loadManifest(pair: pair)
                // A prefix is either a plain chunk, a fan-out split
                // (hot prefixes on continent-scale builds — "st" →
                // st-0-0…; the leaf list lives in `sub_chunks`), or
                // absent. Before 2026-07-02 the split case fell through
                // as "absent", so EVERY name lookup against the
                // California streetzim silently returned nothing
                // ("How far is Stanford University?" → 'not in the
                // loaded maps').
                let leaves: [String]
                let subLeaves = subChunkLeaves(pair: pair, prefix: prefix)
                if manifest[prefix] != nil {
                    leaves = [prefix]
                } else if !subLeaves.isEmpty {
                    leaves = subLeaves
                } else if manifest.isEmpty {
                    leaves = [prefix]   // legacy build, no manifest — try direct
                } else {
                    continue
                }
                // Sub-chunk bucketing hashes the full NAME, so a query
                // can't route to one leaf — pre-filter each leaf by
                // substring and rank the survivors once. Early-exit
                // when we already hold plenty of candidates; leaves go
                // through their own small byte-budgeted LRU (not the
                // chunk cache) so a hot-prefix scan can't pin hundreds
                // of MB.
                var matching: [[String: Any]] = []
                let q = attempt.lowercased()
                let orderedLeaves = Geocoder.prioritizeSubChunkLeaves(
                    leaves, prefix: prefix, query: attempt)
                for (leafIndex, leaf) in orderedLeaves.enumerated() {
                    if leaves.count == 1 {
                        matching = try loadChunk(pair: pair, prefix: leaf)
                    } else {
                        // JSONSerialization creates a large temporary object
                        // graph for each multi-MB shard. Drain it per leaf and
                        // retain only matching records plus the (budgeted)
                        // cached shard; without this pool a 256-leaf `st`
                        // scan peaked +1.45 GB on iPhone.
                        let leafMatches: [[String: Any]] = try autoreleasepool {
                            let records = try loadLeafChunk(pair: pair, leaf: leaf)
                            return records.filter {
                                (($0["n"] as? String) ?? "")
                                    .lowercased().contains(q)
                            }
                        }
                        matching += leafMatches

                        // For a single-result named-place lookup, an exact
                        // case-insensitive name is globally optimal (offset 0
                        // and the shortest possible containing string). Stop
                        // immediately instead of reading the other 240 hot
                        // shards. Substring queries still fan out completely.
                        if limit == 1,
                           let exact = Geocoder.rank(
                               records: leafMatches, query: attempt,
                               limit: 1, kinds: filterSet
                           ).first,
                           exact.name.caseInsensitiveCompare(attempt) == .orderedSame
                        {
                            log("geocode exact hit in hot prefix \(prefix) after \(leafIndex + 1)/\(leaves.count) leaf/leaves")
                            return [(exact, pair.name)]
                        }
                    }
                    if matching.count >= max(200, limit * 8), leaves.count > 1 {
                        break
                    }
                }
                let ranked = Geocoder.rank(records: matching, query: attempt,
                                           limit: limit, kinds: filterSet)
                if !ranked.isEmpty {
                    if attempt != query {
                        log("geocode fallback: '\(query)' → '\(attempt)' matched in \(pair.name)")
                    }
                    return ranked.map { ($0, pair.name) }
                }
            }
        }
        return []
    }

    /// Ordered set of geocoder queries to try — full query first, then
    /// progressively-stripped versions. Splits on "," (typical city /
    /// state suffix) and " in " (natural-language phrasings from TTS
    /// like "Union Square in San Francisco"). Keeps the left-most
    /// fragment only, since that's almost always the venue name.
    /// Internal (not private) so tests can pin the ordering contract.
    static func geocodeVariants(of query: String) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        func push(_ s: String) {
            let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if t.isEmpty || seen.contains(t) { return }
            seen.insert(t); out.append(t)
        }
        push(query)
        if let c = query.range(of: ",") {
            push(String(query[..<c.lowerBound]))
        }
        if let c = query.range(of: " in ", options: [.caseInsensitive]) {
            push(String(query[..<c.lowerBound]))
        }
        // Progressive trailing-token drop, tried only after the variants
        // above. Field evidence 2026-08-03: locate("k1 kart") threw
        // noMatch even though the index holds "K1 Speed" — the query has
        // no comma and no " in ", so the ONLY attempt was the full
        // phrase, and name.contains("k1 kart") matches nothing. Dropping
        // trailing tokens ("k1 kart" → "k1") lets Geocoder.rank surface
        // the venue by prefix. Ordered last so exact/full matches always
        // win; the ≥2-char floor keeps a bare initial from matching half
        // the index; the cap keeps a long phrase from fanning out into a
        // dozen chunk scans.
        var tokens = (out.last ?? query).split(whereSeparator: { $0.isWhitespace })
        while tokens.count > 1, out.count < 5 {
            tokens.removeLast()
            let candidate = tokens.joined(separator: " ")
            if candidate.count < 2 { break }
            push(candidate)
        }
        return out
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
        zim: String?,
        hasWiki: Bool = false
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
        //
        // `isGeneric` (no kind pinned) routes straight to the search-data
        // scan below, which is the exact data the web search box reads —
        // there's no single kind-partitioned file that answers "what's
        // around me", and scanning loads everything in radius regardless.
        let isGeneric = (kinds?.isEmpty ?? true)
        let effectiveKinds: Set<String>
        if let kinds, !kinds.isEmpty {
            // Canonicalize common phrasings ("drugstore" → "pharmacy")
            // BEFORE fan-out, not just inside `chipsFor` — the chip
            // lookup, the niche/broad decision, and scanRecords' subtype
            // filter must all see the same term. Mapping only at chip
            // lookup would send "drugstore" to the health chip but then
            // return the WHOLE chip (the AFA0ECA1 "211 hospitals" shape),
            // because nicheChipKinds and the filter still saw "drugstore".
            effectiveKinds = Set(kinds.map {
                let k = $0.lowercased()
                return Self.kindSynonyms[k] ?? k
            })
        } else {
            effectiveKinds = ["poi", "place"]
        }
        let radiusM = radiusKm * 1000
        var hits: [(Place, Double)] = []

        // Fast path: read only the kind-partitioned index files the web
        // viewer (`places.html`) itself uses, instead of scanning every
        // search-data prefix chunk (~1 GB resident on country ZIMs).
        // Source of truth for the chip ↔ kind mapping is streetzim's
        // `cloud/chip_rules.py`; `Self.chipsForKind` mirrors it.
        //
        // Three index layers, newest first:
        //   1. `category-index/chip-{id}.json` — common place kinds
        //      (restaurants, cafés, … 11 chips). The web data new
        //      `--no-llm-bundle` ZIMs ship; replaces the old poi blob.
        //   2. `category-index/{place,park,water,…}.json` — light web
        //      categories listed in the manifest's `categories` map.
        //   3. `category-index/{poi,addr,street}.json` — the heavy "LLM
        //      bundle" OLD ZIMs shipped (omitted by `--no-llm-bundle`).
        //      Kept only as a fallback for already-installed old ZIMs.
        // Generic ("what's around me") queries skip all of this and scan
        // search-data, which sees every record regardless of kind.
        if !isGeneric, let catManifest = loadCategoryManifest(pair: pair) {
            let chipsMap = catManifest["chips"] as? [String: Any] ?? [:]
            let availableChips = Set(chipsMap.keys.map { $0.lowercased() })
            let categories = catManifest["categories"] as? [String: Any] ?? [:]
            let availableCats = Set(categories.keys.map { $0.lowercased() })

            // Plan the index files to read as (slug, applyKindFilter).
            var plan: [(slug: String, filter: Bool)] = []

            // 1. Chips. A "broad" kind (restaurant → restaurants chip)
            //    returns the whole chip, matching the web's chip-tap
            //    behaviour; a "niche" kind (pizza → restaurants chip)
            //    narrows within it via scanRecords' subtype/name filter.
            var chipFilter: [String: Bool] = [:]
            for kind in effectiveKinds {
                let chips = Self.chipsFor(kind)
                guard !chips.isEmpty else { continue }
                let niche = Self.nicheChipKinds.contains(kind)
                for c in chips where availableChips.contains(c) {
                    if niche { if chipFilter[c] == nil { chipFilter[c] = true } }
                    else { chipFilter[c] = false }
                }
            }
            for (c, f) in chipFilter { plan.append(("chip-\(c)", f)) }

            // 2. Direct category hits (place, park, water, airport, peak),
            //    minus any a chip already covers (parks).
            for slug in effectiveKinds.intersection(availableCats) {
                if slug == "park" && chipFilter["parks"] != nil { continue }
                plan.append((slug, false))
            }

            // 3. Legacy POI bundle — only for old ZIMs that ship `poi.json`
            //    and no chips. New ZIMs intentionally lack it.
            if availableChips.isEmpty,
               availableCats.contains("poi"),
               effectiveKinds.intersection(availableCats).isEmpty {
                let poiish: Set<String> = ["amenity", "tourism", "shop", "leisure", "historic", "poi", "place"]
                for slug in poiish.intersection(availableCats) { plan.append((slug, true)) }
            }

            if !plan.isEmpty {
                let joined = plan.map { $0.slug }.sorted().joined(separator: ",")
                log("nearPlaces via category-index: \(joined) in \(pair.name)")
                for item in plan {
                    guard let recs = loadCategoryChunk(pair: pair, slug: item.slug) else { continue }
                    scanRecords(recs, filter: effectiveKinds, applyKindFilter: item.filter,
                                centerLat: lat, centerLon: lon,
                                radiusMeters: radiusM,
                                requireWiki: hasWiki, hits: &hits)
                }
                return summarize(hits: hits, limit: limit)
            }
        }

        // Fallback: scan the prefix-chunked search-data — the web search
        // box's own data. Covers generic queries, kinds with no chip, and
        // older streetzims that predate the category index entirely.
        //
        // SAFETY GUARD (prevents jetsam): a full scan loads EVERY chunk into
        // memory. On a statewide ZIM that's tens of millions of records →
        // multi-GB → the app gets jetsammed (observed: 453 chunks loaded,
        // 5.4 GB RSS, crash on "nearest coffee shop"). So refuse the full
        // scan when the ZIM ships chips (the chips ARE the place data — a
        // query that matched no chip must NOT drag in the entire index) or
        // when search-data is very large. Specific kinds hit chips above;
        // this only changes unmapped-kind / generic queries on big ZIMs,
        // which now return the chip/category hits gathered so far (possibly
        // empty) instead of crashing the app.
        let manifest = try loadManifest(pair: pair)
        let totalRecords = manifest.values.reduce(0, +)
        if totalRecords > Self.maxFullScanRecords {
            log("nearPlaces: search-data has \(totalRecords) records (> "
                + "\(Self.maxFullScanRecords) cap) — skipping the full scan to "
                + "avoid OOM; returning \(hits.count) chip/category hit(s) from "
                + "\(pair.name). Specific kinds hit a chip and don't need this.")
            if !isGeneric, hits.isEmpty {
                try nearPlacesNameSearchFallback(
                    pair: pair, kinds: effectiveKinds,
                    centerLat: lat, centerLon: lon, radiusMeters: radiusM,
                    requireWiki: hasWiki, hits: &hits)
            }
            return summarize(hits: hits, limit: limit)
        }
        log("nearPlaces full scan: \(manifest.count) chunk(s) in \(pair.name)")
        let prefixes = manifest.isEmpty ? [] : Array(manifest.keys)
        for prefix in prefixes {
            let records = try loadChunk(pair: pair, prefix: prefix)
            scanRecords(records, filter: effectiveKinds, applyKindFilter: true,
                        centerLat: lat, centerLon: lon,
                        radiusMeters: radiusM,
                        requireWiki: hasWiki, hits: &hits)
        }
        // Even a completed full scan finds nothing for a kind the subtype /
        // synonym filters don't know ("go cart place") — the venue is only
        // discoverable by NAME. Same bounded fallback as the guard path.
        if !isGeneric, hits.isEmpty {
            try nearPlacesNameSearchFallback(
                pair: pair, kinds: effectiveKinds,
                centerLat: lat, centerLon: lon, radiusMeters: radiusM,
                requireWiki: hasWiki, hits: &hits)
        }
        return summarize(hits: hits, limit: limit)
    }

    /// Budgets for `nearPlacesNameSearchFallback`: chunk/leaf loads bound
    /// time (a hot prefix can span 256 leaves — never walk them all for a
    /// best-effort fallback), the match cap bounds accumulation. Small on
    /// purpose: this path must stay "a few chunks", never a full scan.
    static let maxFallbackChunkLoads = 48
    static let maxFallbackMatches = 200

    /// Content words shorter than a name-search term or too generic to be
    /// one — "place" as a substring would match half the index by name.
    private static let fallbackFillerWords: Set<String> = [
        "place", "places", "near", "nearby", "the", "a", "an", "and", "of",
    ]

    /// Bounded last-resort NAME search for `near_places` kinds that map to
    /// no chip: load ONLY the search-data prefix chunks selected by the
    /// kind phrase's own words (the same manifest/sub-chunk machinery
    /// `geocodeResolved` uses) and keep in-radius poi/place records whose
    /// name contains a term. Field evidence 2026-08-03: near_places(kinds:
    /// ["go cart place"]) on the statewide California ZIM matched no chip
    /// and the OOM guard (correctly) refused the full scan → 0 hits in
    /// 2 ms with no fallback attempted, even though "Go Cart Raceway"-
    /// style records sit in the name index. When this also misses, the
    /// result stays a clean fast empty — never a full scan.
    private func nearPlacesNameSearchFallback(
        pair: (name: String, reader: ZimReader),
        kinds: Set<String>,
        centerLat: Double, centerLon: Double,
        radiusMeters: Double,
        requireWiki: Bool,
        hits: inout [(Place, Double)]
    ) throws {
        var loadBudget = Self.maxFallbackChunkLoads
        let manifest = try loadManifest(pair: pair)
        // Only kinds with no chip are eligible — mapped kinds already had
        // their shot at the partitioned index / synonym filters.
        for kind in kinds.sorted() where Self.chipsFor(kind).isEmpty {
            // Terms, most-specific first: the full phrase, each content
            // word (≥3 chars, filler dropped), then the filler-stripped
            // phrase spaced and compacted ("go cart place" → "go cart",
            // "gocart"). No spelling variants (kart≠cart) on purpose —
            // literal containment only, same semantics as the geocoder.
            var terms: [String] = []
            var seenTerms = Set<String>()
            func addTerm(_ t: String) {
                let v = t.lowercased()
                guard v.count >= 3, terms.count < 4,
                      seenTerms.insert(v).inserted else { return }
                terms.append(v)
            }
            let words = kind.lowercased()
                .split(whereSeparator: { $0.isWhitespace }).map(String.init)
            addTerm(kind)
            for w in words where w.count >= 3 && !Self.fallbackFillerWords.contains(w) {
                addTerm(w)
            }
            let kept = words.filter { !Self.fallbackFillerWords.contains($0) }
            if kept != words { addTerm(kept.joined(separator: " ")) }
            if kept.count > 1 { addTerm(kept.joined()) }

            var prefixesTried: [String] = []
            var found = 0
            for term in terms {
                let prefix = Geocoder.normalizePrefix(term)
                let subLeaves = subChunkLeaves(pair: pair, prefix: prefix)
                let leaves: [String]
                if manifest[prefix] != nil {
                    leaves = [prefix]
                } else if !subLeaves.isEmpty {
                    leaves = subLeaves
                } else if manifest.isEmpty {
                    leaves = [prefix]   // legacy build, no manifest — try direct
                } else {
                    continue            // prefix absent → no read at all
                }
                if !prefixesTried.contains(prefix) { prefixesTried.append(prefix) }
                for leaf in leaves {
                    guard loadBudget > 0, hits.count < Self.maxFallbackMatches else { break }
                    loadBudget -= 1
                    // Same memory discipline as geocodeResolved: drain the
                    // multi-MB parsed shard per leaf, retain only the name
                    // matches. Streets/addresses are excluded for the same
                    // reason the default kind filter excludes them — a name
                    // hit on "Descartes Street" answers nothing.
                    let matches: [[String: Any]] = try autoreleasepool {
                        let records = leaves.count == 1
                            ? try loadChunk(pair: pair, prefix: leaf)
                            : try loadLeafChunk(pair: pair, leaf: leaf)
                        return records.filter { rec in
                            let t = ((rec["t"] as? String) ?? "").lowercased()
                            guard t == "poi" || t == "place" else { return false }
                            return (((rec["n"] as? String) ?? "")
                                .lowercased().contains(term))
                        }
                    }
                    let before = hits.count
                    scanRecords(matches, filter: [], applyKindFilter: false,
                                centerLat: centerLat, centerLon: centerLon,
                                radiusMeters: radiusMeters,
                                requireWiki: requireWiki, hits: &hits)
                    found += hits.count - before
                }
            }
            log("nearPlaces: kind '\(kind)' unmapped — name-search fallback "
                + "over prefixes \(prefixesTried) found \(found)")
        }
    }

    /// Compute breakdown-by-subtype + top-N-by-distance from an
    /// in-radius hit list. Subtype is preferred (e.g. "cafe", "bar");
    /// if a record is subtype-less we fall back to its kind.
    private func summarize(hits: [(Place, Double)], limit: Int) -> NearPlacesResult {
        // Dedup: the merged OSM + Overture data lists the same venue more
        // than once — e.g. "Peet's Coffee" twice on the same corner — so the
        // model narrated "Peet's, Peet's, Douce France, Douce France…" and
        // the count was inflated. Collapse by name + ~11 m coordinate cell,
        // keeping the nearest instance (we iterate distance-ascending). Two
        // same-name venues a block apart stay distinct (different cell).
        struct DedupKey: Hashable {
            let name: String
            let cellLat: Int32
            let cellLon: Int32
        }
        let sorted = hits.sorted { $0.1 < $1.1 }
        var seen = Set<DedupKey>()
        seen.reserveCapacity(sorted.count)
        var totalDeduped = 0
        var breakdown: [String: Int] = [:]
        let topCap = max(1, limit)
        var top: [(Place, Double)] = []
        for (p, d) in sorted {
            let key = DedupKey(
                name: p.name.lowercased(),
                cellLat: Int32((p.lat * 1e4).rounded()),
                cellLon: Int32((p.lon * 1e4).rounded())
            )
            guard seen.insert(key).inserted else { continue }
            totalDeduped += 1
            breakdown[p.subtype.isEmpty ? p.kind : p.subtype, default: 0] += 1
            if top.count < topCap { top.append((p, d)) }
        }
        return NearPlacesResult(
            totalInRadius: totalDeduped,
            breakdown: breakdown,
            results: top
        )
    }

    private func scanRecords(
        _ records: [[String: Any]],
        filter: Set<String>, applyKindFilter: Bool,
        centerLat: Double, centerLon: Double,
        radiusMeters: Double,
        requireWiki: Bool = false,
        hits: inout [(Place, Double)]
    ) {
        // Expand synonyms once so every record doesn't repeat the
        // work. For each literal filter term, pull in the subtype
        // targets AND name-keyword patterns from userFacingKindSynonyms.
        var expandedSubtypes: Set<String> = filter
        var nameKeywords: [String] = []
        for term in filter {
            if let syn = Self.userFacingKindSynonyms[term] {
                expandedSubtypes.formUnion(syn.subtypes)
                nameKeywords.append(contentsOf: syn.nameKeywords)
            }
        }
        let needKeywordFallback = !nameKeywords.isEmpty
        // Computed once per scan, not per record — see RadiusBoundingBox.
        let bbox = RadiusBoundingBox(centerLat: centerLat, centerLon: centerLon,
                                     radiusMeters: radiusMeters)
        for rec in records {
            guard let rlat = (rec["a"] as? Double) ?? (rec["lat"] as? Double),
                  let rlon = (rec["o"] as? Double) ?? (rec["lon"] as? Double)
            else { continue }
            // A chip/category load or full search-data scan walks up to
            // `maxFullScanRecords` (500k) records for a radius of a few km,
            // where >99% of a country-scale ZIM is nowhere near the centre;
            // paying haversine's 6 transcendentals on all of them was the
            // scan's dominant cost (DS4 perf medium 2026-08-13). Two
            // subtractions reject those first.
            guard bbox.mayBeWithin(lat: rlat, lon: rlon) else { continue }
            let d = haversineMeters(centerLat, centerLon, rlat, rlon)
            guard d <= radiusMeters else { continue }
            if requireWiki {
                let wiki = Self.sanitizedWikiTag(rec["w"] as? String) ?? ""
                let wikidata = rec["q"] as? String ?? ""
                if wiki.isEmpty && wikidata.isEmpty { continue }
            }
            if applyKindFilter {
                let kind = ((rec["t"] as? String) ?? (rec["type"] as? String) ?? "").lowercased()
                let subtype = ((rec["s"] as? String) ?? (rec["subtype"] as? String) ?? "").lowercased()
                // Exact membership — the original OSM-style check.
                var subtypeMatch = expandedSubtypes.contains(kind)
                    || expandedSubtypes.contains(subtype)
                // Component-split match for Overture places' convention
                // of `<thing>_<category>` (pizza_restaurant,
                // italian_restaurant, food_court, coffee_shop, …).
                // Without this, a query for "restaurant" matched only
                // OSM's bare `restaurant` subtype and skipped every
                // Overture-enriched row. Any underscore-separated
                // component that's already in expandedSubtypes means
                // the record counts — so "pizza" matches
                // `pizza_restaurant` and "restaurant" does too.
                if !subtypeMatch, subtype.contains("_") {
                    for part in subtype.split(separator: "_") {
                        if expandedSubtypes.contains(String(part)) {
                            subtypeMatch = true
                            break
                        }
                    }
                }
                if !subtypeMatch {
                    // Name-keyword fallback. Covers two cases:
                    //   • OSM tags a record `amenity=restaurant` but
                    //     streetzim kept subtype=="amenity".
                    //     "Sushi House" still reads as a restaurant.
                    //   • User queries a niche cuisine ("pizza") and
                    //     the record's subtype is a generic bucket
                    //     like `restaurant` or `fast_food`.
                    //     "Round Table Pizza" (subtype=restaurant)
                    //     should match; "Round Table Sushi" (same
                    //     subtype) shouldn't.
                    //
                    // We require TWO things to accept:
                    //   (a) the subtype is a known generic parent —
                    //       specific subtypes like `italian_restaurant`
                    //       weren't matched by the set or the
                    //       component-split, so they genuinely aren't
                    //       about this cuisine and shouldn't sneak
                    //       through on a name keyword.
                    //   (b) the record's NAME contains at least one
                    //       keyword from the synonym's nameKeywords.
                    if needKeywordFallback,
                       Self.genericParentSubtypes.contains(subtype)
                    {
                        let name = ((rec["n"] as? String) ?? (rec["name"] as? String) ?? "").lowercased()
                        var kw = false
                        for key in nameKeywords {
                            if name.contains(key) { kw = true; break }
                        }
                        if !kw { continue }
                    } else {
                        continue
                    }
                }
            }
            let p = Place(
                name: (rec["n"] as? String) ?? (rec["name"] as? String) ?? "",
                kind: (rec["t"] as? String) ?? (rec["type"] as? String) ?? "",
                lat: rlat, lon: rlon,
                subtype: (rec["s"] as? String) ?? (rec["subtype"] as? String) ?? "",
                location: (rec["l"] as? String) ?? (rec["location"] as? String) ?? "",
                wiki: Self.sanitizedWikiTag(rec["w"] as? String),
                wikidata: rec["q"] as? String,
                website: Geocoder.nonEmpty(rec["ws"] as? String),
                phone: Geocoder.nonEmpty(rec["p"] as? String),
                brand: Geocoder.nonEmpty(rec["brand"] as? String)
            )
            hits.append((p, d))
        }
    }

    // MARK: - Category-index helpers (streetzim ≥ a485ce3)

    private func loadCategoryManifest(pair: (name: String, reader: ZimReader)) -> [String: Any]? {
        if let cached = categoryManifests[pair.name] { return cached }
        guard let entry = try? pair.reader.read(path: "category-index/manifest.json"),
              let json = try? JSONSerialization.jsonObject(with: entry.content) as? [String: Any]
        else { return nil }
        categoryManifests[pair.name] = json
        return json
    }

    private func loadCategoryChunk(pair: (name: String, reader: ZimReader), slug: String) -> [[String: Any]]? {
        let cacheKey = "__cat__:\(slug)"
        if let cached = chunks[pair.name]?[cacheKey] {
            touchChunk(zim: pair.name, prefix: cacheKey)
            return cached
        }
        guard let entry = try? pair.reader.read(path: "category-index/\(slug).json"),
              let decoded = (try? JSONSerialization.jsonObject(with: entry.content)) as? [[String: Any]]
        else { return nil }
        cacheChunk(zim: pair.name, prefix: cacheKey, records: decoded)
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
        // Also expose common user-facing food / POI synonyms. They
        // aren't in the streetzim's literal category slugs, but
        // `scanRecords`'s filter-expansion maps them to real subtypes
        // (e.g. "restaurant" → amenity-with-food-name + fast_food).
        // Without this the schema's `enum` doesn't contain
        // "restaurant", "bar", etc., so the model is more likely to
        // invent strings that fall through filtering.
        set.formUnion(Self.userFacingKindSynonyms.keys)
        return set.sorted()
    }

    /// Synonym table mapping common English POI kinds to the
    /// {subtype} set or {name-keyword} patterns in the streetzim. Used
    /// by `scanRecords` to expand a caller's `kinds` filter so that
    /// "restaurant" matches real records even when the data tags them
    /// as generic `amenity` with a food-like name.
    static let userFacingKindSynonyms: [String: (subtypes: Set<String>, nameKeywords: [String])] = [
        "restaurant":   (subtypes: ["restaurant", "fast_food", "food_court"],
                         nameKeywords: ["restaurant", "pizzeria", "pizza", "bistro",
                                        "taqueria", "sushi", "ramen", "noodle",
                                        "taverna", "kitchen", "grill", "diner",
                                        "steakhouse", "burger", "bbq", "curry",
                                        "tacos", "chicken", "seafood", "thai",
                                        "vietnamese", "mexican", "italian",
                                        "chinese", "korean", "japanese"]),
        "food":         (subtypes: ["restaurant", "fast_food", "food_court", "cafe", "bar", "pub"],
                         nameKeywords: ["pizzeria", "pizza", "bistro", "sushi",
                                        "ramen", "taqueria", "kitchen", "grill"]),
        "cafe":         (subtypes: ["cafe"], nameKeywords: ["cafe", "coffee", "café"]),
        "coffee":       (subtypes: ["cafe"], nameKeywords: ["coffee", "cafe", "café",
                                                              "espresso", "roaster"]),
        "bar":          (subtypes: ["bar", "pub"], nameKeywords: ["bar", "pub", "tavern"]),
        "pub":          (subtypes: ["bar", "pub"], nameKeywords: ["pub", "tavern"]),
        "store":        (subtypes: ["shop", "clothing_store", "grocery"], nameKeywords: []),
        "shop":         (subtypes: ["shop", "clothing_store", "grocery"], nameKeywords: []),
        "groceries":    (subtypes: ["grocery"], nameKeywords: ["market", "grocery"]),
        "supermarket":  (subtypes: ["grocery"], nameKeywords: ["market", "supermarket"]),
        "gas":          (subtypes: ["fuel"], nameKeywords: ["gas station", "shell", "chevron", "76"]),
        "pharmacy":     (subtypes: ["pharmacy"], nameKeywords: ["pharmacy", "cvs", "walgreens"]),
        "hotel":        (subtypes: ["lodging", "hotel"], nameKeywords: ["hotel", "inn", "motel", "lodge"]),
        "lodging":      (subtypes: ["lodging", "hotel"], nameKeywords: ["hotel", "inn", "motel"]),
        "atm":          (subtypes: ["bank"], nameKeywords: ["atm"]),
        "bank":         (subtypes: ["bank"], nameKeywords: ["bank", "chase", "wells fargo",
                                                              "bank of america", "citibank"]),
        "hospital":     (subtypes: ["hospital"], nameKeywords: ["hospital", "medical center",
                                                                  "emergency"]),
        "park":         (subtypes: ["park"], nameKeywords: ["park"]),
        "school":       (subtypes: ["school"], nameKeywords: ["school", "academy"]),
        "church":       (subtypes: ["place_of_worship"], nameKeywords: ["church", "mosque",
                                                                          "temple", "synagogue"]),
        // Museum-family synonyms. Name keywords kept deliberately tight
        // (only unambiguous words) because the subtype="amenity"
        // fallback otherwise sweeps in false positives — e.g., a
        // keyword of "heritage" matched "Heritage Park Dental" (a
        // dentist's office) and surfaced it as the #1 museum.
        "museum":       (subtypes: ["museum", "tourism", "gallery"],
                         nameKeywords: ["museum", "gallery"]),
        "gallery":      (subtypes: ["gallery", "tourism"],
                         nameKeywords: ["gallery", "museum"]),
        "attraction":   (subtypes: ["tourism", "museum", "gallery", "viewpoint",
                                     "attraction", "zoo", "theme_park"],
                         nameKeywords: ["museum", "gallery", "zoo"]),
        "landmark":     (subtypes: ["tourism", "historic", "monument", "memorial"],
                         nameKeywords: ["memorial", "monument"]),
        "zoo":          (subtypes: ["zoo", "tourism"], nameKeywords: ["zoo", "aquarium"]),
        "library":      (subtypes: ["library"], nameKeywords: ["library"]),
        // Niche-cuisine / format synonyms.
        //
        // `subtypes` is DELIBERATELY EMPTY for these. We never want a
        // "pizza near me" query to match every restaurant in the set
        // — it should only match:
        //   1. Overture-style `pizza_restaurant` / `pizza_takeout` /
        //      etc.  Caught by the underscore-component-split matcher
        //      in `nearPlaces`: "pizza" is already in expandedSubtypes
        //      (from the user's raw filter) so any compound subtype
        //      whose components include "pizza" passes.
        //   2. Records whose subtype is a generic bucket
        //      (`restaurant`, `fast_food`, `cafe`, `amenity`, …) AND
        //      whose NAME contains one of the keywords — e.g. OSM's
        //      plain `restaurant` subtype on "Round Table Pizza".
        //      Driven by `nameKeywords` + the broadened
        //      generic-bucket fallback in `nearPlaces`.
        // An empty `subtypes` set keeps us honest — we only accept
        // evidence-backed matches.
        "pizza":        (subtypes: [], nameKeywords: ["pizza", "pizzeria"]),
        "pizzeria":     (subtypes: [], nameKeywords: ["pizza", "pizzeria"]),
        "sushi":        (subtypes: [], nameKeywords: ["sushi"]),
        "burger":       (subtypes: [], nameKeywords: ["burger"]),
        "ramen":        (subtypes: [], nameKeywords: ["ramen", "noodle"]),
        "taco":         (subtypes: [], nameKeywords: ["taco", "tacos", "taqueria"]),
        "tacos":        (subtypes: [], nameKeywords: ["taco", "tacos", "taqueria"]),
        "bbq":          (subtypes: [], nameKeywords: ["bbq", "barbecue", "smokehouse"]),
        "thai":         (subtypes: [], nameKeywords: ["thai"]),
        "indian":       (subtypes: [], nameKeywords: ["indian", "curry"]),
        "mexican":      (subtypes: [], nameKeywords: ["mexican", "taqueria"]),
        "italian":      (subtypes: [], nameKeywords: ["italian", "pizzeria", "trattoria"]),
        "chinese":      (subtypes: [], nameKeywords: ["chinese"]),
        "japanese":     (subtypes: [], nameKeywords: ["japanese", "sushi", "ramen"]),
        "korean":       (subtypes: [], nameKeywords: ["korean"]),
        "vietnamese":   (subtypes: [], nameKeywords: ["vietnamese", "pho"]),
        "vegan":        (subtypes: [], nameKeywords: ["vegan", "vegetarian"]),
        "vegetarian":   (subtypes: [], nameKeywords: ["vegan", "vegetarian"]),
        "bakery":       (subtypes: ["bakery"], nameKeywords: ["bakery", "patisserie"]),
        "ice_cream":    (subtypes: ["ice_cream"], nameKeywords: ["ice cream", "gelato", "frozen yogurt"]),
        "diner":        (subtypes: [], nameKeywords: ["diner"]),
        "brunch":       (subtypes: [], nameKeywords: ["brunch", "breakfast"]),
        "breakfast":    (subtypes: [], nameKeywords: ["brunch", "breakfast"]),
    ]

    /// Generic OSM / OMT / Overture subtype buckets that don't carry
    /// enough specificity on their own — when the user's query is a
    /// niche like "pizza" or "sushi" we require the record's NAME to
    /// back up the claim before matching. Records with a specific
    /// subtype (`italian_restaurant`, `coffee_shop`, …) don't need
    /// the name check — the subtype already answers the question.
    static let genericParentSubtypes: Set<String> = [
        "amenity", "restaurant", "fast_food", "food_court",
        "cafe", "shop", "tourism", "attraction", "leisure",
        "historic", "landuse", "poi",
    ]

    /// Maps a user-facing `kinds` term to the streetzim Find-page chip
    /// file(s) that hold its records (`category-index/chip-{id}.json`).
    /// MIRRORS streetzim's `cloud/chip_rules.py` (the build-time source of
    /// truth for the 11 chips). New `--no-llm-bundle` ZIMs ship these chip
    /// files — the same web data `places.html` loads — in place of the old
    /// `category-index/poi.json` LLM bundle. A term absent here has no chip
    /// and falls through to the search-data scan. Keep in sync with
    /// chip_rules.py's `CHIP_RULES`.
    static let chipsForKind: [String: [String]] = [
        // restaurants
        "restaurant": ["restaurants"], "fast_food": ["restaurants"],
        "food_court": ["restaurants"], "diner": ["restaurants"],
        "food": ["restaurants", "cafes", "bars"],
        "pizza": ["restaurants"], "pizzeria": ["restaurants"],
        "sushi": ["restaurants"], "burger": ["restaurants"],
        "ramen": ["restaurants"], "taco": ["restaurants"], "tacos": ["restaurants"],
        "bbq": ["restaurants"], "thai": ["restaurants"], "indian": ["restaurants"],
        "mexican": ["restaurants"], "italian": ["restaurants"],
        "chinese": ["restaurants"], "japanese": ["restaurants"],
        "korean": ["restaurants"], "vietnamese": ["restaurants"],
        "vegan": ["restaurants"], "vegetarian": ["restaurants"],
        "brunch": ["restaurants"], "breakfast": ["restaurants"],
        "ice_cream": ["restaurants", "cafes"],
        // cafés
        "cafe": ["cafes"], "coffee": ["cafes"], "bakery": ["cafes"],
        // bars
        "bar": ["bars"], "pub": ["bars"], "beer": ["bars"], "nightclub": ["bars"],
        // hotels
        "hotel": ["hotels"], "lodging": ["hotels"], "motel": ["hotels"],
        "hostel": ["hotels"],
        // museums / landmarks
        "museum": ["museums"], "gallery": ["museums"],
        "attraction": ["museums", "landmarks"],
        "landmark": ["landmarks"], "monument": ["landmarks"],
        "memorial": ["landmarks"], "historic": ["landmarks"],
        // parks
        "park": ["parks"],
        // libraries
        "library": ["libraries"],
        // health
        "hospital": ["health"], "pharmacy": ["health"], "clinic": ["health"],
        "doctor": ["health"], "doctors": ["health"], "dentist": ["health"],
        "health": ["health"], "veterinary": ["health"], "vet": ["health"],
        // shops
        "shop": ["shops"], "store": ["shops"], "groceries": ["shops"],
        "grocery": ["shops"], "supermarket": ["shops"], "mall": ["shops"],
        "convenience": ["shops"],
        // fuel
        "gas": ["fuel"], "fuel": ["fuel"], "charging": ["fuel"],
        "charging_station": ["fuel"], "ev": ["fuel"],
    ]

    /// Max search-data records `near_places` will full-scan before giving
    /// up (to stay well under the iOS jetsam ceiling — a statewide ZIM has
    /// tens of millions). Chips serve specific kinds without scanning; this
    /// only bounds the generic / no-chip fallback.
    static let maxFullScanRecords = 500_000

    /// Common user phrasings → the canonical `chipsForKind` term they
    /// mean. Consulted by `chipsFor` and by `nearPlaces`' kind
    /// canonicalization so field phrasings land on the existing 11 chips
    /// instead of falling through to the (correctly) OOM-guarded — and
    /// thus empty — search-data scan. Values MUST be `chipsForKind` keys
    /// and never other synonym keys: both call sites apply the map
    /// exactly once. Kept small: only phrasings that unambiguously mean
    /// an existing chip belong here; anything else is served by the
    /// name-search fallback instead.
    static let kindSynonyms: [String: String] = [
        // fuel
        "petrol": "fuel", "petrol station": "fuel", "gasoline": "fuel",
        // health
        "drugstore": "pharmacy", "chemist": "pharmacy",
        "er": "hospital", "emergency room": "hospital",
        "urgent care": "clinic",
        // cafes
        "coffee house": "cafe", "coffeehouse": "cafe",
        // bars
        "tavern": "bar", "boozer": "bar",
        // hotels
        "inn": "hotel", "bnb": "hotel", "b&b": "hotel",
        "bed and breakfast": "hotel",
        // museums
        "art gallery": "gallery",
        // parks
        "playground": "park", "garden": "park", "gardens": "park",
        // shops
        "bookstore": "shop", "bookshop": "shop", "market": "shop",
    ]

    /// Resolve a (possibly multi-word) `kinds` term to chip id(s),
    /// tolerating the phrasings models actually emit. Tries the term as-is,
    /// its synonym, the underscore form (`ice cream`→`ice_cream`), then
    /// each word (`coffee shop`→`coffee`→cafes, `gas station`→`gas`→fuel).
    /// Without this, a two-word kind missed every chip and fell through to
    /// the search-data scan — which jetsams the app on a statewide ZIM.
    static func chipsFor(_ kind: String) -> [String] {
        let k = kind.lowercased()
        if let c = chipsForKind[k] { return c }
        // Synonyms sit between the exact and word-split passes on
        // purpose: "bed and breakfast" must land on hotels via its
        // synonym before the word-split sees "breakfast" and misroutes
        // it to restaurants.
        if let canonical = kindSynonyms[k], let c = chipsForKind[canonical] { return c }
        let underscored = k.replacingOccurrences(of: " ", with: "_")
        if let c = chipsForKind[underscored] { return c }
        for word in k.split(separator: " ") {
            if let c = chipsForKind[String(word)] { return c }
        }
        return []
    }

    /// Kinds that should NARROW within their chip rather than return the
    /// whole chip — e.g. "pizza" maps to the restaurants chip but the user
    /// wants only pizza places, so scanRecords applies its subtype/name
    /// filter. Broad kinds (restaurant, cafe, hotel, …) are absent here and
    /// return the chip's full nearby slice, matching the web chip tap.
    static let nicheChipKinds: Set<String> = [
        "pizza", "pizzeria", "sushi", "burger", "ramen", "taco", "tacos",
        "bbq", "thai", "indian", "mexican", "italian", "chinese", "japanese",
        "korean", "vietnamese", "vegan", "vegetarian", "diner", "brunch",
        "breakfast", "bakery", "ice_cream",
        // Health-chip members. Unlike the restaurants chip — whose broad
        // term "restaurant" legitimately means the whole chip — the health
        // chip bundles distinct things (hospital, pharmacy, dentist, …)
        // with no single headline kind. Asking for "hospital" must return
        // hospitals only, NOT the whole chip, so each specific member
        // narrows via scanRecords' subtype filter. ("health" itself stays
        // broad = the whole chip.) Without this, near_places(kinds=
        // ["hospital"]) returned hospital(104)+dentist(87)+pharmacy(20)
        // and the caption read "Found 211 hospitals".
        "hospital", "pharmacy", "clinic", "doctor", "doctors", "dentist",
        "veterinary", "vet",
    ]

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
            kinds: kinds, zim: first.zim,
            hasWiki: false
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
        // Spatial ZIMs can't be fed into the sync `aStar(graph:…)` code
        // path — they require async cell loading. Detect up front and
        // point callers at the new spatial API instead of silently
        // degrading (or worse, throwing a vague "no graph.bin" error).
        if try pair.reader.read(path: "routing-data/graph-cells-index.bin") != nil {
            log("\(pair.name) uses the spatial SZCI layout — the current "
                + "ZimService A* path is monolithic; use `loadSpatialGraph` "
                + "+ the async spatial router instead, or repackage this ZIM "
                + "without --spatial-chunk-scale.")
            throw ZimServiceError.noStreetzim
        }
        let memStart = MemoryStats.physFootprintMB()
        log("loading routing-data/graph.bin from \(pair.name)…")
        // Prefer the single-entry layout. If it's missing, fall back to
        // the chunked layout — large regions (Japan, Europe) split the
        // graph across N byte-range entries to side-step libzim's per-
        // cluster size ceiling.
        let graphBytes: Data = try timed("read graph.bin") {
            try Self.readRoutingBlob(
                reader: pair.reader,
                primary: "routing-data/graph.bin",
                manifest: "routing-data/graph-chunk-manifest.json"
            )
        }
        log(String(format: "graph.bin = %.1f MB, parsing (skip geoms)…", Double(graphBytes.count) / 1_048_576))
        // Skip per-edge polyline decoding — A* only reads node positions
        // and edge distances. Saves ~600 MB on country-scale graphs. Any
        // client that wants precise polylines can reparse with decodeGeoms=true
        // (and attach the SZGM companion for v5 split ZIMs).
        let g = try timed("parse graph") {
            try SZRGGraph.parse(graphBytes, geomsData: nil, decodeGeoms: false)
        }
        let memAfter = MemoryStats.physFootprintMB()
        let est = g.estimatedBytes
        log(String(
            format: "graph: %d nodes · %d edges · est=%.0f MB · Δmem=%+.0f MB (parse→steady)",
            g.numNodes, g.numEdges, Double(est) / 1_048_576, memAfter - memStart
        ))
        graphs[pair.name] = g
        return g
    }

    /// Load a lazy, cell-based spatial graph from a ZIM that carries the
    /// SZCI index. Returns ``nil`` when the ZIM is monolithic (caller
    /// should fall back to the sync ``loadGraph`` / ``aStar`` path).
    ///
    /// The returned actor eager-holds only the SZCI index (nodes + names
    /// + cell metadata — ~150 MB on Japan-scale ZIMs). Cell edge buffers
    /// fetch lazily via ``pair.reader.read(path:)`` as the async router
    /// walks into them. Callers drive the LRU via ``cacheLimit``.
    public func loadSpatialGraph(
        zimName: String,
        cacheLimit: Int = 32
    ) throws -> SpatialGraph? {
        guard let pair = readers.first(where: { $0.name == zimName }) else {
            throw ZimServiceError.unknownZim(zimName)
        }
        guard let indexEntry = try pair.reader.read(path: "routing-data/graph-cells-index.bin")
        else {
            return nil
        }
        var idx = try SZCIIndex.parse(indexEntry.content)
        // SZCI v2 shards the node table out of the index — assemble it from
        // routing-data/nodes-scaled-NNN.bin before routing (nearestNode and
        // the A* heuristic both read node coords). v1 already inlined it.
        if idx.version == 2 {
            idx.nodesScaled = try Self.loadNodeShards(idx, reader: pair.reader)
        }
        log(String(format: "spatial index loaded from %@: v%d · %d nodes · %d edges · %d cells",
                   pair.name, idx.version, idx.numNodes, idx.numEdges, idx.numCells))
        let reader = pair.reader
        return SpatialGraph(index: idx, cacheLimit: cacheLimit) { cellId in
            // Cell file names: 5-digit zero-pad. Match the writer in
            // streetzim/cloud/repackage_zim.py `_emit_spatial_graph`.
            let path = String(format: "routing-data/graph-cell-%05d.bin", cellId)
            guard let entry = try reader.read(path: path) else {
                throw SZCIError.cellNotFound(cellId)
            }
            return entry.content
        }
    }

    /// Assemble the SZCI v2 node table from its
    /// `routing-data/nodes-scaled-NNN.bin` shards (3-digit zero-pad). Each
    /// shard holds up to `nodesPerShard` nodes × 8 bytes (lat_e7, lon_e7 as
    /// little-endian Int32); shard `i` lands at element offset
    /// `i * nodesPerShard * 2`. Mirrors the JS viewer's `loadNodeShards`.
    ///
    /// Every count here comes from an untrusted header — ZIMs arrive over P2P
    /// nearby-share, and this runs on the routing path. `numNodes` alone used
    /// to size the buffer, so 0xFFFFFFFF asked for ~34 GB (OOM jetsam) and
    /// `shard * nodesPerShard * 2 * 4` trapped on Int overflow before any
    /// bound was consulted (DS4 medium 2026-08-13). `SZCIIndex.parse` has
    /// already rejected shard triples that aren't ceil-consistent; what's left
    /// is to ground the count in bytes the archive actually carries, which is
    /// the v2 equivalent of v1's "needs N×8 bytes, have M". Probing the first
    /// and last shard pins it exactly: non-final shards are full, so
    /// `(shards-1) × nodesPerShard + lastShardNodes` IS `numNodes`, and a
    /// forged count can't survive without shipping the bytes to back it.
    private static func loadNodeShards(_ idx: SZCIIndex, reader: ZimReader) throws -> [Int32] {
        let nodeCount = idx.numNodes
        let shardCount = idx.numNodeShards
        let perShard = idx.nodesPerShard
        if nodeCount == 0 { return [] }
        // `%03d` below can only name 1000 shards, so a larger claim is forged
        // by construction — reject it before it multiplies into an offset.
        guard shardCount >= 1, shardCount <= 1000, perShard >= 1 else {
            throw SZCIError.truncated(
                "SZCI v2: \(shardCount) node shards × \(perShard) unusable "
                + "(nodes-scaled-%03d.bin names at most 1000)")
        }

        func readShard(_ i: Int) throws -> Data {
            let name = String(format: "routing-data/nodes-scaled-%03d.bin", i)
            guard let entry = try reader.read(path: name) else {
                throw SZCIError.truncated("SZCI v2: missing node shard \(name)")
            }
            guard entry.content.count % 8 == 0 else {
                throw SZCIError.truncated(
                    "SZCI v2: node shard \(i) is \(entry.content.count) B, not a multiple of 8")
            }
            return entry.content
        }

        let firstShard = try readShard(0)
        let lastShard = shardCount == 1 ? firstShard : try readShard(shardCount - 1)
        let lastNodes = lastShard.count / 8
        let impliedNodes = shardCount == 1
            ? lastNodes
            : (shardCount - 1) * perShard + lastNodes
        guard shardCount == 1 || firstShard.count / 8 == perShard,
              lastNodes >= 1, lastNodes <= perShard,
              impliedNodes == nodeCount
        else {
            throw SZCIError.truncated(
                "SZCI v2: header claims \(nodeCount) nodes but the shards carry "
                + "\(impliedNodes)")
        }

        var combined = [Int32](repeating: 0, count: nodeCount * 2)
        try combined.withUnsafeMutableBytes { rawDst in
            for shard in 0..<shardCount {
                let content: Data
                switch shard {
                case 0: content = firstShard
                case shardCount - 1: content = lastShard
                default: content = try readShard(shard)
                }
                // Overflow-checked even though the geometry checks above make
                // the product safe: this arithmetic is the one place a header
                // value reaches an unchecked `*`, and a trap here is a crash,
                // not a skipped ZIM.
                let (elements, elemOverflow) =
                    shard.multipliedReportingOverflow(by: perShard)
                let (byteOffset, byteOverflow) =
                    elements.multipliedReportingOverflow(by: 8)
                // A short middle shard would zero-fill a hole in the table
                // instead of failing, i.e. silently place nodes at (0, 0) —
                // so the full-shard invariant is enforced here too, not only
                // on the probed pair.
                let expectedNodes = shard == shardCount - 1 ? lastNodes : perShard
                guard !elemOverflow, !byteOverflow,
                      byteOffset >= 0, byteOffset <= rawDst.count,
                      content.count / 8 == expectedNodes,
                      content.count <= rawDst.count - byteOffset
                else {
                    throw SZCIError.truncated("SZCI v2: node shard \(shard) overruns node table")
                }
                // File bytes are little-endian Int32; on (LE) ARM a raw copy
                // into the Int32-backed buffer is the correct value.
                content.copyBytes(to: UnsafeMutableRawBufferPointer(
                    rebasing: rawDst[byteOffset ..< byteOffset + content.count]))
            }
        }
        return combined
    }

    /// Read a routing-graph-sized blob from a ZIM. If ``primary`` isn't an
    /// entry but ``manifest`` is, reassemble the chunked layout.
    /// ``SZRGError.noStreetzim``-style failures propagate as regular
    /// ``ZimServiceError.noStreetzim`` so upstream fallback logic is
    /// undisturbed.
    private static func readRoutingBlob(
        reader: ZimReader,
        primary: String,
        manifest manifestPath: String
    ) throws -> Data {
        if let entry = try reader.read(path: primary) {
            return entry.content
        }
        guard let manifest = try reader.read(path: manifestPath) else {
            throw ZimServiceError.noStreetzim
        }
        // Manifest lists chunk paths relative to its own directory.
        let dir: String
        if let slash = manifestPath.lastIndex(of: "/") {
            dir = String(manifestPath[..<manifestPath.index(after: slash)])
        } else {
            dir = ""
        }
        return try SZRGChunked.reassembleChunked(manifest: manifest.content) { chunkName in
            guard let ch = try reader.read(path: dir + chunkName) else {
                throw SZRGError.chunkedReassembly(
                    "chunk entry \(dir + chunkName) not found in ZIM"
                )
            }
            return ch.content
        }
    }

    private func loadManifest(pair: (name: String, reader: ZimReader)) throws -> [String: Int] {
        if let cached = manifests[pair.name] { return cached }
        log("loading search-data/manifest.json from \(pair.name)…")
        guard let entry = try pair.reader.read(path: "search-data/manifest.json") else {
            manifests[pair.name] = [:]
            subChunkMaps[pair.name] = [:]
            return [:]
        }
        let parsed = (try? JSONSerialization.jsonObject(with: entry.content)) as? [String: Any]
        let chunks = (parsed?["chunks"] as? [String: Int]) ?? [:]
        manifests[pair.name] = chunks
        // Large builds fan hot prefixes out into FNV-1a sub-chunks
        // ("st" → st-0-0 … st-f-f, recursive to depth 5) and record the
        // leaf list under `sub_chunks`. The bucketing hashes the full
        // record NAME, so a substring query can't route to one leaf —
        // the client contract (the viewer's `expandPrefix`) is: fetch
        // every leaf under the prefix and filter by query content.
        subChunkMaps[pair.name] =
            (parsed?["sub_chunks"] as? [String: [String]]) ?? [:]
        return chunks
    }

    /// Leaf chunk names for a prefix that was fan-out split (see
    /// `loadManifest`). Empty when the prefix is a plain single chunk.
    /// `loadManifest(pair:)` must have run first (it populates the map).
    func subChunkLeaves(
        pair: (name: String, reader: ZimReader), prefix: String
    ) -> [String] {
        subChunkMaps[pair.name]?[prefix] ?? []
    }

    private func loadChunk(
        pair: (name: String, reader: ZimReader), prefix: String
    ) throws -> [[String: Any]] {
        if let cached = chunks[pair.name]?[prefix] {
            touchChunk(zim: pair.name, prefix: prefix)
            return cached
        }
        log("loading search-data/\(prefix).json from \(pair.name)…")
        guard let entry = try pair.reader.read(path: "search-data/\(prefix).json") else {
            return []
        }
        let parsed = (try? JSONSerialization.jsonObject(with: entry.content)) as? [[String: Any]] ?? []
        cacheChunk(zim: pair.name, prefix: prefix, records: parsed)
        return parsed
    }

    /// Fan-out leaf loader with its own LRU, kept apart from the chunk
    /// cache: hot prefixes are by construction the most-geocoded names,
    /// so re-decompressing + re-parsing dozens of multi-MB leaves per
    /// geocode was pure repeat cost — but one hot prefix can also span
    /// hundreds of leaves (the "453 chunks, 5.4 GB, jetsam" war story),
    /// so the budget stays small. Raw shard bytes stand in for parsed
    /// footprint.
    private func loadLeafChunk(
        pair: (name: String, reader: ZimReader), leaf: String
    ) throws -> [[String: Any]] {
        let key = LeafKey(zim: pair.name, leaf: leaf)
        if let cached = leafChunks[key] {
            touchLeaf(key)
            return cached.records
        }
        log("loading search-data/\(leaf).json from \(pair.name)…")
        guard let entry = try pair.reader.read(path: "search-data/\(leaf).json") else {
            return []
        }
        let parsed = (try? JSONSerialization.jsonObject(with: entry.content)) as? [[String: Any]] ?? []
        leafChunks[key] = (records: parsed, bytes: entry.content.count)
        leafLRU.append(key)
        cachedLeafBytes += entry.content.count
        while cachedLeafBytes > Self.maxCachedLeafBytes
            || leafLRU.count > Self.maxCachedLeaves,
            leafLRU.count > 1
        {
            let victim = leafLRU.removeFirst()
            if let evicted = leafChunks.removeValue(forKey: victim) {
                cachedLeafBytes -= evicted.bytes
            }
        }
        return parsed
    }

    // MARK: - Chunk-cache LRU

    /// Budget on total records the chunk cache may pin: one full-scan's
    /// worth. A second ZIM's scan (or a long geocode session) evicts the
    /// least-recently-used chunks instead of accumulating without bound.
    static let maxCachedChunkRecords = maxFullScanRecords

    /// Raw-byte + leaf-count budgets for the fan-out leaf cache — small
    /// enough that even a session of hot-prefix geocodes stays tens of
    /// MB, not the multi-GB an unbounded shard cache once reached.
    static let maxCachedLeafBytes = 24 * 1024 * 1024
    static let maxCachedLeaves = 64

    private func touchLeaf(_ key: LeafKey) {
        guard let idx = leafLRU.firstIndex(of: key),
              idx != leafLRU.count - 1
        else { return }
        leafLRU.append(leafLRU.remove(at: idx))
    }

    private func touchChunk(zim: String, prefix: String) {
        guard let idx = chunkLRU.firstIndex(where: { $0.zim == zim && $0.prefix == prefix }),
              idx != chunkLRU.count - 1
        else { return }
        let key = chunkLRU.remove(at: idx)
        chunkLRU.append(key)
    }

    private func cacheChunk(zim: String, prefix: String, records: [[String: Any]]) {
        if chunks[zim]?[prefix] != nil {
            touchChunk(zim: zim, prefix: prefix)
            return
        }
        chunks[zim, default: [:]][prefix] = records
        chunkLRU.append((zim, prefix))
        // Empty chunks still count 1 so a flood of misses can't grow the
        // bookkeeping arrays unboundedly.
        cachedChunkRecords += max(records.count, 1)
        while cachedChunkRecords > Self.maxCachedChunkRecords, chunkLRU.count > 1 {
            let victim = chunkLRU.removeFirst()
            if let evicted = chunks[victim.zim]?.removeValue(forKey: victim.prefix) {
                cachedChunkRecords -= max(evicted.count, 1)
                log("chunk cache evicted \(victim.zim)/\(victim.prefix) (\(evicted.count) records)")
            }
        }
    }
}
