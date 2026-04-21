// SPDX-License-Identifier: MIT
//
// Deterministic, in-memory `ZimService` for the multi-model eval harness.
//
// The real `DefaultZimService` needs ZIMs on disk (~500 MB+ of libzim +
// geocoder + routing graph). A Mac test target can't drag that around in
// CI, and even when it can, the I/O makes every scenario run several
// seconds. `StubZimService` sidesteps both problems: you hand it a
// `Fixture` keyed on tool arguments, and the service replays the canned
// result verbatim whenever the model (driven through `ChatSession.send`)
// asks for it. Calls that miss the fixture throw `.noFixture(...)` so
// the harness can tell when a model has drifted off-script.
//
// Scope in Phase 1 of `EVAL_HARNESS.md`: just the methods the first three
// scenarios (`restaurants_in_sf`, `directions_to_sf`, `nearest_post_office`)
// exercise. Additional fixtures get added one-per-scenario as the harness
// grows.

import Foundation

public actor StubZimService: ZimService {

    /// Canned-response table. Kept as value-typed dictionaries so test
    /// authors can construct scenarios declaratively without touching
    /// actor internals. Every key is a normalised tuple of the
    /// arguments the model is expected to pass — see the static
    /// `key…` helpers at the bottom for the normalisation rules.
    public struct Fixture: Sendable {
        public var nearNamedPlace: [String: NearNamedPlaceResponse] = [:]
        public var nearPlaces: [String: NearPlacesResponse] = [:]
        public var routeFromPlaces: [String: RouteFromPlacesResponse] = [:]
        public var search: [String: [SearchHitResult]] = [:]
        public var articleByTitle: [String: ArticleByTitleResponse] = [:]
        public var inventory: InventoryResult?

        public init() {}
    }

    public struct NearNamedPlaceResponse: Sendable {
        public let resolved: Place
        public let result: NearPlacesResult
        public init(resolved: Place, result: NearPlacesResult) {
            self.resolved = resolved
            self.result = result
        }
    }

    public struct NearPlacesResponse: Sendable {
        public let result: NearPlacesResult
        public init(result: NearPlacesResult) { self.result = result }
    }

    public struct RouteFromPlacesResponse: Sendable {
        public let resolved: (origin: Place, destination: Place)
        public let route: Route
        public let zimUsed: String?
        public init(resolved: (origin: Place, destination: Place), route: Route, zimUsed: String? = "stub-streetzim") {
            self.resolved = resolved
            self.route = route
            self.zimUsed = zimUsed
        }
    }

    public struct ArticleByTitleResponse: Sendable {
        public let zim: String
        public let path: String
        public let title: String
        public let section: ArticleSection
        public init(zim: String, path: String, title: String, section: ArticleSection) {
            self.zim = zim
            self.path = path
            self.title = title
            self.section = section
        }
    }

    public enum StubError: Error, CustomStringConvertible {
        case noFixture(method: String, key: String)
        public var description: String {
            switch self {
            case .noFixture(let m, let k):
                return "StubZimService has no fixture for \(m) with key '\(k)'"
            }
        }
    }

    private var fixture: Fixture

    public init(fixture: Fixture) {
        self.fixture = fixture
    }

    // MARK: - ZimService (covered in Phase 1)

    public func inventory() async throws -> InventoryResult {
        fixture.inventory ?? InventoryResult(zims: [], capabilities: ["stub"])
    }

    public func nearNamedPlace(
        place: String, radiusKm: Double, limit: Int,
        kinds: [String]?, zim: String?
    ) async throws -> (resolved: Place, result: NearPlacesResult) {
        let k = Self.keyNearNamedPlace(place: place, kinds: kinds)
        guard let hit = fixture.nearNamedPlace[k] else {
            throw StubError.noFixture(method: "nearNamedPlace", key: k)
        }
        return (hit.resolved, hit.result)
    }

    public func nearPlaces(
        lat: Double, lon: Double, radiusKm: Double, limit: Int,
        kinds: [String]?, zim: String?, hasWiki: Bool
    ) async throws -> NearPlacesResult {
        let k = Self.keyNearPlaces(lat: lat, lon: lon, kinds: kinds)
        guard let hit = fixture.nearPlaces[k] else {
            throw StubError.noFixture(method: "nearPlaces", key: k)
        }
        return hit.result
    }

    public func routeFromPlaces(
        origin: String, destination: String, zim: String?
    ) async throws -> (resolved: (origin: Place, destination: Place), route: Route, zimUsed: String?) {
        let k = Self.keyRouteFromPlaces(origin: origin, destination: destination)
        guard let hit = fixture.routeFromPlaces[k] else {
            throw StubError.noFixture(method: "routeFromPlaces", key: k)
        }
        return (hit.resolved, hit.route, hit.zimUsed)
    }

    public func search(query: String, limit: Int, kind: ZimKind?) async throws -> [SearchHitResult] {
        let k = Self.keySearch(query: query)
        guard let hit = fixture.search[k] else {
            throw StubError.noFixture(method: "search", key: k)
        }
        return hit
    }

    public func articleByTitle(
        title: String, zim: String?, section: String?
    ) async throws -> (zim: String, path: String, title: String, section: ArticleSection) {
        let k = Self.keyArticleByTitle(title: title, section: section)
        guard let hit = fixture.articleByTitle[k] else {
            throw StubError.noFixture(method: "articleByTitle", key: k)
        }
        return (hit.zim, hit.path, hit.title, hit.section)
    }

    // MARK: - ZimService (not yet covered — throw so the harness surfaces
    //                      model drift instead of silently succeeding on
    //                      a default value).

    public func article(path: String, zim: String?) async throws -> ArticleResult {
        throw StubError.noFixture(method: "article", key: "path=\(path),zim=\(zim ?? "-")")
    }

    public func articleSections(path: String, zim: String?) async throws -> (zim: String, title: String, sections: [ArticleSection]) {
        throw StubError.noFixture(method: "articleSections", key: "path=\(path)")
    }

    public func articleSection(path: String, section: String, zim: String?) async throws -> (zim: String, title: String, section: ArticleSection) {
        throw StubError.noFixture(method: "articleSection", key: "path=\(path),section=\(section)")
    }

    public func mainPage(zim: String?) async throws -> [ArticleResult] { [] }

    public func planDrivingRoute(_ req: RouteRequest) async throws -> Route {
        throw StubError.noFixture(method: "planDrivingRoute",
                                  key: "(\(req.originLat),\(req.originLon))→(\(req.destLat),\(req.destLon))")
    }

    public func geocode(query: String, limit: Int, zim: String?, kinds: [String]?) async throws -> [Place] {
        throw StubError.noFixture(method: "geocode", key: query.lowercased())
    }

    public func zimInfo(zim: String?) async throws -> [[String: Any]] { [] }

    // MARK: - Key normalisation

    /// Key: lowercased place + optional lowercased kinds (sorted, joined with ',').
    public static func keyNearNamedPlace(place: String, kinds: [String]?) -> String {
        let k = (kinds ?? []).map { $0.lowercased() }.sorted().joined(separator: ",")
        return "\(place.lowercased())|\(k)"
    }

    /// Key: rounded coord (3 decimals ≈ 100 m) + optional kinds.
    public static func keyNearPlaces(lat: Double, lon: Double, kinds: [String]?) -> String {
        let la = String(format: "%.3f", lat)
        let lo = String(format: "%.3f", lon)
        let k = (kinds ?? []).map { $0.lowercased() }.sorted().joined(separator: ",")
        return "\(la),\(lo)|\(k)"
    }

    public static func keyRouteFromPlaces(origin: String, destination: String) -> String {
        "\(origin.lowercased())→\(destination.lowercased())"
    }

    public static func keySearch(query: String) -> String { query.lowercased() }

    public static func keyArticleByTitle(title: String, section: String?) -> String {
        "\(title.lowercased())|\(section?.lowercased() ?? "")"
    }
}
