import XCTest
@testable import MCPZimKit

final class CurrentDocumentBoundaryTests: XCTestCase {
    private let article = OfflineArticle(zim: "fixture.zim", path: "A/Mira", title: "Mira")

    func testOversizedRawDocumentNeverRequestsParsedSections() async throws {
        let service = Probe(bytes: OfflineKnowledge.maxHTMLBytes + 1)
        do {
            _ = try await OfflineKnowledge(service: service).document(article: article)
            XCTFail("Oversized raw document exported")
        } catch OfflineKnowledge.Failure.tooLarge { }
        let calls = await service.sectionCalls
        XCTAssertEqual(calls, 0, "HTML must be rejected before section parsing")
    }

    func testCancellationDuringRawLoadPreventsParsing() async throws {
        try await cancellation(at: .raw, expectedSectionCalls: 0)
    }

    func testCancellationDuringSectionLoadPreventsExport() async throws {
        try await cancellation(at: .sections, expectedSectionCalls: 1)
    }

    private func cancellation(at stage: Probe.Stage, expectedSectionCalls: Int) async throws {
        let service = Probe(stage: stage)
        let article = article
        let task = Task { try await OfflineKnowledge(service: service).document(article: article) }
        let deadline = Date().addingTimeInterval(5)
        while !(await service.isSuspended), Date() < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let suspended = await service.isSuspended
        XCTAssertTrue(suspended, "The request must reach the suspended load")
        task.cancel()
        await service.resume()
        do { _ = try await task.value; XCTFail("Cancelled document exported") }
        catch is CancellationError { }
        let calls = await service.sectionCalls
        XCTAssertEqual(calls, expectedSectionCalls)
    }

    /// A service that deliberately completes its I/O after cancellation.
    /// The exporter must recheck cancellation rather than trust the loader.
    private actor Probe: ZimService {
        enum Stage { case none, raw, sections }
        enum Failure: Error { case unexpectedCall }
        let bytes: Int
        let stage: Stage
        var sectionCalls = 0
        var isSuspended = false
        private var continuation: CheckedContinuation<Void, Never>?
        init(bytes: Int = 100, stage: Stage = .none) {
            self.bytes = bytes; self.stage = stage
        }
        private func pause(at point: Stage) async {
            guard stage == point else { return }
            await withCheckedContinuation { continuation in
                self.continuation = continuation
                isSuspended = true
            }
        }
        func resume() { continuation?.resume(); continuation = nil }
        func article(path: String, zim: String?) async throws -> ArticleResult {
            await pause(at: .raw)
            return .init(zim: "fixture.zim", path: path, title: "Mira",
                         mimetype: "text/html", text: "<p>Mira was a writer.</p>", bytes: bytes)
        }
        func articleSections(path: String, zim: String?) async throws -> (zim: String, title: String, sections: [ArticleSection]) {
            sectionCalls += 1
            await pause(at: .sections)
            return ("fixture.zim", "Mira", [.init(title: "Introduction", level: 1, text: "Mira was a writer.")])
        }
        func inventory() async throws -> InventoryResult { throw Failure.unexpectedCall }
        func search(query: String, limit: Int, kind: ZimKind?) async throws -> [SearchHitResult] { throw Failure.unexpectedCall }
        func articleSection(path: String, section: String, zim: String?) async throws -> (zim: String, title: String, section: ArticleSection) { throw Failure.unexpectedCall }
        func articleByTitle(title: String, zim: String?, section: String?) async throws -> (zim: String, path: String, title: String, section: ArticleSection) { throw Failure.unexpectedCall }
        func mainPage(zim: String?) async throws -> [ArticleResult] { throw Failure.unexpectedCall }
        func planDrivingRoute(_ req: RouteRequest) async throws -> Route { throw Failure.unexpectedCall }
        func geocode(query: String, limit: Int, zim: String?, kinds: [String]?) async throws -> [Place] { throw Failure.unexpectedCall }
        func nearPlaces(lat: Double, lon: Double, radiusKm: Double, limit: Int, kinds: [String]?, zim: String?, hasWiki: Bool) async throws -> NearPlacesResult { throw Failure.unexpectedCall }
        func nearNamedPlace(place: String, radiusKm: Double, limit: Int, kinds: [String]?, zim: String?) async throws -> (resolved: Place, result: NearPlacesResult) { throw Failure.unexpectedCall }
        func zimInfo(zim: String?) async throws -> [[String: Any]] { throw Failure.unexpectedCall }
        func routeFromPlaces(origin: String, destination: String, zim: String?) async throws -> (resolved: (origin: Place, destination: Place), route: Route, zimUsed: String?) { throw Failure.unexpectedCall }
    }
}
