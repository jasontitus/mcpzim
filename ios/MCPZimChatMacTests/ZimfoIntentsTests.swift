// SPDX-License-Identifier: MIT
//
// End-to-end-ish tests for Zimfo's App Intents. Runs the intents'
// `perform()` directly, bypassing Siri / Shortcuts so we don't have
// to drive any UI — handy for iteration on tool-chaining logic.
//
// The tests fall into two buckets:
//   - **Pure** (always run): `ZimfoContext` round-trip, `RouteProgress`
//     math. No ZIMs, no CoreLocation.
//   - **End-to-end** (opt-in): actually spins up `ZimfoRunner`, hits
//     real libzim readers, runs real routing / geocoding. Requires at
//     least one streetzim ZIM in the app's sandbox Documents or in the
//     persisted external bookmarks. Skipped when no data is available.

import XCTest
import CoreLocation
@testable import MCPZimChatMac

final class ZimfoContextTests: XCTestCase {
    func testActiveRoutePersistsAcrossInstances() async throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("zimfo-ctx-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let route = ActiveRoute(
            startedAt: Date(),
            origin: .init(lat: 38.9, lon: -77.05),
            destination: .init(lat: 38.91, lon: -77.04),
            originName: "here",
            destinationName: "White House",
            zim: "osm-dc.zim",
            totalDurationSeconds: 600,
            totalDistanceMeters: 1000,
            polyline: [.init(lat: 38.9, lon: -77.05), .init(lat: 38.91, lon: -77.04)],
            cumulativeDistanceMeters: [0, 1000],
            turnByTurn: ["step 1", "step 2"]
        )

        let a = ZimfoContext(storeURL: tmp)
        await a.setActiveRoute(route)

        let b = ZimfoContext(storeURL: tmp)
        let reloaded = await b.activeRoute
        XCTAssertEqual(reloaded?.destinationName, "White House")
        XCTAssertEqual(reloaded?.totalDistanceMeters, 1000)
        XCTAssertEqual(reloaded?.turnByTurn.count, 2)
    }

    func testClearActiveRouteIsPersistent() async throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("zimfo-ctx-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let ctx = ZimfoContext(storeURL: tmp)
        await ctx.setActiveRoute(.init(
            startedAt: Date(),
            origin: .init(lat: 0, lon: 0),
            destination: .init(lat: 0, lon: 0),
            originName: "", destinationName: "", zim: "",
            totalDurationSeconds: 0, totalDistanceMeters: 0,
            polyline: [], cumulativeDistanceMeters: [], turnByTurn: []
        ))
        await ctx.clearActiveRoute()

        let reloaded = ZimfoContext(storeURL: tmp)
        let r = await reloaded.activeRoute
        XCTAssertNil(r)
    }
}

final class RouteProgressTests: XCTestCase {
    /// 10 km straight east along the equator, evenly sampled.
    private func syntheticRoute(totalMeters: Double, totalSeconds: Double, points: Int = 11) -> ActiveRoute {
        let stepLon = (totalMeters / 111_111) / Double(points - 1) // ~1m = 1/111111° lon at equator
        let coords = (0..<points).map { i in
            ActiveRoute.Coordinate(lat: 0, lon: Double(i) * stepLon)
        }
        var cum: [Double] = [0]
        for i in 1..<points {
            cum.append(Double(i) * totalMeters / Double(points - 1))
        }
        return ActiveRoute(
            startedAt: Date(timeIntervalSinceNow: -60),
            origin: coords.first!,
            destination: coords.last!,
            originName: "start",
            destinationName: "end",
            zim: "test.zim",
            totalDurationSeconds: totalSeconds,
            totalDistanceMeters: totalMeters,
            polyline: coords,
            cumulativeDistanceMeters: cum,
            turnByTurn: ["go east"]
        )
    }

    func testRemainingAtStart() {
        let r = syntheticRoute(totalMeters: 10_000, totalSeconds: 600)
        let (m, s) = RouteProgress.remaining(for: r, current: r.origin)
        XCTAssertEqual(m, 10_000, accuracy: 500)
        XCTAssertEqual(s, 600, accuracy: 30)
    }

    func testRemainingAtHalfway() {
        let r = syntheticRoute(totalMeters: 10_000, totalSeconds: 600)
        let mid = r.polyline[5] // sample #5 of 11 → ~middle
        let (m, s) = RouteProgress.remaining(for: r, current: mid)
        XCTAssertEqual(m, 5_000, accuracy: 600)
        XCTAssertEqual(s, 300, accuracy: 40)
    }

    func testRemainingAtEnd() {
        let r = syntheticRoute(totalMeters: 10_000, totalSeconds: 600)
        let (m, s) = RouteProgress.remaining(for: r, current: r.destination)
        XCTAssertLessThan(m, 500)
        XCTAssertLessThan(s, 30)
    }
}

// MARK: - End-to-end intent tests (require a real streetzim)

final class ZimfoIntentEndToEndTests: XCTestCase {
    /// Paths the test looks for. Uses the first one that exists, so a
    /// developer can drop any streetzim into the app sandbox and run the
    /// tests; if none found, the e2e tests skip rather than fail.
    private static func candidateStreetzimURLs() -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let sandbox = home
            .appendingPathComponent("Library")
            .appendingPathComponent("Containers")
            .appendingPathComponent("org.mcpzim.MCPZimChatMac")
            .appendingPathComponent("Data")
            .appendingPathComponent("Documents")
        let explicit = ProcessInfo.processInfo.environment["ZIMBLE_TEST_STREETZIM"]
            .map { URL(fileURLWithPath: $0) }
        return [
            explicit,
            try? FileManager.default.contentsOfDirectory(at: sandbox, includingPropertiesForKeys: nil)
                .first(where: { $0.pathExtension == "zim" && $0.lastPathComponent.contains("osm") })
        ].compactMap { $0 }
    }

    override func setUp() async throws {
        // Canned "current location" for intents that ask for it. Lat/lon
        // of the White House — picked because our main test ZIMs include
        // DC. If a different streetzim is provided via the env var,
        // override this.
        LocationFetcher.overrideForTesting = {
            CLLocationCoordinate2D(latitude: 38.8977, longitude: -77.0365)
        }
    }

    override func tearDown() async throws {
        LocationFetcher.overrideForTesting = nil
        await ZimfoContext.shared.clearActiveRoute()
    }

    @MainActor
    func testStartRouteIntentThenRemaining() async throws {
        guard !Self.candidateStreetzimURLs().isEmpty else {
            throw XCTSkip("No streetzim ZIM available; drop one into the app's Documents folder or set $ZIMBLE_TEST_STREETZIM.")
        }

        // 1. Start a route via the intent.
        let start = StartRouteIntent()
        start.destination = "Lincoln Memorial"
        _ = try await start.perform()

        let route = await ZimfoContext.shared.activeRoute
        XCTAssertNotNil(route, "StartRouteIntent should store an active route")
        XCTAssertGreaterThan(route?.totalDistanceMeters ?? 0, 100)
        XCTAssertFalse(route?.polyline.isEmpty ?? true)

        // 2. Ask remaining — should be close to total right after start.
        let rem = RemainingRouteIntent()
        _ = try await rem.perform()
        // (We don't assert a specific string; the return type is opaque.
        // Just confirm the context was read without crashing.)
    }

    @MainActor
    func testNearbyHereIntentReturnsSomething() async throws {
        guard !Self.candidateStreetzimURLs().isEmpty else {
            throw XCTSkip("No streetzim ZIM available.")
        }
        let intent = NearbyHereIntent()
        _ = try await intent.perform()
        // The intent dialog isn't easily introspectable from here, but
        // we're asserting it doesn't throw — the underlying
        // `ZimfoRunner.nearbySummary` path does the real work.
    }
}

// Content identity regressions from the adversarial Siri review.
import MCPZimKit

@MainActor
final class SiriArticleIdentityTests: XCTestCase {
    private final class Reader: ZimReader, @unchecked Sendable {
        let title: String
        let lead: String
        init(title: String, lead: String) { self.title = title; self.lead = lead }
        var metadata: ZimMetadata { .init(name: "fixture") }
        var kind: ZimKind { .wikipedia }
        var hasTitleIndex: Bool { false }
        var hasFullTextIndex: Bool { false }
        var hasRoutingData: Bool { false }
        func read(path: String) throws -> ZimEntry? {
            guard path == "A/Mira" else { return nil }
            return .init(path: path, title: title, mimetype: "text/html",
                         content: Data("<p>\(lead)</p>".utf8))
        }
        func readMainPage() throws -> ZimEntry? { nil }
    }
    private func runner(title: String = "Mira", lead: String = "Mira was a writer.") -> ZimfoRunner {
        let reader = Reader(title: title, lead: lead)
        let service = DefaultZimService(readers: [(name: "test.zim", reader: reader)])
        return ZimfoRunner(service: service, adapter: MCPToolAdapter(service: service, hasStreetzim: false),
                           readersByName: ["test.zim": reader])
    }
    private let source = OfflineArticle(zim: "test.zim", path: "A/Mira", title: "Mira")

    func testEntityIdentifierSurvivesJSONRehydration() throws {
        let original = try ZimfoArticleEntity(article: source, version: "version")
        let ref = try JSONDecoder().decode(ZimfoArticleEntity.Reference.self,
                                          from: XCTUnwrap(Data(base64Encoded: original.id)))
        for _ in 0..<20 {
            let restored = try ZimfoArticleEntity(article: ref.article, version: ref.version, excerpt: "Different result text")
            XCTAssertEqual(restored.id, original.id)
        }
    }

    func testValidationIgnoresTamperedDisplayMetadataAndExcerpt() async throws {
        let runner = runner()
        var entity = try ZimfoArticleEntity(article: source, version: runner.libraryVersion, excerpt: "untrusted answer")
        entity.title = "Other person"
        entity.archive = "other.zim"
        let validated = try await entity.validated(using: runner)
        XCTAssertEqual(validated, source)
    }

    func testOldLibraryGenerationAndMissingPathsAreRejected() async throws {
        let runner = runner()
        let old = try ZimfoArticleEntity(article: source, version: "old-version")
        do { _ = try await old.validated(using: runner); XCTFail("Old entity must fail") }
        catch OfflineKnowledge.Failure.staleArticle { }
        let missing = try ZimfoArticleEntity(article: .init(zim: "test.zim", path: "A/Missing", title: "Mira"), version: runner.libraryVersion)
        do { _ = try await missing.validated(using: runner); XCTFail("Missing path must fail") }
        catch { }
    }

    func testHandoffRetainsExactSourceAndGeneration() {
        let draft = SiriQuestionHandoff.Draft(question: "Who were her parents?", source: source, libraryVersion: "edition-1")
        XCTAssertEqual(draft.source, source)
        XCTAssertEqual(draft.libraryVersion, "edition-1")
        XCTAssertEqual(draft.question, "Who were her parents?")
    }

    func testSamePathWithChangedTitleOrDisambiguationIsRejected() async throws {
        for runner in [runner(title: "Other person"), runner(lead: "Mira may refer to:")] {
            let entity = try ZimfoArticleEntity(article: source, version: runner.libraryVersion)
            do { _ = try await entity.validated(using: runner); XCTFail("Replaced or ambiguous content accepted") }
            catch OfflineKnowledge.Failure.staleArticle { }
        }
    }

    func testUnknownArchiveAndOversizedIdentityFieldsAreRejected() async throws {
        let runner = runner()
        let invalidSources = [
            OfflineArticle(zim: "unknown.zim", path: source.path, title: source.title),
            OfflineArticle(zim: source.zim, path: String(repeating: "a", count: 2049), title: source.title),
            OfflineArticle(zim: source.zim, path: source.path, title: String(repeating: "a", count: 501)),
            OfflineArticle(zim: String(repeating: "a", count: 8192), path: source.path, title: source.title)
        ]
        for source in invalidSources {
            let entity = try ZimfoArticleEntity(article: source, version: runner.libraryVersion)
            do { _ = try await entity.validated(using: runner); XCTFail("Invalid identity accepted") }
            catch OfflineKnowledge.Failure.staleArticle { }
        }
    }
}

@MainActor
final class SiriDiagnosticsTests: XCTestCase {
    func testLifecyclePersistsWithoutChatSessionAndSurvivesArchiveReopen() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let archive = LogArchive(directory: dir)
        let diagnostics = SiriDiagnostics(sink: { archive.appendSync($0) })
        let trace = SiriInvocation("AskOfflineQuestionIntent", diagnostics: diagnostics)
        trace.stage("awaiting_question")
        trace.stage("question_received", count: 34)
        trace.end()
        let file = try XCTUnwrap(archive.currentFileURL())
        let reopened = LogArchive(directory: dir)
        XCTAssertNotEqual(file, reopened.currentFileURL())
        let text = reopened.read(file)
        for event in ["begin", "awaiting_question", "question_received count=34", "end outcome=returned"] {
            XCTAssertTrue(text.contains(event), text)
        }
        XCTAssertEqual(diagnostics.entries.count, 4)
        XCTAssertTrue(diagnostics.entries.allSatisfy { $0.message.contains(trace.id) })
    }

    func testFailureDoesNotLeakLocalizedQuestionOrCoordinates() {
        var written = ""
        let diagnostics = SiriDiagnostics(sink: { written += $0 })
        let trace = SiriInvocation("NearbyPlaceIntent", diagnostics: diagnostics)
        trace.failed(NSError(domain: "secret location", code: 7,
                             userInfo: [NSLocalizedDescriptionKey: "Albert Einstein\n37.123,-122.456"]))
        trace.end()
        XCTAssertTrue(written.contains("outcome=failed"))
        XCTAssertTrue(written.contains("code=7"))
        for secret in ["Albert Einstein", "37.123", "secret location"] {
            XCTAssertFalse(written.contains(secret), written)
        }
    }

    func testParameterControlFlowAndCancellationAreDistinctFromRetrievalFailure() {
        let diagnostics = SiriDiagnostics(sink: { _ in })
        let prompt = SiriInvocation("AskOfflineQuestionIntent", diagnostics: diagnostics)
        prompt.stage("awaiting_article_choice", count: 2)
        prompt.failed(NSError(domain: "parameter-control-flow", code: 1))
        prompt.end()
        XCTAssertTrue(diagnostics.entries.last!.message.contains("outcome=parameter_exit"))
        let cancelled = SiriInvocation("AskOfflineQuestionIntent", diagnostics: diagnostics)
        cancelled.failed(CancellationError())
        cancelled.end()
        XCTAssertTrue(diagnostics.entries.last!.message.contains("outcome=cancelled"))
        XCTAssertNotEqual(prompt.id, cancelled.id)
    }

    func testLiveBufferIsBoundedAndClearingDoesNotEraseArchive() {
        var written = 0
        let diagnostics = SiriDiagnostics(sink: { _ in written += 1 })
        for _ in 0..<230 { diagnostics.record("bounded test event") }
        XCTAssertEqual(diagnostics.entries.count, 200)
        diagnostics.clear()
        XCTAssertTrue(diagnostics.entries.isEmpty)
        XCTAssertEqual(written, 230)
    }

    func testInvalidQuestionLogsBeforeLibraryOrLocationAccess() async {
        SiriDiagnostics.shared.clear()
        var intent = AskOfflineQuestionIntent()
        intent.question = " "
        do {
            let result = try await intent.perform()
            XCTAssertNil(result.value ?? nil, "A spoken failure must not return a usable article")
        }
        catch { XCTFail("Unexpected error: \(error)") }
        let logs = SiriDiagnostics.shared.entries.map(\.message).joined(separator: "\n")
        XCTAssertTrue(logs.contains("action=AskOfflineQuestionIntent begin"))
        XCTAssertTrue(logs.contains("question_received count=1"))
        XCTAssertTrue(logs.contains("outcome=failed"))
        XCTAssertTrue(logs.contains("explained_failure"))
        XCTAssertFalse(logs.contains("action=Nearby"))
    }

    func testKnownFailuresHaveExplanationsButSystemControlFlowDoesNot() {
        XCTAssertEqual(AskOfflineQuestionIntent.explainedFailure(OfflineKnowledge.Failure.tooLarge),
                       OfflineKnowledge.Failure.tooLarge.errorDescription)
        XCTAssertNotNil(AskOfflineQuestionIntent.explainedFailure(OfflineKnowledge.Failure.noLibrary))
        XCTAssertNil(AskOfflineQuestionIntent.explainedFailure(CancellationError()))
        XCTAssertNil(AskOfflineQuestionIntent.explainedFailure(NSError(domain: "SiriParameterControl", code: 1)))
    }
}

import AppIntents
import AppKit

@MainActor
final class SiriCurrentArticleTests: XCTestCase {
    private let a = SiriArticleReference(article: .init(zim: "a.zim", path: "A/Mira", title: "Mira"), libraryVersion: "v1")
    private let b = SiriArticleReference(article: .init(zim: "b.zim", path: "A/Mira", title: "Mira"), libraryVersion: "v1")

    func testExactSourceAndOpaqueLeaseReplacePreviousDocument() async throws {
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { Data($0.article.zim.utf8) })
        let owner = UUID()
        context.publish(a, owner: owner, eligible: { true })
        let first = try XCTUnwrap(context.current)
        context.publish(a, owner: owner, eligible: { true })
        XCTAssertEqual(context.current?.id, first.id)
        context.publish(b, owner: owner, eligible: { true })
        XCTAssertNil(context.entity(id: first.id))
        let second = try XCTUnwrap(context.current)
        let output = try await context.export(id: second.id)
        XCTAssertEqual(String(decoding: output, as: UTF8.self), "b.zim")
        XCTAssertFalse(second.id.contains("Mira"))
        context.clear(owner: owner)
        XCTAssertNil(context.entity(id: second.id))
    }

    func testAnotherViewsDisappearanceCannotClearCurrentOwner() throws {
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { _ in Data("source".utf8) })
        let ownerA = UUID(), ownerB = UUID()
        context.publish(a, owner: ownerA, eligible: { true })
        context.publish(b, owner: ownerB, eligible: { true })
        let current = try XCTUnwrap(context.current)
        context.clear(owner: ownerA)
        XCTAssertEqual(context.entity(id: current.id)?.id, current.id)
        context.clear(owner: ownerB)
        XCTAssertNil(context.current)
    }

    func testLibraryChangeAndLiveEligibilityRevokeExportWithoutViewRefresh() async throws {
        var version = "v1", visible = true, reads = 0
        let context = SiriCurrentArticleContext(fingerprint: { version }, loader: { _ in reads += 1; return Data("source".utf8) })
        let owner = UUID()
        context.publish(a, owner: owner, eligible: { visible })
        let id = try XCTUnwrap(context.current?.id)
        visible = false
        do { _ = try await context.export(id: id); XCTFail("Hidden context exported") } catch { }
        XCTAssertEqual(reads, 0)
        visible = true
        version = "v2"
        do { _ = try await context.export(id: id); XCTFail("Changed archive exported") } catch { }
        XCTAssertEqual(reads, 0)
    }

    func testSwitchDuringSuspendedExportRejectsOldDocument() async throws {
        var continuation: CheckedContinuation<Data, Never>?
        let started = expectation(description: "loader suspended")
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { _ in
            await withCheckedContinuation { continuation = $0; started.fulfill() }
        })
        let owner = UUID()
        context.publish(a, owner: owner, eligible: { true })
        let id = try XCTUnwrap(context.current?.id)
        let request = Task { try await context.export(id: id) }
        await fulfillment(of: [started], timeout: 2)
        context.publish(b, owner: owner, eligible: { true })
        continuation?.resume(returning: Data("old article".utf8))
        do { _ = try await request.value; XCTFail("Late A escaped after B was selected") } catch { }
    }

    func testCloseDuringSuspendedExportRejectsOldDocument() async throws {
        var continuation: CheckedContinuation<Data, Never>?
        let started = expectation(description: "loader suspended")
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { _ in
            await withCheckedContinuation { continuation = $0; started.fulfill() }
        })
        let owner = UUID()
        context.publish(a, owner: owner, eligible: { true })
        let id = try XCTUnwrap(context.current?.id)
        let request = Task { try await context.export(id: id) }
        await fulfillment(of: [started], timeout: 2)
        context.clear(owner: owner)
        continuation?.resume(returning: Data("closed article".utf8))
        do { _ = try await request.value; XCTFail("Closed document escaped") } catch { }
    }

    func testActivityIsNotIndexedAndAssociationIsCleared() throws {
        guard #available(macOS 15.2, *) else { throw XCTSkip("Requires activity entity API") }
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { _ in Data("source".utf8) })
        let owner = UUID()
        context.publish(a, owner: owner, eligible: { true })
        let id = try XCTUnwrap(context.current?.id)
        let activity = NSUserActivity(activityType: SiriCurrentArticleContext.activityType)
        context.configure(activity, id: id)
        XCTAssertEqual(activity.appEntityIdentifier?.identifier, id)
        XCTAssertFalse(activity.isEligibleForSearch)
        XCTAssertFalse(activity.isEligibleForPublicIndexing)
        XCTAssertFalse(activity.isEligibleForHandoff)
        XCTAssertNil(activity.contentAttributeSet)
        XCTAssertNil(activity.webpageURL)
        context.clear(owner: owner)
        XCTAssertNil(activity.appEntityIdentifier)
    }

    func testOversizedPayloadAndCancelledRequestFail() async throws {
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { _ in Data(repeating: 65, count: OfflineKnowledge.maxDocumentBytes + 1) })
        context.publish(a, owner: UUID(), eligible: { true })
        let id = try XCTUnwrap(context.current?.id)
        do { _ = try await context.export(id: id); XCTFail("Oversized export") } catch { }
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await context.export(id: id)
        }
        do { _ = try await task.value; XCTFail("Cancelled export") } catch is CancellationError { }
    }

    private func focusWindow(_ window: NSWindow) async throws {
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        // Activation is asynchronous in a hosted test app. Wait for real
        // AppKit focus rather than mistaking Task.yield for a window event.
        for _ in 0..<100 {
            if window.isKeyWindow { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        throw XCTSkip("Cannot acquire actual Mac window focus: active=\(NSApp.isActive), hidden=\(NSApp.isHidden), policy=\(NSApp.activationPolicy().rawValue), canBecomeKey=\(window.canBecomeKey), visible=\(window.isVisible), keyWindow=\(String(describing: NSApp.keyWindow)). Notification-handler coverage runs separately.")
    }

    func testActualMacWindowKeyChangesRestoreCorrectArticle() async throws {
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { Data($0.article.zim.utf8) })
        let ownerA = UUID(), ownerB = UUID()
        let windowA = NSWindow(contentRect: NSRect(x: 20, y: 20, width: 240, height: 120), styleMask: [.titled], backing: .buffered, defer: false)
        let windowB = NSWindow(contentRect: NSRect(x: 40, y: 40, width: 240, height: 120), styleMask: [.titled], backing: .buffered, defer: false)
        windowA.isReleasedWhenClosed = false; windowB.isReleasedWhenClosed = false
        let viewA = CurrentArticleWindowObserver.ObserverView()
        let viewB = CurrentArticleWindowObserver.ObserverView()
        viewA.onKey = { context.publish(self.a, owner: ownerA, eligible: { true }) }
        viewB.onKey = { context.publish(self.b, owner: ownerB, eligible: { true }) }
        viewA.onHide = { context.clear(owner: ownerA) }
        viewB.onHide = { context.clear(owner: ownerB) }
        windowA.contentView = viewA; windowB.contentView = viewB
        defer { windowB.close(); windowA.close(); viewA.stop(); viewB.stop() }
        try await focusWindow(windowA)
        XCTAssertEqual(context.entity(owner: ownerA)?.archive, "a.zim")
        try await focusWindow(windowB)
        XCTAssertEqual(context.entity(owner: ownerB)?.archive, "b.zim")
        windowB.close()
        try await focusWindow(windowA)
        XCTAssertEqual(context.entity(owner: ownerA)?.archive, "a.zim")
        XCTAssertNil(context.entity(owner: ownerB))
    }

    func testFocusingWindowWithoutArticleRevokesPreviousWindow() async throws {
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { _ in Data("source".utf8) })
        let windowA = NSWindow(contentRect: NSRect(x: 20, y: 20, width: 240, height: 120), styleMask: [.titled], backing: .buffered, defer: false)
        let emptyWindow = NSWindow(contentRect: NSRect(x: 40, y: 40, width: 240, height: 120), styleMask: [.titled], backing: .buffered, defer: false)
        windowA.isReleasedWhenClosed = false; emptyWindow.isReleasedWhenClosed = false
        defer { emptyWindow.close(); windowA.close() }
        try await focusWindow(windowA)
        context.publish(a, owner: UUID(), eligible: { true })
        let id = try XCTUnwrap(context.current?.id)
        try await focusWindow(emptyWindow)
        XCTAssertNil(context.entity(id: id))
        do { _ = try await context.export(id: id); XCTFail("Background window article exported") } catch { }
    }

    func testWindowFocusNotificationsRevokeAndRestoreWithoutDesktopAccess() async throws {
        let context = SiriCurrentArticleContext(fingerprint: { "v1" }, loader: { Data($0.article.zim.utf8) })
        let windowA = NSWindow(contentRect: .zero, styleMask: [.titled], backing: .buffered, defer: false)
        let windowB = NSWindow(contentRect: .zero, styleMask: [.titled], backing: .buffered, defer: false)
        windowA.isReleasedWhenClosed = false; windowB.isReleasedWhenClosed = false
        var focused: NSWindow? = windowA
        context.keyWindow = { focused }
        let ownerA = UUID(), ownerB = UUID()
        let viewA = CurrentArticleWindowObserver.ObserverView()
        let viewB = CurrentArticleWindowObserver.ObserverView()
        viewA.onKey = { context.publish(self.a, owner: ownerA, eligible: { true }) }
        // B initially represents a settings/empty window with no publisher.
        windowA.contentView = viewA; windowB.contentView = viewB
        defer { viewA.stop(); viewB.stop(); windowB.close(); windowA.close() }
        NotificationCenter.default.post(name: NSWindow.didBecomeKeyNotification, object: windowA)
        let oldID = try XCTUnwrap(context.entity(owner: ownerA)?.id)
        focused = windowB
        NotificationCenter.default.post(name: NSWindow.didBecomeKeyNotification, object: windowB)
        XCTAssertNil(context.entity(id: oldID))
        do { _ = try await context.export(id: oldID); XCTFail("Old window exported after focus moved") } catch { }
        viewB.onKey = { context.publish(self.b, owner: ownerB, eligible: { true }) }
        NotificationCenter.default.post(name: NSWindow.didBecomeKeyNotification, object: windowB)
        XCTAssertEqual(context.entity(owner: ownerB)?.archive, "b.zim")
        // Same-window notifications must not revoke the newly published owner.
        NotificationCenter.default.post(name: NSWindow.didBecomeKeyNotification, object: windowB)
        XCTAssertEqual(context.entity(owner: ownerB)?.archive, "b.zim")
        focused = windowA
        NotificationCenter.default.post(name: NSWindow.didBecomeKeyNotification, object: windowA)
        XCTAssertEqual(context.entity(owner: ownerA)?.archive, "a.zim")
        XCTAssertNil(context.entity(owner: ownerB))
    }

    func testResolvedSourceRequiresPathAndDoesNotGuessFromTitle() {
        XCTAssertNil(SiriArticleReference.resolved(["title": "Mira", "zim": "a.zim"], version: "v1"))
        XCTAssertNil(SiriArticleReference.resolved(["title": "Mira", "zim": "a.zim", "path": "A/Mira", "error": "miss"], version: "v1"))
        XCTAssertEqual(SiriArticleReference.resolved(["title": "Mira", "zim": "a.zim", "path": "A/Mira"], version: "v1"), a)
    }
}

import CoreTransferable
import UniformTypeIdentifiers

/// Exercises the production chat router and source capture with small local
/// archives. No model download, GPS fix, desktop focus, or user history.
@MainActor
final class SiriDiscussionWiringTests: XCTestCase {
    private final class Reader: ZimReader, @unchecked Sendable {
        let path: String
        let title: String
        init(path: String, title: String) { self.path = path; self.title = title }
        var metadata: ZimMetadata { .init(name: "fixture") }
        var kind: ZimKind { .wikipedia }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { true }
        var hasRoutingData: Bool { false }
        func read(path: String) throws -> ZimEntry? {
            guard path == self.path else { return nil }
            return .init(path: path, title: title, mimetype: "text/html",
                         content: Data("<p>\(title) was a writer.</p><h2>Education</h2><p>\(title) studied in Paris in 1904. Orin Vale studied in Rome in 1908.</p>".utf8))
        }
        func readMainPage() throws -> ZimEntry? { nil }
        func searchTitles(query: String, limit: Int) throws -> [ZimSearchHit] {
            title.caseInsensitiveCompare(query) == .orderedSame
                ? [.init(path: path, title: title)] : []
        }
    }

    private func session() -> ChatSession {
        let service = DefaultZimService(readers: [
            (name: "first.zim", reader: Reader(path: "A/selected-biography", title: "Mira Selene")),
            (name: "second.zim", reader: Reader(path: "A/other-edition", title: "Mira Selene")),
            (name: "third.zim", reader: Reader(path: "A/another-person", title: "Orin Vale")),
        ])
        return .forTesting(providers: [MockProvider(scriptedResponse: "Fixture response.")],
            adapter: MCPToolAdapter(service: service, hasStreetzim: false),
            discussionPreparationStrategy: .none)
    }

    private func finish(_ session: ChatSession) async throws {
        let deadline = Date().addingTimeInterval(10)
        while session.isGenerating, Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        if session.isGenerating { session.stopGeneration(); XCTFail("Fixture turn timed out") }
    }

    private func discuss(_ session: ChatSession) async throws {
        session.send("Let's discuss Mira Selene")
        try await finish(session)
        XCTAssertEqual(session.siriDiscussionArticle?.article.path, "A/selected-biography",
                       session.messages.last?.text ?? "No reply")
    }

    func testResolvedDiscussionSwitchAndReset() async throws {
        let session = session()
        try await discuss(session)
        XCTAssertEqual(session.siriDiscussionArticle?.article.zim, "first.zim")
        XCTAssertEqual(session.siriDiscussionArticle?.libraryVersion, ZimfoRunner.libraryFingerprint())
        session.send("Let's discuss Orin Vale")
        try await finish(session)
        XCTAssertEqual(session.siriDiscussionArticle?.article.path, "A/another-person")
        XCTAssertEqual(session.siriDiscussionArticle?.article.zim, "third.zim")
        session.resetConversation()
        XCTAssertNil(session.siriDiscussionArticle)
    }

    func testSameTitleCardReplacesExactArchiveAndPath() async throws {
        let session = session()
        try await discuss(session)
        let card = DiscoveryThread(label: "Mira Selene", kind: .topic,
                                   source: .wikilink, zim: "second.zim")
        session.send("Mira Selene", suggestion: card)
        try await finish(session)
        XCTAssertEqual(session.siriDiscussionArticle?.article.zim, "second.zim")
        XCTAssertEqual(session.siriDiscussionArticle?.article.path, "A/other-edition")
    }

    func testFailedExplicitHandoffDoesNotRestorePreviousArticle() async throws {
        let session = session()
        try await discuss(session)
        session.send("Where was Orin Vale born?", offlineSource:
            .init(zim: "missing.zim", path: "A/missing", title: "Orin Vale"),
            offlineLibraryVersion: "stale-version")
        try await finish(session)
        XCTAssertNil(session.siriDiscussionArticle)
        XCTAssertFalse(session.messages.last?.text.isEmpty ?? true)
    }

    func testFailedArticleCardDoesNotRestorePreviousArticle() async throws {
        let session = session()
        try await discuss(session)
        session.send("Missing Person", suggestion: .init(label: "Missing Person",
            kind: .topic, source: .wikilink, zim: "second.zim"))
        try await finish(session)
        XCTAssertNil(session.siriDiscussionArticle)
    }

    func testReadButtonRevokesPreviousDiscussionEvenOnMissingArticle() async throws {
        let session = session()
        try await discuss(session)
        session.triggerArticleRead(title: "Missing Person", path: "A/missing")
        try await finish(session)
        XCTAssertNil(session.siriDiscussionArticle)
    }

    func testSpokenReadDoesNotExposePreviousDiscussion() async throws {
        let session = session()
        try await discuss(session)
        session.send("Read the whole article about Orin Vale")
        try await finish(session)
        XCTAssertNil(session.siriDiscussionArticle)
    }

    func testFailedSubjectExplorationCannotRestoreSourceThroughDiscussionFallback() async throws {
        let session = session()
        try await discuss(session)
        let question = "When did she study in Paris?"
        // A real section card establishes the previous dated question using
        // exact source evidence, independent of lexical date-answer scoring.
        session.send(question, suggestion: .init(label: "Education", kind: .topic,
            source: .section, zim: "first.zim", prompt: question,
            articleTitle: "Mira Selene", sectionTitle: "Education"))
        try await finish(session)
        XCTAssertNotNil(session.siriDiscussionArticle)
        XCTAssertTrue(session.messages.last?.text.contains("1904") == true)
        // Exercise the spoken discussion branch, not a generated topic card.
        if let index = session.messages.indices.last { session.messages[index].suggestions = [] }
        let previousIDs = Set(session.debugEntries.map(\.id))
        session.modelState = .notLoaded // Exploration's normal unavailable-model guard.
        session.send("How about Orin Vale?")
        try await finish(session)
        XCTAssertNil(session.siriDiscussionArticle, "Fallback restored the prior article")
        let trace = session.debugEntries.filter { !previousIDs.contains($0.id) }.map(\.message).joined(separator: "\n")
        XCTAssertTrue(trace.contains("follow-up evidence found before section ranking/search"), trace)
        XCTAssertFalse(trace.contains("discussion leave"), trace)
        XCTAssertFalse(trace.contains("fast-path dispatch article_overview"), trace)
    }
}

/// Explicitly selected integration checks run against the Mac app's enabled
/// local library. No model, network request, or change to library settings.
@MainActor
final class SiriCurrentArticleIntegrationTests: XCTestCase {
    func testRepeatedRealDocumentExportsRemainStableAndRevocable() async throws {
        guard #available(macOS 15.2, *) else { throw XCTSkip("Requires Transferable export API") }
        let runner = try await ZimfoRunner.load()
        guard !runner.readersByName.isEmpty else { throw XCTSkip("No enabled Mac archives") }
        let knowledge = OfflineKnowledge(service: runner.service)
        guard case .article(let article) = try await knowledge.resolve(topic: "Vladimir Putin") else {
            throw XCTSkip("This Mac's enabled archives do not resolve the probe article exactly")
        }
        let reference = SiriArticleReference(article: article, libraryVersion: runner.libraryVersion)
        let context = SiriCurrentArticleContext.shared
        let owner = UUID()
        context.publish(reference, owner: owner, eligible: { true })
        defer { context.clear(owner: owner) }
        let entity = try XCTUnwrap(context.current)
        // Keep one baseline document intentionally. Per-iteration results die
        // in the helper; the test never stores a history of exported buffers.
        let baseline = try await entity.exported(as: .plainText)
        for _ in 0..<5 { try await checkExport(entity, baseline: baseline) }
        let initialMB = MemoryStats.physFootprintMB()
        var samples = [initialMB]
        let started = ContinuousClock.now
        for index in 1...100 {
            try await checkExport(entity, baseline: baseline)
            // Allow framework autoreleases and unrelated host work to drain.
            try await Task.sleep(for: .milliseconds(1))
            if index.isMultiple(of: 20) { samples.append(MemoryStats.physFootprintMB()) }
        }
        let elapsed = started.duration(to: .now)
        XCTAssertEqual(context.entity(owner: owner)?.id, entity.id)
        context.clear(owner: owner)
        do { _ = try await entity.exported(as: .plainText); XCTFail("Repeated exports survived revocation") } catch { }
        let unresolved = try await ZimfoCurrentArticleQuery().entities(for: [entity.id])
        XCTAssertTrue(unresolved.isEmpty)
        let report = "CURRENT_ARTICLE_REPEAT_EXPORT iterations=100 bytes_each=\(baseline.count) elapsed=\(elapsed) footprint_MB=\(samples) delta_MB=\((samples.last ?? initialMB) - initialMB). Warm exports; elapsed includes 1 ms yields. Footprint is a process-level diagnostic, not a leak proof or hard memory bound."
        print(report)
        let attachment = XCTAttachment(string: report)
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func checkExport(_ entity: ZimfoCurrentArticleEntity, baseline: Data) async throws {
        guard #available(macOS 15.2, *) else { return }
        let data = try await entity.exported(as: .plainText)
        XCTAssertEqual(data, baseline, "Repeated requests must preserve the exact source document")
        XCTAssertLessThanOrEqual(data.count, OfflineKnowledge.maxDocumentBytes)
    }

    func testRealMacLibraryExportsOnlySelectedDocumentThroughTransferable() async throws {
        guard #available(macOS 15.2, *) else { throw XCTSkip("Requires Transferable export API") }
        let runner = try await ZimfoRunner.load()
        guard !runner.readersByName.isEmpty else { throw XCTSkip("No enabled Mac archives") }
        let knowledge = OfflineKnowledge(service: runner.service)
        guard case .article(let article) = try await knowledge.resolve(topic: "Vladimir Putin") else {
            throw XCTSkip("This Mac's enabled archives do not resolve the probe article exactly")
        }
        let reference = SiriArticleReference(article: article, libraryVersion: runner.libraryVersion)
        let context = SiriCurrentArticleContext.shared
        let owner = UUID()
        context.publish(reference, owner: owner, eligible: { true })
        defer { context.clear(owner: owner) }
        let entity = try XCTUnwrap(context.current)
        XCTAssertEqual(ZimfoCurrentArticleEntity.exportedContentTypes(), [.plainText])
        XCTAssertTrue(ZimfoCurrentArticleEntity.importedContentTypes().isEmpty)
        let queried = try await ZimfoCurrentArticleQuery().entities(for: [entity.id])
        XCTAssertEqual(queried.map(\.id), [entity.id])
        let started = Date()
        let data = try await entity.exported(as: .plainText)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.hasPrefix("Downloaded article: Vladimir Putin\n"))
        XCTAssertNotNil(text.range(of: #"(?i)\bborn\b[^\n]*\b(?:Leningrad|Saint Petersburg)\b"#, options: .regularExpression))
        XCTAssertLessThanOrEqual(data.count, OfflineKnowledge.maxDocumentBytes)
        let attachment = XCTAttachment(string: "Verified current-article Transferable output: \(data.count) bytes, \(Date().timeIntervalSince(started)) seconds; exact selected archive/path; birthplace text present; no transcript.")
        attachment.lifetime = .keepAlways
        add(attachment)
        print("CURRENT_ARTICLE_REAL_EXPORT bytes=\(data.count) elapsed_s=\(Date().timeIntervalSince(started))")
        context.clear(owner: owner)
        do { _ = try await entity.exported(as: .plainText); XCTFail("Closed article still exported through Transferable") } catch { }
        let afterClose = try await ZimfoCurrentArticleQuery().entities(for: [entity.id])
        XCTAssertTrue(afterClose.isEmpty)
    }
}
