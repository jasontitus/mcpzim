// SPDX-License-Identifier: MIT
//
// Siri / Shortcuts actions for offline questions, nearby places, and routes.
// Background answers use downloaded content without a language model.
// Questions carry explicit article entities; route/location actions retain
// their existing ZimfoContext state. Question handoffs are in-memory drafts
// submitted through the foreground chat UI. Xcode 27 adds system search.
// See docs/SIRI_OFFLINE_DESIGN_2026-09-13.md for contracts and limitations.

import Foundation
import AppIntents
import CoreLocation
import MCPZimKit
import Observation

// MARK: - Shared helpers

private func formatDuration(seconds: Double) -> String {
    let total = max(0, Int(seconds.rounded()))
    // Round to whole minutes before splitting, so the carry can't strand
    // 60 minutes beside an hour ("1 hours 60 minutes"), and pluralize each
    // unit independently — Siri SPEAKS this string, and it was saying
    // "1 hours 1 minutes" for 90 minutes (2026-08-13 review).
    let totalMinutes = (total + 30) / 60
    let h = totalMinutes / 60
    let m = totalMinutes % 60
    let hours = h == 1 ? "1 hour" : "\(h) hours"
    let minutes = m == 1 ? "1 minute" : "\(m) minutes"
    if h > 0 && m > 0 { return "\(hours) \(minutes)" }
    if h > 0 { return hours }
    if m < 1 { return "less than a minute" }
    return minutes
}

private func formatDistance(meters: Double) -> String {
    let useImperial = Locale.current.measurementSystem == .us
        || Locale.current.measurementSystem == .uk
    if useImperial {
        let miles = meters / 1609.344
        if miles < 0.1 {
            let feet = meters * 3.28084
            return "\(Int(feet.rounded())) feet"
        }
        return String(format: "%.1f miles", miles)
    } else {
        if meters < 100 { return "\(Int(meters.rounded())) metres" }
        return String(format: "%.1f kilometres", meters / 1000)
    }
}

// MARK: - StartRouteIntent

struct StartRouteIntent: AppIntent {
    static var title: LocalizedStringResource = "Get directions"
    static var description = IntentDescription(
        "Plan a driving route from your current location to a destination, using Zimfo's offline street data."
    )

    @Parameter(title: "Destination") var destination: String

    static var openAppWhenRun: Bool = false

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let trace = SiriInvocation("StartRouteIntent")
        defer { trace.end() }
        do {
            trace.stage("loading_library")
            let runner = try await ZimfoRunner.load()
            trace.stage("library_loaded", count: runner.readersByName.count)

            // Current location
            let origin: CLLocationCoordinate2D
            do {
                origin = try await LocationFetcher.once()
            } catch {
                return .result(dialog: IntentDialog("I couldn't get your current location. Open Zimfo to continue."))
            }

            let body = try await runner.routeFromCoords(
                originLat: origin.latitude, originLon: origin.longitude,
                destination: destination
            )

            guard let totalDist = body["distance_m"] as? Double,
                  let totalDur = body["duration_s"] as? Double,
                  let polyRaw = body["polyline"] as? [[Double]],
                  !polyRaw.isEmpty,
                  polyRaw.allSatisfy({ $0.count >= 2 })
            else {
                return .result(dialog: IntentDialog("Route planning failed."))
            }

            // Build cumulative distance table for later snap-to-route maths.
            var cum: [Double] = [0]
            cum.reserveCapacity(polyRaw.count)
            for i in 1..<polyRaw.count {
                let prev = polyRaw[i - 1], cur = polyRaw[i]
                let d = RouteProgress.haversineMetersApprox(prev[0], prev[1], cur[0], cur[1])
                cum.append(cum[i - 1] + d)
            }

            let destName = (body["destination_resolved"] as? [String: Any])?["name"] as? String
                ?? destination

            let route = ActiveRoute(
                startedAt: Date(),
                origin: .init(lat: origin.latitude, lon: origin.longitude),
                destination: .init(lat: polyRaw.last![0], lon: polyRaw.last![1]),
                originName: "here",
                destinationName: destName,
                zim: (body["zim"] as? String) ?? "",
                totalDurationSeconds: totalDur,
                totalDistanceMeters: totalDist,
                polyline: polyRaw.map { .init(lat: $0[0], lon: $0[1]) },
                cumulativeDistanceMeters: cum,
                turnByTurn: (body["turn_by_turn"] as? [String]) ?? []
            )
            await ZimfoContext.shared.setActiveRoute(route)

            let speech = "Route to \(destName), about \(formatDistance(meters: totalDist)), \(formatDuration(seconds: totalDur)). I'll remember it — say 'ask Zimfo how much longer' on the way."
            return .result(dialog: IntentDialog(stringLiteral: speech))
        } catch {
            trace.failed(error)
            throw error
        }
    }
}

// MARK: - RemainingRouteIntent

struct RemainingRouteIntent: AppIntent {
    static var title: LocalizedStringResource = "How much longer"
    static var description = IntentDescription("Report remaining time on the active route.")
    static var openAppWhenRun: Bool = false

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let trace = SiriInvocation("RemainingRouteIntent")
        defer { trace.end() }
        do {
            guard let route = await ZimfoContext.shared.activeRoute else {
                return .result(dialog: IntentDialog("No active route. Ask Zimfo for directions first."))
            }
            // Prefer a real current-location fix so we can snap to the polyline;
            // fall back to wall-clock elapsed time if location is unavailable.
            let remainingMeters: Double
            let remainingSeconds: Double
            if let here = try? await LocationFetcher.once() {
                let coord = ActiveRoute.Coordinate(lat: here.latitude, lon: here.longitude)
                await ZimfoContext.shared.updateLastLocation(coord)
                let r = RouteProgress.remaining(for: route, current: coord)
                remainingMeters = r.remainingMeters
                remainingSeconds = r.remainingSeconds
            } else {
                let elapsed = Date().timeIntervalSince(route.startedAt)
                remainingSeconds = max(0, route.totalDurationSeconds - elapsed)
                remainingMeters = route.totalDistanceMeters * (remainingSeconds / max(1, route.totalDurationSeconds))
            }

            let speech = "About \(formatDuration(seconds: remainingSeconds)), \(formatDistance(meters: remainingMeters)) to \(route.destinationName)."
            return .result(dialog: IntentDialog(stringLiteral: speech))
        } catch {
            trace.failed(error)
            throw error
        }
    }
}

// MARK: - NearbyHereIntent

struct NearbyHereIntent: AppIntent {
    static var title: LocalizedStringResource = "What's around me"
    static var openAppWhenRun: Bool = false
    static var authenticationPolicy: IntentAuthenticationPolicy = .requiresAuthentication
    @Parameter(title: "Category", default: .all) var category: ZimfoNearbyCategory

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let trace = SiriInvocation("NearbyHereIntent")
        defer { trace.end() }
        do {
            trace.stage("loading_library")
            let runner = try await ZimfoRunner.load()
            trace.stage("library_loaded", count: runner.readersByName.count)
            guard runner.readersByName.values.contains(where: { $0.kind == .streetzim }) else {
                return .result(dialog: "Open Zimfo and download or enable a StreetZim map first.")
            }
            let here: CLLocationCoordinate2D
            do { here = try await LocationFetcher.once(timeout: 5, maxAge: 30) }
            catch { return .result(dialog: "I couldn't get a recent location. Open Zimfo to allow location access, or use What's around a place.") }
            await ZimfoContext.shared.updateLastLocation(.init(lat: here.latitude, lon: here.longitude))
            let summary = try await runner.nearbySummary(lat: here.latitude, lon: here.longitude, kinds: category.kinds)
            guard runner.libraryVersion == ZimfoRunner.libraryFingerprint() else { throw OfflineKnowledge.Failure.staleArticle }
            return .result(dialog: IntentDialog(stringLiteral: summary))
        } catch {
            trace.failed(error)
            throw error
        }
    }
}

// MARK: - NearbyPlaceIntent

struct NearbyPlaceIntent: AppIntent {
    static var title: LocalizedStringResource = "What's around a place"
    @Parameter(title: "Place") var place: String
    static var openAppWhenRun: Bool = false
    static var authenticationPolicy: IntentAuthenticationPolicy = .requiresAuthentication
    @Parameter(title: "Category", default: .all) var category: ZimfoNearbyCategory

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let trace = SiriInvocation("NearbyPlaceIntent")
        defer { trace.end() }
        do {
            trace.stage("loading_library")
            let runner = try await ZimfoRunner.load()
            trace.stage("library_loaded", count: runner.readersByName.count)
            _ = try OfflineKnowledge.validatedInput(place)
            guard runner.readersByName.values.contains(where: { $0.kind == .streetzim }) else {
                return .result(dialog: "Open Zimfo and download or enable a StreetZim map first.")
            }
            let hits = try await runner.service.geocode(query: place, limit: 5, zim: nil, kinds: nil)
            guard var hit = hits.first else {
                return .result(dialog: IntentDialog(stringLiteral: "I couldn't find \(place) in any loaded map."))
            }
            if hits.count > 1 || !EntityResolutionPolicy.sameTitle(hit.name, place) {
                let labels = hits.enumerated().map { i, p in
                    "\(i + 1). \(p.name), \(p.location), \(p.subtype.isEmpty ? p.kind : p.subtype), \(String(format: "%.3f, %.3f", p.lat, p.lon))"
                }
                trace.stage("awaiting_place_choice", count: hits.count)
                let selected = try await $place.requestDisambiguation(among: labels, dialog: "Which place did you mean?")
                guard let index = labels.firstIndex(of: selected) else { throw OfflineKnowledge.Failure.staleArticle }
                hit = hits[index]
                trace.stage("place_selected")
            }
            guard runner.libraryVersion == ZimfoRunner.libraryFingerprint() else { throw OfflineKnowledge.Failure.staleArticle }
            let summary = try await runner.nearbySummary(lat: hit.lat, lon: hit.lon, kinds: category.kinds)
            guard runner.libraryVersion == ZimfoRunner.libraryFingerprint() else { throw OfflineKnowledge.Failure.staleArticle }
            return .result(dialog: IntentDialog(stringLiteral: "Near \(hit.name): \(summary)"))
        } catch {
            trace.failed(error)
            throw error
        }
    }
}

enum ZimfoNearbyCategory: String, AppEnum {
    case all, cafe, restaurant, pharmacy, hospital, hotel, museum
    static var typeDisplayRepresentation = TypeDisplayRepresentation(name: "Place category")
    static var caseDisplayRepresentations: [Self: DisplayRepresentation] = [
        .all: "All places", .cafe: "Cafés", .restaurant: "Restaurants", .pharmacy: "Pharmacies",
        .hospital: "Hospitals", .hotel: "Hotels", .museum: "Museums",
    ]
    var kinds: [String]? { self == .all ? nil : [rawValue] }
}

// MARK: - LookupTopicIntent

struct LookupTopicIntent: AppIntent {
    static var title: LocalizedStringResource = "Look up a topic"
    @Parameter(title: "Topic") var topic: String
    static var openAppWhenRun: Bool = false
    static var authenticationPolicy: IntentAuthenticationPolicy = .requiresAuthentication

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let trace = SiriInvocation("LookupTopicIntent")
        defer { trace.end() }
        do {
            trace.stage("loading_library")
            let runner = try await ZimfoRunner.load()
            trace.stage("library_loaded", count: runner.readersByName.count)
            let source: OfflineArticle
            switch try await OfflineKnowledge(service: runner.service).resolve(topic: topic) {
            case .article(let article): source = article
            case .choices(let choices):
                guard !choices.isEmpty else {
                    return .result(dialog: "No matching article was found in the downloaded library. Try a more specific title.")
                }
                let titles = choices.enumerated().map { "\($0.offset + 1). \($0.element.title) — \($0.element.zim)" }
                trace.stage("awaiting_topic_choice", count: choices.count)
                let chosen = try await $topic.requestDisambiguation(among: titles, dialog: "Which article did you mean?")
                guard let index = titles.firstIndex(of: chosen) else { throw OfflineKnowledge.Failure.staleArticle }
                source = choices[index]
                trace.stage("topic_selected")
            }
            guard runner.libraryVersion == ZimfoRunner.libraryFingerprint() else { throw OfflineKnowledge.Failure.staleArticle }
            let answer = try await OfflineKnowledge(service: runner.service).answer(question: "Tell me about \(source.title)", article: source)
            guard runner.libraryVersion == ZimfoRunner.libraryFingerprint() else { throw OfflineKnowledge.Failure.staleArticle }
            return .result(dialog: IntentDialog(stringLiteral: answer))
        } catch {
            trace.failed(error)
            throw error
        }
    }
}

// MARK: - EndRouteIntent

struct EndRouteIntent: AppIntent {
    static var title: LocalizedStringResource = "End route"
    static var openAppWhenRun: Bool = false

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let trace = SiriInvocation("EndRouteIntent")
        defer { trace.end() }
        do {
            try Task.checkCancellation()
            await ZimfoContext.shared.clearActiveRoute()
            return .result(dialog: IntentDialog("Route cleared."))
        } catch {
            trace.failed(error)
            throw error
        }
    }
}

// MARK: - App shortcuts

// Inline `\(\.$param)` substitution in phrases requires the parameter
// type to be an `AppEntity` / `AppEnum`. Our parameters are plain
// strings for now, so the shortcut phrases trigger the intent and
// Siri prompts for the value after. Natural-language invocation of
// the full form ("directions to Adams Morgan") still works via
// Shortcuts the user creates in the Shortcuts app — it fills the
// parameter at edit time.
struct ZimfoAppShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: AskOfflineQuestionIntent(), phrases: [
            "Ask an offline question with \(.applicationName)",
            "Answer an offline question with \(.applicationName)",
            "\(.applicationName) encyclopedia",
        ], shortTitle: "Offline question", systemImageName: "questionmark.bubble")
        AppShortcut(intent: ContinueQuestionInZimfoIntent(), phrases: [
            "Discuss a question with \(.applicationName)", "Continue a question in \(.applicationName)",
        ], shortTitle: "Discuss in Zimfo", systemImageName: "bubble.left.and.bubble.right")
        AppShortcut(
            intent: StartRouteIntent(),
            phrases: [
                "Get directions with \(.applicationName)",
                "\(.applicationName) directions",
                "Take me somewhere with \(.applicationName)",
            ],
            shortTitle: "Directions",
            systemImageName: "map"
        )
        AppShortcut(
            intent: RemainingRouteIntent(),
            phrases: [
                "\(.applicationName) how much longer",
                "How much longer with \(.applicationName)",
                "\(.applicationName) my ETA",
            ],
            shortTitle: "How much longer",
            systemImageName: "clock"
        )
        AppShortcut(
            intent: NearbyHereIntent(),
            phrases: [
                "What's around here with \(.applicationName)",
                "\(.applicationName) what's nearby",
                "\(.applicationName) nearby",
            ],
            shortTitle: "Nearby",
            systemImageName: "location.circle"
        )
        AppShortcut(
            intent: NearbyPlaceIntent(),
            phrases: [
                "What's around a place with \(.applicationName)",
                "\(.applicationName) what's near there",
            ],
            shortTitle: "What's around a place",
            systemImageName: "map.circle"
        )
        AppShortcut(
            intent: LookupTopicIntent(),
            phrases: [
                "Look something up with \(.applicationName)",
                "\(.applicationName) tell me about a topic",
            ],
            shortTitle: "Look up",
            systemImageName: "book"
        )
        AppShortcut(
            intent: EndRouteIntent(),
            phrases: [
                "\(.applicationName) end route",
                "\(.applicationName) clear route",
            ],
            shortTitle: "End route",
            systemImageName: "xmark.circle"
        )
    }
}

// MARK: - Offline questions and explicit conversational context

/// IDs carry a library generation and exact source path. Display/excerpt data
/// returned to Siri is never trusted when the entity is used again.
struct ZimfoArticleEntity: AppEntity {
    static var typeDisplayRepresentation = TypeDisplayRepresentation(name: "Offline article")
    static var defaultQuery = ZimfoArticleQuery()
    let id: String
    @Property(title: "Title") var title: String
    @Property(title: "Archive") var archive: String
    @Property(title: "Source excerpt") var excerpt: String
    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(title)", subtitle: "\(archive)")
    }
    struct Reference: Codable {
        let version: String
        let article: OfflineArticle
    }
    init(article: OfflineArticle, version: String, excerpt: String = "") throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        id = try encoder.encode(Reference(version: version, article: article)).base64EncodedString()
        title = article.title; archive = article.zim; self.excerpt = excerpt
    }
    @MainActor
    func validated(using runner: ZimfoRunner) async throws -> OfflineArticle {
        guard id.utf8.count <= 8192, let bytes = Data(base64Encoded: id),
              let ref = try? JSONDecoder().decode(Reference.self, from: bytes),
              ref.version == runner.libraryVersion,
              ref.version == ZimfoRunner.libraryFingerprint(),
              runner.readersByName[ref.article.zim] != nil,
              ref.article.path.count <= 2048, ref.article.title.count <= 500
        else { throw OfflineKnowledge.Failure.staleArticle }
        let source = try await runner.service.articleSection(path: ref.article.path, section: "lead", zim: ref.article.zim)
        guard EntityResolutionPolicy.sameTitle(source.title, ref.article.title),
              !ArticleHeuristics.isDisambiguationArticle(title: source.title, leadText: source.section.text)
        else { throw OfflineKnowledge.Failure.staleArticle }
        return ref.article
    }
}

struct ZimfoArticleQuery: EntityStringQuery {
    @MainActor
    func entities(for identifiers: [String]) async throws -> [ZimfoArticleEntity] {
        let trace = SiriInvocation("article_query_ids")
        defer { trace.end() }
        do {
            trace.stage("loading_library")
            let runner = try await ZimfoRunner.load()
            trace.stage("library_loaded", count: runner.readersByName.count)
            var result: [ZimfoArticleEntity] = []
            for id in identifiers.prefix(5) {
                guard id.utf8.count <= 8192, let data = Data(base64Encoded: id),
                      let ref = try? JSONDecoder().decode(ZimfoArticleEntity.Reference.self, from: data),
                      ref.version == runner.libraryVersion else { continue }
                let candidate = try ZimfoArticleEntity(article: ref.article, version: ref.version)
                if let source = try? await candidate.validated(using: runner) {
                    result.append(try ZimfoArticleEntity(article: source, version: ref.version))
                }
            }
            return result
        } catch { trace.failed(error); throw error }
    }
    @MainActor
    func entities(matching string: String) async throws -> [ZimfoArticleEntity] {
        let trace = SiriInvocation("article_query_text")
        defer { trace.end() }
        do {
            trace.stage("loading_library")
            let runner = try await ZimfoRunner.load()
            trace.stage("library_loaded", count: runner.readersByName.count)
            let resolution = try await OfflineKnowledge(service: runner.service).resolve(topic: string)
            let sources: [OfflineArticle]
            switch resolution {
            case .article(let source): sources = [source]
            case .choices(let choices): sources = choices
            }
            return try sources.map { try ZimfoArticleEntity(article: $0, version: runner.libraryVersion) }
        } catch { trace.failed(error); throw error }
    }
    // Do not enumerate/import a user's library into Siri or Spotlight.
    @MainActor
    func suggestedEntities() async throws -> [ZimfoArticleEntity] {
        let trace = SiriInvocation("article_query_suggestions")
        defer { trace.end() }
        return []
    }
}

struct AskOfflineQuestionIntent: AppIntent {
    static var title: LocalizedStringResource = "Ask an offline question"
    static var description = IntentDescription("Ask a question using downloaded Wikipedia or Wikipedia articles bundled in StreetZim. For follow-ups, pass the article returned by a previous answer. No model or internet lookup is used by this action.")
    static var openAppWhenRun = false
    static var authenticationPolicy: IntentAuthenticationPolicy = .requiresAuthentication
    @Parameter(title: "Question", requestValueDialog: "What would you like to know?") var question: String?
    @Parameter(title: "Article", requestValueDialog: "Which offline article should I use?") var article: ZimfoArticleEntity?
    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<ZimfoArticleEntity?> & ProvidesDialog {
        let trace = SiriInvocation("AskOfflineQuestionIntent")
        defer { trace.end() }
        do {
            let suppliedQuestion: String
            if let question {
                suppliedQuestion = question
            } else {
                trace.stage("awaiting_question")
                suppliedQuestion = try await $question.requestValue("What would you like to know?")
            }
            trace.stage("question_received", count: suppliedQuestion.count)
            let question = try OfflineKnowledge.validatedInput(suppliedQuestion)
            let hadExplicitArticle = article != nil
            trace.stage("loading_library")
            var runner = try await ZimfoRunner.load()
            trace.stage("library_loaded", count: runner.readersByName.count)
            guard runner.readersByName.values.contains(where: {
                [.wikipedia, .mdwiki, .streetzim].contains($0.kind)
            }) else { throw OfflineKnowledge.Failure.noLibrary }
            let knowledge = OfflineKnowledge(service: runner.service)
            var selected = article
            let inferred = OfflineKnowledge.topic(in: question)
            trace.stage(inferred == nil ? "topic_not_inferred" : "topic_inferred")
            if selected == nil, let inferred {
                switch try await knowledge.resolve(topic: inferred) {
                case .article(let source):
                    trace.stage("exact_article_match")
                    selected = try ZimfoArticleEntity(article: source, version: runner.libraryVersion)
                case .choices(let choices):
                    trace.stage("article_candidates", count: choices.count)
                    if !choices.isEmpty {
                        trace.stage("awaiting_article_choice", count: choices.count)
                        selected = try await $article.requestDisambiguation(
                            among: choices.map { try ZimfoArticleEntity(article: $0, version: runner.libraryVersion) },
                            dialog: IntentDialog("Which article did you mean?"))
                    }
                }
            }
            if selected == nil {
                trace.stage("awaiting_article")
                selected = try await $article.requestValue("Which offline article should I use? Try its title.")
            }
            guard let selected else { throw OfflineKnowledge.Failure.staleArticle }
            trace.stage("article_selected")
            // Siri may spend minutes clarifying; reload after that suspension.
            runner = try await ZimfoRunner.load()
            let source = try await selected.validated(using: runner)
            if hadExplicitArticle, let inferred, !EntityResolutionPolicy.sameTitle(inferred, source.title) {
                throw QuestionFailure.conflictingTopic
            }
            let effectiveQuestion = OfflineKnowledge.confirmedQuestion(question, requestedTopic: inferred, article: source)
            let answer = try await OfflineKnowledge(service: runner.service).answer(question: effectiveQuestion, article: source) { measurement in
                await trace.measure(measurement)
            }
            try Task.checkCancellation()
            guard runner.libraryVersion == ZimfoRunner.libraryFingerprint() else { throw OfflineKnowledge.Failure.staleArticle }
            let result = try ZimfoArticleEntity(article: source, version: runner.libraryVersion, excerpt: answer)
            return .result(value: Optional(result), dialog: IntentDialog(stringLiteral: answer))
        } catch {
            trace.failed(error)
            if let explanation = Self.explainedFailure(error) {
                trace.stage("explained_failure")
                return .result(value: Optional<ZimfoArticleEntity>.none,
                               dialog: IntentDialog(stringLiteral: explanation))
            }
            throw error
        }
    }
    static func explainedFailure(_ error: Error) -> String? {
        if let failure = error as? OfflineKnowledge.Failure { return failure.errorDescription }
        if let failure = error as? QuestionFailure { return failure.errorDescription }
        if error is ZimServiceError { return "I couldn't read that article in the enabled offline library. Open Zimfo and choose an available article." }
        return nil
    }
    enum QuestionFailure: LocalizedError {
        case conflictingTopic
        var errorDescription: String? { "Your question names a different topic. Clear the Article parameter or choose the article you want to ask about." }
    }
}

@MainActor @Observable
final class SiriQuestionHandoff {
    static let shared = SiriQuestionHandoff()
    struct Draft: Identifiable {
        let id = UUID()
        let question: String
        let source: OfflineArticle?
        let libraryVersion: String?
        var articleTitle: String? { source?.title }
    }
    var pending: Draft?
    struct Search: Identifiable {
        let id = UUID()
        let query: String
        let articles: [ZimfoArticleEntity]
    }
    var search: Search?
}

struct ContinueQuestionInZimfoIntent: AppIntent {
    static var title: LocalizedStringResource = "Continue a question in Zimfo"
    static var description = IntentDescription("Open Zimfo with a question ready to add to the conversation. Your current draft is preserved; tap Use in message, then Send. Download content and a supported model beforehand for offline conversation.")
    static var openAppWhenRun = true
    static var authenticationPolicy: IntentAuthenticationPolicy = .requiresAuthentication
    @Parameter(title: "Question") var question: String
    @Parameter(title: "Article") var article: ZimfoArticleEntity?
    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let trace = SiriInvocation("ContinueQuestionInZimfoIntent")
        defer { trace.end() }
        do {
            let question = try OfflineKnowledge.validatedInput(question)
            let source: OfflineArticle?
            let version: String?
            if let article {
                trace.stage("loading_library")
                let runner = try await ZimfoRunner.load()
                trace.stage("library_loaded", count: runner.readersByName.count)
                source = try await article.validated(using: runner)
                version = runner.libraryVersion
            } else { source = nil; version = nil }
            guard SiriQuestionHandoff.shared.pending == nil else { throw HandoffFailure.pending }
            SiriQuestionHandoff.shared.pending = .init(question: question, source: source, libraryVersion: version)
            return .result(dialog: "Your question is ready in Zimfo. Tap Use in message, then Send to continue.")
        } catch {
            trace.failed(error)
            throw error
        }
    }
    enum HandoffFailure: LocalizedError {
        case pending
        var errorDescription: String? { "A question is already waiting in Zimfo. Use or dismiss it before opening another." }
    }
}

// Xcode 27's schema is intentionally separated from the compatible shortcuts.
// A generic Q&A action must not masquerade as a document editor or messenger.
#if compiler(>=6.4)
@available(iOS 27.0, macOS 27.0, *)
@AppIntent(schema: .system.searchInApp)
struct SearchOfflineContentIntent: ShowInAppSearchResultsIntent {
    static var searchScopes: [StringSearchScope] = [.general]
    static var openAppWhenRun = true
    static var authenticationPolicy: IntentAuthenticationPolicy = .requiresLocalDeviceAuthentication
    var criteria: StringSearchCriteria
    @MainActor
    func perform() async throws -> some IntentResult {
        let trace = SiriInvocation("SearchOfflineContentIntent")
        defer { trace.end() }
        do {
            let query = try OfflineKnowledge.validatedInput(criteria.term)
            let results = try await ZimfoArticleQuery().entities(matching: query)
            SiriQuestionHandoff.shared.search = .init(query: query, articles: results)
            return .result()
        } catch {
            trace.failed(error)
            throw error
        }
    }
}
#endif


extension SiriInvocation {
    func measure(_ measurement: OfflineKnowledge.Measurement) {
        switch measurement {
        case .htmlBytes(let n): stage("html_bytes", count: n)
        case .parsedBytes(let n): stage("parsed_bytes", count: n)
        case .sections(let n): stage("sections", count: n)
        case .answerCharacters(let n): stage("answer_characters", count: n)
        }
    }
}

#if DEBUG
/// Opt-in direct-action smoke using the user's requested public subject.
/// This does not test Siri speech routing or its presentation of the result.
@MainActor
enum SiriEinsteinSmoke {
    private static var didRun = false
    static func run() async {
        guard !didRun else { return }
        didRun = true
        let trace = SiriInvocation("Einstein_direct_intent_smoke")
        defer { trace.end() }
        var intent = AskOfflineQuestionIntent()
        intent.question = "Tell me about Albert Einstein"
        do {
            let result = try await intent.perform()
            let article = result.value ?? nil
            let answered = article?.excerpt.hasPrefix("From the downloaded article") == true
            trace.stage(answered ? "smoke_answer_returned" : "smoke_failed_no_answer", count: article?.excerpt.count ?? 0)
        } catch { trace.failed(error) }
    }
}

@MainActor
enum SiriPutinBirthplaceSmoke {
    private static var didRun = false
    static func run() async {
        guard !didRun else { return }
        didRun = true
        let trace = SiriInvocation("Putin_birthplace_direct_intent_smoke")
        defer { trace.end() }
        var intent = AskOfflineQuestionIntent()
        intent.question = "Tell me where Vladimir Putin was born"
        do {
            let result = try await intent.perform()
            let article = result.value ?? nil
            let excerpt = article?.excerpt ?? ""
            let correctSubject = article?.title == "Vladimir Putin"
            let birthplace = excerpt.range(of: #"(?i)\bborn\b.*\b(?:Leningrad|Saint Petersburg)\b"#,
                                           options: .regularExpression) != nil
            trace.stage(correctSubject && birthplace ? "smoke_birthplace_verified" : "smoke_failed_birthplace",
                        count: excerpt.count)
        } catch { trace.failed(error) }
    }
}
#endif
