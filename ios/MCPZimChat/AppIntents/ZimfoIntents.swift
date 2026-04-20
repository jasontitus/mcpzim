// SPDX-License-Identifier: MIT
//
// Siri / Shortcuts App Intents. Four deterministic intents that don't
// need a language model on the Siri path:
//   - StartRouteIntent           "Ask Zimfo how to get to Adams Morgan from here"
//   - RemainingRouteIntent       "Ask Zimfo how much longer"
//   - NearbyHereIntent           "Ask Zimfo what's around here"
//   - NearbyPlaceIntent          "Ask Zimfo what's around Adams Morgan"
//   - LookupTopicIntent          "Ask Zimfo about the War of 1812"
//
// All intents read/write `ZimfoContext` on disk so Siri calls carry
// context across invocations even after the app has been killed.

import Foundation
import AppIntents
import CoreLocation
import MCPZimKit

// MARK: - Shared helpers

private func formatDuration(seconds: Double) -> String {
    let total = max(0, Int(seconds.rounded()))
    let h = total / 3600
    let m = (total % 3600 + 30) / 60
    if h > 0 && m > 0 { return "\(h) hours \(m) minutes" }
    if h > 0 { return h == 1 ? "1 hour" : "\(h) hours" }
    if m < 1 { return "less than a minute" }
    return "\(m) minutes"
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
        let runner = try await ZimfoRunner.load()

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
              !polyRaw.isEmpty
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
    }
}

// MARK: - RemainingRouteIntent

struct RemainingRouteIntent: AppIntent {
    static var title: LocalizedStringResource = "How much longer"
    static var description = IntentDescription("Report remaining time on the active route.")
    static var openAppWhenRun: Bool = false

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
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
    }
}

// MARK: - NearbyHereIntent

struct NearbyHereIntent: AppIntent {
    static var title: LocalizedStringResource = "What's around me"
    static var openAppWhenRun: Bool = false

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let runner = try await ZimfoRunner.load()
        let here = try await LocationFetcher.once()
        await ZimfoContext.shared.updateLastLocation(.init(lat: here.latitude, lon: here.longitude))
        let summary = try await runner.nearbySummary(lat: here.latitude, lon: here.longitude)
        return .result(dialog: IntentDialog(stringLiteral: summary))
    }
}

// MARK: - NearbyPlaceIntent

struct NearbyPlaceIntent: AppIntent {
    static var title: LocalizedStringResource = "What's around a place"
    @Parameter(title: "Place") var place: String
    static var openAppWhenRun: Bool = false

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let runner = try await ZimfoRunner.load()
        let hits = try await runner.service.geocode(query: place, limit: 1, zim: nil, kinds: nil)
        guard let hit = hits.first else {
            return .result(dialog: IntentDialog(stringLiteral: "I couldn't find \(place) in any loaded map."))
        }
        let summary = try await runner.nearbySummary(lat: hit.lat, lon: hit.lon)
        return .result(dialog: IntentDialog(stringLiteral: "Near \(hit.name): \(summary)"))
    }
}

// MARK: - LookupTopicIntent

struct LookupTopicIntent: AppIntent {
    static var title: LocalizedStringResource = "Look up a topic"
    @Parameter(title: "Topic") var topic: String
    static var openAppWhenRun: Bool = false

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let runner = try await ZimfoRunner.load()
        guard let result = try await runner.lookupTopicLead(topic: topic) else {
            return .result(dialog: IntentDialog(stringLiteral: "I couldn't find anything about \(topic) in Wikipedia."))
        }
        return .result(dialog: IntentDialog(stringLiteral: "\(result.title): \(result.lead)"))
    }
}

// MARK: - EndRouteIntent

struct EndRouteIntent: AppIntent {
    static var title: LocalizedStringResource = "End route"
    static var openAppWhenRun: Bool = false

    func perform() async throws -> some IntentResult & ProvidesDialog {
        await ZimfoContext.shared.clearActiveRoute()
        return .result(dialog: IntentDialog("Route cleared."))
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
                "Search \(.applicationName)",
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
