// SPDX-License-Identifier: MIT

import Foundation

#if canImport(FirebaseCore)
import FirebaseAnalytics
import FirebaseCore
import FirebaseCrashlytics
#if canImport(FirebasePerformance) && !os(macOS)
import FirebasePerformance
#endif
#endif

/// Privacy-safe product and performance telemetry.
///
/// Deliberately never accepts query text, article titles, paths, or ZIM
/// filenames. All dimensions are bounded enums/identifiers so Analytics does
/// not become an accidental record of what a user reads offline.
@MainActor
enum AppTelemetry {
    struct LibraryProfile: Sendable {
        let mix: String
        let wikipediaVariant: String
        let zimCount: Int
        let hasStreetZim: Bool

        init(kinds: [String], filenames: [String]) {
            let normalizedKinds = Set(kinds.map { $0.lowercased() })
            mix = normalizedKinds.isEmpty
                ? "none"
                : normalizedKinds.sorted().joined(separator: "+")
            zimCount = kinds.count
            hasStreetZim = normalizedKinds.contains("streetzim")

            let wikipediaNames = zip(kinds, filenames).compactMap { pair -> String? in
                let (kind, name) = pair
                let normalizedKind = kind.lowercased()
                guard normalizedKind == "wikipedia" || normalizedKind == "mdwiki"
                else { return nil }
                return name.lowercased()
            }
            if wikipediaNames.isEmpty {
                wikipediaVariant = "none"
            } else if wikipediaNames.contains(where: {
                $0.contains("_nopic_") || $0.contains("_nopic.")
            }) {
                wikipediaVariant = "nopic"
            } else if wikipediaNames.contains(where: { $0.contains("_maxi_") }) {
                wikipediaVariant = "maxi"
            } else if wikipediaNames.contains(where: { $0.contains("_mini_") }) {
                wikipediaVariant = "mini"
            } else {
                wikipediaVariant = "unknown"
            }
        }
    }

    static func configure() {
        #if canImport(FirebaseCore)
        guard FirebaseApp.app() == nil else { return }
        #if os(macOS)
        guard let path = Bundle.main.path(
            forResource: "GoogleService-Info-Mac", ofType: "plist"),
              let options = FirebaseOptions(contentsOfFile: path)
        else {
            assertionFailure("Missing GoogleService-Info-Mac.plist")
            return
        }
        FirebaseApp.configure(options: options)
        #else
        // Mirror the macOS guard: bare `configure()` throws an uncaught
        // ObjC exception when the plist is absent, so any build made
        // without it (the plists are gitignored after the 2026-08-03
        // secret incident — a fresh clone has none) crashed at launch
        // instead of simply running without telemetry (2026-08-13 review).
        guard Bundle.main.path(
            forResource: "GoogleService-Info", ofType: "plist") != nil
        else {
            assertionFailure("Missing GoogleService-Info.plist")
            return
        }
        FirebaseApp.configure()
        #endif

        // The generated plist can retain the legacy analytics flag even after
        // the Firebase project is linked. The runtime setting is authoritative.
        Analytics.setAnalyticsCollectionEnabled(true)
        Crashlytics.crashlytics().setCustomValue(platform, forKey: "platform")
        Crashlytics.crashlytics().log("Zimfo telemetry initialized")
        #endif
    }

    static func startQuery(
        type: String,
        modelID: String,
        library: LibraryProfile
    ) -> QueryTrace {
        QueryTrace(
            queryType: bounded(type),
            modelID: bounded(modelID),
            library: library)
    }

    @MainActor
    final class QueryTrace {
        private let queryType: String
        private let modelID: String
        private let library: LibraryProfile
        private let startedAt = ProcessInfo.processInfo.systemUptime
        private var firstResponseAt: TimeInterval?
        private var route = "unknown"
        private var primaryTool = "none"
        private var toolTime: TimeInterval = 0
        private var toolCount = 0
        private var usedZimKinds: Set<String> = []
        private var finished = false

        #if canImport(FirebasePerformance) && !os(macOS)
        private let performanceTrace: Trace?
        #endif

        fileprivate init(
            queryType: String,
            modelID: String,
            library: LibraryProfile
        ) {
            self.queryType = queryType
            self.modelID = modelID
            self.library = library

            #if canImport(FirebasePerformance) && !os(macOS)
            performanceTrace = Performance.startTrace(name: "query_\(queryType)")
            performanceTrace?.setValue(modelID, forAttribute: "model_id")
            performanceTrace?.setValue(
                library.wikipediaVariant, forAttribute: "wiki_variant")
            #endif

            #if canImport(FirebaseAnalytics)
            Analytics.logEvent("query_started", parameters: [
                "query_type": queryType,
                "model_id": modelID,
                "library_mix": library.mix,
                "wiki_variant": library.wikipediaVariant,
                "zim_count": library.zimCount,
                "has_streetzim": library.hasStreetZim ? 1 : 0,
            ])
            Crashlytics.crashlytics().setCustomValue(queryType, forKey: "last_query_type")
            Crashlytics.crashlytics().setCustomValue(library.mix, forKey: "last_library_mix")
            Crashlytics.crashlytics().log("Query started: \(queryType)")
            #endif
        }

        func setRoute(_ value: String, primaryTool: String? = nil) {
            route = AppTelemetry.bounded(value)
            if let primaryTool, self.primaryTool == "none" {
                self.primaryTool = AppTelemetry.bounded(primaryTool)
            }
        }

        func markFirstResponse() {
            guard firstResponseAt == nil else { return }
            firstResponseAt = ProcessInfo.processInfo.systemUptime
        }

        func recordTool(
            name: String,
            duration: TimeInterval,
            usedZimKinds: [String] = []
        ) {
            if primaryTool == "none" {
                primaryTool = AppTelemetry.bounded(name)
            }
            toolCount += 1
            toolTime += max(0, duration)
            self.usedZimKinds.formUnion(
                usedZimKinds.map { AppTelemetry.bounded($0.lowercased()) })
        }

        func finish(cancelled: Bool, responseCharacters: Int) {
            guard !finished else { return }
            finished = true
            let endedAt = ProcessInfo.processInfo.systemUptime
            let totalMS = Self.milliseconds(endedAt - startedAt)
            let firstResponseMS = firstResponseAt.map {
                Self.milliseconds($0 - startedAt)
            } ?? -1
            let toolMS = Self.milliseconds(toolTime)
            let usedZimMix = usedZimKinds.isEmpty
                ? "unknown"
                : usedZimKinds.sorted().joined(separator: "+")

            #if canImport(FirebasePerformance) && !os(macOS)
            performanceTrace?.setValue(route, forAttribute: "route")
            performanceTrace?.setValue(primaryTool, forAttribute: "primary_tool")
            performanceTrace?.setValue(usedZimMix, forAttribute: "used_zim_mix")
            if firstResponseMS >= 0 {
                performanceTrace?.incrementMetric(
                    "first_response_ms", by: Int64(firstResponseMS))
            }
            performanceTrace?.incrementMetric("tool_time_ms", by: Int64(toolMS))
            performanceTrace?.incrementMetric("tool_count", by: Int64(toolCount))
            performanceTrace?.incrementMetric(
                "response_chars", by: Int64(max(0, responseCharacters)))
            performanceTrace?.stop()
            #endif

            #if canImport(FirebaseAnalytics)
            Analytics.logEvent("query_completed", parameters: [
                "query_type": queryType,
                "route": route,
                "primary_tool": primaryTool,
                "model_id": modelID,
                "library_mix": library.mix,
                "used_zim_mix": usedZimMix,
                "wiki_variant": library.wikipediaVariant,
                "zim_count": library.zimCount,
                "has_streetzim": library.hasStreetZim ? 1 : 0,
                "total_ms": totalMS,
                "first_response_ms": firstResponseMS,
                "tool_time_ms": toolMS,
                "tool_count": toolCount,
                "response_chars": max(0, responseCharacters),
                "cancelled": cancelled ? 1 : 0,
            ])
            Crashlytics.crashlytics().setCustomValue(route, forKey: "last_query_route")
            Crashlytics.crashlytics().setCustomValue(
                usedZimMix, forKey: "last_used_zim_mix")
            Crashlytics.crashlytics().setCustomValue(totalMS, forKey: "last_query_total_ms")
            Crashlytics.crashlytics().log(
                "Query completed: \(queryType), route=\(route), total_ms=\(totalMS)")
            #endif
        }

        private static func milliseconds(_ seconds: TimeInterval) -> Int {
            Int((max(0, seconds) * 1_000).rounded())
        }
    }

    private static var platform: String {
        #if os(macOS)
        return "macos"
        #else
        return "ios"
        #endif
    }

    /// Firebase permits longer values, but a small bound prevents accidental
    /// high-cardinality dimensions if a future caller passes an uncontrolled ID.
    private static func bounded(_ value: String) -> String {
        String(value.prefix(64))
    }
}
