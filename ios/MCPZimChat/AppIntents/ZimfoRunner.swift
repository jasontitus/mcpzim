// SPDX-License-Identifier: MIT
//
// Lightweight tool executor used by App Intents. Opens whatever ZIMs are
// recorded in the library (via ChatSession's existing persisted
// bookmarks) and dispatches `MCPZimKit` tools directly — no language
// model, no UI. Kept separate from `ChatSession` so Siri can invoke
// intents even before (or without) the main UI launching.

import Foundation
import MCPZimKit

@MainActor
final class ZimfoRunner {
    let service: DefaultZimService
    let adapter: MCPToolAdapter
    /// Filename → reader, for resolving back after a tool answers.
    let readersByName: [String: any ZimReader]

    static func load() async throws -> ZimfoRunner {
        var readers: [(name: String, reader: any ZimReader)] = []
        // 1) Anything in the app's sandbox Documents folder (auto-scan).
        let fm = FileManager.default
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                  appropriateFor: nil, create: false) {
            let urls = (try? fm.contentsOfDirectory(at: docs,
                                                    includingPropertiesForKeys: nil))?
                .filter { $0.pathExtension.lowercased() == "zim" } ?? []
            for url in urls {
                if let r = try? LibzimReader(url: url) {
                    readers.append((url.lastPathComponent, r))
                }
            }
        }
        // 2) External bookmarks stored by the main app (same key as
        //    ChatSession.persistBookmarks()).
        if let blobs = UserDefaults.standard.array(forKey: "library.externalBookmarks") as? [Data] {
            for blob in blobs {
                var stale = false
                #if os(macOS)
                let url = try? URL(resolvingBookmarkData: blob,
                                   options: [.withSecurityScope],
                                   relativeTo: nil,
                                   bookmarkDataIsStale: &stale)
                #else
                let url = try? URL(resolvingBookmarkData: blob,
                                   options: [],
                                   relativeTo: nil,
                                   bookmarkDataIsStale: &stale)
                #endif
                if let url, url.startAccessingSecurityScopedResource(),
                   let r = try? LibzimReader(url: url) {
                    readers.append((url.lastPathComponent, r))
                }
            }
        }
        let service = DefaultZimService(readers: readers)
        let adapter = await MCPToolAdapter.from(service: service)
        // Same bridge ChatSession uses — lets Siri intents that end up
        // calling `route_status` / `what_is_here` read from the same
        // persistent route + GPS state as the in-app chat.
        await adapter.installHostStateProvider {
            await ZimfoContext.shared.mcpSnapshot()
        }
        let byName = Dictionary(uniqueKeysWithValues: readers.map { ($0.name, $0.reader) })
        return ZimfoRunner(service: service, adapter: adapter, readersByName: byName)
    }

    init(service: DefaultZimService, adapter: MCPToolAdapter,
         readersByName: [String: any ZimReader]) {
        self.service = service
        self.adapter = adapter
        self.readersByName = readersByName
    }

    // MARK: - Canned flows

    func routeFromCoords(
        originLat: Double, originLon: Double,
        destination: String
    ) async throws -> ([String: Any]) {
        // Geocode destination, then plan_driving_route.
        let hits = try await service.geocode(query: destination, limit: 1, zim: nil, kinds: nil)
        guard let dest = hits.first else { throw ZimServiceError.noMatch(destination) }
        let route = try await service.planDrivingRoute(RouteRequest(
            originLat: originLat, originLon: originLon,
            destLat: dest.lat, destLon: dest.lon,
            zim: nil
        ))
        // Re-use the encoded form of route so we can reach into
        // polyline / distance fields uniformly with the in-app path.
        var body = await adapter.dispatchLocal_plan(req: RouteRequest(
            originLat: originLat, originLon: originLon,
            destLat: dest.lat, destLon: dest.lon, zim: nil
        ))
        body["destination_resolved"] = [
            "name": dest.name, "lat": dest.lat, "lon": dest.lon,
            "type": dest.kind
        ] as [String: Any]
        _ = route
        return body
    }

    /// Search for a topic in a wikipedia ZIM and return the article's
    /// lead paragraph (first chunk of text before the first empty line
    /// or after 400 chars). No LLM involved.
    func lookupTopicLead(topic: String) async throws -> (title: String, lead: String)? {
        let hits = try await service.search(query: topic, limit: 1, kind: .wikipedia)
        guard let hit = hits.first else { return nil }
        let article = try await service.article(path: hit.path, zim: hit.zim)
        let text = article.text
        let lead = Self.firstLead(of: text)
        return (hit.title, lead)
    }

    private static func firstLead(of text: String) -> String {
        // Naive: take first paragraph up to ~400 chars.
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lines = trimmed.split(separator: "\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        for line in lines {
            let s = line.trimmingCharacters(in: .whitespaces)
            if s.count > 40 { return String(s.prefix(400)) }
        }
        return String(trimmed.prefix(400))
    }

    /// Places near a lat/lon with optional category filter. Returns
    /// a compact list pre-formatted for TTS.
    func nearbySummary(lat: Double, lon: Double, limit: Int = 5) async throws -> String {
        let result = try await service.nearPlaces(
            lat: lat, lon: lon, radiusKm: 1.0,
            limit: limit, kinds: nil, zim: nil
        )
        guard !result.results.isEmpty else { return "I don't see anything notable nearby." }
        let topBreakdown = result.breakdown
            .sorted { $0.value > $1.value }.prefix(3)
            .map { "\($0.value) \($0.key)" }.joined(separator: ", ")
        let names = result.results.prefix(limit).enumerated().map { (i, pair) -> String in
            let dist = Int(pair.distanceMeters.rounded())
            return "\(i + 1). \(pair.place.name) (\(dist)m)"
        }.joined(separator: "; ")
        if result.totalInRadius > result.results.count {
            return "Within 1km: \(result.totalInRadius) total (\(topBreakdown)). Nearest: \(names)."
        }
        return "Nearby: \(names)."
    }
}

// Helper to synthesize the route body WITHOUT model-facing trimming —
// we need the full polyline to persist into ZimfoContext.
extension MCPToolAdapter {
    func dispatchLocal_plan(req: RouteRequest) async -> [String: Any] {
        (try? await self.dispatch(tool: "plan_driving_route", args: [
            "origin_lat": req.originLat,
            "origin_lon": req.originLon,
            "dest_lat": req.destLat,
            "dest_lon": req.destLon,
        ])) ?? [:]
    }
}
