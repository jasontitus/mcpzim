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

    /// Memoized instance + the library fingerprint it was built from.
    /// `load()` used to rebuild everything per App Intent — re-scanning
    /// Documents, re-opening every multi-GB ZIM archive (metadata +
    /// title/fulltext index reads), re-resolving bookmarks, and
    /// re-constructing the tool stack — adding seconds of disk I/O to
    /// every Siri/Shortcuts invocation for an identical library.
    private static var cached: ZimfoRunner?
    private static var cachedFingerprint: String?

    static func load() async throws -> ZimfoRunner {
        let fingerprint = libraryFingerprint()
        if let cached, cachedFingerprint == fingerprint {
            return cached
        }
        let runner = try await buildFresh()
        cached = runner
        cachedFingerprint = fingerprint
        return runner
    }

    /// Cheap change-detection key: ZIM filename + size + mtime for everything
    /// in Documents, plus the bookmark blob signatures. No archive opens —
    /// just one directory listing (with the two stat values the enumerator
    /// already prefetches) and a defaults read. (`Data.hashValue` is
    /// per-process seeded, which is fine for an in-memory, per-process cache.)
    ///
    /// Size + mtime are part of the key because the name alone isn't: a `.zim`
    /// replaced by a newer edition under the SAME filename (re-download of a
    /// monthly Wikipedia dump, a friend's nearby-share overwrite) looked
    /// identical here, so `load()` kept handing Siri readers onto the old file
    /// for the rest of the process (review 2026-08-13, bugs ZimfoRunner:43).
    private static func libraryFingerprint() -> String {
        var parts: [String] = []
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                  appropriateFor: nil, create: false),
           let urls = try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: keys) {
            parts = urls.filter { $0.pathExtension.lowercased() == "zim" }
                .map { url in
                    let values = try? url.resourceValues(forKeys: Set(keys))
                    let size = values?.fileSize ?? -1
                    let mtime = values?.contentModificationDate?.timeIntervalSince1970 ?? -1
                    return "\(url.lastPathComponent):\(size):\(Int(mtime))"
                }
                .sorted()
        }
        if let blobs = UserDefaults.standard.array(forKey: "library.externalBookmarks") as? [Data] {
            for b in blobs { parts.append("bm:\(b.hashValue)") }
        }
        return parts.joined(separator: "|")
    }

    private static func buildFresh() async throws -> ZimfoRunner {
        // Archive opening is detached because it is blocking disk I/O —
        // libzim metadata plus title/fulltext index reads, "seconds" for a
        // multi-GB library. `ZimfoRunner` is `@MainActor`, so this used to run
        // on the main actor and the first Siri/Shortcuts intent of a fresh
        // process (empty `cached`) paid all of it there, risking Siri's
        // execution budget / an in-app freeze (review 2026-08-13, "Perf: fix
        // first" #1). Everything after the detach is actor hops or cheap
        // bookkeeping, so the main actor only ever suspends.
        let readers = await Task.detached(priority: .userInitiated) {
            openReaders()
        }.value
        let service = DefaultZimService(readers: readers)
        let adapter = await MCPToolAdapter.from(service: service)
        // Same bridge ChatSession uses — lets Siri intents that end up
        // calling `route_status` / `what_is_here` read from the same
        // persistent route + GPS state as the in-app chat.
        await adapter.installHostStateProvider {
            await ZimfoContext.shared.mcpSnapshot()
        }
        let byName = Dictionary(
            readers.map { ($0.name, $0.reader) },
            uniquingKeysWith: { first, _ in first })
        return ZimfoRunner(service: service, adapter: adapter, readersByName: byName)
    }

    /// Opens every ZIM the library knows about. `nonisolated` so `buildFresh`
    /// can run it off the main actor; `LibzimReader` is `@unchecked Sendable`,
    /// so the opened readers cross back to the main actor safely.
    private nonisolated static func openReaders() -> [(name: String, reader: any ZimReader)] {
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
                guard let url, url.startAccessingSecurityScopedResource() else { continue }
                if let r = try? LibzimReader(url: url) {
                    // Scope stays open on purpose: the reader mmaps the file
                    // for as long as the runner lives.
                    readers.append((url.lastPathComponent, r))
                } else {
                    // …but a failed open leaves nothing to read through it, so
                    // balance the start here instead of leaking the scope for
                    // the process lifetime (review 2026-08-13, bugs
                    // ZimfoRunner:90).
                    url.stopAccessingSecurityScopedResource()
                }
            }
        }
        return readers
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
