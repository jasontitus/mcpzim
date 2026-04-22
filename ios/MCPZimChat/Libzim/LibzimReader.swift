// SPDX-License-Identifier: MIT
//
// ZimReader implementation backed by the Kiwix C++ libzim (bundled inside
// CoreKiwix.xcframework). The actual C++ calls live in `LibzimBridge.mm`
// (Obj-C++) and are exposed to Swift as a thin Obj-C class.
//
// This file compiles *without* the xcframework — the bridging-header check
// below replaces the reader with a stub so the rest of the app keeps
// building until you drop CoreKiwix.xcframework into MCPZimChat/Frameworks/.

import Foundation
import MCPZimKit

#if canImport(CoreKiwix)
import CoreKiwix

/// Real reader: opens a ZIM file via libzim and answers `ZimReader` calls.
public final class LibzimReader: ZimReader, @unchecked Sendable {
    public static let isLinked = true
    private let archive: ZimArchive   // Obj-C++ wrapper around zim::Archive
    private let url: URL

    public init(url: URL) throws {
        self.url = url
        self.archive = try ZimArchive(fileURL: url)
        self.metadata = Self.readMetadata(archive)
        self.kind = classifyZim(
            filename: url.lastPathComponent,
            metadata: metadata,
            hasRoutingEntry: archive.hasEntry("routing-data/graph.bin")
                || archive.hasEntry("routing-data/graph.json"),
            hasStreetzimConfig: archive.hasEntry("map-config.json")
        )
        self.hasFullTextIndex = archive.hasFulltextIndex
        self.hasTitleIndex = archive.hasTitleIndex
        self.hasRoutingData = archive.hasEntry("routing-data/graph.bin")
        // Cap libzim's per-archive cluster cache. The default (~32 MB in
        // recent libzim) suits Wikipedia's small article clusters, but
        // streetzim packs its entire 700 MB routing graph into a single
        // huge cluster — without a cap that cluster would linger in the
        // LRU long after `SZRGGraph.parse` is done with it, pinning
        // ~700 MB of resident memory we already extracted everything
        // we need from. Set the cap small enough that post-read eviction
        // happens promptly; reads still pay full peak during the single
        // `getData()` call, but steady-state drops back to arrays-only.
        let cacheCapBytes: UInt = 64 * 1024 * 1024   // 64 MB
        archive.setClusterCacheMaxSizeBytes(cacheCapBytes)
    }

    public let metadata: ZimMetadata
    public let kind: ZimKind
    public let hasFullTextIndex: Bool
    public let hasTitleIndex: Bool
    public let hasRoutingData: Bool

    public func read(path: String) throws -> ZimEntry? {
        guard let raw = archive.readEntry(atPath: path) else { return nil }
        return ZimEntry(path: raw.path, title: raw.title, mimetype: raw.mimetype, content: raw.content)
    }

    public func readMainPage() throws -> ZimEntry? {
        guard let raw = archive.readMainPage() else { return nil }
        return ZimEntry(path: raw.path, title: raw.title, mimetype: raw.mimetype, content: raw.content)
    }

    public func searchFullText(query: String, limit: Int) throws -> [ZimSearchHit] {
        archive.searchFulltext(query, limit: Int32(limit)).map {
            ZimSearchHit(path: $0.path, title: $0.title)
        }
    }

    public func searchTitles(query: String, limit: Int) throws -> [ZimSearchHit] {
        archive.suggestTitles(query, limit: Int32(limit)).map {
            ZimSearchHit(path: $0.path, title: $0.title)
        }
    }

    private static func readMetadata(_ archive: ZimArchive) -> ZimMetadata {
        ZimMetadata(
            name: archive.metadataValue("Name") ?? "",
            title: archive.metadataValue("Title") ?? "",
            description: archive.metadataValue("Description") ?? "",
            language: archive.metadataValue("Language") ?? "",
            creator: archive.metadataValue("Creator") ?? "",
            publisher: archive.metadataValue("Publisher") ?? "",
            date: archive.metadataValue("Date") ?? "",
            tags: (archive.metadataValue("Tags") ?? "")
                .split(separator: ";")
                .map { $0.trimmingCharacters(in: .whitespaces) },
            articleCount: Int(archive.articleCount)
        )
    }
}

#else

/// Stub used until you bundle CoreKiwix.xcframework. The chat app falls back
/// to a user-visible warning instead of failing to build.
public final class LibzimReader: ZimReader, @unchecked Sendable {
    public static let isLinked = false
    public let metadata = ZimMetadata()
    public let kind: ZimKind = .generic
    public let hasFullTextIndex = false
    public let hasTitleIndex = false
    public let hasRoutingData = false

    public init(url: URL) throws {
        throw LibzimError.notLinked
    }

    public func read(path: String) throws -> ZimEntry? { throw LibzimError.notLinked }
    public func readMainPage() throws -> ZimEntry? { throw LibzimError.notLinked }
    public func searchFullText(query: String, limit: Int) throws -> [ZimSearchHit] { [] }
    public func searchTitles(query: String, limit: Int) throws -> [ZimSearchHit] { [] }
}

public enum LibzimError: Error, CustomStringConvertible {
    case notLinked
    public var description: String {
        "CoreKiwix.xcframework is not linked. See ios/README.md for the vendoring steps."
    }
}

#endif
