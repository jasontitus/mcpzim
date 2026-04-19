// SPDX-License-Identifier: MIT
//
// Protocol that abstracts the underlying ZIM reader. MCPZimKit itself does
// NOT depend on libzim — the iOS / macOS app supplies a concrete reader by
// linking a C++ libzim via CoreKiwix.xcframework, an Obj-C++ bridge module,
// or (eventually) a pure-Swift implementation.
//
// Host apps only need to implement `read(path:)` and `searchTitles(query:limit:)`
// to get every tool in MCPZimServerKit; full-text search is optional.

import Foundation

/// Broad classification of a ZIM archive. Drives conditional tool registration
/// in MCPZimServerKit (routing tools appear only when a `.streetzim` reader
/// is present).
public enum ZimKind: String, Sendable, CaseIterable {
    case wikipedia
    case mdwiki
    case streetzim
    case generic
}

public struct ZimMetadata: Sendable {
    public let name: String
    public let title: String
    public let description: String
    public let language: String
    public let creator: String
    public let publisher: String
    public let date: String
    public let tags: [String]
    public let articleCount: Int

    public init(
        name: String = "",
        title: String = "",
        description: String = "",
        language: String = "",
        creator: String = "",
        publisher: String = "",
        date: String = "",
        tags: [String] = [],
        articleCount: Int = 0
    ) {
        self.name = name
        self.title = title
        self.description = description
        self.language = language
        self.creator = creator
        self.publisher = publisher
        self.date = date
        self.tags = tags
        self.articleCount = articleCount
    }
}

public struct ZimEntry: Sendable {
    public let path: String
    public let title: String
    public let mimetype: String
    public let content: Data

    public init(path: String, title: String, mimetype: String, content: Data) {
        self.path = path
        self.title = title
        self.mimetype = mimetype
        self.content = content
    }
}

public struct ZimSearchHit: Sendable {
    public let path: String
    public let title: String

    public init(path: String, title: String) {
        self.path = path
        self.title = title
    }
}

/// A single opened ZIM archive. Implementations are expected to be thread-safe
/// — MCPZimServerKit may call `read` concurrently from multiple tool handlers.
public protocol ZimReader: AnyObject, Sendable {
    var metadata: ZimMetadata { get }
    var kind: ZimKind { get }
    var hasFullTextIndex: Bool { get }
    var hasTitleIndex: Bool { get }
    /// Signals whether this reader can serve streetzim-specific tools. Default
    /// implementations of `ZimService` gate route tools on this.
    var hasRoutingData: Bool { get }

    func read(path: String) throws -> ZimEntry?
    func readMainPage() throws -> ZimEntry?
    func searchFullText(query: String, limit: Int) throws -> [ZimSearchHit]
    func searchTitles(query: String, limit: Int) throws -> [ZimSearchHit]
}

public extension ZimReader {
    func searchFullText(query: String, limit: Int) throws -> [ZimSearchHit] { [] }
    func searchTitles(query: String, limit: Int) throws -> [ZimSearchHit] { [] }

    /// Best available search: full-text first, then title suggestions.
    func search(query: String, limit: Int) throws -> [ZimSearchHit] {
        if hasFullTextIndex {
            let hits = try searchFullText(query: query, limit: limit)
            if !hits.isEmpty { return hits }
        }
        if hasTitleIndex {
            return try searchTitles(query: query, limit: limit)
        }
        return []
    }
}

/// Classify a ZIM by its metadata + presence of signature entries. Mirrors
/// the Python `library.classify()` function so both hosts agree on kinds.
public func classifyZim(
    filename: String,
    metadata: ZimMetadata,
    hasRoutingEntry: Bool,
    hasStreetzimConfig: Bool
) -> ZimKind {
    let tags = Set(metadata.tags.map { $0.lowercased() })
    let fn = filename.lowercased()
    let nm = metadata.name.lowercased()
    let ti = metadata.title.lowercased()
    let cr = metadata.creator.lowercased()
    let pb = metadata.publisher.lowercased()

    if hasRoutingEntry || hasStreetzimConfig { return .streetzim }
    if nm.hasPrefix("streetzim") || nm.hasPrefix("street_")
        || fn.hasPrefix("streetzim") || tags.contains("streetzim") {
        return .streetzim
    }
    if tags.contains("mdwiki") || tags.contains("medical") { return .mdwiki }
    if nm.hasPrefix("mdwiki") || fn.hasPrefix("mdwiki")
        || pb.contains("wikiprojectmed") || ti.contains("mdwiki") {
        return .mdwiki
    }
    if tags.contains("wikipedia") || tags.contains("_category:wikipedia") {
        return .wikipedia
    }
    if nm.hasPrefix("wikipedia") || fn.hasPrefix("wikipedia") || cr == "wikipedia" {
        return .wikipedia
    }
    return .generic
}
