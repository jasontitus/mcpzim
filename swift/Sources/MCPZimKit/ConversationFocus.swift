// SPDX-License-Identifier: MIT
//
// Discourse state for the "walking companion" conversation model.
//
// The old design treated each turn in isolation: `IntentRouter` matched
// the raw text against a fixed set of shapes and dispatched one tool. There
// was nowhere for "it", "the second one", or "tell me more" to resolve,
// because nothing in the code remembered what the conversation was *about*.
//
// `ConversationFocus` is that memory. The host (`ChatSession`) owns one
// instance and mutates it each turn:
//
//   * `entities`   — most-recent-first stack of subjects in play, each with
//                    its ZIM path / coordinates so a follow-up can re-fetch
//                    the EXACT thing without re-searching.
//   * `lastList`   — the enumerated list we last showed the user, in display
//                    order, so "the second one" / "the other church" resolve
//                    deterministically rather than via the 4B model's guess.
//   * `openThreads`— vetted topic-drift candidates surfaced from the last
//                    result (the deterministic half of the hybrid drift
//                    engine — see `ConversationThreads`).
//   * `here`+`trail`— current GPS plus a short movement trail, so the host
//                    can tell the user has walked into a new area.
//
// Everything here is pure value-semantics: `Sendable`, `Equatable`, no UI or
// ZIM dependency, so it is exercised by `swift test` alongside the rest of
// the kit.

import Foundation

/// A subject the conversation is (or was recently) about.
public struct FocusEntity: Equatable, Sendable, Hashable {
    public enum Kind: String, Sendable, Hashable {
        /// A Wikipedia subject — person, event, concept, named place that we
        /// reach encyclopedically (article-first).
        case topic
        /// A geocoded location / POI we reach map-first (has coordinates).
        case place
        /// An active driving route (origin → destination).
        case route
    }

    /// Canonical display + lookup name ("Stanford Memorial Church"). This is
    /// what we substitute for a pronoun and what we feed back to a fetch.
    public var name: String
    public var kind: Kind
    /// ZIM entry path when known, so a follow-up re-fetches the exact article
    /// instead of re-running search and possibly landing on a variant.
    public var zimPath: String?
    public var lat: Double?
    public var lon: Double?
    /// Turn index at which this entity was last brought into focus. Higher =
    /// more recent. Used for recency ordering and "did the topic just change".
    public var lastTouchedTurn: Int

    public init(
        name: String,
        kind: Kind,
        zimPath: String? = nil,
        lat: Double? = nil,
        lon: Double? = nil,
        lastTouchedTurn: Int = 0
    ) {
        self.name = name
        self.kind = kind
        self.zimPath = zimPath
        self.lat = lat
        self.lon = lon
        self.lastTouchedTurn = lastTouchedTurn
    }

    /// Case/whitespace-folded key for dedupe + reference matching.
    public var matchKey: String {
        name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// A vetted "where you could go next" candidate. Always grounded in a real
/// artefact from a tool result (a wikilink that exists in the article, a POI
/// the geocoder actually returned) — never invented by the model.
public struct DiscoveryThread: Equatable, Sendable, Hashable {
    public enum Source: String, Sendable, Hashable {
        /// Outbound link from the article body to another article.
        case wikilink
        /// A POI returned by a places/stories tool.
        case nearbyPlace
        /// A section heading worth expanding within the current article.
        case section
        /// A related entity surfaced by a relations-article probe.
        case relation
    }

    /// What we offer the user and what they can name back to pick it up.
    public var label: String
    public var kind: FocusEntity.Kind
    public var source: Source
    public var zimPath: String?
    public var lat: Double?
    public var lon: Double?
    /// Optional one-line gloss for the offer ("the architect", "350 m away").
    public var note: String?

    public init(
        label: String,
        kind: FocusEntity.Kind,
        source: Source,
        zimPath: String? = nil,
        lat: Double? = nil,
        lon: Double? = nil,
        note: String? = nil
    ) {
        self.label = label
        self.kind = kind
        self.source = source
        self.zimPath = zimPath
        self.lat = lat
        self.lon = lon
        self.note = note
    }

    public var matchKey: String {
        label.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Promote a thread the user picked into a focus entity.
    public func asEntity(turn: Int) -> FocusEntity {
        FocusEntity(
            name: label, kind: kind, zimPath: zimPath,
            lat: lat, lon: lon, lastTouchedTurn: turn
        )
    }
}

public struct ConversationFocus: Equatable, Sendable {

    public struct Coord: Equatable, Sendable, Hashable {
        public var lat: Double
        public var lon: Double
        public init(lat: Double, lon: Double) {
            self.lat = lat
            self.lon = lon
        }
    }

    /// Monotonic user-turn counter. `beginUserTurn` bumps it; entities and
    /// threads stamp it so recency is comparable across the whole session.
    public private(set) var turn: Int = 0

    /// Most-recent-first, deduped, bounded. `entities.first` is the primary
    /// subject a bare pronoun / elliptical follow-up binds to.
    public private(set) var entities: [FocusEntity] = []

    /// The enumerated list last shown to the user (places hits / search hits
    /// / compared titles), in DISPLAY order — index 0 is "the first one".
    public private(set) var lastList: [FocusEntity] = []

    /// Vetted drift candidates from the last result.
    public private(set) var openThreads: [DiscoveryThread] = []

    /// Current GPS fix and a short most-recent-first movement trail.
    public private(set) var here: Coord?
    public private(set) var trail: [Coord] = []

    static let maxEntities = 12
    static let maxTrail = 8
    /// Minimum move (metres) before we push a new trail point — filters GPS
    /// jitter so "have I moved" means actual walking, not noise.
    static let trailEpsilonM: Double = 25

    public init() {}

    // MARK: - Queries

    /// The subject a bare "it"/"that"/elliptical follow-up resolves to.
    public var primaryEntity: FocusEntity? { entities.first }

    /// Most-recent entity of a given kind (e.g. the active place even if a
    /// topic was discussed more recently).
    public func mostRecent(kind: FocusEntity.Kind) -> FocusEntity? {
        entities.first { $0.kind == kind }
    }

    public var isEmpty: Bool {
        entities.isEmpty && lastList.isEmpty
    }

    // MARK: - Mutation

    public mutating func beginUserTurn() { turn += 1 }

    /// Bring `entity` to the front (most-recent), folding any prior mention of
    /// the same name+kind so the stack stays deduped and recency-ordered.
    public mutating func remember(_ entity: FocusEntity) {
        var e = entity
        if e.lastTouchedTurn == 0 { e.lastTouchedTurn = turn }
        entities.removeAll { $0.matchKey == e.matchKey && $0.kind == e.kind }
        entities.insert(e, at: 0)
        if entities.count > Self.maxEntities {
            entities.removeLast(entities.count - Self.maxEntities)
        }
    }

    /// Replace the "list on screen". Pass the items in display order. The
    /// first item also becomes the primary entity (the user was just shown
    /// it), but the whole list is retained for ordinal selection.
    public mutating func setLastList(_ items: [FocusEntity]) {
        lastList = items
        if let head = items.first { remember(head) }
    }

    public mutating func setThreads(_ threads: [DiscoveryThread]) {
        openThreads = threads
    }

    public mutating func clearThreads() { openThreads = [] }

    /// Record a GPS fix. Pushes onto the trail only if we've moved more than
    /// `trailEpsilonM` from the last recorded point (jitter filter).
    public mutating func updateLocation(lat: Double, lon: Double) {
        let c = Coord(lat: lat, lon: lon)
        defer { here = c }
        guard let last = trail.first else {
            trail = [c]
            return
        }
        if Self.haversineMeters(last, c) >= Self.trailEpsilonM {
            trail.insert(c, at: 0)
            if trail.count > Self.maxTrail {
                trail.removeLast(trail.count - Self.maxTrail)
            }
        }
    }

    /// Straight-line distance (metres) the user has moved since `coord`.
    public func movedMeters(since coord: Coord) -> Double {
        guard let here else { return 0 }
        return Self.haversineMeters(coord, here)
    }

    // MARK: - Geo

    static func haversineMeters(_ a: Coord, _ b: Coord) -> Double {
        let R = 6_371_000.0
        let dLat = (b.lat - a.lat) * .pi / 180
        let dLon = (b.lon - a.lon) * .pi / 180
        let la1 = a.lat * .pi / 180
        let la2 = b.lat * .pi / 180
        let h = sin(dLat / 2) * sin(dLat / 2)
            + cos(la1) * cos(la2) * sin(dLon / 2) * sin(dLon / 2)
        return 2 * R * asin(min(1, sqrt(h)))
    }
}
