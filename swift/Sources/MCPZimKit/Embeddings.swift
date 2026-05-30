// SPDX-License-Identifier: MIT
//
// On-device semantic recall for conversational drift — the "build a little
// embeddings DB from the articles we touch" idea, done incrementally.
//
// Nothing here is pre-built or bundled: the index is empty at launch and grows
// as the user opens articles and nearby-place leads during the walk. That
// keeps it tiny (kilobytes per article), private, and shaped exactly to the
// user's interests — far cheaper than embedding all ~6M Wikipedia articles.
//
// The embedding *model* is pluggable via `TextEmbedder`. The iOS app supplies
// a Core ML / `NLEmbedding` implementation for real sentence semantics; a
// dependency-free `HashingEmbedder` ships here as a baseline so the feature
// (and its tests) work with zero model assets.
//
// What it powers (see CONVERSATIONAL_REDESIGN.md):
//   * fuzzy follow-ups that don't keyword-match ("the romantic one");
//   * ranking drift threads by similarity to the WHOLE conversation
//     (centroid of the focus), not just the last sentence;
//   * "what does this remind you of" lateral jumps across touched topics.
//
// Pure Swift + Foundation; the actor is the only stateful piece. Fully
// exercised by `swift test`.

import Foundation

// MARK: - Vector math

public enum VectorMath {

    /// L2-normalise. A zero vector is returned unchanged (cosine against it is
    /// defined as 0 by the helpers below).
    public static func normalized(_ v: [Float]) -> [Float] {
        var sum: Float = 0
        for x in v { sum += x * x }
        let n = sum.squareRoot()
        guard n > 0 else { return v }
        return v.map { $0 / n }
    }

    /// Dot product. With normalised inputs this equals cosine similarity.
    public static func dot(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var s: Float = 0
        var i = 0
        while i < n { s += a[i] * b[i]; i += 1 }
        return s
    }

    /// Cosine similarity of two (not-necessarily-normalised) vectors.
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let na = magnitude(a), nb = magnitude(b)
        guard na > 0, nb > 0 else { return 0 }
        return dot(a, b) / (na * nb)
    }

    /// Element-wise mean of equal-length vectors (the centroid), normalised.
    /// Returns nil for an empty input.
    public static func meanNormalized(_ vs: [[Float]]) -> [Float]? {
        guard let first = vs.first else { return nil }
        var acc = [Float](repeating: 0, count: first.count)
        var used = 0
        for v in vs where v.count == acc.count {
            for i in 0..<acc.count { acc[i] += v[i] }
            used += 1
        }
        guard used > 0 else { return nil }
        for i in 0..<acc.count { acc[i] /= Float(used) }
        return normalized(acc)
    }

    static func magnitude(_ v: [Float]) -> Float {
        var s: Float = 0
        for x in v { s += x * x }
        return s.squareRoot()
    }
}

// MARK: - Embedder

/// Turns text into a dense vector. The host conforms a Core ML / NLEmbedding
/// model to this; `HashingEmbedder` is the zero-asset fallback.
public protocol TextEmbedder: Sendable {
    var dimension: Int { get }
    func embed(_ text: String) -> [Float]
}

/// Dependency-free feature-hashing embedder. Not a semantic model — it maps
/// shared vocabulary to shared dimensions, so it captures lexical overlap
/// (good enough to rank "Roman architecture" nearer "the Colosseum" than
/// "blood plasma"). Deterministic across runs (FNV-1a, not Swift's randomised
/// `Hasher`) so a persisted index stays valid. Useful as a real baseline and
/// to keep the pipeline testable without model assets.
public struct HashingEmbedder: TextEmbedder {
    public let dimension: Int

    public init(dimension: Int = 512) {
        self.dimension = max(16, dimension)
    }

    public func embed(_ text: String) -> [Float] {
        var v = [Float](repeating: 0, count: dimension)
        for token in Self.tokenize(text) {
            bump(&v, token)
            // Subword char 3-grams (boundary-padded) so morphological
            // variants share features: "efficient"~"efficiency",
            // "perovskite"~"perovskites". Exact-token hashing alone misses
            // these, leaving the answer-bearing section unranked ("I don't
            // see it" even though the article was pulled — real capture
            // 2026-05-30: "how efficient are they?").
            if token.count >= 4 {
                let padded = Array("<" + token + ">")
                var j = 0
                while j + 3 <= padded.count {
                    bump(&v, "n:" + String(padded[j ..< j + 3]))
                    j += 1
                }
            }
        }
        return VectorMath.normalized(v)
    }

    /// Hash `feature` into a dimension with a signed contribution (second
    /// hash bit) — the standard hashing-trick sign that reduces collision
    /// bias vs. always-adding.
    private func bump(_ v: inout [Float], _ feature: String) {
        let h = Int(Self.fnv1a(feature) % UInt64(dimension))
        let sign: Float = (Self.fnv1a("#" + feature) & 1) == 0 ? 1 : -1
        v[h] += sign
    }

    static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !($0.isLetter || $0.isNumber) })
            .map(String.init)
            .filter { $0.count >= 2 }
    }

    static func fnv1a(_ s: String) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in s.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x00000100000001B3
        }
        return hash
    }
}

// MARK: - Index

/// The incremental touch-index: vectors keyed by ZIM path (or any stable
/// string), capped LRU so memory stays bounded on a long walk. Vectors are
/// stored normalised, so nearest-neighbour is a dot product.
public actor EmbeddingIndex {

    public struct Entry: Sendable, Equatable {
        public let key: String
        public let title: String
        public let vector: [Float]
    }

    public struct Hit: Sendable, Equatable {
        public let key: String
        public let title: String
        public let score: Float
    }

    private var entries: [Entry] = []   // oldest-first; append = most recent
    private let maxEntries: Int

    public init(maxEntries: Int = 2000) {
        self.maxEntries = max(1, maxEntries)
    }

    public var count: Int { entries.count }

    public func contains(_ key: String) -> Bool {
        entries.contains { $0.key == key }
    }

    /// Insert or refresh an entry. Re-adding a key moves it to most-recent and
    /// updates its vector. Stored normalised. Over-cap drops the oldest.
    public func add(key: String, title: String, vector: [Float]) {
        guard !key.isEmpty, !vector.isEmpty else { return }
        let v = VectorMath.normalized(vector)
        entries.removeAll { $0.key == key }
        entries.append(Entry(key: key, title: title, vector: v))
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }
    }

    /// k nearest entries to a query vector, highest cosine first, optionally
    /// excluding keys already in play.
    public func nearest(
        to query: [Float], k: Int, excluding: Set<String> = []
    ) -> [Hit] {
        guard k > 0 else { return [] }
        let q = VectorMath.normalized(query)
        let scored = entries.compactMap { e -> Hit? in
            if excluding.contains(e.key) { return nil }
            return Hit(key: e.key, title: e.title,
                       score: VectorMath.dot(q, e.vector))
        }
        return Array(scored.sorted { $0.score > $1.score }.prefix(k))
    }

    /// The normalised centroid of the given keys — the "gist" of everything
    /// discussed so far when you pass the focus's entity keys.
    public func centroid(of keys: [String]) -> [Float]? {
        let set = Set(keys)
        let vs = entries.filter { set.contains($0.key) }.map(\.vector)
        return VectorMath.meanNormalized(vs)
    }

    /// Cosine of each requested key against `query`, for callers that want to
    /// re-rank an existing candidate list (e.g. drift threads) rather than do
    /// a global kNN. Missing keys are simply absent from the result.
    public func scores(for keys: [String], against query: [Float]) -> [String: Float] {
        let q = VectorMath.normalized(query)
        var out: [String: Float] = [:]
        for e in entries where keys.contains(e.key) {
            out[e.key] = VectorMath.dot(q, e.vector)
        }
        return out
    }
}
