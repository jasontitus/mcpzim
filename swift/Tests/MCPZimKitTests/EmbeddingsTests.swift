// SPDX-License-Identifier: MIT
//
// Tests for the on-device semantic-recall layer: vector math, the
// dependency-free hashing embedder, the incremental index actor, and the
// thread re-ranking helper.

import Foundation
import XCTest

@testable import MCPZimKit

final class EmbeddingsTests: XCTestCase {

    // MARK: - VectorMath

    func testCosineEndpoints() {
        let a: [Float] = [1, 2, 3]
        XCTAssertEqual(VectorMath.cosine(a, a), 1, accuracy: 1e-5)
        XCTAssertEqual(VectorMath.cosine([1, 0], [0, 1]), 0, accuracy: 1e-5)
        XCTAssertEqual(VectorMath.cosine([1, 0], [-1, 0]), -1, accuracy: 1e-5)
    }

    func testNormalizedIsUnitLength() {
        let n = VectorMath.normalized([3, 4])
        XCTAssertEqual(VectorMath.magnitude(n), 1, accuracy: 1e-5)
    }

    func testMeanNormalized() {
        XCTAssertNil(VectorMath.meanNormalized([]))
        let m = VectorMath.meanNormalized([[1, 0], [0, 1]])
        XCTAssertEqual(VectorMath.magnitude(m ?? []), 1, accuracy: 1e-5)
    }

    // MARK: - HashingEmbedder

    func testHashingEmbedderDeterministicAndNormalized() {
        let e = HashingEmbedder(dimension: 128)
        let v1 = e.embed("Stanford Memorial Church")
        let v2 = e.embed("Stanford Memorial Church")
        XCTAssertEqual(v1, v2, "embedding must be deterministic across calls")
        XCTAssertEqual(VectorMath.magnitude(v1), 1, accuracy: 1e-5)
    }

    func testHashingEmbedderCapturesLexicalOverlap() {
        let e = HashingEmbedder(dimension: 512)
        let colosseum = e.embed("the Colosseum ancient Roman amphitheatre in Rome")
        let rome = e.embed("ancient Rome Roman empire architecture")
        let plasma = e.embed("blood plasma medical haematology transfusion")
        let near = VectorMath.cosine(colosseum, rome)
        let far = VectorMath.cosine(colosseum, plasma)
        XCTAssertGreaterThan(near, far,
            "shared Rome/Roman vocab should score nearer than an unrelated topic")
    }

    // MARK: - EmbeddingIndex

    func testIndexAddDedupeAndCap() async {
        let idx = EmbeddingIndex(maxEntries: 3)
        let e = HashingEmbedder(dimension: 64)
        await idx.add(key: "A/One", title: "One", vector: e.embed("one"))
        await idx.add(key: "A/Two", title: "Two", vector: e.embed("two"))
        await idx.add(key: "A/One", title: "One", vector: e.embed("one again"))
        var c = await idx.count
        XCTAssertEqual(c, 2, "re-adding a key must not grow the index")
        await idx.add(key: "A/Three", title: "Three", vector: e.embed("three"))
        await idx.add(key: "A/Four", title: "Four", vector: e.embed("four"))
        c = await idx.count
        XCTAssertEqual(c, 3, "LRU cap enforced")
        // Re-adding "A/One" moved it to most-recent, so the oldest remaining
        // entry ("A/Two") is the one evicted past the cap — not "A/One".
        let hasTwo = await idx.contains("A/Two")
        XCTAssertFalse(hasTwo, "oldest entry dropped past the cap")
        let hasOne = await idx.contains("A/One")
        XCTAssertTrue(hasOne, "recently-refreshed entry survives eviction")
    }

    func testIndexNearestAndExclusion() async {
        let idx = EmbeddingIndex()
        let e = HashingEmbedder(dimension: 512)
        await idx.add(key: "A/Rome", title: "Rome", vector: e.embed("ancient Rome Roman empire"))
        await idx.add(key: "A/Plasma", title: "Plasma", vector: e.embed("blood plasma medical"))
        let q = e.embed("Roman architecture in Rome")
        let top = await idx.nearest(to: q, k: 1)
        XCTAssertEqual(top.first?.key, "A/Rome")
        let excluded = await idx.nearest(to: q, k: 1, excluding: ["A/Rome"])
        XCTAssertEqual(excluded.first?.key, "A/Plasma")
    }

    func testCentroidAndScores() async {
        let idx = EmbeddingIndex()
        let e = HashingEmbedder(dimension: 256)
        await idx.add(key: "A/Rome", title: "Rome", vector: e.embed("ancient Rome"))
        await idx.add(key: "A/Italy", title: "Italy", vector: e.embed("Italy Mediterranean"))
        let centroid = await idx.centroid(of: ["A/Rome", "A/Italy"])
        XCTAssertNotNil(centroid)
        let scores = await idx.scores(for: ["A/Rome", "A/Italy"], against: e.embed("Rome"))
        XCTAssertNotNil(scores["A/Rome"])
        XCTAssertGreaterThan(scores["A/Rome"] ?? 0, scores["A/Italy"] ?? 1)
    }

    // MARK: - orderBySimilarity

    func testOrderBySimilarityReordersAndIsStable() {
        let threads = [
            DiscoveryThread(label: "Far", kind: .topic, source: .wikilink, zimPath: "A/Far"),
            DiscoveryThread(label: "Near", kind: .topic, source: .wikilink, zimPath: "A/Near"),
            DiscoveryThread(label: "Unscored", kind: .topic, source: .section),
        ]
        let scores: [String: Float] = ["A/Far": 0.1, "A/Near": 0.9]
        let ordered = ConversationThreads.orderBySimilarity(threads, scores: scores)
        XCTAssertEqual(ordered.map(\.label), ["Near", "Far", "Unscored"],
            "highest score first; unscored keeps trailing original order")
    }
}
