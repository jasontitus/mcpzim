// SPDX-License-Identifier: MIT

import Foundation
import XCTest
@testable import MCPZimKit

final class HotSplitGeocoderTests: XCTestCase {
    private final class TrackingReader: ZimReader, @unchecked Sendable {
        private let store: [String: Data]
        private let lock = NSLock()
        private var paths: [String] = []

        init(store: [String: Data]) { self.store = store }
        var metadata: ZimMetadata { ZimMetadata(name: "osm-test") }
        var kind: ZimKind { .streetzim }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { false }
        var hasRoutingData: Bool { true }

        func read(path: String) throws -> ZimEntry? {
            lock.lock(); paths.append(path); lock.unlock()
            guard let data = store[path] else { return nil }
            return ZimEntry(path: path, title: path,
                            mimetype: "application/json", content: data)
        }
        func readMainPage() throws -> ZimEntry? { nil }

        func searchChunkReads() -> [String] {
            lock.lock(); defer { lock.unlock() }
            return paths.filter {
                $0.hasPrefix("search-data/st-") && $0.hasSuffix(".json")
            }
        }
    }

    private func fixture() throws -> (DefaultZimService, TrackingReader, Int) {
        let leaves = (0..<16).flatMap { first in
            (0..<16).map { second in "st-\(String(first, radix: 16))-\(String(second, radix: 16))" }
        }
        let targetBucket = Geocoder.subBucketFor(name: "Stanford University")
        let targetLeaf = "st-\(String(targetBucket, radix: 16))-f"
        var chunks = Dictionary(uniqueKeysWithValues: leaves.map { ($0, 1) })
        chunks[targetLeaf] = 1
        let manifest: [String: Any] = [
            "chunks": chunks,
            "sub_chunks": ["st": leaves],
        ]
        var store: [String: Data] = [
            "search-data/manifest.json": try JSONSerialization.data(withJSONObject: manifest),
        ]
        for leaf in leaves {
            let records: [[String: Any]] = leaf == targetLeaf ? [[
                "n": "Stanford University", "t": "poi", "s": "university",
                "a": 37.4275, "o": -122.1697, "l": "Stanford, California",
            ]] : [[
                "n": "Street fixture \(leaf)", "t": "street",
                "a": 0.0, "o": 0.0,
            ]]
            store["search-data/\(leaf).json"] = try JSONSerialization.data(
                withJSONObject: records)
        }
        let reader = TrackingReader(store: store)
        return (DefaultZimService(readers: [("osm-test", reader)]), reader, targetBucket)
    }

    func testExactCanonicalNameOnlyReadsPredictedFirstLevelBranch() async throws {
        let (service, reader, bucket) = try fixture()
        let hits = try await service.geocode(
            query: "Stanford University", limit: 1, zim: nil,
            kinds: ["place", "poi"])
        XCTAssertEqual(hits.first?.name, "Stanford University")
        let reads = reader.searchChunkReads()
        XCTAssertEqual(reads.count, 16,
                       "target sits in the last leaf of its 16-leaf branch")
        XCTAssertTrue(reads.allSatisfy {
            $0.hasPrefix("search-data/st-\(String(bucket, radix: 16))-")
        })
    }

    func testLowercaseVoiceQueryTriesTitleCaseBranchBeforeFullFanout() async throws {
        let (service, reader, _) = try fixture()
        let hits = try await service.geocode(
            query: "stanford university", limit: 1, zim: nil,
            kinds: ["place", "poi"])
        XCTAssertEqual(hits.first?.name, "Stanford University")
        XCTAssertLessThanOrEqual(reader.searchChunkReads().count, 32,
            "lowercase hash branch plus title-case hash branch, not all 256")
    }

    func testPrioritizationPreservesEveryLeafForSubstringFallback() {
        let leaves = (0..<16).flatMap { first in
            (0..<16).map { second in "st-\(String(first, radix: 16))-\(String(second, radix: 16))" }
        }
        let ordered = Geocoder.prioritizeSubChunkLeaves(
            leaves, prefix: "st", query: "Stanford University")
        XCTAssertEqual(Set(ordered), Set(leaves))
        XCTAssertEqual(ordered.count, leaves.count)
        let bucket = String(Geocoder.subBucketFor(name: "Stanford University"), radix: 16)
        XCTAssertTrue(ordered.prefix(16).allSatisfy { $0.hasPrefix("st-\(bucket)-") })
    }
}
