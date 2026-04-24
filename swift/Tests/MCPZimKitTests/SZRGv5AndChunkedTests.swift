// SPDX-License-Identifier: MIT
//
// Coverage for SZRG v5 split + chunked-reassembly handling. Mirrors the
// Python suite under streetzim/tests/test_szrg_v5_split.py +
// test_graph_chunking.py — identical input means identical node /edge /
// name arrays out.

import CryptoKit
import Foundation
import XCTest
@testable import MCPZimKit

final class SZRGv5AndChunkedTests: XCTestCase {
    // MARK: - v5 split

    /// Build a SZRG v5 main buffer + matching SZGM companion for the
    /// same 3-node graph used in SZRGGraphTests. class_access column is
    /// always 0 — we only care about byte layout here.
    private static func buildV5Pair(includeGeoms: Bool = true) -> (main: Data, szgm: Data) {
        // Graph topology: A(0,0) -- B(0,0.01) -- C(0.01,0.01) at 50 km/h.
        let nodes: [(Double, Double)] = [(0, 0), (0, 0.01), (0.01, 0.01)]
        let d = haversineMeters(0, 0, 0, 0.01)
        let distDm = UInt32((d * 10).rounded())
        let names = ["", "Main St"]

        // Build adjacency (CSR).
        let adjOffsets: [UInt32] = [0, 1, 2, 2]

        // Edge column (v5 = v4 stride = 5 × u32):
        //   (target, speed<<24 | distDm24, geom_idx, name_idx, class_access)
        func edgeRow(target: UInt32, speed: UInt32, dm: UInt32,
                     geom: UInt32, name: UInt32, ca: UInt32) -> [UInt32] {
            return [target, (speed << 24) | (dm & 0x00FFFFFF), geom, name, ca]
        }
        let edges = [
            edgeRow(target: 1, speed: 50, dm: distDm, geom: 0, name: 1, ca: 0),
            edgeRow(target: 2, speed: 50, dm: distDm, geom: 1, name: 1, ca: 0),
        ]

        // Geom blob: two geoms, each a single int32 (lon,lat) pair = 8 B.
        // Geom 0: (lon0, lat0) = (0, 0.005 * 1e7)
        // Geom 1: (lon0, lat0) = (0.005 * 1e7, 0.01 * 1e7)
        func geomBytes(lon: Int32, lat: Int32) -> Data {
            var d = Data()
            withUnsafeBytes(of: lon.littleEndian) { d.append(contentsOf: $0) }
            withUnsafeBytes(of: lat.littleEndian) { d.append(contentsOf: $0) }
            return d
        }
        let g0 = geomBytes(lon: 0, lat: 50_000)
        let g1 = geomBytes(lon: 50_000, lat: 100_000)
        let geomBlob = g0 + g1
        let geomOffsets: [UInt32] = [0, UInt32(g0.count), UInt32(g0.count + g1.count)]

        // Name table: UTF-8 blob + offset index.
        var nameOffsets: [UInt32] = [0]
        var namesBlob = Data()
        for n in names {
            namesBlob.append(n.data(using: .utf8) ?? Data())
            nameOffsets.append(UInt32(namesBlob.count))
        }

        // ---- Main v5 buffer ----
        var main = Data()
        main.append(contentsOf: [0x53, 0x5A, 0x52, 0x47]) // SZRG
        func appendU32(_ x: UInt32) {
            var v = x.littleEndian
            withUnsafeBytes(of: &v) { main.append(contentsOf: $0) }
        }
        appendU32(5)                                    // version
        appendU32(UInt32(nodes.count))
        appendU32(UInt32(edges.count))
        appendU32(UInt32(geomOffsets.count - 1))        // numGeoms
        appendU32(0)                                    // geomBytes=0 sentinel
        appendU32(UInt32(names.count))
        appendU32(UInt32(namesBlob.count))
        // Nodes
        for (lat, lon) in nodes {
            var lx = Int32((lat * 1e7).rounded()).littleEndian
            withUnsafeBytes(of: &lx) { main.append(contentsOf: $0) }
            var ly = Int32((lon * 1e7).rounded()).littleEndian
            withUnsafeBytes(of: &ly) { main.append(contentsOf: $0) }
        }
        // adj
        for off in adjOffsets { appendU32(off) }
        // edges (v5 stride 5)
        for row in edges { for v in row { appendU32(v) } }
        // (no geom section in v5 main)
        // name offsets + blob
        for off in nameOffsets { appendU32(off) }
        main.append(namesBlob)

        // ---- SZGM companion ----
        var szgm = Data()
        szgm.append(contentsOf: [0x53, 0x5A, 0x47, 0x4D]) // SZGM
        func appendU32S(_ x: UInt32) {
            var v = x.littleEndian
            withUnsafeBytes(of: &v) { szgm.append(contentsOf: $0) }
        }
        appendU32S(1)                                        // SZGM version
        appendU32S(UInt32(geomOffsets.count - 1))            // numGeoms
        let geomBytesTotal = UInt32(includeGeoms ? geomBlob.count : 0)
        appendU32S(geomBytesTotal)
        for off in geomOffsets { appendU32S(off) }
        if includeGeoms { szgm.append(geomBlob) }

        return (main: main, szgm: szgm)
    }

    func testV5ParsesWithoutCompanionForRouting() throws {
        let (main, _) = Self.buildV5Pair()
        // decodeGeoms=false is the mcpzim-routing path. No companion
        // needed; geom array comes back empty but adjacency + edge
        // distances + names are populated.
        let g = try SZRGGraph.parse(main, geomsData: nil, decodeGeoms: false)
        XCTAssertEqual(g.numNodes, 3)
        XCTAssertEqual(g.numEdges, 2)
        XCTAssertEqual(g.names, ["", "Main St"])
        XCTAssertEqual(Int(g.adjOffsets.last ?? 0), 2)
        XCTAssertEqual(g.geoms.count, 2, "geom array should be sized-but-empty")
        XCTAssertTrue(g.geoms.allSatisfy { $0.isEmpty })
    }

    func testV5AttachesCompanionGeoms() throws {
        let (main, szgm) = Self.buildV5Pair()
        let g = try SZRGGraph.parse(main, geomsData: szgm, decodeGeoms: true)
        XCTAssertEqual(g.geoms.count, 2)
        XCTAssertEqual(g.geoms[0].count, 1)
        XCTAssertEqual(g.geoms[1].count, 1)
        // First geom encoded as (lat=50000 * 1e-7, lon=0).
        XCTAssertEqual(g.geoms[0][0].lat, 50_000.0 / 1e7, accuracy: 1e-12)
        XCTAssertEqual(g.geoms[0][0].lon, 0.0, accuracy: 1e-12)
    }

    func testV5CompanionWithMismatchedGeomCountFails() throws {
        let (main, _) = Self.buildV5Pair()
        // Build a SZGM declaring 1 geom instead of 2.
        var szgm = Data()
        szgm.append(contentsOf: [0x53, 0x5A, 0x47, 0x4D])
        for v in [UInt32(1), UInt32(1), UInt32(0)] {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { szgm.append(contentsOf: $0) }
        }
        // offsets for 1 geom = 2 u32s
        for v in [UInt32(0), UInt32(0)] {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { szgm.append(contentsOf: $0) }
        }
        XCTAssertThrowsError(try SZRGGraph.parse(main, geomsData: szgm, decodeGeoms: true)) { err in
            guard case SZRGError.companionMismatch = err else {
                return XCTFail("expected companionMismatch, got \(err)")
            }
        }
    }

    func testV5HeaderGeomBytesNonZeroIsRejected() throws {
        // Take a valid v5 main buffer and overwrite geomBytes=N to simulate
        // a drift where a future writer forgot to zero the field.
        var (main, _) = Self.buildV5Pair()
        main.withUnsafeMutableBytes { raw in
            // Byte 20..24 is geomBytes (7 u32s after "SZRG" magic, index 4 → byte 20).
            let p = raw.baseAddress!.advanced(by: 20).assumingMemoryBound(to: UInt32.self)
            p.pointee = 999
        }
        XCTAssertThrowsError(try SZRGGraph.parse(main, geomsData: nil, decodeGeoms: false))
    }

    // MARK: - Chunked reassembly

    func testChunkedReassemblyConcatenatesInOrder() throws {
        // Payload = ascending bytes 0…999 so any reordering jumps out.
        var payload = Data(capacity: 1000)
        for i in 0..<1000 { payload.append(UInt8(i & 0xFF)) }
        let chunks: [(String, Data)] = [
            ("graph-chunk-0000.bin", payload.prefix(300)),
            ("graph-chunk-0001.bin", payload[300..<700]),
            ("graph-chunk-0002.bin", payload.suffix(300)),
        ]
        let sha = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": 1000,
            "sha256": sha,
            "chunks": chunks.map { ["path": $0.0, "bytes": $0.1.count] },
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifest)

        let reassembled = try SZRGChunked.reassembleChunked(manifest: manifestData) { name in
            guard let c = chunks.first(where: { $0.0 == name })?.1 else {
                XCTFail("loader asked for unknown chunk \(name)")
                return Data()
            }
            return c
        }
        XCTAssertEqual(reassembled, payload)
    }

    func testChunkedReassemblyRejectsSizeMismatch() throws {
        let payload = Data(repeating: 0xAB, count: 100)
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": 100,
            "chunks": [["path": "c.bin", "bytes": 50]],  // lies: payload is 100
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(try SZRGChunked.reassembleChunked(manifest: md) { _ in payload }) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("size"))
        }
    }

    func testChunkedReassemblyRejectsShaMismatch() throws {
        let payload = Data(repeating: 0xCD, count: 100)
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": 100,
            "sha256": String(repeating: "0", count: 64),  // bogus
            "chunks": [["path": "c.bin", "bytes": 100]],
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(try SZRGChunked.reassembleChunked(manifest: md) { _ in payload }) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("sha256"))
        }
    }

    func testChunkedReassemblyRejectsUnsupportedSchema() throws {
        let manifest: [String: Any] = [
            "schema": 99,
            "total_bytes": 0,
            "chunks": [],
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(try SZRGChunked.reassembleChunked(manifest: md) { _ in Data() }) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("schema"))
        }
    }

    func testChunkedReassemblyThenParseSzrg() throws {
        // End-to-end: chunk a real v2 graph into 3 pieces, reassemble via
        // SZRGChunked, parse with SZRGGraph.parse — should succeed.
        let nodes: [(Double, Double)] = [(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)]
        let d = haversineMeters(0, 0, 0, 0.01)
        let edges: [(Int, Int, Double, Int, Int)] = [
            (0, 1, d, 50, 0), (1, 2, d, 50, 0),
            (0, 3, d, 30, 0), (3, 2, d, 30, 0),
        ]
        let original = encodeGraphV2(nodes: nodes, edges: edges, names: [""])

        // Chunk into 3 ~even pieces.
        let n = original.count
        let size1 = n / 3
        let size2 = n / 3
        let c0 = original.prefix(size1)
        let c1 = original[size1..<(size1 + size2)]
        let c2 = original.suffix(n - size1 - size2)
        let sha = SHA256.hash(data: original).map { String(format: "%02x", $0) }.joined()
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": n,
            "sha256": sha,
            "chunks": [
                ["path": "graph-chunk-0000.bin", "bytes": c0.count],
                ["path": "graph-chunk-0001.bin", "bytes": c1.count],
                ["path": "graph-chunk-0002.bin", "bytes": c2.count],
            ],
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)

        let parts: [String: Data] = [
            "graph-chunk-0000.bin": Data(c0),
            "graph-chunk-0001.bin": Data(c1),
            "graph-chunk-0002.bin": Data(c2),
        ]
        let reassembled = try SZRGChunked.reassembleChunked(manifest: md) { parts[$0] ?? Data() }
        XCTAssertEqual(reassembled, original)

        let g = try SZRGGraph.parse(reassembled)
        XCTAssertEqual(g.numNodes, 4)
        XCTAssertEqual(g.numEdges, 4)
    }

    // MARK: - Hostile-manifest guardrails

    /// A torn or hostile manifest with an absurd `total_bytes` must be
    /// rejected before any allocation — otherwise `reserveCapacity` on a
    /// 10 GB value would OOM iOS the moment anyone tried to open the
    /// ZIM.
    func testChunkedReassemblyRejectsAbsurdTotalBytes() throws {
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": 9_999_999_999,
            "chunks": [],
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(
            try SZRGChunked.reassembleChunked(manifest: md) { _ in Data() }
        ) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("total_bytes"), "got: \(m)")
        }
    }

    func testChunkedReassemblyRejectsNegativeTotalBytes() throws {
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": -1,
            "chunks": [],
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(
            try SZRGChunked.reassembleChunked(manifest: md) { _ in Data() }
        ) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("total_bytes"), "got: \(m)")
        }
    }

    func testChunkedReassemblyRejectsTooManyChunks() throws {
        // Each entry tiny so the test itself stays fast; point is the count
        // cap kicks in before we even touch the data.
        var chunks: [[String: Any]] = []
        for i in 0...SZRGChunked.maxChunks {
            chunks.append(["path": "c\(i).bin", "bytes": 1])
        }
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": chunks.count,
            "chunks": chunks,
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(
            try SZRGChunked.reassembleChunked(manifest: md) { _ in Data([0]) }
        ) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("chunk count"), "got: \(m)")
        }
    }

    func testChunkedReassemblyRejectsChunkSumExceedsTotal() throws {
        // Manifest claims 50 total bytes but lists three 30-byte chunks
        // summing to 90. Fail BEFORE loading anything — the writer would
        // never emit this, so it signals corruption or attack.
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": 50,
            "chunks": [
                ["path": "a.bin", "bytes": 30],
                ["path": "b.bin", "bytes": 30],
                ["path": "c.bin", "bytes": 30],
            ],
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(
            try SZRGChunked.reassembleChunked(manifest: md) { _ in Data(count: 30) }
        ) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("sum of chunk bytes"), "got: \(m)")
        }
    }

    func testChunkedReassemblyRejectsChunkBytesOverflow() throws {
        let manifest: [String: Any] = [
            "schema": 1,
            "total_bytes": 1,
            "chunks": [
                ["path": "a.bin", "bytes": SZRGChunked.maxChunkBytes + 1],
            ],
        ]
        let md = try JSONSerialization.data(withJSONObject: manifest)
        XCTAssertThrowsError(
            try SZRGChunked.reassembleChunked(manifest: md) { _ in Data() }
        ) { err in
            guard case SZRGError.chunkedReassembly(let m) = err else {
                return XCTFail("expected chunkedReassembly, got \(err)")
            }
            XCTAssertTrue(m.contains("bytes"), "got: \(m)")
        }
    }
}
