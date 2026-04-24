// SPDX-License-Identifier: MIT
//
// Parser + lazy-loader tests for SZCI/SZRC. Mirrors
// streetzim/tests/test_spatial_chunking.py in spirit — hand-packed
// buffers with known topology, then assertions on what the parsed
// structure contains.

import Foundation
import XCTest
@testable import MCPZimKit


final class SZRGSpatialTests: XCTestCase {
    // MARK: - Encoders

    /// Build a minimal SZCI buffer by hand. One cell at lat=0, lon=0
    /// containing two nodes. Matches the format documented at the top
    /// of SZRGSpatial.swift.
    private func packIndex(
        nodes: [(lat_e7: Int32, lon_e7: Int32)],
        cellEntries: [(lat: Int32, lon: Int32, nodes: UInt32, edges: UInt32, geoms: UInt32)],
        names: [String],
        cellScale: Int32 = 1
    ) -> Data {
        var out = Data()
        out.append(contentsOf: [0x53, 0x5A, 0x43, 0x49])  // SZCI
        appendU32(&out, 1)  // version
        appendU32(&out, UInt32(nodes.count))       // numNodes
        // Sum of cell edge counts gives numEdges
        let totalEdges: UInt32 = cellEntries.reduce(0) { $0 + $1.edges }
        appendU32(&out, totalEdges)
        appendU32(&out, UInt32(names.count))       // numNames

        var nameOffsets: [UInt32] = [0]
        var namesBlob = Data()
        for n in names {
            namesBlob.append(n.data(using: .utf8) ?? Data())
            nameOffsets.append(UInt32(namesBlob.count))
        }
        appendU32(&out, UInt32(namesBlob.count))   // namesBytes
        appendU32(&out, UInt32(cellEntries.count)) // numCells
        appendI32(&out, cellScale)

        for node in nodes {
            appendI32(&out, node.lat_e7)
            appendI32(&out, node.lon_e7)
        }
        for entry in cellEntries {
            appendI32(&out, entry.lat)
            appendI32(&out, entry.lon)
            appendU32(&out, entry.nodes)
            appendU32(&out, entry.edges)
            appendU32(&out, entry.geoms)
        }
        for off in nameOffsets { appendU32(&out, off) }
        out.append(namesBlob)
        return out
    }

    private func packCell(
        cellId: UInt32,
        nodesGlobal: [UInt32],
        adjOffsets: [UInt32],
        edges: [SpatialEdge],
        geoms: [Data] = []
    ) -> Data {
        var out = Data()
        out.append(contentsOf: [0x53, 0x5A, 0x52, 0x43])  // SZRC
        appendU32(&out, 1)                        // version
        appendU32(&out, cellId)                   // cell_id
        appendU32(&out, UInt32(nodesGlobal.count))
        appendU32(&out, UInt32(edges.count))
        appendU32(&out, UInt32(geoms.count))
        let geomBytes = geoms.reduce(0) { $0 + $1.count }
        appendU32(&out, UInt32(geomBytes))

        for n in nodesGlobal { appendU32(&out, n) }
        for off in adjOffsets { appendU32(&out, off) }
        for e in edges {
            appendU32(&out, e.target)
            appendU32(&out, e.speedDist)
            appendU32(&out, e.geomLocal)
            appendU32(&out, e.nameIdx)
            appendU32(&out, e.classAccess)
        }
        var geomOffsets: [UInt32] = [0]
        var geomBlob = Data()
        for g in geoms {
            geomBlob.append(g)
            geomOffsets.append(UInt32(geomBlob.count))
        }
        for off in geomOffsets { appendU32(&out, off) }
        out.append(geomBlob)
        return out
    }

    // MARK: - Index parser

    func testParseIndexHeaderRoundtrips() throws {
        // At scale=1 (1° cells), node 0 at lat=0° is in cell 0, and node
        // 1 at lat=1° is in cell (1, 0) — NOT lat=0.1° which would still
        // land in cell 0.
        let data = packIndex(
            nodes: [(0, 0), (10_000_000, 0)],  // 1° north
            cellEntries: [
                (lat: 0, lon: 0, nodes: 1, edges: 1, geoms: 0),
                (lat: 1, lon: 0, nodes: 1, edges: 0, geoms: 0),
            ],
            names: ["", "Main St"],
            cellScale: 1
        )
        let idx = try SZCIIndex.parse(data)
        XCTAssertEqual(idx.numNodes, 2)
        XCTAssertEqual(idx.numCells, 2)
        XCTAssertEqual(idx.numNames, 2)
        XCTAssertEqual(idx.cellScale, 1)
        XCTAssertEqual(idx.name(1), "Main St")
        // Cell lookup by (lat, lon) key.
        XCTAssertEqual(idx.cellForNode(0), 0)
        XCTAssertEqual(idx.cellForNode(1), 1)
    }

    func testParseIndexRejectsBadMagic() {
        var bad = Data("XXXX".utf8)
        bad.append(Data(repeating: 0, count: 60))
        XCTAssertThrowsError(try SZCIIndex.parse(bad)) { err in
            guard case SZCIError.badMagic = err else {
                return XCTFail("expected badMagic, got \(err)")
            }
        }
    }

    func testParseIndexRejectsFutureVersion() throws {
        var data = packIndex(
            nodes: [(0, 0)],
            cellEntries: [(lat: 0, lon: 0, nodes: 1, edges: 0, geoms: 0)],
            names: [""],
            cellScale: 1
        )
        // Stomp the version u32 (at offset 4) to 99.
        data.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!.advanced(by: 4).assumingMemoryBound(to: UInt32.self)
            p.pointee = 99
        }
        XCTAssertThrowsError(try SZCIIndex.parse(data)) { err in
            guard case SZCIError.unsupportedVersion(99, _) = err else {
                return XCTFail("expected unsupportedVersion(99), got \(err)")
            }
        }
    }

    func testCellForNodeHandlesNegativeCoords() throws {
        // A node at lat = -0.5° should land in cell (-1, 0) with scale=1
        // (floor semantics). Swift's integer / rounds toward zero, so
        // without the explicit floor helper we'd get cell 0 instead.
        let data = packIndex(
            nodes: [(-5_000_000, 0)],
            cellEntries: [(lat: -1, lon: 0, nodes: 1, edges: 0, geoms: 0)],
            names: [""],
            cellScale: 1
        )
        let idx = try SZCIIndex.parse(data)
        XCTAssertEqual(idx.cellForNode(0), 0, "floor semantics must bucket -0.5° into cell -1")
    }

    // MARK: - Cell parser

    func testParseCellHeaderAndFields() throws {
        let e = SpatialEdge(target: 1, speedDist: (50 << 24) | 12345,
                            geomLocal: 0, nameIdx: 1, classAccess: 0x100 | 2)
        let geomA = Data([1, 2, 3, 4])
        let data = packCell(
            cellId: 7,
            nodesGlobal: [0, 1],
            adjOffsets: [0, 1, 1],
            edges: [e],
            geoms: [geomA]
        )
        let cell = try SZRCCell.parse(data)
        XCTAssertEqual(cell.cellId, 7)
        XCTAssertEqual(cell.nodeCount, 2)
        XCTAssertEqual(cell.edgeCount, 1)
        XCTAssertEqual(cell.geomCount, 1)
        XCTAssertEqual(Array(cell.cellNodesGlobal), [0, 1])
        XCTAssertEqual(cell.localIdx(for: 1), 1)
        XCTAssertNil(cell.localIdx(for: 99))

        let edges = cell.edges
        XCTAssertEqual(edges.count, 5)  // stride 5
        XCTAssertEqual(edges[0], 1)     // target
        XCTAssertEqual(edges[4], 0x102) // class_access
    }

    func testSpatialEdgeDecoderFlags() {
        let linkEdge = SpatialEdge(target: 0, speedDist: (60 << 24) | 1000,
                                    geomLocal: 0xFFFFFFFF,
                                    nameIdx: 0,
                                    classAccess: 2)  // motorway_link ordinal
        XCTAssertTrue(linkEdge.isLink)
        XCTAssertFalse(linkEdge.isRoundabout)
        XCTAssertEqual(linkEdge.speedKmh, 60)
        XCTAssertEqual(linkEdge.distanceMeters, 100.0, accuracy: 0.001)

        let roundEdge = SpatialEdge(target: 0, speedDist: 0,
                                     geomLocal: 0xFFFFFFFF,
                                     nameIdx: 0,
                                     classAccess: 0x100)  // roundabout bit
        XCTAssertTrue(roundEdge.isRoundabout)
    }

    // MARK: - Lazy graph + cell cache

    func testEdgesOfNodeLazyLoadsSingleCell() async throws {
        let idx = try SZCIIndex.parse(packIndex(
            nodes: [(0, 0), (1_000_000, 0)],
            cellEntries: [
                (lat: 0, lon: 0, nodes: 2, edges: 1, geoms: 0),
            ],
            names: [""],
            cellScale: 1
        ))
        let cellData = packCell(
            cellId: 0,
            nodesGlobal: [0, 1],
            adjOffsets: [0, 1, 1],
            edges: [SpatialEdge(target: 1, speedDist: (50 << 24) | 1000,
                                 geomLocal: 0xFFFFFFFF, nameIdx: 0, classAccess: 0)],
            geoms: []
        )
        actor FetchCounter { var n = 0; func bump() { n += 1 } }
        let counter = FetchCounter()
        let sg = SpatialGraph(index: idx) { cid in
            await counter.bump()
            XCTAssertEqual(cid, 0)
            return cellData
        }

        let e0 = try await sg.edgesOfNode(0)
        XCTAssertEqual(e0.count, 1)
        XCTAssertEqual(e0[0].target, 1)
        XCTAssertEqual(e0[0].speedKmh, 50)
        // A second call on a node from the same cell must not re-fetch.
        _ = try await sg.edgesOfNode(1)
        let fetches = await counter.n
        XCTAssertEqual(fetches, 1, "second query to same cell must hit cache")
    }

    func testEdgesOfNodeTouchesMultipleCells() async throws {
        // Two cells at scale=10 (0.1° grid). Node 0 at lat=0 ⇒ cell (0,0);
        // node 1 at lat=0.2° ⇒ floor(0.2 * 10) = 2 ⇒ cell (2, 0).
        let idx = try SZCIIndex.parse(packIndex(
            nodes: [(0, 0), (2_000_000, 0)],
            cellEntries: [
                (lat: 0, lon: 0, nodes: 1, edges: 1, geoms: 0),
                (lat: 2, lon: 0, nodes: 1, edges: 1, geoms: 0),
            ],
            names: [""],
            cellScale: 10
        ))
        let cell0 = packCell(
            cellId: 0, nodesGlobal: [0], adjOffsets: [0, 1],
            edges: [SpatialEdge(target: 1, speedDist: (50 << 24) | 1000,
                                 geomLocal: 0xFFFFFFFF, nameIdx: 0, classAccess: 0)]
        )
        let cell1 = packCell(
            cellId: 1, nodesGlobal: [1], adjOffsets: [0, 1],
            edges: [SpatialEdge(target: 0, speedDist: (50 << 24) | 1000,
                                 geomLocal: 0xFFFFFFFF, nameIdx: 0, classAccess: 0)]
        )
        actor FetchLog { var cids: [Int] = []; func add(_ c: Int) { cids.append(c) } }
        let log = FetchLog()
        let sg = SpatialGraph(index: idx) { cid in
            await log.add(cid)
            return cid == 0 ? cell0 : cell1
        }

        _ = try await sg.edgesOfNode(0)
        _ = try await sg.edgesOfNode(1)
        let cids = await log.cids
        XCTAssertEqual(cids.sorted(), [0, 1])
    }

    func testSpatialGraphCacheLimitEvicts() async throws {
        // Build 3 cells, cache limit 1 → each query evicts the prior.
        let idx = try SZCIIndex.parse(packIndex(
            nodes: [(0, 0), (2_000_000, 0), (4_000_000, 0)],  // 3 cells at scale=10
            cellEntries: [
                (lat: 0, lon: 0, nodes: 1, edges: 0, geoms: 0),
                (lat: 2, lon: 0, nodes: 1, edges: 0, geoms: 0),
                (lat: 4, lon: 0, nodes: 1, edges: 0, geoms: 0),
            ],
            names: [""],
            cellScale: 10
        ))
        let cellDatas: [Int: Data] = [
            0: packCell(cellId: 0, nodesGlobal: [0], adjOffsets: [0, 0], edges: []),
            1: packCell(cellId: 1, nodesGlobal: [1], adjOffsets: [0, 0], edges: []),
            2: packCell(cellId: 2, nodesGlobal: [2], adjOffsets: [0, 0], edges: []),
        ]
        actor FetchLog { var cids: [Int] = []; func add(_ c: Int) { cids.append(c) } }
        let log = FetchLog()
        let sg = SpatialGraph(index: idx, cacheLimit: 1) { cid in
            await log.add(cid)
            return cellDatas[cid]!
        }
        _ = try await sg.edgesOfNode(0)
        _ = try await sg.edgesOfNode(1)
        _ = try await sg.edgesOfNode(2)
        _ = try await sg.edgesOfNode(0)  // should re-fetch — evicted
        let cids = await log.cids
        XCTAssertEqual(cids, [0, 1, 2, 0],
                       "cacheLimit=1 means each query evicts the prior; "
                       + "re-query of cell 0 should miss")
        let loaded = await sg.cellsLoaded
        XCTAssertLessThanOrEqual(loaded, 1)
    }
}


// MARK: - Little-endian writers (test-only)

private func appendU32(_ data: inout Data, _ v: UInt32) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}
private func appendI32(_ data: inout Data, _ v: Int32) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}
