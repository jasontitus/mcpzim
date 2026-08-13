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

    func testParseCellRejectsInvalidAdjacencyOffsets() {
        let edge = SpatialEdge(
            target: 1, speedDist: (50 << 24) | 100,
            geomLocal: 0xFFFF_FFFF, nameIdx: 0, classAccess: 0)
        let data = packCell(
            cellId: 0,
            nodesGlobal: [0, 1],
            adjOffsets: [0, 2, 1],
            edges: [edge])
        XCTAssertThrowsError(try SZRCCell.parse(data))
    }

    func testDecodeGeomRejectsShortHeader() throws {
        let data = packCell(
            cellId: 0,
            nodesGlobal: [0],
            adjOffsets: [0, 0],
            edges: [],
            geoms: [Data([1, 2, 3, 4])])
        let cell = try SZRCCell.parse(data)
        XCTAssertThrowsError(try cell.decodeGeom(localIdx: 0))
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

    // MARK: - Spatial A* (the async router wired into planDrivingRoute)

    func testSpatialAStarRoutesThreeNodeLine() async throws {
        // A(0,0) — B(0,0.001) — C(0,0.002): a 222 m line on "Main St", one
        // cell, bidirectional edges. routeSpatial(A,C) must walk A→B→C.
        let idx = try SZCIIndex.parse(packIndex(
            nodes: [(0, 0), (0, 10_000), (0, 20_000)],
            cellEntries: [(lat: 0, lon: 0, nodes: 3, edges: 4, geoms: 0)],
            names: ["", "Main St"]))
        let sd = (UInt32(50) << 24) | 1110     // 50 km/h, 111.0 m (1110 dm)
        func edge(_ target: UInt32) -> SpatialEdge {
            SpatialEdge(target: target, speedDist: sd,
                        geomLocal: 0xFFFF_FFFF, nameIdx: 1, classAccess: 1)
        }
        let cell = packCell(
            cellId: 0, nodesGlobal: [0, 1, 2], adjOffsets: [0, 1, 3, 4],
            edges: [edge(1), edge(0), edge(2), edge(1)])  // A→B, B→A, B→C, C→B
        let g = SpatialGraph(index: idx) { _ in cell }

        let routeOpt = await routeSpatial(graph: g, index: idx, origin: 0, goal: 2)
        let route = try XCTUnwrap(routeOpt, "A→B→C must be routable")
        XCTAssertEqual(route.originNode, 0)
        XCTAssertEqual(route.destinationNode, 2)
        XCTAssertEqual(route.distanceMeters, 222.0, accuracy: 1.0)   // 2 × 111 m
        XCTAssertGreaterThan(route.durationSeconds, 0)
        XCTAssertEqual(route.roads.map { $0.name }, ["Main St"],
                       "consecutive same-name edges merge into one road")
        XCTAssertGreaterThanOrEqual(route.polyline.count, 3)  // origin + B + C

        // nearestNode picks the closest of the three.
        XCTAssertEqual(nearestNodeSpatial(index: idx, lat: 0, lon: 0.0019), 2)
        XCTAssertEqual(nearestNodeSpatial(index: idx, lat: 0, lon: 0.0001), 0)
    }

    /// Minimal streetzim reader serving raw routing-data blobs.
    private final class SpatialMapReader: ZimReader, @unchecked Sendable {
        let store: [String: Data]
        init(_ s: [String: Data]) { store = s }
        var metadata: ZimMetadata { ZimMetadata(name: "osm-v2") }
        var kind: ZimKind { .streetzim }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { false }
        var hasRoutingData: Bool { true }
        func read(path: String) throws -> ZimEntry? {
            store[path].map { ZimEntry(path: path, title: path,
                                       mimetype: "application/octet-stream", content: $0) }
        }
        func readMainPage() throws -> ZimEntry? { nil }
    }

    /// SZCI v2 header (40 B, sharded nodes — no inline node table).
    private func packIndexV2(
        numNodes: Int, numNodeShards: UInt32, nodesPerShard: UInt32,
        cellEntries: [(lat: Int32, lon: Int32, nodes: UInt32, edges: UInt32, geoms: UInt32)],
        names: [String]
    ) -> Data {
        var out = Data()
        out.append(contentsOf: [0x53, 0x5A, 0x43, 0x49])  // SZCI
        appendU32(&out, 2)                                 // version 2
        appendU32(&out, UInt32(numNodes))
        appendU32(&out, cellEntries.reduce(0) { $0 + $1.edges })  // numEdges
        appendU32(&out, UInt32(names.count))
        var nameOffsets: [UInt32] = [0]; var namesBlob = Data()
        for n in names { namesBlob.append(n.data(using: .utf8) ?? Data()); nameOffsets.append(UInt32(namesBlob.count)) }
        appendU32(&out, UInt32(namesBlob.count))           // namesBytes
        appendU32(&out, UInt32(cellEntries.count))         // numCells
        appendI32(&out, 1)                                  // cellScale
        appendU32(&out, numNodeShards)                      // @32
        appendU32(&out, nodesPerShard)                      // @36  → offset 40, no inline nodes
        for e in cellEntries {
            appendI32(&out, e.lat); appendI32(&out, e.lon)
            appendU32(&out, e.nodes); appendU32(&out, e.edges); appendU32(&out, e.geoms)
        }
        for off in nameOffsets { appendU32(&out, off) }
        out.append(namesBlob)
        return out
    }

    func testV2ShardedIndexRoutesEndToEnd() async throws {
        // 3 nodes A(0,0)-B(0,0.001)-C(0,0.002) on "Main St", one cell, but the
        // node table is V2-SHARDED: shard0 = nodes 0,1; shard1 = node 2
        // (nodesPerShard=2). planDrivingRoute must parse v2, assemble the
        // shards, and route A→C. Catches header-offset + shard-assembly bugs.
        func i32le(_ vals: [Int32]) -> Data {
            var d = Data(); for v in vals { var le = v.littleEndian; withUnsafeBytes(of: &le) { d.append(contentsOf: $0) } }; return d
        }
        let index = packIndexV2(
            numNodes: 3, numNodeShards: 2, nodesPerShard: 2,
            cellEntries: [(lat: 0, lon: 0, nodes: 3, edges: 4, geoms: 0)],
            names: ["", "Main St"])
        let shard0 = i32le([0, 0, 0, 10_000])    // node 0 (A), node 1 (B)
        let shard1 = i32le([0, 20_000])           // node 2 (C)
        let sd = (UInt32(50) << 24) | 1110
        func edge(_ t: UInt32) -> SpatialEdge {
            SpatialEdge(target: t, speedDist: sd, geomLocal: 0xFFFF_FFFF, nameIdx: 1, classAccess: 1)
        }
        let cell = packCell(cellId: 0, nodesGlobal: [0, 1, 2], adjOffsets: [0, 1, 3, 4],
                            edges: [edge(1), edge(0), edge(2), edge(1)])
        let reader = SpatialMapReader([
            "routing-data/graph-cells-index.bin": index,
            "routing-data/nodes-scaled-000.bin": shard0,
            "routing-data/nodes-scaled-001.bin": shard1,
            "routing-data/graph-cell-00000.bin": cell,
        ])
        let svc = DefaultZimService(readers: [(name: "osm-v2", reader: reader)])
        let route = try await svc.planDrivingRoute(RouteRequest(
            originLat: 0, originLon: 0.00001, destLat: 0, destLon: 0.00199, zim: nil))
        XCTAssertEqual(route.originNode, 0)
        XCTAssertEqual(route.destinationNode, 2)
        XCTAssertEqual(route.distanceMeters, 222.0, accuracy: 1.0)
        XCTAssertEqual(route.roads.map { $0.name }, ["Main St"])
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

    // MARK: - Adversarial SZCI / SZRC parser inputs

    /// A crafted SZCI with numNodes = 0xFFFFFFFF should throw a truncated
    /// error — NOT trap on Int overflow or read off the end of the buffer.
    func testParseIndexRejectsAdversarialNumNodes() {
        var d = Data()
        d.append(contentsOf: [0x53, 0x5A, 0x43, 0x49])       // SZCI
        appendU32(&d, 1)                                       // version
        appendU32(&d, 0xFFFFFFFF)                              // numNodes (hostile)
        appendU32(&d, 0)                                       // numEdges
        appendU32(&d, 0)                                       // numNames
        appendU32(&d, 0)                                       // namesBytes
        appendU32(&d, 0)                                       // numCells
        appendI32(&d, 10)                                      // cellScale
        // Header is complete (32 B) but no room for nodes; parser must
        // reject before allocating 34 GB of UInt32s.
        XCTAssertThrowsError(try SZCIIndex.parse(d)) { err in
            guard case SZCIError.truncated(let msg) = err else {
                return XCTFail("expected truncated, got \(err)")
            }
            XCTAssertTrue(msg.contains("nodes"), "got: \(msg)")
        }
    }

    /// Same hostile pattern inside an SZRC cell payload: edgeCount
    /// 0xFFFFFFFF would walk past the buffer end if unchecked.
    func testParseCellRejectsAdversarialEdgeCount() {
        var d = Data()
        d.append(contentsOf: [0x53, 0x5A, 0x52, 0x43])        // SZRC
        appendU32(&d, 1)                                       // version
        appendU32(&d, 0)                                       // cellId
        appendU32(&d, 0)                                       // nodeCount
        appendU32(&d, 0xFFFFFFFF)                              // edgeCount (hostile)
        appendU32(&d, 0)                                       // geomCount
        appendU32(&d, 0)                                       // geomBytes
        // Single adj offset (nodeCount + 1 = 1 entry, 4 B).
        appendU32(&d, 0)
        XCTAssertThrowsError(try SZRCCell.parse(d)) { err in
            guard case SZCIError.truncated(let msg) = err else {
                return XCTFail("expected truncated, got \(err)")
            }
            XCTAssertTrue(msg.contains("edges"), "got: \(msg)")
        }
    }

    // MARK: - Adversarial SZCI v2 (sharded nodes)
    //
    // v1's node table is bounded by `requireBytes` against the blob it was
    // read from. v2 moved that table into nodes-scaled-NNN.bin, so numNodes /
    // numNodeShards / nodesPerShard reached the `[Int32](count: numNodes * 2)`
    // allocation and the `shard * nodesPerShard * 2 * 4` offset unchecked —
    // ~34 GB and an Int-overflow trap respectively, on the routing path of a
    // P2P-shared ZIM (DS4 medium 2026-08-13). Each test below asserts a clean
    // throw AND counts node-shard reads: the allocation happens after the
    // first/last-shard probe, so a low read count is the evidence that no
    // giant buffer was ever requested.

    /// Reader that records every path it served, so a test can prove where
    /// the load bailed out.
    private final class CountingSpatialReader: ZimReader, @unchecked Sendable {
        let store: [String: Data]
        private let lock = NSLock()
        private var served: [String] = []
        init(_ s: [String: Data]) { store = s }
        var metadata: ZimMetadata { ZimMetadata(name: "osm-v2") }
        var kind: ZimKind { .streetzim }
        var hasFullTextIndex: Bool { false }
        var hasTitleIndex: Bool { false }
        var hasRoutingData: Bool { true }
        var shardReads: Int {
            lock.lock()
            defer { lock.unlock() }
            return served.filter { $0.contains("nodes-scaled") }.count
        }
        func read(path: String) throws -> ZimEntry? {
            lock.lock()
            served.append(path)
            lock.unlock()
            return store[path].map {
                ZimEntry(path: path, title: path,
                         mimetype: "application/octet-stream", content: $0)
            }
        }
        func readMainPage() throws -> ZimEntry? { nil }
    }

    private func i32le(_ vals: [Int32]) -> Data {
        var d = Data()
        for v in vals { appendI32(&d, v) }
        return d
    }

    /// Drive the real `loadSpatialGraph` → `loadNodeShards` path with a
    /// hand-forged v2 header and return (thrown message, node-shard reads).
    private func loadV2(
        numNodes: Int, numNodeShards: UInt32, nodesPerShard: UInt32,
        shards: [Int: Data] = [0: Data(count: 16), 1: Data(count: 8)]
    ) async -> (message: String?, shardReads: Int) {
        var store: [String: Data] = [
            "routing-data/graph-cells-index.bin": packIndexV2(
                numNodes: numNodes, numNodeShards: numNodeShards,
                nodesPerShard: nodesPerShard,
                cellEntries: [(lat: 0, lon: 0, nodes: 1, edges: 0, geoms: 0)],
                names: [""]),
        ]
        for (i, d) in shards {
            store[String(format: "routing-data/nodes-scaled-%03d.bin", i)] = d
        }
        let reader = CountingSpatialReader(store)
        let svc = DefaultZimService(readers: [(name: "osm-v2", reader: reader)])
        do {
            _ = try await svc.loadSpatialGraph(zimName: "osm-v2")
            return (nil, reader.shardReads)
        } catch let e as SZCIError {
            guard case .truncated(let msg) = e else {
                return ("unexpected SZCIError \(e)", reader.shardReads)
            }
            return (msg, reader.shardReads)
        } catch {
            return ("unexpected \(error)", reader.shardReads)
        }
    }

    /// numNodes = 0xFFFFFFFF with a 1×1 shard table can't be backed by any
    /// archive; the header cross-check must reject it before a byte of shard
    /// data (or 34 GB of Int32s) is touched.
    func testV2HostileNumNodesRejectedBeforeAnyShardRead() async {
        let (message, reads) = await loadV2(
            numNodes: 0xFFFF_FFFF, numNodeShards: 1, nodesPerShard: 1)
        let msg = try? XCTUnwrap(message, "hostile numNodes must throw")
        XCTAssertTrue((msg ?? "").contains("ceil-consistent"), "got: \(message ?? "no throw")")
        XCTAssertEqual(reads, 0, "rejected at parse — no shard should be read")
    }

    /// numNodeShards × nodesPerShard at 0xFFFFFFFF each is ~1.8e19, past
    /// Int.max: the old `shard * nodesPerShard * 2 * 4` trapped here. The
    /// overflow-reporting multiply must turn it into a throw.
    func testV2OverflowingShardGeometryThrowsInsteadOfTrapping() async {
        let (message, reads) = await loadV2(
            numNodes: 0xFFFF_FFFF, numNodeShards: 0xFFFF_FFFF,
            nodesPerShard: 0xFFFF_FFFF)
        XCTAssertTrue((message ?? "").contains("ceil-consistent"),
                      "got: \(message ?? "no throw")")
        XCTAssertEqual(reads, 0)
    }

    /// A ceil-consistent but still absurd header (1000 × 4 294 968 ≈ 2³²
    /// nodes) survives the parse-time cross-check, so the byte-grounded probe
    /// in `loadNodeShards` is what has to stop it — after two small reads and
    /// before the allocation.
    func testV2ForgedNodeCountRejectedAgainstActualShardBytes() async {
        let (message, reads) = await loadV2(
            numNodes: 0xFFFF_FFFF, numNodeShards: 1000, nodesPerShard: 4_294_968,
            shards: [0: Data(count: 16), 999: Data(count: 8)])
        XCTAssertTrue((message ?? "").contains("shards carry"),
                      "got: \(message ?? "no throw")")
        XCTAssertLessThanOrEqual(reads, 2, "only the first/last probe may run")
    }

    /// `nodes-scaled-%03d.bin` can only name 1000 shards, so a bigger claim is
    /// forged by construction and must not reach the offset arithmetic.
    func testV2ShardCountBeyondNameSpaceRejected() async {
        // 1999×2 < 3999 ≤ 2000×2, so the parse-time ceil check passes.
        let (message, reads) = await loadV2(
            numNodes: 3999, numNodeShards: 2000, nodesPerShard: 2)
        XCTAssertTrue((message ?? "").contains("at most 1000"),
                      "got: \(message ?? "no throw")")
        XCTAssertEqual(reads, 0)
    }

    /// A shard whose byte count isn't a whole number of (lat, lon) pairs is
    /// truncated — a partial copy would silently mis-place every later node.
    func testV2TruncatedShardRejected() async {
        let (message, _) = await loadV2(
            numNodes: 3, numNodeShards: 2, nodesPerShard: 2,
            shards: [0: i32le([0, 0, 0, 10_000]), 1: Data(count: 5)])
        XCTAssertTrue((message ?? "").contains("multiple of 8"),
                      "got: \(message ?? "no throw")")
    }

    /// Non-final shards must be exactly `nodesPerShard` long — the
    /// `shard * nodesPerShard` placement assumes it, and it is what ties
    /// numNodes to bytes that exist.
    func testV2ShortLeadingShardRejected() async {
        let (message, _) = await loadV2(
            numNodes: 3, numNodeShards: 2, nodesPerShard: 2,
            shards: [0: i32le([0, 0]), 1: i32le([0, 20_000])])
        XCTAssertTrue((message ?? "").contains("shards carry"),
                      "got: \(message ?? "no throw")")
    }

    /// The first/last probe can't see a short shard in the middle, and a
    /// partial copy there would zero-fill a hole — silently placing real nodes
    /// at (0, 0) — so the copy loop re-checks every shard's length.
    func testV2ShortMiddleShardRejected() async {
        let (message, _) = await loadV2(
            numNodes: 5, numNodeShards: 3, nodesPerShard: 2,
            shards: [0: i32le([0, 0, 0, 10_000]),
                     1: i32le([0, 20_000]),
                     2: i32le([0, 40_000])])
        XCTAssertTrue((message ?? "").contains("overruns node table"),
                      "got: \(message ?? "no throw")")
    }

    /// The honest header still loads: 2 full-then-partial shards, 3 nodes.
    func testV2WellFormedShardGeometryStillLoads() async {
        let (message, reads) = await loadV2(
            numNodes: 3, numNodeShards: 2, nodesPerShard: 2,
            shards: [0: i32le([0, 0, 0, 10_000]), 1: i32le([0, 20_000])])
        XCTAssertNil(message, "well-formed v2 must load")
        XCTAssertEqual(reads, 2)
    }

    /// A single-shard table has no "full non-final shard" to check, so its
    /// only bound is the one shard's own byte count.
    func testV2SingleShardCountMustMatchItsBytes() async {
        let (bad, _) = await loadV2(
            numNodes: 2, numNodeShards: 1, nodesPerShard: 2,
            shards: [0: i32le([0, 0])])
        XCTAssertTrue((bad ?? "").contains("shards carry"), "got: \(bad ?? "no throw")")
        let (good, _) = await loadV2(
            numNodes: 2, numNodeShards: 1, nodesPerShard: 2,
            shards: [0: i32le([0, 0, 0, 10_000])])
        XCTAssertNil(good, "honest single-shard table must load")
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
