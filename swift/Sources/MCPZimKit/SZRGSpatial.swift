// SPDX-License-Identifier: MIT
//
// Spatial-chunked routing graph support: SZCI index (eager) + SZRC per-
// cell edge/geom buffers (lazy). Mirrors the Python reference in
// streetzim/tests/szrg_spatial.py and the JS viewer in
// streetzim/resources/viewer/index.html — same on-disk layout, same
// cell-of-node arithmetic, same binary search for local indices.
//
// Concurrency: ``SpatialGraph`` is an actor so cell loads serialise per
// instance — we never fire two overlapping fetches for the same cell.
// Callers that need to route multiple concurrent requests should hold
// the graph as a shared reference; the actor semantics handle the rest.

import Foundation


/// One outbound edge from a node in a spatial graph. Mirrors the flat
/// stride-5 layout inside an SZRC cell — we unpack into fields here
/// rather than a `SIMD5` (stdlib only provides SIMD2/3/4/8 etc.).
public struct SpatialEdge: Sendable, Hashable {
    public let target: UInt32
    public let speedDist: UInt32
    public let geomLocal: UInt32
    public let nameIdx: UInt32
    public let classAccess: UInt32

    /// Convenience: unpack the speed (top 8 bits) and distance in meters
    /// (bottom 24 bits, in decimetres) as v4/v5 SZRG does.
    public var speedKmh: UInt8 { UInt8((speedDist >> 24) & 0xFF) }
    public var distanceMeters: Double { Double(speedDist & 0x00FFFFFF) / 10.0 }
    public var isRoundabout: Bool { (classAccess >> 8) & 1 != 0 }
    public var isLink: Bool {
        let cls = classAccess & 0x1F
        return cls == 2 || cls == 4 || cls == 6 || cls == 8 || cls == 10
    }
}


public enum SZCIError: Error, CustomStringConvertible {
    case badMagic(String)
    case unsupportedVersion(UInt32, String)
    case truncated(String)
    case cellNotFound(Int)
    case cellIDMismatch(expected: Int, got: Int)

    public var description: String {
        switch self {
        case .badMagic(let m): return "SZCI/SZRC bad magic: \(m)"
        case .unsupportedVersion(let v, let kind): return "Unsupported \(kind) version \(v)"
        case .truncated(let s): return "truncated \(s)"
        case .cellNotFound(let cid): return "cell \(cid) not in ZIM"
        case .cellIDMismatch(let expected, let got):
            return "SZRC cell_id mismatch: expected \(expected), got \(got)"
        }
    }
}


/// Eager-loaded spatial index. Mirrors
/// streetzim/tests/szrg_spatial.SZCIIndex.
public struct SZCIIndex: Sendable {
    public let version: UInt32
    public let numNodes: Int
    public let numEdges: Int
    public let numNames: Int
    public let numCells: Int
    /// Degrees-per-cell scaling factor. 1 ⇒ 1° cells; 10 ⇒ 0.1° cells.
    public let cellScale: Int32
    /// Global nodes in source order: index 2*n is lat_e7, 2*n+1 is lon_e7.
    public let nodesScaled: [Int32]
    /// Parallel arrays of cell metadata.
    public let cellLatIdx: [Int32]
    public let cellLonIdx: [Int32]
    public let cellNodeCount: [UInt32]
    public let cellEdgeCount: [UInt32]
    public let cellGeomCount: [UInt32]
    /// (lat_cell_idx, lon_cell_idx) → cell_id lookup.
    public let cellIdByKey: [Int64: Int]

    /// Names table. Held as bytes + offsets to defer UTF-8 decoding to
    /// the few names a routing request actually emits.
    public let nameOffsets: [UInt32]
    public let namesBlob: [UInt8]

    public func name(_ idx: UInt32) -> String {
        let i = Int(idx)
        guard i > 0, i < numNames else { return "" }
        let s = Int(nameOffsets[i])
        let e = Int(nameOffsets[i + 1])
        if e == s { return "" }
        return namesBlob.withUnsafeBufferPointer { bp -> String in
            let slice = UnsafeBufferPointer(start: bp.baseAddress!.advanced(by: s),
                                             count: e - s)
            return String(decoding: slice, as: UTF8.self)
        }
    }

    /// Cell bucket for a node, matching Python cell_of() floor semantics.
    public func cellForNode(_ nodeIdx: Int) -> Int? {
        let latE7 = nodesScaled[nodeIdx * 2]
        let lonE7 = nodesScaled[nodeIdx * 2 + 1]
        // Python: (lat_e7 * scale) // 10_000_000. Swift's integer / rounds
        // toward zero, so apply explicit floor to match on negatives.
        let latProd = Int64(latE7) * Int64(cellScale)
        let lonProd = Int64(lonE7) * Int64(cellScale)
        let latCell = Int32(floorDiv(latProd, 10_000_000))
        let lonCell = Int32(floorDiv(lonProd, 10_000_000))
        return cellIdByKey[Self.encodeKey(lat: latCell, lon: lonCell)]
    }

    @inline(__always)
    static func encodeKey(lat: Int32, lon: Int32) -> Int64 {
        // Pack two signed 32s into a 64-bit key — cheaper than
        // allocating a (Int32, Int32) tuple as a dictionary key.
        return (Int64(lat) << 32) | Int64(UInt32(bitPattern: lon))
    }
}


/// One parsed SZRC cell. Immutable once constructed; safe to share.
public struct SZRCCell: Sendable {
    public let cellId: Int
    public let nodeCount: Int
    public let edgeCount: Int
    public let geomCount: Int
    public let cellNodesGlobal: [UInt32]      // sorted ascending
    public let cellAdj: [UInt32]              // length nodeCount + 1
    public let edges: [UInt32]                // stride 5: target, speed_dist, geom_local, name, class_access
    public let geomOffsets: [UInt32]          // length geomCount + 1
    public let geomBlob: [UInt8]

    /// Returns the local position of `globalNodeIdx` in this cell, or
    /// nil if the cell doesn't own that node.
    public func localIdx(for globalNodeIdx: UInt32) -> Int? {
        var lo = 0
        var hi = cellNodesGlobal.count
        while lo < hi {
            let mid = (lo + hi) >> 1
            let v = cellNodesGlobal[mid]
            if v < globalNodeIdx { lo = mid + 1 }
            else if v > globalNodeIdx { hi = mid }
            else { return mid }
        }
        return nil
    }

    /// Decode a cell-local geom index to [(lat, lon)] points. Zigzag-
    /// varint format, identical to SZRG/SZGM polyline blobs.
    public func decodeGeom(localIdx gi: Int) throws -> [(lat: Double, lon: Double)] {
        let start = Int(geomOffsets[gi])
        let end = Int(geomOffsets[gi + 1])
        if end <= start { return [] }
        return geomBlob.withUnsafeBufferPointer { bp -> [(lat: Double, lon: Double)] in
            let base = UnsafeRawBufferPointer(bp)
            let lon0 = SZRGInt.readInt32LE(base, at: start)
            let lat0 = SZRGInt.readInt32LE(base, at: start + 4)
            var pts: [(lat: Double, lon: Double)] = [(Double(lat0) / 1e7, Double(lon0) / 1e7)]
            if end - start <= 8 { return pts }
            var lonE7 = lon0
            var latE7 = lat0
            var i = start + 8
            while i < end {
                let (dlonRaw, ni1) = SZRGInt.readVarintLE(base, at: i, end: end)
                let dlon = zigzagDecode(dlonRaw)
                let (dlatRaw, ni2) = SZRGInt.readVarintLE(base, at: ni1, end: end)
                let dlat = zigzagDecode(dlatRaw)
                lonE7 &+= Int32(dlon)
                latE7 &+= Int32(dlat)
                pts.append((Double(latE7) / 1e7, Double(lonE7) / 1e7))
                i = ni2
            }
            return pts
        }
    }
}


/// Actor-protected lazy graph: eager SZCI index + cache of fetched cells.
public actor SpatialGraph {
    public let index: SZCIIndex

    private var cells: [Int: SZRCCell] = [:]
    private var inFlight: [Int: Task<SZRCCell, Error>] = [:]
    private var lru: [Int] = []
    private let cacheLimit: Int

    /// Caller-supplied async fetcher. Receives a cell_id, returns the
    /// raw SZRC bytes. Typically wraps a libzim read or HTTP fetch.
    private let fetch: @Sendable (Int) async throws -> Data

    public init(
        index: SZCIIndex,
        cacheLimit: Int = 32,
        fetch: @escaping @Sendable (Int) async throws -> Data
    ) {
        self.index = index
        self.cacheLimit = cacheLimit
        self.fetch = fetch
    }

    /// Returns the list of outbound edges for a global node. Fetches +
    /// parses the owning cell on first access.
    public func edgesOfNode(_ globalNodeIdx: Int) async throws -> [SpatialEdge] {
        guard let cid = index.cellForNode(globalNodeIdx) else { return [] }
        let cell = try await ensureCell(cid)
        guard let local = cell.localIdx(for: UInt32(globalNodeIdx)) else { return [] }
        let eStart = Int(cell.cellAdj[local])
        let eEnd = Int(cell.cellAdj[local + 1])
        var out: [SpatialEdge] = []
        out.reserveCapacity(eEnd - eStart)
        for ei in eStart..<eEnd {
            let base = ei * 5
            out.append(SpatialEdge(
                target: cell.edges[base],
                speedDist: cell.edges[base + 1],
                geomLocal: cell.edges[base + 2],
                nameIdx: cell.edges[base + 3],
                classAccess: cell.edges[base + 4]
            ))
        }
        return out
    }

    /// Decode polyline for an edge whose source is `globalNodeIdx`.
    /// Returns nil for the NO_GEOM sentinel.
    public func decodeGeomForEdge(
        sourceNode globalNodeIdx: Int,
        geomLocal: UInt32
    ) async throws -> [(lat: Double, lon: Double)]? {
        if geomLocal == 0xFFFFFFFF { return nil }
        guard let cid = index.cellForNode(globalNodeIdx) else { return nil }
        let cell = try await ensureCell(cid)
        return try cell.decodeGeom(localIdx: Int(geomLocal))
    }

    // MARK: - Cell cache

    public var cellsLoaded: Int { cells.count }

    private func ensureCell(_ cid: Int) async throws -> SZRCCell {
        if let c = cells[cid] {
            // LRU touch.
            if let idx = lru.firstIndex(of: cid) {
                lru.remove(at: idx)
            }
            lru.append(cid)
            return c
        }
        if let pending = inFlight[cid] {
            return try await pending.value
        }
        let task = Task { @Sendable in
            let data = try await self.fetch(cid)
            let cell = try SZRCCell.parse(data)
            if cell.cellId != cid {
                throw SZCIError.cellIDMismatch(expected: cid, got: cell.cellId)
            }
            return cell
        }
        inFlight[cid] = task
        do {
            let cell = try await task.value
            inFlight[cid] = nil
            cells[cid] = cell
            lru.append(cid)
            if cells.count > cacheLimit {
                let evict = lru.removeFirst()
                cells[evict] = nil
            }
            return cell
        } catch {
            inFlight[cid] = nil
            throw error
        }
    }
}


// MARK: - Parsers

public extension SZCIIndex {
    static func parse(_ data: Data) throws -> SZCIIndex {
        guard data.count >= 32 else { throw SZCIError.truncated("SZCI header") }
        return try data.withUnsafeBytes { raw -> SZCIIndex in
            if !compareMagic(raw, bytes: [0x53, 0x5A, 0x43, 0x49] /* "SZCI" */) {
                throw SZCIError.badMagic("expected SZCI")
            }
            let version = SZRGInt.readUInt32LE(raw, at: 4)
            if version != 1 {
                throw SZCIError.unsupportedVersion(version, "SZCI")
            }
            let numNodes = Int(SZRGInt.readUInt32LE(raw, at: 8))
            let numEdges = Int(SZRGInt.readUInt32LE(raw, at: 12))
            let numNames = Int(SZRGInt.readUInt32LE(raw, at: 16))
            let namesBytes = Int(SZRGInt.readUInt32LE(raw, at: 20))
            let numCells = Int(SZRGInt.readUInt32LE(raw, at: 24))
            let cellScale = SZRGInt.readInt32LE(raw, at: 28)

            var off = 32
            var nodes: [Int32] = []
            nodes.reserveCapacity(numNodes * 2)
            for _ in 0..<(numNodes * 2) {
                nodes.append(SZRGInt.readInt32LE(raw, at: off))
                off += 4
            }

            var cellLat = [Int32](); cellLat.reserveCapacity(numCells)
            var cellLon = [Int32](); cellLon.reserveCapacity(numCells)
            var cellNC = [UInt32](); cellNC.reserveCapacity(numCells)
            var cellEC = [UInt32](); cellEC.reserveCapacity(numCells)
            var cellGC = [UInt32](); cellGC.reserveCapacity(numCells)
            var keyToId: [Int64: Int] = [:]
            keyToId.reserveCapacity(numCells)
            for cid in 0..<numCells {
                let la = SZRGInt.readInt32LE(raw, at: off)
                let lo = SZRGInt.readInt32LE(raw, at: off + 4)
                let nc = SZRGInt.readUInt32LE(raw, at: off + 8)
                let ec = SZRGInt.readUInt32LE(raw, at: off + 12)
                let gc = SZRGInt.readUInt32LE(raw, at: off + 16)
                cellLat.append(la); cellLon.append(lo)
                cellNC.append(nc); cellEC.append(ec); cellGC.append(gc)
                keyToId[encodeKey(lat: la, lon: lo)] = cid
                off += 20
            }

            var nameOffsets = [UInt32](); nameOffsets.reserveCapacity(numNames + 1)
            for _ in 0...numNames {
                nameOffsets.append(SZRGInt.readUInt32LE(raw, at: off))
                off += 4
            }
            var namesBlob = [UInt8](); namesBlob.reserveCapacity(namesBytes)
            for i in 0..<namesBytes {
                namesBlob.append(raw[off + i])
            }

            return SZCIIndex(
                version: version,
                numNodes: numNodes,
                numEdges: numEdges,
                numNames: numNames,
                numCells: numCells,
                cellScale: cellScale,
                nodesScaled: nodes,
                cellLatIdx: cellLat,
                cellLonIdx: cellLon,
                cellNodeCount: cellNC,
                cellEdgeCount: cellEC,
                cellGeomCount: cellGC,
                cellIdByKey: keyToId,
                nameOffsets: nameOffsets,
                namesBlob: namesBlob
            )
        }
    }
}


public extension SZRCCell {
    static func parse(_ data: Data) throws -> SZRCCell {
        guard data.count >= 28 else { throw SZCIError.truncated("SZRC header") }
        return try data.withUnsafeBytes { raw -> SZRCCell in
            if !compareMagic(raw, bytes: [0x53, 0x5A, 0x52, 0x43] /* "SZRC" */) {
                throw SZCIError.badMagic("expected SZRC")
            }
            let version = SZRGInt.readUInt32LE(raw, at: 4)
            if version != 1 {
                throw SZCIError.unsupportedVersion(version, "SZRC")
            }
            let cellId = Int(SZRGInt.readUInt32LE(raw, at: 8))
            let nodeCount = Int(SZRGInt.readUInt32LE(raw, at: 12))
            let edgeCount = Int(SZRGInt.readUInt32LE(raw, at: 16))
            let geomCount = Int(SZRGInt.readUInt32LE(raw, at: 20))
            let geomBytes = Int(SZRGInt.readUInt32LE(raw, at: 24))
            var off = 28

            var cellNodes = [UInt32](); cellNodes.reserveCapacity(nodeCount)
            for _ in 0..<nodeCount {
                cellNodes.append(SZRGInt.readUInt32LE(raw, at: off))
                off += 4
            }
            var cellAdj = [UInt32](); cellAdj.reserveCapacity(nodeCount + 1)
            for _ in 0...nodeCount {
                cellAdj.append(SZRGInt.readUInt32LE(raw, at: off))
                off += 4
            }
            var edges = [UInt32](); edges.reserveCapacity(edgeCount * 5)
            for _ in 0..<(edgeCount * 5) {
                edges.append(SZRGInt.readUInt32LE(raw, at: off))
                off += 4
            }
            var geomOffsets = [UInt32](); geomOffsets.reserveCapacity(geomCount + 1)
            for _ in 0...geomCount {
                geomOffsets.append(SZRGInt.readUInt32LE(raw, at: off))
                off += 4
            }
            var geomBlob = [UInt8](); geomBlob.reserveCapacity(geomBytes)
            for i in 0..<geomBytes {
                geomBlob.append(raw[off + i])
            }

            return SZRCCell(
                cellId: cellId,
                nodeCount: nodeCount,
                edgeCount: edgeCount,
                geomCount: geomCount,
                cellNodesGlobal: cellNodes,
                cellAdj: cellAdj,
                edges: edges,
                geomOffsets: geomOffsets,
                geomBlob: geomBlob
            )
        }
    }
}


// MARK: - Shared low-level helpers

enum SZRGInt {
    @inline(__always)
    static func readUInt32LE(_ raw: UnsafeRawBufferPointer, at pos: Int) -> UInt32 {
        return UInt32(raw[pos])
            | (UInt32(raw[pos + 1]) << 8)
            | (UInt32(raw[pos + 2]) << 16)
            | (UInt32(raw[pos + 3]) << 24)
    }

    @inline(__always)
    static func readInt32LE(_ raw: UnsafeRawBufferPointer, at pos: Int) -> Int32 {
        return Int32(bitPattern: readUInt32LE(raw, at: pos))
    }

    @inline(__always)
    static func readVarintLE(_ raw: UnsafeRawBufferPointer, at pos: Int, end: Int) -> (UInt64, Int) {
        var shift: UInt64 = 0
        var result: UInt64 = 0
        var i = pos
        while i < end {
            let b = raw[i]
            i += 1
            result |= UInt64(b & 0x7F) << shift
            if (b & 0x80) == 0 { return (result, i) }
            shift += 7
            if shift >= 64 { return (result, i) }
        }
        return (result, i)
    }
}


@inline(__always)
private func compareMagic(_ raw: UnsafeRawBufferPointer, bytes: [UInt8]) -> Bool {
    for i in 0..<bytes.count {
        if raw[i] != bytes[i] { return false }
    }
    return true
}

@inline(__always)
private func zigzagDecode(_ n: UInt64) -> Int64 {
    Int64(bitPattern: n >> 1) ^ -(Int64(bitPattern: n & 1))
}

/// Floor division for Int64 — equivalent to Python's // operator. Swift's
/// built-in / rounds toward zero, which would place e.g. -122.001 in the
/// wrong cell for cell_of() calls.
@inline(__always)
private func floorDiv(_ a: Int64, _ b: Int64) -> Int64 {
    let q = a / b
    let r = a % b
    if (r != 0) && ((r < 0) != (b < 0)) {
        return q - 1
    }
    return q
}
