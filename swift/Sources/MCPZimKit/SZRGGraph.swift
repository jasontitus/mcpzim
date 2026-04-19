// SPDX-License-Identifier: MIT
//
// Parser for streetzim's SZRG v2 binary routing graph (entry
// "routing-data/graph.bin" inside the ZIM). Byte layout mirrors the writer in
// streetzim/create_osm_zim.py and the JS reader in resources/viewer/index.html.
// All multi-byte integers are little-endian.

import Foundation

public enum SZRGError: Error, CustomStringConvertible {
    case tooSmall
    case badMagic([UInt8])
    case unsupportedVersion(UInt32)
    case truncated(String)

    public var description: String {
        switch self {
        case .tooSmall: return "graph.bin too small"
        case .badMagic(let b): return "bad magic \(b)"
        case .unsupportedVersion(let v): return "unsupported SZRG version \(v)"
        case .truncated(let s): return "truncated \(s)"
        }
    }
}

public struct SZRGGraph: Sendable {
    // Nodes in degrees.
    public let lat: [Double]
    public let lon: [Double]
    // CSR adjacency.
    public let adjOffsets: [UInt32]      // len == numNodes + 1
    // Parallel edge arrays.
    public let edgeTargets: [UInt32]
    public let edgeDistMeters: [Double]
    public let edgeSpeedKmh: [UInt8]
    public let edgeGeomIdx: [Int32]      // -1 when no geometry
    public let edgeNameIdx: [UInt32]     // 0 for unnamed
    // String table — index 0 is always "" (the unnamed sentinel).
    public let names: [String]
    // Polylines — each a flat array alternating latitude/longitude doubles.
    public let geoms: [[(lat: Double, lon: Double)]]

    public var numNodes: Int { lat.count }
    public var numEdges: Int { edgeTargets.count }

    public func name(_ idx: UInt32) -> String {
        let i = Int(idx)
        guard i > 0, i < names.count else { return "" }
        return names[i]
    }

    public static func parse(_ data: Data) throws -> SZRGGraph {
        guard data.count >= 32 else { throw SZRGError.tooSmall }
        var cursor = Cursor(data: data)
        let magic = cursor.readBytes(4)
        guard magic == [0x53, 0x5A, 0x52, 0x47] else { throw SZRGError.badMagic(magic) }
        let version: UInt32 = try cursor.readU32()
        guard version == 2 else { throw SZRGError.unsupportedVersion(version) }
        let numNodes = Int(try cursor.readU32())
        let numEdges = Int(try cursor.readU32())
        let numGeoms = Int(try cursor.readU32())
        let geomBytes = Int(try cursor.readU32())
        let numNames = Int(try cursor.readU32())
        let namesBytes = Int(try cursor.readU32())

        // Nodes: int32 lat_e7, int32 lon_e7.
        var lat = [Double](); lat.reserveCapacity(numNodes)
        var lon = [Double](); lon.reserveCapacity(numNodes)
        for _ in 0..<numNodes {
            let latE7 = try cursor.readI32()
            let lonE7 = try cursor.readI32()
            lat.append(Double(latE7) / 1e7)
            lon.append(Double(lonE7) / 1e7)
        }

        // Adjacency offsets.
        var adjOffsets = [UInt32](); adjOffsets.reserveCapacity(numNodes + 1)
        for _ in 0...numNodes { adjOffsets.append(try cursor.readU32()) }

        // Edges.
        var edgeTargets = [UInt32](); edgeTargets.reserveCapacity(numEdges)
        var edgeDistMeters = [Double](); edgeDistMeters.reserveCapacity(numEdges)
        var edgeSpeedKmh = [UInt8](); edgeSpeedKmh.reserveCapacity(numEdges)
        var edgeGeomIdx = [Int32](); edgeGeomIdx.reserveCapacity(numEdges)
        var edgeNameIdx = [UInt32](); edgeNameIdx.reserveCapacity(numEdges)
        for _ in 0..<numEdges {
            let target = try cursor.readU32()
            let distDm = try cursor.readU32()
            let speedGeom = try cursor.readU32()
            let nameIdx = try cursor.readU32()
            edgeTargets.append(target)
            edgeDistMeters.append(Double(distDm) / 10.0)
            edgeSpeedKmh.append(UInt8((speedGeom >> 24) & 0xFF))
            let geomIdx24 = speedGeom & 0x00FFFFFF
            edgeGeomIdx.append(geomIdx24 == 0xFFFFFF ? -1 : Int32(geomIdx24))
            edgeNameIdx.append(nameIdx)
        }

        // Geometry offsets + blob.
        var geomOffsets = [UInt32](); geomOffsets.reserveCapacity(numGeoms + 1)
        for _ in 0...numGeoms { geomOffsets.append(try cursor.readU32()) }
        let geomBlob = try cursor.readData(count: geomBytes)

        // Name offsets + blob.
        var nameOffsets = [UInt32](); nameOffsets.reserveCapacity(numNames + 1)
        for _ in 0...numNames { nameOffsets.append(try cursor.readU32()) }
        let namesBlob = try cursor.readData(count: namesBytes)

        // Decode polylines.
        var geoms: [[(lat: Double, lon: Double)]] = []
        geoms.reserveCapacity(numGeoms)
        for g in 0..<numGeoms {
            let start = Int(geomOffsets[g])
            let end = Int(geomOffsets[g + 1])
            geoms.append(try Self.decodeGeom(geomBlob, start: start, end: end))
        }

        // Decode names.
        var names: [String] = []
        names.reserveCapacity(numNames)
        for n in 0..<numNames {
            let s = Int(nameOffsets[n])
            let e = Int(nameOffsets[n + 1])
            let slice = namesBlob.subdata(in: s..<e)
            names.append(String(data: slice, encoding: .utf8) ?? "")
        }

        return SZRGGraph(
            lat: lat,
            lon: lon,
            adjOffsets: adjOffsets,
            edgeTargets: edgeTargets,
            edgeDistMeters: edgeDistMeters,
            edgeSpeedKmh: edgeSpeedKmh,
            edgeGeomIdx: edgeGeomIdx,
            edgeNameIdx: edgeNameIdx,
            names: names,
            geoms: geoms
        )
    }

    static func decodeGeom(_ blob: Data, start: Int, end: Int) throws -> [(lat: Double, lon: Double)] {
        if end <= start { return [] }
        var cursor = Cursor(data: blob.subdata(in: start..<end))
        let lon0 = try cursor.readI32()
        let lat0 = try cursor.readI32()
        var points: [(lat: Double, lon: Double)] = [(Double(lat0) / 1e7, Double(lon0) / 1e7)]
        var lonE7 = lon0
        var latE7 = lat0
        while !cursor.atEnd {
            let dlon = Self.zigzagDecode(try cursor.readVarint())
            let dlat = Self.zigzagDecode(try cursor.readVarint())
            lonE7 &+= Int32(dlon)
            latE7 &+= Int32(dlat)
            points.append((Double(latE7) / 1e7, Double(lonE7) / 1e7))
        }
        return points
    }

    static func zigzagDecode(_ n: UInt64) -> Int64 {
        Int64(bitPattern: n >> 1) ^ -(Int64(bitPattern: n & 1))
    }

    /// Linear-scan nearest node by haversine distance. Good enough for the
    /// graph sizes streetzim ships (city or state scale); if profiling reveals
    /// this as a hotspot, swap in a k-d tree on `(lat, lon)`.
    public func nearestNode(lat: Double, lon: Double) -> Int {
        var best = -1
        var bestD = Double.infinity
        for i in 0..<numNodes {
            let d = haversineMeters(lat, lon, self.lat[i], self.lon[i])
            if d < bestD { bestD = d; best = i }
        }
        return best
    }
}

// MARK: - Byte cursor

private struct Cursor {
    let data: Data
    private(set) var pos: Int = 0

    var atEnd: Bool { pos >= data.count }

    mutating func readBytes(_ count: Int) -> [UInt8] {
        let end = pos + count
        let slice = Array(data[pos..<end])
        pos = end
        return slice
    }

    mutating func readData(count: Int) throws -> Data {
        let end = pos + count
        guard end <= data.count else { throw SZRGError.truncated("blob") }
        let slice = data.subdata(in: pos..<end)
        pos = end
        return slice
    }

    mutating func readU32() throws -> UInt32 {
        let end = pos + 4
        guard end <= data.count else { throw SZRGError.truncated("u32") }
        var v: UInt32 = 0
        withUnsafeMutableBytes(of: &v) { buf in
            data.copyBytes(to: buf, from: pos..<end)
        }
        pos = end
        return UInt32(littleEndian: v)
    }

    mutating func readI32() throws -> Int32 {
        Int32(bitPattern: try readU32())
    }

    mutating func readVarint() throws -> UInt64 {
        var shift: UInt64 = 0
        var result: UInt64 = 0
        while true {
            guard pos < data.count else { throw SZRGError.truncated("varint") }
            let b = data[pos]; pos += 1
            result |= UInt64(b & 0x7F) << shift
            if (b & 0x80) == 0 { return result }
            shift += 7
            if shift >= 64 { throw SZRGError.truncated("varint too long") }
        }
    }
}

// MARK: - Great-circle distance

public func haversineMeters(_ lat1: Double, _ lon1: Double, _ lat2: Double, _ lon2: Double) -> Double {
    let R = 6_371_000.0
    let rlat1 = lat1 * .pi / 180
    let rlat2 = lat2 * .pi / 180
    let dlat = (lat2 - lat1) * .pi / 180
    let dlon = (lon2 - lon1) * .pi / 180
    let a = sin(dlat / 2) * sin(dlat / 2)
        + cos(rlat1) * cos(rlat2) * sin(dlon / 2) * sin(dlon / 2)
    return 2 * R * asin(min(1.0, sqrt(a)))
}
