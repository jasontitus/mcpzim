// SPDX-License-Identifier: MIT
//
// Parser for streetzim's SZRG v2 binary routing graph (entry
// "routing-data/graph.bin" inside the ZIM). Byte layout mirrors the writer in
// streetzim/create_osm_zim.py and the JS reader in resources/viewer/index.html.
// All multi-byte integers are little-endian.
//
// The parser is zero-copy: it reads everything via `Data.withUnsafeBytes`
// into raw pointers and offsets, never slicing `Data` itself. The earlier
// `subdata(in:)`-based implementation allocated a fresh `Data` per polyline
// segment, which for a country-scale routing graph (millions of segments)
// blew resident memory up by 100× the input size.

import Foundation

public enum SZRGError: Error, CustomStringConvertible {
    case tooSmall
    case badMagic([UInt8])
    case unsupportedVersion(UInt32)
    case truncated(String)
    /// Richer variant that carries enough context (pos / size / section /
    /// parsed header) to diagnose format drift — e.g. an unexpected
    /// edge-stride on a future SZRG version.
    case truncatedAt(String, pos: Int, size: Int, header: String)

    public var description: String {
        switch self {
        case .tooSmall: return "graph.bin too small"
        case .badMagic(let b): return "bad magic \(b)"
        case .unsupportedVersion(let v): return "unsupported SZRG version \(v)"
        case .truncated(let s): return "truncated \(s)"
        case .truncatedAt(let s, let pos, let size, let hdr):
            return "truncated \(s) at pos=\(pos) size=\(size) header=[\(hdr)]"
        }
    }
}

public struct SZRGGraph: Sendable {
    public let lat: [Double]
    public let lon: [Double]
    public let adjOffsets: [UInt32]
    public let edgeTargets: [UInt32]
    public let edgeDistMeters: [Double]
    public let edgeSpeedKmh: [UInt8]
    public let edgeGeomIdx: [Int32]
    public let edgeNameIdx: [UInt32]
    public let names: [String]
    public let geoms: [[(lat: Double, lon: Double)]]

    public var numNodes: Int { lat.count }
    public var numEdges: Int { edgeTargets.count }

    public func name(_ idx: UInt32) -> String {
        let i = Int(idx)
        guard i > 0, i < names.count else { return "" }
        return names[i]
    }

    /// Rough lower bound on the parsed graph's steady-state footprint —
    /// sum of the fixed-stride arrays. Excludes `names` (Swift String
    /// overhead per entry) and `geoms` (skipped when parsed with
    /// `decodeGeoms=false`). Useful for attributing the `+2 GB on graph
    /// load` memory spike to the specific arrays driving it.
    public var estimatedBytes: Int {
        let n = numNodes, e = numEdges
        return 8 * n  // lat Double
             + 8 * n  // lon Double
             + 4 * (n + 1)  // adjOffsets UInt32
             + 4 * e  // edgeTargets UInt32
             + 8 * e  // edgeDistMeters Double
             + 1 * e  // edgeSpeedKmh UInt8
             + 4 * e  // edgeGeomIdx Int32
             + 4 * e  // edgeNameIdx UInt32
    }

    /// Parse the binary routing graph. When `decodeGeoms` is `false` the
    /// per-edge polyline blob is skipped entirely — on a country-scale
    /// graph (Baltics, 977 MB ZIM) that saves ~600 MB of Swift tuples and
    /// around 3 s of parse time. Routing still works because A* only needs
    /// node lat/lon + edge distances; the LLM-facing `polyline` field in
    /// the route result falls back to node-sequence coordinates.
    public static func parse(_ data: Data, decodeGeoms: Bool = true) throws -> SZRGGraph {
        guard data.count >= 32 else { throw SZRGError.tooSmall }
        return try data.withUnsafeBytes { raw -> SZRGGraph in
            var p = RawCursor(base: raw, count: data.count, pos: 0)
            try p.expectMagic([0x53, 0x5A, 0x52, 0x47])
            let version = try p.readU32()
            // v2: edge = (target_u32, dist_dm_u32_full, (speed<<24|geom24)_u32, name_u32)
            //     — geom index capped at 2^24 ≈ 16.78 M, which truncates
            //     continental-scale graphs (Japan, Europe).
            // v3: edge = (target_u32, (speed<<24|dist_dm24)_u32, geom_u32_full, name_u32)
            //     — speed/dist are packed instead, geom widened to full u32
            //     (0xFFFFFFFF = no geom). Everything outside the edge record
            //     is identical to v2.
            // v4: same 4 u32s as v3 + one trailing class_access_u32 per edge.
            //     Bits 0..4 encode a road-class ordinal (motorway / footway /
            //     …), bits 5..7 are access-override flags (foot=no, bicycle=no,
            //     oneway). The rest is reserved. Routing here still only
            //     consumes (dist, speed, geom, name), so we skip-parse the
            //     extra u32 — enough to make v4 ZIMs load. When road-class
            //     warnings graduate out of the design doc in
            //     streetzim/docs/driving-mode-road-class-warnings.md, widen
            //     the graph struct to carry it.
            guard version == 2 || version == 3 || version == 4 else {
                throw SZRGError.unsupportedVersion(version)
            }
            let numNodes = Int(try p.readU32())
            let numEdges = Int(try p.readU32())
            let numGeoms = Int(try p.readU32())
            let geomBytes = Int(try p.readU32())
            let numNames = Int(try p.readU32())
            let namesBytes = Int(try p.readU32())
            let headerStr = "v=\(version) nodes=\(numNodes) edges=\(numEdges) "
                + "geoms=\(numGeoms) geomBytes=\(geomBytes) "
                + "names=\(numNames) namesBytes=\(namesBytes) size=\(data.count)"

            // Nodes — lat/lon in 1e-7 degrees.
            var lat = [Double](); lat.reserveCapacity(numNodes)
            var lon = [Double](); lon.reserveCapacity(numNodes)
            for _ in 0..<numNodes {
                let latE7 = try p.readI32()
                let lonE7 = try p.readI32()
                lat.append(Double(latE7) / 1e7)
                lon.append(Double(lonE7) / 1e7)
            }

            var adjOffsets = [UInt32](); adjOffsets.reserveCapacity(numNodes + 1)
            for _ in 0...numNodes { adjOffsets.append(try p.readU32()) }

            var edgeTargets = [UInt32](); edgeTargets.reserveCapacity(numEdges)
            var edgeDistMeters = [Double](); edgeDistMeters.reserveCapacity(numEdges)
            var edgeSpeedKmh = [UInt8](); edgeSpeedKmh.reserveCapacity(numEdges)
            var edgeGeomIdx = [Int32](); edgeGeomIdx.reserveCapacity(numEdges)
            var edgeNameIdx = [UInt32](); edgeNameIdx.reserveCapacity(numEdges)
            for _ in 0..<numEdges {
                let target = try p.readU32()
                if version == 2 {
                    let distDm = try p.readU32()
                    let speedGeom = try p.readU32()
                    let nameIdx = try p.readU32()
                    edgeTargets.append(target)
                    edgeDistMeters.append(Double(distDm) / 10.0)
                    edgeSpeedKmh.append(UInt8((speedGeom >> 24) & 0xFF))
                    let geomIdx24 = speedGeom & 0x00FFFFFF
                    edgeGeomIdx.append(geomIdx24 == 0xFFFFFF ? -1 : Int32(geomIdx24))
                    edgeNameIdx.append(nameIdx)
                } else {
                    // v3 + v4: speed+dist packed together, geom_idx gets a
                    // full u32. v4 adds a trailing class_access u32 we don't
                    // consume yet — read + discard so the cursor lands at
                    // the next edge.
                    let speedDist = try p.readU32()
                    let geom = try p.readU32()
                    let nameIdx = try p.readU32()
                    if version == 4 { _ = try p.readU32() }
                    edgeTargets.append(target)
                    let distDm24 = speedDist & 0x00FFFFFF
                    edgeDistMeters.append(Double(distDm24) / 10.0)
                    edgeSpeedKmh.append(UInt8((speedDist >> 24) & 0xFF))
                    edgeGeomIdx.append(geom == 0xFFFFFFFF ? -1 : Int32(bitPattern: geom))
                    edgeNameIdx.append(nameIdx)
                }
            }

            // Geometry offsets + blob (blob stays in place; we just track
            // its start offset and length and decode into `geoms` in one
            // pass without any `subdata` copies).
            let posAfterEdges = p.pos
            var geomOffsets = [UInt32](); geomOffsets.reserveCapacity(numGeoms + 1)
            for _ in 0...numGeoms { geomOffsets.append(try p.readU32()) }
            let geomBase = p.pos
            do {
                try p.advance(geomBytes)
            } catch {
                throw SZRGError.truncatedAt(
                    "geomBlob (posAfterEdges=\(posAfterEdges))",
                    pos: p.pos, size: data.count, header: headerStr
                )
            }

            // Name offsets + blob.
            var nameOffsets = [UInt32](); nameOffsets.reserveCapacity(numNames + 1)
            for _ in 0...numNames { nameOffsets.append(try p.readU32()) }
            let namesBase = p.pos
            do {
                try p.advance(namesBytes)
            } catch {
                throw SZRGError.truncatedAt(
                    "namesBlob",
                    pos: p.pos, size: data.count, header: headerStr
                )
            }

            // Decode polylines directly against the unsafe buffer — no copies.
            // With `decodeGeoms=false` we allocate the outer array as empty
            // placeholders so indices still line up; each edge's polyline
            // falls back to its endpoint nodes' lat/lon.
            var geoms: [[(lat: Double, lon: Double)]] = []
            if decodeGeoms {
                geoms.reserveCapacity(numGeoms)
                for g in 0..<numGeoms {
                    let start = geomBase + Int(geomOffsets[g])
                    let end = geomBase + Int(geomOffsets[g + 1])
                    geoms.append(try Self.decodeGeom(raw, start: start, end: end))
                }
            } else {
                geoms = Array(repeating: [], count: numGeoms)
            }

            // Decode names — construct a Swift `String` from each bounded
            // slice. Each string allocates (Swift strings are not zero-copy
            // over an external buffer), but the *blob itself* never gets
            // copied.
            var names: [String] = []
            names.reserveCapacity(numNames)
            for n in 0..<numNames {
                let s = namesBase + Int(nameOffsets[n])
                let e = namesBase + Int(nameOffsets[n + 1])
                let bytesPointer = raw.baseAddress!.advanced(by: s).assumingMemoryBound(to: UInt8.self)
                let length = e - s
                let buffer = UnsafeBufferPointer(start: bytesPointer, count: length)
                names.append(String(decoding: buffer, as: UTF8.self))
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
    }

    /// Decode one delta-encoded polyline from raw bytes. `start` and `end`
    /// are byte offsets into `raw`; no allocation happens here beyond the
    /// returned `[(lat, lon)]` array.
    static func decodeGeom(
        _ raw: UnsafeRawBufferPointer,
        start: Int,
        end: Int
    ) throws -> [(lat: Double, lon: Double)] {
        if end <= start { return [] }
        var cursor = RawCursor(base: raw, count: end, pos: start)
        let lon0 = try cursor.readI32()
        let lat0 = try cursor.readI32()
        var points: [(lat: Double, lon: Double)] = [(Double(lat0) / 1e7, Double(lon0) / 1e7)]
        points.reserveCapacity(8) // small buckets amortize; arrays grow as needed.
        var lonE7 = lon0
        var latE7 = lat0
        while cursor.pos < end {
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

// MARK: - Raw cursor over UnsafeRawBufferPointer

private struct RawCursor {
    let base: UnsafeRawBufferPointer
    let count: Int
    var pos: Int

    mutating func expectMagic(_ expected: [UInt8]) throws {
        guard pos + expected.count <= count else { throw SZRGError.truncated("magic") }
        for i in 0..<expected.count {
            if base[pos + i] != expected[i] {
                let actual = Array(base[pos..<pos + expected.count])
                throw SZRGError.badMagic(actual)
            }
        }
        pos += expected.count
    }

    mutating func advance(_ n: Int) throws {
        guard pos + n <= count else { throw SZRGError.truncated("advance") }
        pos += n
    }

    mutating func readU32() throws -> UInt32 {
        let end = pos + 4
        guard end <= count else { throw SZRGError.truncated("u32") }
        // Little-endian: base[0] is LSB.
        let v = UInt32(base[pos])
            | UInt32(base[pos + 1]) << 8
            | UInt32(base[pos + 2]) << 16
            | UInt32(base[pos + 3]) << 24
        pos = end
        return v
    }

    mutating func readI32() throws -> Int32 {
        Int32(bitPattern: try readU32())
    }

    mutating func readVarint() throws -> UInt64 {
        var shift: UInt64 = 0
        var result: UInt64 = 0
        while true {
            guard pos < count else { throw SZRGError.truncated("varint") }
            let b = base[pos]
            pos += 1
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
