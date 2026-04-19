// SPDX-License-Identifier: MIT
//
// Test-only encoder that produces an SZRG v2 blob from a small in-memory
// graph description. Mirrors `mcpzim.routing.encode_graph_v2` in the Python
// library so the two implementations can be exercised with identical fixtures.
//
// Exposed as an API entry point rather than kept test-private so downstream
// host apps can also construct a synthetic graph for their own tests.

import Foundation

/// Build an SZRG v2 `graph.bin` blob in memory.
///
/// - Parameters:
///   - nodes: array of `(lat, lon)` in degrees.
///   - edges: tuples `(src, dst, distance_m, speed_kmh, name_idx)`. Edges
///     written by this helper always carry the "no geometry" sentinel.
///   - names: string table. Index 0 MUST be `""` (the unnamed sentinel).
public func encodeGraphV2(
    nodes: [(Double, Double)],
    edges: [(Int, Int, Double, Int, Int)],
    names: [String]
) -> Data {
    precondition(names.isEmpty || names[0].isEmpty, "names[0] must be \"\"")

    // Build CSR adjacency.
    var buckets: [[(dst: UInt32, distDm: UInt32, speedKmh: UInt8, nameIdx: UInt32)]] =
        Array(repeating: [], count: nodes.count)
    for (src, dst, distM, speedKmh, nameIdx) in edges {
        let dm = UInt32((distM * 10).rounded())
        buckets[src].append((UInt32(dst), dm, UInt8(speedKmh), UInt32(nameIdx)))
    }

    var adjOffsets: [UInt32] = [0]
    var flat: [(dst: UInt32, distDm: UInt32, speedKmh: UInt8, nameIdx: UInt32)] = []
    for b in buckets {
        flat.append(contentsOf: b)
        adjOffsets.append(UInt32(flat.count))
    }

    // Names blob.
    var nameOffsets: [UInt32] = [0]
    var namesBlob = Data()
    for s in names {
        let bytes = s.data(using: .utf8) ?? Data()
        namesBlob.append(bytes)
        nameOffsets.append(UInt32(namesBlob.count))
    }

    // No geoms in this helper.
    let geomOffsets: [UInt32] = [0]
    let geomBlob = Data()

    var out = Data()
    appendMagic(&out, "SZRG")
    appendU32(&out, 2)                              // version
    appendU32(&out, UInt32(nodes.count))            // numNodes
    appendU32(&out, UInt32(flat.count))             // numEdges
    appendU32(&out, 0)                              // numGeoms
    appendU32(&out, UInt32(geomBlob.count))         // geomBytes
    appendU32(&out, UInt32(names.count))            // numNames
    appendU32(&out, UInt32(namesBlob.count))        // namesBytes

    for (lat, lon) in nodes {
        appendI32(&out, Int32((lat * 1e7).rounded()))
        appendI32(&out, Int32((lon * 1e7).rounded()))
    }
    for off in adjOffsets { appendU32(&out, off) }
    for edge in flat {
        appendU32(&out, edge.dst)
        appendU32(&out, edge.distDm)
        let speedGeom: UInt32 = (UInt32(edge.speedKmh) << 24) | 0x00FFFFFF
        appendU32(&out, speedGeom)
        appendU32(&out, edge.nameIdx)
    }
    for off in geomOffsets { appendU32(&out, off) }
    out.append(geomBlob)
    for off in nameOffsets { appendU32(&out, off) }
    out.append(namesBlob)
    return out
}

// MARK: - Little-endian writers

private func appendMagic(_ data: inout Data, _ s: String) {
    data.append(s.data(using: .ascii) ?? Data())
}

private func appendU32(_ data: inout Data, _ value: UInt32) {
    var v = value.littleEndian
    withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
}

private func appendI32(_ data: inout Data, _ value: Int32) {
    var v = value.littleEndian
    withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
}
