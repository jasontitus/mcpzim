// SPDX-License-Identifier: MIT
//
// Chunked routing-graph reassembly.
//
// Very large streetzim ZIMs (continent-scale — US, Europe, Japan) can
// overflow libzim's per-cluster size budget. In that case the writer
// splits the main graph.bin (and the SZGM geoms companion) into N byte-
// range ZIM entries ``graph-chunk-NNNN.bin`` plus a JSON manifest:
//
//   {
//     "schema": 1,
//     "total_bytes": 2_012_345_678,
//     "sha256": "…",
//     "chunks": [
//       { "path": "graph-chunk-0000.bin", "bytes": 268_435_456 },
//       …
//     ]
//   }
//
// ``reassembleChunked`` reads the manifest, dispatches a caller-supplied
// ``loader(chunkPath)`` for each chunk, and concatenates them in manifest
// order. It verifies:
//   * schema == 1
//   * each chunk's ``bytes`` matches the payload length
//   * total length matches ``total_bytes``
//   * the concatenation's sha256 matches ``sha256`` (when present)
//
// Any mismatch throws ``SZRGError.chunkedReassembly`` with a message
// pointing at the offending chunk. The point is: torn uploads must fail
// loud rather than silently produce a corrupt graph.

import CryptoKit
import Foundation


public enum SZRGChunked {
    // Hard caps so a malformed or hostile manifest can't drive the iOS
    // process into OOM before any payload is even read. The writer emits
    // ~20-30 chunks for a continent-scale ZIM at ~256 MB each (cluster
    // ceiling), so 256 chunks × 2 GB each is well above any legitimate
    // value and safely under iOS's per-process memory ceiling.
    public static let maxChunks = 256
    public static let maxChunkBytes = 2 * 1024 * 1024 * 1024   // 2 GB
    public static let maxTotalBytes = 4 * 1024 * 1024 * 1024   // 4 GB

    /// Parse the manifest, pull every chunk through `loader`, concatenate,
    /// and verify. `loader(path)` receives each chunk's manifest path
    /// (e.g. `"graph-chunk-0000.bin"`) — the caller maps that to a ZIM
    /// entry (typically by prefixing with the manifest's directory).
    public static func reassembleChunked(
        manifest manifestData: Data,
        loader: (String) throws -> Data
    ) throws -> Data {
        struct Chunk: Decodable {
            let path: String
            let bytes: Int
        }
        struct Manifest: Decodable {
            let schema: Int
            let total_bytes: Int
            let sha256: String?
            let chunks: [Chunk]
        }

        let m: Manifest
        do {
            m = try JSONDecoder().decode(Manifest.self, from: manifestData)
        } catch {
            throw SZRGError.chunkedReassembly(
                "manifest JSON decode failed: \(error)"
            )
        }
        if m.schema != 1 {
            throw SZRGError.chunkedReassembly(
                "unsupported manifest schema \(m.schema)"
            )
        }
        if m.total_bytes < 0 || m.total_bytes > maxTotalBytes {
            throw SZRGError.chunkedReassembly(
                "total_bytes \(m.total_bytes) outside [0, \(maxTotalBytes)]"
            )
        }
        if m.chunks.count > maxChunks {
            throw SZRGError.chunkedReassembly(
                "chunk count \(m.chunks.count) exceeds \(maxChunks)"
            )
        }
        // Per-chunk size sanity + overflow-safe sum check against total_bytes.
        var declaredSum = 0
        for ch in m.chunks {
            if ch.bytes < 0 || ch.bytes > maxChunkBytes {
                throw SZRGError.chunkedReassembly(
                    "chunk \(ch.path) bytes \(ch.bytes) outside [0, \(maxChunkBytes)]"
                )
            }
            let (sum, overflow) = declaredSum.addingReportingOverflow(ch.bytes)
            if overflow || sum > m.total_bytes {
                throw SZRGError.chunkedReassembly(
                    "sum of chunk bytes exceeds manifest total_bytes \(m.total_bytes)"
                )
            }
            declaredSum = sum
        }

        var out = Data()
        out.reserveCapacity(m.total_bytes)
        for ch in m.chunks {
            let data = try loader(ch.path)
            if data.count != ch.bytes {
                throw SZRGError.chunkedReassembly(
                    "chunk \(ch.path) size \(data.count) != manifest \(ch.bytes)"
                )
            }
            out.append(data)
        }

        if out.count != m.total_bytes {
            throw SZRGError.chunkedReassembly(
                "reassembled \(out.count) B != manifest total_bytes \(m.total_bytes)"
            )
        }

        if let expectedSha = m.sha256 {
            let digest = SHA256.hash(data: out)
            let got = digest.map { String(format: "%02x", $0) }.joined()
            if got != expectedSha {
                throw SZRGError.chunkedReassembly(
                    "sha256 mismatch: manifest=\(expectedSha) got=\(got)"
                )
            }
        }

        return out
    }
}
