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
