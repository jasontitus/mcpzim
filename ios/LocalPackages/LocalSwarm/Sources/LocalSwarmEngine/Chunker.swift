import Foundation
import CryptoKit

/// Builds a `SwarmManifest` from a set of source files by slicing each file into
/// fixed-size chunks and hashing every chunk with SHA-256.
public enum Chunker {
    public static let defaultChunkSize = 1 << 20 // 1 MiB

    public enum ChunkerError: Error {
        case noFiles
        case unreadable(URL)
    }

    /// Convenience over the `items:` variant for flat shares — every file
    /// keeps its bare filename as the manifest path.
    public static func buildManifest(name: String,
                                     fileURLs: [URL],
                                     chunkSize: Int = defaultChunkSize,
                                     progress: (@Sendable (Double) -> Void)? = nil) throws -> (manifest: SwarmManifest, orderedURLs: [URL]) {
        try buildManifest(name: name,
                          items: fileURLs.map { ShareItem(url: $0) },
                          chunkSize: chunkSize,
                          progress: progress)
    }

    /// Produces a manifest plus the source URLs in manifest-file order (so a
    /// seeding `ChunkStore` can map file index -> URL). Each item carries the
    /// relative path it gets in the manifest, allowing directory shares whose
    /// layout the receiver recreates.
    public static func buildManifest(name: String,
                                     items: [ShareItem],
                                     chunkSize: Int = defaultChunkSize,
                                     progress: (@Sendable (Double) -> Void)? = nil) throws -> (manifest: SwarmManifest, orderedURLs: [URL]) {
        guard !items.isEmpty else { throw ChunkerError.noFiles }

        var files: [SwarmFile] = []
        var hashes: [String] = []
        var orderedURLs: [URL] = []
        var globalIndex = 0
        var total: Int64 = 0

        // Total bytes to hash, so progress can be reported as a fraction.
        let grandTotal = (try? items.reduce(Int64(0)) { try $0 + fileSize(of: $1.url) }) ?? 0
        var hashed: Int64 = 0

        for item in items {
            let url = item.url
            let size = try fileSize(of: url)
            let startIndex = globalIndex

            if size > 0 {
                guard let handle = try? FileHandle(forReadingFrom: url) else {
                    throw ChunkerError.unreadable(url)
                }
                defer { try? handle.close() }
                var remaining = size
                while remaining > 0 {
                    // Drain each chunk's autoreleased Data immediately. Without
                    // this, hashing a multi-GB file in one loop accumulates the
                    // whole file in memory and the OS jetsams the app.
                    let consumed: Int = try autoreleasepool {
                        let wanted = Int(min(Int64(chunkSize), remaining))
                        let data = try handle.readFully(wanted)
                        guard !data.isEmpty else { return 0 }
                        hashes.append(Hashing.sha256Hex(data))
                        return data.count
                    }
                    if consumed == 0 { break }
                    globalIndex += 1
                    remaining -= Int64(consumed)
                    hashed += Int64(consumed)
                    if let progress, grandTotal > 0, globalIndex % 64 == 0 || hashed >= grandTotal {
                        progress(Double(hashed) / Double(grandTotal))
                    }
                }
            }

            let endIndex = size > 0 ? globalIndex - 1 : startIndex
            files.append(SwarmFile(path: item.relativePath,
                                   sizeBytes: size,
                                   startChunkIndex: startIndex,
                                   endChunkIndex: endIndex))
            orderedURLs.append(url)
            total += size
        }

        let swarmID = contentAddressedID(files: files, hashes: hashes, chunkSize: chunkSize)
        let manifest = SwarmManifest(protocolVersion: 1,
                                     swarmID: swarmID,
                                     name: name,
                                     totalBytes: total,
                                     chunkSizeBytes: chunkSize,
                                     chunkHashAlgo: "sha256",
                                     files: files,
                                     chunkHashes: hashes)
        return (manifest, orderedURLs)
    }

    private static func fileSize(of url: URL) throws -> Int64 {
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            return Int64(size)
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs[.size] as? NSNumber)?.int64Value ?? 0
    }

    /// SHA-256 over the canonical description of the swarm's content. Two
    /// independent hosts that share the same bytes derive the same id. Internal
    /// so `SwarmManifest.validate()` can recompute and verify it.
    static func contentAddressedID(files: [SwarmFile], hashes: [String], chunkSize: Int) -> String {
        var hasher = SHA256()
        hasher.update(data: Data("v1:\(chunkSize):".utf8))
        for file in files {
            hasher.update(data: Data("\(file.path):\(file.sizeBytes);".utf8))
        }
        for hash in hashes {
            hasher.update(data: Data(hash.utf8))
        }
        return Hashing.sha256Hex(of: hasher.finalize())
    }
}
