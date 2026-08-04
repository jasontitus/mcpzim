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

    /// Produces a manifest plus the source URLs in manifest-file order (so a
    /// seeding `ChunkStore` can map file index -> URL). Each file's manifest path
    /// is its last path component — the flat, single-or-multi-file case.
    public static func buildManifest(name: String,
                                     fileURLs: [URL],
                                     chunkSize: Int = defaultChunkSize,
                                     progress: (@Sendable (Double) -> Void)? = nil) throws -> (manifest: SwarmManifest, orderedURLs: [URL]) {
        try buildManifest(name: name,
                          sources: fileURLs.map { (url: $0, path: $0.lastPathComponent) },
                          chunkSize: chunkSize, progress: progress)
    }

    /// Builds a manifest from explicit (source URL, manifest relative path) pairs.
    /// Folder shares use this to preserve the directory layout: the relative path
    /// (forward-slash) becomes `SwarmFile.path`, chunking every file independently
    /// exactly as the Go seeder does, so the content-addressed id matches
    /// byte-for-byte across platforms.
    public static func buildManifest(name: String,
                                     sources: [(url: URL, path: String)],
                                     chunkSize: Int = defaultChunkSize,
                                     progress: (@Sendable (Double) -> Void)? = nil) throws -> (manifest: SwarmManifest, orderedURLs: [URL]) {
        guard !sources.isEmpty else { throw ChunkerError.noFiles }

        var files: [SwarmFile] = []
        var hashes: [String] = []
        var orderedURLs: [URL] = []
        var globalIndex = 0
        var total: Int64 = 0

        // Total bytes to hash, so progress can be reported as a fraction.
        let grandTotal = (try? sources.reduce(Int64(0)) { try $0 + fileSize(of: $1.url) }) ?? 0
        var hashed: Int64 = 0

        for source in sources {
            let url = source.url
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
            files.append(SwarmFile(path: source.path,
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

    /// Enumerates every regular file under `root` as (url, relativePath) pairs,
    /// with forward-slash relative paths and sorted by UTF-8 bytes so the derived
    /// swarm id is deterministic and matches the Go seeder's ordering for
    /// identical content. Hidden files are included (as the Go walk does), and
    /// symlinks/directories are skipped. The caller must already hold
    /// security-scoped access to `root`.
    public static func folderSources(_ root: URL) throws -> [(url: URL, path: String)] {
        let fm = FileManager.default
        let base = root.standardizedFileURL
        let basePath = base.path
        guard let enumerator = fm.enumerator(at: base,
                                             includingPropertiesForKeys: [.isRegularFileKey],
                                             options: [],
                                             errorHandler: nil) else {
            throw ChunkerError.noFiles
        }
        var out: [(url: URL, path: String)] = []
        for case let url as URL in enumerator {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { continue }
            let full = url.standardizedFileURL.path
            guard full.hasPrefix(basePath + "/") else { continue } // never escape root
            out.append((url: url, path: String(full.dropFirst(basePath.count + 1))))
        }
        guard !out.isEmpty else { throw ChunkerError.noFiles }
        out.sort { $0.path.utf8.lexicographicallyPrecedes($1.path.utf8) }
        return out
    }

    /// Sources for a *mixed* share — plain files alongside whole directories
    /// (a host app sharing "library files + a couple of model folders" in one
    /// swarm). Files keep their bare filename; each directory is expanded via
    /// `folderSources` with its folder name prepended
    /// ("kokoro_mlx/voices.npz"), so two folders with identical inner layouts
    /// can't collide and a receiver can route each tree by its top-level name.
    ///
    /// Deliberately distinct from the single-folder share, which stays
    /// unprefixed (paths relative to the folder) to remain byte-for-byte
    /// conformant with the Go seeder's folder swarms.
    public static func mixedSources(_ urls: [URL]) throws -> [(url: URL, path: String)] {
        let fm = FileManager.default
        var sources: [(url: URL, path: String)] = []
        for url in urls {
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
                throw ChunkerError.unreadable(url)
            }
            guard isDirectory.boolValue else {
                sources.append((url: url, path: url.lastPathComponent))
                continue
            }
            let prefix = url.standardizedFileURL.lastPathComponent
            for entry in try folderSources(url) {
                sources.append((url: entry.url, path: "\(prefix)/\(entry.path)"))
            }
        }
        return sources
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
