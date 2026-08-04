import Foundation

/// The single source of truth describing the contents of a swarm: which files
/// it holds, how they are sliced into fixed-size chunks, and the SHA-256 hash
/// of every chunk so a receiver can verify each block independently.
///
/// Chunks never span file boundaries — every file begins on a fresh chunk and
/// its final chunk may be shorter than `chunkSizeBytes`. This keeps file-level
/// download selection trivial (each file owns a contiguous, exclusive range of
/// chunk indices).
public struct SwarmManifest: Codable, Identifiable, Sendable, Hashable {
    public var protocolVersion: Int
    /// Content-addressed identifier (SHA-256 over the canonicalized manifest).
    /// Identical content => identical id, so two peers offering the same files
    /// are recognized as one swarm and can be pulled from in parallel.
    public var swarmID: String
    public var name: String
    public var totalBytes: Int64
    public var chunkSizeBytes: Int
    public var chunkHashAlgo: String
    public var files: [SwarmFile]
    /// Hex-encoded SHA-256 of each chunk, indexed by global chunk index.
    public var chunkHashes: [String]

    public var id: String { swarmID }
    public var chunkCount: Int { chunkHashes.count }

    public init(protocolVersion: Int,
                swarmID: String,
                name: String,
                totalBytes: Int64,
                chunkSizeBytes: Int,
                chunkHashAlgo: String,
                files: [SwarmFile],
                chunkHashes: [String]) {
        self.protocolVersion = protocolVersion
        self.swarmID = swarmID
        self.name = name
        self.totalBytes = totalBytes
        self.chunkSizeBytes = chunkSizeBytes
        self.chunkHashAlgo = chunkHashAlgo
        self.files = files
        self.chunkHashes = chunkHashes
    }

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case swarmID = "swarm_id"
        case name
        case totalBytes = "total_bytes"
        case chunkSizeBytes = "chunk_size_bytes"
        case chunkHashAlgo = "chunk_hash_algo"
        case files
        case chunkHashes = "chunk_hashes"
    }
}

public struct SwarmFile: Codable, Identifiable, Sendable, Hashable {
    /// Relative path within the swarm (e.g. `videos/lecture.mp4`).
    public var path: String
    public var sizeBytes: Int64
    public var startChunkIndex: Int
    public var endChunkIndex: Int

    public var id: String { path }
    /// True when the file is empty and therefore owns no chunks.
    public var isEmpty: Bool { sizeBytes == 0 }
    public var chunkCount: Int { isEmpty ? 0 : (endChunkIndex - startChunkIndex + 1) }

    public init(path: String, sizeBytes: Int64, startChunkIndex: Int, endChunkIndex: Int) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.startChunkIndex = startChunkIndex
        self.endChunkIndex = endChunkIndex
    }

    enum CodingKeys: String, CodingKey {
        case path
        case sizeBytes = "size_bytes"
        case startChunkIndex = "start_chunk_index"
        case endChunkIndex = "end_chunk_index"
    }
}

public extension SwarmManifest {
    enum ValidationError: Error, CustomStringConvertible, Equatable {
        case unsupportedProtocol(Int)
        case unsupportedHashAlgo(String)
        case badChunkSize(Int)
        case nameTooLong
        case tooManyFiles(Int)
        case tooManyChunks(Int)
        case badHash(Int)
        case badPath(String)
        case duplicatePath(String)
        case badFileRange(String)
        case chunkCoverageMismatch
        case totalBytesMismatch
        case swarmIDMismatch

        public var description: String {
            switch self {
            case let .unsupportedProtocol(v): return "unsupported protocol version \(v)"
            case let .unsupportedHashAlgo(a): return "unsupported hash algo \(a)"
            case let .badChunkSize(s): return "invalid chunk size \(s)"
            case .nameTooLong: return "name too long"
            case let .tooManyFiles(n): return "too many files (\(n))"
            case let .tooManyChunks(n): return "too many chunks (\(n))"
            case let .badHash(i): return "invalid chunk hash at \(i)"
            case let .badPath(p): return "unsafe file path \"\(p)\""
            case let .duplicatePath(p): return "duplicate file path \"\(p)\""
            case let .badFileRange(p): return "inconsistent chunk range for \"\(p)\""
            case .chunkCoverageMismatch: return "files do not cover the chunk hashes exactly"
            case .totalBytesMismatch: return "totalBytes does not match file sizes"
            case .swarmIDMismatch: return "swarmID is not the content address of the manifest"
            }
        }
    }

    // Bounds for untrusted manifests received from peers.
    static let maxFiles = 100_000
    static let maxChunks = 1 << 24       // ~16.7M chunks ≈ 16 TiB at 1 MiB
    static let maxChunkSizeBytes = 64 << 20
    static let maxNameBytes = 4096
    static let maxPathBytes = 1024
    private static let sidecarNames: Set<String> = [".localswarm-bitfield", ".localswarm-manifest.json"]

    /// Validates an untrusted manifest. On success every `file.path` is a safe
    /// relative path, ranges/sizes/hashes are internally consistent, and
    /// `swarmID` is the true content address — so it can be joined to a
    /// destination directory and trusted by the rest of the engine. Throws
    /// `ValidationError` otherwise. Must be called at every network boundary
    /// before a manifest reaches `ChunkStore`.
    func validate() throws {
        guard protocolVersion == 1 else { throw ValidationError.unsupportedProtocol(protocolVersion) }
        guard chunkHashAlgo == "sha256" else { throw ValidationError.unsupportedHashAlgo(chunkHashAlgo) }
        guard chunkSizeBytes > 0, chunkSizeBytes <= Self.maxChunkSizeBytes else { throw ValidationError.badChunkSize(chunkSizeBytes) }
        guard name.utf8.count <= Self.maxNameBytes else { throw ValidationError.nameTooLong }
        guard files.count <= Self.maxFiles else { throw ValidationError.tooManyFiles(files.count) }
        guard chunkHashes.count <= Self.maxChunks else { throw ValidationError.tooManyChunks(chunkHashes.count) }

        for (i, hash) in chunkHashes.enumerated() where !Self.isLowerHex64(hash) {
            throw ValidationError.badHash(i)
        }

        var seen = Set<String>()
        var nextChunk = 0
        var total: Int64 = 0
        for file in files {
            try Self.validatePath(file.path)
            guard seen.insert(file.path.lowercased()).inserted else { throw ValidationError.duplicatePath(file.path) }
            guard file.sizeBytes >= 0 else { throw ValidationError.badFileRange(file.path) }
            total += file.sizeBytes
            if file.isEmpty { continue }
            let expected = Int((file.sizeBytes + Int64(chunkSizeBytes) - 1) / Int64(chunkSizeBytes))
            guard file.startChunkIndex == nextChunk,
                  file.endChunkIndex == file.startChunkIndex + expected - 1,
                  file.endChunkIndex < chunkHashes.count else {
                throw ValidationError.badFileRange(file.path)
            }
            nextChunk = file.endChunkIndex + 1
        }
        guard nextChunk == chunkHashes.count else { throw ValidationError.chunkCoverageMismatch }
        guard total == totalBytes else { throw ValidationError.totalBytesMismatch }
        guard Chunker.contentAddressedID(files: files, hashes: chunkHashes, chunkSize: chunkSizeBytes) == swarmID else {
            throw ValidationError.swarmIDMismatch
        }
    }

    private static func validatePath(_ path: String) throws {
        guard !path.isEmpty, path.utf8.count <= Self.maxPathBytes, !path.hasPrefix("/"),
              !sidecarNames.contains(path) else {
            throw ValidationError.badPath(path)
        }
        for component in path.split(separator: "/", omittingEmptySubsequences: false) {
            guard !component.isEmpty, component != ".", component != ".." else {
                throw ValidationError.badPath(path)
            }
        }
    }

    private static func isLowerHex64(_ s: String) -> Bool {
        let bytes = s.utf8
        guard bytes.count == 64 else { return false }
        return bytes.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}

/// Resolves a global chunk index to a concrete location on disk.
public struct ChunkInfo: Sendable, Equatable {
    public let index: Int
    public let fileIndex: Int
    public let offsetInFile: Int64
    public let length: Int
}

public extension SwarmManifest {
    /// Ordered layout for every chunk in the swarm. `layout[i].index == i`.
    func chunkLayout() -> [ChunkInfo] {
        var infos: [ChunkInfo] = []
        infos.reserveCapacity(chunkCount)
        for (fileIndex, file) in files.enumerated() where !file.isEmpty {
            var remaining = file.sizeBytes
            var offset: Int64 = 0
            var index = file.startChunkIndex
            while remaining > 0 {
                let length = Int(min(Int64(chunkSizeBytes), remaining))
                infos.append(ChunkInfo(index: index,
                                       fileIndex: fileIndex,
                                       offsetInFile: offset,
                                       length: length))
                remaining -= Int64(length)
                offset += Int64(length)
                index += 1
            }
        }
        return infos
    }

    /// Byte length of a single chunk (the last chunk of a file may be short).
    func length(ofChunk index: Int) -> Int {
        for file in files where !file.isEmpty
            && file.startChunkIndex <= index && index <= file.endChunkIndex {
            let within = Int64(index - file.startChunkIndex)
            let offset = within * Int64(chunkSizeBytes)
            return Int(min(Int64(chunkSizeBytes), file.sizeBytes - offset))
        }
        return 0
    }

    /// The set of global chunk indices covered by the given files.
    func chunkIndices(for selection: [SwarmFile]) -> [Int] {
        var indices: [Int] = []
        for file in selection where !file.isEmpty {
            indices.append(contentsOf: file.startChunkIndex...file.endChunkIndex)
        }
        return indices
    }
}
