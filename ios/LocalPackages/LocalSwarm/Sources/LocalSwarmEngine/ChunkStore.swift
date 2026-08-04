import Foundation

/// Thread-safe chunk-level storage backed by real files on disk.
///
/// Two roles:
///   * **Seeding** — wraps the original source files; every chunk is present and
///     served read-only.
///   * **Downloading** — wraps a destination directory; files are pre-allocated
///     to full size (sparse), chunks are written and verified as they arrive,
///     and a persisted bitfield allows a transfer to resume after interruption.
///
/// A completed download keeps its store, so the receiver immediately becomes a
/// seeder for everyone else (BitTorrent-style leech-then-seed).
public final class ChunkStore {
    public enum StoreError: Error {
        case chunkOutOfRange(Int)
        case hashMismatch(Int)
        case missingFile(Int)
        case chunkLengthMismatch(Int)
        case unsafePath(String)
    }

    public let manifest: SwarmManifest
    private let layout: [ChunkInfo]
    private let fileURLs: [URL]            // indexed by manifest file index
    private let bitfieldURL: URL?         // nil when seeding (no persistence needed)
    private let readOnly: Bool

    private var bitfield: [Bool]
    /// The packed form of `bitfield`, maintained bit-by-bit so persisting never
    /// has to re-serialize the whole array.
    private var packedBitfield: Data
    /// False entries remaining in `bitfield` — O(1) completeness checks.
    private var missingCount: Int
    /// Chunks stored since the bitfield was last persisted.
    private var dirtyChunks = 0
    private var lastPersist = DispatchTime.now()
    /// Checkpoint cadence: persist after this many chunks or this much time,
    /// whichever comes first (was: one atomic file rewrite per chunk written).
    private static let persistEveryChunks = 64
    private static let persistIntervalNs: UInt64 = 1_000_000_000
    private var readHandles: [Int: FileHandle] = [:]
    private var writeHandles: [Int: FileHandle] = [:]
    private let lock = NSRecursiveLock()

    private init(manifest: SwarmManifest,
                 fileURLs: [URL],
                 bitfield: [Bool],
                 bitfieldURL: URL?,
                 readOnly: Bool) {
        self.manifest = manifest
        self.layout = manifest.chunkLayout()
        self.fileURLs = fileURLs
        self.bitfield = bitfield
        self.packedBitfield = Self.packBitfield(bitfield)
        self.missingCount = bitfield.lazy.filter { !$0 }.count
        self.bitfieldURL = bitfieldURL
        self.readOnly = readOnly
    }

    deinit {
        flush()
        for handle in readHandles.values { try? handle.close() }
        for handle in writeHandles.values { try? handle.close() }
    }

    // MARK: - Factories

    /// A read-only store over original source files. All chunks present.
    public static func forSeeding(manifest: SwarmManifest, sourceURLs: [URL]) -> ChunkStore {
        ChunkStore(manifest: manifest,
                   fileURLs: sourceURLs,
                   bitfield: [Bool](repeating: true, count: manifest.chunkCount),
                   bitfieldURL: nil,
                   readOnly: true)
    }

    /// A writable store inside `directory`. Only the files containing a selected
    /// chunk are created/pre-allocated (`selecting` empty = all files); a
    /// previously persisted bitfield is loaded so the transfer resumes.
    public static func forDownloading(manifest: SwarmManifest,
                                      directory: URL,
                                      selecting files: [SwarmFile] = []) throws -> ChunkStore {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        // Only create the selected files (nil = all). `fileURLs` is still indexed
        // by manifest file index for chunk→file mapping; unselected entries are
        // never opened because their chunks are never requested.
        let selectedPaths: Set<String>? = files.isEmpty ? nil : Set(files.map(\.path))

        // The caller must have validated the manifest (safe relative paths), but
        // re-check that every resolved destination stays inside `directory` so a
        // path can never escape via "..", a symlink, or a future regression.
        let root = directory.standardizedFileURL.path
        var urls: [URL] = []
        for file in manifest.files {
            let url = directory.appendingPathComponent(file.path)
            let resolved = url.standardizedFileURL.path
            guard resolved == root || resolved.hasPrefix(root + "/") else {
                throw StoreError.unsafePath(file.path)
            }
            urls.append(url)
            guard selectedPaths == nil || selectedPaths!.contains(file.path) else { continue }
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !fm.fileExists(atPath: url.path) {
                fm.createFile(atPath: url.path, contents: nil)
                if file.sizeBytes > 0 {
                    let handle = try FileHandle(forWritingTo: url)
                    try handle.truncate(atOffset: UInt64(file.sizeBytes))
                    try handle.close()
                }
            }
        }

        let bitfieldURL = directory.appendingPathComponent(".localswarm-bitfield")
        let bitfield = loadBitfield(at: bitfieldURL, count: manifest.chunkCount)

        // Persist the manifest alongside the data so a resumed session can
        // reconstruct without the network.
        let manifestURL = directory.appendingPathComponent(".localswarm-manifest.json")
        if let data = try? JSONEncoder().encode(manifest) {
            try? data.write(to: manifestURL)
        }

        return ChunkStore(manifest: manifest,
                          fileURLs: urls,
                          bitfield: bitfield,
                          bitfieldURL: bitfieldURL,
                          readOnly: false)
    }

    /// Bytes already present on disk for `indices`, read from the persisted
    /// bitfield without opening the data files. 0 if nothing is downloaded yet.
    /// Lets a caller size the *remaining* space a (possibly resumed) download
    /// needs, before creating any files.
    public static func persistedBytes(manifest: SwarmManifest, directory: URL, indices: [Int]) -> Int64 {
        let bits = loadBitfield(at: directory.appendingPathComponent(".localswarm-bitfield"),
                                count: manifest.chunkCount)
        let layout = manifest.chunkLayout()
        var total: Int64 = 0
        for index in indices where index >= 0 && index < bits.count && bits[index] {
            total += Int64(layout[index].length)
        }
        return total
    }

    // MARK: - State

    public func hasChunk(_ index: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard index >= 0 && index < bitfield.count else { return false }
        return bitfield[index]
    }

    public func bitfieldSnapshot() -> [Bool] {
        lock.lock(); defer { lock.unlock() }
        return bitfield
    }

    public var completedChunkCount: Int {
        lock.lock(); defer { lock.unlock() }
        return bitfield.count - missingCount
    }

    public var isComplete: Bool {
        lock.lock(); defer { lock.unlock() }
        return missingCount == 0
    }

    /// Bytes held so far, counting only the selected chunk indices.
    public func completedBytes(in indices: [Int]) -> Int64 {
        lock.lock(); defer { lock.unlock() }
        var total: Int64 = 0
        for index in indices where index >= 0 && index < bitfield.count && bitfield[index] {
            total += Int64(layout[index].length)
        }
        return total
    }

    // MARK: - I/O

    public func readChunk(_ index: Int) throws -> Data {
        lock.lock(); defer { lock.unlock() }
        guard index >= 0 && index < layout.count else { throw StoreError.chunkOutOfRange(index) }
        let info = layout[index]
        let handle = try readHandle(for: info.fileIndex)
        try handle.seek(toOffset: UInt64(info.offsetInFile))
        return try handle.readFully(info.length)
    }

    /// Verifies the chunk against the manifest hash, writes it at the correct
    /// offset, marks it present, and persists the bitfield. Returns true if the
    /// chunk was newly stored.
    @discardableResult
    public func writeChunk(_ index: Int, data: Data) throws -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !readOnly else { throw StoreError.missingFile(index) }
        guard index >= 0 && index < layout.count else { throw StoreError.chunkOutOfRange(index) }
        // Reject a wrong-length chunk before paying for SHA-256 (a peer can't
        // make us hash an oversized payload).
        guard data.count == layout[index].length else { throw StoreError.chunkLengthMismatch(index) }
        guard Hashing.sha256Hex(data) == manifest.chunkHashes[index] else {
            throw StoreError.hashMismatch(index)
        }
        if bitfield[index] { return false } // already have it

        let info = layout[index]
        let handle = try writeHandle(for: info.fileIndex)
        try handle.seek(toOffset: UInt64(info.offsetInFile))
        try handle.write(contentsOf: data)
        bitfield[index] = true
        packedBitfield[index / 8] |= UInt8(1 << (7 - (index % 8)))
        missingCount -= 1
        dirtyChunks += 1
        // Checkpoint rather than persist per chunk: an atomic full-file rewrite
        // per 1 MiB written gutted large downloads with ~100k extra
        // serialize+rename cycles. A crash now re-fetches at most the chunks
        // since the last checkpoint (they're on disk but unclaimed — harmless).
        if missingCount == 0 || dirtyChunks >= Self.persistEveryChunks ||
            DispatchTime.now().uptimeNanoseconds - lastPersist.uptimeNanoseconds >= Self.persistIntervalNs {
            persistBitfield()
        }
        return true
    }

    // MARK: - Handles

    private func readHandle(for fileIndex: Int) throws -> FileHandle {
        if let handle = readHandles[fileIndex] { return handle }
        guard fileIndex < fileURLs.count else { throw StoreError.missingFile(fileIndex) }
        let handle = try FileHandle(forReadingFrom: fileURLs[fileIndex])
        readHandles[fileIndex] = handle
        return handle
    }

    private func writeHandle(for fileIndex: Int) throws -> FileHandle {
        if let handle = writeHandles[fileIndex] { return handle }
        guard fileIndex < fileURLs.count else { throw StoreError.missingFile(fileIndex) }
        let handle = try FileHandle(forWritingTo: fileURLs[fileIndex])
        writeHandles[fileIndex] = handle
        return handle
    }

    // MARK: - Bitfield persistence

    /// Persists any unflushed bitfield state. Call at natural boundaries
    /// (session stop, completion); also runs on deinit.
    public func flush() {
        lock.lock(); defer { lock.unlock() }
        guard dirtyChunks > 0 else { return }
        persistBitfield()
    }

    private func persistBitfield() {
        // Failure keeps the dirty count so a later checkpoint or flush()
        // retries; bumping lastPersist regardless rate-limits retries against
        // a failing disk to the time-based cadence rather than every write.
        lastPersist = DispatchTime.now()
        guard let url = bitfieldURL else {
            dirtyChunks = 0
            return
        }
        // Atomic write of the incrementally maintained packed form. The
        // per-swarm directory is keyed by the content-addressed swarmID, so a
        // loaded bitfield can only belong to this exact content. We do not
        // fsync chunk bytes (it would gut throughput); a hard crash can
        // therefore lose the last seconds of unflushed writes while the
        // bitfield claims them — surfaced here so it's observable rather
        // than silent.
        do {
            try packedBitfield.write(to: url, options: .atomic)
            dirtyChunks = 0
        } catch {
            swarmDiag("bitfield persist FAILED (\(url.lastPathComponent)): \(error)")
        }
    }

    private static func packBitfield(_ bits: [Bool]) -> Data {
        var data = Data(count: (bits.count + 7) / 8)
        for (i, bit) in bits.enumerated() where bit {
            data[i / 8] |= UInt8(1 << (7 - (i % 8)))
        }
        return data
    }

    private static func loadBitfield(at url: URL, count: Int) -> [Bool] {
        var bits = [Bool](repeating: false, count: count)
        guard let data = try? Data(contentsOf: url) else { return bits }
        let bytes = [UInt8](data)
        for i in 0..<count {
            let byteIndex = i / 8
            guard byteIndex < bytes.count else { break }
            bits[i] = (bytes[byteIndex] & UInt8(1 << (7 - (i % 8)))) != 0
        }
        return bits
    }
}
