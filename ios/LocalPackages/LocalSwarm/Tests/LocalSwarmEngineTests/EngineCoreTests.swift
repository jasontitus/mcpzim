import XCTest
import Foundation
@testable import LocalSwarmEngine

final class EngineCoreTests: XCTestCase {

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("localswarm-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Full pipeline: build a manifest from source files, then transfer every
    /// chunk into a fresh store and confirm the reassembled bytes match exactly.
    func testChunkAndReassembleRoundTrip() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Two files: one spanning several chunks (with a short tail), one tiny.
        let chunkSize = 4096
        let fileA = dir.appendingPathComponent("a.bin")
        let fileB = dir.appendingPathComponent("b.bin")
        let dataA = Data((0..<(chunkSize * 3 + 123)).map { UInt8($0 % 251) })
        let dataB = Data((0..<10).map { UInt8($0) })
        try dataA.write(to: fileA)
        try dataB.write(to: fileB)

        let (manifest, ordered) = try Chunker.buildManifest(
            name: "test", fileURLs: [fileA, fileB], chunkSize: chunkSize)

        XCTAssertEqual(manifest.totalBytes, Int64(dataA.count + dataB.count))
        XCTAssertEqual(manifest.files.count, 2)
        // a.bin -> 4 chunks (3 full + tail), b.bin -> 1 chunk.
        XCTAssertEqual(manifest.chunkCount, 5)
        XCTAssertEqual(manifest.files[0].chunkCount, 4)
        XCTAssertEqual(manifest.files[1].chunkCount, 1)

        let seedStore = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)
        XCTAssertTrue(seedStore.isComplete)

        let destDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: destDir) }
        let recvStore = try ChunkStore.forDownloading(manifest: manifest, directory: destDir)
        XCTAssertEqual(recvStore.completedChunkCount, 0)

        // Transfer chunks out of order to prove offsets are correct.
        for index in manifest.chunkCount.indicesShuffledDeterministically() {
            let chunk = try seedStore.readChunk(index)
            let stored = try recvStore.writeChunk(index, data: chunk)
            XCTAssertTrue(stored)
        }

        XCTAssertTrue(recvStore.isComplete)
        let reassembledA = try Data(contentsOf: destDir.appendingPathComponent("a.bin"))
        let reassembledB = try Data(contentsOf: destDir.appendingPathComponent("b.bin"))
        XCTAssertEqual(reassembledA, dataA)
        XCTAssertEqual(reassembledB, dataB)
    }

    func testWriteRejectsCorruptedChunk() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("c.bin")
        try Data((0..<5000).map { UInt8($0 % 255) }).write(to: file)
        let (manifest, ordered) = try Chunker.buildManifest(name: "c", fileURLs: [file], chunkSize: 4096)
        let seed = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)
        let destDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: destDir) }
        let recv = try ChunkStore.forDownloading(manifest: manifest, directory: destDir)

        var corrupt = try seed.readChunk(0)
        corrupt[corrupt.startIndex] = corrupt[corrupt.startIndex] &+ 1
        XCTAssertThrowsError(try recv.writeChunk(0, data: corrupt))
        XCTAssertFalse(recv.hasChunk(0))
    }

    func testResumeFromPersistedBitfield() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("d.bin")
        try Data((0..<20000).map { UInt8($0 % 200) }).write(to: file)
        let (manifest, ordered) = try Chunker.buildManifest(name: "d", fileURLs: [file], chunkSize: 4096)
        let seed = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)

        let destDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: destDir) }
        do {
            let recv = try ChunkStore.forDownloading(manifest: manifest, directory: destDir)
            try recv.writeChunk(0, data: try seed.readChunk(0))
            try recv.writeChunk(2, data: try seed.readChunk(2))
        }
        // Re-open the same directory; previously written chunks should persist.
        let resumed = try ChunkStore.forDownloading(manifest: manifest, directory: destDir)
        XCTAssertTrue(resumed.hasChunk(0))
        XCTAssertFalse(resumed.hasChunk(1))
        XCTAssertTrue(resumed.hasChunk(2))
        XCTAssertEqual(resumed.completedChunkCount, 2)
    }

    func testPersistedBytesReflectsDownloadedChunks() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("e.bin")
        try Data((0..<20000).map { UInt8($0 % 200) }).write(to: file)
        let (manifest, ordered) = try Chunker.buildManifest(name: "e", fileURLs: [file], chunkSize: 4096)
        let seed = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)

        let destDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: destDir) }
        let all = Array(0..<manifest.chunkCount)

        // Nothing downloaded yet → 0 bytes present, so the space precheck sees
        // the full file as still-needed.
        XCTAssertEqual(ChunkStore.persistedBytes(manifest: manifest, directory: destDir, indices: all), 0)

        let recv = try ChunkStore.forDownloading(manifest: manifest, directory: destDir)
        try recv.writeChunk(0, data: try seed.readChunk(0))
        try recv.writeChunk(1, data: try seed.readChunk(1))

        let expected = Int64(manifest.length(ofChunk: 0) + manifest.length(ofChunk: 1))
        XCTAssertEqual(ChunkStore.persistedBytes(manifest: manifest, directory: destDir, indices: all), expected)
    }

    func testAvailableBytesResolvesEvenForNonexistentSubpath() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        // A path several levels below an existing dir should still resolve the
        // volume by walking up to `dir`.
        let deep = dir.appendingPathComponent("a/b/c", isDirectory: true)
        let free = SwarmManager.availableBytes(forWritingTo: deep)
        XCTAssertNotNil(free)
        XCTAssertGreaterThan(free ?? 0, 0)
    }

    func testLargeManifestFitsFrameCeilingAndRoundTrips() throws {
        // A ~110 GiB ZIM at 1 MiB chunks: the manifest's chunk-hash list alone is
        // several MB. It must still fit one frame, or the file list can't be
        // fetched (regression: 100 GB shares failed with WireError "malformed").
        let oneMiB = 1 << 20
        let totalBytes = Int64(110) * Int64(1 << 30)
        let chunkCount = Int(totalBytes / Int64(oneMiB))
        let hash = String(repeating: "a", count: 64)
        let manifest = SwarmManifest(protocolVersion: 1,
                                     swarmID: String(repeating: "s", count: 64),
                                     name: "wikipedia_en_all_maxi.zim",
                                     totalBytes: totalBytes,
                                     chunkSizeBytes: oneMiB,
                                     chunkHashAlgo: "sha256",
                                     files: [SwarmFile(path: "wiki.zim", sizeBytes: totalBytes,
                                                       startChunkIndex: 0, endChunkIndex: chunkCount - 1)],
                                     chunkHashes: [String](repeating: hash, count: chunkCount))

        let frame = try Wire.encode(.manifestResponse(manifest))
        XCTAssertLessThanOrEqual(frame.count, PeerConnection.maxFrameLength,
                                 "manifest frame (\(frame.count) bytes) exceeds the ceiling — file list can't be fetched")

        // The receiver strips the 4-byte length prefix before decoding.
        let body = frame.subdata(in: (frame.startIndex + 4)..<frame.endIndex)
        guard case let .manifestResponse(decoded) = try Wire.decode(body: body) else {
            return XCTFail("expected manifestResponse")
        }
        XCTAssertEqual(decoded.chunkHashes.count, chunkCount)
        XCTAssertEqual(decoded.totalBytes, totalBytes)
    }

    func testWireRoundTripAllMessages() throws {
        let manifest = SwarmManifest(protocolVersion: 1, swarmID: "abc", name: "demo",
                                     totalBytes: 100, chunkSizeBytes: 50, chunkHashAlgo: "sha256",
                                     files: [SwarmFile(path: "f", sizeBytes: 100, startChunkIndex: 0, endChunkIndex: 1)],
                                     chunkHashes: ["h0", "h1"])
        let messages: [Message] = [
            .handshake(peerID: "peer-1", swarmID: "abc"),
            .manifestRequest(swarmID: "abc"),
            .manifestResponse(manifest),
            .bitfield(swarmID: "abc", bits: [true, false, true, true, false, false, false, true, true]),
            .have(swarmID: "abc", chunkIndex: 7),
            .chunkRequest(swarmID: "abc", chunkIndex: 42),
            .chunkResponse(swarmID: "abc", chunkIndex: 42, data: Data((0..<300).map { UInt8($0 % 256) })),
        ]
        for message in messages {
            let frame = try Wire.encode(message)
            // Strip the 4-byte length prefix the way PeerConnection does.
            let body = frame.subdata(in: (frame.startIndex + 4)..<frame.endIndex)
            let decoded = try Wire.decode(body: body)
            assertEqual(message, decoded)
        }
    }

    private func assertEqual(_ lhs: Message, _ rhs: Message, file: StaticString = #file, line: UInt = #line) {
        switch (lhs, rhs) {
        case let (.handshake(a, b), .handshake(c, d)):
            XCTAssertEqual(a, c, file: file, line: line); XCTAssertEqual(b, d, file: file, line: line)
        case let (.manifestRequest(a), .manifestRequest(b)):
            XCTAssertEqual(a, b, file: file, line: line)
        case let (.manifestResponse(a), .manifestResponse(b)):
            XCTAssertEqual(a, b, file: file, line: line)
        case let (.bitfield(a, x), .bitfield(b, y)):
            XCTAssertEqual(a, b, file: file, line: line); XCTAssertEqual(x, y, file: file, line: line)
        case let (.have(a, x), .have(b, y)):
            XCTAssertEqual(a, b, file: file, line: line); XCTAssertEqual(x, y, file: file, line: line)
        case let (.chunkRequest(a, x), .chunkRequest(b, y)):
            XCTAssertEqual(a, b, file: file, line: line); XCTAssertEqual(x, y, file: file, line: line)
        case let (.chunkResponse(a, x, d1), .chunkResponse(b, y, d2)):
            XCTAssertEqual(a, b, file: file, line: line); XCTAssertEqual(x, y, file: file, line: line)
            XCTAssertEqual(d1, d2, file: file, line: line)
        default:
            XCTFail("Message mismatch", file: file, line: line)
        }
    }
}

private extension Int {
    /// Deterministic non-trivial ordering of 0..<self (no RNG, for repeatable tests).
    func indicesShuffledDeterministically() -> [Int] {
        guard self > 0 else { return [] }
        var indices = Array(0..<self)
        // Reverse halves and interleave — order-independence check without randomness.
        let mid = self / 2
        indices = Array(indices[mid...].reversed()) + Array(indices[..<mid])
        return indices
    }
}
