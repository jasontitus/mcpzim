import XCTest
@testable import LocalSwarmEngine

/// Adversarial coverage for the trust boundary: a manifest/wire frame from a
/// nearby peer is untrusted input and must not be able to drive unsafe paths,
/// substitute content, or force pathological allocations.
final class SecurityTests: XCTestCase {

    // A genuine, internally-consistent manifest (correct content-addressed id).
    private func makeValidManifest() throws -> SwarmManifest {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ls-sec-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("a.bin")
        try Data((0..<10_000).map { UInt8($0 % 251) }).write(to: file)
        return try Chunker.buildManifest(name: "a", fileURLs: [file], chunkSize: 4096).manifest
    }

    private func assertBadPath(_ manifest: SwarmManifest, file: StaticString = #file, line: UInt = #line) {
        XCTAssertThrowsError(try manifest.validate(), file: file, line: line) {
            guard case .badPath = $0 as? SwarmManifest.ValidationError else {
                return XCTFail("expected badPath, got \($0)", file: file, line: line)
            }
        }
    }

    func testValidManifestPasses() throws {
        XCTAssertNoThrow(try makeValidManifest().validate())
    }

    func testRejectsPathTraversal() throws {
        var m = try makeValidManifest()
        m.files[0].path = "../../../../etc/evil"
        assertBadPath(m)
    }

    func testRejectsAbsolutePath() throws {
        var m = try makeValidManifest()
        m.files[0].path = "/etc/passwd"
        assertBadPath(m)
    }

    func testRejectsSidecarCollision() throws {
        var m = try makeValidManifest()
        m.files[0].path = ".localswarm-bitfield"
        assertBadPath(m)
    }

    func testRejectsEmptyAndDotPaths() throws {
        for bad in ["", ".", "..", "a/../b", "a//b", "sub/./x"] {
            var m = try makeValidManifest()
            m.files[0].path = bad
            assertBadPath(m)
        }
    }

    func testRejectsForgedSwarmID() throws {
        var m = try makeValidManifest()
        m.swarmID = String(repeating: "0", count: 64)
        XCTAssertThrowsError(try m.validate()) {
            XCTAssertEqual($0 as? SwarmManifest.ValidationError, .swarmIDMismatch)
        }
    }

    func testRejectsTamperedHash() throws {
        var m = try makeValidManifest()
        // Swap a real hash for a non-hex one — fails before the id recompute.
        m.chunkHashes[0] = String(repeating: "z", count: 64)
        XCTAssertThrowsError(try m.validate()) {
            guard case .badHash = $0 as? SwarmManifest.ValidationError else {
                return XCTFail("expected badHash, got \($0)")
            }
        }
    }

    func testRejectsDuplicatePaths() throws {
        let hash = String(repeating: "a", count: 64)
        let m = SwarmManifest(protocolVersion: 1, swarmID: String(repeating: "0", count: 64),
                              name: "dup", totalBytes: 2, chunkSizeBytes: 1, chunkHashAlgo: "sha256",
                              files: [SwarmFile(path: "dup", sizeBytes: 1, startChunkIndex: 0, endChunkIndex: 0),
                                      SwarmFile(path: "dup", sizeBytes: 1, startChunkIndex: 1, endChunkIndex: 1)],
                              chunkHashes: [hash, hash])
        XCTAssertThrowsError(try m.validate()) {
            guard case .duplicatePath = $0 as? SwarmManifest.ValidationError else {
                return XCTFail("expected duplicatePath, got \($0)")
            }
        }
    }

    func testRejectsInconsistentChunkRange() throws {
        var m = try makeValidManifest()
        m.files[0].endChunkIndex += 5 // claim more chunks than the file owns
        XCTAssertThrowsError(try m.validate())
    }

    // MARK: - Wire decode limits

    func testBitfieldDecodeRejectsMismatchedPackedLength() throws {
        var body = ByteWriter()
        body.u8(MessageType.bitfield.rawValue)
        body.string("swarm")
        body.u32(100)            // 100 bits → expects 13 packed bytes
        body.u32(5)              // lie
        body.raw(Data(count: 5))
        XCTAssertThrowsError(try Wire.decode(body: body.data))
    }

    func testBitfieldDecodeRejectsHugeBitCount() throws {
        var body = ByteWriter()
        body.u8(MessageType.bitfield.rawValue)
        body.string("swarm")
        body.u32(UInt32(SwarmManifest.maxChunks + 1)) // would allocate a giant array
        body.u32(0)
        XCTAssertThrowsError(try Wire.decode(body: body.data))
    }

    func testChunkStoreRejectsWrongLengthChunk() throws {
        let m = try makeValidManifest()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ls-sec-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = try ChunkStore.forDownloading(manifest: m, directory: dir)
        // Right index, wrong length → rejected before hashing.
        XCTAssertThrowsError(try store.writeChunk(0, data: Data(count: m.chunkSizeBytes + 1))) {
            guard case .chunkLengthMismatch = $0 as? ChunkStore.StoreError else {
                return XCTFail("expected chunkLengthMismatch, got \($0)")
            }
        }
    }
}
