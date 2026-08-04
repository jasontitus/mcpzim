import XCTest
@testable import LocalSwarmEngine

/// Byte-exact conformance vectors for the wire protocol. These pin the framing
/// so a second implementation (Android/Kotlin, Linux/Go, …) can validate its
/// encoder/decoder against the same bytes. The hex strings here are mirrored in
/// PROTOCOL.md and are the source of truth — changing them is a wire break.
///
/// Frame = [UInt32 bodyLength BE][UInt8 type][body…]. Strings = [UInt32 len BE][utf8].
final class ConformanceTests: XCTestCase {

    private func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    private func assertEncodes(_ message: Message, to expected: String,
                               file: StaticString = #file, line: UInt = #line) throws {
        let encoded = try Wire.encode(message)
        XCTAssertEqual(hex(encoded), expected, file: file, line: line)
        // And it must decode back to an equivalent message (round-trip).
        let body = encoded.subdata(in: (encoded.startIndex + 4)..<encoded.endIndex)
        XCTAssertNoThrow(try Wire.decode(body: body), file: file, line: line)
    }

    func testHandshakeVector() throws {
        // type 1; peerID "ab"; swarmID "cd"
        try assertEncodes(.handshake(peerID: "ab", swarmID: "cd"),
                          to: "0000000d01000000026162000000026364")
    }

    func testManifestRequestVector() throws {
        // type 2; swarmID "cd"
        try assertEncodes(.manifestRequest(swarmID: "cd"),
                          to: "0000000702000000026364")
    }

    func testBitfieldVector() throws {
        // type 4; swarmID "cd"; 4 bits [1,0,1,1] -> 1 packed byte 0xB0
        try assertEncodes(.bitfield(swarmID: "cd", bits: [true, false, true, true]),
                          to: "00000010040000000263640000000400000001b0")
    }

    func testHaveVector() throws {
        // type 5; swarmID "cd"; chunkIndex 5
        try assertEncodes(.have(swarmID: "cd", chunkIndex: 5),
                          to: "0000000b0500000002636400000005")
    }

    func testChunkRequestVector() throws {
        // type 6; swarmID "cd"; chunkIndex 5
        try assertEncodes(.chunkRequest(swarmID: "cd", chunkIndex: 5),
                          to: "0000000b0600000002636400000005")
    }

    func testChunkResponseVector() throws {
        // type 7; swarmID "cd"; chunkIndex 5; data 00 01 02
        try assertEncodes(.chunkResponse(swarmID: "cd", chunkIndex: 5, data: Data([0, 1, 2])),
                          to: "00000012070000000263640000000500000003000102")
    }

    /// The content-addressed swarmID must be reproducible from the manifest's
    /// canonical fields alone, so independent implementations (Go/Kotlin) derive
    /// the same id. This pins the exact value for the canonical input below.
    func testContentAddressedIDVector() {
        let files = [SwarmFile(path: "a.bin", sizeBytes: 3, startChunkIndex: 0, endChunkIndex: 0)]
        let hashes = ["0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"]
        let id = Chunker.contentAddressedID(files: files, hashes: hashes, chunkSize: 1 << 20)
        XCTAssertEqual(id, "dca3e34ef976cec3f0c8ee890473181ec194fbaac6ea60a6ed9506fdd5ec6b29")
    }
}
