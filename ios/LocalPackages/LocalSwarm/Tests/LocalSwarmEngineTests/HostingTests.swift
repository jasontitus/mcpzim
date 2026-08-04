import XCTest
import Network
@testable import LocalSwarmEngine

/// Exercises the seeding path that runs when a user "chooses a file to share":
/// build a manifest, open a seeding store, and start advertising. This is the
/// flow that was crashing on device, so we reproduce it off-UI here.
final class HostingTests: XCTestCase {
    private func tempFile(bytes: Int) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ls-host-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("sample.bin")
        try Data((0..<bytes).map { UInt8($0 % 256) }).write(to: url)
        return url
    }

    private func runHosting(transport: Transport) throws {
        let file = try tempFile(bytes: 2_500_000) // ~3 chunks
        let (manifest, ordered) = try Chunker.buildManifest(name: "sample", fileURLs: [file])
        let store = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)
        let netQueue = DispatchQueue(label: "test.net")
        let ioQueue = DispatchQueue(label: "test.io", attributes: .concurrent)
        let session = SwarmSession(manifest: manifest,
                                   store: store,
                                   localPeerID: "Peer_test",
                                   isDownloading: false,
                                   selectedIndices: Array(0..<manifest.chunkCount),
                                   transport: transport,
                                   netQueue: netQueue,
                                   ioQueue: ioQueue)
        let snapshot = expectation(description: "snapshot emitted")
        snapshot.assertForOverFulfill = false
        session.onSnapshot = { _ in snapshot.fulfill() }
        netQueue.async {
            session.startAdvertising(peerID: "Peer_test")
            session.start()
        }
        wait(for: [snapshot], timeout: 5)
        netQueue.sync { session.stop() }
    }

    func testHostingOverTCPDoesNotCrash() throws {
        try runHosting(transport: .tcp)
    }

    func testHostingOverQUICDoesNotCrash() throws {
        try runHosting(transport: .quic)
    }
}
