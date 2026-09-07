import XCTest
@testable import LocalSwarmEngine

final class ManagerLifecycleTests: XCTestCase {
    private func fixture() throws -> (root: URL, source: URL, manifest: SwarmManifest) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ls-lifecycle-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("fixture.bin")
        try Data(repeating: 0x6b, count: 8193).write(to: source)
        let manifest = try Chunker.buildManifest(name: "fixture", fileURLs: [source], chunkSize: 4096).manifest
        return (root, source, manifest)
    }

    @MainActor
    private func settle(_ manager: SwarmManager) async {
        // Startup crosses net -> main -> net. Drain each hop without blocking
        // the main actor, then verify the state after late callbacks arrive.
        for _ in 0..<4 {
            await withCheckedContinuation { continuation in
                manager.netQueue.async { DispatchQueue.main.async { continuation.resume() } }
            }
        }
    }

    @MainActor
    func testCancelDuringStartupCannotReviveTransferOrFiles() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let manager = SwarmManager(downloadDirectory: f.root.appendingPathComponent("incoming"))
        let swarm = DiscoveredSwarm(swarmID: f.manifest.swarmID, name: "fixture",
                                    totalBytes: f.manifest.totalBytes, chunkCount: f.manifest.chunkCount, peers: [])
        manager.startDownload(manifest: f.manifest, selecting: [], from: swarm)
        XCTAssertEqual(manager.transfers.first?.role, .downloading)
        manager.cancelDownload(swarmID: swarm.swarmID)
        await settle(manager)
        XCTAssertTrue(manager.transfers.isEmpty)
        XCTAssertTrue(manager.completedFileURLs(swarmID: swarm.swarmID).isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: manager.directory(for: f.manifest).path))
    }

    @MainActor
    func testPauseDuringStartupRemainsPaused() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let manager = SwarmManager(downloadDirectory: f.root.appendingPathComponent("incoming"))
        let swarm = DiscoveredSwarm(swarmID: f.manifest.swarmID, name: "fixture",
                                    totalBytes: f.manifest.totalBytes, chunkCount: f.manifest.chunkCount, peers: [])
        manager.startDownload(manifest: f.manifest, selecting: [], from: swarm, pin: "fixture-pin")
        manager.pauseDownload(swarmID: swarm.swarmID)
        XCTAssertEqual(manager.transfers.first?.role, .paused)
        await settle(manager)
        XCTAssertEqual(manager.transfers.first?.role, .paused)
        manager.cancelDownload(swarmID: swarm.swarmID)
        await settle(manager)
    }

    @MainActor
    func testPublicDownloadRejectsUntrustedManifestBeforeWriting() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let manager = SwarmManager(downloadDirectory: f.root.appendingPathComponent("incoming"))
        var malformed = f.manifest
        malformed.files[0].sizeBytes = Int64.max
        let swarm = DiscoveredSwarm(swarmID: malformed.swarmID, name: "fixture",
                                    totalBytes: malformed.totalBytes, chunkCount: malformed.chunkCount, peers: [])
        manager.startDownload(manifest: malformed, selecting: [], from: swarm)
        await settle(manager)
        XCTAssertNotNil(manager.lastError)
        XCTAssertTrue(manager.transfers.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: manager.downloadBase.path))
    }

    func testStopDoesNotWaitForUnrelatedHashingOnSharedIOQueue() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let net = DispatchQueue(label: "test.stop.net")
        let io = DispatchQueue(label: "test.stop.io", attributes: .concurrent)
        let gate = DispatchSemaphore(value: 0)
        io.async { gate.wait() } // another share's long-running hashing task
        defer { gate.signal() }
        let store = ChunkStore.forSeeding(manifest: f.manifest, sourceURLs: [f.source])
        let session = SwarmSession(manifest: f.manifest, store: store, localPeerID: "test",
            isDownloading: false, selectedIndices: [], transport: .quic, netQueue: net, ioQueue: io)
        let stopped = expectation(description: "stop is independent of unrelated I/O")
        net.async { session.stop(); stopped.fulfill() }
        await fulfillment(of: [stopped], timeout: 1)
    }

    @MainActor
    func testStopHostingCancelsPendingPreparation() async throws {
        let f = try fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let manager = SwarmManager()
        // Hold hashing so Stop deterministically wins before host completion.
        let gate = DispatchSemaphore(value: 0)
        manager.ioQueue.async(flags: .barrier) { gate.wait() }
        var didHost = false
        manager.hostFiles(at: [f.source]) { _ in didHost = true }
        manager.stopHosting()
        gate.signal()
        await withCheckedContinuation { continuation in
            manager.ioQueue.async(flags: .barrier) { continuation.resume() }
        }
        await settle(manager)
        XCTAssertFalse(didHost)
        XCTAssertTrue(manager.hostedManifests.isEmpty)
        XCTAssertTrue(manager.hostPreparations.isEmpty)
        XCTAssertTrue(manager.transfers.isEmpty)
        manager.stopHosting()
    }
}
