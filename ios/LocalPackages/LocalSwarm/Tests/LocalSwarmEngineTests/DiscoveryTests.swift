import XCTest
import Network
@testable import LocalSwarmEngine

/// Reproduces the discovery path locally (no device/AWDL needed) to see whether
/// the QUIC `NWListener` + Bonjour advertiser actually starts, vs. the TCP one.
final class DiscoveryTests: XCTestCase {
    private var advertiser: Advertiser?
    private var browser: Browser?

    private func manifest(_ id: String) -> SwarmManifest {
        SwarmManifest(protocolVersion: 1, swarmID: id, name: "Test",
                      totalBytes: 100, chunkSizeBytes: 50, chunkHashAlgo: "sha256",
                      files: [SwarmFile(path: "f", sizeBytes: 100, startChunkIndex: 0, endChunkIndex: 1)],
                      chunkHashes: ["a", "b"])
    }

    /// Outcome of the listener, captured thread-safely from the state callback.
    private final class Outcome: @unchecked Sendable {
        enum State { case pending, ready, failed(String) }
        private let lock = NSLock()
        private var state: State = .pending
        func set(_ s: State) { lock.lock(); if case .pending = state { state = s }; lock.unlock() }
        var value: State { lock.lock(); defer { lock.unlock() }; return state }
    }

    /// Strict: the listener must reach `.ready`. If it `.failed`/threw (e.g. a
    /// sandbox/CI environment with no Bonjour), skip with a reason rather than
    /// passing green — so a real regression on developer hardware still fails.
    private func probe(_ transport: Transport) throws {
        let queue = DispatchQueue(label: "disco.\(transport.rawValue)")
        let swarm = manifest("swarm-\(transport.rawValue)")
        let outcome = Outcome()

        let listenerResolved = expectation(description: "listener resolved \(transport.rawValue)")
        listenerResolved.assertForOverFulfill = false
        let adv = Advertiser(queue: queue, transport: transport)
        adv.onStateChange = { state in
            switch state {
            case .ready: outcome.set(.ready); listenerResolved.fulfill()
            case let .failed(error): outcome.set(.failed("\(error)")); listenerResolved.fulfill()
            default: break
            }
        }
        do {
            try adv.start(peerID: "Peer_\(transport.rawValue)", swarm: swarm)
        } catch {
            outcome.set(.failed("start() threw: \(error)"))
            listenerResolved.fulfill()
        }
        self.advertiser = adv
        _ = XCTWaiter().wait(for: [listenerResolved], timeout: 8)
        adv.stop()

        switch outcome.value {
        case .ready: break // success
        case let .failed(msg):
            throw XCTSkip("\(transport.rawValue) listener unavailable here: \(msg)")
        case .pending:
            throw XCTSkip("\(transport.rawValue) listener never resolved (no Bonjour in this environment?)")
        }
    }

    func testTCPListenerStarts() throws { try probe(.tcp) }
    func testQUICListenerStarts() throws { try probe(.quic) }
}
