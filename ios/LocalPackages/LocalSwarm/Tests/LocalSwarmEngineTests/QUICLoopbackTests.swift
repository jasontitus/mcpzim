import XCTest
import Network
@testable import LocalSwarmEngine

/// Verifies, on the real Apple QUIC stack over loopback (no AWDL needed), that
/// the production `PeerSecurity.quicParameters` config works end to end: the
/// TLS 1.3 handshake completes with the embedded self-signed identity, the
/// client pins the certificate, and a bidirectional stream carries a request
/// answered on the same stream (the chunk request/response shape).
///
/// Network.framework's real-world QUIC pattern (and what open-source iOS code
/// uses) is a single `NWConnection` acting as one bidirectional stream — which
/// is exactly the v1 transport. Multiplexed `NWConnectionGroup` streams remain a
/// documented future optimization.
final class QUICLoopbackTests: XCTestCase {
    private var listener: NWListener?
    private var clientConn: NWConnection?
    private var serverConn: NWConnection?

    func testQUICProductionParamsHandshakeAndRoundTrip() throws {
        let queue = DispatchQueue(label: "com.localswarm.quic.test")
        let gotReply = expectation(description: "client received reply over QUIC")

        let listener = try NWListener(using: PeerSecurity.quicParameters(peerToPeer: false))
        self.listener = listener
        listener.newConnectionHandler = { [weak self] conn in
            self?.serverConn = conn
            conn.start(queue: queue)
            conn.receive(minimumIncompleteLength: 1, maximumLength: 64) { data, _, _, _ in
                guard let data, String(decoding: data, as: UTF8.self) == "hello" else { return }
                conn.send(content: Data("world".utf8), completion: .contentProcessed { _ in })
            }
        }
        listener.stateUpdateHandler = { state in
            if case let .failed(error) = state { XCTFail("listener failed: \(error)") }
            guard case .ready = state, let port = listener.port else { return }
            let endpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: port)
            let conn = NWConnection(to: endpoint, using: PeerSecurity.quicParameters(peerToPeer: false))
            self.clientConn = conn
            conn.stateUpdateHandler = { s in
                if case .ready = s {
                    conn.send(content: Data("hello".utf8), completion: .contentProcessed { _ in })
                    conn.receive(minimumIncompleteLength: 1, maximumLength: 64) { data, _, _, _ in
                        if let data, String(decoding: data, as: UTF8.self) == "world" { gotReply.fulfill() }
                    }
                }
                if case let .failed(error) = s { XCTFail("client failed: \(error)") }
            }
            conn.start(queue: queue)
        }
        listener.start(queue: queue)

        wait(for: [gotReply], timeout: 20)
        clientConn?.cancel()
        listener.cancel()
    }
}
