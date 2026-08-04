import Foundation
import Network
import os

/// One connection to a single peer, speaking the framed wire protocol. Frames
/// are read with a length-prefix so messages never interleave, and every
/// callback fires on the connection's dispatch queue.
final class PeerConnection {
    /// Upper bound on a single frame, used as a sanity guard before allocating
    /// the receive buffer (the buffer itself tracks the frame's real size, so a
    /// generous ceiling costs nothing until a frame is actually that big).
    ///
    /// It must hold the largest control payload — the manifest, whose JSON grows
    /// ~70 KB per GiB at 1 MiB chunks (a 100 GiB ZIM ≈ 6.9 MB, well past the old
    /// 5 MiB limit). 64 MiB covers a manifest for ~900 GiB, plus any chunk frame.
    static let maxFrameLength = 64 << 20
    private static let pathLog = Logger(subsystem: "com.localswarm", category: "path")

    let connection: NWConnection
    private let queue: DispatchQueue
    private let forcedDirect: Bool

    var onReady: (() -> Void)?
    var onMessage: ((Message) -> Void)?
    var onClosed: ((Error?) -> Void)?

    private var didClose = false

    /// Outbound: dial a discovered peer over the chosen transport. `forceDirect`
    /// steers a bulk-data connection onto the AWDL link (off the slow router).
    /// `peerToPeer` must be **false** for a non-AWDL (Linux/Android-over-infra)
    /// peer — otherwise Network.framework tries the AWDL path to a Bonjour
    /// endpoint that isn't on AWDL and the connection hangs in `preparing`.
    init(endpoint: NWEndpoint, queue: DispatchQueue, transport: Transport,
         peerToPeer: Bool = true, forceDirect: Bool = false) {
        self.queue = queue
        self.forcedDirect = forceDirect
        self.connection = NWConnection(to: endpoint, using: transport.parameters(peerToPeer: peerToPeer, forceDirect: forceDirect))
    }

    /// Inbound: accept a connection handed up by the listener.
    init(connection: NWConnection, queue: DispatchQueue) {
        self.queue = queue
        self.forcedDirect = false
        self.connection = connection
    }

    /// The physical link this connection is running over. A `forceDirect` dial
    /// prohibits every infrastructure interface, so a live forced connection is
    /// guaranteed to be on AWDL; otherwise we read it from the actual path.
    var linkKind: LinkKind {
        if forcedDirect { return .directAWDL }
        guard let path = connection.currentPath else { return .unknown }
        if path.availableInterfaces.contains(where: { $0.name.hasPrefix("awdl") || $0.name.hasPrefix("llw") }) {
            return .directAWDL
        }
        if path.usesInterfaceType(.wiredEthernet) { return .wired }
        if path.usesInterfaceType(.wifi) { return .wifi }
        if path.usesInterfaceType(.cellular) { return .cellular }
        return .other
    }

    func start() {
        connection.stateUpdateHandler = { [weak self] state in
            guard let self = self else { return }
            switch state {
            case .ready:
                // Log which interface(s) the link uses: "awdl0" = direct AWDL
                // (fast), "en0"/"en1" = infrastructure Wi-Fi via a router (slow).
                if let interfaces = self.connection.currentPath?.availableInterfaces {
                    let names = interfaces.map { "\($0.name)(\($0.type))" }.joined(separator: ",")
                    PeerConnection.pathLog.notice("connection ready over: \(names, privacy: .public)")
                    swarmDiag("connection ready over: \(names)")
                }
                self.onReady?()
                self.receiveNextFrame()
            case let .failed(error):
                swarmDiag("connection FAILED: \(error)")
                self.close(error)
            case let .waiting(error):
                swarmDiag("connection waiting: \(error)")
            case .cancelled:
                self.close(nil)
            default:
                break
            }
        }
        connection.start(queue: queue)
    }

    /// `whenSent` fires (on the connection's queue) once Network.framework has
    /// accepted the content for sending. For a flow-controlled connection that is
    /// gated by the send window, so chaining sends on it provides backpressure.
    func send(_ message: Message, whenSent: (() -> Void)? = nil) {
        guard let data = try? Wire.encode(message) else { whenSent?(); return }
        connection.send(content: data, completion: .contentProcessed { [weak self] error in
            if let error = error { self?.close(error) }
            whenSent?()
        })
    }

    func cancel() {
        connection.cancel()
    }

    private func close(_ error: Error?) {
        guard !didClose else { return }
        didClose = true
        onClosed?(error)
    }

    // MARK: - Framed receive

    private func receiveNextFrame() {
        connection.receive(minimumIncompleteLength: 4, maximumLength: 4) { [weak self] data, _, isComplete, error in
            guard let self = self else { return }
            if let error = error { self.close(error); return }
            guard let data = data, data.count == 4 else {
                if isComplete { self.close(nil) }
                return
            }
            let length = data.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            self.receiveBody(length: Int(length))
        }
    }

    private func receiveBody(length: Int) {
        guard length > 0, length <= PeerConnection.maxFrameLength else {
            close(WireError.malformed)
            cancel()
            return
        }
        connection.receive(minimumIncompleteLength: length, maximumLength: length) { [weak self] data, _, isComplete, error in
            guard let self = self else { return }
            if let error = error { self.close(error); return }
            guard let data = data, data.count == length else {
                if isComplete { self.close(nil) }
                return
            }
            if let message = try? Wire.decode(body: data) {
                self.onMessage?(message)
            }
            self.receiveNextFrame()
        }
    }
}
