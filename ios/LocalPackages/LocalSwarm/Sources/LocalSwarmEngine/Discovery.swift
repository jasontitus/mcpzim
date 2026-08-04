import Foundation
import Network
import os

private let discoveryLog = Logger(subsystem: "com.localswarm", category: "discovery")

/// A swarm source found nearby over Bonjour/AWDL. The same swarm (identical
/// `swarmID`) may be advertised by several peers — each becomes one source to
/// pull chunks from in parallel.
public struct DiscoveredPeer: Identifiable, Hashable, Sendable {
    public let endpoint: NWEndpoint
    public let peerID: String
    public let swarmID: String
    public let name: String
    public let totalBytes: Int64
    public let chunkCount: Int
    /// Which transport this peer was discovered on. A host advertises on every
    /// transport, so the same host appears once per transport it offers.
    public let transport: Transport
    /// Whether this peer is reachable over the AWDL direct link (Apple peers).
    /// A non-Apple peer (Linux/Avahi, Android over infra) advertises without the
    /// flag, so we connect to it over infrastructure instead of waiting out an
    /// AWDL attempt that can never succeed.
    public let supportsAWDL: Bool
    /// The peer's wire protocol version (for cross-implementation compatibility).
    public let protocolVersion: Int
    /// Whether the share requires a PIN (the filename is hidden when locked).
    public let locked: Bool

    public var id: String { "\(peerID)@\(swarmID)@\(transport.rawValue)" }

    init?(result: NWBrowser.Result) {
        guard case let .bonjour(txt) = result.metadata else { return nil }
        let swarmID = txt["sid"] ?? ""
        guard !swarmID.isEmpty, let transport = Transport(rawValue: txt["tp"] ?? "") else { return nil }
        self.endpoint = result.endpoint
        self.swarmID = swarmID
        self.name = txt["name"] ?? "Unknown"
        self.peerID = txt["pid"] ?? UUID().uuidString
        self.totalBytes = Int64(txt["bytes"] ?? "") ?? 0
        self.chunkCount = Int(txt["chunks"] ?? "") ?? 0
        self.transport = transport
        self.supportsAWDL = txt["awdl"] == "1"
        self.protocolVersion = Int(txt["pv"] ?? "") ?? 1
        self.locked = txt["lk"] == "1"
    }
}

/// Advertises the locally hosted swarm and accepts inbound peers.
final class Advertiser {
    private var listener: NWListener?
    private let queue: DispatchQueue
    private let transport: Transport

    var onInboundConnection: ((NWConnection) -> Void)?
    var onStateChange: ((NWListener.State) -> Void)?

    init(queue: DispatchQueue, transport: Transport) {
        self.queue = queue
        self.transport = transport
    }

    func start(peerID: String, swarm: SwarmManifest, locked: Bool = false) throws {
        let listener = try NWListener(using: transport.parameters())

        var txt = NWTXTRecord()
        txt["sid"] = swarm.swarmID
        txt["name"] = locked ? "Protected share" : swarm.name // don't leak a locked filename
        txt["bytes"] = String(swarm.totalBytes)
        if locked { txt["lk"] = "1" }
        txt["chunks"] = String(swarm.chunkCount)
        txt["pid"] = peerID
        txt["tp"] = transport.rawValue // transport carried in TXT, not the service type
        txt["awdl"] = "1"              // this (Apple) peer is reachable over AWDL
        txt["pv"] = String(swarm.protocolVersion) // wire protocol version
        // Unique service instance name per advertisement. The stable identity is
        // the peerID carried in the TXT record ("pid"); using a fresh Bonjour
        // name each time ensures a stop-then-share re-appears on browsers instead
        // of colliding with the just-removed registration.
        let instanceName = "\(peerID)-\(transport.rawValue)-\(String(UUID().uuidString.prefix(4)))"
        listener.service = NWListener.Service(name: instanceName,
                                              type: PeerSecurity.discoveryServiceType,
                                              txtRecord: txt)

        listener.newConnectionHandler = { [weak self] connection in
            self?.onInboundConnection?(connection)
        }
        listener.stateUpdateHandler = { [weak self] state in
            swarmDiag("advertiser(\(self?.transport.rawValue ?? "?")) [\(PeerSecurity.discoveryServiceType)] state: \(state)")
            self?.onStateChange?(state)
        }
        listener.start(queue: queue)
        swarmDiag("advertiser starting \(transport.rawValue) on \(PeerSecurity.discoveryServiceType) as \(instanceName)")
        self.listener = listener
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }
}

/// Browses for nearby swarms over Bonjour/AWDL.
final class Browser {
    private var browser: NWBrowser?
    private let queue: DispatchQueue

    var onResultsChanged: (([DiscoveredPeer]) -> Void)?
    var onStateChanged: ((NWBrowser.State) -> Void)?

    init(queue: DispatchQueue) {
        self.queue = queue
    }

    func start() {
        let parameters = NWParameters()
        parameters.includePeerToPeer = true
        let descriptor = NWBrowser.Descriptor.bonjourWithTXTRecord(
            type: PeerSecurity.discoveryServiceType, domain: nil)
        let browser = NWBrowser(for: descriptor, using: parameters)

        browser.browseResultsChangedHandler = { [weak self] results, _ in
            let peers = results.compactMap { DiscoveredPeer(result: $0) }
            swarmDiag("browser results: \(results.count) raw, \(peers.count) usable [\(peers.map { $0.transport.rawValue }.joined(separator: ","))]")
            self?.onResultsChanged?(peers)
        }
        browser.stateUpdateHandler = { [weak self] state in
            swarmDiag("browser [\(PeerSecurity.discoveryServiceType)] state: \(state)")
            self?.onStateChanged?(state)
        }
        browser.start(queue: queue)
        self.browser = browser
    }

    func stop() {
        browser?.cancel()
        browser = nil
    }
}
