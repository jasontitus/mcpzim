import Foundation
import Network
import Security
import CryptoKit

/// Builds peer-to-peer transport parameters, gated by a pre-shared key, with
/// `includePeerToPeer` so connections ride AWDL / peer-to-peer Wi-Fi (no router
/// or internet required).
///
/// Two transports are provided:
///   * **TCP** — `tcpParameters()`, TLS 1.2 external PSK (Apple's proven
///     peer-to-peer path).
///   * **QUIC** — `quicParameters()`, TLS 1.3 external PSK (QUIC mandates TLS
///     1.3), enabling multiplexed streams with no head-of-line blocking.
///
/// The PSK gives encryption plus "LocalSwarm apps only" gating. v1 uses a fixed
/// app-wide passcode; a per-swarm pairing code can replace `sharedPasscode`
/// later without touching the rest of the stack.
enum PeerSecurity {
    /// One Bonjour service type for discovery over *every* transport. macOS
    /// local-network privacy only authorizes the service types present when the
    /// user first granted permission, so a newly-added type (e.g. a separate one
    /// for QUIC) fails browsing with -65555 NoAuth. We advertise every transport
    /// under this single already-authorized type and carry the transport in the
    /// TXT record ("tp") instead. The wire protocol is set by NWParameters, not
    /// by this label.
    static let discoveryServiceType = "_localswarm._tcp"
    static let alpnToken = "localswarm-v1"
    static let sharedPasscode = "localswarm-v1"

    /// Per-stream QUIC flow-control window. Generous so a single QUIC stream can
    /// keep the AWDL link saturated (otherwise the window, not the radio, caps a
    /// 1-connection download).
    static let maxStreamData = 16 * 1024 * 1024

    // MARK: - TCP (TLS 1.2 external PSK)

    static func tcpParameters(peerToPeer: Bool = true, forceDirect: Bool = false) -> NWParameters {
        let tls = NWProtocolTLS.Options()
        addSharedPSK(to: tls.securityProtocolOptions)
        // TLS 1.2 PSK suite (0x00A8) for the TCP path.
        sec_protocol_options_append_tls_ciphersuite(
            tls.securityProtocolOptions,
            tls_ciphersuite_t(rawValue: UInt16(TLS_PSK_WITH_AES_128_GCM_SHA256))!)
        let parameters = NWParameters(tls: tls)
        parameters.includePeerToPeer = peerToPeer
        applyDirectLink(to: parameters, forceDirect: forceDirect)
        return parameters
    }

    /// Prohibits the infrastructure interfaces so the connection is forced onto
    /// the direct AWDL link (used for bulk data transfer). No-op if no infra
    /// interface is currently known.
    private static func applyDirectLink(to parameters: NWParameters, forceDirect: Bool) {
        guard forceDirect else { return }
        let infra = InterfaceTracker.shared.infrastructureInterfaces()
        guard !infra.isEmpty else {
            swarmDiag("forceDirect: no infrastructure interfaces known yet — not prohibiting")
            return
        }
        parameters.prohibitedInterfaces = infra
        swarmDiag("forceDirect: prohibiting [\(infra.map { $0.name }.joined(separator: ","))] to force AWDL")
    }

    // MARK: - QUIC (TLS 1.3, shared self-signed identity, cert pinning)

    private static let verifyQueue = DispatchQueue(label: "com.localswarm.tls-verify")

    static func quicParameters(peerToPeer: Bool = true, forceDirect: Bool = false) -> NWParameters {
        let quic = NWProtocolQUIC.Options(alpn: [alpnToken])
        let sec = quic.securityProtocolOptions

        // Present the shared identity (a QUIC listener requires a certificate).
        if let identity = EmbeddedIdentity.secIdentity,
           let secIdentity = sec_identity_create(identity) {
            sec_protocol_options_set_local_identity(sec, secIdentity)
        }
        // Pin the shared cert: accept the peer only if it presents exactly our
        // self-signed certificate. This gives encryption + app gating.
        sec_protocol_options_set_verify_block(sec, { _, trust, complete in
            guard let pinned = EmbeddedIdentity.pinnedCertDER else { complete(false); return }
            let secTrust = sec_trust_copy_ref(trust).takeRetainedValue()
            if let chain = SecTrustCopyCertificateChain(secTrust) as? [SecCertificate],
               let leaf = chain.first {
                complete((SecCertificateCopyData(leaf) as Data) == pinned)
            } else {
                complete(false)
            }
        }, verifyQueue)

        // Headroom for a healthy in-flight request window.
        quic.initialMaxStreamsBidirectional = 256
        quic.initialMaxStreamsUnidirectional = 256
        quic.initialMaxData = 64 * 1024 * 1024
        quic.initialMaxStreamDataBidirectionalLocal = maxStreamData
        quic.initialMaxStreamDataBidirectionalRemote = maxStreamData
        quic.idleTimeout = 30_000 // ms

        let parameters = NWParameters(quic: quic)
        parameters.includePeerToPeer = peerToPeer
        applyDirectLink(to: parameters, forceDirect: forceDirect)
        return parameters
    }

    // MARK: - Shared PSK

    private static func addSharedPSK(to sec: sec_protocol_options_t) {
        let key = SymmetricKey(data: Data(sharedPasscode.utf8))
        let mac = HMAC<SHA256>.authenticationCode(for: Data("LocalSwarm".utf8), using: key)
        let keyData = dispatchData(Data(mac))
        let identityData = dispatchData(Data("localswarm-psk-identity".utf8))
        sec_protocol_options_add_pre_shared_key(sec,
                                                keyData as __DispatchData,
                                                identityData as __DispatchData)
    }

    private static func dispatchData(_ data: Data) -> DispatchData {
        data.withUnsafeBytes { DispatchData(bytes: $0) }
    }
}

/// Selects the wire transport. Both peers must use the same one (each advertises
/// a different Bonjour service type), so it is an app-wide choice.
public enum Transport: String, Sendable, CaseIterable, Identifiable {
    /// TLS 1.2 over TCP, PSK-gated — Apple's proven peer-to-peer path.
    case tcp
    /// QUIC (TLS 1.3, shared self-signed identity) — single stream per peer in
    /// v1. Better loss recovery on lossy AWDL, faster handshake, 0-RTT-capable.
    case quic

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .tcp: return "TCP + TLS"
        case .quic: return "QUIC"
        }
    }

    func parameters(peerToPeer: Bool = true, forceDirect: Bool = false) -> NWParameters {
        switch self {
        case .tcp: return PeerSecurity.tcpParameters(peerToPeer: peerToPeer, forceDirect: forceDirect)
        case .quic: return PeerSecurity.quicParameters(peerToPeer: peerToPeer, forceDirect: forceDirect)
        }
    }
}
