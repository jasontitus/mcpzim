import Foundation

/// One file offered to a swarm, together with the relative path it gets in
/// the manifest. Flat shares use the bare filename; folder shares carry each
/// file's path relative to the shared directory ("styles/f1.json"), which the
/// receiving `ChunkStore` recreates — `Manifest.validate()` has always
/// accepted safe nested relative paths, this is the sending-side handle.
public struct ShareItem: Sendable, Hashable {
    public let url: URL
    public let relativePath: String

    public init(url: URL, relativePath: String? = nil) {
        self.url = url
        self.relativePath = relativePath ?? url.lastPathComponent
    }
}

/// A swarm available to download, aggregated across every nearby peer that
/// advertises the same content (`swarmID`). `peers` are the parallel sources.
public struct DiscoveredSwarm: Identifiable, Hashable, Sendable {
    public let swarmID: String
    public let name: String
    public let totalBytes: Int64
    public let chunkCount: Int
    /// All discovered peers across every transport (a host appears once per
    /// transport it advertises).
    public var peers: [DiscoveredPeer]

    public var id: String { swarmID }

    /// Distinct devices offering this swarm (collapsing transport variants).
    public var sourceCount: Int { Set(peers.map(\.peerID)).count }

    /// Whether this share requires a PIN (any advertising peer flags it locked).
    public var locked: Bool { peers.contains { $0.locked } }

    /// Transports this swarm can be pulled over right now.
    public var availableTransports: Set<Transport> { Set(peers.map(\.transport)) }

    /// Peers reachable over a given transport.
    public func peers(for transport: Transport) -> [DiscoveredPeer] {
        peers.filter { $0.transport == transport }
    }
}

/// Progress while a host is hashing a file into chunks, before it starts
/// advertising (a big file takes a while to prepare).
public struct HostPreparation: Sendable, Identifiable {
    public let id: String
    public let name: String
    public var fraction: Double
}

/// A receive the user has requested but whose manifest is still being fetched,
/// so no transfer row exists yet. Lets the UI show "Connecting…" the instant
/// Receive is tapped (a big swarm's manifest takes a moment to transfer).
public struct PendingReceive: Identifiable, Sendable {
    public let swarmID: String
    public let name: String
    public var id: String { swarmID }
}

/// What a node is currently doing with a given swarm.
public enum SwarmRole: String, Sendable {
    case seeding       // hosting original files
    case downloading   // pulling chunks
    case paused        // download stopped by the user; partial data kept for resume
    case complete      // finished downloading; now re-seeding
}

/// A snapshot of one active transfer for display. Value type so it crosses
/// queues safely and drives SwiftUI directly.
/// The physical link a transfer is actually running over — so the UI can tell
/// the user whether they're on the fast direct path or a slower fallback.
public enum LinkKind: String, Sendable, Equatable {
    case directAWDL   // awdl0 / llw* — Apple's peer-to-peer link (AirDrop-class)
    case wifi         // infrastructure Wi-Fi, via a router or a hotspot
    case wired        // Ethernet
    case cellular
    case other
    case unknown      // not yet connected / can't tell

    /// Lower is better/faster — used to pick the best link across several peers.
    public var rank: Int {
        switch self {
        case .directAWDL: return 0
        case .wired: return 1
        case .wifi: return 2
        case .cellular: return 3
        case .other: return 4
        case .unknown: return 5
        }
    }

    /// True when this is the fastest available path (nothing to gain by switching).
    public var isBest: Bool { self == .directAWDL || self == .wired }

    public var label: String {
        switch self {
        case .directAWDL: return "Direct"
        case .wifi: return "Wi-Fi"
        case .wired: return "Wired"
        case .cellular: return "Cellular"
        case .other: return "Network"
        case .unknown: return "Connecting…"
        }
    }
}

public struct TransferStatus: Identifiable, Sendable, Equatable {
    public let swarmID: String
    public let name: String
    public let totalBytes: Int64
    public var completedBytes: Int64
    public var bytesPerSecond: Double
    public var connectedPeers: Int
    public var role: SwarmRole
    public var link: LinkKind = .unknown

    public var id: String { swarmID }

    public var fractionComplete: Double {
        if role == .complete { return 1 }
        guard totalBytes > 0 else { return 0 }
        return min(1, Double(completedBytes) / Double(totalBytes))
    }

    public var isComplete: Bool { role == .complete }
}
