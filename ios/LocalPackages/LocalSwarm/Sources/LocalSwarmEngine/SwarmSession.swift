import Foundation
import Network

/// Per-peer connection state. Mutated only on the session's `netQueue`.
final class PeerSession {
    let id: String
    let connection: PeerConnection
    var remoteBitfield: [Bool]
    var inFlight: Set<Int> = []
    var handshaken = false
    var authed = false   // proved the share PIN (always true for an unprotected share)
    let downRate = RateMeter()
    let upRate = RateMeter()

    /// Target number of outstanding chunk requests (keeps the pipe full). Larger
    /// windows fill higher-RTT paths (e.g. infrastructure Wi-Fi) better.
    let window = 48

    /// Seeder-side backpressure: chunk indices this peer has requested but that we
    /// haven't read + sent yet, and how many reads/sends are in flight right now.
    /// Bounds how much one peer can make us buffer — a peer that requests the
    /// whole file at once must never OOM us (a 133 GB / 127k-chunk request did).
    var serveQueue: [Int] = []
    var servesInFlight = 0
    let serveWindow = 16

    init(id: String, connection: PeerConnection, chunkCount: Int) {
        self.id = id
        self.connection = connection
        self.remoteBitfield = [Bool](repeating: false, count: chunkCount)
    }

    func has(_ index: Int) -> Bool {
        index >= 0 && index < remoteBitfield.count && remoteBitfield[index]
    }

    var availableSlots: Int { max(0, window - inFlight.count) }
}

/// Participation in a single swarm — seeding, downloading, or both at once.
///
/// All mutable state is confined to `netQueue`; every public method must be
/// invoked on that queue (the `SwarmManager` wraps calls accordingly). Disk I/O
/// is offloaded to `ioQueue`; the `ChunkStore` is internally locked so parallel
/// reads are safe.
final class SwarmSession: @unchecked Sendable {
    let manifest: SwarmManifest
    let store: ChunkStore
    let localPeerID: String
    let isDownloading: Bool
    let selectedIndices: [Int]
    let transport: Transport

    private let netQueue: DispatchQueue
    private let ioQueue: DispatchQueue

    private var peers: [String: PeerSession] = [:]
    private var neededOrder: [Int] = []
    private var neededSet: Set<Int> = []
    /// Index of the first `neededOrder` entry that may still be needed; the
    /// prefix before it is verified-and-stored and never rescanned. The cursor
    /// only advances past chunks that left `neededSet` — an in-flight chunk can
    /// return to circulation on peer loss, so it must stay scannable.
    private var neededHead = 0
    private var globalInFlight: Set<Int> = []
    /// Total bytes of chunks requested-but-not-yet-received, across all streams.
    /// Capped (on top of the per-peer window) so a high stream count can't balloon
    /// outstanding requests / transport buffers on a memory-constrained device.
    private var globalInFlightBytes: Int64 = 0
    private static let maxInFlightBytes: Int64 = 256 << 20
    /// Per-chunk byte lengths, indexed by global chunk index. Precomputed once:
    /// `manifest.length(ofChunk:)` is a linear scan over the file list, and the
    /// transfer hot path needs a length per requested/received chunk.
    private let chunkLengths: [Int]
    /// Byte size of the selection, summed once instead of per status snapshot.
    private let totalSelectedBytes: Int64
    /// Running count of selected bytes present in the store, maintained per
    /// stored chunk so a snapshot never rescans the whole bitfield.
    private var completedBytesCache: Int64
    private var advertisers: [Advertiser] = []
    private var advertisingSuspended = false
    private var pendingAdvertisePeerID: String?
    private var snapshotTimer: DispatchSourceTimer?
    private var completed = false

    private let downloadRate = RateMeter()
    private let uploadRate = RateMeter()
    private var startedAt = Date()
    private var lastSampleAt = Date.distantPast
    private var lastEmitAt = Date.distantPast
    private var lastRateLogAt = Date.distantPast

    // Diagnostics counters (netQueue only).
    private var servedOK = 0
    private var serveFailed = 0
    private var receivedOK = 0
    private var requestsSent = 0

    /// When this few chunks remain, request them redundantly from several peers
    /// so one slow source can't stall completion.
    private let endgameThreshold = 16

    /// Emitted on `netQueue`.
    var onSnapshot: ((TransferStatus) -> Void)?
    var onComplete: (() -> Void)?
    var onAdvertiserState: ((Transport, NWListener.State) -> Void)?

    init(manifest: SwarmManifest,
         store: ChunkStore,
         localPeerID: String,
         isDownloading: Bool,
         selectedIndices: [Int],
         transport: Transport,
         netQueue: DispatchQueue,
         ioQueue: DispatchQueue,
         pin: String? = nil) {
        self.manifest = manifest
        self.store = store
        self.localPeerID = localPeerID
        self.isDownloading = isDownloading
        self.selectedIndices = selectedIndices
        self.transport = transport
        self.netQueue = netQueue
        self.ioQueue = ioQueue
        let lengths = manifest.chunkLayout().map(\.length)
        self.chunkLengths = lengths
        self.totalSelectedBytes = selectedIndices.reduce(Int64(0)) {
            $0 + Int64($1 >= 0 && $1 < lengths.count ? lengths[$1] : 0)
        }
        self.completedBytesCache = isDownloading
            ? store.completedBytes(in: selectedIndices)
            : self.totalSelectedBytes
        if let pin = pin, !pin.isEmpty {
            let token = swarmAuthToken(swarmID: manifest.swarmID, pin: pin)
            // Seeder side gates on this; downloader side presents it.
            if isDownloading { self.authTokenToSend = token } else { self.expectedToken = token }
        }
        if isDownloading {
            let remaining = selectedIndices.filter { !store.hasChunk($0) }
            self.neededOrder = remaining
            self.neededSet = Set(remaining)
            self.completed = remaining.isEmpty
        }
    }

    /// Set when seeding a PIN-protected share — a peer must present a matching
    /// auth token before the manifest or any chunk is served. nil = open.
    private var expectedToken: String?
    /// Set when downloading a PIN-protected share — sent right after handshake.
    private var authTokenToSend: String?

    // MARK: - Lifecycle (call on netQueue)

    func start() {
        dispatchPrecondition(condition: .onQueue(netQueue))
        startedAt = Date()
        let timer = DispatchSource.makeTimerSource(queue: netQueue)
        timer.schedule(deadline: .now() + 0.4, repeating: 0.4)
        timer.setEventHandler { [weak self] in self?.emitSnapshot(force: true) }
        timer.resume()
        snapshotTimer = timer
        TransferLogger.shared.record(event: "start", status: currentStatus(), transport: transport, elapsed: 0)
        emitSnapshot(force: true)
    }

    func startAdvertising(peerID: String) {
        dispatchPrecondition(condition: .onQueue(netQueue))
        pendingAdvertisePeerID = peerID
        // Suspended (this node is leeching) or already advertising: do nothing.
        guard !advertisingSuspended, advertisers.isEmpty else { return }
        for transport in Transport.allCases {
            let advertiser = Advertiser(queue: netQueue, transport: transport)
            advertiser.onInboundConnection = { [weak self] connection in
                guard let self = self else { return }
                let pc = PeerConnection(connection: connection, queue: self.netQueue)
                self.attach(pc, id: "in-" + UUID().uuidString)
            }
            advertiser.onStateChange = { [weak self] state in
                self?.onAdvertiserState?(transport, state)
            }
            do {
                try advertiser.start(peerID: peerID, swarm: manifest, locked: expectedToken != nil)
            } catch {
                swarmDiag("advertiser start FAILED (\(transport.rawValue)): \(error)")
            }
            advertisers.append(advertiser)
        }
    }

    /// Stops advertising while remembering the intent. An active AWDL listener on
    /// a device that's also leeching collapses the download's AWDL throughput
    /// (radio role conflict), so every advertiser is suspended for the duration
    /// of any download and resumed afterward.
    func suspendAdvertising() {
        dispatchPrecondition(condition: .onQueue(netQueue))
        let had = !advertisers.isEmpty
        advertisingSuspended = true
        advertisers.forEach { $0.stop() }
        advertisers.removeAll()
        if had { swarmDiag("advertising SUSPENDED for \(manifest.name) (leeching — avoid AWDL role conflict)") }
    }

    func resumeAdvertising() {
        dispatchPrecondition(condition: .onQueue(netQueue))
        guard advertisingSuspended else { return }
        advertisingSuspended = false
        if let peerID = pendingAdvertisePeerID {
            swarmDiag("advertising RESUMED for \(manifest.name)")
            startAdvertising(peerID: peerID)
        }
    }

    /// Recreates advertisers (e.g. after the app was suspended and the system
    /// tore the listeners down). No-op if this session wasn't advertising.
    func restartAdvertising(peerID: String) {
        dispatchPrecondition(condition: .onQueue(netQueue))
        guard !advertisers.isEmpty else { return }
        advertisers.forEach { $0.stop() }
        advertisers.removeAll()
        startAdvertising(peerID: peerID)
    }

    /// Parallel connections opened per source. AWDL appears to cap a single
    /// stream well below the link's aggregate capacity; striping requests over
    /// several connections (pump dedupes via globalInFlight) tests whether the
    /// inbound limit is per-connection or per-radio.
    private static let streamsPerSource = 8

    func startDownload(from sources: [DiscoveredPeer]) {
        dispatchPrecondition(condition: .onQueue(netQueue))
        swarmDiag("download start: \(sources.count) source(s) × \(Self.streamsPerSource) streams, need \(neededSet.count)/\(manifest.chunkCount) chunks over \(transport.rawValue)")
        if sources.isEmpty { swarmDiag("download has NO sources — nothing to connect to") }
        for source in sources {
            // Force the AWDL direct link only for AWDL-capable (Apple) peers; a
            // non-Apple peer connects over infrastructure immediately rather than
            // waiting out an AWDL attempt that can't succeed.
            let direct = source.supportsAWDL
            for slot in 0..<Self.streamsPerSource {
                connectSource(source, forceDirect: direct, slot: slot)
            }
        }
    }

    /// Dials a source for bulk data. The first attempt forces the direct AWDL
    /// link (off the slow router); if it isn't handshaken within a few seconds,
    /// it's torn down and re-dialed over the infrastructure path so a transfer
    /// never gets stuck chasing a direct link that can't form. `slot` distinguishes
    /// the parallel streams to one source.
    private func connectSource(_ source: DiscoveredPeer, forceDirect: Bool, slot: Int) {
        // Distinct id per stream + a fallback suffix so the forced attempt's
        // onClosed can't remove the replacement peer (it keys off the id).
        let base = source.id + "#\(slot)"
        let id = forceDirect ? base : base + "#infra"
        // peer-to-peer (AWDL) on only when forcing the direct link; an infra
        // connection (incl. to a non-AWDL Linux/Android peer) must keep it off.
        let pc = PeerConnection(endpoint: source.endpoint, queue: netQueue, transport: transport,
                                peerToPeer: forceDirect, forceDirect: forceDirect)
        attach(pc, id: id)
        guard forceDirect else { return }
        netQueue.asyncAfter(deadline: .now() + 6) { [weak self] in
            guard let self = self, let peer = self.peers[id], !peer.handshaken else { return }
            swarmDiag("peer \(id): direct link not ready in 6s — falling back to infrastructure")
            self.peers[id] = nil
            peer.connection.cancel()
            self.connectSource(source, forceDirect: false, slot: slot)
        }
    }

    func stop() {
        dispatchPrecondition(condition: .onQueue(netQueue))
        onSnapshot = nil
        snapshotTimer?.cancel()
        snapshotTimer = nil
        advertisers.forEach { $0.stop() }
        advertisers.removeAll()
        for peer in peers.values { peer.connection.cancel() }
        peers.removeAll()
        // The bitfield persists on a checkpoint cadence; flush the tail so an
        // interrupted download resumes from exactly what's on disk.
        store.flush()
    }

    // MARK: - Peer wiring

    private func attach(_ connection: PeerConnection, id: String) {
        guard peers[id] == nil else { connection.cancel(); return }
        let peer = PeerSession(id: id, connection: connection, chunkCount: manifest.chunkCount)
        peers[id] = peer

        let inbound = id.hasPrefix("in-")
        swarmDiag("attach peer \(id) (\(inbound ? "inbound" : "outbound"))")
        connection.onReady = { [weak self, weak peer] in
            guard let self = self, let peer = peer else { return }
            swarmDiag("peer \(peer.id) ready → sending handshake + bitfield")
            peer.connection.send(.handshake(peerID: self.localPeerID, swarmID: self.manifest.swarmID))
            // On a locked share the downloader proves the PIN before anything else.
            if let token = self.authTokenToSend {
                peer.connection.send(.auth(swarmID: self.manifest.swarmID, token: token))
            }
            peer.connection.send(.bitfield(swarmID: self.manifest.swarmID, bits: self.store.bitfieldSnapshot()))
        }
        connection.onMessage = { [weak self, weak peer] message in
            guard let self = self, let peer = peer else { return }
            self.handle(message, from: peer)
        }
        connection.onClosed = { [weak self, weak peer] _ in
            guard let self = self, let peer = peer else { return }
            self.removePeer(peer)
        }
        connection.start()
    }

    private func removePeer(_ peer: PeerSession) {
        guard peers[peer.id] != nil else { return }
        swarmDiag("peer \(peer.id) closed (served \(servedOK) total, \(peer.serveQueue.count) queued, \(peer.servesInFlight) in flight)")
        for index in peer.inFlight { untrackInFlight(index) }
        peers[peer.id] = nil
        emitSnapshot(force: true)
        if isDownloading { pump() }
    }

    // MARK: - Message handling

    private func handle(_ message: Message, from peer: PeerSession) {
        // Every swarm-scoped message must name THIS swarm; a peer that sends a
        // different swarmID is confused or hostile, so drop the connection.
        if let mid = message.swarmID, mid != manifest.swarmID {
            swarmDiag("peer \(peer.id) wrong swarmID \(mid.prefix(8)) — closing")
            peer.connection.cancel()
            return
        }
        // Data-bearing messages are only valid after a handshake.
        let needsHandshake: Bool
        switch message {
        case .bitfield, .have, .chunkRequest, .chunkResponse: needsHandshake = true
        default: needsHandshake = false
        }
        if needsHandshake && !peer.handshaken {
            swarmDiag("peer \(peer.id) sent data before handshake — closing")
            peer.connection.cancel()
            return
        }
        // Gate a protected share: nothing is served until the peer proves the
        // PIN. Constant-time compare so a wrong PIN leaks no timing signal.
        if let expected = expectedToken {
            switch message {
            case let .auth(_, token):
                peer.authed = token.utf8CStringConstantTimeEquals(expected)
                if !peer.authed {
                    swarmDiag("peer \(peer.id) wrong PIN — closing")
                    peer.connection.cancel()
                }
                return
            case .manifestRequest, .chunkRequest, .have, .bitfield:
                if !peer.authed {
                    swarmDiag("peer \(peer.id) sent \(message) before auth on a locked share — closing")
                    peer.connection.cancel()
                    return
                }
            default:
                break
            }
        }
        switch message {
        case .handshake:
            peer.handshaken = true
            swarmDiag("peer \(peer.id) handshaken")
            emitSnapshot(force: true)
            if isDownloading { pump() }
        case .auth:
            break // handled above (or ignored on an open share)
        case .manifestRequest:
            swarmDiag("manifestRequest from \(peer.id) → sending manifest (\(manifest.chunkCount) chunks)")
            peer.connection.send(.manifestResponse(manifest))
        case .manifestResponse:
            break // manifest is already known for an active session
        case let .bitfield(_, bits):
            let have = bits.lazy.filter { $0 }.count
            swarmDiag("peer \(peer.id) bitfield: \(have)/\(bits.count) available (expect \(manifest.chunkCount))")
            if bits.count == manifest.chunkCount { peer.remoteBitfield = bits }
            else { swarmDiag("peer \(peer.id) bitfield IGNORED (count mismatch) — will request nothing") }
            if isDownloading { pump() }
        case let .have(_, index):
            if index >= 0 && index < peer.remoteBitfield.count { peer.remoteBitfield[index] = true }
            if isDownloading { pump() }
        case let .chunkRequest(_, index):
            // Make the first requests visible so an in-app Debug log can tell
            // "requests never arrived" from "arrived but never served".
            if index < 3 {
                swarmDiag("chunkRequest recv idx \(index) from \(peer.id) "
                    + "(chunks=\(manifest.chunkCount), queue=\(peer.serveQueue.count), inFlight=\(peer.servesInFlight))")
            }
            serveChunk(index, to: peer)
        case let .chunkResponse(_, index, data):
            receiveChunk(index, data: data, from: peer)
        }
    }

    // MARK: - Serving (seeder side)

    private func serveChunk(_ index: Int, to peer: PeerSession) {
        // Queue the request and serve at most `serveWindow` at a time, so a peer
        // that asks for the whole file at once only ever makes us read + buffer
        // that many chunks — never the entire file. (A 133 GB / 127k-chunk request
        // with no cap OOM-crashed the seeder.)
        peer.serveQueue.append(index)
        drainServeQueue(peer)
    }

    /// Serve queued chunks up to the per-peer window, advancing as each send is
    /// accepted by Network.framework (itself gated by the connection's flow
    /// control), so at most `serveWindow` chunks are ever buffered per peer.
    private func drainServeQueue(_ peer: PeerSession) {
        while peer.servesInFlight < peer.serveWindow, !peer.serveQueue.isEmpty {
            let index = peer.serveQueue.removeFirst()
            guard store.hasChunk(index) else {
                swarmDiag("serveChunk \(index): NOT in store (can't serve)")
                continue
            }
            peer.servesInFlight += 1
            ioQueue.async { [weak self, weak peer] in
                guard let self = self else { return }
                do {
                    if index < 3 { swarmDiag("serveChunk \(index): reading…") }
                    let data = try self.store.readChunk(index)
                    if index < 3 { swarmDiag("serveChunk \(index): read \(data.count)B — sending") }
                    self.netQueue.async {
                        guard let peer = peer else { return }
                        peer.connection.send(.chunkResponse(swarmID: self.manifest.swarmID,
                                                            chunkIndex: index, data: data)) { [weak self, weak peer] in
                            guard let self = self else { return }
                            self.netQueue.async {
                                guard let peer = peer else { return }
                                peer.servesInFlight -= 1
                                self.drainServeQueue(peer) // send accepted → refill the window
                            }
                        }
                        peer.upRate.record(data.count)
                        self.uploadRate.record(data.count)
                        self.servedOK += 1
                        if self.servedOK <= 3 || self.servedOK % 1024 == 0 {
                            swarmDiag("served \(self.servedOK) chunks (\(self.uploadRate.bytesPerSecond.formattedByteRate))")
                        }
                    }
                } catch {
                    // The silent killer: if reading the source file fails (e.g. a
                    // security-scoped ZIM whose access lapsed), the requester just
                    // never gets the chunk. Make it loud — and free the slot.
                    self.netQueue.async {
                        guard let peer = peer else { return }
                        peer.servesInFlight -= 1
                        self.serveFailed += 1
                        if self.serveFailed <= 5 || self.serveFailed % 256 == 0 {
                            swarmDiag("serveChunk \(index): READ FAILED (\(self.serveFailed) total): \(error)")
                        }
                        self.drainServeQueue(peer)
                    }
                }
            }
        }
    }

    // MARK: - Receiving (leecher side)

    private func receiveChunk(_ index: Int, data: Data, from peer: PeerSession) {
        guard isDownloading else { return }
        ioQueue.async { [weak self] in
            guard let self = self else { return }
            var stored = false
            do {
                stored = try self.store.writeChunk(index, data: data)
            } catch {
                // Distinguish a bad-data peer (hash/length mismatch) from a local
                // disk error — both leave the chunk un-stored, but the cause matters.
                swarmDiag("writeChunk \(index) from \(peer.id) failed: \(error)")
            }
            self.netQueue.async {
                peer.inFlight.remove(index)
                self.untrackInFlight(index)
                if stored {
                    // Count toward the selection only if this chunk was needed —
                    // an unsolicited chunk outside the selection is stored but
                    // must not inflate the progress counter.
                    if self.neededSet.remove(index) != nil {
                        self.completedBytesCache += Int64(data.count)
                    }
                    self.downloadRate.record(data.count)
                    peer.downRate.record(data.count)
                    self.receivedOK += 1
                    let now = Date()
                    if now.timeIntervalSince(self.lastRateLogAt) >= 2.0 {
                        self.lastRateLogAt = now
                        let peers = self.peers.values.filter { $0.handshaken }.count
                        swarmDiag("down \(self.downloadRate.bytesPerSecond.formattedByteRate) · \(self.receivedOK)/\(self.selectedIndices.count) chunks · \(peers) streams · inFlight=\(self.globalInFlight.count)")
                    }
                    // Announce only to peers that lack the chunk. A peer that
                    // already has it (every seeder stream — their bitfield is
                    // full) gains nothing, and the unfiltered broadcast put one
                    // control frame per chunk × peer on the bulk-transfer path.
                    for other in self.peers.values where other !== peer && !other.has(index) {
                        other.connection.send(.have(swarmID: self.manifest.swarmID, chunkIndex: index))
                    }
                }
                self.emitSnapshot()
                if self.neededSet.isEmpty {
                    self.finishDownload()
                } else {
                    self.pump()
                }
            }
        }
    }

    // MARK: - Scheduling

    /// O(1) chunk length (0 for an out-of-range index, like `length(ofChunk:)`).
    private func chunkLength(_ index: Int) -> Int {
        index >= 0 && index < chunkLengths.count ? chunkLengths[index] : 0
    }

    /// Marks a chunk as outstanding (requested) and accounts its bytes.
    private func trackInFlight(_ index: Int) {
        guard globalInFlight.insert(index).inserted else { return }
        globalInFlightBytes += Int64(chunkLength(index))
    }

    /// Clears an outstanding chunk and its byte accounting.
    private func untrackInFlight(_ index: Int) {
        guard globalInFlight.remove(index) != nil else { return }
        globalInFlightBytes -= Int64(chunkLength(index))
    }

    private func pump() {
        guard isDownloading, !completed else { return }
        // Advance the cursor past the completed front once per pump (amortized
        // O(1) per finished chunk — pump used to rescan the whole consumed
        // prefix on every call), then compact when the dead entries dominate.
        while neededHead < neededOrder.count, !neededSet.contains(neededOrder[neededHead]) {
            neededHead += 1
        }
        if neededHead > 2048 || neededOrder.count - neededHead > neededSet.count * 2 {
            neededOrder = neededOrder[neededHead...].filter { neededSet.contains($0) }
            neededHead = 0
        }
        let endgame = neededSet.count <= endgameThreshold

        var requestedThisPump = 0
        let handshakenPeers = peers.values.filter { $0.handshaken }
        for peer in handshakenPeers {
            guard globalInFlightBytes < Self.maxInFlightBytes else { break }
            var slots = peer.availableSlots
            guard slots > 0 else { continue }
            for position in neededHead..<neededOrder.count {
                let index = neededOrder[position]
                guard slots > 0, globalInFlightBytes < Self.maxInFlightBytes else { break }
                guard neededSet.contains(index) else { continue }
                guard peer.has(index), !peer.inFlight.contains(index) else { continue }
                if !endgame && globalInFlight.contains(index) { continue }
                peer.inFlight.insert(index)
                trackInFlight(index)
                peer.connection.send(.chunkRequest(swarmID: manifest.swarmID, chunkIndex: index))
                slots -= 1
                requestedThisPump += 1
            }
        }
        requestsSent += requestedThisPump
        // A pump that requests nothing despite work remaining is the stall
        // signature — log why (no handshaken peers, or none hold what we need).
        if requestedThisPump > 0 {
            if requestsSent <= 100 || requestsSent % 4096 == 0 {
                swarmDiag("pump: requested \(requestedThisPump) (total \(requestsSent), \(neededSet.count) remaining, \(handshakenPeers.count) peer(s))")
            }
        } else if !neededSet.isEmpty {
            let front = neededOrder[neededHead...].prefix(256)
            let withData = handshakenPeers.filter { p in front.contains { p.has($0) } }.count
            swarmDiag("pump: requested 0 — \(handshakenPeers.count) handshaken peer(s), \(withData) hold needed chunks, inFlight=\(globalInFlight.count)")
        }
    }

    private func finishDownload() {
        guard !completed else { return }
        completed = true
        store.flush()
        // Only now start advertising/seeding. Running an AWDL listener while a
        // download's AWDL connection is active causes a severe throughput
        // collapse (radio role conflict), so a leecher stays quiet until done.
        TransferLogger.shared.record(event: "complete", status: currentStatus(),
                                     transport: transport, elapsed: Date().timeIntervalSince(startedAt))
        if advertisers.isEmpty {
            startAdvertising(peerID: localPeerID)
        }
        emitSnapshot(force: true)
        onComplete?()
    }

    // MARK: - Snapshot

    func currentStatus() -> TransferStatus {
        // Cached counters: a snapshot fires ~7×/s during a transfer, and the
        // rescans this used to do (bitfield walk × linear per-chunk length
        // lookups) grew with swarm size on the serial netQueue.
        let completedBytes = completedBytesCache
        let role: SwarmRole = !isDownloading ? .seeding : (completed ? .complete : .downloading)
        let rate = (role == .seeding) ? uploadRate.bytesPerSecond : downloadRate.bytesPerSecond
        let total = totalSelectedBytes
        let active = peers.values.filter { $0.handshaken }
        // Report the best (fastest) link any active peer is on — that's the path
        // carrying the transfer, so the UI can flag whether we're on AWDL or a
        // slower fallback.
        let link = active.map { $0.connection.linkKind }.min(by: { $0.rank < $1.rank }) ?? .unknown
        return TransferStatus(swarmID: manifest.swarmID,
                              name: manifest.name,
                              totalBytes: total,
                              completedBytes: completedBytes,
                              bytesPerSecond: rate,
                              connectedPeers: active.count,
                              role: role,
                              link: link)
    }

    /// Pushes a status snapshot to the UI. `force` bypasses the rate limit for
    /// state-change events (handshake, peer loss, completion, timer heartbeat).
    ///
    /// At high throughput chunks land ~70×/s; emitting a `@Published` update per
    /// chunk floods the main thread and starves SwiftUI's render pass (the view
    /// looks frozen until the transfer stops). Rate-limiting the per-chunk path
    /// to ~5×/s keeps the UI live while a heartbeat timer guarantees freshness.
    private func emitSnapshot(force: Bool = false) {
        let now = Date()
        if !force && now.timeIntervalSince(lastEmitAt) < 0.2 { return }
        lastEmitAt = now
        let status = currentStatus()
        onSnapshot?(status)
        if now.timeIntervalSince(lastSampleAt) >= 1.0, isDownloading || status.bytesPerSecond > 0 {
            lastSampleAt = now
            TransferLogger.shared.record(event: "sample", status: status,
                                         transport: transport, elapsed: now.timeIntervalSince(startedAt))
        }
    }
}
