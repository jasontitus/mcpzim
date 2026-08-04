import Foundation
import Network

/// The public entry point for the framework and the single `ObservableObject`
/// a SwiftUI app binds to. Hosts files, discovers nearby swarms, fetches a
/// manifest for file selection, and drives multi-source downloads — while
/// re-seeding whatever it has.
@MainActor
public final class SwarmManager: ObservableObject {
    @Published public private(set) var localPeerID: String
    @Published public private(set) var transport: Transport
    @Published public private(set) var isDiscovering = false
    @Published public private(set) var discoveredSwarms: [DiscoveredSwarm] = []
    /// Every swarm this node is currently seeding (a node can host many at once).
    @Published public private(set) var hostedManifests: [SwarmManifest] = []
    /// Files currently being hashed into chunks before sharing starts.
    @Published public private(set) var hostPreparations: [HostPreparation] = []
    /// Receives whose manifest is still being fetched (no transfer row yet).
    @Published public private(set) var pendingReceives: [PendingReceive] = []
    @Published public private(set) var transfers: [TransferStatus] = []
    /// Manifests of downloads started this session, kept so completed files can
    /// be resolved back to on-disk URLs for sharing/opening.
    private var downloadManifests: [String: SwarmManifest] = [:]
    @Published public private(set) var lastError: String?

    @Published public private(set) var benchmarkRunning = false
    @Published public private(set) var benchmarkProgress: BenchmarkProgress?
    @Published public private(set) var benchmarkReport: BenchmarkReport?

    /// Human-readable per-transport browse/advertise state, for on-device
    /// diagnosis (the unified log isn't queryable for this app).
    @Published public private(set) var diagnostics: [String] = []

    // Immutable infrastructure, safe to touch from any queue. ioQueue is
    // concurrent at userInitiated QoS so chunk hashing + disk writes keep up with
    // a fast AWDL link — on iOS a lower QoS (e.g. .utility) gets throttled hard
    // and collapses receive throughput to Wi-Fi speeds. The UI stays responsive
    // because the transfer views render outside the List and refresh on a timer,
    // not because the transfer is deprioritized.
    nonisolated let netQueue = DispatchQueue(label: "com.localswarm.net", qos: .userInitiated)
    nonisolated let ioQueue = DispatchQueue(label: "com.localswarm.io", qos: .userInitiated, attributes: .concurrent)

    private var browsers: [Browser] = []
    private var discoveredByTransport: [Transport: [DiscoveredPeer]] = [:]
    private var browserState = "—"
    private var advertiserStateByTransport: [Transport: String] = [:]
    private var hostSessions: [String: SwarmSession] = [:]
    private var downloadSessions: [String: SwarmSession] = [:]
    private var statusByID: [String: TransferStatus] = [:]

    /// Enough to recreate (resume) a download after a pause.
    private struct DownloadParams {
        let manifest: SwarmManifest
        let files: [SwarmFile]
        var swarm: DiscoveredSwarm
    }
    private var downloadParams: [String: DownloadParams] = [:]

    /// swarmIDs whose late session snapshots must be ignored (paused/canceled),
    /// so a user action isn't immediately undone by an in-flight status update.
    private var suppressedSwarmIDs: Set<String> = []

    /// swarmIDs currently downloading. While non-empty, all advertising is
    /// suspended app-wide: an AWDL listener active during a download collapses
    /// the download's throughput (radio role conflict).
    private var activeDownloads: Set<String> = []

    private func updateAdvertisingForDownloads() {
        let suspend = !activeDownloads.isEmpty
        let sessions = Array(hostSessions.values) + Array(downloadSessions.values)
        netQueue.async {
            for session in sessions {
                if suspend { session.suspendAdvertising() } else { session.resumeAdvertising() }
            }
        }
    }

    /// Base directory downloads are written under (one subfolder per swarm). A
    /// host app (e.g. Kiwix) can point this at its own library folder so received
    /// files land where it expects, avoiding a large copy afterward.
    nonisolated let downloadBase: URL

    /// Called on the main queue when a download finishes, with the swarm id and
    /// the on-disk URLs of the received files (ready to import/move).
    public var onDownloadComplete: ((_ swarmID: String, _ fileURLs: [URL]) -> Void)?

    public init(transport: Transport = .tcp, downloadDirectory: URL? = nil) {
        self.transport = transport
        self.localPeerID = "Peer_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(6))
        let fallback = ((try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                      appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory)
            .appendingPathComponent("LocalSwarm", isDirectory: true)
        self.downloadBase = downloadDirectory ?? fallback
    }

    /// The on-disk folder for a swarm's download (base/swarmID).
    nonisolated func directory(for manifest: SwarmManifest) -> URL {
        downloadBase.appendingPathComponent(manifest.swarmID, isDirectory: true)
    }

    /// Destination URLs of a finished download's files — only those that exist on
    /// disk (a partial download creates just the selected files).
    /// Downloaded-file URLs for a swarm we've fetched this session (empty if the
    /// swarm is unknown or nothing landed yet). Used by the UI's share/export.
    public func completedFileURLs(swarmID: String) -> [URL] {
        guard let manifest = downloadManifests[swarmID] else { return [] }
        return completedFileURLs(for: manifest)
    }

    public nonisolated func completedFileURLs(for manifest: SwarmManifest) -> [URL] {
        let dir = directory(for: manifest)
        let fm = FileManager.default
        return manifest.files
            .map { dir.appendingPathComponent($0.path) }
            .filter { fm.fileExists(atPath: $0.path) }
    }

    /// Free space on the volume that will hold downloads, accounting for
    /// purgeable space the OS can reclaim ("important usage"). Walks up to the
    /// nearest existing directory so the volume can be resolved before the
    /// per-swarm folder exists. nil if it can't be determined.
    nonisolated static func availableBytes(forWritingTo url: URL) -> Int64? {
        let fm = FileManager.default
        var probe = url
        while !fm.fileExists(atPath: probe.path) {
            let parent = probe.deletingLastPathComponent()
            if parent.path == probe.path { break }
            probe = parent
        }
        let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    nonisolated static func byteString(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    /// The transport new downloads prefer. Discovery and hosting use *all*
    /// transports regardless, so changing this never disrupts an active session —
    /// it only affects which transport the next download dials.
    public func setTransport(_ newTransport: Transport) {
        transport = newTransport
    }

    /// Dismisses the current error (e.g. after the UI has shown it).
    public func clearError() {
        lastError = nil
    }

    // MARK: - Discovery

    public func startDiscovery() {
        guard !isDiscovering else { return }
        isDiscovering = true
        // Warm the interface tracker now so the first download can already steer
        // onto the direct AWDL link (it needs a path update to know the infra
        // interfaces to prohibit).
        InterfaceTracker.shared.start()
        let myID = localPeerID
        let browser = Browser(queue: netQueue)
        browser.onResultsChanged = { [weak self] peers in
            DispatchQueue.main.async { self?.updateDiscovered(peers, excluding: myID) }
        }
        browser.onStateChanged = { [weak self] state in
            DispatchQueue.main.async {
                self?.browserState = Self.describe(state)
                self?.rebuildDiagnostics()
            }
        }
        browser.start()
        browsers.append(browser)
    }

    private func rebuildDiagnostics() {
        var lines: [String] = ["Browse: \(browserState)"]
        for transport in Transport.allCases {
            lines.append("  \(transport.displayName) peers: \(discoveredByTransport[transport]?.count ?? 0)")
        }
        if !hostSessions.isEmpty {
            for transport in Transport.allCases {
                lines.append("Advertise \(transport.displayName): \(advertiserStateByTransport[transport] ?? "—")")
            }
        }
        diagnostics = lines
    }

    private static func describe(_ state: NWBrowser.State) -> String {
        switch state {
        case .ready: return "ready"
        case .setup: return "setup"
        case .cancelled: return "cancelled"
        case let .failed(error): return "failed: \(error)"
        case let .waiting(error): return "waiting: \(error)"
        @unknown default: return "?"
        }
    }

    private static func describe(_ state: NWListener.State) -> String {
        switch state {
        case .ready: return "ready"
        case .setup: return "setup"
        case .cancelled: return "cancelled"
        case let .failed(error): return "failed: \(error)"
        case let .waiting(error): return "waiting: \(error)"
        @unknown default: return "?"
        }
    }

    public func stopDiscovery() {
        browsers.forEach { $0.stop() }
        browsers.removeAll()
        discoveredByTransport.removeAll()
        isDiscovering = false
        discoveredSwarms = []
    }

    /// Rebuilds discovery and re-advertises hosting. Call when the app returns to
    /// the foreground — iOS tears down NWBrowser/NWListener while suspended, so
    /// after a sleep/lock nothing is discoverable until they're recreated.
    public func refreshConnectivity() {
        if isDiscovering {
            stopDiscovery()
            startDiscovery()
        }
        if !hostSessions.isEmpty {
            let peerID = localPeerID
            let sessions = Array(hostSessions.values)
            netQueue.async { sessions.forEach { $0.restartAdvertising(peerID: peerID) } }
        }
    }

    private func updateDiscovered(_ peers: [DiscoveredPeer], excluding myID: String) {
        let mine = peers.filter { $0.peerID != myID }
        var byTransport: [Transport: [DiscoveredPeer]] = [:]
        for peer in mine { byTransport[peer.transport, default: []].append(peer) }
        discoveredByTransport = byTransport

        var groups: [String: DiscoveredSwarm] = [:]
        for peer in mine {
            if var group = groups[peer.swarmID] {
                group.peers.append(peer)
                groups[peer.swarmID] = group
            } else {
                groups[peer.swarmID] = DiscoveredSwarm(swarmID: peer.swarmID,
                                                       name: peer.name,
                                                       totalBytes: peer.totalBytes,
                                                       chunkCount: peer.chunkCount,
                                                       peers: [peer])
            }
        }
        discoveredSwarms = groups.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        rebuildDiagnostics()
    }

    // MARK: - Hosting

    /// Slices the chosen files into a new swarm and begins seeding it alongside
    /// any others already hosted. A *directory* URL is shared with its internal
    /// layout preserved (each file gets `<dirname>/<subpath>` as its manifest
    /// path, which the receiving side recreates); plain files keep their bare
    /// filename. Hashing runs off the main thread; `completion` fires on the
    /// main thread with the manifest once seeding starts (so a caller can
    /// record the content-addressed swarmID).
    public func hostFiles(at urls: [URL], name: String? = nil, pin: String? = nil, completion: ((SwarmManifest) -> Void)? = nil) {
        let displayName = name ?? Self.defaultName(for: urls)
        let peerID = localPeerID
        let transport = self.transport
        let prepID = UUID().uuidString
        hostPreparations.append(HostPreparation(id: prepID, name: displayName, fraction: 0))
        ioQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let items = Self.expandForSharing(urls)
                guard !items.isEmpty else { throw Chunker.ChunkerError.noFiles }
                let manifest: SwarmManifest
                let ordered: [URL]
                if let cached = ManifestCache.lookup(name: displayName, items: items) {
                    // Unchanged files (same paths, sizes, mtimes) — skip hashing
                    // entirely so a relaunch re-shares a 100 GB file instantly.
                    (manifest, ordered) = cached
                    swarmDiag("manifest cache HIT for \(displayName) (\(manifest.chunkCount) chunks) — skipping hash")
                } else {
                    (manifest, ordered) = try Chunker.buildManifest(name: displayName, items: items,
                        progress: { [weak self] fraction in
                            DispatchQueue.main.async { self?.updatePreparation(prepID, fraction: fraction) }
                        })
                    ManifestCache.store(manifest: manifest, ordered: ordered, name: displayName, items: items)
                }
                let store = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)
                self.netQueue.async {
                    let session = SwarmSession(manifest: manifest,
                                               store: store,
                                               localPeerID: peerID,
                                               isDownloading: false,
                                               selectedIndices: Array(0..<manifest.chunkCount),
                                               transport: transport,
                                               netQueue: self.netQueue,
                                               ioQueue: self.ioQueue,
                                               pin: pin)
                    session.onSnapshot = { [weak self] status in
                        DispatchQueue.main.async { self?.applyStatus(status) }
                    }
                    session.onAdvertiserState = { [weak self] transport, state in
                        DispatchQueue.main.async {
                            self?.advertiserStateByTransport[transport] = Self.describe(state)
                            self?.rebuildDiagnostics()
                        }
                    }
                    session.startAdvertising(peerID: peerID)
                    session.start()
                    DispatchQueue.main.async {
                        let previous = self.hostSessions[manifest.swarmID] // re-sharing same content
                        self.hostSessions[manifest.swarmID] = session
                        self.hostedManifests = self.hostSessions.values.map { $0.manifest }
                        self.hostPreparations.removeAll { $0.id == prepID }
                        self.netQueue.async { previous?.stop() }
                        // If a download is in progress, this new host session must
                        // not advertise either (AWDL role conflict).
                        self.updateAdvertisingForDownloads()
                        completion?(manifest)
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.hostPreparations.removeAll { $0.id == prepID }
                    self.lastError = "Couldn't share files: \(error.localizedDescription)"
                }
            }
        }
    }

    private func updatePreparation(_ id: String, fraction: Double) {
        guard let index = hostPreparations.firstIndex(where: { $0.id == id }) else { return }
        hostPreparations[index].fraction = fraction
    }

    /// Stops seeding one swarm (keeps its files; just stops advertising/serving).
    public func stopHosting(swarmID: String) {
        guard let session = hostSessions.removeValue(forKey: swarmID) else { return }
        hostedManifests = hostSessions.values.map { $0.manifest }
        statusByID[swarmID] = nil
        recomputeTransfers()
        netQueue.async { session.stop() }
    }

    /// Stops seeding every hosted swarm.
    public func stopHosting() {
        let sessions = Array(hostSessions.values)
        for id in hostSessions.keys { statusByID[id] = nil }
        hostSessions = [:]
        hostedManifests = []
        recomputeTransfers()
        netQueue.async { sessions.forEach { $0.stop() } }
    }

    // MARK: - Downloading

    /// One-call receive: shows a "Connecting…" entry immediately, fetches the
    /// manifest, then downloads every file. Use this when the UI doesn't let the
    /// user pick a subset (the manifest fetch for a big swarm is otherwise a
    /// silent wait).
    public func receive(_ swarm: DiscoveredSwarm) {
        guard statusByID[swarm.swarmID] == nil,
              !pendingReceives.contains(where: { $0.swarmID == swarm.swarmID }) else { return }
        pendingReceives.append(PendingReceive(swarmID: swarm.swarmID, name: swarm.name))
        Task { @MainActor in
            let manifest = await fetchManifest(for: swarm)
            if let manifest {
                startDownload(manifest: manifest, selecting: [], from: swarm)
            }
            // Cleared after startDownload has seeded its initial row, so the
            // "Connecting…" entry hands off to the transfer row with no gap.
            pendingReceives.removeAll { $0.swarmID == swarm.swarmID }
        }
    }

    /// Connects to one source and retrieves the full manifest so the user can
    /// pick which files to download.
    public func fetchManifest(for swarm: DiscoveredSwarm, pin: String? = nil) async -> SwarmManifest? {
        // Prefer TCP for the control fetch (most reliable); fall back to any peer.
        guard let source = swarm.peers(for: .tcp).first ?? swarm.peers.first else { return nil }
        do {
            return try await fetchManifest(from: source, transport: source.transport, pin: pin)
        } catch {
            self.lastError = "Couldn't load file list: \(error.localizedDescription)"
            return nil
        }
    }

    /// Begins a multi-source download of the selected files (empty selection =
    /// all files), pulling chunks from every source in the swarm in parallel and
    /// re-seeding as chunks land.
    public func startDownload(manifest: SwarmManifest,
                              selecting files: [SwarmFile],
                              from swarm: DiscoveredSwarm,
                              using preferredTransport: Transport? = nil,
                              pin: String? = nil) {
        let peerID = localPeerID
        // Use the requested transport if the swarm offers it; otherwise fall back
        // to any transport it does offer (so a download never silently no-ops).
        let wanted = preferredTransport ?? self.transport
        let transport = swarm.availableTransports.contains(wanted)
            ? wanted
            : (swarm.availableTransports.first ?? wanted)
        let indices = files.isEmpty
            ? Array(0..<manifest.chunkCount)
            : manifest.chunkIndices(for: files).sorted()
        downloadManifests[manifest.swarmID] = manifest // for completedFileURLs(swarmID:)
        let directory = directory(for: manifest)
        let sources = swarm.peers(for: transport)
        let selectedBytes = files.isEmpty
            ? manifest.totalBytes
            : files.reduce(Int64(0)) { $0 + $1.sizeBytes }

        // Refuse a download that can't fit before creating any files, so a huge
        // ZIM doesn't fill the volume and fail deep into the transfer. Only the
        // bytes still missing count — an interrupted partial is already on disk.
        let alreadyOnDisk = ChunkStore.persistedBytes(manifest: manifest, directory: directory, indices: indices)
        let needed = max(0, selectedBytes - alreadyOnDisk)
        if let free = Self.availableBytes(forWritingTo: downloadBase), free < needed {
            lastError = "Not enough space for “\(manifest.name)”. Needs \(Self.byteString(needed)) more, but only \(Self.byteString(free)) is free."
            return
        }

        suppressedSwarmIDs.remove(manifest.swarmID)
        downloadParams[manifest.swarmID] = DownloadParams(manifest: manifest, files: files, swarm: swarm)
        // Stop advertising app-wide while this download runs (AWDL role conflict).
        activeDownloads.insert(manifest.swarmID)
        updateAdvertisingForDownloads()

        // Show the transfer immediately (role .downloading, 0 bytes) so there's
        // visible feedback before the first network snapshot — and so a Receive's
        // "Connecting…" entry hands off to a real row with no gap. A later
        // snapshot (incl. resumed progress) overwrites this.
        if statusByID[manifest.swarmID] == nil {
            applyStatus(TransferStatus(swarmID: manifest.swarmID,
                                       name: manifest.name,
                                       totalBytes: selectedBytes,
                                       completedBytes: 0,
                                       bytesPerSecond: 0,
                                       connectedPeers: 0,
                                       role: .downloading))
        }

        netQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let store = try ChunkStore.forDownloading(manifest: manifest, directory: directory, selecting: files)
                let session = SwarmSession(manifest: manifest,
                                           store: store,
                                           localPeerID: peerID,
                                           isDownloading: true,
                                           selectedIndices: indices,
                                           transport: transport,
                                           netQueue: self.netQueue,
                                           ioQueue: self.ioQueue,
                                           pin: pin)
                session.onSnapshot = { [weak self] status in
                    DispatchQueue.main.async { self?.applyStatus(status) }
                }
                session.onComplete = { [weak self] in
                    DispatchQueue.main.async {
                        guard let self = self else { return }
                        // Download done → this node can advertise/seed again.
                        self.activeDownloads.remove(manifest.swarmID)
                        self.updateAdvertisingForDownloads()
                        self.onDownloadComplete?(manifest.swarmID, self.completedFileURLs(for: manifest))
                    }
                }
                // Note: a leecher does NOT advertise while downloading — the
                // session begins seeding only once the download completes (an
                // active AWDL listener throttles the AWDL download badly).
                session.startDownload(from: sources)
                session.start()
                DispatchQueue.main.async {
                    self.downloadSessions[manifest.swarmID]?.stop()
                    self.downloadSessions[manifest.swarmID] = session
                }
            } catch {
                DispatchQueue.main.async {
                    // Startup failed after we optimistically suspended advertising
                    // and showed a row — undo both so hosting stays discoverable
                    // and the failed transfer doesn't linger.
                    self.activeDownloads.remove(manifest.swarmID)
                    self.updateAdvertisingForDownloads()
                    self.statusByID[manifest.swarmID] = nil
                    self.recomputeTransfers()
                    self.lastError = "Couldn't start download: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Pauses an active download. Partial data + bitfield stay on disk; the
    /// transfer can be resumed.
    public func pauseDownload(swarmID: String) {
        guard let session = downloadSessions[swarmID] else { return }
        suppressedSwarmIDs.insert(swarmID)
        downloadSessions[swarmID] = nil
        activeDownloads.remove(swarmID)
        updateAdvertisingForDownloads()
        netQueue.async { session.stop() }
        if var status = statusByID[swarmID] {
            status.role = .paused
            status.bytesPerSecond = 0
            status.connectedPeers = 0
            statusByID[swarmID] = status
            recomputeTransfers()
            TransferLogger.shared.record(event: "pause", status: status, transport: transport, elapsed: 0)
        }
    }

    /// Resumes a paused download, reconnecting to whatever sources are currently
    /// nearby and continuing from the persisted bitfield.
    public func resumeDownload(swarmID: String) {
        guard downloadSessions[swarmID] == nil, var params = downloadParams[swarmID] else { return }
        if let live = discoveredSwarms.first(where: { $0.swarmID == swarmID }) {
            params.swarm = live
            downloadParams[swarmID] = params
        }
        guard !params.swarm.peers.isEmpty else {
            lastError = "No source nearby to resume “\(params.manifest.name)”."
            return
        }
        startDownload(manifest: params.manifest, selecting: params.files, from: params.swarm)
    }

    /// Stops a download (active or paused) and deletes its partial data. A
    /// completed/re-seeding transfer is stopped but its finished files are kept.
    public func cancelDownload(swarmID: String) {
        let wasComplete = statusByID[swarmID]?.role == .complete
        if let status = statusByID[swarmID] {
            TransferLogger.shared.record(event: "cancel", status: status, transport: transport, elapsed: 0)
        }
        suppressedSwarmIDs.insert(swarmID)
        let session = downloadSessions[swarmID]
        let manifest = downloadParams[swarmID]?.manifest
        downloadSessions[swarmID] = nil
        downloadParams[swarmID] = nil
        statusByID[swarmID] = nil
        activeDownloads.remove(swarmID)
        updateAdvertisingForDownloads()
        recomputeTransfers()
        netQueue.async { session?.stop() }
        if !wasComplete, let manifest {
            let directory = directory(for: manifest)
            ioQueue.async { try? FileManager.default.removeItem(at: directory) }
        }
    }

    // MARK: - Benchmark

    /// Downloads the selected files over each available transport (TCP, then
    /// QUIC) back-to-back and reports overall / min / max throughput.
    public func runBenchmark(manifest: SwarmManifest, selecting files: [SwarmFile], from swarm: DiscoveredSwarm) {
        guard !benchmarkRunning else { return }
        let indices = files.isEmpty ? Array(0..<manifest.chunkCount) : manifest.chunkIndices(for: files).sorted()
        let total = indices.reduce(Int64(0)) { $0 + Int64(manifest.length(ofChunk: $1)) }
        let order: [Transport] = [.tcp, .quic].filter { swarm.availableTransports.contains($0) }
        guard !order.isEmpty else { lastError = "No sources available to benchmark."; return }

        benchmarkRunning = true
        benchmarkReport = nil
        benchmarkProgress = nil
        let peerID = localPeerID

        Task { [weak self] in
            guard let self else { return }
            var legs: [BenchmarkLeg] = []
            for transport in order {
                let leg = await self.runBenchmarkLeg(transport: transport, manifest: manifest,
                                                     indices: indices, totalBytes: total,
                                                     sources: swarm.peers(for: transport), peerID: peerID)
                legs.append(leg)
            }
            for transport in order {
                try? FileManager.default.removeItem(at: benchmarkDirectory(for: manifest, transport: transport))
            }
            await MainActor.run {
                self.benchmarkReport = BenchmarkReport(name: manifest.name, legs: legs)
                self.benchmarkProgress = nil
                self.benchmarkRunning = false
            }
        }
    }

    private func runBenchmarkLeg(transport: Transport, manifest: SwarmManifest, indices: [Int],
                                 totalBytes: Int64, sources: [DiscoveredPeer], peerID: String) async -> BenchmarkLeg {
        let directory = benchmarkDirectory(for: manifest, transport: transport)
        try? FileManager.default.removeItem(at: directory) // fresh: force a real re-download

        return await withCheckedContinuation { continuation in
            netQueue.async {
                guard !sources.isEmpty,
                      let store = try? ChunkStore.forDownloading(manifest: manifest, directory: directory) else {
                    continuation.resume(returning: BenchmarkLeg(
                        transport: transport, totalBytes: totalBytes, durationSeconds: 0,
                        averageBytesPerSecond: 0, minBytesPerSecond: 0, maxBytesPerSecond: 0, succeeded: false))
                    return
                }
                let session = SwarmSession(manifest: manifest, store: store, localPeerID: peerID,
                                           isDownloading: true, selectedIndices: indices,
                                           transport: transport, netQueue: self.netQueue, ioQueue: self.ioQueue)
                let started = Date()
                var progress: [(t: Double, bytes: Int64)] = [(0, 0)]
                var finished = false
                func finish(_ succeeded: Bool) {
                    guard !finished else { return }
                    finished = true
                    let duration = Date().timeIntervalSince(started)
                    let avg = (succeeded && duration > 0) ? Double(totalBytes) / duration : 0
                    // Min/max are the slowest/fastest *sustained* 1-second windows
                    // (warmup + partial tail excluded) — the slow window is where
                    // bad conditions show up. Falls back to the average for very
                    // short transfers with no full windows.
                    let (lo, hi) = Self.windowedRates(progress: progress, duration: duration)
                    let leg = BenchmarkLeg(transport: transport, totalBytes: totalBytes, durationSeconds: duration,
                                           averageBytesPerSecond: avg,
                                           minBytesPerSecond: lo > 0 ? lo : avg,
                                           maxBytesPerSecond: hi > 0 ? hi : avg,
                                           succeeded: succeeded)
                    session.onComplete = nil
                    session.stop()
                    continuation.resume(returning: leg)
                }
                session.onSnapshot = { status in
                    let elapsed = Date().timeIntervalSince(started)
                    progress.append((elapsed, status.completedBytes))
                    let fraction = totalBytes > 0 ? Double(status.completedBytes) / Double(totalBytes) : 0
                    DispatchQueue.main.async {
                        self.benchmarkProgress = BenchmarkProgress(transport: transport, fractionComplete: fraction, bytesPerSecond: status.bytesPerSecond)
                    }
                }
                session.onComplete = { finish(true) }
                self.netQueue.asyncAfter(deadline: .now() + 180) { finish(false) }
                session.startDownload(from: sources)
                session.start()
            }
        }
    }

    /// Sustained throughput (bytes/sec) over each fixed 1-second window, returning
    /// (min, max). The first window (warmup) and the partial tail window are
    /// excluded so the numbers reflect steady-state transfer, and min ≤ avg ≤ max
    /// holds. `progress` is cumulative (elapsedSeconds, bytesCompleted) samples.
    nonisolated static func windowedRates(progress: [(t: Double, bytes: Int64)], duration: Double) -> (min: Double, max: Double) {
        let fullWindows = Int(duration)
        guard fullWindows >= 2, progress.count >= 2 else { return (0, 0) }
        let sorted = progress.sorted { $0.t < $1.t }
        func bytesAt(_ time: Double) -> Int64 {
            var result: Int64 = 0
            for sample in sorted {
                if sample.t <= time { result = sample.bytes } else { break }
            }
            return result
        }
        var rates: [Double] = []
        for i in 1..<fullWindows { // skip window [0,1) warmup; ignore partial tail
            rates.append(Double(bytesAt(Double(i + 1)) - bytesAt(Double(i))))
        }
        guard let lo = rates.min(), let hi = rates.max() else { return (0, 0) }
        return (lo, hi)
    }

    private nonisolated func benchmarkDirectory(for manifest: SwarmManifest, transport: Transport) -> URL {
        downloadBase.appendingPathComponent("benchmark-\(transport.rawValue)-\(manifest.swarmID)", isDirectory: true)
    }

    // MARK: - Status plumbing

    private func applyStatus(_ status: TransferStatus) {
        guard !suppressedSwarmIDs.contains(status.swarmID) else { return }
        statusByID[status.swarmID] = status
        recomputeTransfers()
    }

    private func recomputeTransfers() {
        transfers = statusByID.values.sorted { lhs, rhs in
            if lhs.role == rhs.role {
                return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
            return lhs.role.sortOrder < rhs.role.sortOrder
        }
    }

    // MARK: - Manifest fetch (nonisolated; runs its own connection)

    enum FetchError: Error { case closed, timeout, swarmIDMismatch }

    nonisolated func fetchManifest(from source: DiscoveredPeer, transport: Transport, pin: String? = nil) async throws -> SwarmManifest {
        try await withCheckedThrowingContinuation { continuation in
            netQueue.async {
                // Peer-to-peer (AWDL) only for AWDL-capable peers; a non-Apple
                // peer is reached over infrastructure (else the dial hangs).
                let connection = PeerConnection(endpoint: source.endpoint, queue: self.netQueue,
                                                transport: transport, peerToPeer: source.supportsAWDL)
                var finished = false
                func finish(_ result: Result<SwarmManifest, Error>) {
                    guard !finished else { return }
                    finished = true
                    connection.cancel()
                    continuation.resume(with: result)
                }
                connection.onReady = {
                    connection.send(.handshake(peerID: "manifest-fetch", swarmID: source.swarmID))
                    if let pin = pin, !pin.isEmpty {
                        connection.send(.auth(swarmID: source.swarmID,
                                              token: swarmAuthToken(swarmID: source.swarmID, pin: pin)))
                    }
                    connection.send(.manifestRequest(swarmID: source.swarmID))
                }
                connection.onMessage = { message in
                    guard case let .manifestResponse(manifest) = message else { return }
                    // Validate the untrusted manifest before it can drive disk
                    // paths / allocation, and require it to be the swarm we asked
                    // for (a peer can't substitute different content).
                    do {
                        try manifest.validate()
                        guard manifest.swarmID == source.swarmID else { throw FetchError.swarmIDMismatch }
                        finish(.success(manifest))
                    } catch {
                        swarmDiag("rejected manifest from \(source.peerID): \(error)")
                        finish(.failure(error))
                    }
                }
                connection.onClosed = { error in finish(.failure(error ?? FetchError.closed)) }
                self.netQueue.asyncAfter(deadline: .now() + 10) { finish(.failure(FetchError.timeout)) }
                connection.start()
            }
        }
    }

    // MARK: - Helpers

    private static func defaultName(for urls: [URL]) -> String {
        if urls.count == 1 { return urls[0].lastPathComponent }
        return "\(urls.count) files"
    }

    /// Turns a user-facing share list into concrete manifest items. Plain
    /// files pass through with their bare filename; a directory is walked
    /// recursively and every regular file inside becomes
    /// `<dirname>/<subpath>` (hidden files skipped). Expanded entries are
    /// sorted by relative path so two hosts sharing identical content derive
    /// the identical content-addressed swarmID.
    nonisolated static func expandForSharing(_ urls: [URL]) -> [ShareItem] {
        let fm = FileManager.default
        var items: [ShareItem] = []
        for url in urls {
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else { continue }
            guard isDirectory.boolValue else {
                items.append(ShareItem(url: url))
                continue
            }
            let root = url.standardizedFileURL
            let prefix = root.lastPathComponent
            guard let enumerator = fm.enumerator(at: root,
                                                 includingPropertiesForKeys: [.isRegularFileKey],
                                                 options: [.skipsHiddenFiles]) else { continue }
            var expanded: [ShareItem] = []
            for case let child as URL in enumerator {
                guard (try? child.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
                else { continue }
                let childPath = child.standardizedFileURL.path
                guard childPath.hasPrefix(root.path + "/") else { continue }
                let subpath = String(childPath.dropFirst(root.path.count + 1))
                expanded.append(ShareItem(url: child, relativePath: "\(prefix)/\(subpath)"))
            }
            items.append(contentsOf: expanded.sorted { $0.relativePath < $1.relativePath })
        }
        return items
    }

}

private extension SwarmRole {
    var sortOrder: Int {
        switch self {
        case .downloading: return 0
        case .paused: return 1
        case .complete: return 2
        case .seeding: return 3
        }
    }
}
