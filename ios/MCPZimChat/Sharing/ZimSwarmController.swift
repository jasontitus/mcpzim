// SPDX-License-Identifier: MIT

import Combine
import Foundation
import LocalSwarmEngine
#if canImport(UIKit)
import UIKit
#endif

/// Bridges the LocalSwarm peer-to-peer engine to the Zimfo library.
///
/// Sharing side: seeds the user's loaded ZIMs to nearby devices over
/// AWDL / peer-to-peer Wi-Fi (no router or internet required) so a friend
/// who just installed the app can copy a working library in minutes.
///
/// Receiving side: swarm downloads are staged under
/// `Documents/Incoming/<swarmID>/`; when a transfer completes, the received
/// `.zim` files are moved into the Documents root (same volume — an instant
/// rename), the engine's re-seed session for the moved files is stopped, and
/// the library opens them immediately. Partial downloads therefore never
/// appear in the library scan, and resume-after-interruption is inherited
/// from the engine's persisted chunk bitfields.
@MainActor
final class ZimSwarmController: ObservableObject {
    let manager: SwarmManager

    /// Whether this device is currently seeding its library.
    @Published private(set) var isSharingLibrary = false
    /// One-line outcome of the last completed import ("Added 2 files (7.1 GB)").
    @Published private(set) var lastImportSummary: String?
    /// Files from the last completed transfer that could not be moved out of
    /// staging (rare: disk-level failures). Surfaced so the UI can say so.
    @Published private(set) var lastSkippedCount = 0

    /// Supplies the URLs worth seeding (the enabled library entries).
    var shareableFiles: () -> [URL] = { [] }
    /// Imports freshly received ZIMs into the chat session's library.
    var importFiles: (([URL]) async -> Void)?

    private var browsingViewCount = 0
    private var cancellable: AnyCancellable?
    /// Where swarm downloads are staged (mirrors what the engine was told;
    /// the engine's own `downloadBase` is not public).
    private let stagingBase: URL?

    init() {
        let docs = try? FileManager.default.url(for: .documentDirectory,
                                                in: .userDomainMask,
                                                appropriateFor: nil, create: true)
        let staging = docs?.appendingPathComponent("Incoming", isDirectory: true)
        stagingBase = staging
        manager = SwarmManager(transport: .quic, downloadDirectory: staging)
        manager.onDownloadComplete = { [weak self] swarmID, fileURLs in
            Task { @MainActor in
                await self?.importCompletedSwarm(swarmID: swarmID, fileURLs: fileURLs)
            }
        }
        // Track engine activity for the keep-awake assertion. objectWillChange
        // fires before the mutation lands, so evaluate on the next main-queue
        // hop.
        cancellable = manager.objectWillChange.sink { [weak self] _ in
            DispatchQueue.main.async { self?.updateSleepBlocker() }
        }
    }

    // MARK: - Sharing (seeding the library)

    static var defaultShareName: String {
        #if canImport(UIKit)
        let device = UIDevice.current.name
        #else
        let device = Host.current().localizedName ?? "Mac"
        #endif
        return "Zimfo · \(device)"
    }

    var hasShareableFiles: Bool { !shareableFiles().isEmpty }

    func setSharing(_ enabled: Bool) {
        if enabled {
            let urls = shareableFiles()
            guard !urls.isEmpty else { return }
            isSharingLibrary = true
            // Hosting also implies being discoverable to the friend's browser;
            // keep our own browser running so the Nearby screen stays live.
            manager.startDiscovery()
            manager.hostFiles(at: urls, name: Self.defaultShareName)
        } else {
            isSharingLibrary = false
            manager.stopHosting()
            stopDiscoveryIfIdle()
        }
    }

    /// Re-seeds with the current library contents (called after an import so
    /// a newly received ZIM is immediately shareable onward to a third
    /// device). The engine's manifest cache makes re-hosting unchanged files
    /// instant — only the new file gets hashed.
    private func refreshSharingIfActive() {
        guard isSharingLibrary else { return }
        let urls = shareableFiles()
        guard !urls.isEmpty else {
            setSharing(false)
            return
        }
        manager.stopHosting()
        manager.hostFiles(at: urls, name: Self.defaultShareName)
    }

    // MARK: - Browsing lifecycle (driven by the Nearby screen)

    func beginBrowsing() {
        browsingViewCount += 1
        manager.startDiscovery()
    }

    func endBrowsing() {
        browsingViewCount = max(0, browsingViewCount - 1)
        stopDiscoveryIfIdle()
    }

    private var engineBusy: Bool {
        !manager.hostPreparations.isEmpty
            || !manager.pendingReceives.isEmpty
            || manager.transfers.contains { $0.role == .downloading }
    }

    private func stopDiscoveryIfIdle() {
        // Keep radios up while anything is in flight or the user is sharing
        // or looking at the Nearby screen; otherwise let them wind down.
        guard browsingViewCount == 0, !isSharingLibrary, !engineBusy else { return }
        manager.stopDiscovery()
    }

    /// iOS tears down Bonjour browsers/listeners while the app is suspended;
    /// call on every return to the foreground so nearby devices reappear.
    func refreshConnectivity() {
        guard isSharingLibrary || browsingViewCount > 0 || engineBusy else { return }
        manager.refreshConnectivity()
    }

    // MARK: - Import of completed swarm downloads

    private func importCompletedSwarm(swarmID: String, fileURLs: [URL]) async {
        let fm = FileManager.default
        guard let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                     appropriateFor: nil, create: true) else { return }
        var imported: [URL] = []
        var importedBytes: Int64 = 0
        var skipped = 0

        // Move *everything* out of the per-swarm staging folder (which gets
        // cleaned up below). Only .zim files are imported into the library;
        // anything else a generic LocalSwarm peer sent just lands in
        // Documents, where the library scan ignores it.
        for source in fileURLs {
            let isZim = source.pathExtension.lowercased() == "zim"
            let sourceSize = (try? fm.attributesOfItem(atPath: source.path)[.size] as? Int64)
                .flatMap { $0 } ?? 0
            var destination = docs.appendingPathComponent(source.lastPathComponent)
            if fm.fileExists(atPath: destination.path) {
                let existingSize = (try? fm.attributesOfItem(atPath: destination.path)[.size] as? Int64)
                    .flatMap { $0 } ?? -1
                if existingSize == sourceSize {
                    // Same published archive already in the library — drop the
                    // duplicate copy rather than storing it twice.
                    try? fm.removeItem(at: source)
                    continue
                }
                destination = uniqueDestination(for: source.lastPathComponent, in: docs)
            }
            do {
                try fm.moveItem(at: source, to: destination)
                if isZim {
                    imported.append(destination)
                    importedBytes += sourceSize
                }
            } catch {
                skipped += 1
            }
        }

        // The completed session would keep re-seeding from the staging folder
        // we just emptied — stop it. (Files already moved; a completed
        // transfer's cancel does not delete anything.)
        manager.cancelDownload(swarmID: swarmID)
        cleanupStagingFolder(swarmID: swarmID)

        lastSkippedCount = skipped
        if !imported.isEmpty {
            let size = ByteCountFormatter.string(fromByteCount: importedBytes, countStyle: .file)
            lastImportSummary = imported.count == 1
                ? "Added \(imported[0].lastPathComponent) (\(size))"
                : "Added \(imported.count) files (\(size))"
            await importFiles?(imported)
            refreshSharingIfActive()
        }
        updateSleepBlocker()
    }

    private func uniqueDestination(for filename: String, in directory: URL) -> URL {
        let base = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        var attempt = 2
        while true {
            let candidate = directory.appendingPathComponent("\(base) \(attempt).\(ext)")
            if !FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            attempt += 1
        }
    }

    private func cleanupStagingFolder(swarmID: String) {
        guard let stagingBase else { return }
        let dir = stagingBase.appendingPathComponent(swarmID, isDirectory: true)
        let fm = FileManager.default
        // Remove the per-swarm folder only once nothing useful remains —
        // just the bitfield sidecars — so a move failure never turns into
        // data loss here.
        if let leftovers = try? fm.contentsOfDirectory(atPath: dir.path),
           leftovers.allSatisfy({ $0.hasSuffix(".lsbits") || $0 == ".DS_Store" }) {
            try? fm.removeItem(at: dir)
        }
    }

    private func updateSleepBlocker() {
        SleepBlocker.set(engineBusy, reason: "nearby-share")
    }
}
