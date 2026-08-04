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
    /// Supplies the selected AI model's shareable file(s) — empty when the
    /// model isn't a single-file GGUF or isn't fully downloaded yet.
    var shareableModelFiles: () -> [URL] = { [] }
    /// Offers a received file to the model providers; true means a provider
    /// claimed it and moved it into its own cache slot.
    var importModelFile: ((URL) async -> Bool)?
    /// Whether the share also seeds the chat model, so the friend can chat
    /// entirely offline. On by default — that's the point of the bootstrap.
    @Published private(set) var includeModelInShare = true
    /// Whether the share also seeds the voice models (Kokoro MLX assets +
    /// Supertonic Core ML bundles), shared as whole directories.
    @Published private(set) var includeVoiceInShare = true

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

    var hasShareableFiles: Bool { !currentShareSet().isEmpty }

    /// Everything the share seeds right now: enabled library ZIMs, the
    /// selected model's GGUF, and the voice-model folders, per the toggles.
    private func currentShareSet() -> [URL] {
        var urls = shareableFiles()
        if includeModelInShare {
            urls.append(contentsOf: shareableModelFiles())
        }
        if includeVoiceInShare {
            urls.append(contentsOf: Self.shareableVoiceDirectories)
        }
        return urls
    }

    /// Voice-model folders worth seeding, shared as directories — the engine
    /// preserves their internal layout, and the receiver routes them back
    /// into `Application Support/models/`.
    nonisolated static var shareableVoiceDirectories: [URL] {
        var directories: [URL] = []
        if KokoroAssets.isDownloaded {
            directories.append(KokoroAssets.modelDirectory)
        }
        #if canImport(FluidAudio)
        if Supertonic3Assets.currentBytesOnDisk > 0 {
            directories.append(Supertonic3Assets.modelDirectory)
        }
        #endif
        return directories
    }

    /// Bytes across every shareable voice asset — for the UI's toggle label.
    /// Zero means "nothing to offer" and the toggle hides.
    nonisolated static var shareableVoiceBytes: Int64 {
        var total: Int64 = 0
        if KokoroAssets.isDownloaded {
            total += KokoroAssets.currentBytesOnDisk
        }
        #if canImport(FluidAudio)
        total += Supertonic3Assets.currentBytesOnDisk
        #endif
        return total
    }

    func setSharing(_ enabled: Bool) {
        if enabled {
            let urls = currentShareSet()
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
        updateSleepBlocker()
    }

    /// Flips whether the chat model rides along; if sharing is live, re-hosts
    /// so the advertised file list matches the toggle immediately.
    func setIncludeModel(_ on: Bool) {
        guard includeModelInShare != on else { return }
        includeModelInShare = on
        refreshSharingIfActive()
    }

    /// Same, for the voice models.
    func setIncludeVoice(_ on: Bool) {
        guard includeVoiceInShare != on else { return }
        includeVoiceInShare = on
        refreshSharingIfActive()
    }

    /// Re-seeds with the current share set (called after an import so a
    /// newly received ZIM is immediately shareable onward to a third
    /// device, and when the model toggle flips). The engine's manifest
    /// cache makes re-hosting unchanged files instant — only new files get
    /// hashed.
    private func refreshSharingIfActive() {
        guard isSharingLibrary else { return }
        let urls = currentShareSet()
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
        var importedModelCount = 0
        var importedVoiceCount = 0
        var skipped = 0
        let swarmDir = stagingBase?.appendingPathComponent(swarmID, isDirectory: true)

        // Move *everything* out of the per-swarm staging folder (which gets
        // cleaned up below). Voice-model trees ("kokoro_mlx/…",
        // "supertonic_3/…") route back into Application Support/models where
        // the TTS engines look; .zim files are imported into the library; a
        // .gguf is offered to the model providers, which adopt it into their
        // own cache slot; anything else a generic LocalSwarm peer sent just
        // lands in Documents, where the library scan ignores it.
        for source in fileURLs {
            let isZim = source.pathExtension.lowercased() == "zim"
            if let swarmDir,
               let relative = Self.relativePath(of: source, under: swarmDir),
               let voiceDestination = Self.voiceModelDestination(forRelativePath: relative) {
                if moveReplacing(source: source, destination: voiceDestination) {
                    importedVoiceCount += 1
                } else {
                    skipped += 1
                }
                continue
            }
            if source.pathExtension.lowercased() == "gguf",
               await importModelFile?(source) == true {
                importedModelCount += 1
                continue
            }
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
        if !imported.isEmpty || importedModelCount > 0 || importedVoiceCount > 0 {
            var parts: [String] = []
            if !imported.isEmpty {
                let size = ByteCountFormatter.string(fromByteCount: importedBytes, countStyle: .file)
                parts.append(imported.count == 1
                    ? "Added \(imported[0].lastPathComponent) (\(size))"
                    : "Added \(imported.count) files (\(size))")
            }
            if importedModelCount > 0 {
                parts.append(parts.isEmpty ? "Chat model installed" : "chat model included")
            }
            if importedVoiceCount > 0 {
                parts.append(parts.isEmpty ? "Voice models installed" : "voice models included")
            }
            lastImportSummary = parts.joined(separator: " · ")
            if !imported.isEmpty {
                await importFiles?(imported)
            }
            refreshSharingIfActive()
        }
        updateSleepBlocker()
    }

    /// The manifest-relative path of a received file ("kokoro_mlx/voices.npz"),
    /// or nil when the file isn't under the swarm's staging folder.
    nonisolated static func relativePath(of file: URL, under directory: URL) -> String? {
        let root = directory.standardizedFileURL.path
        let path = file.standardizedFileURL.path
        guard path.hasPrefix(root + "/") else { return nil }
        return String(path.dropFirst(root.count + 1))
    }

    /// Where a received voice-model file belongs, or nil when it isn't one.
    /// Only known voice trees/filenames are honored — an arbitrary swarm can
    /// never write elsewhere into Application Support. Two share shapes
    /// arrive here: the usual mixed swarm carries the folder-name prefix
    /// ("kokoro_mlx/voices.npz"), while a share consisting of *only* one
    /// voice folder is a folder swarm with unprefixed paths (the engine's
    /// Go-conformant form) — recognized by Kokoro's two known filenames or
    /// Supertonic's "supertonic-3-coreml/" bundle root.
    nonisolated static func voiceModelDestination(forRelativePath relative: String) -> URL? {
        let components = relative.split(separator: "/").map(String.init)
        guard let first = components.first else { return nil }

        switch first {
        case "kokoro_mlx" where components.count >= 2:
            return components.dropFirst()
                .reduce(KokoroAssets.modelDirectory) { $0.appendingPathComponent($1) }
        case "supertonic_3" where components.count >= 2:
            #if canImport(FluidAudio)
            return components.dropFirst()
                .reduce(Supertonic3Assets.modelDirectory) { $0.appendingPathComponent($1) }
            #else
            return nil
            #endif
        case "kokoro-v1_0.safetensors", "voices.npz":
            guard components.count == 1 else { return nil }
            return KokoroAssets.modelDirectory.appendingPathComponent(first)
        case "supertonic-3-coreml":
            #if canImport(FluidAudio)
            guard components.count >= 2 else { return nil }
            return components.reduce(Supertonic3Assets.modelDirectory) { $0.appendingPathComponent($1) }
            #else
            return nil
            #endif
        default:
            return nil
        }
    }

    /// Moves `source` over `destination`, creating parent directories and
    /// replacing an existing file (voice assets are interchangeable published
    /// files — a friend's copy and a downloaded copy are the same bytes).
    private nonisolated func moveReplacing(source: URL, destination: URL) -> Bool {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: destination.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: source, to: destination)
            return true
        } catch {
            return false
        }
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
        // Remove the per-swarm folder only when nothing but the engine's
        // sidecars and emptied-out subdirectories remain (a directory share
        // leaves its folder skeleton behind after the files move out) — so a
        // move failure never turns into data loss here.
        let disposable: Set<String> = [".localswarm-bitfield",
                                       ".localswarm-manifest.json",
                                       ".DS_Store"]
        guard let enumerator = fm.enumerator(at: dir,
                                             includingPropertiesForKeys: [.isRegularFileKey])
        else { return }
        for case let url as URL in enumerator {
            let isFile = (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
            if isFile, !disposable.contains(url.lastPathComponent) {
                return // real data still inside — keep everything
            }
        }
        try? fm.removeItem(at: dir)
    }

    private func updateSleepBlocker() {
        // The *sender* must stay awake too: if the sharing device dozes, iOS
        // suspends the app, the AWDL listener dies, and the friend's copy
        // stalls mid-transfer. Sharing is an explicit, session-scoped toggle,
        // so treating it as "keep me awake" is what the user asked for.
        SleepBlocker.set(engineBusy || isSharingLibrary, reason: "nearby-share")
    }
}
