// SPDX-License-Identifier: MIT
//
// LlamaCppProvider — ModelProvider conformance that runs GGUF
// models via the llama.cpp C API (vendored as Prism ML's
// `prism-b9591` XCFramework, exposed through the
// LocalPackages/llama.cpp-swift wrapper → `import LlamaCppSwift`).
//
// Why a direct-to-C provider instead of a Swift wrapper:
// `llama_context_params` now exposes `swa_full`, `type_k`,
// `type_v`, and `flash_attn_type`. None of the currently
// maintained Swift wrappers (LocalLLMClient, SwiftLlama, …)
// surface all four, and our shipping config needs every one of
// them (Q8_0 KV + iSWA rotation-pruning). The wrappers also
// assume they own chat formatting; we don't want that — our
// `ModelTemplate` protocol builds transcripts byte-exactly to
// match how the MLX path formats them.
//
// 2026-04-23 bench on Mac (bars_sc_caltrain_chain, Gemma 3 4B
// Q4_K_M + q8_0 KV + flash_attn + swa_full=false):
//   peak_rss = 3567 MB at 5k tokens, 3212 MB at 20k tokens
// vs MLX Gemma 3 4B on the same scenario: 6560 MB peak.
// ~2.9–3.4 GB of headroom reclaimed on iPhone.

import Foundation
import MCPZimKit
import OSLog

#if canImport(LlamaCppSwift)
import LlamaCppSwift

private let log = Logger(subsystem: "org.mcpzim.MCPZimChat", category: "LlamaCpp")

public enum LlamaKVCacheType: String, Sendable {
    case q4_0 = "Q4_0"
    case q8_0 = "Q8_0"
}

public struct LlamaPrefixCacheResult: Sendable {
    public let mode: String
    public let tokens: Int
    public let bytes: UInt64
    public let seconds: TimeInterval
}

public final class LlamaCppProvider: ModelProvider, @unchecked Sendable {

    // MARK: - ModelProvider conformance

    public let id: String
    public let displayName: String
    /// Approximate resident MB — weights + KV cache reservation.
    /// For Q4_K_M Gemma 3 4B with q8_0 KV and flash-attn iSWA
    /// pruning we see ~3.2 GB peak at 20 k tokens on Mac, so
    /// 3200 is a safe picker-UI figure on iPhone after the
    /// ~700 MB of WebKit/UIKit/Kokoro overhead is added on top.
    public let approximateMemoryMB: Int
    public let supportsToolCalls = true
    public let template: any ModelTemplate

    public let huggingFaceRepo: String
    public let ggufFilename: String
    /// Exact byte count for models whose publisher provides a stable file.
    /// When set, a truncated cache entry is never handed to llama.cpp. This
    /// also enables side-loaded `.part-NN` files to be assembled atomically,
    /// which is useful because CoreDevice can drop multi-GB copy sockets.
    public let expectedGGUFBytes: Int64?
    /// Exact SHA-256 (lowercase hex) of the pinned GGUF. When set, both the
    /// size and the checksum must validate before the weights are handed to
    /// llama.cpp — a same-size-but-corrupt file is rejected rather than
    /// half-loaded.
    public let expectedGGUFSHA256: String?
    /// When set to an existing file, load this GGUF directly instead of
    /// downloading from HuggingFace — lets the local Mac eval harness run
    /// the on-disk shipping model without a multi-GB fetch.
    public let localGGUFPath: String?
    /// Minimum per-turn reply-token budget, regardless of the device
    /// profile's conservative default. llama.cpp's KV cache is fixed at
    /// n_ctx (≈6 KB/token for LFM2.5's 6 attention layers), so long replies
    /// cost almost no extra memory here — the small default exists for the
    /// MLX path + TTS latency, not this one. Set high so grounded/discuss
    /// answers (which can run long, especially if the FT opens a <think>)
    /// don't truncate mid-sentence. `nil` = use the device default.
    public let replyTokensFloor: Int?
    /// Context window (n_ctx). llama.cpp PRE-ALLOCATES the KV buffer for the
    /// whole window at load, so this is a constant resident cost per model:
    /// LFM2.5 has 6 attention layers of 24 (8 KV-heads × 64 dim ≈ 6.9 KB/tok
    /// at q8_0 KV) → 32k ≈ 226 MB, paid for by the 2026-06-10 IQ3_XS requant
    /// (−0.53 GB). Heavier-KV models (Gemma 3 GGUF fallbacks) keep the 8k
    /// default. Full budget math: CONTEXT_BUDGET.md.
    public let contextTokens: Int
    /// Per-provider KV precision. Existing Gemma/LFM builds retain Q8_0;
    /// phone-class Bonsai uses Q4_0, matching Prism's recommended compact
    /// context configuration and leaving more iOS jetsam headroom.
    public let kvCacheType: LlamaKVCacheType
    /// Optional publisher/model-specific sampler settings. Bonsai uses
    /// Qwen 3.6's non-thinking recipe, while the older app
    /// default was 0.3 / 0.9 / top-k 40. Keeping this on the provider avoids
    /// accidentally applying Bonsai's higher-entropy recipe to every model.
    public let samplingProfile: GenerationSamplingProfile?
    /// Seed for probabilistic sampling. Production uses llama.cpp's random
    /// default; the Mac evaluator supplies a fixed seed so paired A/B runs
    /// compare retrieval strategies rather than different random draws.
    public let samplingSeed: UInt32

    // MARK: - State + llama.cpp handles

    private let queue = DispatchQueue(label: "LlamaCppProvider.state")
    private var state: ModelLoadState = .notLoaded
    private var continuations: [UUID: AsyncStream<ModelLoadState>.Continuation] = [:]

    /// Opaque handles from llama.cpp. Guarded by `modelLock` below —
    /// generate() can take a long time and we don't want `unload()`
    /// racing it.
    private let modelLock = NSLock()
    /// Separate from `modelLock`: cancellation must be writable while the
    /// detached llama.cpp task holds the model for a multi-second decode.
    private let generationControlLock = NSLock()
    private var activeGenerationID: UUID?
    private var cancelledGenerationIDs: Set<UUID> = []
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var vocab: OpaquePointer?
    private static let defaultSequenceIDs: [llama_seq_id] = [0]

    /// KV-cache mirror for follow-up LCP matching. Analogous to
    /// `Gemma4Provider.cachedTokens`. llama.cpp itself keeps the cache
    /// in-context; we only track this to decide whether the next
    /// turn's prefix can reuse it (same-prefix rule).
    private var cachedTokens: [Int32] = []

    /// One bounded recurrent-state snapshot. Attention KV stays in seq-0.
    /// Host storage can be released on reset; the optional device experiment
    /// instead retains one backend allocation until context destruction.
    private var recoveryCheckpoint: (tokens: [Int32], data: [UInt8], flags: UInt32)?
    private var isPhoneBonsaiQ1: Bool {
        #if os(iOS)
        return ggufFilename == "Bonsai-27B-Q1_0.gguf"
        #else
        return false
        #endif
    }
    private var recoveryCheckpointEnabled: Bool {
        #if DEBUG
        if let override = ProcessInfo.processInfo.environment["MCPZIM_LLAMA_RECOVERY_CHECKPOINT"] {
            return override == "1"
        }
        #endif
        return isPhoneBonsaiQ1
    }
    private var recoveryFlags: UInt32 {
        #if DEBUG
        if ProcessInfo.processInfo.environment["MCPZIM_LLAMA_RECOVERY_STORAGE"] == "device" {
            return UInt32(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)
        }
        #endif
        return UInt32(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
    }
    private var generationPrefillBatchSize: Int {
        #if DEBUG
        if let raw = ProcessInfo.processInfo.environment["MCPZIM_LLAMA_PREFILL_BATCH"],
           let size = Int(raw), (16...512).contains(size) { return size }
        #endif
        // Shorter submissions passed the A19 Pro hot-run recovery suite and
        // bound cancellation latency. Keep the 512-token context capacity.
        return isPhoneBonsaiQ1 ? 128 : 512
    }

    /// DEBUG-only one-shot guard for the explicit llama.cpp session-file
    /// benchmark. Set MCPZIM_BENCH_STATE_CACHE=1 on launch to measure a full
    /// save + restore without paying the cost on normal turns.
    private var didBenchmarkStateCache = false

    /// In-window + persistent-log sink. ChatSession wires this after provider
    /// construction so stage timings survive a crash/jetsam and are visible
    /// without an attached Xcode console.
    public var debugSink: (@Sendable (String) -> Void)?

    /// Uniform cross-runtime stats for the last completed generate().
    /// Generation writes this from a detached worker while ChatSession reads
    /// it on the main actor, so use a dedicated lock rather than relying on
    /// the model lock (which generation already holds for the whole turn).
    private let generationStatsLock = NSLock()
    private var storedGenerationStats: GenerationStats?
    public var lastGenerationStats: GenerationStats? {
        generationStatsLock.lock()
        defer { generationStatsLock.unlock() }
        return storedGenerationStats
    }

    // MARK: - Init

    public init(
        id: String = "gemma3-4b-it-q4km-gguf",
        displayName: String = "Gemma 3 4B IT (Q4_K_M · llama.cpp)",
        huggingFaceRepo: String = "bartowski/google_gemma-3-4b-it-GGUF",
        ggufFilename: String = "google_gemma-3-4b-it-Q4_K_M.gguf",
        expectedGGUFBytes: Int64? = nil,
        expectedGGUFSHA256: String? = nil,
        localGGUFPath: String? = nil,
        replyTokensFloor: Int? = nil,
        approximateMemoryMB: Int = 3200,
        contextTokens: Int = 8192,
        kvCacheType: LlamaKVCacheType = .q8_0,
        samplingProfile: GenerationSamplingProfile? = nil,
        samplingSeed: UInt32 = 0xFFFF_FFFF,
        template: any ModelTemplate = Gemma3Template()
    ) {
        self.id = id
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.ggufFilename = ggufFilename
        self.expectedGGUFBytes = expectedGGUFBytes
        self.expectedGGUFSHA256 = expectedGGUFSHA256
        self.localGGUFPath = localGGUFPath
        self.replyTokensFloor = replyTokensFloor
        self.approximateMemoryMB = approximateMemoryMB
        self.contextTokens = contextTokens
        self.kvCacheType = kvCacheType
        self.samplingProfile = samplingProfile
        self.samplingSeed = samplingSeed
        self.template = template
        // One-time global init. Safe to call repeatedly per
        // llama.cpp docs; the backend keeps a refcount.
        llama_backend_init()
    }

    // MARK: - State stream

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { continuation in
            let id = UUID()
            queue.sync {
                continuation.yield(self.state)
                self.continuations[id] = continuation
            }
            // Drop dead subscribers so `continuations` doesn't grow
            // unboundedly (and we don't yield to terminated streams) as
            // models are switched over a long session.
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.queue.async { self.continuations[id] = nil }
            }
        }
    }

    private func set(_ new: ModelLoadState) {
        queue.sync {
            state = new
            for c in continuations.values { c.yield(new) }
        }
    }

    // MARK: - Load

    public func load() async throws {
        set(.loading)
        LogArchive.shared.trace("load: \(displayName) · id=\(id) · cached=\(hasCompleteCachedGGUF)")
        do {
            if localGGUFPath == nil, !ModelCatalog.modelFitsDevice(approximateMemoryMB) {
                LogArchive.shared.trace("load: REJECTED — needs ~\(approximateMemoryMB) MB")
                throw LlamaCppError.custom(
                    "\(displayName) needs ~\(approximateMemoryMB) MB, but this device can't hold it. Pick a model that fits — Settings shows which.")
            }
            let path: URL
            if let lp = localGGUFPath, FileManager.default.fileExists(atPath: lp) {
                path = URL(fileURLWithPath: lp)
            } else {
                path = try await ensureGGUFDownloaded()
            }
            try openModel(at: path)
            set(.ready)
        } catch {
            set(.failed(String(describing: error)))
            throw error
        }
    }

    /// True only when the complete, byte-validated GGUF is already present.
    /// The Mac Models menu uses this to distinguish "Download" from "Use"
    /// instead of asking the user to infer cache state from a spinner.
    public var hasCompleteCachedGGUF: Bool {
        if let localGGUFPath {
            return FileManager.default.fileExists(atPath: localGGUFPath)
        }
        guard let url = cachedGGUFURL else { return false }
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        return (try? cachedFileHasExpectedSize(url)) == true
    }

    private var cachedGGUFURL: URL? {
        guard let cachesDir = FileManager.default.urls(
            for: .cachesDirectory, in: .userDomainMask).first
        else { return nil }
        let repoSlug = huggingFaceRepo.replacingOccurrences(of: "/", with: "--")
        return cachesDir
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--\(repoSlug)")
            .appendingPathComponent("snapshots")
            .appendingPathComponent("main")
            .appendingPathComponent(ggufFilename)
    }

    /// The complete, byte-validated GGUF on disk, if any — what Nearby
    /// Sharing offers to a friend so a fresh install can chat without ever
    /// touching the internet.
    public var shareableGGUFURL: URL? {
        if let localGGUFPath {
            let url = URL(fileURLWithPath: localGGUFPath)
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
        guard let url = cachedGGUFURL,
              FileManager.default.fileExists(atPath: url.path),
              (try? cachedFileHasExpectedSize(url)) == true else { return nil }
        return url
    }

    /// Claim a GGUF received from a nearby device. The filename must match
    /// this provider's pinned file and the byte count must validate — the
    /// swarm already SHA-256-verified every chunk in transit, so the size
    /// check's job is rejecting a same-named file from a different release.
    /// On success the file is *moved* into the provider's cache slot (the
    /// same place `ensureGGUFDownloaded()` would have put it) and the next
    /// load skips the network entirely.
    public func adoptSharedGGUF(at url: URL) -> Bool {
        guard url.lastPathComponent == ggufFilename,
              (try? cachedFileHasExpectedSize(url)) == true,
              let destination = cachedGGUFURL else { return false }
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            if (try? cachedFileHasExpectedSize(destination)) == true {
                // Already have a good copy; drop the duplicate.
                try? fm.removeItem(at: url)
                return true
            }
            // Replace a truncated/failed earlier download.
            try? fm.removeItem(at: destination)
        }
        do {
            try fm.createDirectory(at: destination.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            try fm.moveItem(at: url, to: destination)
            log.info("adopted shared GGUF into cache: \(self.ggufFilename, privacy: .public)")
            return true
        } catch {
            log.error("failed to adopt shared GGUF: \(error.localizedDescription, privacy: .public)")
            return false
        }
    }

    /// Download the GGUF via HuggingFace's direct resolve URL if not
    /// already on disk. We store under the same
    /// `<caches>/huggingface/hub/models--<repo>/snapshots/main/<file>`
    /// layout the MLX path uses — single cache directory for both
    /// runtimes, saves duplicate downloads when switching providers.
    ///
    /// The actual transfer is handed to `ZimDownloadManager`, the same
    /// background `URLSession` downloader the ZIM catalog uses. That buys:
    ///   * the transfer survives suspension / app termination (the system
    ///     relaunches us to hand the finished file back);
    ///   * a partial pull resumes across relaunches from persisted resume
    ///     data, and reconnects to in-flight tasks on launch;
    ///   * the file is staged and size **and** SHA-256-verified (when pinned)
    ///     before it moves into the cache slot, so llama.cpp never sees a
    ///     truncated or corrupt GGUF;
    ///   * the storage gate refuses to start a multi-GB pull the volume
    ///     can't hold, naming the shortfall.
    private func ensureGGUFDownloaded() async throws -> URL {
        guard let destURL = cachedGGUFURL else {
            throw LlamaCppError.custom("could not resolve the app cache directory")
        }
        let destDir = destURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: destDir, withIntermediateDirectories: true)

        LogArchive.shared.trace("ensureGGUF: \(ggufFilename) · pinned=\(expectedGGUFBytes ?? -1) · cached=\(FileManager.default.fileExists(atPath: destURL.path))")

        // Cache hit: size + (pinned) SHA-256 must both validate before the
        // file is handed to llama.cpp. SHA-256 is streamed off the main
        // actor, so a multi-GB cache hit doesn't block the UI.
        if FileManager.default.fileExists(atPath: destURL.path) {
            if try await Self.cachedGGUFIsValid(
                destURL, expectedBytes: expectedGGUFBytes,
                sha256: expectedGGUFSHA256) {
                LogArchive.shared.trace("ensureGGUF: cache hit · \(ggufFilename)")
                debugSink?("model cache hit · \(ggufFilename) · \(Self.byteCountDescription(try Self.fileSize(destURL)))")
                return destURL
            }
            LogArchive.shared.trace("ensureGGUF: cache INVALID — re-downloading · \(ggufFilename)")
            let bytes = try Self.fileSize(destURL)
            log.warning(
                "Removing incomplete/corrupt GGUF cache entry: \(bytes, privacy: .public) bytes at \(destURL.path, privacy: .public)"
            )
            try FileManager.default.removeItem(at: destURL)
        }

        // `devicectl` can drop its file-service socket during one multi-GB
        // transfer. Deployment tooling may instead place ordered files named
        // `<model>.part-00`, `.part-01`, … beside the destination. Assemble
        // only when their total exactly matches the publisher's byte count;
        // the final rename is atomic, so a killed app never exposes a partial
        // GGUF as complete.
        if let expectedGGUFBytes {
            let assembled = try await Task.detached(priority: .utility) {
                try Self.assembleSideLoadedParts(
                    in: destDir,
                    filename: self.ggufFilename,
                    expectedBytes: expectedGGUFBytes
                )
            }.value
            if assembled {
                log.info(
                    "Assembled side-loaded GGUF: \(expectedGGUFBytes, privacy: .public) bytes"
                )
                return destURL
            }
        }
        let urlStr =
            "https://huggingface.co/\(huggingFaceRepo)/resolve/main/\(ggufFilename)"
        guard let url = URL(string: urlStr) else {
            throw LlamaCppError.custom("bad HF URL: \(urlStr)")
        }
        log.info("GGUF download start: \(urlStr, privacy: .public)")
        debugSink?("model download start · \(ggufFilename) · \(Self.byteCountDescription(expectedGGUFBytes))")
        // Emit "0%" immediately so the UI flips from `.loading` to
        // `.downloading(0)` before the first byte arrives.
        set(.downloading(0))
        LogArchive.shared.trace("ensureGGUF: handing off to downloader · \(ggufFilename)")

        let spec = ModelDownloadSpec(
            id: id, title: displayName, url: url,
            expectedBytes: expectedGGUFBytes ?? 0,
            sha256: expectedGGUFSHA256,
            destination: destURL)
        do {
            let url = try await ZimDownloadManager.shared.downloadModel(
                spec: spec,
                onProgress: { [weak self] fraction in
                    self?.set(.downloading(fraction))
                    let percent = Int((fraction * 100).rounded(.down))
                    if percent > 0, percent.isMultiple(of: 10) {
                        self?.debugSink?("model download progress · \(percent)%")
                    }
                },
                onWaiting: { [weak self] waiting in
                    if waiting { self?.set(.waitingForNetwork) }
                }
            )
            log.info("GGUF download complete: \(url.path, privacy: .public)")
            debugSink?("model download complete · \(ggufFilename) · \(Self.byteCountDescription(try? Self.fileSize(url)))")
            // Transition back to .loading so the UI knows we're now doing
            // the model open + Metal warm-up, not still downloading.
            set(.loading)
            return url
        } catch {
            throw LlamaCppError.custom(Self.downloadErrorMessage(for: error))
        }
    }

    private func cachedFileHasExpectedSize(_ url: URL) throws -> Bool {
        guard let expectedGGUFBytes else { return true }
        return try Self.fileSize(url) == expectedGGUFBytes
    }

    /// Full cache validity: size (when pinned) and SHA-256 (when pinned).
    /// This is the authoritative gate before llama.cpp loads a file; the
    /// cheap `cachedFileHasExpectedSize` is only a menu hint.
    ///
    /// SHA-256 is streamed off the main actor. The result is memoized per
    /// file version (path + size + mtime), so a launch that calls `load()`
    /// several times doesn't re-hash a multi-GB GGUF on every call — the
    /// hash is paid at most once per process for a given on-disk file.
    nonisolated static func cachedGGUFIsValid(
        _ url: URL, expectedBytes: Int64?, sha256: String?
    ) async throws -> Bool {
        if let expectedBytes {
            guard try fileSize(url) == expectedBytes else { return false }
        }
        guard let sha256 else { return true }
        let key = shaVerifyKey(url) + "|" + sha256.lowercased()
        if let cached = memoizedSHA256Verdict(key) {
            return cached
        }
        let digest = try await Task.detached(priority: .utility) {
            try ZimDownloadManager.sha256Hex(of: url)
        }.value
        let ok = digest.lowercased() == sha256.lowercased()
        storeSHA256Verdict(key, ok)
        return ok
    }

    /// Sync memo read/write kept out of the async body so the lock isn't
    /// held across an `await` (Swift 6 would reject it).
    private static let shaVerifyLock = NSLock()
    private static var shaVerified: [String: Bool] = [:]

    private nonisolated static func memoizedSHA256Verdict(_ key: String) -> Bool? {
        shaVerifyLock.lock()
        defer { shaVerifyLock.unlock() }
        return shaVerified[key]
    }

    private nonisolated static func storeSHA256Verdict(_ key: String, _ ok: Bool) {
        shaVerifyLock.lock()
        defer { shaVerifyLock.unlock() }
        shaVerified[key] = ok
    }

    /// File-version key so a re-download (new mtime/size) invalidates the
    /// memoized SHA-256 verdict.
    private nonisolated static func shaVerifyKey(_ url: URL) -> String {
        let attrs = (try? FileManager.default.attributesOfItem(atPath: url.path)) ?? [:]
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return "\(url.path)|\(size)|\(mtime)"
    }

    /// Turn downloader failures into an actionable message. The storage gate
    /// already names the shortfall; a transport/checksum failure tells the
    /// user to retry.
    private nonisolated static func downloadErrorMessage(for error: Error) -> String {
        if let zimError = error as? ZimDownloadError {
            return zimError.localizedDescription
        }
        return "Model download failed: \(error.localizedDescription). Check your connection and retry."
    }

    private static func fileSize(_ url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = attributes[.size] as? NSNumber else {
            throw LlamaCppError.custom("could not read file size for \(url.path)")
        }
        return size.int64Value
    }

    private static func byteCountDescription(_ bytes: Int64?) -> String {
        guard let bytes, bytes > 0 else { return "size unknown" }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    /// Join a complete set of deployment parts without ever exposing a
    /// truncated destination. Returns false when no complete set is present,
    /// allowing the normal resumable HTTP downloader to take over.
    private static func assembleSideLoadedParts(
        in directory: URL,
        filename: String,
        expectedBytes: Int64
    ) throws -> Bool {
        let manager = FileManager.default
        let prefix = filename + ".part-"
        let parts = try manager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        .filter { $0.lastPathComponent.hasPrefix(prefix) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !parts.isEmpty else { return false }
        let total = try parts.reduce(Int64(0)) { partial, url in
            partial + (try fileSize(url))
        }
        guard total == expectedBytes else { return false }

        let destination = directory.appendingPathComponent(filename)
        let staging = directory.appendingPathComponent(filename + ".assembling")
        try? manager.removeItem(at: staging)
        guard manager.createFile(atPath: staging.path, contents: nil) else {
            throw LlamaCppError.custom("could not create GGUF assembly file")
        }

        do {
            do {
                let output = try FileHandle(forWritingTo: staging)
                defer { try? output.close() }
                for part in parts {
                    do {
                        let input = try FileHandle(forReadingFrom: part)
                        defer { try? input.close() }
                        while let data = try input.read(upToCount: 8 * 1_024 * 1_024),
                              !data.isEmpty {
                            try output.write(contentsOf: data)
                        }
                    }
                }
                try output.synchronize()
            }

            guard try fileSize(staging) == expectedBytes else {
                throw LlamaCppError.custom("assembled GGUF size mismatch")
            }
            try? manager.removeItem(at: destination)
            try manager.moveItem(at: staging, to: destination)
            for part in parts { try? manager.removeItem(at: part) }
            return true
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }
    }

    /// Open the model + create a context configured with our
    /// shipping-config flags: Metal-all-layers, Q8_0 KV, flash-attn,
    /// iSWA pruning. All four of these were load-bearing in the
    /// bench that chose llama.cpp over MLX for Gemma 3 — see file
    /// header for the numbers.
    private func openModel(at path: URL) throws {
        modelLock.lock()
        defer { modelLock.unlock() }
        if let c = ctx { llama_free(c); ctx = nil }
        if let m = model { llama_model_free(m); model = nil }
        vocab = nil
        cachedTokens = []
        recoveryCheckpoint = nil

        // Model
        var mp = llama_model_default_params()
        mp.n_gpu_layers = -1          // all on Metal
        // Startup crash breadcrumb: these two llama.cpp calls are the
        // known iOS kill points. Synchronous writes so a SIGABRT leaves
        // the stage on disk (LogArchive's async queue would lose them).
        LogArchive.shared.trace("openModel: loading \(path.lastPathComponent) · n_gpu_layers=\(mp.n_gpu_layers)")
        guard let m = llama_model_load_from_file(path.path, mp) else {
            throw LlamaCppError.custom("llama_model_load_from_file failed for \(path.lastPathComponent)")
        }
        self.model = m
        self.vocab = llama_model_get_vocab(m)

        // Context — shipping config (same knobs that crash iOS,
        // testing them now on Mac via MCPZimChatMac to see whether
        // it's iOS-Metal-specific or a Swift invocation bug).
        var cp = llama_context_default_params()
        cp.n_ctx = UInt32(contextTokens)
        cp.n_batch = 512
        cp.n_ubatch = 512
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        switch kvCacheType {
        case .q4_0:
            cp.type_k = GGML_TYPE_Q4_0
            cp.type_v = GGML_TYPE_Q4_0
        case .q8_0:
            cp.type_k = GGML_TYPE_Q8_0
            cp.type_v = GGML_TYPE_Q8_0
        }
        cp.swa_full = false
        cp.offload_kqv = true
        LogArchive.shared.trace("openModel: init context · n_ctx=\(cp.n_ctx) · kv=\(kvCacheType.rawValue) · fa=\(cp.flash_attn_type) · swa_full=\(cp.swa_full)")
        guard let c = llama_init_from_model(m, cp) else {
            llama_model_free(m)
            self.model = nil
            self.vocab = nil
            self.cachedTokens = []
            throw LlamaCppError.custom("llama_init_from_model failed")
        }
        self.ctx = c
        log.notice("loaded \(self.ggufFilename, privacy: .public) · n_ctx=\(cp.n_ctx) kv=\(self.kvCacheType.rawValue, privacy: .public) fa=on swa_full=false")
        debug("loaded context · n_ctx=\(cp.n_ctx) · kv=\(kvCacheType.rawValue) · flash-attn=on · swa-full=false · prefill-batch=\(generationPrefillBatchSize) · recovery=\(recoveryCheckpointEnabled) · recovery-flags=\(recoveryFlags)")
    }

    public func unload() async {
        modelLock.lock()
        defer { modelLock.unlock() }
        if let c = ctx { llama_free(c); ctx = nil }
        if let m = model { llama_model_free(m); model = nil }
        vocab = nil
        cachedTokens = []
        recoveryCheckpoint = nil
        set(.notLoaded)
    }

    /// Exact token count using the loaded model's tokenizer. ChatSession uses
    /// this to compact an append-only grounded transcript before it consumes
    /// the reply reservation in n_ctx. Tokenisation is cheap (~5 ms for a
    /// 2k-token prompt in current device captures).
    public func promptTokenCount(_ prompt: String) -> Int? {
        modelLock.lock()
        defer { modelLock.unlock() }
        guard let vocab else { return nil }
        return Self.tokenize(vocab: vocab, prompt: prompt).count
    }

    /// Ensure the context is ready to extend `fullPrompt` without paying to
    /// prefill its large static system+tools prefix.
    ///
    /// Three paths are possible:
    /// - the live context is already an exact prefix of `fullPrompt`
    ///   (ordinary append-only conversation);
    /// - a serialized static-prefix state is restored from SSD;
    /// - the prefix is evaluated once, persisted, then reused in memory.
    ///
    /// llama.cpp state files are snapshots, not an automatic SSD cache tier.
    /// The caller owns the cache key/path and must change it whenever the
    /// model, runtime, template, system prompt, or tools change.
    public func preparePromptPrefix(
        prefixPrompt: String,
        fullPrompt: String,
        cacheURL: URL,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> LlamaPrefixCacheResult {
        let worker = Task.detached(priority: .utility) { [weak self] in
            guard let self else { throw LlamaCppError.notLoaded }
            return try self.preparePromptPrefixLocked(
                prefixPrompt: prefixPrompt,
                fullPrompt: fullPrompt,
                cacheURL: cacheURL,
                progress: progress)
        }
        return try await withTaskCancellationHandler {
            try await worker.value
        } onCancel: {
            // `Task.detached` does not inherit cancellation from its waiter.
            // Propagate it explicitly so a model/library switch only waits for
            // the current 512-token batch instead of the entire prefill.
            worker.cancel()
        }
    }

    private func preparePromptPrefixLocked(
        prefixPrompt: String,
        fullPrompt: String,
        cacheURL: URL,
        progress: (@Sendable (Double) -> Void)?
    ) throws -> LlamaPrefixCacheResult {
        let started = ProcessInfo.processInfo.systemUptime
        modelLock.lock()
        defer { modelLock.unlock() }
        guard let ctx, let vocab else { throw LlamaCppError.notLoaded }

        let prefixTokens = Self.tokenize(vocab: vocab, prompt: prefixPrompt)
        let fullTokens = Self.tokenize(vocab: vocab, prompt: fullPrompt)
        guard !prefixTokens.isEmpty,
              prefixTokens.count < fullTokens.count,
              fullTokens.prefix(prefixTokens.count).elementsEqual(prefixTokens)
        else {
            throw LlamaCppError.custom(
                "static prompt cache is not an exact token prefix of the request")
        }

        // Best case: the previous request is itself a strict prefix of this
        // one. Keep the richer live state; restoring only the static prefix
        // would throw away useful conversation KV/recurrent state.
        if !cachedTokens.isEmpty,
           cachedTokens.count < fullTokens.count,
           fullTokens.prefix(cachedTokens.count).elementsEqual(cachedTokens)
        {
            return LlamaPrefixCacheResult(
                mode: "live-append",
                tokens: cachedTokens.count,
                bytes: 0,
                seconds: ProcessInfo.processInfo.systemUptime - started)
        }

        // A prior restore/prime may already have left exactly the reusable
        // prefix in memory.
        if cachedTokens == prefixTokens {
            return LlamaPrefixCacheResult(
                mode: "memory-prefix",
                tokens: prefixTokens.count,
                bytes: 0,
                seconds: ProcessInfo.processInfo.systemUptime - started)
        }

        let fm = FileManager.default
        // The following paths replace the attention cache underneath a
        // recovery snapshot. Exact live-prefix returns above retain it.
        recoveryCheckpoint = nil
        if fm.fileExists(atPath: cacheURL.path) {
            var restoredTokens = [Int32](
                repeating: 0, count: contextTokens)
            var restoredCount = 0
            if let mem = llama_get_memory(ctx) {
                _ = llama_memory_seq_rm(mem, 0, 0, -1)
            }
            let loadStarted = ProcessInfo.processInfo.systemUptime
            let restoredBytes = cacheURL.path.withCString { path in
                restoredTokens.withUnsafeMutableBufferPointer { buf in
                    guard let base = buf.baseAddress else { return 0 }
                    return llama_state_seq_load_file(
                        ctx, path, 0, base, buf.count, &restoredCount)
                }
            }
            let loadedTokens = Array(restoredTokens.prefix(restoredCount))
            if restoredBytes > 0, loadedTokens == prefixTokens {
                cachedTokens = loadedTokens
                let bytes = Self.fileSizeIfPresent(cacheURL)
                Self.touchAndPrunePrefixCache(keeping: cacheURL)
                let seconds = ProcessInfo.processInfo.systemUptime - loadStarted
                debug(String(format:
                    "prefix cache · disk restore · %d tok · %.1f MB · %.3fs",
                    restoredCount, Double(bytes) / 1_048_576, seconds))
                return LlamaPrefixCacheResult(
                    mode: "disk-restore",
                    tokens: restoredCount,
                    bytes: bytes,
                    seconds: ProcessInfo.processInfo.systemUptime - started)
            }

            // A state file from a different model/runtime must never be
            // handed to generation. Clear any partially restored state and
            // rebuild under the caller's current cache key.
            if let mem = llama_get_memory(ctx) {
                _ = llama_memory_seq_rm(mem, 0, 0, -1)
            }
            cachedTokens = []
            try? fm.removeItem(at: cacheURL)
            debug("prefix cache · rejected stale/incompatible state; rebuilding")
        }

        if let mem = llama_get_memory(ctx) {
            _ = llama_memory_seq_rm(mem, 0, 0, -1)
        }
        cachedTokens = []

        let primeStarted = ProcessInfo.processInfo.systemUptime
        // This is background work and can be preempted by a compact grounded
        // turn. Keep batches small so cancellation releases `modelLock` in
        // roughly a quarter of the prior worst case instead of making the
        // foreground request wait through another 512-token decode.
        let nBatch = 128
        var batch = llama_batch_init(Int32(nBatch), 0, 1)
        defer { llama_batch_free(batch) }
        var pos: Int32 = 0
        var i = 0
        progress?(0)
        do {
            while i < prefixTokens.count {
                if Task.isCancelled { throw CancellationError() }
                let end = min(i + nBatch, prefixTokens.count)
                batch.n_tokens = 0
                for j in i..<end {
                    Self.batchAdd(
                        &batch,
                        token: prefixTokens[j],
                        pos: pos,
                        seqIds: Self.defaultSequenceIDs,
                        logits: j == prefixTokens.count - 1)
                    pos += 1
                }
                let rc = llama_decode(ctx, batch)
                guard rc == 0 else {
                    throw LlamaCppError.custom(
                        "llama_decode prefix prefill rc=\(rc) at \(i)..<\(end)")
                }
                i = end
                progress?(Double(i) / Double(prefixTokens.count))
            }
            cachedTokens = prefixTokens
        } catch {
            if let mem = llama_get_memory(ctx) {
                _ = llama_memory_seq_rm(mem, 0, 0, -1)
            }
            cachedTokens = []
            throw error
        }
        let primeSeconds = ProcessInfo.processInfo.systemUptime - primeStarted

        var savedBytes: UInt64 = 0
        do {
            let directory = cacheURL.deletingLastPathComponent()
            try fm.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let staging = directory.appendingPathComponent(
                ".\(cacheURL.lastPathComponent).\(UUID().uuidString).tmp")
            defer { try? fm.removeItem(at: staging) }
            let serializedBytes = staging.path.withCString { path in
                prefixTokens.withUnsafeBufferPointer { buf in
                    guard let base = buf.baseAddress else { return 0 }
                    return llama_state_seq_save_file(
                        ctx, path, 0, base, buf.count)
                }
            }
            guard serializedBytes > 0 else {
                throw LlamaCppError.custom(
                    "llama_state_seq_save_file failed for static prefix")
            }
            try? fm.removeItem(at: cacheURL)
            try fm.moveItem(at: staging, to: cacheURL)
            #if os(iOS)
            try? fm.setAttributes(
                [.protectionKey:
                    FileProtectionType.completeUntilFirstUserAuthentication],
                ofItemAtPath: cacheURL.path)
            #endif
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var mutableURL = cacheURL
            try? mutableURL.setResourceValues(values)
            savedBytes = Self.fileSizeIfPresent(cacheURL)
            Self.touchAndPrunePrefixCache(keeping: cacheURL)
        } catch {
            // The in-memory prefix is still valid. A full disk should not
            // turn a model answer into an error; log and continue.
            debug("prefix cache · SSD save skipped: \(error)")
        }

        let totalSeconds = ProcessInfo.processInfo.systemUptime - started
        debug(String(format:
            "prefix cache · built · %d tok · prefill %.3fs · file %.1f MB · total %.3fs",
            prefixTokens.count, primeSeconds,
            Double(savedBytes) / 1_048_576, totalSeconds))
        return LlamaPrefixCacheResult(
            mode: savedBytes > 0 ? "built-and-saved" : "built-memory-only",
            tokens: prefixTokens.count,
            bytes: savedBytes,
            seconds: totalSeconds)
    }

    /// Keep SSD state useful without letting prompt variants grow without
    /// bound. Entries can differ by model, runtime, context/KV configuration,
    /// and tool schema; stale variants are discarded oldest first. Always
    /// retain the state that was just saved/restored even if it alone exceeds
    /// the soft byte budget.
    private static func touchAndPrunePrefixCache(keeping cacheURL: URL) {
        let fm = FileManager.default
        try? fm.setAttributes(
            [.modificationDate: Date()], ofItemAtPath: cacheURL.path)
        let directory = cacheURL.deletingLastPathComponent()
        guard let files = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [
                .contentModificationDateKey, .fileSizeKey,
            ],
            options: [.skipsHiddenFiles])
        else { return }

        struct Entry {
            let url: URL
            let date: Date
            let bytes: UInt64
        }
        let entries: [Entry] = files.compactMap { url in
            guard url.pathExtension == "bin", url != cacheURL else {
                return nil
            }
            let values = try? url.resourceValues(forKeys: [
                .contentModificationDateKey, .fileSizeKey,
            ])
            return Entry(
                url: url,
                date: values?.contentModificationDate ?? .distantPast,
                bytes: UInt64(max(0, values?.fileSize ?? 0)))
        }.sorted { $0.date > $1.date }

        #if os(iOS)
        let byteBudget: UInt64 = 640 * 1_048_576
        #else
        let byteBudget: UInt64 = 1_536 * 1_048_576
        #endif
        var keptFiles = 1
        var keptBytes = fileSizeIfPresent(cacheURL)
        for entry in entries {
            if keptFiles < 2, keptBytes + entry.bytes <= byteBudget {
                keptFiles += 1
                keptBytes += entry.bytes
            } else {
                try? fm.removeItem(at: entry.url)
            }
        }
    }

    /// Forget provider-side conversation state. The serialized static prefix
    /// remains on SSD and can be restored on the next generic request.
    public func resetPromptCache() {
        modelLock.lock()
        defer { modelLock.unlock() }
        if let ctx, let mem = llama_get_memory(ctx) {
            _ = llama_memory_seq_rm(mem, 0, 0, -1)
        }
        cachedTokens = []
        recoveryCheckpoint = nil
    }

    // MARK: - Generate

    public func generate(
        prompt: String, parameters: GenerationParameters
    ) -> AsyncThrowingStream<String, Error> {
        let generationID = UUID()
        generationControlLock.lock()
        activeGenerationID = generationID
        cancelledGenerationIDs.remove(generationID)
        generationControlLock.unlock()
        return AsyncThrowingStream { continuation in
            Task.detached { [weak self] in
                guard let self else {
                    continuation.finish(throwing: LlamaCppError.notLoaded)
                    return
                }
                do {
                    try self.generateLocked(
                        prompt: prompt,
                        parameters: parameters,
                        generationID: generationID,
                        emit: { continuation.yield($0) })
                    self.finishGeneration(generationID)
                    continuation.finish()
                } catch {
                    self.finishGeneration(generationID)
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [weak self] termination in
                if case .cancelled = termination {
                    self?.cancelGeneration(generationID)
                }
            }
        }
    }

    public func cancelGeneration() {
        generationControlLock.lock()
        if let id = activeGenerationID { cancelledGenerationIDs.insert(id) }
        generationControlLock.unlock()
    }

    private func cancelGeneration(_ id: UUID) {
        generationControlLock.lock()
        cancelledGenerationIDs.insert(id)
        generationControlLock.unlock()
    }

    private func finishGeneration(_ id: UUID) {
        generationControlLock.lock()
        cancelledGenerationIDs.remove(id)
        if activeGenerationID == id { activeGenerationID = nil }
        generationControlLock.unlock()
    }

    private func generationIsCancelled(_ id: UUID) -> Bool {
        generationControlLock.lock()
        defer { generationControlLock.unlock() }
        return cancelledGenerationIDs.contains(id)
    }

    /// Synchronous generation body. Runs on a Task-detached background
    /// thread (llama.cpp's `llama_decode` is blocking). Holds
    /// `modelLock` for the whole turn so `unload()` can't yank the
    /// model out from under us mid-decode.
    private func generateLocked(
        prompt: String,
        parameters: GenerationParameters,
        generationID: UUID,
        emit: @escaping (String) -> Void
    ) throws {
        if generationIsCancelled(generationID) { throw CancellationError() }
        let requestStarted = ProcessInfo.processInfo.systemUptime
        let memoryStarted = MemoryStats.physFootprintMB()
        let thermalStarted = Self.thermalStateLabel()
        let lockStarted = requestStarted
        modelLock.lock()
        defer { modelLock.unlock() }
        let lockSeconds = ProcessInfo.processInfo.systemUptime - lockStarted
        guard let ctx = ctx, let vocab = vocab else {
            throw LlamaCppError.notLoaded
        }

        let tokenizeStarted = ProcessInfo.processInfo.systemUptime
        let tokens = Self.tokenize(vocab: vocab, prompt: prompt)
        let tokenizeSeconds = ProcessInfo.processInfo.systemUptime - tokenizeStarted
        debug(String(format:
            "perf start · prompt=%d chars · lock=%.3fs · tokenize=%d tok/%.3fs · thermal=%@ · footprint=%.0f MB",
            prompt.count, lockSeconds, tokens.count, tokenizeSeconds,
            thermalStarted, memoryStarted))
        guard tokens.count < contextTokens - 16 else {
            throw LlamaCppError.custom(
                "prompt (\(tokens.count) tok) exceeds n_ctx=\(contextTokens) — "
                + "trim the transcript before generating")
        }

        // Cross-turn KV prefix reuse (2026-06-10). ChatSession rebuilds the
        // transcript byte-for-byte each turn (`toolRoundTrips` exists for
        // exactly this), so turn N's prompt is normally turn N-1's prompt +
        // reply + one new exchange. Instead of wiping seq-0 and re-prefilling
        // the whole transcript (cost grows linearly with conversation length:
        // ~4 s at a full 8k, ~15 s at 32k), keep `cachedTokens` as a mirror of
        // what seq-0's KV actually contains, drop only the divergent tail, and
        // prefill only the new suffix.
        //
        // History that shaped this:
        // - 2026-04-25 (gist 80daf913): writing turn N at pos=0 OVER stale
        //   seq-0 cells made llama_decode rc=-1 (slot pos overlap). The fix
        //   then was a full `llama_memory_seq_rm(0, 0, -1)` wipe per turn.
        //   The reuse below never writes over live cells either — it removes
        //   [lcp, ∞) first and starts the new batch at pos=lcp.
        // - `llama_memory_clear` (global) was suspected in a silent iOS death
        //   2026-04-24; we still avoid it.
        var lcp = 0
        while lcp < tokens.count, lcp < cachedTokens.count,
              tokens[lcp] == cachedTokens[lcp] { lcp += 1 }
        let lcpCandidate = lcp
        var cacheMode = "cold"
        // Need at least one fresh token in the batch to obtain logits — if
        // the new prompt is entirely a prefix of the cache, re-decode its
        // final token (forces the divergence path below).
        if lcp == tokens.count { lcp = max(0, lcp - 1) }
        // Two-tier reuse. LFM2.5 is a HYBRID (6 attention layers + recurrent
        // shortconv): llama.cpp can only wipe its recurrent state entirely,
        // not truncate it to a midpoint — a PARTIAL `llama_memory_seq_rm`
        // returns false and removes nothing (first probe of this feature
        // died exactly there: stale cells + new batch → rc=-1).
        //   Tier 1 — pure append (lcp == cachedTokens.count): the new prompt
        //   strictly extends what's in the KV; nothing to remove, valid for
        //   hybrids too. The common case, because ChatSession rebuilds
        //   transcripts byte-for-byte and ChatML's special-token markers stop
        //   BPE merges from crossing turn boundaries.
        //   Tier 2 — divergence: try the partial rm (pure-attention models
        //   accept it); if it returns false, fall back to a FULL wipe +
        //   full prefill — slower, never wrong.
        if lcp == cachedTokens.count, lcp > 0 {
            // Pure append — KV already holds exactly tokens[0..<lcp].
            cacheMode = "append"
        } else if let mem = llama_get_memory(ctx) {
            if lcp > 0, llama_memory_seq_rm(mem, 0, Int32(lcp), -1) {
                // Partial truncation accepted (pure-attention model).
                cacheMode = "truncate"
            } else if let restored = restoreRecoveryCheckpoint(
                ctx: ctx, prompt: tokens, liveTokens: cachedTokens)
            {
                lcp = restored
                cacheMode = "recovery-checkpoint"
            } else {
                _ = llama_memory_seq_rm(mem, 0, 0, -1)
                recoveryCheckpoint = nil
                lcp = 0
                cacheMode = lcpCandidate > 0 ? "reset-after-divergence" : "reset"
            }
        }
        log.notice("generate: \(tokens.count) prompt tokens · KV reuse \(lcp) · prefill \(tokens.count - lcp)")
        // OSLog from unsigned CLI processes is hard to retrieve with `log
        // show`; the Mac probes set MCPZIM_KV_DEBUG=1 to see reuse stats on
        // stderr directly.
        if ProcessInfo.processInfo.environment["MCPZIM_KV_DEBUG"] == "1" {
            FileHandle.standardError.write(Data(
                "[kv] prompt=\(tokens.count) reuse=\(lcp) prefill=\(tokens.count - lcp)\n".utf8))
        }
        // Pessimistic until the prefill lands: if we throw mid-prefill the
        // KV holds a partial suffix; an empty mirror forces the next turn to
        // lcp=0 → full seq_rm(0,…) wipe → consistent state.
        cachedTokens = []
        var prefillComplete = false
        defer {
            if !prefillComplete { recoveryCheckpoint = nil }
        }

        // Prefill via manual chunking — llama.cpp b8911 asserts
        // `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` in
        // llama-context.cpp:1599, so llama_batch_get_one(whole_prompt)
        // aborts for any prompt > n_batch. Confirmed on Mac probe
        // 2026-04-24 (MCPZimEvalCLI --probe-llama): the SAME abort
        // that was killing iOS silently. Split within the 512-token capacity
        // (128 on phone Bonsai to keep hot GPU submissions shorter)
        // and call llama_decode per chunk. Only set logits=true on
        // the final token of the entire prompt — llama.cpp accepts
        // intermediate batches with logits=false as pure KV-populate.
        let nBatch = generationPrefillBatchSize
        var batch = llama_batch_init(Int32(nBatch), 0, 1)
        defer { llama_batch_free(batch) }
        var pos: Int32 = Int32(lcp)
        var i = lcp
        var prefillBatches = 0
        let prefillTokens = tokens.count - lcp
        let prefillStarted = ProcessInfo.processInfo.systemUptime
        // Capture immediately before the final prompt token, so even an
        // exact retry has one fresh token from which to obtain logits.
        let recoveryBoundary = tokens.count - 1
        let captureRecovery = recoveryCheckpointEnabled && tokens.count >= 256
            && model.map { llama_model_is_hybrid($0) || llama_model_is_recurrent($0) } == true
        while i < tokens.count {
            if generationIsCancelled(generationID) { throw CancellationError() }
            let end = captureRecovery && i < recoveryBoundary
                ? min(i + nBatch, recoveryBoundary)
                : min(i + nBatch, tokens.count)
            batch.n_tokens = 0
            for j in i..<end {
                let isFinalOfPrompt = (j == tokens.count - 1)
                Self.batchAdd(
                    &batch, token: tokens[j], pos: pos,
                    seqIds: Self.defaultSequenceIDs, logits: isFinalOfPrompt)
                pos += 1
            }
            let rc = llama_decode(ctx, batch)
            if rc != 0 {
                throw LlamaCppError.custom(
                    "llama_decode prefill rc=\(rc) at batch \(i)..<\(end) of \(tokens.count)")
            }
            i = end
            prefillBatches += 1
            if captureRecovery, i == recoveryBoundary {
                captureRecoveryCheckpoint(ctx: ctx, tokens: Array(tokens.prefix(i)))
            }
        }
        // Metal decode can return before execution finishes. Attribute its
        // synchronization to prefill, rather than hiding it in first sample.
        llama_synchronize(ctx)
        let prefillFinished = ProcessInfo.processInfo.systemUptime
        let prefillSeconds = prefillFinished - prefillStarted
        let prefillRate = prefillSeconds > 0
            ? Double(prefillTokens) / prefillSeconds : 0
        // Prefill complete — the KV now holds exactly `tokens`.
        cachedTokens = tokens
        prefillComplete = true
        let prefillMemory = MemoryStats.physFootprintMB()
        debug(String(format:
            "perf prefill · prompt=%d tok · LCP candidate=%d · reused=%d · mode=%@ · batches=%d · %.3fs · %.1f tok/s · footprint=%.0f MB (Δ%+.0f)",
            tokens.count, lcpCandidate, lcp, cacheMode, prefillBatches,
            prefillSeconds, prefillRate, prefillMemory,
            prefillMemory - memoryStarted))

        benchmarkSSDStateCacheIfRequested(ctx: ctx, tokens: tokens)

        // Sampler chain. Use a publisher/model-specific profile when one is
        // installed; otherwise preserve the caller's task-level parameters.
        // `temp=0.0` → greedy.
        let modelProfile = parameters.useModelSamplingProfile
            ? samplingProfile : nil
        let temperature = modelProfile?.temperature ?? parameters.temperature
        let topP = modelProfile?.topP ?? parameters.topP
        let topK = modelProfile?.topK ?? parameters.topK
        let presencePenalty = modelProfile?.presencePenalty ?? 0
        debug(String(format: "sampler · temp=%.2f · top-p=%.2f · top-k=%d · presence=%.2f",
                     temperature, topP, topK, presencePenalty))
        let sp = llama_sampler_chain_init(llama_sampler_chain_default_params())
        defer { llama_sampler_free(sp) }
        if temperature <= 0 {
            llama_sampler_chain_add(sp, llama_sampler_init_greedy())
        } else {
            if presencePenalty != 0 {
                #if MCPZIM_LLAMA_UPSTREAM
                // b10816 adds n_vocab and requires an explicit history size.
                // Only the isolated upstream A/B build sets this condition.
                llama_sampler_chain_add(sp, llama_sampler_init_penalties(
                    llama_vocab_n_tokens(vocab), Int32(contextTokens),
                    1.0, 0.0, Float(presencePenalty)))
                #else
                llama_sampler_chain_add(sp, llama_sampler_init_penalties(
                    -1, 1.0, 0.0, Float(presencePenalty)))
                #endif
            }
            if topK > 0 {
                llama_sampler_chain_add(sp, llama_sampler_init_top_k(Int32(topK)))
            }
            llama_sampler_chain_add(sp, llama_sampler_init_top_p(
                Float(topP), 1))
            llama_sampler_chain_add(sp, llama_sampler_init_temp(
                Float(temperature)))
            llama_sampler_chain_add(sp, llama_sampler_init_dist(
                samplingSeed))
        }

        // Decode loop. Sample → detokenise → emit → feed back.
        var newTokens = 0
        var sampledTokens = 0
        var firstVisibleSeconds: Double?
        var firstVisibleAt: TimeInterval?
        var stopReason = "max_tokens"
        let decodeStarted = ProcessInfo.processInfo.systemUptime
        let maxTokens = parameters.maxTokens
        // Rolling tail for stop-sequence detection: a stop marker completed
        // by this piece must END inside it, so only the last few dozen
        // characters ever need scanning — the old whole-buffer `contains`
        // re-scanned the entire accumulated reply on every sampled token
        // (O(n²) over a long grounded answer). The tail keeps at least
        // 2×maxStopLen characters, so a marker spanning pieces still lands
        // fully inside it.
        let maxStopLen = parameters.stopSequences.map(\.count).max() ?? 0
        let stopTailCap = max(128, maxStopLen * 2)
        var stopTail = ""
        // Reused detokenise scratch — allocating a fresh 64-byte buffer
        // (plus a mapped array) per token was pure churn on the hottest
        // loop.
        var pieceBuf = [UInt8](repeating: 0, count: 64)
        while newTokens < maxTokens {
            if generationIsCancelled(generationID) { throw CancellationError() }
            let id = llama_sampler_sample(sp, ctx, -1)
            if llama_vocab_is_eog(vocab, id) {
                stopReason = "eog"
                break
            }
            sampledTokens += 1
            // Detokenise the piece — llama.cpp returns raw bytes; callers
            // expect UTF-8 strings. Occasional partial multi-byte chars
            // are fine; they round-trip on the next chunk.
            let n = pieceBuf.withUnsafeMutableBufferPointer { buf in
                buf.baseAddress!.withMemoryRebound(
                    to: CChar.self, capacity: buf.count
                ) { p in
                    llama_token_to_piece(
                        vocab, id, p, Int32(buf.count),
                        /*lstrip*/ 0, /*special*/ false)
                }
            }
            if n > 0 {
                let piece = String(decoding: pieceBuf[0..<Int(n)], as: UTF8.self)
                if maxStopLen > 0 {
                    stopTail += piece
                    if stopTail.count > stopTailCap * 2 {
                        stopTail = String(stopTail.suffix(stopTailCap))
                    }
                }
                if firstVisibleSeconds == nil {
                    let now = ProcessInfo.processInfo.systemUptime
                    firstVisibleAt = now
                    firstVisibleSeconds = now - requestStarted
                    debug(String(format:
                        "perf first token · TTFT=%.3fs · after-prefill=%.3fs",
                        firstVisibleSeconds!, now - prefillFinished))
                }
                emit(piece)
            }
            // Check stop sequences. Post-emit so we don't clip the
            // stop marker on the caller side — mirrors how
            // Gemma4Provider watches for `<turn|>`.
            if maxStopLen > 0,
               parameters.stopSequences.contains(where: {
                   stopTail.contains($0)
               })
            {
                stopReason = "stop_sequence"
                break
            }
            // Feed the new token back for the next decode.
            batch.n_tokens = 0
            Self.batchAdd(
                &batch, token: id, pos: pos,
                seqIds: Self.defaultSequenceIDs, logits: true)
            pos += 1
            if llama_decode(ctx, batch) != 0 {
                // A failed backend submission may have partially mutated
                // attention/recurrent state. Neither its mirror nor the
                // partial checkpoint is safe to reuse on the next request.
                cachedTokens = []
                recoveryCheckpoint = nil
                throw LlamaCppError.custom("llama_decode step failed at pos=\(pos)")
            }
            // Mirror what's actually in the KV so next turn's LCP can reuse
            // it. (A sampled token we break on BEFORE decoding — EOG, stop
            // sequence — is deliberately NOT appended: it never entered KV.)
            cachedTokens.append(id)
            newTokens += 1
        }
        let requestFinished = ProcessInfo.processInfo.systemUptime
        let decodeSeconds = requestFinished - decodeStarted
        let decodeRate = decodeSeconds > 0
            ? Double(sampledTokens) / decodeSeconds : 0
        let steadyDecodeSeconds = firstVisibleAt.map {
            max(0, requestFinished - $0)
        } ?? 0
        let steadyDecodeTokens = max(0, sampledTokens - 1)
        let steadyDecodeRate = steadyDecodeSeconds > 0
            ? Double(steadyDecodeTokens) / steadyDecodeSeconds : 0
        let ttft = firstVisibleSeconds ?? -1
        let memoryFinished = MemoryStats.physFootprintMB()
        debug(String(format:
            "perf complete · output=%d tok · generation=%.3fs/%.2f tok/s · steady-decode=%d tok/%.3fs/%.2f tok/s · TTFT=%.3fs · total=%.3fs · stop=%@ · thermal=%@ · footprint=%.0f MB (Δ%+.0f)",
            sampledTokens, decodeSeconds, decodeRate,
            steadyDecodeTokens, steadyDecodeSeconds, steadyDecodeRate, ttft,
            requestFinished - requestStarted, stopReason,
            Self.thermalStateLabel(), memoryFinished,
            memoryFinished - memoryStarted))
        let completedStats = GenerationStats(
            runtime: "llamacpp", modelID: id,
            promptTokens: tokens.count, reusedTokens: lcp,
            prefillSeconds: prefillSeconds, ttftSeconds: ttft,
            outputTokens: sampledTokens,
            decodeTokensPerSecond: steadyDecodeRate,
            totalSeconds: requestFinished - requestStarted,
            peakFootprintMB: max(prefillMemory, memoryFinished),
            stopReason: stopReason)
        generationStatsLock.lock()
        storedGenerationStats = completedStats
        generationStatsLock.unlock()
        log.notice("generate: \(newTokens) new tokens")
    }

    // MARK: - Helpers

    #if DEBUG
    /// Target-only cost curve, not a speculative decoder. All rows request
    /// logits as real verification would. Snapshot/restore and drafting costs
    /// are outside the timed interval, so ratios are optimistic lower bounds.
    public func benchmarkVerificationWidths(prompt: String) async throws {
        try await Task.detached { [self] in
            modelLock.lock()
            defer { modelLock.unlock() }
            guard let ctx, let vocab, let model, llama_model_is_hybrid(model),
                  let mem = llama_get_memory(ctx) else { return }
            defer {
                _ = llama_memory_seq_rm(mem, 0, 0, -1)
                cachedTokens = []
                recoveryCheckpoint = nil
            }
            let tokens = Self.tokenize(vocab: vocab, prompt: prompt)
            guard !tokens.isEmpty, tokens.count + 8 < contextTokens else { return }
            _ = llama_memory_seq_rm(mem, 0, 0, -1)
            cachedTokens = []
            recoveryCheckpoint = nil
            var batch = llama_batch_init(512, 0, 1)
            defer { llama_batch_free(batch) }
            let prefillBatch = generationPrefillBatchSize
            for start in stride(from: 0, to: tokens.count, by: prefillBatch) {
                batch.n_tokens = 0
                for j in start..<min(start + prefillBatch, tokens.count) {
                    Self.batchAdd(&batch, token: tokens[j], pos: Int32(j),
                        seqIds: Self.defaultSequenceIDs, logits: j == tokens.count - 1)
                }
                guard llama_decode(ctx, batch) == 0 else {
                    throw LlamaCppError.custom("verification curve prefill failed")
                }
            }
            captureRecoveryCheckpoint(ctx: ctx, tokens: tokens)
            guard let checkpoint = recoveryCheckpoint else {
                throw LlamaCppError.custom("verification curve checkpoint unavailable")
            }
            let suffix = Array(tokens.suffix(8))
            guard suffix.count == 8 else { return }
            for trial in 0..<4 {
                // Reverse alternate sweeps to reduce warm/thermal order bias.
                let widths = trial % 2 == 0 ? [1, 2, 4, 8] : [8, 4, 2, 1]
                for width in widths {
                    let restored = checkpoint.data.withUnsafeBufferPointer {
                        llama_state_seq_set_data_ext(ctx, $0.baseAddress, $0.count, 0, checkpoint.flags)
                    }
                    guard restored == checkpoint.data.count,
                          llama_memory_seq_rm(mem, 0, Int32(tokens.count), -1) else {
                        throw LlamaCppError.custom("verification curve restore failed")
                    }
                    batch.n_tokens = 0
                    for j in 0..<width {
                        Self.batchAdd(&batch, token: suffix[j], pos: Int32(tokens.count + j),
                            seqIds: Self.defaultSequenceIDs, logits: true)
                    }
                    llama_synchronize(ctx)
                    let started = ProcessInfo.processInfo.systemUptime
                    let rc = llama_decode(ctx, batch)
                    llama_synchronize(ctx)
                    guard rc == 0 else {
                        throw LlamaCppError.custom("verification curve decode failed")
                    }
                    debug(String(format:
                        "verify curve · trial=%d · width=%d · seconds=%.6f · thermal=%@ · warmup=%@",
                        trial, width, ProcessInfo.processInfo.systemUptime - started,
                        Self.thermalStateLabel(), trial == 0 ? "yes" : "no"))
                }
            }
        }.value
    }
    #endif

    private func captureRecoveryCheckpoint(ctx: OpaquePointer, tokens: [Int32]) {
        let queuedAt = ProcessInfo.processInfo.systemUptime
        llama_synchronize(ctx)
        let started = ProcessInfo.processInfo.systemUptime
        let queuedPrefillSeconds = started - queuedAt
        let tensorBytes = llama_state_seq_get_size_ext(
            ctx, 0, UInt32(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY))
        // Bound the real state, not just optional on-device metadata. Host
        // buffers are reused across captures and released on reset/unload.
        guard tensorBytes > 0, tensorBytes <= 256 * 1_048_576 else {
            recoveryCheckpoint = nil
            debug("recovery capture skipped · state=\(tensorBytes) bytes exceeds budget")
            return
        }
        let flags = recoveryFlags
        let bytes = llama_state_seq_get_size_ext(ctx, 0, flags)
        guard bytes > 0, bytes <= 256 * 1_048_576 else {
            recoveryCheckpoint = nil
            return
        }
        // A new ON_DEVICE capture invalidates the previous handle even if
        // this operation fails. Never leave that old handle reachable.
        var data = recoveryCheckpoint?.data ?? []
        recoveryCheckpoint = nil
        if data.count != bytes { data = [UInt8](repeating: 0, count: bytes) }
        let saved = data.withUnsafeMutableBufferPointer {
            llama_state_seq_get_data_ext(ctx, $0.baseAddress, $0.count, 0, flags)
        }
        llama_synchronize(ctx)
        guard saved == bytes else {
            debug("recovery capture failed · \(saved)/\(bytes) bytes")
            return
        }
        recoveryCheckpoint = (tokens, data, flags)
        debug(String(format:
            "recovery captured · tokens=%d · tensors=%.1f MB · data=%d bytes · %.3fs · queued-prefill=%.3fs · flags=%d",
            tokens.count, Double(tensorBytes) / 1_048_576, bytes,
            ProcessInfo.processInfo.systemUptime - started, queuedPrefillSeconds, flags))
    }

    private func restoreRecoveryCheckpoint(
        ctx: OpaquePointer, prompt: [Int32], liveTokens: [Int32]
    ) -> Int? {
        guard recoveryCheckpointEnabled, let checkpoint = recoveryCheckpoint,
              let count = PromptRecoveryPolicy.reusableTokenCount(
                prompt: prompt, liveTokens: liveTokens,
                checkpointTokens: checkpoint.tokens),
              let mem = llama_get_memory(ctx)
        else { return nil }
        let started = ProcessInfo.processInfo.systemUptime
        llama_synchronize(ctx)
        let restored = checkpoint.data.withUnsafeBufferPointer {
            llama_state_seq_set_data_ext(ctx, $0.baseAddress, $0.count, 0, checkpoint.flags)
        }
        // Rewind recurrent state first; then the hybrid memory can remove
        // the attention tail. Any failure is handled by the caller's full reset.
        guard restored == checkpoint.data.count,
              llama_memory_seq_rm(mem, 0, Int32(count), -1)
        else {
            recoveryCheckpoint = nil
            debug("recovery restore failed; rebuilding full prompt")
            return nil
        }
        llama_synchronize(ctx)
        debug(String(format: "recovery restored · tokens=%d · %.3fs",
            count, ProcessInfo.processInfo.systemUptime - started))
        return count
    }

    /// llama.cpp's `--prompt-cache` is a serialization of the complete
    /// context, not a transparent SSD tier. Keep this as an opt-in DEBUG
    /// experiment: it measures the exact file size plus save/restore latency
    /// on the target device, then deletes the temporary snapshot. Normal
    /// conversation turns never touch disk.
    private func benchmarkSSDStateCacheIfRequested(
        ctx: OpaquePointer, tokens: [Int32]
    ) {
        #if DEBUG
        guard !didBenchmarkStateCache,
              ProcessInfo.processInfo.environment["MCPZIM_BENCH_STATE_CACHE"] == "1",
              !tokens.isEmpty
        else { return }
        didBenchmarkStateCache = true

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcpzim-llama-state-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        let sizeStarted = ProcessInfo.processInfo.systemUptime
        let advertisedBytes = llama_state_get_size(ctx)
        let sizeSeconds = ProcessInfo.processInfo.systemUptime - sizeStarted

        let saveStarted = ProcessInfo.processInfo.systemUptime
        let saved = url.path.withCString { path in
            tokens.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return false }
                return llama_state_save_file(ctx, path, base, buf.count)
            }
        }
        let saveSeconds = ProcessInfo.processInfo.systemUptime - saveStarted
        let fileBytes = ((try? FileManager.default.attributesOfItem(
            atPath: url.path)[.size]) as? NSNumber)?.uint64Value ?? 0

        var restoredTokens = [Int32](repeating: 0, count: contextTokens)
        var restoredCount = 0
        let loadStarted = ProcessInfo.processInfo.systemUptime
        let loaded = saved && url.path.withCString { path in
            restoredTokens.withUnsafeMutableBufferPointer { buf in
                guard let base = buf.baseAddress else { return false }
                return llama_state_load_file(
                    ctx, path, base, buf.count, &restoredCount)
            }
        }
        let loadSeconds = ProcessInfo.processInfo.systemUptime - loadStarted
        let tokenMatch = loaded && restoredCount == tokens.count
            && Array(restoredTokens.prefix(restoredCount)) == tokens

        debug(String(format:
            "SSD state benchmark · state=%.1f MB (size %.3fs) · file=%.1f MB · save=%.3fs · load=%.3fs · tokens=%d/%d match=%@",
            Double(advertisedBytes) / 1_048_576,
            sizeSeconds,
            Double(fileBytes) / 1_048_576,
            saveSeconds,
            loadSeconds,
            restoredCount,
            tokens.count,
            tokenMatch ? "yes" : "NO"))
        #endif
    }

    private func debug(_ message: String) {
        if let debugSink {
            debugSink(message)
        } else {
            log.notice("\(message, privacy: .public)")
        }
    }

    private static func thermalStateLabel() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func fileSizeIfPresent(_ url: URL) -> UInt64 {
        ((try? FileManager.default.attributesOfItem(
            atPath: url.path)[.size]) as? NSNumber)?.uint64Value ?? 0
    }

    private static func tokenize(vocab: OpaquePointer, prompt: String) -> [llama_token] {
        let utf8 = Array(prompt.utf8)
        let nMax = Int32(utf8.count + 8)
        var buf = [llama_token](repeating: 0, count: Int(nMax))
        let n = utf8.withUnsafeBufferPointer { inBuf -> Int32 in
            inBuf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: utf8.count) { cstr in
                buf.withUnsafeMutableBufferPointer { outBuf in
                    llama_tokenize(
                        vocab, cstr, Int32(utf8.count),
                        outBuf.baseAddress, Int32(outBuf.count),
                        /*add_special*/ true, /*parse_special*/ true)
                }
            }
        }
        if n < 0 { return [] }
        return Array(buf.prefix(Int(n)))
    }

    /// Helper lifted from llama.cpp/examples/llama.swiftui — one
    /// token per call with a variable-length seq_ids list. llama.cpp's
    /// C struct exposes these as parallel arrays; we append at the
    /// current `batch.n_tokens` and bump it.
    private static func batchAdd(
        _ batch: inout llama_batch,
        token: llama_token,
        pos: llama_pos,
        seqIds: [llama_seq_id],
        logits: Bool
    ) {
        let i = Int(batch.n_tokens)
        batch.token[i] = token
        batch.pos[i] = pos
        batch.n_seq_id[i] = Int32(seqIds.count)
        for (k, sid) in seqIds.enumerated() {
            batch.seq_id[i]![k] = sid
        }
        batch.logits[i] = logits ? 1 : 0
        batch.n_tokens += 1
    }

    public func formatTranscript(systemPreamble: String, turns: [ChatTurn]) -> String {
        // Defer to the injected template (Gemma3Template by default)
        // — same pattern Gemma4Provider uses so a swap to Qwen
        // GGUF just swaps the template.
        template.renderTranscript(
            systemPreamble: systemPreamble, tools: [], turns: turns)
    }
}

private enum LlamaCppError: Error {
    case notLoaded
    case custom(String)
}

#else

/// Fallback stub — kept so the app still compiles if the
/// LlamaCppSwift package isn't linked (e.g. an older project.yml).
/// The provider reports failed state and never drives generation.
public final class LlamaCppProvider: ModelProvider, @unchecked Sendable {
    public let id: String
    public let displayName: String
    public let approximateMemoryMB: Int
    public let supportsToolCalls = true
    public let template: any ModelTemplate
    public var debugSink: (@Sendable (String) -> Void)?
    public init(
        id: String = "llamacpp-unlinked",
        displayName: String = "llama.cpp (unlinked)",
        approximateMemoryMB: Int = 0,
        template: any ModelTemplate = Gemma3Template()
    ) {
        self.id = id
        self.displayName = displayName
        self.approximateMemoryMB = approximateMemoryMB
        self.template = template
    }
    /// Model-sharing surface (matches the real provider; nothing to share
    /// or adopt when llama.cpp isn't linked).
    public var shareableGGUFURL: URL? { nil }
    public func adoptSharedGGUF(at url: URL) -> Bool { false }
    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { $0.finish() }
    }
    public func load() async throws {
        throw NSError(
            domain: "LlamaCpp", code: -1,
            userInfo: [NSLocalizedDescriptionKey: "LlamaCppSwift not linked"])
    }
    public func unload() async {}
    public func generate(
        prompt: String, parameters: GenerationParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream {
            $0.finish(throwing: NSError(
                domain: "LlamaCpp", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "LlamaCppSwift not linked"]))
        }
    }
}

#endif
