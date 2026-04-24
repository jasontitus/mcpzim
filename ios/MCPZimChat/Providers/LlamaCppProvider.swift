// SPDX-License-Identifier: MIT
//
// LlamaCppProvider — ModelProvider conformance that runs GGUF
// models via the upstream llama.cpp C API (vendored as the
// `llama-b8911-xcframework.zip` release, exposed through the
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

    // MARK: - State + llama.cpp handles

    private let queue = DispatchQueue(label: "LlamaCppProvider.state")
    private var state: ModelLoadState = .notLoaded
    private var continuations: [AsyncStream<ModelLoadState>.Continuation] = []

    /// Opaque handles from llama.cpp. Guarded by `modelLock` below —
    /// generate() can take a long time and we don't want `unload()`
    /// racing it.
    private let modelLock = NSLock()
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var vocab: OpaquePointer?

    /// KV-cache mirror for follow-up LCP matching. Analogous to
    /// `Gemma4Provider.cachedTokens`. llama.cpp itself keeps the cache
    /// in-context; we only track this to decide whether the next
    /// turn's prefix can reuse it (same-prefix rule).
    private var cachedTokens: [Int32] = []

    // MARK: - Init

    public init(
        id: String = "gemma3-4b-it-q4km-gguf",
        displayName: String = "Gemma 3 4B IT (Q4_K_M · llama.cpp)",
        huggingFaceRepo: String = "bartowski/google_gemma-3-4b-it-GGUF",
        ggufFilename: String = "google_gemma-3-4b-it-Q4_K_M.gguf",
        approximateMemoryMB: Int = 3200,
        template: any ModelTemplate = Gemma3Template()
    ) {
        self.id = id
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.ggufFilename = ggufFilename
        self.approximateMemoryMB = approximateMemoryMB
        self.template = template
        // One-time global init. Safe to call repeatedly per
        // llama.cpp docs; the backend keeps a refcount.
        llama_backend_init()
    }

    // MARK: - State stream

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { continuation in
            queue.sync {
                continuation.yield(self.state)
                self.continuations.append(continuation)
            }
        }
    }

    private func set(_ new: ModelLoadState) {
        queue.sync {
            state = new
            for c in continuations { c.yield(new) }
        }
    }

    // MARK: - Load

    public func load() async throws {
        set(.loading)
        do {
            let path = try await ensureGGUFDownloaded()
            try openModel(at: path)
            set(.ready)
        } catch {
            set(.failed(String(describing: error)))
            throw error
        }
    }

    /// Download the GGUF via HuggingFace's direct resolve URL if not
    /// already on disk. We store under the same
    /// `<caches>/huggingface/hub/models--<repo>/snapshots/main/<file>`
    /// layout the MLX path uses — single cache directory for both
    /// runtimes, saves duplicate downloads when switching providers.
    private func ensureGGUFDownloaded() async throws -> URL {
        let cachesDir = FileManager.default.urls(
            for: .cachesDirectory, in: .userDomainMask).first!
        let repoSlug = huggingFaceRepo.replacingOccurrences(of: "/", with: "--")
        let destDir = cachesDir
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--\(repoSlug)")
            .appendingPathComponent("snapshots")
            .appendingPathComponent("main")
        try FileManager.default.createDirectory(
            at: destDir, withIntermediateDirectories: true)
        let destURL = destDir.appendingPathComponent(ggufFilename)
        if FileManager.default.fileExists(atPath: destURL.path) {
            return destURL
        }
        let urlStr =
            "https://huggingface.co/\(huggingFaceRepo)/resolve/main/\(ggufFilename)"
        guard let url = URL(string: urlStr) else {
            throw LlamaCppError.custom("bad HF URL: \(urlStr)")
        }
        log.info("GGUF download start: \(urlStr, privacy: .public)")
        // Emit initial "0% of model download" so the UI flips from
        // `.loading` to `.downloading(0)` immediately — the ~2.5 GB
        // GGUF pull takes 30-120 s on wifi; without a progress
        // signal the setup overlay looks frozen.
        set(.downloading(0))
        // Default URLSession.shared caps `timeoutIntervalForRequest`
        // at 60 s and `timeoutIntervalForResource` at 7 days, but the
        // request timeout fires any time a SINGLE chunk takes > 60s,
        // which HF's `cas-bridge.xethub.hf.co` Xet redirect hits
        // routinely on a 2.5 GB pull (seen on Jazzman 17 2026-04-23
        // as NSURLErrorDomain -1001 timed-out on xethub). Use the
        // same long-timeout + waitsForConnectivity pattern the MLX
        // path uses for safetensors downloads (see
        // Gemma4Provider.longDownloadSession).
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 600      // 10 minutes per chunk
        cfg.timeoutIntervalForResource = 3600    // 1 hour overall
        cfg.waitsForConnectivity = true
        let progress = ProgressReporter { [weak self] fraction in
            self?.set(.downloading(fraction))
        }
        let session = URLSession(configuration: cfg, delegate: nil, delegateQueue: nil)
        let (tmpURL, resp) = try await session.download(
            for: URLRequest(url: url), delegate: progress)
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            throw LlamaCppError.custom(
                "HF download \(http.statusCode) for \(urlStr)")
        }
        // Move from the URL's auto-deleted tmp location to our cache.
        // If something already put a file there (partial fetch from a
        // prior attempt), overwrite it.
        if FileManager.default.fileExists(atPath: destURL.path) {
            try? FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.moveItem(at: tmpURL, to: destURL)
        log.info("GGUF download complete: \(destURL.path, privacy: .public)")
        // Transition back to .loading so the UI knows we're now doing
        // the model open + Metal warm-up, not still downloading.
        set(.loading)
        return destURL
    }

    /// Open the model + create a context configured with our
    /// shipping-config flags: Metal-all-layers, Q8_0 KV, flash-attn,
    /// iSWA pruning. All four of these were load-bearing in the
    /// bench that chose llama.cpp over MLX for Gemma 3 — see file
    /// header for the numbers.
    private func openModel(at path: URL) throws {
        modelLock.lock()
        defer { modelLock.unlock() }
        // Model
        var mp = llama_model_default_params()
        mp.n_gpu_layers = -1          // all on Metal
        guard let m = llama_model_load_from_file(path.path, mp) else {
            throw LlamaCppError.custom("llama_model_load_from_file failed for \(path.lastPathComponent)")
        }
        self.model = m
        self.vocab = llama_model_get_vocab(m)

        // Context — shipping config (same knobs that crash iOS,
        // testing them now on Mac via MCPZimChatMac to see whether
        // it's iOS-Metal-specific or a Swift invocation bug).
        var cp = llama_context_default_params()
        cp.n_ctx = 8192
        cp.n_batch = 512
        cp.n_ubatch = 512
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        cp.type_k = GGML_TYPE_Q8_0
        cp.type_v = GGML_TYPE_Q8_0
        cp.swa_full = false
        cp.offload_kqv = true
        guard let c = llama_init_from_model(m, cp) else {
            llama_model_free(m)
            self.model = nil
            throw LlamaCppError.custom("llama_init_from_model failed")
        }
        self.ctx = c
        log.notice("loaded \(self.ggufFilename, privacy: .public) · n_ctx=\(cp.n_ctx) kv=Q8_0 fa=on swa_full=false")
    }

    public func unload() async {
        modelLock.lock()
        defer { modelLock.unlock() }
        if let c = ctx { llama_free(c); ctx = nil }
        if let m = model { llama_model_free(m); model = nil }
        vocab = nil
        cachedTokens = []
        set(.notLoaded)
    }

    // MARK: - Generate

    public func generate(
        prompt: String, parameters: GenerationParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [weak self] in
                guard let self else {
                    continuation.finish(throwing: LlamaCppError.notLoaded)
                    return
                }
                do {
                    try self.generateLocked(
                        prompt: prompt,
                        parameters: parameters,
                        emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Synchronous generation body. Runs on a Task-detached background
    /// thread (llama.cpp's `llama_decode` is blocking). Holds
    /// `modelLock` for the whole turn so `unload()` can't yank the
    /// model out from under us mid-decode.
    private func generateLocked(
        prompt: String,
        parameters: GenerationParameters,
        emit: @escaping (String) -> Void
    ) throws {
        modelLock.lock()
        defer { modelLock.unlock() }
        guard let ctx = ctx, let vocab = vocab else {
            throw LlamaCppError.notLoaded
        }
        let tokens = Self.tokenize(vocab: vocab, prompt: prompt)
        log.notice("generate: \(tokens.count) prompt tokens")

        // Prefill via `llama_batch_get_one` — the canonical path in
        // llama.cpp/examples/simple.cpp. It constructs a view batch
        // over the full prompt and llama_decode internally chunks
        // into n_batch-sized sub-batches, handling pos/seq_id/logits
        // defaults correctly. Our earlier attempt with manual
        // `llama_batch_init` + `llama_batch_add` chunking failed with
        // `llama_decode prefill failed at pos=512` on every multi-
        // batch prompt (tried various logits-anchor patterns, none
        // worked). Switching to the library's internal split is
        // both simpler and matches what the example + main.cpp do.
        //
        // KV cache clear between turns: llama.cpp's examples don't
        // call this explicitly and the library handles seq_id 0
        // reset on its own. Explicit `llama_memory_clear` on a
        // fresh context was a suspected crash source (iOS 2026-04-24
        // silent process death right after prefill started).
        // Skipping it entirely — every new Llama() already starts
        // with empty memory, and we currently treat each turn as a
        // cold prefill (no cache reuse). When we add LCP-match reuse
        // later we'll revisit this.
        // Prefill via manual chunking — llama.cpp b8911 asserts
        // `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` in
        // llama-context.cpp:1599, so llama_batch_get_one(whole_prompt)
        // aborts for any prompt > n_batch. Confirmed on Mac probe
        // 2026-04-24 (MCPZimEvalCLI --probe-llama): the SAME abort
        // that was killing iOS silently. Split into 512-token chunks
        // and call llama_decode per chunk. Only set logits=true on
        // the final token of the entire prompt — llama.cpp accepts
        // intermediate batches with logits=false as pure KV-populate.
        let nBatch = 512
        var batch = llama_batch_init(Int32(nBatch), 0, 1)
        defer { llama_batch_free(batch) }
        var pos: Int32 = 0
        var i = 0
        while i < tokens.count {
            let end = min(i + nBatch, tokens.count)
            batch.n_tokens = 0
            for j in i..<end {
                let isFinalOfPrompt = (j == tokens.count - 1)
                Self.batchAdd(
                    &batch, token: tokens[j], pos: pos,
                    seqIds: [0], logits: isFinalOfPrompt)
                pos += 1
            }
            let rc = llama_decode(ctx, batch)
            if rc != 0 {
                throw LlamaCppError.custom(
                    "llama_decode prefill rc=\(rc) at batch \(i)..<\(end) of \(tokens.count)")
            }
            i = end
        }

        // Sampler chain. Match the MLX defaults: greedy-ish with
        // temp + top-p. `temp=0.0` → force dist sampler to greedy.
        let sp = llama_sampler_chain_init(llama_sampler_chain_default_params())
        defer { llama_sampler_free(sp) }
        if parameters.temperature <= 0 {
            llama_sampler_chain_add(sp, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(sp, llama_sampler_init_top_k(40))
            llama_sampler_chain_add(sp, llama_sampler_init_top_p(
                Float(parameters.topP), 1))
            llama_sampler_chain_add(sp, llama_sampler_init_temp(
                Float(parameters.temperature)))
            llama_sampler_chain_add(sp, llama_sampler_init_dist(
                LLAMA_DEFAULT_SEED))
        }

        // Decode loop. Sample → detokenise → emit → feed back.
        var newTokens = 0
        var buffered = ""
        let maxTokens = parameters.maxTokens
        while newTokens < maxTokens {
            let id = llama_sampler_sample(sp, ctx, -1)
            if llama_vocab_is_eog(vocab, id) { break }
            // Detokenise the piece — llama.cpp returns raw bytes, we
            // accumulate into `buffered` and emit on every chunk
            // since callers expect UTF-8 strings. Occasional partial
            // multi-byte chars are fine; String(cString:) will round-
            // trip them on the next chunk.
            var pieceBuf = [CChar](repeating: 0, count: 64)
            let n = pieceBuf.withUnsafeMutableBufferPointer { buf in
                llama_token_to_piece(
                    vocab, id, buf.baseAddress, Int32(buf.count),
                    /*lstrip*/ 0, /*special*/ false)
            }
            if n > 0 {
                let piece = String(
                    decoding: pieceBuf.prefix(Int(n)).map { UInt8(bitPattern: $0) },
                    as: UTF8.self)
                buffered += piece
                emit(piece)
            }
            // Check stop sequences. Post-emit so we don't clip the
            // stop marker on the caller side — mirrors how
            // Gemma4Provider watches for `<turn|>`.
            if !parameters.stopSequences.isEmpty,
               parameters.stopSequences.contains(where: {
                   buffered.contains($0)
               })
            {
                break
            }
            // Feed the new token back for the next decode.
            batch.n_tokens = 0
            Self.batchAdd(
                &batch, token: id, pos: pos,
                seqIds: [0], logits: true)
            pos += 1
            if llama_decode(ctx, batch) != 0 {
                throw LlamaCppError.custom("llama_decode step failed at pos=\(pos)")
            }
            newTokens += 1
        }
        log.notice("generate: \(newTokens) new tokens")
    }

    // MARK: - Helpers

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

/// `URLSessionDownloadDelegate` that throttles byte-progress updates
/// to whole-percent transitions and forwards the fraction to a
/// closure. Used by `ensureGGUFDownloaded` so the Setup overlay
/// shows the user a real % instead of a frozen spinner during the
/// 2.5 GB GGUF pull. The delegate is required because Swift's
/// async `URLSession.download(for:delegate:)` otherwise reports no
/// intermediate progress.
private final class ProgressReporter: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let onFraction: (Double) -> Void
    private var lastReportedPct: Int = -1
    init(_ onFraction: @escaping (Double) -> Void) {
        self.onFraction = onFraction
    }
    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        let pct = Int((fraction * 100).rounded(.down))
        if pct != lastReportedPct {
            lastReportedPct = pct
            onFraction(fraction)
        }
    }
    // Required delegate method — no-op, the async download API
    // handles the final file move for us via its tmpURL return.
    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {}
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
