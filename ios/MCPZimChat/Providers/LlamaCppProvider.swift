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
        localGGUFPath: String? = nil,
        replyTokensFloor: Int? = nil,
        approximateMemoryMB: Int = 3200,
        contextTokens: Int = 8192,
        template: any ModelTemplate = Gemma3Template()
    ) {
        self.id = id
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.ggufFilename = ggufFilename
        self.localGGUFPath = localGGUFPath
        self.replyTokensFloor = replyTokensFloor
        self.approximateMemoryMB = approximateMemoryMB
        self.contextTokens = contextTokens
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
        // Emit "0%" immediately so the UI flips from `.loading` to
        // `.downloading(0)` before the first byte arrives.
        set(.downloading(0))

        // Rewritten 2026-06-10 after a real on-device failure: the old
        // single-shot download (waitsForConnectivity + 600 s chunk timeout,
        // no retry, no resume, progress gated on Content-Length) could sit
        // visually frozen for many minutes on a stall, then throw away a
        // nearly-complete multi-GB pull and land in `.failed`. New shape:
        //   • HEAD first → expected size, so progress is accurate even when
        //     HF's Xet CDN streams chunked (no Content-Length on the GET).
        //   • Short 60 s idle timeout (resets whenever bytes arrive) +
        //     waitsForConnectivity OFF → a stall surfaces in ≤60 s instead
        //     of silently hanging.
        //   • Up to 6 attempts with capped backoff, resuming via URLSession
        //     resume data — persisted to disk, so even an app relaunch
        //     continues where it left off instead of restarting from byte 0.
        //   • Final size validation, so a truncated file can never be
        //     half-loaded by llama.cpp.
        let resumeBlobURL = destDir.appendingPathComponent(ggufFilename + ".resume")
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60        // idle stall detector
        cfg.timeoutIntervalForResource = 86_400   // retries govern, not this
        cfg.waitsForConnectivity = false
        let session = URLSession(configuration: cfg, delegate: nil, delegateQueue: nil)

        // Expected byte size via HEAD (HF serves Content-Length or
        // x-linked-size for LFS files). 0 = unknown → reporter falls back.
        var expectedBytes: Int64 = 0
        var headReq = URLRequest(url: url)
        headReq.httpMethod = "HEAD"
        if let (_, headResp) = try? await session.data(for: headReq),
           let http = headResp as? HTTPURLResponse {
            if http.expectedContentLength > 0 {
                expectedBytes = http.expectedContentLength
            } else if let linked = http.value(forHTTPHeaderField: "x-linked-size"),
                      let n = Int64(linked) {
                expectedBytes = n
            }
        }
        let progress = ProgressReporter(expectedBytes: expectedBytes) { [weak self] fraction in
            self?.set(.downloading(fraction))
        }

        var resumeData: Data? = try? Data(contentsOf: resumeBlobURL)
        if resumeData != nil {
            log.info("GGUF download: found resume blob (\(resumeData!.count) bytes) from a previous attempt")
        }
        let maxAttempts = 6
        var lastError: Error = LlamaCppError.custom("download never attempted")
        var tmpURL: URL? = nil
        for attempt in 1...maxAttempts {
            do {
                let (got, resp): (URL, URLResponse)
                if let rd = resumeData {
                    (got, resp) = try await session.download(resumeFrom: rd, delegate: progress)
                } else {
                    (got, resp) = try await session.download(for: URLRequest(url: url), delegate: progress)
                }
                if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
                    throw LlamaCppError.custom("HF download HTTP \(http.statusCode) for \(urlStr)")
                }
                tmpURL = got
                break
            } catch {
                lastError = error
                // Salvage partial progress: URLSession attaches resume data
                // to the error when the server supports ranges (HF does).
                let ns = error as NSError
                if let rd = ns.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
                    resumeData = rd
                    try? rd.write(to: resumeBlobURL)   // survives relaunch
                } else {
                    // Stale/unusable resume state — start clean next attempt.
                    resumeData = nil
                    try? FileManager.default.removeItem(at: resumeBlobURL)
                }
                guard attempt < maxAttempts else { break }
                let backoff = min(30, 1 << attempt)    // 2,4,8,16,30,30 s
                log.warning("GGUF download attempt \(attempt)/\(maxAttempts) failed (\(ns.code)): \(ns.localizedDescription, privacy: .public) — retrying in \(backoff)s\(resumeData != nil ? " (resuming)" : " (from scratch)")")
                try? await Task.sleep(nanoseconds: UInt64(backoff) * 1_000_000_000)
            }
        }
        guard let finalTmp = tmpURL else {
            throw LlamaCppError.custom(
                "model download failed after \(maxAttempts) attempts: \(lastError.localizedDescription)")
        }
        try? FileManager.default.removeItem(at: resumeBlobURL)

        // Truncation guard — a clipped GGUF must never reach llama.cpp
        // (a partial file can load far enough to crash mid-prefill).
        if expectedBytes > 0 {
            let gotBytes = (try? FileManager.default
                .attributesOfItem(atPath: finalTmp.path)[.size] as? Int64) ?? 0
            guard gotBytes >= expectedBytes else {
                try? FileManager.default.removeItem(at: finalTmp)
                throw LlamaCppError.custom(
                    "model download truncated (\(gotBytes)/\(expectedBytes) bytes) — please retry")
            }
        }

        if FileManager.default.fileExists(atPath: destURL.path) {
            try? FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.moveItem(at: finalTmp, to: destURL)
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
        cp.n_ctx = UInt32(contextTokens)
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
        } else if let mem = llama_get_memory(ctx) {
            if lcp > 0, llama_memory_seq_rm(mem, 0, Int32(lcp), -1) {
                // Partial truncation accepted (pure-attention model).
            } else {
                _ = llama_memory_seq_rm(mem, 0, 0, -1)
                lcp = 0
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
        var pos: Int32 = Int32(lcp)
        var i = lcp
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
        // Prefill complete — the KV now holds exactly `tokens`.
        cachedTokens = tokens

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
            // Mirror what's actually in the KV so next turn's LCP can reuse
            // it. (A sampled token we break on BEFORE decoding — EOG, stop
            // sequence — is deliberately NOT appended: it never entered KV.)
            cachedTokens.append(id)
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
    /// HEAD-derived size used when the GET response is chunked (HF's Xet CDN
    /// often omits Content-Length, which used to freeze the UI at 0%).
    private let expectedBytes: Int64
    init(expectedBytes: Int64 = 0, _ onFraction: @escaping (Double) -> Void) {
        self.expectedBytes = expectedBytes
        self.onFraction = onFraction
    }
    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        // Prefer the response's own total; fall back to the HEAD size; as a
        // last resort crawl against a 5 GB ceiling so the bar still moves.
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite
                  : expectedBytes > 0 ? expectedBytes
                  : 5_000_000_000
        let fraction = min(0.999, Double(totalBytesWritten) / Double(total))
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
