// SPDX-License-Identifier: MIT
//
// Top-level observable app state. Owns the list of opened ZIM readers, the
// set of available models, the current chat transcript, and a reference to
// the MCPZim tool adapter that a Gemma 4 tool loop can dispatch through.

import CryptoKit
import Foundation
import MCPZimKit
import Observation
import OSLog
#if canImport(UIKit)
import UIKit
#endif

private let chatLog = Logger(subsystem: "org.mcpzim.MCPZimChat", category: "Chat")

@MainActor
@Observable
public final class ChatSession {
    // MARK: - Library (opened ZIMs)

    public struct LibraryEntry: Identifiable, Sendable {
        public let id = UUID()
        public let url: URL
        public let reader: ZimReader
        public var isEnabled: Bool = true
        public var kind: ZimKind { reader.kind }
        public var displayName: String {
            reader.metadata.title.isEmpty ? url.lastPathComponent : reader.metadata.title
        }
        /// True when this ZIM lives inside the app sandbox's Documents
        /// directory — i.e. "Remove" can safely trash the underlying
        /// file. External entries only hold security-scoped bookmarks
        /// and Remove just unlinks them from the library.
        public var isInSandboxDocuments: Bool {
            guard let docs = try? FileManager.default.url(
                for: .documentDirectory, in: .userDomainMask,
                appropriateFor: nil, create: false
            ) else { return false }
            return url.path.hasPrefix(docs.path)
        }
    }

    public var library: [LibraryEntry] = []
    public var libraryError: String?

    // MARK: - Models

    public private(set) var models: [any ModelProvider]
    public var selectedModel: any ModelProvider
    public var modelState: ModelLoadState = .notLoaded

    // MARK: - Transcript

    public var messages: [ChatMessage] = []
    public var isGenerating = false
    public var lastError: String?

    // MARK: - Plumbing

    public var service: DefaultZimService?
    public var adapter: MCPToolAdapter?

    // MARK: - In-window debug log
    //
    // Keeping a bounded ring of log entries on the observable session means
    // we can render them live in the chat window — much better feedback
    // than spelunking through `log stream` while the model is loading.

    public struct DebugEntry: Identifiable, Sendable {
        public let id = UUID()
        public let timestamp: Date
        public let category: String
        public let message: String
    }
    public var debugEntries: [DebugEntry] = []
    public var showDebugPane = false
    /// Debug-pane cap. Tuned for interactive use; tests that want to
    /// scan the full log can bump this before a long scenario.
    public var maxDebugEntries = 500

    /// When true, after a routing tool (`route_from_places` /
    /// `plan_driving_route`) returns, skip the model's iter-1 summary
    /// turn and render the reply directly from the tool result
    /// (distance + duration + first-few turn_by_turn steps). Saves
    /// the ~5 s generation cost of iter 1 for every routing question
    /// at the price of a more mechanical reply wording. User-toggleable
    /// in Library → Settings so both flavors can be A/B'd live.
    /// Persisted to UserDefaults via `didSet` so the choice survives
    /// relaunch.
    public var routingSkipModelReply: Bool = UserDefaults.standard.bool(
        forKey: "routingSkipModelReply"
    ) {
        didSet {
            UserDefaults.standard.set(routingSkipModelReply, forKey: "routingSkipModelReply")
        }
    }

    /// When true, double the per-turn reply token budget over the
    /// DeviceProfile default. Trades KV-cache headroom (and ~seconds
    /// of generation time) for fuller, less-clipped answers. With
    /// 4-bit KV-cache quantization enabled on phones the memory tax
    /// is ~4× cheaper than it used to be, so this is usually safe on
    /// 8 GB+ iPhones. Persisted so the choice survives relaunch.
    public var longerReplies: Bool = UserDefaults.standard.bool(
        forKey: "longerReplies"
    ) {
        didSet {
            UserDefaults.standard.set(longerReplies, forKey: "longerReplies")
        }
    }

    /// Device default × 2 when the user has opted in. All reply-generating
    /// sites (iter 0, iter 1, section reduce) read this instead of
    /// `DeviceProfile.current.maxReplyTokens` directly.
    public var effectiveMaxReplyTokens: Int {
        let base = DeviceProfile.current.maxReplyTokens
        let withToggle = longerReplies ? base * 2 : base
        // Per-provider floor — small models with reasoning modes
        // (Qwen 3 1.7B's `<think>` burns the default budget) get a
        // bigger budget because their weight footprint leaves plenty
        // of headroom. Only raises the budget, never lowers it.
        if let floor = (selectedModel as? Gemma4Provider)?.replyTokensFloor {
            return max(withToggle, floor)
        }
        return withToggle
    }

    /// When true, construct the `FoundationModelsProvider` variants at
    /// launch (model picker shows "Apple Foundation Models"). When
    /// false, skip them entirely — saves the FoundationModels.framework
    /// dylib load (~10–30 MB) and any Swift heap associated with the
    /// per-provider Tool schemas. Default: `false` on iOS while we're
    /// memory-constrained; flip via Library → Settings to restore
    /// the picker. Takes effect on next app launch (providers are
    /// constructed in ChatSession's init).
    public static let enableAppleFMKey = "enableAppleFM"
    public static var enableAppleFM: Bool {
        UserDefaults.standard.object(forKey: enableAppleFMKey) as? Bool ?? false
    }
    public var enableAppleFMBinding: Bool {
        get { Self.enableAppleFM }
        set { UserDefaults.standard.set(newValue, forKey: Self.enableAppleFMKey) }
    }

    public func debug(_ message: String, category: String = "App") {
        // Prefix every log line with resident-memory so it's easy to eyeball
        // which step moved the needle. Uses `phys_footprint` — the same number
        // the OS uses to decide whether to jetsam this process.
        let decorated = "\(message) · mem=\(MemoryStats.formatted())"
        let entry = DebugEntry(timestamp: Date(), category: category, message: decorated)
        debugEntries.append(entry)
        if debugEntries.count > maxDebugEntries {
            debugEntries.removeFirst(debugEntries.count - maxDebugEntries)
        }
        print("[\(category)] \(decorated)")
        // OSLog so idevicesyslog / Console.app can see these lines too.
        // print() only lands in Xcode's console when attached, which
        // we aren't when the app crashes/hangs on-device.
        chatLog.notice("[\(category, privacy: .public)] \(decorated, privacy: .public)")
        // Persistent on-disk archive. Survives crashes / jetsam so
        // Settings → Past Logs can show the last N runs for
        // post-mortem + Share (AirDrop to Mac, Mail, Save to Files).
        let tsFormatter = ChatSession.logTimestampFormatter
        let ts = tsFormatter.string(from: entry.timestamp)
        LogArchive.shared.append("\(ts) [\(category)] \(decorated)")
    }

    /// Shared formatter for persistent log rows. Isolated statically
    /// so we don't rebuild it on every debug() call.
    private static let logTimestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone.current
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    @ObservationIgnored private var stateObservationTask: Task<Void, Never>?

    /// Last known current location, if the user has granted permission.
    /// Injected into the system preamble so "directions to X" can
    /// default to "from here" without the user having to name an
    /// origin. Refreshed lazily on each new turn.
    public var currentLocation: (lat: Double, lon: Double)? = nil
    @ObservationIgnored private var lastLocationFetch: Date = .distantPast

    /// One-time setup state — drives the "Setting things up…" overlay
    /// at launch. `send()` refuses to run until this is `.ready` so a
    /// user-triggered generate never races with the prompt-cache
    /// prewarm.
    public enum SetupState: Equatable, Sendable {
        case pending
        case running(stage: String, progress: Double?)
        case ready
        case failed(String)
    }
    public var setupState: SetupState = .pending

    /// Guards `runLaunchSequence()` so SwiftUI's `.task` firing twice
    /// (common with NavigationStack view re-identification) doesn't
    /// double-open the library, double-rebuild the ZIM service, or
    /// double-warm the streetzim routing graph (+2 GB temporarily on
    /// each load). Always read/written on the main actor.
    @ObservationIgnored private var launchSequenceRan = false

    /// Single idempotent entry point for RootView's `.task`. SwiftUI
    /// can fire `.task` more than once across a NavigationStack's
    /// lifecycle (e.g., the Library push re-identifies the root and
    /// re-fires the closure); without this guard we were opening the
    /// library twice, rebuilding the ZIM service twice, and loading
    /// the streetzim graph.bin (~700 MB → +2 GB resident) twice.
    @MainActor
    public func runLaunchSequence() async {
        guard !launchSequenceRan else {
            debug("launch sequence already ran; skipping", category: "App")
            return
        }
        launchSequenceRan = true
        await scanDocumentsFolder()
        await restoreExternalBookmarks()
        LocationFetcher.requestAuthorizationIfNeeded()
        LocationFetcher.start()
        refreshLocationIfStale()
        prewarmBackgroundCaches()
        await runSetupIfNeeded()
        // Prewarm the KV cache with the system-turn + tool-declaration
        // prefix now. Re-added 2026-04-21 after a brief rollback: the
        // 2.6 GB of KV tensors this allocates gets allocated EITHER WAY
        // as soon as the user sends their first real query, so
        // postponing doesn't lower the steady-state peak — it just
        // delays it. Prewarming at launch means the first `send()`
        // finds a warm cache and skips the ~18 s cold prefill. The
        // real fix for the "cache gets dropped after one turn" symptom
        // isn't to skip the prewarm; it's to make the memory-warning
        // handler stop nuking the cache on the first iOS gripe (see
        // the handler at `didReceiveMemoryWarningNotification`).
        prewarmGemmaKVCacheIfIdle()
    }

    /// Warm the expensive start-up caches off the user's hot path.
    /// Called from `RootView` at launch. Intentionally concurrent —
    /// streetzim graph parse, reranker asset load, and location fix
    /// are all independent, so there's no point serialising them.
    /// Run the one-time setup that warms the KV cache with the static
    /// system prompt + tool declarations. Gates the chat UI so the
    /// user can't send a query until the cache is populated — that
    /// closes the race that hung the first build of this. Subsequent
    /// launches load the cache from disk and this returns quickly.
    @MainActor
    public func runSetupIfNeeded() async {
        guard setupState == .pending else { return }
        setupState = .running(stage: "Loading model…", progress: nil)
        // Wait for Gemma weights to be in memory. The download/load
        // itself fires from `loadSelectedModel()` which is kicked off
        // at init time.
        if case .ready = modelState {
            // already loaded
        } else {
            for _ in 0..<120 { // ~60 s
                if case .ready = modelState { break }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            guard case .ready = modelState else {
                setupState = .failed("Model failed to load within 60 s.")
                return
            }
        }
        // primeCache disabled: pre-filling ~4500 tokens of KV state
        // at launch left ~500-700 MB of extra resident memory
        // permanently allocated. On an iPhone 17 Pro Max already
        // carrying Gemma 4 weights (~2.6 GB) + Kokoro TTS (~400 MB)
        // + ZIMs + WebKit, that pushed peak into iOS jetsam territory
        // and the app got killed mid-reply repeatedly. First user
        // turn now pays ~3 s of full prefill instead, which is cheap
        // vs. getting terminated. Disk cache code below left intact
        // for when we resurrect a memory-safe variant (e.g.,
        // load-from-disk on first send, then evict).
        setupState = .ready
        return
        #if PRIMECACHE_ENABLED
        guard let gemma = selectedModel as? Gemma4Provider else {
            setupState = .ready
            return
        }
        // Cache key = static preamble + tools + model id + enabled
        // ZIMs (by filename). Anything that changes those should
        // invalidate the stored cache.
        let cacheKey = makePromptCacheKey()
        let cacheURL = promptCacheURL(for: cacheKey)
        let exists = FileManager.default.fileExists(atPath: cacheURL.path)
        let size = (try? FileManager.default.attributesOfItem(atPath: cacheURL.path)[.size] as? Int64) ?? 0
        debug("setup: cacheURL=\(cacheURL.path) exists=\(exists) size=\(size) bytes",
              category: selectedModel.template.logCategory)
        if exists {
            setupState = .running(stage: "Restoring saved prompt cache…", progress: nil)
            do {
                try await gemma.loadPromptCache(from: cacheURL)
                setupState = .ready
                debug("loaded prompt cache from disk (key=\(cacheKey.prefix(12))…)",
                      category: selectedModel.template.logCategory)
                return
            } catch {
                debug("disk cache load failed: \(error) — will re-prewarm",
                      category: selectedModel.template.logCategory)
                try? FileManager.default.removeItem(at: cacheURL)
            }
        }
        setupState = .running(stage: "Pre-filling system prompt…", progress: nil)
        do {
            try await warmPromptCacheOnce(gemma: gemma)
            try await gemma.savePromptCache(to: cacheURL, keyHint: cacheKey)
            debug("prewarmed + saved prompt cache (key=\(cacheKey.prefix(12))…)",
                  category: selectedModel.template.logCategory)
            setupState = .ready
        } catch {
            debug("prompt-cache warmup failed: \(error)", category: selectedModel.template.logCategory)
            // Fall through — user can still chat, just without the
            // cache benefit.
            setupState = .ready
        }
        #endif
    }

    /// Invalidate the on-disk cache so the next `runSetupIfNeeded()`
    /// rebuilds it. Called when the enabled ZIM set changes or the
    /// user swaps models.
    @MainActor
    public func invalidateSetupCache() {
        setupState = .pending
        // Best-effort — we don't know the old cache key, so nuke all
        // files under our prompt-cache dir. Cheap: they're < 1 GB.
        let dir = promptCacheDirectory()
        if let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) {
            for f in files where f.pathExtension == "safetensors" || f.pathExtension == "json" {
                try? FileManager.default.removeItem(at: f)
            }
        }
    }

    private func warmPromptCacheOnce(gemma: Gemma4Provider) async throws {
        guard let adapter else { return }
        let registry = await adapter.registry
        let toolDecls = toolDeclarations(registry: registry)
        lastQueryComplexity = .topical
        let preamble = systemMessageText(for: .topical)
        // Build a prewarm prompt that is a BYTE-EXACT prefix of what the
        // first real turn will look like. No user message, no trailing
        // `<|turn>model\n` — just `<bos>` + the tool-system turn. Iter 0
        // will tokenize its full prompt starting with the same bytes, so
        // LCP == cachedTokens.count on that first user send → cache hit,
        // skipping ~4000 tokens of prefill.
        //
        // We can't use `template.renderTranscript(... turns: [])` because
        // that appends the assistant-open marker, which diverges from
        // iter 0 (which has `<|turn>user\n{msg}…` there). Call the
        // system-turn formatter directly via the template.
        let template = selectedModel.template
        let systemTurn = template.formatSystemTurn(
            systemMessage: preamble, tools: toolDecls
        )
        let prompt = template.bos + systemTurn
        try await gemma.primeCache(prompt: prompt)
    }

    private func makePromptCacheKey() -> String {
        // Strip the location block — it's dynamic and we want the
        // cache to survive a GPS fix landing later.
        let preamble = Self.composeSystemMessage(
            categoryHint: Self.categoryHint(for: .topical),
            locationLine: ""
        )
        let toolNames = library
            .filter(\.isEnabled)
            .map { $0.url.lastPathComponent }
            .sorted()
            .joined(separator: ",")
        let modelID = selectedModel.id
        let raw = preamble + "\n\(toolNames)\n\(modelID)"
        return Self.sha256Hex(raw)
    }

    private static func sha256Hex(_ s: String) -> String {
        let h = SHA256.hash(data: Data(s.utf8))
        return h.map { String(format: "%02x", $0) }.joined()
    }

    private func promptCacheDirectory() -> URL {
        // Application Support persists across launches and is NOT
        // evicted by iOS under storage pressure (Caches is). Our
        // prompt cache is expensive to rebuild (5+ s of prefill), so
        // we want it to stick around.
        let fm = FileManager.default
        let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = base.appendingPathComponent("PromptCache", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func promptCacheURL(for key: String) -> URL {
        promptCacheDirectory()
            .appendingPathComponent("gemma-\(key.prefix(16)).safetensors")
    }

    public func prewarmBackgroundCaches() {
        Task { [weak self] in
            guard let self else { return }
            let started = Date()
            await self.service?.prewarmStreetzims()
            let dt = Date().timeIntervalSince(started)
            await MainActor.run {
                self.debug(String(format: "prewarmed streetzims in %.2fs", dt),
                           category: "ZimSvc")
            }
        }
        Task { [weak self] in
            // Poke the semantic reranker so `NLContextualEmbedding`
            // loads before the first search instead of blocking the
            // first tool_call round-trip.
            let started = Date()
            _ = await SemanticReranker.shared.rerank(query: "warmup", hits: [])
            let dt = Date().timeIntervalSince(started)
            await MainActor.run {
                self?.debug(String(format: "prewarmed reranker in %.2fs", dt),
                            category: "Rerank")
            }
        }
        // NOTE: Gemma prompt-cache prewarm disabled — racing with
        // the user's first query caused the app to hang (two
        // `ModelContainer` reads serialise, and tearing down the
        // prewarm's inner stream while the user's task awaited the
        // actor blocked indefinitely). Cross-turn cache hits still
        // work via the LCP match in `Gemma4Provider.generate`. The
        // disk-serialised cache (planned next) avoids this race by
        // loading state directly without touching `container.perform`.
    }

    /// Run a silent 1-token "hi" generation so Gemma's KV cache is
    /// populated with the static system-prompt + tool-declaration
    /// prefix. Next real send() does an LCP match against that
    /// prefix, skipping most of the prefill.
    @MainActor
    private func prewarmPromptCache() async {
        guard let gemma = selectedModel as? Gemma4Provider else { return }
        // Wait until weights are actually loaded; container ready =
        // modelState == .ready. If we fire before load, we'd just
        // spin.
        if case .ready = modelState {
            // ok
        } else {
            for _ in 0..<40 { // wait up to ~20 s
                try? await Task.sleep(nanoseconds: 500_000_000)
                if case .ready = modelState { break }
            }
            guard case .ready = modelState else {
                debug("prompt-cache warmup: model not ready in 20 s, skipping",
                      category: selectedModel.template.logCategory)
                return
            }
        }
        let started = Date()
        // Build a preamble+tools prefix that matches what a real
        // send() will build, but with a throwaway user message. The
        // location + category blocks will still land at the end (so
        // they're outside the cached prefix), but the big static
        // header + all tool declarations land inside it.
        guard let adapter else {
            debug("prompt-cache warmup: no tool adapter yet, skipping",
                  category: selectedModel.template.logCategory)
            return
        }
        let registry = await adapter.registry
        let toolDecls = toolDeclarations(registry: registry)
        lastQueryComplexity = .topical
        let preamble = self.systemMessageText(for: .topical)
        let turns = [ChatTurn(role: .user, text: "hi")]
        let finalPrompt = selectedModel.template.renderTranscript(
            systemPreamble: preamble, tools: toolDecls, turns: turns
        )
        do {
            let params = GenerationParameters(
                maxTokens: 1, temperature: 0.3, topP: 0.9
            )
            for try await _ in gemma.generate(prompt: finalPrompt, parameters: params) {
                break // one token is enough to bake the cache
            }
            let dt = Date().timeIntervalSince(started)
            debug(String(format: "prewarmed prompt cache in %.2fs", dt),
                  category: selectedModel.template.logCategory)
        } catch {
            debug("prompt-cache warmup failed: \(error)", category: selectedModel.template.logCategory)
        }
    }

    /// Build the `<|turn>system\n…` body used by `runGenerationLoop`
    /// and the prompt-cache warmup. Fully invariant — we used to
    /// fold a per-turn classification hint into either the preamble
    /// or the user-turn body, but a Mac behavior test (see
    /// `tools/gemma-smoke` `prompt-experiment`) confirmed Gemma 4
    /// picks the same tool calls with or without that hint, so we
    /// dropped it. The `_` argument is kept for call-site
    /// compatibility.
    fileprivate func systemMessageText(for _: QueryComplexity) -> String {
        let locationLine = self.locationLineText()
        return Self.composeSystemMessage(categoryHint: "", locationLine: locationLine)
    }

    /// Static assembly of the preamble body. Mirrors exactly the
    /// inline string that used to live in `runGenerationLoop`; do
    /// NOT reorder without bumping the prompt-cache version key.
    /// The `categoryHint` argument is now unused for the live path
    /// (kept for the cache-key hash to stay backwards-compatible).
    static func composeSystemMessage(categoryHint: String, locationLine: String) -> String {
        return """
        You are a helpful assistant with access to tools over locally-loaded \
        ZIM archives. Call tools immediately whenever they can answer the \
        user's question — do NOT ask the user to confirm, and do NOT ask \
        which ZIM to use (if there is a streetzim, use it for routing; \
        if there is a wikipedia, use it for general knowledge; if there \
        is an mdwiki, use it for medical questions). Pick sensible \
        defaults for optional arguments. Only respond in prose after \
        you have the tool result.

        Follow-up interpretation: when the user's current message is
        SHORT (under ~8 words) or begins with "and", "what about", "how
        about", "ok", "then", "also", "more on", "more about", treat it
        as a follow-up to the immediately previous turn in THIS
        conversation. Carry the prior subject forward — if the last
        turn was about "Iraq–United States relations" and the user
        says "and what about modern relations?", answer about the
        MODERN U.S.–Iraq relationship. Do NOT reply "could you
        specify what you mean" when the prior-turn subject makes the
        answer obvious; instead, search/fetch articles that extend
        that subject. Only ask for clarification when the short
        follow-up could plausibly mean several very different topics.

        Medical questions are in-scope: this app ships with WikiMed \
        (the mdwiki ZIM), an open encyclopedia of medical articles \
        written for clinicians and patients. For clearly clinical \
        queries (conditions, drugs, dosages, first aid), search it \
        for better-calibrated answers and relay what the article \
        says. Do NOT refuse with "I'm not a doctor" boilerplate — the \
        user is asking for the mdwiki's content, not your opinion.

        IMPORTANT: do NOT set `kind: "mdwiki"` (or any `kind` filter) \
        unless the user's question is unambiguously medical. Setting \
        `kind="mdwiki"` on a general query like "plasma physics" or \
        "Billy Crystal" blinds the search to Wikipedia and returns \
        nonsense. Default behaviour: OMIT `kind` entirely and let the \
        unified search pick the best ZIM for you.

        For routing questions, keep the reply SHORT — the user also \
        sees the map and the full list on-screen, and a spoken reply of \
        30+ turns is unusable. Your reply MUST include:
        1) total distance and duration from the tool result,
        2) a single-sentence summary of the major roads involved \
           (name the one or two freeways / arterials from the \
           `turn_by_turn` list that cover most of the distance),
        3) at most the FIRST 3–4 turns from `turn_by_turn`, then stop. \
           If `turn_by_turn_total` is present just say "about N steps \
           total" — do NOT enumerate the rest.

        For "what's nearby" style questions, lead your reply with the \
        `by_category` breakdown from the tool response. Only names from \
        the current `results` array are trustworthy — don't invent items \
        from counts or from earlier turns. The tool's own description \
        spells out when to re-call with `kinds` to drill into a bucket.

        For "tell me about X" / "what is X" / "how does X work" / \
        "explain X" questions, the preferred chain is:
        1. `search` — pick the best matching hit.
        2. `list_article_sections` on that hit's `path` — pick the 1–3 \
           sections that actually answer the question (skip "See also", \
           "References", etc. which the tool already strips).
        3. `get_article_section` once per chosen section.
        4. Answer from the sections you read. Write in natural prose — \
           DO NOT open with "per the 'lead' section" or "according to \
           the article"; the user already knows the answer is grounded. \
           Only name a section when it genuinely clarifies (e.g. \
           contrasting two sections of the same article).

        When the question is a short factoid, one `get_article_section` \
        on `lead` is usually enough. When it's broader ("tell me about \
        the French Revolution"), read 2–3 sections. When you truly \
        need the whole article (rare) fall back to `get_article`. \
        NEVER stop after `search` to ask "would you like me to fetch \
        it?" — just proceed through the chain.

        === Grounding policy ===
        This app's value to the user is that answers are grounded in \
        the loaded ZIM archives — not in your training priors. So: \
        * Every factual claim in your reply should trace to a tool result \
          you have seen this turn OR an earlier turn in this conversation. \
        * If the user asks a follow-up that refers back to a prior topic \
          ("when was that?", "tell me more about it"), reuse the article(s) \
          from the earlier turn rather than re-running the full search. \
        * Cite section / article names inline (e.g. "per 'Article' § \
          Causes…") whenever a claim isn't obviously common knowledge. \
        * If the loaded ZIMs genuinely don't cover the question, say that \
          — do not guess.\(locationLine)
        """
        // NB: `locationLine` is deliberately the LAST thing in the
        // preamble. It changes on every GPS fix (and is empty until the
        // first fix lands), so keeping it at the tail means the
        // everything-but-the-last-block prefix tokenises identically
        // across prewarm (no-fix state) and runtime (fix obtained) —
        // which keeps `Gemma4Provider.generate`'s LCP match near
        // `cached.count` and skips ~7000 tokens of prefill on iter 0.
    }

    private static func categoryHint(for complexity: QueryComplexity) -> String {
        switch complexity {
        case .navigational:
            return """
            The user's current turn looks NAVIGATIONAL (routing / "what's \
            around" / nearest-X). Use streetzim tools (`near_named_place`, \
            `route_from_places`). Do NOT call `search` or read Wikipedia \
            articles for this turn — that's a different surface.
            """
        case .factoid:
            return """
            The user's current turn looks FACTOID (short, single-fact \
            lookup). You MUST ground the answer in a tool-result — \
            never answer a factual claim from prior knowledge alone. \
            Either: \
            (a) call `search` → `get_article_section(section: "lead")` \
            and cite the article, OR \
            (b) if this is a follow-up (short question with pronouns \
            like "that"/"those"/"it"/"them"), reuse an article from \
            an earlier turn in THIS conversation and cite that \
            specific article + section. If you genuinely can't find \
            the fact in the loaded ZIMs, say so — don't guess.
            """
        case .topical:
            return """
            The user's current turn looks TOPICAL ("tell me about X" / \
            "what is X"). Fixed chain you MUST follow before writing a \
            reply: \
            1. `search` (one call). The search result's top hits include \
               a `preview` field (first ~400 chars of the article's \
               lead). READ every preview and pick the hit whose \
               preview actually matches what the user asked — do NOT \
               default to `hits[0]`. For "origin of pizza", skip \
               "Chicago-style pizza" (a regional variant) and choose \
               the general "Pizza" article. For "plasma", pick the \
               physics article over "plasma actuators" or "blood \
               plasma" unless the user specifically asked about those.
            2. `list_article_sections` on the chosen hit. \
            3. `get_article_section(section: "lead")`. \
            4. `get_article_section` on AT LEAST ONE more section \
               whose title bears on the user's question (history, \
               applications, current status, impact, mechanism, …). \
            5. Only then write the answer. \
            Skipping step 4 leaves the user with a lead-only summary \
            and that's what the model was asked NOT to do.
            """
        case .explanatory:
            return """
            The user's current turn looks EXPLANATORY ("explain how X \
            works" / "why did X happen" / "compare X and Y"). This is \
            a SYNTHESIS question. Fixed chain you MUST follow: \
            1. `search`. \
            2. `list_article_sections` on the best hit. \
            3. `get_article_section` on the lead + at least ONE \
               content section (mechanism / causes / effects / …). \
            4. For compare/contrast questions, or when the first \
               article alone can't answer the question, do a second \
               `search`/`list_article_sections`/`get_article_section` \
               cycle on a second article. \
            5. Only after steps 1–4 write the user-facing reply. \
            Total minimum `get_article_section` calls this turn: 2. \
            Answering after just one section or from snippets alone \
            is a failure — do not do it.
            """
        }
    }

    private func locationLineText() -> String {
        guard let here = currentLocation else {
            return """

            === Current location ===
            Location permission hasn't resolved yet. If the user asks \
            a location-relative question, tell them you can't get a \
            fix right now rather than guessing coordinates.
            """
        }
        let latStr = String(format: "%.5f", here.lat)
        let lonStr = String(format: "%.5f", here.lon)
        return """

        === Current location ===
        The user is physically at lat=\(latStr), lon=\(lonStr) right \
        now. Treat this as load-bearing context for every "where" / \
        "here" / "nearby" / "directions" / "nearest" question — \
        NEVER ask the user where they are.

        Tool recipes when the question references the user's \
        position (implicitly or explicitly):
          * "what's around (here|me)?" → `near_places(lat=\(latStr), \
            lon=\(lonStr), radius_km=1)` (no `kinds` unless the \
            user asked for a specific type).
          * "nearest <kind>" / "where's the closest <kind>" → \
            `near_places(lat=\(latStr), lon=\(lonStr), radius_km=5, \
            kinds=["<kind>"])`, then pick the single best hit.

        Tool recipes when the question references a DIFFERENT, \
        NAMED place (not the user's current position):
          * "<kind> in <named place>" / "restaurants in San Francisco" \
            / "museums near Berkeley" → `near_named_place(place="<named \
            place>", kinds=["<kind>"], radius_km=<default 1>)`. NEVER \
            use `near_places` with the user's lat/lon for these — that \
            would search their neighborhood, not the place they asked \
            about. `near_named_place` geocodes the string internally \
            and searches from there.
          * "tell me about <named place>" with no category filter → \
            `near_named_place(place="<named place>")` for a mixed list, \
            or `get_article(title="<named place>")` for an encyclopedic \
            summary.
          * "directions to <place>" / "how do I get to <place>" → \
            ALWAYS call `route_from_places(origin="my location", \
            destination="<place>")`. The host auto-fills the \
            origin lat/lon from the user's current fix when \
            `origin="my location"` is passed, and geocodes the \
            destination name. Do NOT invent dest_lat / dest_lon — \
            you DO NOT know the coordinates of place names, \
            guessing them produces a route that goes nowhere \
            (e.g. San Francisco is NOT at the user's \
            coordinates). Only use `plan_driving_route` with \
            raw lat/lons when BOTH endpoints came from a prior \
            tool result.
          * "directions to the nearest <kind>" → first \
            `near_places` (as above) to get the winning hit's \
            lat/lon, then `plan_driving_route` from \
            (\(latStr), \(lonStr)) to those coords.
          * "map of where I am" / "what neighborhood is this" → \
            `show_map(place="<the nearest named place>")`, OR \
            fall back to `near_places` and describe the top result.
        """
    }

    /// Kick off a location fetch if we haven't had a fresh fix in the
    /// last two minutes. Non-blocking — the preamble uses whatever we
    /// last saw, so a first-query user's reply doesn't stall on GPS.
    /// The first launch also triggers the `WhenInUse` permission
    /// prompt via `CLLocationManager`, so use a generous timeout so
    /// the user has time to tap Allow.
    @ObservationIgnored private var locationFetchTask: Task<Void, Never>?

    public func refreshLocationIfStale() {
        // No-op. `ChatSession.init` subscribes to `LocationFetcher.shared`,
        // so `currentLocation` auto-updates on every CL delegate callback.
        // Callers (RootView.task, RouteWebView.onAppear) still invoke this
        // for legacy reasons; kept as a symbol to avoid touching every
        // call site.
    }

    /// Replace any string-valued tool arg whose value is a user-facing
    /// "my location" synonym with the literal `"lat,lon"` string so
    /// ZimService.geocode's parseLatLon short-circuit picks it up. The
    /// preamble tells the model to use `origin:"my location"` on
    /// route_from_places, but the geocoder has no concept of "me" —
    /// this is where that shortcut gets resolved.
    private func substituteCurrentLocation(in args: [String: Any]) -> [String: Any] {
        guard let here = currentLocation else { return args }
        let coord = String(format: "%.5f,%.5f", here.lat, here.lon)
        let synonyms: Set<String> = [
            "my location", "my current location",
            "current location", "here", "me",
        ]
        var out = args
        // 1) Only resolve MY-LOCATION synonyms in string fields — never
        //    touch a string field that holds a real place name ("San
        //    Francisco", "the museum"). Otherwise we'd silently lie to
        //    the user: a query for restaurants in SF would come back
        //    with restaurants near the user's couch.
        for (key, val) in args {
            guard let s = val as? String else { continue }
            let lower = s.lowercased().trimmingCharacters(in: .whitespaces)
            if synonyms.contains(lower) {
                out[key] = coord
            }
        }
        // 2) Detect the tool shape. `near_places` / `near_named_place`
        //    expect numeric `lat`+`lon`; routing tools use `origin` /
        //    `destination` strings. ONLY inject the user's coords when
        //    we can tell the proximity tool is being called AND the
        //    model gave us nothing location-like to work with — no
        //    `origin` string, no existing lat/lon. If the model passed
        //    a real-looking origin (a place name), leave it alone —
        //    the tool adapter is responsible for geocoding it.
        let isProximityTool = out["kinds"] != nil || out["radius_km"] != nil
            || out["has_wiki"] != nil
        let hasNumericOrigin = toDouble(out["lat"]) != nil
            && toDouble(out["lon"]) != nil
        let originString = (out["origin"] as? String)?
            .trimmingCharacters(in: .whitespaces) ?? ""
        let hasMeaningfulOriginString = !originString.isEmpty
        if isProximityTool && !hasNumericOrigin && !hasMeaningfulOriginString {
            out["lat"] = here.lat
            out["lon"] = here.lon
        }
        // If the origin string is our own "lat,lon" synonym substitution
        // AND numeric lat/lon are also present, the string is redundant
        // — drop it so the tool adapter doesn't try to geocode a coord.
        if isProximityTool,
           hasNumericOrigin,
           originString == coord
        {
            out.removeValue(forKey: "origin")
        }
        // The model often pins `zim` to a Wikipedia ZIM by mistake on
        // near_places (prompt contamination). near_places requires a
        // streetzim — drop any wikipedia/mdwiki pin so the service's
        // fallback picks the right one. Applies only to proximity
        // tools; article lookups need the wikipedia ZIM.
        if isProximityTool,
           let z = out["zim"] as? String,
           z.contains("wikipedia") || z.contains("mdwiki")
        {
            out.removeValue(forKey: "zim")
        }
        // Numeric fallback: the model sometimes emits `origin_lat:0,
        // origin_lon:0` when the preamble lacked a location block (no
        // GPS at turn start). If we now have a fix by dispatch time,
        // inject it so the route still goes through with "my
        // location" semantics.
        if let la = toDouble(out["origin_lat"]), let lo = toDouble(out["origin_lon"]),
           la == 0 && lo == 0 {
            out["origin_lat"] = here.lat
            out["origin_lon"] = here.lon
        }
        // Same for destination zeros (rare — usually the dest is a
        // named place — but covers the edge where the model got
        // confused and omitted the destination coords too).
        if let la = toDouble(out["destination_lat"]), let lo = toDouble(out["destination_lon"]),
           la == 0 && lo == 0,
           (out["destination"] as? String).map({ $0.isEmpty }) ?? true {
            // Don't auto-inject destination — ambiguous. Just clear
            // the zeros so the geocoder uses the `destination` string.
            out.removeValue(forKey: "destination_lat")
            out.removeValue(forKey: "destination_lon")
        }
        // Final sweep: if the tool is route_from_places but has no
        // `origin` string at all and no origin_lat/lon, inject the
        // user's coords as the origin string.
        if out["origin"] == nil, out["origin_lat"] == nil, out["origin_lon"] == nil {
            out["origin"] = coord
        }
        return out
    }

    private func toDouble(_ any: Any?) -> Double? {
        if let d = any as? Double { return d }
        if let i = any as? Int { return Double(i) }
        if let s = any as? String { return Double(s) }
        return nil
    }

    /// Block the caller for up to `maxWait` seconds to let an
    /// in-flight `LocationFetcher.once()` land. Used at the top of a
    /// navigational / topical turn so "directions to X" doesn't fire
    /// the model while `currentLocation` is still nil. Returns
    /// immediately if we already have a fix (or the task is done).
    ///
    /// We poll `currentLocation` instead of awaiting `locationFetchTask.value` —
    /// `LocationFetcher.once()` wraps CoreLocation in a `CheckedContinuation`
    /// that cancellation cannot resume, so if CL never calls the delegate
    /// back (e.g. permission prompt pending, airplane mode, watch-GPS
    /// silent fail) the fetch task leaks forever and a TaskGroup.next()
    /// join on it hangs indefinitely. Polling side-steps that: we give
    /// up at the deadline and let the model answer without location.
    public func awaitLocationIfAny(maxWait: TimeInterval = 5) async {
        if currentLocation != nil { return }
        let deadline = Date().addingTimeInterval(maxWait)
        while currentLocation == nil, Date() < deadline {
            try? await Task.sleep(nanoseconds: 100_000_000) // 100 ms
        }
    }

    /// - Parameter autoLoadOnInit: when true (the default, used by the
    ///   app), kicks off an immediate `loadSelectedModel()` so users
    ///   don't have to press Load. Tests pass `false` so they can swap
    ///   the selected provider before any weights get downloaded.
    public init(autoLoadOnInit: Bool = true) {
        let defaults = UserDefaults.standard
        let storedCap = defaults.integer(forKey: Self.articleCapKBKey)
        // Default to the device-tier cap so phones don't blow RAM on
        // first launch. User can override via Library → Generation
        // and that override persists.
        self.articleCapKB = storedCap > 0 ? storedCap : DeviceProfile.current.articleCapKB
        let mock = MockProvider()
        // Default Gemma — multimodal repo that sanitize()'s down to
        // text-only at load time. Already in the on-device HF cache
        // on most dev phones, so launches instantly.
        let gemma = Gemma4Provider(
            id: "gemma4-e2b-it-4bit",
            displayName: "Gemma 4 E2B (4-bit · multimodal)",
            huggingFaceRepo: "mlx-community/gemma-4-e2b-it-4bit",
            approximateMemoryMB: 2600
        )
        // Pure text-only 4-bit quant. Kept in the picker so eval
        // harnesses can A/B it against the multimodal baseline even
        // though its tool-calling fidelity is weak under long prompts
        // (reproducer: `GemmaToolEmissionTests
        // .testEachVariantPicksNearNamedPlaceForNamedCity`). Expose it
        // anyway — the picker is what lets you run the eval harness
        // against it from the same binary.
        let gemmaText = Gemma4Provider(
            id: "gemma4-e2b-it-4bit-text",
            displayName: "Gemma 4 E2B Text (4-bit · text-only)",
            huggingFaceRepo: "mlx-community/Gemma4-E2B-IT-Text-int4",
            approximateMemoryMB: 2200
        )
        // Qwen 3 family — ChatML tool-call format, registered upstream
        // (`qwen3` / `qwen3_5_text` in `LLMModelFactory`). Same provider
        // class, same streaming path as Gemma; only the `template`
        // differs. 4B peers Gemma 4 E2B on memory; 1.7B is the small
        // slot for ≤4 GB iPhones.
        let qwen3_4b = Gemma4Provider(
            id: "qwen3-4b-4bit",
            displayName: "Qwen 3 4B (4-bit)",
            huggingFaceRepo: "mlx-community/Qwen3-4B-4bit",
            approximateMemoryMB: 2200,
            template: QwenChatMLTemplate()
        )
        // Qwen 3.5 4B 4-bit. Hybrid-attention sibling of Qwen 3 —
        // full-attention every 4th layer, linear/SSM on the others
        // (via our vendored mlx-swift-lm's `Qwen35TextModel`). Scored
        // 9/9 on the evaluator matrix matching Qwen 3 4B's perfect
        // score, with slightly smaller per-turn KV growth thanks to
        // the mostly-linear layers. Same `QwenChatMLTemplate` +
        // `/no_think` directive — our tool-call parser accepts all
        // four JSON shapes Qwen 3.5 emits.
        let qwen35_4b = Gemma4Provider(
            id: "qwen35-4b-4bit",
            displayName: "Qwen 3.5 4B (4-bit)",
            huggingFaceRepo: "mlx-community/Qwen3.5-4B-MLX-4bit",
            approximateMemoryMB: 2400,
            template: QwenChatMLTemplate()
        )
        let qwen3_1_7b = Gemma4Provider(
            id: "qwen3-1-7b-4bit",
            displayName: "Qwen 3 1.7B (4-bit)",
            huggingFaceRepo: "mlx-community/Qwen3-1.7B-4bit",
            approximateMemoryMB: 1000,
            template: QwenChatMLTemplate(),
            // Qwen 3's `<think>` reasoning mode spends the default
            // 320–384-token budget on scratchpad before reaching the
            // tool call. 1.7B has ~4 GB of memory headroom vs the
            // default 2600 MB Gemma budget, so give it a bigger token
            // budget so it can reliably finish both reasoning + tool.
            replyTokensFloor: 1024
        )
        var providers: [any ModelProvider] = [gemma, gemmaText, qwen3_4b, qwen35_4b, qwen3_1_7b]
        if #available(macOS 26.0, iOS 19.0, *), Self.enableAppleFM {
            // FoundationModels.framework gets linked into the app at
            // load time on iOS 19+/macOS 26+, costing ~10–30 MB. Only
            // construct the providers when the user has explicitly
            // opted into trying the Apple FM runtime — Gemma is the
            // default and we'd rather claw that headroom back for
            // KV-cache / Kokoro spikes.
            providers.append(FoundationModelsProvider())
            providers.append(FoundationModelsProvider(useNativeTools: true))
        }
        providers.append(mock)
        self.models = providers
        // Gemma 4 is the intended runtime; Apple Foundation Models is
        // an alternate on-device option; Mock is kept for UI debugging.
        // Persist the last-selected provider across launches so you
        // don't have to re-pick (and re-pay load costs for models you
        // weren't using).
        let savedId = UserDefaults.standard.string(forKey: Self.selectedModelKey)
        self.selectedModel = providers.first(where: { $0.id == savedId }) ?? gemma
        startObservingSelectedModel()
        // Wire every Gemma4Provider instance (which includes Qwen
        // variants — same class, different template) to the debug
        // pane. Each uses its own template's log category so "Qwen
        // 3 4B" lines aren't tagged `[Gemma4]`.
        for p in providers {
            guard let prov = p as? Gemma4Provider else { continue }
            let cat = prov.template.logCategory
            prov.debugSink = { [weak self] msg in
                Task { @MainActor [weak self] in
                    self?.debug(msg, category: cat)
                }
            }
        }
        // Wire the Apple FM debug sink now that `self` is fully initialised.
        if #available(macOS 26.0, iOS 19.0, *) {
            for p in providers where p.id.hasPrefix("apple-foundation-models") {
                if let apple = p as? FoundationModelsProvider {
                    apple.debugSink = { [weak self] msg in
                        Task { @MainActor [weak self] in
                            self?.debug(msg, category: "AppleFM")
                        }
                    }
                }
            }
        }
        // Auto-load on launch. The session is created exactly once by
        // `@State private var session = ChatSession()` in the @main scene,
        // and `Gemma4Provider.load()` is idempotent (early-returns if a
        // container already exists), so there's no way this double-loads.
        // Test harnesses pass `autoLoadOnInit: false` so they can control
        // memory probing around the load.
        if autoLoadOnInit {
            Task { @MainActor in await self.loadSelectedModel() }
        }
        // Subscribe to the LocationFetcher singleton. Every CL delegate
        // callback pushes a new coord into `currentLocation` with zero
        // polling / timeout machinery — replaces the fragile
        // `refreshLocationIfStale` + `LocationFetcher.once()` pair.
        #if canImport(UIKit)
        LocationFetcher.subscribe { [weak self] coord in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.currentLocation = (coord.latitude, coord.longitude)
            }
            // Mirror into ZimfoContext so the route_status / what_is_here
            // tools (dispatched off-main through the adapter's actor) have
            // a thread-safe source of the latest GPS fix.
            Task {
                await ZimfoContext.shared.updateLastLocation(
                    .init(lat: coord.latitude, lon: coord.longitude)
                )
            }
        }
        // Listen for iOS memory warnings and aggressively free the KV
        // cache + MLX Metal pool when they fire. Across a long
        // conversation the KV mirror grows by thousands of tokens
        // (each ~100 KB of Metal state) and combined with the 2.6 GB
        // Gemma weights + Kokoro TTS + WebKit map, the process can
        // drift into the zone where iOS jetsam kicks in. Dropping
        // the cache costs one full prefill on the next turn (~3 s)
        // which is cheap compared to getting killed.
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                // Don't touch the KV cache during an active generate
                // OR an in-flight KV prewarm — `resetPromptCache()`
                // calls `MLX.GPU.clearCache()` which synchronously
                // drains the Metal stream. Thrashing it mid-operation
                // slows things down AND would wipe the cache we're
                // mid-way through building. iOS fires 5–10 warnings
                // in quick succession while MLX does a big prefill;
                // they all become no-ops here until the operation
                // finishes.
                if self.isGenerating {
                    self.debug("memory warning ignored (isGenerating=true)",
                               category: "Chat")
                    return
                }
                if self.kvPrewarmTask != nil {
                    self.debug("memory warning ignored (kvPrewarm in flight)",
                               category: "Chat")
                    return
                }
                self.debug("memory warning — dropping KV cache + MLX pool",
                           category: "Chat")
                if let gemma = self.selectedModel as? Gemma4Provider {
                    gemma.resetPromptCache()
                }
            }
        }
        // Drop the KV cache + MLX buffer pool when the app moves to
        // background. iOS suspends us at our current RSS, and if the
        // suspended footprint is the biggest on the device, the
        // jetsam compressor will kill us to reclaim memory. We've
        // seen this repeatedly: MCPZimChat suspended at ~5 GB ends up
        // as "largestProcess" in JetsamEvent reports and gets
        // terminated. Shrinking the suspension footprint to just the
        // model weights + small working set avoids the kill.
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.debug("backgrounded — dropping KV cache + MLX pool",
                           category: "Chat")
                if let gemma = self.selectedModel as? Gemma4Provider {
                    gemma.resetPromptCache()
                }
            }
        }
        #endif
    }

    /// Test-only factory. Skips the normal init's Documents scan + ZIM
    /// reader bootstrap + model-picker restoration, and instead takes an
    /// explicit providers list + a pre-built `MCPToolAdapter` (typically
    /// backed by `StubZimService`). The harness can then immediately call
    /// `loadSelectedModel()` / `send(...)` / `waitForIdle()` without any
    /// real HF downloads or libzim I/O.
    ///
    /// `autoLoadOnInit: false` is forced — tests decide when to load so
    /// they can measure memory baselines first.
    public static func forTesting(
        providers: [any ModelProvider],
        adapter: MCPToolAdapter,
        initialModelId: String? = nil
    ) -> ChatSession {
        let session = ChatSession(autoLoadOnInit: false)
        session.models = providers
        if let id = initialModelId,
           let picked = providers.first(where: { $0.id == id })
        {
            session.selectedModel = picked
        } else if let first = providers.first {
            session.selectedModel = first
        }
        session.adapter = adapter
        // `send(...)` early-returns when `setupState != .ready` —
        // tests bypass the real setup flow and inject their own
        // adapter, so mark the session ready straight away.
        session.setupState = .ready
        return session
    }

    private func startObservingSelectedModel() {
        stateObservationTask?.cancel()
        stateObservationTask = Task { [weak self] in
            guard let self else { return }
            for await state in self.selectedModel.stateStream() {
                self.modelState = state
            }
        }
    }

    // MARK: - Library management

    public func scanDocumentsFolder() async {
        let fm = FileManager.default
        guard let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        else { return }
        let files = (try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil))?.filter {
            $0.pathExtension.lowercased() == "zim"
        } ?? []
        await openReaders(urls: files)
    }

    /// Open a fresh set of readers (typically from `scanDocumentsFolder()`).
    /// Replaces the existing library — callers that want to *add* should use
    /// `addReaders(urls:)` instead, which preserves the current entries.
    public func openReaders(urls: [URL]) async {
        let opened = await openEach(urls: urls, useSecurityScope: false)
        library = opened
        await rebuildService()
    }

    /// Append external (user-picked) ZIMs to the library. Distinct from
    /// `openReaders(urls:)` in two ways:
    ///   1. It *appends* (prior library stays loaded).
    ///   2. It enters each URL's security-scoped resource so libzim can
    ///      read the file even though it lives outside the app sandbox
    ///      (e.g. in `~/Downloads` where Kiwix also reads from).
    /// Also persists each picked URL as a bookmark so next launch reopens
    /// it without another pick.
    public func addReaders(urls: [URL]) async {
        let opened = await openEach(urls: urls, useSecurityScope: true)
        // Skip duplicates — the user picking the same file again shouldn't
        // create a second entry.
        let existingPaths = Set(library.map { $0.url.path })
        let fresh = opened.filter { !existingPaths.contains($0.url.path) }
        library.append(contentsOf: fresh)
        persistBookmarks()
        await rebuildService()
        // New ZIM changes the tool preamble (per-ZIM guidance) so the
        // saved prompt cache no longer matches. Force a rebuild on
        // next launch.
        if !fresh.isEmpty {
            invalidateSetupCache()
            Task { await runSetupIfNeeded() }
        }
    }

    private func openEach(urls: [URL], useSecurityScope: Bool) async -> [LibraryEntry] {
        var opened: [LibraryEntry] = []
        for url in urls {
            let memBefore = MemoryStats.physFootprintMB()
            // External-picked files live outside our sandbox; we must enter
            // the security scope before libzim can open them.
            var scoped = false
            if useSecurityScope {
                scoped = url.startAccessingSecurityScopedResource()
            }
            do {
                let reader = try LibzimReader(url: url)
                let delta = MemoryStats.physFootprintMB() - memBefore
                let mb = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64).flatMap { $0 } ?? 0
                debug(String(format: "opened %@ (file=%.0f MB, Δmem=%+.1f MB%@)",
                             url.lastPathComponent, Double(mb) / 1_048_576, delta,
                             scoped ? ", external" : ""),
                      category: "Library")
                opened.append(LibraryEntry(url: url, reader: reader))
            } catch {
                if scoped { url.stopAccessingSecurityScopedResource() }
                debug("open failed for \(url.lastPathComponent): \(error)", category: "Library")
                libraryError = "Could not open \(url.lastPathComponent): \(error)"
            }
        }
        return opened
    }

    // MARK: - Persistent bookmarks (for user-picked external ZIMs)
    //
    // Dropped into UserDefaults rather than a standalone plist so we don't
    // need a Codable model or a file layout. Each stored blob is a security-
    // scoped bookmark produced by `URL.bookmarkData(options: .withSecurityScope)`
    // (macOS) or `URL.bookmarkData()` (iOS).

    private static let bookmarksKey = "library.externalBookmarks"
    private static let selectedModelKey = "chat.selectedModelId"
    private static let articleCapKBKey = "chat.articleCapKB"

    /// How many KB of a single `get_article` response we pass to the
    /// model. Bigger = more complete context, slower first-token, and
    /// higher KV-cache memory spike on stream open. Persisted.
    public var articleCapKB: Int {
        didSet {
            UserDefaults.standard.set(articleCapKB, forKey: Self.articleCapKBKey)
        }
    }

    private func persistBookmarks() {
        // Store bookmarks for every entry whose URL is outside the sandbox
        // Documents dir — i.e. those opened via `addReaders`. Plain Documents
        // entries are rediscovered via `scanDocumentsFolder()` each launch.
        let docs = (try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false))?.path
        let externalURLs = library.map { $0.url }.filter { url in
            guard let docs else { return true }
            return !url.path.hasPrefix(docs)
        }
        let blobs: [Data] = externalURLs.compactMap { url in
            #if os(macOS)
            return try? url.bookmarkData(options: [.withSecurityScope])
            #else
            return try? url.bookmarkData()
            #endif
        }
        UserDefaults.standard.set(blobs, forKey: Self.bookmarksKey)
        debug("persisted \(blobs.count) external bookmark(s)", category: "Library")
    }

    /// Resolve previously-persisted bookmarks and open them as external
    /// readers. Called once at launch, before/after the Documents scan.
    public func restoreExternalBookmarks() async {
        guard let blobs = UserDefaults.standard.array(forKey: Self.bookmarksKey) as? [Data],
              !blobs.isEmpty else { return }
        var urls: [URL] = []
        for blob in blobs {
            var isStale = false
            #if os(macOS)
            let url = try? URL(
                resolvingBookmarkData: blob,
                options: [.withSecurityScope],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            #else
            let url = try? URL(
                resolvingBookmarkData: blob,
                options: [],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            #endif
            if let url {
                if isStale {
                    debug("bookmark stale for \(url.lastPathComponent); will refresh after open", category: "Library")
                }
                urls.append(url)
            }
        }
        if !urls.isEmpty {
            await addReaders(urls: urls)
        }
    }

    /// Rebuild the in-process tool service from the *enabled* subset of the
    /// library. Called after `openReaders` and on every toggle so the tool
    /// registry (and thus what the model sees) stays in sync with user intent.
    private func rebuildService() async {
        let pairs = library
            .filter { $0.isEnabled }
            .map { ($0.url.lastPathComponent, $0.reader as ZimReader) }
        let svc = DefaultZimService(readers: pairs)
        // Forward service-side progress into the debug pane so the user can
        // see slow steps (graph parse, geocode chunk load) as they happen.
        await svc.setLogger { [weak self] msg in
            Task { @MainActor [weak self] in
                self?.debug(msg, category: "ZimSvc")
            }
        }
        self.service = svc
        let adapter = await MCPToolAdapter.from(service: svc, surface: .conversational)
        // Phase 3: semantic reranker on top of BM25. Uses Apple's
        // `NLContextualEmbedding` when available — graceful no-op
        // otherwise. Reordering happens inside the `search` tool
        // dispatch so the model always sees the semantically-best
        // candidates first.
        SemanticReranker.log = { [weak self] msg in
            Task { @MainActor [weak self] in
                self?.debug(msg, category: "Rerank")
            }
        }
        await adapter.installHitReranker { query, hits in
            await SemanticReranker.shared.rerank(query: query, hits: hits)
        }
        // Host-state provider for the location-aware tools
        // (`route_status`, `what_is_here`). Reads from `ZimfoContext`
        // so the adapter stays framework-free — the iOS side mirrors
        // CL updates + route plans into ZimfoContext at their source.
        await adapter.installHostStateProvider {
            await ZimfoContext.shared.mcpSnapshot()
        }
        self.adapter = adapter
        // Wire the native-tools Apple-FM variant to the freshly-built
        // service so its Tool conformances dispatch to the same
        // backend the text-loop path uses. No-op when the framework
        // isn't linked or when only Gemma is around.
        if #available(macOS 26.0, iOS 19.0, *),
           let native = models.first(where: { $0.id == "apple-foundation-models-native" })
               as? FoundationModelsProvider {
            // Mirror the conversational surface — same seven tools
            // the text-loop path exposes, minus the raw-coord trio.
            native.installNativeTools([
                NearNamedPlaceNativeTool(service: svc),
                RouteFromPlacesNativeTool(service: svc),
                SearchNativeTool(service: svc),
                GetArticleNativeTool(service: svc),
                GetMainPageNativeTool(service: svc),
                ListLibrariesNativeTool(service: svc),
                ZimInfoNativeTool(service: svc),
            ])
        }
    }

    /// Toggle whether a ZIM contributes to the tool registry. The reader is
    /// kept open either way — only `rebuildService` gates visibility — so
    /// flipping the switch is cheap.
    /// Unlink a library entry. If the ZIM lives in the app's Documents
    /// folder, the underlying file is moved to the Trash (recoverable);
    /// external entries just lose their security-scoped bookmark.
    public func removeEntry(_ entryID: LibraryEntry.ID) async {
        guard let idx = library.firstIndex(where: { $0.id == entryID }) else { return }
        let entry = library[idx]
        if entry.isInSandboxDocuments {
            do {
                var trashURL: NSURL?
                try FileManager.default.trashItem(at: entry.url, resultingItemURL: &trashURL)
                debug("trashed \(entry.url.lastPathComponent)", category: "Library")
            } catch {
                debug("failed to trash \(entry.url.lastPathComponent): \(error)", category: "Library")
                libraryError = "Couldn't remove \(entry.url.lastPathComponent): \(error.localizedDescription)"
                return
            }
        } else {
            debug("unlinked external \(entry.url.lastPathComponent)", category: "Library")
        }
        library.remove(at: idx)
        persistBookmarks()
        await rebuildService()
        invalidateSetupCache()
        Task { await runSetupIfNeeded() }
    }

    public func setEnabled(_ enabled: Bool, for entryID: LibraryEntry.ID) async {
        guard let idx = library.firstIndex(where: { $0.id == entryID }) else { return }
        guard library[idx].isEnabled != enabled else { return }
        library[idx].isEnabled = enabled
        await rebuildService()
        // Enabled-set feeds into the cache key via filenames, so a
        // toggle changes the static prefix. Invalidate + rewarm.
        invalidateSetupCache()
        Task { await runSetupIfNeeded() }
    }

    // MARK: - Model switching

    public func select(modelId: String) async {
        guard let found = models.first(where: { $0.id == modelId }) else { return }
        guard found.id != selectedModel.id else { return }
        // Unload the previous model — iOS memory budget is tight.
        await selectedModel.unload()
        selectedModel = found
        UserDefaults.standard.set(found.id, forKey: Self.selectedModelKey)
        startObservingSelectedModel()
        // Auto-load the freshly-selected model. Without this, the UI
        // looked switched but `send(_:)` would throw
        // `ModelError.notLoaded` because nothing ever kicked off a
        // load. Users pressing the menu expect "pick a model = ready
        // to use". Tests can still opt out via their own gating.
        await loadSelectedModel()
        // Cache key embeds modelId — any on-disk cache now points at
        // a different model's tokenizer/architecture. Drop it and
        // rewarm once the new model is ready.
        invalidateSetupCache()
        await runSetupIfNeeded()
    }

    /// Hint to the selected model that a user turn is imminent — e.g.
    /// fired when the composer text field gains focus. Providers that
    /// support it (Apple FM native tools) eagerly reload their
    /// session's KV cache before the user finishes typing, so first
    /// token comes back faster. No-op for providers without the hook.
    public func prewarmSelectedModel() {
        if #available(macOS 26.0, iOS 19.0, *),
           let fm = selectedModel as? FoundationModelsProvider,
           fm.useNativeTools {
            fm.prewarmIfIdle()
        }
        // Gemma 4: prewarm the KV cache so iter 0 of the upcoming
        // turn hits cache instead of paying a 5 s full prefill. We
        // trigger this from the composer's focus handler so the work
        // overlaps with the user typing — a few seconds of hot
        // cache is usually done before they hit send. Idempotent: a
        // second call while the first is in flight is a no-op, and a
        // call when the cache is already warm returns immediately.
        // Deliberately NOT called at launch — the Kokoro TTS model
        // needs ~400 MB and the combined footprint crossed iOS
        // jetsam threshold when both lived in memory simultaneously.
        prewarmGemmaKVCacheIfIdle()
    }

    @ObservationIgnored private var kvPrewarmTask: Task<Void, Never>?

    /// Build the Gemma prompt KV cache in the background if we don't
    /// already have one. Safe to call repeatedly — it noops if a
    /// prewarm is in-flight or already finished.
    @MainActor
    public func prewarmGemmaKVCacheIfIdle() {
        guard let gemma = selectedModel as? Gemma4Provider else { return }
        if isGenerating { return }
        // Already warm?
        if gemma.hasPromptKVCache { return }
        if kvPrewarmTask != nil { return }
        kvPrewarmTask = Task { [weak self] in
            guard let self else { return }
            defer { Task { @MainActor [weak self] in self?.kvPrewarmTask = nil } }
            guard let adapter = self.adapter else { return }
            let registry = await adapter.registry
            let toolDecls = await MainActor.run { self.toolDeclarations(registry: registry) }
            let preamble = await MainActor.run { self.systemMessageText(for: .topical) }
            // Build exactly the byte-prefix an iter-0 prompt starts
            // with: `<bos>` + system-turn. No user turn, no trailing
            // model-open — the upcoming send's encode will land on
            // the same first N tokens and LCP-hit.
            let template = self.selectedModel.template
            let systemTurn = template.formatSystemTurn(
                systemMessage: preamble, tools: toolDecls
            )
            let prompt = template.bos + systemTurn
            let cat = template.logCategory
            self.debug("prewarming KV cache in background…", category: cat)
            do {
                try await gemma.primeCache(prompt: prompt)
            } catch {
                self.debug("KV prewarm failed: \(error)", category: cat)
            }
        }
    }

    public func loadSelectedModel() async {
        debug("Loading \(selectedModel.displayName)…", category: "Load")
        do {
            try await selectedModel.load()
            debug("Loaded \(selectedModel.displayName)", category: "Load")
        } catch {
            debug("Load failed: \(error)", category: "Load")
            lastError = String(describing: error)
        }
    }

    // MARK: - Send a user turn

    /// Drop the current transcript so the next `send(_:)` starts from a
    /// clean slate. Tool context, debug log, and loaded model are kept —
    /// this only resets the conversation itself. Also clears any
    /// provider-side conversation state (Apple FM's warm session owns
    /// the transcript internally and needs to be told).
    public func resetConversation() {
        messages.removeAll()
        if #available(macOS 26.0, iOS 19.0, *),
           let fm = selectedModel as? FoundationModelsProvider {
            fm.resetNativeConversation()
        }
        // Drop Gemma's KV prompt cache — the next send() starts a
        // completely new transcript, so anything cached from the
        // previous conversation is garbage.
        if let gemma = selectedModel as? Gemma4Provider {
            gemma.resetPromptCache()
        }
        debug("conversation reset", category: "Chat")
    }

    /// Classification of the most recent user turn. Set in `send()`
    /// and read by the generation loop so the system message can
    /// carry category-specific guidance (Phase 2b).
    public private(set) var lastQueryComplexity: QueryComplexity = .topical

    public func send(_ text: String) {
        // Setup must have finished (prompt-cache prewarm / load) before
        // we let a real turn hit the generator — otherwise the user's
        // first query races with the prewarm's container.perform and
        // hangs. The SetupOverlayView keeps the UI blocked; this guard
        // is a belt-and-braces for anything that slips past it (e.g.
        // the voice mic).
        if setupState != .ready {
            debug("send() ignored — setup still running (\(setupState))", category: "Chat")
            return
        }
        // Refresh GPS if our last fix is stale — the preamble built in
        // `runGenerationLoop` injects `currentLocation`, so a recent
        // snapshot means "directions to X" queries Just Work.
        refreshLocationIfStale()
        if let here = currentLocation {
            debug(String(format: "session location: (%.5f, %.5f)", here.lat, here.lon),
                  category: "Location")
        } else {
            debug("session location: <none> — preamble will omit it", category: "Location")
        }
        debug(text, category: "User")
        // Phase 2a classification → stashed for Phase 2b retrieval
        // routing in `runGenerationLoop`. Logged to the debug pane
        // either way so we can keep calibrating keyword rules
        // against real usage.
        let complexity = QueryComplexity.classify(text)
        lastQueryComplexity = complexity
        debug("query complexity: \(complexity.rawValue)", category: "Router")
        debug("model=\(selectedModel.displayName), state=\(modelState)", category: "Chat")
        let user = ChatMessage(role: .user, text: text)
        messages.append(user)
        messages.append(ChatMessage(role: .assistant, text: "", startedAt: Date()))
        isGenerating = true
        Task {
            // Fast-path intent router. Match a small set of simple
            // user patterns ("<category> in <place>", "directions to
            // <place>", "<category> near me") and dispatch the tool
            // directly — no model generate, no 13 s prefill. If no
            // pattern matches, fall through to the full LLM loop.
            // Logic lives in `MCPZimKit.IntentRouter` so it's covered
            // by `swift test` (see `IntentRouterTests`).
            if let intent = IntentRouter.classify(
                text, currentLocation: currentLocation
            ) {
                debug("fast-path intent: \(intent.toolName) — skipping LLM",
                      category: "Router")
                let handled = await executeDirectIntent(intent)
                if handled {
                    isGenerating = false
                    if let idx = messages.indices.last,
                       messages[idx].role == .assistant
                    {
                        messages[idx].finishedAt = Date()
                    }
                    return
                }
                debug("fast-path intent: dispatch failed, falling back to LLM",
                      category: "Router")
            }
            await runGenerationLoop()
            // Phase 2c: for explanatory turns, if the model pulled
            // >=2 sections, run a stateless map-reduce synthesis
            // over those sections to ground the final answer. Peak
            // memory stays flat (one section in prompt at a time).
            // The model's first-pass synthesis is discarded and
            // replaced with the reduced output — yes, that's a
            // wasted generation; worth it for the quality lift on
            // multi-source questions.
            if complexity == .explanatory {
                // `runGenerationLoop` already flipped isGenerating
                // false via its defer; keep the UI disabled while
                // the extra phase runs.
                isGenerating = true
                await maybeMapReduceExplanatory(userQuery: text)
                isGenerating = false
            }
        }
    }

    /// Core tool-aware generation loop.
    ///
    /// The strategy is deliberately transport-agnostic: we build a plain-text
    /// transcript from the current messages, stream tokens from the selected
    /// `ModelProvider`, and watch for `<tool_call>...</tool_call>` blocks in
    /// the stream. When one is detected, the loop halts generation, calls
    /// `MCPToolAdapter.dispatch(...)`, appends a synthetic tool response to
    /// the transcript, and restarts.
    private func runGenerationLoop() async {
        defer {
            isGenerating = false
            if let idx = messages.indices.last, messages[idx].role == .assistant {
                messages[idx].finishedAt = Date()
            }
        }
        debug("runGenerationLoop: entered", category: "Chat")
        // Navigational / topical queries lean heavily on location.
        // If a fetch is in flight, give it a short window to land
        // before we bake the preamble — otherwise the model sees
        // "Location permission hasn't resolved yet" and apologises
        // instead of answering.
        if lastQueryComplexity == .navigational || lastQueryComplexity == .topical {
            debug("runGenerationLoop: awaiting location (max 4s)", category: "Chat")
            await awaitLocationIfAny(maxWait: 4)
            debug("runGenerationLoop: location await done", category: "Chat")
        }
        guard let adapter else {
            debug("No adapter — library is empty.", category: "Chat")
            appendAssistant("[No ZIMs loaded — add .zim files to the app's Documents folder, then tap Refresh Library.]")
            return
        }

        debug("runGenerationLoop: fetching adapter registry", category: "Chat")
        let registry = await adapter.registry
        let toolDecls = self.toolDeclarations(registry: registry)
        debug("runGenerationLoop: \(toolDecls.count) tools declared", category: "Chat")
        let complexity = self.lastQueryComplexity
        let systemMessage = self.systemMessageText(for: complexity)
        debug("runGenerationLoop: system message \(systemMessage.count) chars", category: "Chat")
        // Preamble body lives in `Self.composeSystemMessage(...)` so
        // the startup prompt-cache warmup can reproduce the exact
        // bytes this loop emits.
        // Apple Foundation Models native-tools path: short-circuit
        // the text tool loop entirely. The framework owns the
        // transcript and dispatches tool calls internally via
        // `Tool.call()`, so we pass just the new user message (not a
        // re-rendered transcript), let one `streamResponse` consume
        // the whole turn (tool round-trips included), and stream
        // tokens straight to the UI.
        if #available(macOS 26.0, iOS 19.0, *),
           let fm = selectedModel as? FoundationModelsProvider,
           fm.useNativeTools {
            fm.setNativeInstructions(systemMessage)
            await runNativeToolsTurn(provider: fm)
            return
        }

        let promptFormatLabel = selectedModel is Gemma4Provider
            ? "Gemma-4 native format"
            : "generic <tool_call> preamble"
        debug("Dispatch loop: \(toolDecls.count) tools available (\(promptFormatLabel))", category: "Chat")

        // Structured turns that survive across tool-loop iterations. We drop
        // the final (empty) assistant placeholder since the provider template
        // appends the "open assistant" marker itself.
        //
        // Each assistant message may carry `toolRoundTrips` — the exact
        // intermediate (asst tool_call emission) + (tool response) text
        // from every round of its tool loop. We expand those into
        // separate ChatTurns before the final reply so the rebuilt
        // prompt BYTE-MATCHES what the KV cache was left in. Without
        // this, turn 2's iter 0 would diverge at the position of the
        // first tool_call emission and pay a full prefill.
        var turns: [ChatTurn] = []
        for msg in messages.dropLast() {
            if msg.role == .assistant {
                for rt in msg.toolRoundTrips {
                    if !rt.assistantEmission.isEmpty {
                        turns.append(ChatTurn(role: .assistant, text: rt.assistantEmission))
                    }
                    if !rt.toolResponseTurn.isEmpty {
                        turns.append(ChatTurn(role: .tool, text: rt.toolResponseTurn))
                    }
                }
            }
            if !msg.text.isEmpty {
                turns.append(ChatTurn(role: msg.role.asChatTurnRole, text: msg.text))
            }
        }
        // The last message is the empty assistant placeholder the
        // provider template's open-assistant tag resumes into — we
        // normally drop it with `dropLast` so it doesn't get rendered
        // twice. But the fast-path injector may have pre-populated
        // tool round-trips on it (compare_articles / article_overview
        // dispatched without the LLM's iter 0, now the LLM just has
        // to summarise). Include those round-trips here so the
        // rebuilt prompt ends in `<tool_response>…</tool_response>`
        // and the model's next emission is the prose.
        if let last = messages.last, last.role == .assistant {
            for rt in last.toolRoundTrips {
                if !rt.assistantEmission.isEmpty {
                    turns.append(ChatTurn(role: .assistant, text: rt.assistantEmission))
                }
                if !rt.toolResponseTurn.isEmpty {
                    turns.append(ChatTurn(role: .tool, text: rt.toolResponseTurn))
                }
            }
        }

        // Up to 6 tool loops per user turn — enough for small models
        // that burn iterations exploring (small search → wrong zim →
        // retry) before landing on a useful answer. Still capped so
        // a genuinely stuck loop terminates.
        let maxIters = 6
        for iter in 0..<maxIters {
            // Preemptive memory-pressure guard. MLX's Metal backend
            // doesn't surface command-buffer errors as Swift errors —
            // when the GPU runs out of memory mid-eval the underlying
            // C++ throws and the process terminates before our catch
            // below can fire. Short-circuit BEFORE we kick off another
            // prefill/sample if available memory has dropped below the
            // safe-headroom threshold: surface a Swift error the user
            // can read instead of an abort trap.
            // `os_proc_available_memory()` is iOS-only. The macOS
            // eval CLI links the same source tree but there's no
            // jetsam cap to respect there — leave availableMB at 0
            // so the guard below short-circuits.
            #if os(iOS)
            let availableMB = Double(os_proc_available_memory()) / (1024 * 1024)
            #else
            let availableMB: Double = 0
            #endif
            let minHeadroomMB: Double = 700   // rough KV-cache + Metal scratch floor for a 4B 4-bit Qwen turn
            if availableMB > 0, availableMB < minHeadroomMB {
                debug(String(format:
                    "runGenerationLoop: abort — only %.0f MB available, need ≥%.0f MB headroom before next generate() (protects against MLX abort_trap)",
                    availableMB, minHeadroomMB
                ), category: "Chat")
                lastError = String(format:
                    "Out of memory — only %.0f MB free, need %.0f MB to generate safely. "
                    + "Try resetting the conversation or closing any open fullscreen maps.",
                    availableMB, minHeadroomMB
                )
                return
            }

            // Prompt shape per provider:
            //   • Gemma 4 → native Gemma system turn + tool DSL.
            //   • Everyone else → generic preamble + formatTranscript.
            //     The preamble carries both tool schemas AND
            //     behavioural rules (how to read `by_category`,
            //     "don't invent names", etc.) — dropping it for
            //     native-tools models saves ~2 KB of prefill but
            //     degrades answer quality noticeably, so we keep it.
            let prompt: String
            if selectedModel is Gemma4Provider {
                prompt = selectedModel.template.renderTranscript(
                    systemPreamble: systemMessage,
                    tools: toolDecls,
                    turns: turns
                )
            } else {
                let preamble = Self.toolsPreamble(registry: registry)
                prompt = selectedModel.formatTranscript(systemPreamble: preamble, turns: turns)
            }
            let genStart = Date()
            debug("iter \(iter) · generate(prompt=\(prompt.count) chars)", category: "Chat")

            var buffer = ""
            var chunkCount = 0
            var toolCall: (range: Range<String.Index>, name: String, args: [String: Any])?

            // Lower temperature than the default 0.7 so Gemma 4 E2B commits to
            // a tool call instead of hedging with "Would you like me to…?"
            // follow-ups. If later turns need more variety, widen it.
            //
            // maxTokens drives how aggressively MLX pre-reserves KV-cache on
            // stream open — the spike we saw at `stream opened in …` scales
            // with this number. 512 comfortably covers a full Kaunas→Vilnius
            // turn-by-turn reply (~350 tokens visible) with headroom;
            // dropping to 256 clipped the list and the model started
            // emitting distance/duration only. Trade ~100 MB of cache spike
            // for reliably complete answers. On-device default comes
            // from DeviceProfile (256 on 4 GB phones, up to 512 on 8+ GB
            // and macOS) so MLX's KV-cache reservation fits the
            // jetsam budget.
            let params = GenerationParameters(
                maxTokens: effectiveMaxReplyTokens,
                temperature: 0.3, topP: 0.9
            )
            // Throttle UI pushes to ~10 Hz. Each `appendToAssistant` mutates
            // the observable `messages` array, which cascades into
            // re-layout of the chat bubble + scroll view + debug pane.
            // Doing that per token pegs a CPU core; 10 Hz is indistinguishable
            // to the eye and recovers most of the main-thread headroom.
            var lastUIPush = Date.distantPast
            let uiMinInterval: TimeInterval = 0.1
            do {
                for try await chunk in selectedModel.generate(prompt: prompt, parameters: params) {
                    buffer += chunk
                    chunkCount += 1
                    let now = Date()
                    if now.timeIntervalSince(lastUIPush) >= uiMinInterval {
                        appendToAssistant(buffer)
                        lastUIPush = now
                    }
                    if let call = self.extractToolCall(in: buffer) {
                        appendToAssistant(buffer)
                        toolCall = call
                        break
                    }
                }
                // Flush the final tokens — the throttle may have skipped them.
                appendToAssistant(buffer)
                // Post-stream rescue: generation ended naturally without
                // the strict streaming parser matching a tool call. Qwen
                // 3.5 (and occasionally Qwen 3) sometimes gets clipped by
                // the `<|im_end|>` stop token mid-`</tool_call>`, leaving
                // a partial closer like `</tool` or bare JSON. The
                // template's `firstToolCallAfterClip` retries with a
                // lenient closer — only called here, not during stream —
                // so a clipped emission still dispatches instead of the
                // loop silently returning "done, no tool call".
                if toolCall == nil,
                   let rescued = selectedModel.template.firstToolCallAfterClip(in: buffer)
                {
                    debug("iter \(iter) · recovered clipped tool_call via firstToolCallAfterClip",
                          category: "Chat")
                    toolCall = (range: rescued.range, name: rescued.name, args: rescued.arguments)
                }
            } catch {
                debug("generate threw: \(error)", category: "Chat")
                lastError = String(describing: error)
                return
            }

            let dt = Date().timeIntervalSince(genStart)
            guard let call = toolCall else {
                debug(String(format: "iter %d · done (%d chunks, %.2fs, %d chars)",
                              iter, chunkCount, dt, buffer.count),
                      category: "Chat")
                // Mirror the final assistant text into the debug pane so
                // you see the full Q/A pair inline with the tool calls.
                let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    debug(trimmed, category: "Assistant")
                }
                return
            }
            debug(String(format: "iter %d · tool_call %@ after %.2fs (%d chunks)",
                          iter, call.name, dt, chunkCount),
                  category: "Chat")

            // Substitute "my location" / "here" / "me" / "current location"
            // in routing + proximity tool args with the user's lat,lon so the
            // geocoder doesn't try to find a place literally named "my
            // location". Covers the `origin` arg on `route_from_places` and
            // anywhere the model used the preamble's shortcut phrasing.
            let resolvedArgs = substituteCurrentLocation(in: call.args)
            let argsData = try? JSONSerialization.data(withJSONObject: resolvedArgs)
            let argsStr = argsData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            let pre = String(buffer[..<call.range.lowerBound])
            // Use the FULL buffer (not trimmed at call.range.upperBound).
            // The sampler's last token often decodes to text that spans
            // the `<tool_call|>` marker AND a few chars past it (e.g. a
            // trailing newline). Those post-marker chars are already
            // in the Gemma4Provider KV cache; if we trimmed them off
            // here, encode(iter-1 prompt) would not match the cache
            // mirror's last token and LCP would fall short by 1 token —
            // every follow-up turn in the same conversation would pay
            // a full prefill. Feeding the whole buffer keeps the cache
            // mirror and the re-encoded prompt in sync.
            let assistantTurnText = buffer
            let memBefore = MemoryStats.physFootprintMB()
            let toolStart = Date()
            debug("dispatching \(call.name)(\(argsStr)) — first call against a ZIM may block on graph/index load", category: "Tool")
            do {
                let fullResult = try await adapter.dispatch(tool: call.name, args: resolvedArgs)
                // Pass-through sentinel: the tool wants its `text` emitted
                // verbatim to the user without another model pass. Used by
                // `narrate_article` so Wikipedia prose reaches TTS unaltered
                // (no paraphrase, no re-summarization, no KV-cost for the
                // model re-encoding an article it already decided to read).
                let isPassThrough = (fullResult["pass_through"] as? Bool) == true
                let passThroughText = (fullResult["text"] as? String) ?? ""
                // Routing results carry a polyline with thousands of points
                // and a turn-by-turn list that together inflate to 50+ KB.
                // Feeding that verbatim into the next prompt turns into
                // ~30 000 tokens of context, which is expensive and useless
                // to the model (it can't navigate by lat/lons anyway). Trim
                // to a summary before re-prompting.
                var preTrim = fullResult
                if isPassThrough {
                    // Swap the body for a compact ack so the tool-response
                    // turn the model sees on re-prompt stays cheap. The
                    // full body is still available in `rawResult` for
                    // debug + UI.
                    preTrim = [
                        "pass_through": true,
                        "title": (fullResult["title"] as? String) ?? "",
                        "bytes": (fullResult["bytes"] as? Int) ?? 0,
                        "delivered": true,
                        "note": "Full article body was read directly to the "
                            + "user; no further narration or summary needed.",
                    ]
                }
                if call.name == "search" {
                    preTrim = self.enrichSearchHits(preTrim)
                }
                let result = Self.trimForModel(toolName: call.name, result: preTrim, articleCapKB: self.articleCapKB)
                let resultData = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
                let resultStr = String(data: resultData, encoding: .utf8) ?? "{}"
                // Also serialize the UNTRIMMED result so UI extras (like the
                // route map) can use the full polyline without re-dispatching.
                let rawData = (try? JSONSerialization.data(withJSONObject: fullResult, options: [.sortedKeys])) ?? resultData
                let rawStr = String(data: rawData, encoding: .utf8) ?? resultStr
                let toolDt = Date().timeIntervalSince(toolStart)
                let delta = MemoryStats.physFootprintMB() - memBefore
                debug(String(format: "tool %@ returned %d bytes in %.2fs · Δmem=%+.1f MB (trimmed for model: %d bytes)",
                              call.name, rawStr.count,
                              toolDt, delta, resultStr.count),
                      category: "Tool")
                // For name-resolving tools, echo the `resolved` record's
                // name/location/coords so we can tell whether a weird
                // "Silver Spring, MD" in the model's reply came from the
                // streetzim data or was confabulated by the model.
                if call.name == "near_named_place" || call.name == "route_from_places",
                   let resolved = (fullResult["resolved"] as? [String: Any])
                       ?? ((fullResult["origin_resolved"] as? [String: Any]))
                {
                    let name = resolved["name"] as? String ?? "?"
                    let loc = resolved["location"] as? String ?? ""
                    let lat = resolved["lat"] as? Double ?? 0
                    let lon = resolved["lon"] as? Double ?? 0
                    debug("resolved: name=\"\(name)\" location=\"\(loc)\" (\(lat), \(lon))",
                          category: "Tool")
                }
                recordToolTrace(ToolCallTrace(
                    name: call.name,
                    arguments: argsStr,
                    result: resultStr,
                    rawResult: rawStr,
                    error: nil
                ))
                updateAssistant(pre) // truncate the <tool_call> block from the visible message.
                if !assistantTurnText.isEmpty {
                    turns.append(ChatTurn(role: .assistant, text: assistantTurnText))
                }
                // Format tool response in the provider's native wire format.
                let toolTurnText: String
                if selectedModel is Gemma4Provider {
                    toolTurnText = selectedModel.template.formatToolResponse(name: call.name, payload: result)
                } else {
                    toolTurnText = resultStr
                }
                turns.append(ChatTurn(role: .tool, text: toolTurnText))
                // Persist the exact round-trip text onto the assistant
                // ChatMessage so the next user turn can rebuild the prompt
                // byte-for-byte and hit the KV cache at iter 0.
                recordToolRoundTrip(assistantEmission: assistantTurnText,
                                    toolResponse: toolTurnText)

                // Pass-through short-circuit: emit the tool's `text` body
                // as the assistant reply and skip iter 1 — saves both the
                // prefill of the full article body AND the generation cost
                // of the model re-narrating what's already clean prose.
                if isPassThrough, !passThroughText.isEmpty {
                    updateAssistant(passThroughText)
                    debug("narrate pass-through: emitted \(passThroughText.count) chars (iter 1 skipped)",
                          category: "Chat")
                    return
                }

                // Optional fast path for routing tools — skip iter 1.
                // Saves ~5 s per routing turn by synthesizing the reply
                // directly from the tool result instead of asking the
                // model to rephrase it. Controlled by
                // `routingSkipModelReply` (Library → Settings).
                let routingTools: Set<String> = ["route_from_places", "plan_driving_route"]
                if routingSkipModelReply && routingTools.contains(call.name) {
                    let synth = Self.synthesizeRoutingReply(from: fullResult)
                    if !synth.isEmpty {
                        updateAssistant(synth)
                        debug("routing skip-model-reply: synthesized \(synth.count) chars (iter 1 skipped)",
                              category: "Chat")
                        return
                    }
                }

                // Same fast path for the places-returning families. The
                // map bubble below the message carries the actual
                // answer — pins + popups with Wikipedia intros where
                // available — so the LLM's prose summary is both
                // redundant AND slow (Qwen 3.5 pays the full hybrid-
                // cache prefill every turn, ~13 s). Synthesise a
                // one-line caption and skip iter 1.
                let placesTools: Set<String> = [
                    "near_named_place", "near_places",
                    "nearby_stories", "nearby_stories_at_place",
                ]
                if placesTools.contains(call.name) {
                    let synth = IntentRouter.synthesizePlacesReply(
                        toolName: call.name,
                        args: call.args,
                        fullResult: fullResult
                    )
                    if !synth.isEmpty {
                        updateAssistant(synth)
                        debug("places skip-model-reply: synthesized \(synth.count) chars (iter 1 skipped)",
                              category: "Chat")
                        return
                    }
                }
            } catch {
                let err = String(describing: error)
                debug("tool \(call.name) failed: \(err)", category: "Tool")
                recordToolTrace(ToolCallTrace(name: call.name, arguments: argsStr, result: "", error: err))
                if !assistantTurnText.isEmpty {
                    turns.append(ChatTurn(role: .assistant, text: assistantTurnText))
                }
                let errPayload: [String: Any] = ["error": err]
                let toolTurnText: String
                if selectedModel is Gemma4Provider {
                    toolTurnText = selectedModel.template.formatToolResponse(name: call.name, payload: errPayload)
                } else {
                    toolTurnText = "[error] \(err)"
                }
                turns.append(ChatTurn(role: .tool, text: toolTurnText))
                // Persist the error round-trip too — the next turn
                // needs the same bytes whether the tool succeeded or
                // errored, or LCP will miss on failed queries.
                recordToolRoundTrip(assistantEmission: assistantTurnText,
                                    toolResponse: toolTurnText)
            }
        }
        // Loop exhausted with unresolved tool results — force one last
        // no-tool-call generation so the user sees a reply instead of
        // an empty assistant bubble. Happens on small/slower models
        // that burn iterations exploring.
        if let last = messages.last, last.role == .assistant, last.text.isEmpty {
            debug("tool loop exhausted after \(maxIters) iters — forcing a summary turn",
                  category: "Chat")
            let summaryPrompt: String
            let summaryInstruction = ChatTurn(
                role: .user,
                text: "You've used your tool budget. Without calling any more "
                    + "tools, summarize what you found for the user in 1–3 "
                    + "sentences based on the tool results above."
            )
            var finalTurns = turns
            finalTurns.append(summaryInstruction)
            if selectedModel is Gemma4Provider {
                summaryPrompt = selectedModel.template.renderTranscript(
                    systemPreamble: systemMessage, tools: toolDecls, turns: finalTurns
                )
            } else {
                let preamble = Self.toolsPreamble(registry: registry)
                summaryPrompt = selectedModel.formatTranscript(
                    systemPreamble: preamble, turns: finalTurns
                )
            }
            var buffer = ""
            let params = GenerationParameters(maxTokens: 256, temperature: 0.3, topP: 0.9)
            do {
                for try await chunk in selectedModel.generate(prompt: summaryPrompt, parameters: params) {
                    buffer += chunk
                    appendToAssistant(buffer)
                }
            } catch {
                debug("summary generation failed: \(error)", category: "Chat")
            }
            let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                debug(trimmed, category: "Assistant")
            }
        }
    }

    // MARK: - Map-reduce synthesis for explanatory turns

    /// Guard + extract: only run map-reduce if the last assistant
    /// turn pulled multiple `get_article_section` sources. One source
    /// is better served by the direct-answer path.
    private func maybeMapReduceExplanatory(userQuery: String) async {
        guard let lastIdx = messages.lastIndex(where: { $0.role == .assistant })
        else { return }
        let sectionTraces = messages[lastIdx].toolCalls
            .filter { $0.name == "get_article_section" && $0.succeeded }
        guard sectionTraces.count >= 2 else {
            debug("explanatory: only \(sectionTraces.count) section source(s), skipping map-reduce",
                  category: "MapReduce")
            return
        }
        await runMapReduce(userQuery: userQuery, sectionTraces: sectionTraces)
    }

    /// Decode a `get_article_section` result JSON into its human
    /// fields. Returns nil for malformed / non-article traces.
    private struct MapReduceSection {
        let article: String
        let section: String
        let body: String
    }

    private func decodeSectionTrace(_ trace: ToolCallTrace) -> MapReduceSection? {
        guard let data = trace.result.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = obj["text"] as? String, !text.isEmpty
        else { return nil }
        let article = (obj["title"] as? String)
            ?? (obj["path"] as? String)
            ?? "(unknown)"
        let section = (obj["section"] as? String) ?? "lead"
        return MapReduceSection(article: article, section: section, body: text)
    }

    /// Stateless map-reduce over the sections the model fetched:
    ///   • Map — for each section, generate a short, section-only
    ///     digest of points that answer the user's question. Each
    ///     call runs in its own generation with a minimal prompt
    ///     (one section body at a time), so peak MLX KV-cache
    ///     reservation is bounded by the largest single section
    ///     rather than the sum.
    ///   • Reduce — feed the digests back as notes and stream one
    ///     final answer to the UI, replacing the first-pass text.
    private func runMapReduce(userQuery: String, sectionTraces: [ToolCallTrace]) async {
        let sections = sectionTraces.compactMap(decodeSectionTrace)
        guard sections.count >= 2 else { return }

        debug("map-reduce: \(sections.count) sections → per-section digests",
              category: "MapReduce")

        // ===== Map phase =====
        var summaries: [String] = []
        let mapParams = GenerationParameters(maxTokens: 256, temperature: 0.2, topP: 0.9)
        for (i, sec) in sections.enumerated() {
            let mapUserTurn = """
            User's question: \(userQuery)

            Text from the article "\(sec.article)" (section: \(sec.section)):

            \(sec.body)

            List 3–6 concise bullet points from THIS TEXT that help answer \
            the user's question. Only include facts explicitly present in \
            the text above. No outside knowledge, no invention.
            """
            let preamble = "You are a careful note-taker."
            let turns = [ChatTurn(role: .user, text: mapUserTurn)]
            let prompt: String
            if selectedModel is Gemma4Provider {
                prompt = selectedModel.template.renderTranscript(
                    systemPreamble: preamble, tools: [], turns: turns
                )
            } else {
                prompt = selectedModel.formatTranscript(
                    systemPreamble: preamble, turns: turns
                )
            }
            debug("map \(i + 1)/\(sections.count): \(sec.article) § \(sec.section) · \(sec.body.count) chars",
                  category: "MapReduce")
            var buf = ""
            do {
                for try await chunk in selectedModel.generate(
                    prompt: prompt, parameters: mapParams
                ) {
                    buf += chunk
                }
            } catch {
                debug("map \(i + 1) failed: \(error)", category: "MapReduce")
                continue
            }
            let trimmed = buf.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                summaries.append(
                    "### From \"\(sec.article)\" § \(sec.section)\n\(trimmed)"
                )
            }
        }
        guard !summaries.isEmpty else {
            debug("map-reduce: no non-empty digests, keeping first-pass answer",
                  category: "MapReduce")
            return
        }

        // ===== Reduce phase =====
        debug("reduce: synthesising from \(summaries.count) digest(s)",
              category: "MapReduce")
        let reduceUserTurn = """
        User's question: \(userQuery)

        Notes I gathered from the available articles:

        \(summaries.joined(separator: "\n\n"))

        Write a clear, thorough answer to the user's question, grounded \
        only in the notes above. Use natural prose — DO NOT open with \
        "per the 'lead' section…" or "according to the article…". Cite \
        a specific source only when the user would genuinely benefit \
        (e.g. contrasting two sources). Do NOT add facts that aren't \
        in the notes.
        """
        let preamble = "You are a helpful, grounded writer."
        let turns = [ChatTurn(role: .user, text: reduceUserTurn)]
        let prompt: String
        if selectedModel is Gemma4Provider {
            prompt = Gemma4PromptTemplate.render(
                systemMessage: preamble, tools: [], turns: turns
            )
        } else {
            prompt = selectedModel.formatTranscript(
                systemPreamble: preamble, turns: turns
            )
        }

        // Replace first-pass text with a visible placeholder so the
        // user sees the phase transition.
        if let idx = messages.lastIndex(where: { $0.role == .assistant }) {
            messages[idx].text = "_Synthesising from \(summaries.count) grounded sources…_\n\n"
        }

        var buffer = ""
        let reduceParams = GenerationParameters(
            maxTokens: effectiveMaxReplyTokens,
            temperature: 0.3, topP: 0.9
        )
        var lastUIPush = Date.distantPast
        do {
            for try await chunk in selectedModel.generate(
                prompt: prompt, parameters: reduceParams
            ) {
                buffer += chunk
                let now = Date()
                if now.timeIntervalSince(lastUIPush) >= 0.1 {
                    appendToAssistant(buffer)
                    lastUIPush = now
                }
            }
            appendToAssistant(buffer)
        } catch {
            debug("reduce failed: \(error)", category: "MapReduce")
            return
        }
        debug("map-reduce complete: \(buffer.count) chars", category: "MapReduce")
    }

    /// Single-turn dispatch for Apple Foundation Models native-tools.
    /// The framework is stateful — it owns the transcript across calls
    /// and handles tool round-trips inside one `streamResponse` — so
    /// we don't drive the text-based tool loop here. Just grab the
    /// latest user message, stream the reply, and move on.
    @available(macOS 26.0, iOS 19.0, *)
    private func runNativeToolsTurn(provider: FoundationModelsProvider) async {
        guard let lastUser = messages.last(where: { $0.role == .user })?.text,
              !lastUser.isEmpty
        else { return }
        debug("native-tools turn: userMessage=\(lastUser.count) chars", category: "Chat")
        let params = GenerationParameters(
            maxTokens: effectiveMaxReplyTokens,
            temperature: 0.3, topP: 0.9
        )
        var buffer = ""
        let genStart = Date()
        var lastUIPush = Date.distantPast
        let uiMinInterval: TimeInterval = 0.1
        do {
            for try await chunk in provider.generateNativeTurn(userMessage: lastUser, parameters: params) {
                buffer += chunk
                let now = Date()
                if now.timeIntervalSince(lastUIPush) >= uiMinInterval {
                    appendToAssistant(buffer)
                    lastUIPush = now
                }
            }
            appendToAssistant(buffer)
        } catch {
            debug("native-tools turn threw: \(error)", category: "Chat")
            lastError = String(describing: error)
            return
        }
        let dt = Date().timeIntervalSince(genStart)
        debug(String(format: "native-tools turn · done (%.2fs, %d chars)",
                     dt, buffer.count),
              category: "Chat")
        let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            debug(trimmed, category: "Assistant")
        }
    }

    // MARK: - Transcript helpers

    private func appendAssistant(_ text: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = text
        }
    }

    private func appendToAssistant(_ replacement: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = scrubReasoning(replacement)
        }
    }

    private func updateAssistant(_ newText: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = scrubReasoning(newText)
        }
    }

    /// Run the selected model's template-specific reasoning scrubber
    /// over `text` before it lands in the chat bubble. Gemma's default
    /// is no-op; Qwen removes `<think>…</think>` blocks. Partial /
    /// still-open spans stay visible so we don't flash mid-stream.
    private func scrubReasoning(_ text: String) -> String {
        selectedModel.template.stripReasoning(text)
    }

    /// Pull the lead paragraph for the top-3 search hits and add it
    /// as a `preview` field so the model can judge relevance from
    /// real article content, not the ~200-char BM25 snippet that
    /// Wikipedia/mdwiki full-text search returns. Makes a huge
    /// difference on disambiguation-prone queries (pizza → "origin"
    /// vs "Chicago-style"; plasma → "plasma actuators" vs
    /// "plasma (physics)").
    private func enrichSearchHits(_ result: [String: Any]) -> [String: Any] {
        guard var hits = result["hits"] as? [[String: Any]], !hits.isEmpty else { return result }
        let limit = min(hits.count, 3)
        for i in 0..<limit {
            guard let zim = hits[i]["zim"] as? String,
                  let path = hits[i]["path"] as? String,
                  let entry = library.first(where: {
                      $0.url.lastPathComponent == zim && $0.isEnabled
                  }),
                  let data = try? entry.reader.read(path: path)?.content,
                  let html = String(data: data, encoding: .utf8)
            else { continue }
            let stripped = ArticleSections.stripHTML(html)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            // First ~400 chars of the cleaned lead is enough signal
            // for the model to disambiguate without blowing the prompt.
            let preview = String(stripped.prefix(400))
            var updated = hits[i]
            if !preview.isEmpty {
                updated["preview"] = preview
            }
            hits[i] = updated
        }
        var out = result
        out["hits"] = hits
        return out
    }

    private func recordToolTrace(_ trace: ToolCallTrace) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].toolCalls.append(trace)
        }
    }

    // MARK: - Fast-path direct-dispatch router

    // Fast-path intent classification + reply synthesis live in
    // `MCPZimKit.IntentRouter` so they're exercised by `swift test`.
    // The iOS side just calls `IntentRouter.classify(...)` and passes
    // the resulting `DirectIntent` into `executeDirectIntent`.

    /// Article-sheet presentation intent — set by
    /// `presentArticleSheet` and observed by `PlacesWebView` to mount
    /// a native `.sheet(item:)` hosting a WKWebView of the article.
    /// Replaces the previous "dispatch `get_article_section` and
    /// render as a chat turn" flow, which showed "Results below." in
    /// the bubble and hid the actual article behind a tap.
    public struct ArticleSheetRequest: Equatable, Identifiable {
        public let id = UUID()
        public let zim: String
        public let path: String
        public let title: String
    }
    public var articleSheetIntent: ArticleSheetRequest? = nil

    /// Resolve the Wikipedia ZIM from the library and post an
    /// `ArticleSheetRequest`. `PlacesWebView` observes this via
    /// `.onChange(of:)` and mounts the sheet.
    public func presentArticleSheet(title: String, path: String) {
        let zimName = library
            .first(where: { $0.isEnabled && $0.reader.kind == .wikipedia })?
            .url.lastPathComponent ?? "wikipedia"
        articleSheetIntent = ArticleSheetRequest(
            zim: zimName, path: path, title: title
        )
    }

    /// Public entry for the "Read article" affordance on pin popups
    /// and list rows — dispatches `get_article_section(path, lead)`
    /// directly (no LLM roundtrip) and lets the existing
    /// `traceHasArticle` branch on `MessageRow.assistant` render the
    /// hero image + prose like any other article-bearing trace.
    public func triggerArticleRead(title: String, path: String) {
        guard setupState == .ready else {
            debug("triggerArticleRead ignored — setup still running",
                  category: "Chat")
            return
        }
        guard !path.isEmpty else {
            debug("triggerArticleRead: empty path, ignoring", category: "Chat")
            return
        }
        let caption = title.isEmpty
            ? "Read article at \(path)"
            : "Read \(title)"
        debug(caption, category: "User")
        messages.append(ChatMessage(role: .user, text: caption))
        messages.append(ChatMessage(role: .assistant, text: "", startedAt: Date()))
        isGenerating = true
        Task {
            let intent = DirectIntent(toolName: "get_article_section", args: [
                "path":    .string(path),
                "section": .string("lead")
            ])
            let ok = await executeDirectIntent(intent)
            if !ok {
                debug("triggerArticleRead: get_article_section dispatch failed",
                      category: "Tool")
            }
            isGenerating = false
            if let idx = messages.indices.last,
               messages[idx].role == .assistant
            {
                messages[idx].finishedAt = Date()
            }
        }
    }

    /// Public entry for the pin-popup Directions button. Appends a
    /// new user turn ("Directions to <name>") and dispatches our OWN
    /// `plan_driving_route` — not Apple Maps — against the exact
    /// lat/lon of the pin (no geocoding round-trip; the name is just
    /// the label shown in chat). Ends up with a route bubble in chat
    /// carrying the polyline + Drive/Walk/Bike pills.
    public func triggerDirectionsToCoord(
        name: String, lat: Double, lon: Double
    ) {
        guard setupState == .ready else {
            debug("triggerDirections ignored — setup still running",
                  category: "Chat")
            return
        }
        refreshLocationIfStale()
        guard let origin = currentLocation else {
            lastError = "Can't route — no GPS fix yet."
            return
        }
        let caption = name.isEmpty
            ? String(format: "Directions to (%.5f, %.5f)", lat, lon)
            : "Directions to \(name)"
        debug(caption, category: "User")
        messages.append(ChatMessage(role: .user, text: caption))
        messages.append(ChatMessage(role: .assistant, text: "", startedAt: Date()))
        isGenerating = true
        Task {
            let intent = DirectIntent(toolName: "plan_driving_route", args: [
                "origin_lat": .double(origin.lat),
                "origin_lon": .double(origin.lon),
                "dest_lat":   .double(lat),
                "dest_lon":   .double(lon)
            ])
            let ok = await executeDirectIntent(intent)
            if !ok {
                debug("triggerDirections: plan_driving_route dispatch failed",
                      category: "Tool")
            }
            isGenerating = false
            if let idx = messages.indices.last,
               messages[idx].role == .assistant
            {
                messages[idx].finishedAt = Date()
            }
        }
    }

    /// Run the dispatched tool, record the trace, and synthesize a
    /// one-line assistant caption. Returns `true` when the fast path
    /// was used successfully and the LLM should be skipped.
    @MainActor
    private func executeDirectIntent(_ intent: DirectIntent) async -> Bool {
        guard let adapter else { return false }
        let dictArgs = intent.anyArgs
        let argsStr: String = {
            guard let data = try? JSONSerialization.data(
                withJSONObject: dictArgs, options: [.sortedKeys]
            ), let s = String(data: data, encoding: .utf8) else { return "{}" }
            return s
        }()
        debug("fast-path dispatch \(intent.toolName)(\(argsStr))", category: "Tool")
        do {
            let fullResult = try await adapter.dispatch(
                tool: intent.toolName, args: dictArgs
            )
            let resultData = (try? JSONSerialization.data(
                withJSONObject: fullResult, options: [.sortedKeys]
            )) ?? Data()
            let rawStr = String(data: resultData, encoding: .utf8) ?? "{}"
            recordToolTrace(ToolCallTrace(
                name: intent.toolName,
                arguments: argsStr,
                result: rawStr,
                rawResult: rawStr,
                error: nil
            ))
            let placesTools: Set<String> = [
                "near_named_place", "near_places",
                "nearby_stories", "nearby_stories_at_place",
            ]
            let routingTools: Set<String> = [
                "route_from_places", "plan_driving_route",
            ]
            // Fast-path usability gate — if the tool technically
            // succeeded but didn't produce anything the user will
            // find useful (no articles found, no place resolved),
            // bail with `false` so the caller falls through to the
            // LLM loop. Saves the user from a dead-end fast-path
            // message when the model could at least try a different
            // approach.
            let usable: Bool = {
                switch intent.toolName {
                case "compare_articles":
                    return IntentRouter.compareResultIsUsable(fullResult)
                case "article_overview":
                    return IntentRouter.articleOverviewResultIsUsable(fullResult)
                case "what_is_here":
                    return IntentRouter.whatIsHereResultIsUsable(fullResult)
                default:
                    // Places + routing tools have their own
                    // empty-results handling in their synth.
                    return true
                }
            }()
            if !usable {
                debug("fast-path result not usable — handing off to LLM",
                      category: "Router")
                // Drop the trace row we just recorded so the LLM's
                // retry doesn't see a pre-populated tool call that
                // would confuse its dispatch state.
                if let idx = messages.indices.last,
                   messages[idx].role == .assistant
                {
                    messages[idx].toolCalls.removeAll()
                    messages[idx].text = ""
                }
                return false
            }
            if placesTools.contains(intent.toolName) {
                let synth = IntentRouter.synthesizePlacesReply(
                    toolName: intent.toolName,
                    args: dictArgs,
                    fullResult: fullResult
                )
                updateAssistant(synth.isEmpty ? "Results below." : synth)
            } else if routingTools.contains(intent.toolName) {
                let synth = Self.synthesizeRoutingReply(from: fullResult)
                updateAssistant(synth.isEmpty ? "Route below." : synth)
            } else if intent.toolName == "article_overview"
                   || intent.toolName == "compare_articles"
            {
                // The fast path dispatches the tool (saving iter 0's
                // ~13 s prefill-and-decide cost), then hands off to
                // the LLM to generate the prose. Synthesising a
                // caption ourselves works for simple places / routing
                // replies — a map bubble below carries the real
                // answer — but for compare / article_overview the
                // reply IS the prose, and the LLM needs to see the
                // tool result and summarise it.
                //
                // We inject a synthetic round-trip (assistant
                // tool-call emission + tool response) into the
                // transcript in the model's native wire format, then
                // return `false` so the caller falls through to
                // `runGenerationLoop`. That loop rebuilds the prompt,
                // sees the round-trip already done, and the model's
                // next emission is the summarising prose — no
                // second-tool-call round.
                let template = selectedModel.template
                let assistantEmission = template.formatToolCall(
                    name: intent.toolName, arguments: dictArgs
                )
                let trimmed = Self.trimForModel(
                    toolName: intent.toolName,
                    result: fullResult,
                    articleCapKB: self.articleCapKB
                )
                let toolResponse = template.formatToolResponse(
                    name: intent.toolName, payload: trimmed
                )
                recordToolRoundTrip(
                    assistantEmission: assistantEmission,
                    toolResponse: toolResponse
                )
                // Leave the assistant bubble empty — `runGenerationLoop`
                // will stream the prose into it.
                updateAssistant("")
                debug("fast-path injected \(intent.toolName) round-trip → LLM will summarise",
                      category: "Router")
                return false
            } else if intent.toolName == "what_is_here" {
                let synth = IntentRouter.synthesizeWhatIsHereReply(
                    fullResult: fullResult
                )
                updateAssistant(synth.isEmpty ? "Location below." : synth)
            } else {
                updateAssistant("Results below.")
            }
            return true
        } catch {
            debug("fast-path dispatch failed: \(error)", category: "Tool")
            if let idx = messages.indices.last,
               messages[idx].role == .assistant
            {
                messages[idx].text = ""
                messages[idx].toolCalls.removeAll()
            }
            return false
        }
    }

    // `synthesizePlacesReply` + its helpers moved to
    // `MCPZimKit.IntentRouter` so they're covered by `swift test`.

    /// Build a human-readable reply from a `route_from_places` /
    /// `plan_driving_route` result, without going through another
    /// model generate pass. Used by the `routingSkipModelReply`
    /// setting. Assumes the tool-adapter localised distances already
    /// (miles for US locale, km elsewhere) — reads the
    /// `distance_localized` + `duration_min` fields verbatim.
    static func synthesizeRoutingReply(from fullResult: [String: Any]) -> String {
        // Fields laid out by `MCPToolAdapter.encodeRoute` (raw) and
        // `trimForModel` (the model-facing trim that localises
        // units). We read the UNTRIMMED result to get the richer
        // fields; fall back gracefully when some are missing.
        var bits: [String] = []
        let distance: String? = {
            if let s = fullResult["distance_localized"] as? String { return s }
            if let km = fullResult["distance_km"] as? Double {
                return String(format: "%.1f km", km)
            }
            if let m = fullResult["distance_m"] as? Int {
                return String(format: "%.1f km", Double(m) / 1000)
            }
            return nil
        }()
        let duration: String? = {
            if let s = fullResult["duration_localized"] as? String { return s }
            if let min = fullResult["duration_min"] as? Double {
                return String(format: "%d min", Int(min.rounded()))
            }
            if let sec = fullResult["duration_s"] as? Int {
                return String(format: "%d min", max(1, Int((Double(sec) / 60).rounded())))
            }
            return nil
        }()
        if let d = distance, let t = duration {
            bits.append("Route: \(d), about \(t).")
        } else if let d = distance {
            bits.append("Route: \(d).")
        } else if let t = duration {
            bits.append("Route time: \(t).")
        }
        if let origin = (fullResult["origin_resolved"] as? [String: Any])?["name"] as? String,
           let dest = (fullResult["destination_resolved"] as? [String: Any])?["name"] as? String,
           !origin.isEmpty, !dest.isEmpty
        {
            bits.append("From \(origin) to \(dest).")
        }
        // Include the first 4 turn_by_turn steps (keep the reply
        // short — the map + full list are on-screen).
        if let turns = fullResult["turn_by_turn"] as? [String], !turns.isEmpty {
            let head = Array(turns.prefix(4))
            let rest = turns.count - head.count
            var steps = head
            if rest > 0 { steps.append("(\(rest) more steps — tap Directions for the full list)") }
            bits.append("Start: " + steps.joined(separator: "; "))
        } else if let totalTurns = fullResult["turn_by_turn_total"] as? Int, totalTurns > 0 {
            bits.append("About \(totalTurns) steps.")
        }
        return bits.joined(separator: " ")
    }

    /// Store a tool-call → tool-response round-trip on the current
    /// assistant message so subsequent turns can reconstruct the
    /// prompt byte-for-byte for KV-cache LCP matching.
    private func recordToolRoundTrip(assistantEmission: String, toolResponse: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].toolRoundTrips.append(
                ToolRoundTripEntry(assistantEmission: assistantEmission,
                                   toolResponseTurn: toolResponse)
            )
        }
    }

    // MARK: - Prompt formatting

    /// Drop context-heavy fields from a tool result before feeding it back
    /// into the next model turn. The original (untrimmed) payload is still
    /// what lands in the UI's tool trace — we only strip for the LLM.
    private static func trimForModel(toolName: String, result: [String: Any], articleCapKB: Int) -> [String: Any] {
        switch toolName {
        case "plan_driving_route", "route_from_places":
            var out = result
            // Attach a human-readable duration ("2h 32m", "45m", "1h") so
            // the model doesn't dump the raw minutes/seconds in the reply.
            // We also drop the raw second/minute fields to push the model
            // toward the formatted one.
            if let totalSeconds = (out["duration_s"] as? Double) ?? (out["duration_min"] as? Double).map({ $0 * 60 }) {
                out["duration"] = Self.formatDuration(seconds: totalSeconds)
                out["duration_s"] = nil
                out["duration_min"] = nil
            }
            // Respect the host's measurement system: emit a single
            // pre-formatted `distance` string ("104.8 mi" / "168.6 km") so
            // the model doesn't have to guess units or do conversion math
            // (E2B-sized models get it wrong roughly half the time).
            if let km = out["distance_km"] as? Double {
                out["distance"] = Self.formatDistance(km: km)
            }
            out["distance_km"] = nil
            out["distance_m"] = nil
            // Also convert per-road distances so the turn-by-turn list
            // matches the summary units.
            if var turns = out["turn_by_turn"] as? [String] {
                turns = turns.map { Self.localizeDistanceInTurnString($0) }
                out["turn_by_turn"] = turns
            }
            // Polyline: keep first/last points + length so the model knows
            // it exists, but don't feed thousands of lat/lons into its context.
            if let poly = out["polyline"] as? [[Double]] {
                let first = poly.first ?? []
                let last = poly.last ?? []
                out["polyline"] = ["points": poly.count, "first": first, "last": last]
            }
            // Turn-by-turn: keep only the first 8 instructions. On a
            // cross-metro route the model will otherwise dutifully
            // enumerate 30+ turns, which drives a 40 s generation that
            // grows Gemma's KV cache into jetsam territory on iPhone
            // and leaves the user listening to 2 minutes of street
            // names. 8 is enough for the "freeway summary + last few"
            // flavor the voice assistant should produce.
            if let turns = out["turn_by_turn"] as? [String], turns.count > 8 {
                out["turn_by_turn_total"] = turns.count
                out["turn_by_turn"] = Array(turns.prefix(8)) + ["… (\(turns.count - 8) more)"]
            }
            // Roads: same idea — cap at 8.
            if let roads = out["roads"] as? [[String: Any]], roads.count > 8 {
                out["roads_total"] = roads.count
                out["roads"] = Array(roads.prefix(8)) + [["name": "… (\(roads.count - 8) more)"]]
            }
            return out
        case "search":
            // Cap hit snippets so a full-text match flood doesn't blow context.
            if let hits = result["hits"] as? [[String: Any]], hits.count > 10 {
                var out = result
                out["hits_total"] = hits.count
                out["hits"] = Array(hits.prefix(10))
                return out
            }
            return result
        case "get_article":
            // Gemma 4 E2B has a 32 K-token context — ~96 KB of text — so we
            // can comfortably pass a ~24 KB article (~6 K tokens) and still
            // leave room for the system turn, tool declarations, and reply.
            // Below that cap, feed the article verbatim so the model can
            // actually summarise rather than complain about truncation.
            let capBytes = max(2, articleCapKB) * 1024
            if let text = result["text"] as? String, text.count > capBytes {
                var out = result
                out["text"] = String(text.prefix(capBytes)) + "\n… (truncated, \(text.count - capBytes) more bytes)"
                return out
            }
            return result
        case "article_overview", "compare_articles":
            // Wikipedia leads are designed to be standalone summaries —
            // the opening paragraphs carry the entire "what is this"
            // answer the LLM needs for a compare / overview pass.
            // Feeding additional sections just burns prompt budget
            // AND memory: on-device repro showed two full articles
            // (15–30 KB of raw text) jetsam'd the app mid-summary.
            //
            // Keep ONLY the lead section, truncated to ~160 words
            // (≈1000 chars ≈ 200 tokens). At two articles that's
            // ~400 extra tokens of prompt — safely under any
            // threshold, with enough room for a second or third
            // paragraph of the lead so the model has real
            // comparable material. Word-based truncation (vs char-
            // based) keeps words and trailing punctuation intact so
            // the model doesn't see "...founded in 19" or mid-entity
            // mangling at the boundary. The relations-article shape
            // (top-level `sections`) gets the same treatment.
            let leadWordCap = 160
            func keepLeadOnly(_ sections: [[String: Any]]) -> [[String: Any]] {
                guard let lead = sections.first else { return [] }
                var trimmedLead = lead
                if let text = lead["text"] as? String {
                    // Run word-based cap across the whole lead — not
                    // just the first paragraph. At 160 words we
                    // typically land inside paragraph 2 or 3, which
                    // for most Wikipedia leads is exactly the "enough
                    // context to actually compare" sweet spot.
                    let words = text.split(separator: " ",
                                           omittingEmptySubsequences: false)
                    let truncated = words.count > leadWordCap
                    let out = truncated
                        ? words.prefix(leadWordCap).joined(separator: " ") + "…"
                        : text
                    trimmedLead["text"] = out
                    if truncated || sections.count > 1 {
                        trimmedLead["truncated"] = true
                    }
                }
                return [trimmedLead]
            }
            var out = result
            if let sections = out["sections"] as? [[String: Any]] {
                out["sections"] = keepLeadOnly(sections)
            }
            if let articles = out["articles"] as? [[String: Any]] {
                out["articles"] = articles.map { a -> [String: Any] in
                    var inner = a
                    if let sections = a["sections"] as? [[String: Any]] {
                        inner["sections"] = keepLeadOnly(sections)
                    }
                    return inner
                }
            }
            return out
        default:
            return result
        }
    }

    /// `"2h 32m"`, `"45m"`, `"1h"` — whichever is most natural for the
    /// supplied duration. Passed to the model so it doesn't echo raw
    /// `duration_min: 152.48…`.
    private static func formatDuration(seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        let h = total / 3600
        let m = (total % 3600 + 30) / 60 // round minutes to nearest
        if h > 0 && m > 0 { return "\(h)h \(m)m" }
        if h > 0 { return "\(h)h" }
        return "\(m)m"
    }

    /// Format a distance in kilometres to the user's locale's preferred unit.
    /// Imperial (US, UK, Myanmar, Liberia) → miles. Metric elsewhere.
    /// Rounded to 1 decimal — the model's gonna echo this string verbatim.
    private static func formatDistance(km: Double) -> String {
        // Use the full word ("miles" / "kilometres") rather than the
        // abbreviation — the small Gemma E2B sometimes drops the number
        // when trying to expand "mi" to "miles" in prose, producing
        // "traveling ___ miles". Whole words are pronounced cleanly by
        // Kokoro TTS too, which otherwise botches bare "mi" / "km".
        if Self.useImperialDistance {
            let miles = km * 0.621371
            return "\(Self.round1(miles)) miles"
        } else {
            return "\(Self.round1(km)) kilometres"
        }
    }

    /// `turn_by_turn` entries from `MCPToolAdapter.encodeRoute` look like
    /// `"Žemaičių pl. (A1) for 0.50 km (~0.4 min)"`. Rewrite the `X.XX km`
    /// segment in-place to the host's preferred unit — leave the rest
    /// (road names, durations) untouched.
    private static func localizeDistanceInTurnString(_ s: String) -> String {
        guard Self.useImperialDistance else { return s }
        // Replace the LAST "N.NN km" occurrence — road names like "A2 km-5"
        // won't trip us up since they don't end with " km".
        guard let range = s.range(of: #"(\d+\.\d+)\s*km"#, options: .regularExpression) else { return s }
        let matched = String(s[range])
        let numberText = matched
            .replacingOccurrences(of: " km", with: "")
            .replacingOccurrences(of: "km", with: "")
        guard let km = Double(numberText) else { return s }
        let miles = km * 0.621371
        return s.replacingCharacters(in: range, with: "\(round1(miles)) mi")
    }

    private static func round1(_ v: Double) -> Double {
        (v * 10).rounded() / 10
    }

    /// `Locale.measurementSystem` is iOS 16 / macOS 13+. Cache at first use.
    private static let useImperialDistance: Bool = {
        Locale.current.measurementSystem == .us
            || Locale.current.measurementSystem == .uk
    }()

    /// Converts MCPToolRegistry schemas into Gemma-4 tool declarations.
    /// We only try to translate the top-level parameter shape; anything
    /// exotic (deeply nested objects, oneOf, etc.) falls through as a
    /// best-effort OBJECT so the model at least knows the param exists.
    ///
    /// For the `zim` parameter specifically, we also inject the actual
    /// list of loaded ZIM filenames as an enum — small models sometimes
    /// invent plausible names (`"streetzim"`, `"wikipedia"`) otherwise.
    private func toolDeclarations(registry: MCPToolRegistry) -> [ModelToolDeclaration] {
        let zimNames = library.filter { $0.isEnabled }.map { $0.url.lastPathComponent }
        return registry.tools.map { tool -> ModelToolDeclaration in
            let schema = (try? JSONSerialization.jsonObject(with: tool.inputSchemaJSON)) as? [String: Any] ?? [:]
            let properties = schema["properties"] as? [String: Any] ?? [:]
            let required = Set((schema["required"] as? [String]) ?? [])
            let params: [ModelToolDeclaration.Parameter] = properties.keys.sorted().map { key in
                let raw = (properties[key] as? [String: Any]) ?? [:]
                let typeStr = ((raw["type"] as? String) ?? "string").lowercased()
                let type: ModelToolDeclaration.Parameter.ParamType = {
                    switch typeStr {
                    case "integer": return .integer
                    case "number":  return .number
                    case "boolean": return .boolean
                    case "array":   return .array
                    case "object":  return .object
                    default:        return .string
                    }
                }()
                let description = raw["description"] as? String
                var enumValues = (raw["enum"] as? [Any])?.compactMap { $0 as? String }
                if key == "zim" && !zimNames.isEmpty {
                    enumValues = zimNames
                }
                return .init(
                    name: key,
                    type: type,
                    description: description,
                    required: required.contains(key),
                    enumValues: (enumValues?.isEmpty ?? true) ? nil : enumValues
                )
            }
            return .init(name: tool.name, description: tool.description, parameters: params)
        }
    }

    /// Produces the plain-text instructions the model sees at the top of the
    /// transcript. Providers wrap this in their own template — Gemma-4 folds
    /// it into the first user turn, the generic template emits a `<|system|>`
    /// block.
    private static func toolsPreamble(registry: MCPToolRegistry) -> String {
        var lines: [String] = [
            "You are a helpful assistant running on-device. You have access to the following tools.",
            "To call a tool, emit a single line:",
            "<tool_call>{\"name\":\"TOOL_NAME\",\"arguments\":{...}}</tool_call>",
            "and wait for the <tool_response> turn before continuing.",
            "",
            "Available tools:",
        ]
        for tool in registry.tools {
            let schema = String(data: tool.inputSchemaJSON, encoding: .utf8) ?? "{}"
            lines.append("- \(tool.name): \(tool.description)")
            lines.append("  input: \(schema)")
        }
        return lines.joined(separator: "\n")
    }

    /// Accept the selected model's native tool-call format first, then
    /// fall back to the generic `<tool_call>{…json…}</tool_call>` Mock
    /// and older prompts use. Whichever fires first wins.
    func extractToolCall(in buffer: String) -> (range: Range<String.Index>, name: String, args: [String: Any])? {
        if let m = selectedModel.template.firstToolCall(in: buffer) {
            return (m.range, m.name, m.arguments)
        }
        if let m = ChatToolCallParser.firstCall(in: buffer) {
            return (m.range, m.name, m.arguments)
        }
        return nil
    }
}
