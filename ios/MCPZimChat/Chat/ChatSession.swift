// SPDX-License-Identifier: MIT
//
// Top-level observable app state. Owns the list of opened ZIM readers, the
// set of available models, the current chat transcript, and a reference to
// the MCPZim tool adapter that a Gemma 4 tool loop can dispatch through.

import CoreLocation
import CryptoKit
import Foundation
import MCPZimKit
import Observation
import OSLog
#if canImport(UIKit)
import UIKit
#endif

private let chatLog = Logger(subsystem: "org.mcpzim.MCPZimChat", category: "Chat")

/// Synchronous adapter over section vectors prepared by
/// `NLContextualEmbedding`. `ArticleHeuristics` still supplies its strong
/// heading/keyword score; these vectors replace its hashing-only semantic
/// tiebreak without recomputing every section on each follow-up.
private struct PreparedDiscussionEmbedder: TextEmbedder {
    let dimension: Int
    let vectors: [String: [Float]]
    private let zero: [Float]

    init(vectors: [String: [Float]]) {
        self.vectors = vectors
        dimension = vectors.values.first?.count ?? 512
        zero = [Float](repeating: 0, count: dimension)
    }

    func embed(_ text: String) -> [Float] {
        vectors[Self.key(text)] ?? zero
    }

    static func key(_ text: String) -> String {
        Data(SHA256.hash(data: Data(text.utf8))).base64EncodedString()
    }
}

@MainActor
@Observable
public final class ChatSession {
    /// Stable id used by the macOS app menu for the laptop-class Bonsai
    /// operating point. The provider itself is registered only on macOS.
    public static let ternaryBonsai27BModelID = "bonsai-27b-q2-ternary-gguf"

    // MARK: - Library (opened ZIMs)

    public struct LibraryEntry: Identifiable, Sendable {
        public let id = UUID()
        public let url: URL
        public let reader: ZimReader
        public var isEnabled: Bool = true

        public init(
            url: URL,
            reader: ZimReader,
            isEnabled: Bool = true
        ) {
            self.url = url
            self.reader = reader
            self.isEnabled = isEnabled
        }

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
    /// True once the launch-time Documents scan and bookmark restore have
    /// completed. A cold "Open in Zimfo" event waits for this so the scan's
    /// replace operation cannot erase the newly opened external file.
    public private(set) var libraryBootstrapComplete = false

    // MARK: - Models

    public private(set) var models: [any ModelProvider]
    public var selectedModel: any ModelProvider
    public var modelState: ModelLoadState = .notLoaded
    /// Wall-clock stamp of when the current download started — stamped
    /// on the `.notLoaded → .downloading` transition and cleared on
    /// transitions back to `.notLoaded` / `.ready` / `.failed`. Used by
    /// `LibraryView` to render "Ns elapsed" alongside the Hub's percent
    /// so the UI stays honest when `fractionCompleted` coalesces and
    /// sits at 1% for minutes on large safetensors downloads.
    public var downloadStartedAt: Date? = nil

    #if os(macOS)
    /// Drives the Mac Models menu's honest Download/Use wording. The provider
    /// validates the pinned byte count, so a partial 7 GB transfer is never
    /// reported as cached.
    public var isTernaryBonsai27BCached: Bool {
        guard let provider = models.first(where: {
            $0.id == Self.ternaryBonsai27BModelID
        }) as? LlamaCppProvider else { return false }
        return provider.hasCompleteCachedGGUF
    }
    #endif

    // MARK: - Transcript

    public var messages: [ChatMessage] = []
    public var isGenerating = false
    public var lastError: String?
    private var generationTask: Task<Void, Never>?
    private var activeQueryTelemetry: AppTelemetry.QueryTrace?

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
    // Default OFF — this ships as a discussion app, not a dev console, so a
    // first launch shouldn't show tool traces + memory pressure. Turn it back
    // on in Library → Debug (that toggle also re-exposes the "Send Debug
    // Report" button used for the gist diagnostics workflow).
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

    /// Where we are in reading an article aloud, section by section.
    /// Set when `article_overview` / `narrate_article` runs; consumed +
    /// advanced when the user says "continue" / "keep reading" / "tell
    /// me more". `next` is the document-order index of the next section
    /// to narrate (0 = lead); `total` is the article's section count.
    /// Cleared whenever the user starts a different (non-continue) turn,
    /// so a later "continue" never resumes a stale article.
    private struct ReadingState {
        let title: String
        let zim: String?
        let total: Int
        var next: Int
    }
    private var readingState: ReadingState?

    /// Switchable operating points for the prepared-discussion experiment.
    /// The shipping app uses `semanticSections`; the Mac harness can select
    /// `none` to produce a same-model, same-retrieval lexical baseline without
    /// editing source or rebuilding the app.
    public enum DiscussionPreparationStrategy: String, Sendable {
        case none
        case semanticSections = "semantic-sections"
    }

    public struct DiscussionPreparationStats: Sendable {
        public let strategy: DiscussionPreparationStrategy
        public let title: String
        public let sectionCount: Int
        public let vectorCount: Int
        public let vectorBytes: Int
        public let elapsedSeconds: Double
    }

    @ObservationIgnored
    public var discussionPreparationStrategy: DiscussionPreparationStrategy =
        .semanticSections
    @ObservationIgnored
    public private(set) var lastDiscussionPreparationStats:
        DiscussionPreparationStats?

    /// Pinned topic for "let's discuss X" — grounded MULTI-article RAG.
    /// `sources[0]` is the anchor article; when a follow-up isn't covered by
    /// the articles in hand, the host searches the corpus and appends the
    /// best-matching article, so the discussion can range beyond a single
    /// page (population → Demographics of a different article; perovskites →
    /// a solar-cell section). Each turn the model sees only the few passages
    /// the question needs. Cleared when the user navigates away or says
    /// "stop". `topic` is the broad subject (topicCore of the anchor title)
    /// used to build corpus-fallback queries.
    private struct DiscussionState {
        let anchorTitle: String
        let topic: String
        let zim: String?
        var sources: [(title: String, sections: [ArticleSection])]
        /// Normalized titles of articles directly linked from any source
        /// currently in hand. Corpus expansion must follow one of these
        /// Wikipedia edges; lexical similarity alone is not enough.
        var linkedArticleTitles: Set<String>
        /// Exact-text hash → prepared semantic vector for section headings,
        /// bodies, and (per turn) the current question.
        var sectionEmbeddings: [String: [Float]]
        /// Raw prior user question. A chronological continuation such as
        /// “Then what happened in Soviet times?” inherits its facet for
        /// retrieval and answer framing without changing the visible text.
        var lastQuestion: String?
    }
    private var discussionState: DiscussionState?

    /// Exact grounded transcript retained across Wikipedia follow-ups. The
    /// llama.cpp provider can reuse a hybrid model's recurrent/KV state only
    /// when the next prompt is a strict append. Re-sending a freshly ranked
    /// standalone passage bundle changes the prompt near the beginning and
    /// forces Bonsai to prefill all ~2k tokens again. This cache instead
    /// appends only newly selected passages + the next question, while the
    /// provider keeps the matching prefix in RAM.
    private struct GroundedPromptCache {
        let topic: String
        let modelID: String
        let systemPreamble: String
        var turns: [ChatTurn]
        var passageKeys: Set<String>
    }
    private var groundedPromptCache: GroundedPromptCache?

    /// Tools whose intent means the user has LEFT a "discuss X" session
    /// (navigation/places, or a fresh discuss/compare). `article_overview`
    /// is intentionally absent: mid-discussion "tell me about its economy"
    /// is a question about the pinned article, not a topic change.
    private static func exitsDiscussion(_ toolName: String) -> Bool {
        switch toolName {
        case "route_from_places", "plan_driving_route", "near_places",
             "near_named_place", "what_is_here", "locate",
             "discuss_article", "compare_articles":
            return true
        default:
            return false
        }
    }

    /// Subject-aware exit check: navigation/compare tools always leave the
    /// pinned discussion; article tools leave only when their title names a
    /// DIFFERENT subject than the articles in hand. Real capture 2026-07-01:
    /// "Tell me about Donald Trump" while discussing Putin stayed pinned and
    /// answered from Putin's sections. "Tell me about Putin's wealth" (same
    /// subject) still stays grounded in the discussion.
    private func intentLeavesDiscussion(
        _ intent: DirectIntent, state: DiscussionState, userText: String
    ) -> Bool {
        if Self.exitsDiscussion(intent.toolName) { return true }
        let articleTools: Set<String> = [
            "article_overview", "article_factoid", "narrate_article",
            "get_article_section",
        ]
        guard articleTools.contains(intent.toolName),
              let raw = intent.anyArgs["title"] as? String
        else { return false }
        let title = raw.lowercased().trimmingCharacters(in: .whitespaces)
        guard !title.isEmpty else { return false }
        // Stateless routing can parse "Who were the combatants?" as an
        // article named "the combatants". Inside a pinned discussion this is
        // plainly a facet, not a request to abandon the current subject.
        if IntentRouter.isDiscussionFacetTitle(title) { return false }
        var inHand = state.sources.map { $0.title.lowercased() }
        inHand.append(state.topic.lowercased())
        inHand.append(state.anchorTitle.lowercased())
        // Same subject when either name contains the other
        // ("putin" ⊂ "vladimir putin").
        if inHand.contains(where: { $0.contains(title) || title.contains($0) }) {
            return false
        }
        // Explicit deictics are stronger than the stateless article parse:
        // "Buddhism there" must remain a Mongolia facet even when the router
        // happens to emit article_overview(title: "buddhism there").
        if IntentRouter.isDiscussionDeicticFollowUp(userText) {
            return false
        }
        // "Tell me about Y" is an explicit hand-off. Otherwise give the
        // prepared topic resistance: stateless routing often invents a fresh
        // article title for an ordinary facet question, but if the pinned
        // evidence covers the user's words we should keep the discussion.
        if IntentRouter.isExplicitDiscussionTopicChange(userText) {
            return true
        }
        if state.sources.contains(where: {
            ArticleHeuristics.sectionsCoverQuestion(
                $0.sections, userText, articleTitle: $0.title)
        }) {
            return false
        }
        // Elliptical "How/what about X?" phrasing inherits the prepared
        // subject when X is at least mentioned in that article. The stricter
        // coverage gate above requires repeated body evidence and therefore
        // let "And how about Christianity?" escape Lithuania to the global
        // Christianity article even though Lithuania's religion/history
        // sections contain the requested facet.
        if IntentRouter.isEllipticalDiscussionFollowUp(userText),
           state.sources.contains(where: {
               ArticleHeuristics.sectionsMentionQuestion(
                   $0.sections, userText, articleTitle: $0.title)
           }) {
            return false
        }
        return true
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
        // of headroom. Only raises the budget, never lowers it. Applies
        // to both the MLX (Gemma) and llama.cpp (LFM2.5) providers — the
        // latter's KV is fixed at n_ctx, so long replies are nearly free.
        if let floor = (selectedModel as? Gemma4Provider)?.replyTokensFloor
            ?? (selectedModel as? LlamaCppProvider)?.replyTokensFloor {
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
            // Bulk-drop with slack: removeFirst(k) shifts the whole array,
            // so trimming one row per call re-copied ~maxDebugEntries
            // elements on every debug() once the ring filled. Dropping a
            // chunk below the cap makes the shift run 1/slack as often;
            // nothing user-visible reads the oldest rows.
            let slack = min(63, maxDebugEntries / 8)
            debugEntries.removeFirst(debugEntries.count - (maxDebugEntries - slack))
        }
        // `print` is synchronous stdout on the @MainActor hot path (every
        // tool dispatch + generation stage) and only reaches a console
        // when Xcode is attached — Release builds get the same line via
        // os_log + LogArchive below.
        #if DEBUG
        print("[\(category)] \(decorated)")
        #endif
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
    /// Token for the LocationFetcher subscription, released in `deinit`
    /// so short-lived sessions (eval harness builds many per process)
    /// don't accumulate dead subscriber closures in the singleton.
    @ObservationIgnored private var locationSubscription: UUID?

    /// Conversational discourse state — what the conversation is *about*,
    /// the last enumerated list shown (for "the second one"), the vetted
    /// topic-drift threads, and a GPS movement trail. Lets follow-ups
    /// ("who built it", "the other one", "tell me more") resolve
    /// deterministically in `IntentRouter` / `ReferenceResolver` instead of
    /// relying on the small model's coreference. Mutated only on the main
    /// actor; `@ObservationIgnored` so updating it doesn't churn the view.
    @ObservationIgnored var focus = ConversationFocus()
    /// Predicate to reapply when the previous fast path asked the user to
    /// choose among ambiguous Wikipedia entities.
    @ObservationIgnored private var pendingFactoidPredicate: String?

    /// Incremental on-device semantic recall — embeds the lead of every
    /// article we open (via `SemanticReranker`'s `NLContextualEmbedding`) so
    /// drift offers can be re-ranked by similarity to the whole conversation.
    /// Empty at launch, grows with the walk, never bundled. Degrades cleanly
    /// to the deterministic thread order when the embedding model is cold.
    @ObservationIgnored private let embeddingIndex = EmbeddingIndex()

    /// Last location the local-area seed-index ran for, so we only re-seed
    /// after the user has walked into a genuinely new area (jitter/repeat-fix
    /// filter). nil until the first seed.
    @ObservationIgnored private var lastSeedLocation: (lat: Double, lon: Double)?
    /// Thread keys offered in recent replies, most-recent-last, capped.
    /// Stops the appended "Want to hear about X or Y?" line repeating the
    /// SAME offer turn after turn when the user ignores it — a fresh offer
    /// or silence both beat nagging.
    @ObservationIgnored private var recentlyOfferedThreadKeys: [String] = []

    /// Local-area seed-index tuning. Re-seed only after moving this far; pull
    /// wiki-backed places within this radius; cap how many we embed per area so
    /// a fresh fix never turns into a long burst of article reads.
    private static let seedReseedMeters: Double = 600
    private static let seedRadiusKm: Double = 2
    private static let seedMaxPlaces: Int = 30

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

    /// Non-blocking preparation of llama.cpp's invariant system + tool-schema
    /// prefix. The model is already usable while this runs: the composer stays
    /// live, deterministic/direct routes can answer immediately, and a turn
    /// that needs this same model serializes behind the worker safely.
    public enum PromptOptimizationState: Equatable, Sendable {
        case idle
        case checking
        case restoring
        case building(progress: Double)
        case ready
        case failed

        public var isActive: Bool {
            switch self {
            case .checking, .restoring, .building: return true
            case .idle, .ready, .failed: return false
            }
        }
    }
    public private(set) var promptOptimizationState:
        PromptOptimizationState = .idle
    @ObservationIgnored private var llamaPromptOptimizationTask:
        Task<Void, Never>?
    @ObservationIgnored private var llamaPromptOptimizationGeneration = UUID()

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
        libraryBootstrapComplete = true
        // Route LocationFetcher's auth-state events into the
        // in-app debug pane + the gist report so "why does it keep
        // prompting me" is answerable from the log rather than by
        // re-walking the CoreLocation state machine.
        LocationFetcher.debug = { [weak self] msg in
            Task { @MainActor [weak self] in
                self?.debug(msg, category: "Location")
            }
        }
        // Do not put a CoreLocation permission sheet over first launch. On a
        // fresh install it can race the first Speech recognizer/audio-engine
        // start and leave voice chat "listening" without recording until the
        // process is restarted. Authorized installs still start updates now;
        // undetermined installs prompt after the first navigational request
        // has already been transcribed and submitted.
        LocationFetcher.start()
        refreshLocationIfStale()
        prewarmBackgroundCaches()
        await runSetupIfNeeded()
        // `runSetupIfNeeded` now starts llama.cpp's static-prefix work here,
        // after the model becomes usable, without blocking the composer.
        // Direct routes can still answer immediately; a generic LLM turn
        // waits for this one coalesced worker instead of racing it. Unlike the
        // older launch-only KV prime, the resulting sequence state is saved
        // to Application Support and normally restores in milliseconds.
    }

    /// Warm the expensive start-up caches off the user's hot path.
    /// Called from `RootView` at launch. Intentionally concurrent —
    /// streetzim graph parse, reranker asset load, and location fix
    /// are all independent, so there's no point serialising them.
    /// Finish blocking model setup, then start the static system + tool-schema
    /// prefix as a separate background phase. The chat UI becomes available
    /// as soon as the model is ready; the persistent prefix is an optimization,
    /// not a launch gate.
    @MainActor
    public func runSetupIfNeeded() async {
        guard setupState == .pending else { return }
        setupState = .running(stage: "Loading model…", progress: nil)
        // Wait for the model download/load kicked off during init. A fresh
        // Bonsai install is several gigabytes and legitimately takes longer
        // than the old 60-second deadline; timing it out made a healthy first
        // download look broken. Stay here for as long as bytes are flowing,
        // expose honest progress, and stop only on a provider failure.
        modelWait: while true {
            switch modelState {
            case .ready:
                break modelWait
            case .downloading(let fraction):
                setupState = .running(
                    stage: "Downloading \(selectedModel.displayName) — \(Int(fraction * 100))%\nOne-time download; keep Zimfo open.",
                    progress: fraction
                )
            case .loading:
                setupState = .running(
                    stage: "Opening \(selectedModel.displayName)…",
                    progress: nil
                )
            case .notLoaded:
                setupState = .running(
                    stage: "Preparing \(selectedModel.displayName)…",
                    progress: nil
                )
            case .failed(let message):
                setupState = .failed(message)
                return
            }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        // The blocking setup ends as soon as the model can accept work. For
        // llama.cpp, build/restore the expensive invariant system+tools state
        // behind the composer: a user can type immediately, and the resulting
        // SSD snapshot survives future launches. This is materially different
        // from the old Gemma/MLX launch prime below, whose Metal allocation
        // compounded with TTS and caused jetsam on phones.
        setupState = .ready
        startLlamaPromptOptimizationIfNeeded()
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
        cancelLlamaPromptOptimization()
        // Keep llama.cpp states. Their key includes the runtime, model,
        // context/KV configuration, exact system prompt, and exact generated
        // tool schemas; a mismatched state is therefore unreachable and is
        // pruned by LRU later. Deleting every `.bin` here made harmless model
        // or library toggles throw away a 70-second cache build.
        let dir = promptCacheDirectory()
        if let files = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)
        {
            for f in files where
                f.pathExtension == "safetensors"
                || f.pathExtension == "json"
            {
                try? FileManager.default.removeItem(at: f)
            }
        }
    }

    /// Let the user reach Settings after a model download/load failure. The
    /// chat header still shows the failed model and its Load retry button;
    /// unlike the old no-op "Continue anyway" control, this actually removes
    /// the blocking launch overlay.
    public func dismissSetupFailure() {
        guard case .failed = setupState else { return }
        setupState = .ready
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

    private func llamaPromptCacheDirectory() -> URL {
        let fm = FileManager.default
        // This state takes roughly a minute to rebuild on the phone. Store it
        // in Application Support so iOS does not purge it as disposable cache
        // data; each file is separately excluded from iCloud backup by the
        // provider after its atomic save.
        let base = (try? fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true))
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = base.appendingPathComponent(
            "LlamaPromptCache", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)

        // One-time, best-effort migration from builds that stored the exact
        // same keyed snapshots in Library/Caches. Preserve an already-migrated
        // destination and let the normal LRU remove obsolete variants.
        if let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let legacy = caches.appendingPathComponent(
                "LlamaPromptCache", isDirectory: true)
            if legacy != dir,
               let files = try? fm.contentsOfDirectory(
                   at: legacy, includingPropertiesForKeys: nil)
            {
                for source in files where source.pathExtension == "bin" {
                    let destination = dir.appendingPathComponent(
                        source.lastPathComponent)
                    if !fm.fileExists(atPath: destination.path) {
                        try? fm.moveItem(at: source, to: destination)
                    }
                }
            }
        }
        return dir
    }

    private func llamaPromptCacheURL(
        provider: LlamaCppProvider,
        prefixPrompt: String
    ) -> URL {
        // Manual format/runtime tag is intentional. llama.cpp state files
        // are not a stable interchange format; changing the bundled Prism
        // runtime must produce a fresh file even when the prompt/model did
        // not change.
        let raw = [
            // v2 stores only sequence 0 (KV + recurrent state), omitting
            // unrelated whole-context state such as logits and RNG.
            "mcpzim-llama-prefix-sequence-state-v2-prism-b9591",
            provider.id,
            provider.ggufFilename,
            String(provider.expectedGGUFBytes ?? -1),
            String(provider.contextTokens),
            provider.kvCacheType.rawValue,
            prefixPrompt,
        ].joined(separator: "\n")
        let key = Self.sha256Hex(raw)
        return llamaPromptCacheDirectory()
            .appendingPathComponent("llama-\(key.prefix(24)).bin")
    }

    public func prewarmBackgroundCaches() {
        #if !MCPZIM_EVAL
        if SpeechRecognizerFactory.prewarmIfAuthorized() {
            debug("prewarmed on-device speech recognizer (no microphone access)",
                  category: "Voice")
        }
        #endif
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
        // Location is deliberately NOT part of this system turn. It changes
        // with GPS drift and used to invalidate Bonsai's hybrid recurrent
        // cache before the tool declarations. Generic navigational fallbacks
        // capture it on the user message instead; ordinary knowledge turns
        // no longer carry ~1,000 tokens of irrelevant routing recipes. Reply
        // length is also per-user-turn context below, keeping this one prefix
        // byte-identical across the Longer Replies toggle.
        return Self.composeSystemMessage(categoryHint: "", locationLine: "")
    }

    private func modelContextForCurrentTurn() -> String? {
        var blocks: [String] = []
        if longerReplies {
            blocks.append("""
            === Reply length preference ===
            The user enabled Longer replies. For broad, explanatory, or
            biographical questions, give a substantially fuller answer:
            usually four to eight sentences covering the main context,
            chronology, and consequences supported by retrieved evidence.
            Short factoids and routing answers should remain short. Do not pad
            or repeat yourself.
            """)
        }
        if lastQueryComplexity == .navigational {
            blocks.append(locationLineText())
        }
        return blocks.isEmpty ? nil : blocks.joined(separator: "\n\n")
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
        1. `article_overview(title: "X")` — returns the lead plus the \
           most informative section headings with their excerpts. For \
           most single-entity queries this is the WHOLE answer — skip \
           the search step.
        2. Only if `article_overview` doesn't have the angle the user \
           asked for, follow up with `get_article_section(title, \
           section: "<heading from the overview's section list>")`. \
        3. Answer from the sections you read. Write in natural prose — \
           DO NOT open with "per the 'lead' section" or "according to \
           the article"; the user already knows the answer is grounded. \
           Only name a section when it genuinely clarifies.

        When the question is a short factoid, one `article_overview` \
        is usually enough. When it's broader ("tell me about the \
        French Revolution"), the overview's section list tells you \
        which follow-up fetches are worthwhile. `search` is a fallback \
        — use it only for ambiguous topics where you don't know the \
        article title.

        === Follow-up rule (critical for conversational flow) ===
        When the CURRENT user turn is a short follow-up — starts with \
        "wait", "so", "then", "why", "how", "but", "what about", \
        "and", "ok", "also", or is under 10 words — FIRST check \
        whether the answer is already in the prior turns of this \
        conversation:
        * If an `article_overview` / `get_article_section` / \
          `compare_articles` call in the LAST 2 ASSISTANT TURNS \
          already contains the content needed to answer, \
          **ANSWER DIRECTLY from that context — DO NOT call any tool \
          again.** Cite inline if helpful. \
        * Only call a tool when the follow-up genuinely needs \
          information you HAVEN'T already fetched this conversation. \
          "Path length" / "inverse fourth power" / "what does that \
          imply" are things you can answer from cached context. \
          "Was there an earlier experiment?" is a new fact — fetch.
        * Re-fetching an article you already have in context on a \
          clarify turn is WRONG. It wastes the user's time and \
          doesn't improve grounding, because the cached content is \
          the same source.

        === Grounding policy ===
        This app's value to the user is that answers are grounded in \
        the loaded ZIM archives — not in your training priors. So: \
        * Every factual claim in your reply should trace to a tool \
          result you have seen this turn OR an earlier turn in this \
          conversation. \
        * Cite section / article names inline (e.g. "per 'Article' § \
          Causes…") whenever a claim isn't obviously common knowledge. \
        * If the loaded ZIMs genuinely don't cover the question, say \
          that — do not guess.\(locationLine)
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

    /// Coordinate frozen into the preamble for this conversation. The
    /// preamble is the FIRST turn of the prompt, so any byte change there
    /// invalidates the entire KV prefix — and the shipping hybrid models
    /// can't partially truncate, so a moving user would pay a full 4–15 s
    /// re-prefill EVERY turn if live GPS were interpolated directly
    /// (PERFORMANCE_REVIEW.md A1). Refreshed only after real movement;
    /// tool dispatch separately substitutes precise live coordinates via
    /// `substituteCurrentLocation`, so the preamble never needed meter
    /// precision.
    private var preambleLocationSnapshot: (lat: Double, lon: Double)?
    /// Set after a location wait times out with no fix; suppresses further
    /// pre-turn GPS waits for the session (a late fix still flows in via
    /// the normal delegate path and clears nothing — it just means future
    /// waits are unnecessary anyway).
    private var locationFixTimedOut = false
    /// Movement past this distance re-freezes the preamble coordinate
    /// (one deliberate re-prefill, matching the seed-reseed hysteresis).
    private let preambleLocationRefreshMeters: Double = 300

    private func locationLineText() -> String {
        guard let live = currentLocation else {
            return """

            === Current location ===
            Location permission hasn't resolved yet. If the user asks \
            a location-relative question, tell them you can't get a \
            fix right now rather than guessing coordinates.
            """
        }
        let here: (lat: Double, lon: Double)
        if let snap = preambleLocationSnapshot,
           GeoMath.haversineMeters(snap.lat, snap.lon, live.lat, live.lon)
               < preambleLocationRefreshMeters
        {
            here = snap
        } else {
            preambleLocationSnapshot = (live.lat, live.lon)
            here = (live.lat, live.lon)
        }
        // %.3f ≈ 110 m — consistent with the hundred-meter accuracy the
        // app actually requests from CoreLocation.
        let latStr = String(format: "%.3f", here.lat)
        let lonStr = String(format: "%.3f", here.lon)
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
          * "where is <place>?" / "show me <place>" / "find \
            <place>" → `locate(place="<place>")` — resolves the one \
            named place and drops a pin on it. NOT `near_places` \
            (that lists what's nearby) and NOT `get_article_section` \
            (that's a Wikipedia article, not a map location).
          * "map of where I am" / "what neighborhood is this" → \
            `locate(place="<the nearest named place>")`, OR \
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
        // A named `place` (e.g. near_places(place:"Berkeley")) is also
        // location-like: injecting our coords here would make the tool
        // adapter prefer them over geocoding the place, scanning near the
        // user instead of Berkeley. (A "my location" synonym in `place`
        // was already rewritten to a "lat,lon" string above, which the
        // adapter parses back to coords — so skipping injection is safe.)
        let placeString = (out["place"] as? String)?
            .trimmingCharacters(in: .whitespaces) ?? ""
        let hasMeaningfulPlaceString = !placeString.isEmpty
        if isProximityTool && !hasNumericOrigin && !hasMeaningfulOriginString
            && !hasMeaningfulPlaceString {
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
        // FIRST line of every session: did the previous run die mid-work?
        // Two on-device llama.cpp deaths (2026-07-02) left no system crash
        // report — the previous log's tail is the only evidence, so shout
        // it here where the debug pane, OSLog, and the persisted archive
        // all see it.
        if let tail = LogArchive.shared.previousSessionUncleanTail() {
            let msg = "⚠️ PREVIOUS SESSION ENDED UNCLEANLY (no clean background/terminate). Last lines of \(tail)"
            print(msg)
            LogArchive.shared.append(msg)
        }
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
        // Gemma 3 4B IT text-only 4-bit — the mlx-swift-lm-compatible
        // quant. `mlx-community/gemma-3-4b-it-4bit` is the multimodal
        // `Gemma3ForConditionalGeneration` wrapper and its packed 4-bit
        // weights mismatch our vendored `Gemma3TextModel`'s
        // `o_proj` shape (seen on 2026-04-23: expectedShape [2560,128]
        // vs actualShape [2560,256]). The `-text-4b-it-` variant is the
        // text-only checkpoint designed for the mlx-swift-lm `Gemma3Text`
        // path and loads cleanly. Benched 7/9 on the mac-only eval
        // scorecard; dense attention (no Qwen 3.5 MambaCache reuse bug).
        let gemma3_4b = Gemma4Provider(
            id: "gemma3-4b-it-text-4bit",
            displayName: "Gemma 3 4B IT (4-bit · text)",
            huggingFaceRepo: "mlx-community/gemma-3-text-4b-it-4bit",
            approximateMemoryMB: 2700,
            template: Gemma3Template()
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
        // the mostly-linear layers. Same `QwenChatMLTemplate` hard
        // non-thinking prefill — our tool-call parser accepts all
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
        // Gemma 3 4B IT Q4_K_M via llama.cpp. The memory-first path we
        // ported to on 2026-04-23 after confirming the MLX provider
        // peaks ~6.3 GB on multi-turn 5-6k-token prompts (iPhone 17
        // Pro Max jetsams at ~6 GB). llama.cpp + iSWA rotation-pruning
        // + Q8_0 KV keeps the same model under 3.2 GB peak at 20k
        // tokens. See tools/llama-smoke/RESULTS_2026-04-23_SEQ.md +
        // LlamaCppProvider.swift header for the full numbers.
        let gemma3_4b_gguf = LlamaCppProvider(
            id: "gemma3-4b-it-q4km-gguf",
            displayName: "Gemma 3 4B IT (Q4_K_M · llama.cpp)",
            huggingFaceRepo: "bartowski/google_gemma-3-4b-it-GGUF",
            ggufFilename: "google_gemma-3-4b-it-Q4_K_M.gguf",
            approximateMemoryMB: 3200,
            template: Gemma3Template()
        )
        // LoRA-fine-tuned Gemma 3 4B IT — tool-calling-grounded variant
        // (V7C, run 2026-04-25). Same Q4_K_M / iSWA-pruning / Q8_0 KV
        // layout as stock so memory profile is identical (~3.2 GB peak).
        // V7C scored 10/13 on the llama-smoke eval grid vs stock's 6/13
        // (+4: gained sky_is_blue, nearby_stories, tell_me_about,
        // french_revolution, crispr; lost narrate_hp_garage). Default
        // for new installs and migrated for existing stock-Q4 picks.
        // See tools/llama-smoke/GRID_RESULTS_FT_V7C.md.
        let gemma3_4b_gguf_ft = LlamaCppProvider(
            id: "gemma3-4b-it-q4km-gguf-ft",
            displayName: "Gemma 3 4B IT FT (Q4_K_M · llama.cpp)",
            huggingFaceRepo: "sliderforthewin/gemma-3-4b-it-ft-GGUF",
            ggufFilename: "gemma-3-4b-it-ft.Q4_K_M.gguf",
            approximateMemoryMB: 3200,
            template: Gemma3Template()
        )
        // LoRA-fine-tuned LFM2.5-8B-A1B (v7-full, run 2026-05-29). 8.3B
        // total / 1.5B active hybrid MoE. Trained on the SAME tool-call
        // corpus as the Gemma 3 4B FT (train_v4_combined + chain-heavy +
        // 317 targeted hard-case rows) so it drives the JSON tool format
        // via LFM25Template (ChatML markers, Gemma-3 body/parse logic).
        // Quant: IQ3_XS with an importance matrix calibrated on our own
        // tool-call transcripts (2026-06-10) — a strict Pareto win over the
        // previous Q3_K_M: same stable 12/13 on the llama-smoke grid,
        // 3.64 GB peak RSS (−0.53 GB), and ~+24% decode (136 vs 110 t/s on
        // the b9434 runtime) because the bandwidth-bound MoE gains more from
        // smaller weights than it loses to i-quant decode cost. Recipe +
        // sweep table: tools/llama-smoke/LFM25_MEMORY_PERF_FRONTIER.md.
        // The id keeps the legacy "q3km" so existing devices' model
        // selection persists across the quant swap.
        let lfm25_ft = LlamaCppProvider(
            id: "lfm2.5-8b-a1b-q3km-gguf-ft",
            displayName: "LFM2.5 8B-A1B FT (IQ3_XS · llama.cpp)",
            huggingFaceRepo: "sliderforthewin/lfm2.5-8b-a1b-ft-GGUF",
            ggufFilename: "lfm2.5-8b-a1b-ft.imx.IQ3_XS.gguf",
            // llama.cpp KV is fixed at n_ctx, so long replies are nearly free
            // here — give the FT room for thorough/discuss answers (and a
            // <think> preamble) instead of the device's TTS-tuned default.
            replyTokensFloor: 1024,
            approximateMemoryMB: 3700,
            // 32k window (model trains to 131k). LFM2.5's KV is cheap — only
            // 6 of 24 layers are attention (8 KV-heads × 64 dim) ≈ 6.9 KB/tok
            // at q8_0 → ~226 MB resident for the whole window, paid for by
            // the IQ3_XS requant (−0.53 GB). Turn latency stays flat via the
            // provider's cross-turn KV prefix reuse. Gemma GGUF fallbacks
            // keep 8k. Full budget math: CONTEXT_BUDGET.md.
            contextTokens: 32768,
            template: LFM25Template()
        )
        // Phone-class Bonsai 27B operating point. The same 1-bit weights
        // scored at near-parity with the ternary reference in our 16-scenario
        // Mac conversational suite; deterministic Swift routing handles the
        // one observed wrong-tool turn. Prism reports ~5.2 GB llama.cpp peak
        // at 4K context, versus ~5.9 GB for MLX before app/UI overhead, so the
        // GGUF runtime gives mcpzim the safer side of the iPhone 17 Pro Max's
        // 6144 MB per-process ceiling. The pinned Prism XCFramework provides
        // the packed Q1_0_g128 Metal kernels — weights stay packed on GPU.
        let bonsai27b_1bit = LlamaCppProvider(
            id: "bonsai-27b-q1-gguf",
            displayName: "Bonsai 27B (1-bit · Metal)",
            huggingFaceRepo: "prism-ml/Bonsai-27B-gguf",
            ggufFilename: "Bonsai-27B-Q1_0.gguf",
            expectedGGUFBytes: 3_803_452_480,
            replyTokensFloor: 512,
            approximateMemoryMB: 5500,
            // Bonsai has only 16 attention layers and uses Q4 KV: 16K costs
            // ~288 MB total KV, just ~144 MB more than the verified 8K build.
            // Ordinary grounded chat still rolls at 6K for speed; the larger
            // allocation is safety capacity for large tools and long turns.
            contextTokens: 16384,
            kvCacheType: .q4_0,
            // Qwen 3.6's official instruct/non-thinking recipe. The former
            // 1.0 / 0.95 setting is Qwen's thinking-mode recipe; it scored
            // one extra turn in an early 22-turn A/B, but the July 16 device
            // capture also showed reasoning bleed and a duplicated answer.
            // Use the mode-matched recipe for direct, grounded conversation.
            samplingProfile: GenerationSamplingProfile(
                temperature: 0.7, topP: 0.8, topK: 20,
                presencePenalty: 1.5),
            template: QwenChatMLTemplate()
        )
        var providers: [any ModelProvider] = [
            bonsai27b_1bit,
            lfm25_ft,
            gemma3_4b_gguf_ft,
            gemma3_4b_gguf,
            gemma, gemmaText, gemma3_4b, qwen3_4b, qwen35_4b, qwen3_1_7b,
        ]
        #if os(macOS)
        // Quality-oriented Bonsai operating point for Macs. This is the
        // publisher's native packed ternary Q2_0_g128 build, not a generic
        // low-bit requant: our pinned Prism llama.cpp XCFramework contains
        // the matching Q2_0 Metal kernels and Qwen 3.5 hybrid architecture.
        // The exact 7,165,121,600-byte size is pinned so an interrupted
        // multi-GB download can never be mistaken for a loadable GGUF.
        // Prism measures ~8.4 GB peak at 4K with uncompressed KV; Q4 KV and
        // a 32K allocation keep this comfortable on the development Mac but
        // far beyond iOS's per-process memory budget.
        let ternaryBonsai27b = LlamaCppProvider(
            id: Self.ternaryBonsai27BModelID,
            displayName: "Bonsai 27B Ternary (2-bit · Metal · Mac)",
            huggingFaceRepo: "prism-ml/Ternary-Bonsai-27B-gguf",
            ggufFilename: "Ternary-Bonsai-27B-Q2_0.gguf",
            expectedGGUFBytes: 7_165_121_600,
            replyTokensFloor: 1024,
            approximateMemoryMB: 9000,
            contextTokens: 32768,
            kvCacheType: .q4_0,
            samplingProfile: GenerationSamplingProfile(
                temperature: 0.7, topP: 0.8, topK: 20,
                presencePenalty: 1.5),
            template: QwenChatMLTemplate()
        )
        // Keep the two Bonsai operating points adjacent at the top of every
        // model menu. Selecting this entry invokes the provider's resumable
        // Hugging Face download and switches to it once loading completes.
        providers.insert(ternaryBonsai27b, at: 1)

        // No MLX Bonsai picker entry: the 2026-07-19 quant×runtime A/B
        // (docs/BONSAI_MLX_VS_LLAMACPP.md) settled it — llama.cpp decodes
        // 2-3.5× faster on Apple Silicon with equal answer quality, so the
        // MLX operating points exist only in the eval harness
        // (`--probe-discuss --runtime mlx`), not the user-facing menu.

        // Gemma 3 12B IT QAT-4bit — mac-only reference model. Benched 9/9
        // on the mac-only eval scorecard (perfect tool-calling across
        // every scenario), but peak memory scales from 9.2 GB @ 7k to
        // 13.4 GB @ 40k tokens, well past the iPhone 17 Pro Max 6 GB
        // jetsam cap. Useful on the Mac app for A/B comparisons against
        // the 4B candidates and as an upper bound the smaller models
        // can be measured against. See ON_DEVICE_MODEL_REPORT_2026-04-23.md.
        // Pull the text-only 12B quant to match the text-only 4B choice —
        // same `Gemma3TextModel` load path, no multimodal wrapper to
        // mismatch on.
        let gemma3_12b = Gemma4Provider(
            id: "gemma3-12b-it-text-4bit",
            displayName: "Gemma 3 12B IT (4-bit · text · mac)",
            huggingFaceRepo: "mlx-community/gemma-3-text-12b-it-4bit",
            approximateMemoryMB: 9200,
            template: Gemma3Template()
        )
        providers.append(gemma3_12b)
        #endif
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
        // Default flipped to Gemma 3 4B Q4_K_M via llama.cpp on
        // 2026-04-23. See tools/llama-smoke/RESULTS_2026-04-23_SEQ.md.
        // The MLX variants of Gemma 3 blew past the iPhone 17 Pro Max
        // 6 GB jetsam cap on multi-turn prefill; llama.cpp's
        // iSWA-rotation pruning + Q8_0 KV + flash-attn keeps the same
        // model at 3.2 GB peak even at 20k-token context. Qwen 3 4B
        // MLX is kept as a picker alternative (4.6 GB peak, correct
        // but tight) for anyone who wants to A/B the runtimes.
        //
        // On iOS, silently redirect a previously-saved Gemma-3-MLX
        // choice to the llama.cpp variant — we'd rather the user land
        // on a model that doesn't crash than on "the one they picked
        // last time". They can still repick the MLX one via the
        // picker to reproduce the crash.
        #if os(iOS)
        let crashesOnDevice: Set<String> = [
            "gemma3-4b-it-text-4bit",
            "gemma3-12b-it-text-4bit",
        ]
        // Migrate previous-default stock Gemma 3 4B Q4 picks to the FT
        // variant (V7C, +4 vs stock on the eval grid). Same memory
        // footprint and template; pickers can repick stock to A/B.
        var resolvedId: String? = savedId
        // Select Bonsai once on the development phone so this build actually
        // exercises the new runtime instead of silently reopening the prior
        // LFM/Gemma choice. Subsequent launches and manual picker changes are
        // preserved normally.
        // DEBUG-only: on TestFlight/App Store installs this would force a
        // ~5.5 GB-peak model onto EVERY first launch — at or over the
        // 6 GB jetsam line before TTS/WebKit overhead, and a much larger
        // download than the shipping LFM2.5 default
        // (PERFORMANCE_REVIEW.md H1). Release fresh installs fall back to
        // lfm25_ft below; the dev phone keeps exercising Bonsai.
        #if DEBUG
        let bonsaiSelectionMigrationKey = "chat.didSelectBonsai27B1BitV2"
        if !defaults.bool(forKey: bonsaiSelectionMigrationKey) {
            resolvedId = bonsai27b_1bit.id
            defaults.set(bonsai27b_1bit.id, forKey: Self.selectedModelKey)
            defaults.set(true, forKey: bonsaiSelectionMigrationKey)
        } else if crashesOnDevice.contains(savedId ?? "") {
            resolvedId = gemma3_4b_gguf_ft.id
        } else if savedId == gemma3_4b_gguf.id {
            resolvedId = gemma3_4b_gguf_ft.id
        }
        self.selectedModel = providers.first(where: { $0.id == resolvedId }) ?? bonsai27b_1bit
        #else
        if crashesOnDevice.contains(savedId ?? "") {
            resolvedId = gemma3_4b_gguf_ft.id
        } else if savedId == gemma3_4b_gguf.id {
            resolvedId = gemma3_4b_gguf_ft.id
        }
        self.selectedModel = providers.first(where: { $0.id == resolvedId }) ?? lfm25_ft
        #endif
        #else
        // Give the Mac app the same initial Bonsai operating point as the
        // phone. This one-time migration also moves existing development
        // installs off an older saved Gemma choice, while subsequent manual
        // picker changes remain persistent.
        var resolvedId: String? = savedId
        let macBonsaiSelectionMigrationKey = "chat.didSelectBonsai27B1BitMacV1"
        if !defaults.bool(forKey: macBonsaiSelectionMigrationKey) {
            resolvedId = bonsai27b_1bit.id
            defaults.set(bonsai27b_1bit.id, forKey: Self.selectedModelKey)
            defaults.set(true, forKey: macBonsaiSelectionMigrationKey)
        }
        self.selectedModel = providers.first(where: { $0.id == resolvedId }) ?? bonsai27b_1bit
        #endif
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
        // llama.cpp runs its blocking decode loop off-main. Forward its
        // stage-level performance telemetry through ChatSession so it lands
        // in the debug pane, OSLog, and the persistent crash-safe archive.
        for p in providers {
            guard let prov = p as? LlamaCppProvider else { continue }
            prov.debugSink = { [weak self] msg in
                Task { @MainActor [weak self] in
                    self?.debug(msg, category: "LlamaCpp")
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
        locationSubscription = LocationFetcher.subscribe { [weak self] coord in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.currentLocation = (coord.latitude, coord.longitude)
                self.focus.updateLocation(lat: coord.latitude, lon: coord.longitude)
                // Proactively embed wiki-backed places around the new fix so
                // semantic recall works for where the user physically is, not
                // just articles they've opened. Throttled + fire-and-forget.
                self.seedNearbyPlacesIfMoved(
                    lat: coord.latitude, lon: coord.longitude)
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

    deinit {
        // On-device the session lives for the app's lifetime, but eval
        // harnesses build one per variant — release the LocationFetcher
        // subscription so the singleton doesn't accumulate dead closures.
        // Same guard as the subscription site in init.
        #if canImport(UIKit)
        if let token = locationSubscription {
            LocationFetcher.unsubscribe(token)
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
        initialModelId: String? = nil,
        discussionPreparationStrategy: DiscussionPreparationStrategy =
            .semanticSections
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
        session.discussionPreparationStrategy = discussionPreparationStrategy
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
                // Stamp/clear download start time on state transitions
                // so the UI can show elapsed seconds during the window
                // where Hub progress is stuck at 1%.
                switch state {
                case .downloading:
                    if self.downloadStartedAt == nil {
                        self.downloadStartedAt = Date()
                    }
                case .notLoaded, .loading, .ready, .failed:
                    self.downloadStartedAt = nil
                }
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
        await addReaders(urls: urls, invalidatePromptCache: true)
    }

    private func addReaders(
        urls: [URL],
        invalidatePromptCache: Bool
    ) async {
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
        // next launch. Bookmark restoration is different: it reconstructs
        // the same persisted library on every process launch, so deleting
        // the keyed llama.cpp state here would make SSD restoration
        // impossible. Explicit library edits still take this path with
        // invalidation enabled.
        if !fresh.isEmpty, invalidatePromptCache {
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
            await addReaders(
                urls: urls,
                invalidatePromptCache: false)
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
        // A toggle can change the available tool registry (for example when
        // the last StreetZIM is disabled), so recompute the exact static
        // prefix key and restore/build its matching state.
        invalidateSetupCache()
        Task { await runSetupIfNeeded() }
    }

    // MARK: - Model switching

    public func select(modelId: String) async {
        guard let found = models.first(where: { $0.id == modelId }) else { return }
        if found.id == selectedModel.id {
            // A menu selection is also the user's retry gesture. Previously
            // this returned unconditionally, leaving a failed/not-loaded
            // model stuck while the menu appeared to do nothing.
            guard !modelState.isReady else {
                debug("\(found.displayName) is already selected and ready", category: "Load")
                return
            }
            debug("Retrying \(found.displayName)…", category: "Load")
            await loadSelectedModel()
            if modelState.isReady {
                invalidateSetupCache()
                await runSetupIfNeeded()
            }
            return
        }
        // Unload the previous model — iOS memory budget is tight.
        cancelLlamaPromptOptimization()
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
        startLlamaPromptOptimizationIfNeeded()
    }

    @ObservationIgnored private var kvPrewarmTask: Task<Void, Never>?

    /// Start the persistent llama.cpp static-prefix preparation without
    /// making it part of blocking app setup. Repeated calls coalesce; if the
    /// model/library changed while an older worker is unwinding, one fresh
    /// worker is scheduled as soon as the old lock is released.
    @MainActor
    private func startLlamaPromptOptimizationIfNeeded() {
        guard setupState == .ready,
              modelState.isReady,
              !isGenerating,
              groundedPromptCache == nil,
              !messages.contains(where: { $0.role == .user }),
              let llama = selectedModel as? LlamaCppProvider,
              let adapter
        else { return }
        if llamaPromptOptimizationTask != nil {
            llamaPromptOptimizationRestartRequested = true
            return
        }

        llamaPromptOptimizationRestartRequested = false
        let generation = UUID()
        llamaPromptOptimizationGeneration = generation
        let modelID = selectedModel.id
        promptOptimizationState = .checking
        llamaPromptOptimizationTask = Task { @MainActor [weak self, weak llama] in
            guard let self, let llama else { return }
            defer {
                self.llamaPromptOptimizationTask = nil
                if self.llamaPromptOptimizationRestartRequested,
                   self.setupState == .ready
                {
                    self.llamaPromptOptimizationRestartRequested = false
                    self.startLlamaPromptOptimizationIfNeeded()
                }
            }

            let registry = await adapter.registry
            guard !Task.isCancelled,
                  generation == self.llamaPromptOptimizationGeneration,
                  modelID == self.selectedModel.id
            else { return }

            let toolDecls = self.toolDeclarations(registry: registry)
            let template = self.selectedModel.template
            let systemMessage = self.systemMessageText(for: .topical)
            let prefix = template.bos + template.formatSystemTurn(
                systemMessage: systemMessage, tools: toolDecls)
            let cacheURL = self.llamaPromptCacheURL(
                provider: llama, prefixPrompt: prefix)
            // The provider only evaluates `prefix`; this throwaway turn proves
            // at the tokenizer level that it is an exact prefix of a real
            // request before any state is restored or saved.
            let validationPrompt = template.renderTranscript(
                systemPreamble: systemMessage,
                tools: toolDecls,
                turns: [ChatTurn(
                    role: .user,
                    text: "__zimfo_static_prefix_validation__")])
            let hasDiskState = FileManager.default.fileExists(
                atPath: cacheURL.path)
            self.promptOptimizationState = hasDiskState
                ? .restoring
                : .building(progress: 0)
            self.debug(
                hasDiskState
                    ? "background prefix optimization: restoring persisted state…"
                    : "background prefix optimization: building persisted state…",
                category: "LlamaCpp")

            do {
                let result = try await llama.preparePromptPrefix(
                    prefixPrompt: prefix,
                    fullPrompt: validationPrompt,
                    cacheURL: cacheURL,
                    progress: { [weak self] fraction in
                        Task { @MainActor [weak self] in
                            guard let self,
                                  generation == self.llamaPromptOptimizationGeneration
                            else { return }
                            self.promptOptimizationState = .building(
                                progress: min(1, max(0, fraction)))
                        }
                    })
                guard generation == self.llamaPromptOptimizationGeneration,
                      modelID == self.selectedModel.id
                else { return }
                self.promptOptimizationState = .ready
                self.debug(String(format:
                    "background prefix optimization ready: %@ · %d tok · %.1f MB · %.3fs",
                    result.mode, result.tokens,
                    Double(result.bytes) / 1_048_576,
                    result.seconds),
                    category: "LlamaCpp")
            } catch is CancellationError {
                self.debug("background prefix optimization cancelled",
                           category: "LlamaCpp")
            } catch {
                guard generation == self.llamaPromptOptimizationGeneration
                else { return }
                self.promptOptimizationState = .failed
                self.debug("background prefix optimization unavailable: \(error)",
                           category: "LlamaCpp")
            }
        }
    }

    @ObservationIgnored private var llamaPromptOptimizationRestartRequested = false

    @MainActor
    private func cancelLlamaPromptOptimization() {
        llamaPromptOptimizationGeneration = UUID()
        llamaPromptOptimizationRestartRequested = false
        llamaPromptOptimizationTask?.cancel()
        promptOptimizationState = .idle
    }

    /// A compact grounded discussion prompt cannot use the generic
    /// system+12-tools prefix. If that background build still owns llama.cpp,
    /// waiting for it adds tens of seconds without helping this answer. Stop
    /// it at the next prefix batch; a later fresh conversation can prepare
    /// the persisted generic prefix again.
    @MainActor
    private func preemptLlamaPromptOptimizationForGroundedTurn() {
        guard llamaPromptOptimizationTask != nil else { return }
        debug("grounded turn preempting background prefix optimization",
              category: "LlamaCpp")
        cancelLlamaPromptOptimization()
    }

    /// A generic tool-dispatch turn uses this exact prefix. Wait for an
    /// already-running build instead of racing a second restore/build against
    /// the same llama.cpp context. Direct routes remain free to complete while
    /// the worker runs.
    @MainActor
    private func awaitLlamaPromptOptimizationIfNeeded() async {
        guard selectedModel is LlamaCppProvider,
              let task = llamaPromptOptimizationTask
        else { return }
        debug("question waiting for background prefix optimization",
              category: "LlamaCpp")
        await task.value
        debug("question continuing after prefix optimization",
              category: "LlamaCpp")
    }

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
            let template = self.selectedModel.template
            let cat = template.logCategory
            // Skip the launch-time prewarm for families whose MLX
            // model can't safely reuse the KV cache anyway. On
            // Gemma 3 our `hasStaleScratchStateBug` guard forces
            // full prefill every turn (see Gemma4Provider.generate
            // + mlx-swift-lm#157). The prewarm is burning 2 GB
            // transient RSS (observed 2026-04-23 on device — 5 GB
            // peak on launch, memory warnings flooding) for a cache
            // that will never be reused. Skip it.
            if template.hasStaleScratchStateBug {
                self.debug(
                    "skipping KV prewarm — \(cat) forces full prefill every turn",
                    category: cat
                )
                return
            }
            let registry = await adapter.registry
            let toolDecls = await MainActor.run { self.toolDeclarations(registry: registry) }
            let preamble = await MainActor.run { self.systemMessageText(for: .topical) }
            // Build exactly the byte-prefix an iter-0 prompt starts
            // with: `<bos>` + system-turn. No user turn, no trailing
            // model-open — the upcoming send's encode will land on
            // the same first N tokens and LCP-hit.
            let systemTurn = template.formatSystemTurn(
                systemMessage: preamble, tools: toolDecls
            )
            let prompt = template.bos + systemTurn
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
        // Discourse state is conversation-scoped — a "new chat" must forget
        // it, or the next turn stays pinned to the old topic (real bug
        // 2026-05-30: a fresh "how do solar panels work?" was answered "I
        // don't see it in Lithuanian history" because discussionState
        // survived the clear). Offer history too, or the first offers of
        // the new chat get suppressed as "already offered".
        readingState = nil
        discussionState = nil
        lastDiscussionPreparationStats = nil
        groundedPromptCache = nil
        preambleLocationSnapshot = nil
        focus.reset()
        pendingFactoidPredicate = nil
        recentlyOfferedThreadKeys.removeAll()
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
        if let llama = selectedModel as? LlamaCppProvider {
            llama.resetPromptCache()
        }
        debug("conversation reset", category: "Chat")
    }

    /// Stop the active turn at the provider's next safe boundary. We keep
    /// `isGenerating` true until the worker actually unwinds, preventing a
    /// new turn from racing a still-owned llama.cpp context.
    public func stopGeneration() {
        guard isGenerating else { return }
        debug("user requested generation stop", category: "Chat")
        selectedModel.cancelGeneration()
        generationTask?.cancel()
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
        guard !isGenerating else {
            debug("send() ignored — a turn is still generating", category: "Chat")
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
        let enabledLibraries = library.filter(\.isEnabled)
        let queryTelemetry = AppTelemetry.startQuery(
            type: complexity.rawValue,
            modelID: selectedModel.id,
            library: AppTelemetry.LibraryProfile(
                kinds: enabledLibraries.map { $0.kind.rawValue },
                filenames: enabledLibraries.map { $0.url.lastPathComponent }))
        activeQueryTelemetry = queryTelemetry
        let user = ChatMessage(role: .user, text: text)
        messages.append(user)
        messages.append(ChatMessage(role: .assistant, text: "", startedAt: Date()))
        // Advance the discourse-state turn counter so entities/threads
        // recorded this turn stamp the right recency.
        focus.beginUserTurn()
        isGenerating = true
        generationTask = Task {
            defer {
                let responseCharacters = messages.last?.role == .assistant
                    ? (messages.last?.text.count ?? 0)
                    : 0
                queryTelemetry.finish(
                    cancelled: Task.isCancelled,
                    responseCharacters: responseCharacters)
                if activeQueryTelemetry === queryTelemetry {
                    activeQueryTelemetry = nil
                }
            }
            // Request location lazily, after voice recognition has finished
            // submitting the turn. General Wikipedia questions never need to
            // see a location sheet; nearest-place/directions questions do.
            // Waiting here also lets the deterministic near_places router see
            // the newly granted coordinate instead of falling into the LLM.
            // Skip the wait entirely when a fix can never arrive: denied
            // authorization, or a previous wait already timed out this
            // session (airplane mode, no GPS) — otherwise every
            // navigational turn stalls the full 4 s before prefill
            // (PERFORMANCE_REVIEW.md D7).
            if complexity == .navigational, currentLocation == nil,
               !LocationFetcher.isAuthorizationDenied,
               !locationFixTimedOut {
                debug("location needed — requesting permission/fix after transcription",
                      category: "Location")
                LocationFetcher.requestAuthorizationIfNeeded()
                LocationFetcher.start()
                await awaitLocationIfAny(maxWait: 4)
                if let here = currentLocation {
                    debug(String(format: "location ready: (%.5f, %.5f)",
                                 here.lat, here.lon), category: "Location")
                } else {
                    locationFixTimedOut = true
                    debug("location unavailable after 4s — continuing without it (and skipping future waits this session)",
                          category: "Location")
                }
            }
            let pendingFactoidIntent: DirectIntent? = {
                guard let predicate = pendingFactoidPredicate else {
                    return nil
                }
                // A pending clarification applies to one turn only. If this
                // isn't a list pick, ordinary routing handles the fresh query.
                pendingFactoidPredicate = nil
                return IntentRouter.factoidSelectionIntent(
                    text, predicate: predicate, focus: focus)
            }()
            // A tentative factoid may be tried while deciding whether a turn
            // leaves a pinned discussion. If that probe misses and the turn
            // then falls through to normal routing, do not dispatch the exact
            // same intent a second time before invoking the general model.
            var failedDiscussionIntent: DirectIntent?
            // "Read the whole article" is an action on the focused
            // Wikipedia subject, not a question for discussion mode. Handle
            // it before the pinned-discussion branch so the full article goes
            // straight to the narration pass-through instead of asking the
            // model to answer from its compact evidence window.
            if let readingIntent = IntentRouter.readArticleIntent(
                text, focus: focus),
               await executeDirectIntent(readingIntent)
            {
                isGenerating = false
                if let idx = messages.indices.last,
                   messages[idx].role == .assistant
                {
                    messages[idx].finishedAt = Date()
                }
                return
            }
            // "Continue" / "keep reading" / "tell me more" — page the
            // next chunk of the article we're currently reading aloud.
            // Checked before `classify` so a bare "continue" never tries
            // to geocode "continue". Only fires when there's an active
            // reading position; a bare "more" with none (e.g. after a
            // places search) falls straight through to normal routing.
            let wantsContinue = IntentRouter.isContinueReading(text)
            if wantsContinue, readingState != nil {
                activeQueryTelemetry?.setRoute("reading")
                await continueReadingArticle()
                isGenerating = false
                if let idx = messages.indices.last,
                   messages[idx].role == .assistant
                {
                    messages[idx].finishedAt = Date()
                }
                return
            }
            // Any other turn abandons an in-progress read — the article
            // tools below re-establish it if this turn is itself a
            // reading request (`noteReadingState`).
            if !wantsContinue { readingState = nil }

            // Explicit source control outranks both the pinned discussion and
            // stateless location patterns. This is the user's recovery path
            // when routing selected the wrong corpus or a disambiguation page:
            // "Use the Wikipedia article on Santa Rosa, California and tell
            // me what it says about the 1906 earthquake." Load that exact
            // article's full section set, re-anchor the discussion, and answer
            // the requested facet from it.
            if let sourceDirective = IntentRouter.wikipediaSourceDirective(
                text, focus: focus)
            {
                discussionState = nil
                groundedPromptCache = nil
                debug("explicit source directive: Wikipedia article “\(sourceDirective.title)”"
                    + (sourceDirective.question.map { " · question=\($0)" } ?? ""),
                    category: "Router")
                _ = await handleWikipediaSourceDirective(sourceDirective)
                activeQueryTelemetry?.setRoute("source_directive")
                isGenerating = false
                if let idx = messages.indices.last,
                   messages[idx].role == .assistant {
                    messages[idx].finishedAt = Date()
                }
                return
            }

            if let intent = pendingFactoidIntent {
                debug("factoid clarification selected — reapplying original predicate",
                      category: "Router")
                let handled = await executeDirectIntent(intent)
                if handled {
                    isGenerating = false
                    if let idx = messages.indices.last,
                       messages[idx].role == .assistant {
                        messages[idx].finishedAt = Date()
                    }
                    return
                }
            }

            // Discussion mode: while an article is pinned ("let's discuss
            // X"), answer follow-ups from its sections instead of routing
            // each turn afresh. An explicit "stop" or a navigation / new-
            // topic intent exits; anything else is a grounded question
            // about the pinned article.
            if let ds = discussionState, !wantsContinue {
                if IntentRouter.isDiscussionExit(text) {
                    activeQueryTelemetry?.setRoute("discussion")
                    discussionState = nil
                    groundedPromptCache = nil
                    updateAssistant("Okay — we can stop there. What next?")
                    isGenerating = false
                    if let i = messages.indices.last, messages[i].role == .assistant {
                        messages[i].finishedAt = Date()
                    }
                    return
                }
                let switchIntent = IntentRouter.classify(
                    text, currentLocation: currentLocation, focus: focus)
                // A founding-date factoid is grounded and deterministic even
                // inside a pinned discussion. Execute it here instead of
                // handing the same article lead to Bonsai for another long
                // prefill. If it names a fresh subject, clear the old pin;
                // if it says "when was it founded?", retain/re-anchor it.
                if let intent = switchIntent,
                   intent.toolName == "article_factoid" {
                    if intentLeavesDiscussion(intent, state: ds, userText: text) {
                        discussionState = nil
                        groundedPromptCache = nil
                    }
                    let handled = await executeDirectIntent(intent)
                    if handled {
                        isGenerating = false
                        if let i = messages.indices.last,
                           messages[i].role == .assistant {
                            messages[i].finishedAt = Date()
                        }
                        return
                    }
                    failedDiscussionIntent = intent
                    // A same-topic evidence miss can still use the richer
                    // pinned sections. A new-topic miss falls through to the
                    // normal router below instead of querying the old article.
                    if discussionState != nil {
                        await answerWithinDiscussion(ds, question: text)
                        isGenerating = false
                        if let i = messages.indices.last,
                           messages[i].role == .assistant {
                            messages[i].finishedAt = Date()
                        }
                        return
                    }
                } else if let intent = switchIntent,
                          intentLeavesDiscussion(intent, state: ds, userText: text) {
                    discussionState = nil   // topic change → normal routing below
                    groundedPromptCache = nil
                } else {
                    await answerWithinDiscussion(ds, question: text)
                    isGenerating = false
                    if let i = messages.indices.last, messages[i].role == .assistant {
                        messages[i].finishedAt = Date()
                    }
                    return
                }
            }

            // Genuine ambiguity → ask, don't guess. When a descriptive
            // selector ("the church") matches SEVERAL items from the
            // list we just showed, the resolver flags `.ambiguous` —
            // previously that fell through to the stateless patterns,
            // which guessed (`article_overview(title: "the church")`).
            // A deterministic clarifying question is faster and right;
            // `focus.lastList` stays intact so the user's pick ("the
            // second one" / the name) resolves next turn.
            let reference = ReferenceResolver.resolve(text, focus: focus)
            if case .ambiguous(let candidates) = reference.binding,
               candidates.count > 1 {
                activeQueryTelemetry?.setRoute("clarification")
                let names = candidates.prefix(3).map(\.name)
                let list = names.count == 2
                    ? "\(names[0]) or \(names[1])"
                    : names.dropLast().joined(separator: ", ")
                        + ", or \(names.last!)"
                updateAssistant("Which one do you mean — \(list)?")
                debug("ambiguous reference (\(names.joined(separator: " / "))) — asked for clarification",
                      category: "Router")
                isGenerating = false
                if let idx = messages.indices.last,
                   messages[idx].role == .assistant
                {
                    messages[idx].finishedAt = Date()
                }
                return
            }

            // Fast-path intent router. Match a small set of simple
            // user patterns ("<category> in <place>", "directions to
            // <place>", "<category> near me") and dispatch the tool
            // directly — no model generate, no 13 s prefill. If no
            // pattern matches, fall through to the full LLM loop.
            // Logic lives in `MCPZimKit.IntentRouter` so it's covered
            // by `swift test` (see `IntentRouterTests`).
            if let intent = IntentRouter.classify(
                text, currentLocation: currentLocation, focus: focus
            ) {
                if intent == failedDiscussionIntent {
                    debug("fast-path intent already missed during discussion switch — falling back to LLM",
                          category: "Router")
                } else {
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
            }
            // Only a genuine generic-model fallback needs the per-turn
            // location block. Capture it on the visible user message without
            // changing the bubble text; subsequent transcript rebuilds then
            // reproduce the exact bytes that entered the provider cache.
            if let idx = messages.lastIndex(where: { $0.role == .user }),
               messages[idx].modelContext == nil
            {
                messages[idx].modelContext = modelContextForCurrentTurn()
            }
            await awaitLlamaPromptOptimizationIfNeeded()
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
            // Drift: end the reply by offering 1–3 vetted threads
            // (related wikilinks / nearby places) surfaced from this
            // turn's tool results, so the conversation can keep moving.
            await appendThreadOfferIfUseful()
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
        activeQueryTelemetry?.setRoute("llm")
        defer {
            isGenerating = false
            if let idx = messages.indices.last, messages[idx].role == .assistant {
                messages[idx].finishedAt = Date()
            }
        }
        debug("runGenerationLoop: entered", category: "Chat")
        // Location-dependent turns already request/wait immediately after
        // transcription in `send`. Topical Wikipedia turns deliberately do
        // not wait for GPS; doing so added four seconds to ordinary questions
        // whenever permission was absent.
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
        let llamaStaticPrefix: (
            provider: LlamaCppProvider,
            prompt: String,
            cacheURL: URL
        )? = {
            guard let llama = selectedModel as? LlamaCppProvider else {
                return nil
            }
            let template = selectedModel.template
            let prompt = template.bos + template.formatSystemTurn(
                systemMessage: systemMessage, tools: toolDecls)
            return (
                provider: llama,
                prompt: prompt,
                cacheURL: llamaPromptCacheURL(
                    provider: llama, prefixPrompt: prompt))
        }()

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
                turns.append(ChatTurn(
                    role: msg.role.asChatTurnRole,
                    text: msg.modelText))
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

        // Bounded history window. The transcript used to be fed back
        // in FULL every turn — on a long walking session the prompt
        // grew without limit until the memory-headroom guard below
        // aborted the turn ("reset the conversation"). Cap it at the
        // last `maxExchanges` user exchanges, trimming in CHUNKS
        // (down to `keepExchanges`) rather than sliding one turn at a
        // time, so the prompt prefix — and with it the provider's
        // KV-cache LCP match — stays stable for several turns between
        // trims. Older subjects aren't lost to the conversation: the
        // discourse state (`focus`) carries entities/threads
        // deterministically outside the transcript.
        let maxExchanges = 10
        let keepExchanges = 6
        let userTurnIdxs = turns.indices.filter { turns[$0].role == .user }
        if userTurnIdxs.count > maxExchanges {
            let cut = userTurnIdxs[userTurnIdxs.count - keepExchanges]
            turns.removeFirst(cut)
            debug("history window: \(userTurnIdxs.count) exchanges > \(maxExchanges) — dropped \(cut) oldest turns, keeping last \(keepExchanges) exchanges",
                  category: "Chat")
        }

        // TOKEN-budget guard on top of the exchange-count window: a few
        // article-heavy round-trips can overflow n_ctx long before 10
        // exchanges (real capture 2026-07-02: 6 turns hit 10.3k tokens
        // and generate() threw "exceeds n_ctx", leaving an empty reply).
        // Drop oldest exchanges while the estimated prompt exceeds the
        // provider's window minus the reply + preamble reservation.
        // ~3.3 chars/token is conservative for English + JSON payloads.
        let contextTokens = (selectedModel as? LlamaCppProvider)?.contextTokens ?? 8192
        let promptTokenBudget = max(2048, contextTokens - effectiveMaxReplyTokens - 512)
        let charBudget = promptTokenBudget * 3
        func turnsChars() -> Int {
            turns.reduce(systemMessage.count + 2048) { $0 + $1.text.count + 16 }
        }
        // Trim to a LOWER watermark in one step, mirroring the count-based
        // path above. Trimming minimally (just under budget) means a session
        // that reaches the budget slides the window start on EVERY
        // subsequent turn — the prompt prefix diverges each time, and on
        // hybrid models that means a full ~30k-token re-prefill every turn,
        // forever (PERFORMANCE_REVIEW.md A3). Cutting to 75% buys several
        // stable-prefix turns per trim.
        if turnsChars() > charBudget {
            let watermark = charBudget * 3 / 4
            var dropped = 0
            // Running total: re-running the full reduce after every dropped
            // exchange made this loop O(turns²) in transcript size.
            var total = turnsChars()
            while total > watermark {
                let userIdxs = turns.indices.filter { turns[$0].role == .user }
                // Keep at least the current exchange (last user turn onward).
                guard userIdxs.count > 1 else { break }
                let cut = userIdxs[1]
                for i in 0..<cut { total -= turns[i].text.count + 16 }
                turns.removeFirst(cut)
                dropped += 1
            }
            debug("history window: token budget — dropped \(dropped) oldest exchange(s) to watermark (\(total) chars vs \(charBudget) budget)",
                  category: "Chat")
        }

        // Up to 6 tool loops per user turn — enough for small models
        // that burn iterations exploring (small search → wrong zim →
        // retry) before landing on a useful answer. Still capped so
        // a genuinely stuck loop terminates.
        let maxIters = 6
        var toolLoopGuard = ToolLoopGuard()
        var forcedSummaryReason: String?
        toolLoop: for iter in 0..<maxIters {
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

            // Every provider renders via its OWN template, which folds the
            // static systemMessage and tool schemas into the model's TRAINED
            // prompt shape. A generic navigational fallback carries its
            // dynamic `=== Current location ===` block on the user turn. The
            // LFM2.5 and Gemma-3-gguf FTs were trained on exactly this
            // (system prose + JSON tool block folded into the first user
            // turn) — also what the llama-smoke eval scores 12/13 on. The old
            // `toolsPreamble` + formatTranscript path fed them an
            // off-distribution generic format, so even with the location
            // present in the prompt they ignored it and refused "nearest X".
            var prompt = selectedModel.template.renderTranscript(
                systemPreamble: systemMessage,
                tools: toolDecls,
                turns: turns
            )
            // Exact guard after templating: the earlier character estimate
            // cannot account precisely for tool schemas or tokenizer density
            // (real Bonsai capture: 5,877 tokens reached a 4K provider even
            // after the approximate history trim). Remove whole old exchanges
            // while preserving the current one, then reserve the full reply
            // budget so decode never walks off the end of n_ctx.
            if let llama = selectedModel as? LlamaCppProvider {
                let budget = max(256,
                    llama.contextTokens - effectiveMaxReplyTokens - 32)
                var exactTokens = llama.promptTokenCount(prompt)
                while let count = exactTokens, count > budget {
                    // Shed enough whole exchanges to clear the budget (with
                    // 5% slack) BEFORE paying the next render + tokenize —
                    // the old loop re-rendered and re-tokenized the full
                    // ~10k-token prompt once per dropped exchange. The
                    // outer `while` re-checks the exact count, so the
                    // char-density estimate only sizes the batch.
                    let charsPerToken = max(1.0, Double(prompt.count) / Double(count))
                    var estimate = count
                    var droppedAny = false
                    while estimate > budget * 95 / 100 {
                        let userIdxs = turns.indices.filter {
                            turns[$0].role == .user
                        }
                        guard userIdxs.count > 1 else { break }
                        let cut = userIdxs[1]
                        var cutChars = 0
                        for i in 0..<cut { cutChars += turns[i].text.count + 16 }
                        turns.removeFirst(cut)
                        estimate -= Int(Double(cutChars) / charsPerToken)
                        droppedAny = true
                    }
                    guard droppedAny else { break }
                    prompt = selectedModel.template.renderTranscript(
                        systemPreamble: systemMessage,
                        tools: toolDecls,
                        turns: turns)
                    exactTokens = llama.promptTokenCount(prompt)
                    debug("exact context guard: dropped oldest exchange(s); \(exactTokens ?? -1)/\(budget) prompt tokens",
                          category: "Chat")
                }
                if let count = exactTokens, count > budget {
                    let msg = "This request needs \(count) prompt tokens, but \(selectedModel.displayName) has \(budget) available after reserving its reply. Try a new conversation or a narrower request."
                    debug("exact context guard: refusing oversized current exchange (\(count)/\(budget))",
                          category: "Chat")
                    lastError = msg
                    updateAssistant(msg)
                    return
                }
                debug("exact context guard: \(exactTokens ?? -1)/\(budget) prompt tokens",
                      category: "Chat")
            }
            // Yield to SwiftUI before iter 0 so the prior assistant's
            // `PlacesWebView` / `RouteWebView` (guarded by
            // `isLatestAssistant` in ChatView.MessageRow) has time to
            // swap to `MapPlaceholder` and WebKit can release its
            // ~700–1000 MB of tile/JS buffers BEFORE our prefill
            // allocates another ~1 GB of Metal activations. Without
            // this, the follow-up voice query after "Bars in SC"
            // stacked the live places map on top of a 5,726-token
            // cache-miss prefill and jetsammed the app at ~5 GB RSS
            // (2026-04-23, confirmed via mcp-logs tail + new
            // `cache MISS — no existing cache` diagnostic). Only yields
            // on iter 0 — tool-response round-trips inside the same
            // turn don't remount the WebView, so no need to delay.
            if iter == 0 {
                let hasPriorAssistant = messages
                    .dropLast()
                    .contains(where: { $0.role == .assistant && !$0.toolCalls.isEmpty })
                if hasPriorAssistant {
                    debug("yielding 250ms for prior map WebView teardown before prefill",
                          category: "Chat")
                    try? await Task.sleep(nanoseconds: 250_000_000)
                    debug("map teardown wait done", category: "Chat")
                }
            }
            if let prefix = llamaStaticPrefix {
                do {
                    let result = try await prefix.provider.preparePromptPrefix(
                        prefixPrompt: prefix.prompt,
                        fullPrompt: prompt,
                        cacheURL: prefix.cacheURL)
                    debug(String(format:
                        "static prefix cache: %@ · %d tok · %.1f MB · %.3fs",
                        result.mode, result.tokens,
                        Double(result.bytes) / 1_048_576,
                        result.seconds),
                        category: "LlamaCpp")
                } catch {
                    // Cache acceleration is opportunistic. Generation's own
                    // LCP/reset logic remains the correctness fallback.
                    debug("static prefix cache unavailable: \(error)",
                          category: "LlamaCpp")
                }
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
            // Every tool-call opener across templates starts with "<" or a
            // "```" fence, so until one of those characters has streamed in
            // there is nothing for the parsers to find — skip the per-chunk
            // whole-buffer scans (they made plain-prose streaming O(n²) in
            // reply length). Once seen, scan per chunk as before.
            var sawMarkerChar = false
            do {
                for try await chunk in selectedModel.generate(prompt: prompt, parameters: params) {
                    buffer += chunk
                    chunkCount += 1
                    let now = Date()
                    if now.timeIntervalSince(lastUIPush) >= uiMinInterval {
                        appendToAssistant(buffer)
                        lastUIPush = now
                    }
                    if !sawMarkerChar {
                        sawMarkerChar = chunk.contains("<") || chunk.contains("`")
                    }
                    if sawMarkerChar, let call = self.extractToolCall(in: buffer) {
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
            } catch is CancellationError {
                debug("generation stopped by user", category: "Chat")
                if buffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    updateAssistant("Stopped.")
                }
                return
            } catch {
                debug("generate threw: \(error)", category: "Chat")
                lastError = String(describing: error)
                // Never leave a silently EMPTY assistant bubble — the user
                // has no idea the turn died (real capture 2026-07-02: an
                // n_ctx overflow threw here and the reply was blank).
                if buffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    updateAssistant(
                        "Sorry — I hit an error generating that reply. "
                        + "Try asking again, or reset the conversation if it keeps happening.")
                }
                return
            }

            let dt = Date().timeIntervalSince(genStart)
            // One uniform [Perf] line per generate call, identical field
            // layout for every runtime — this is the row the Bonsai
            // llama.cpp-vs-MLX A/B greps for. Providers without
            // instrumentation (Mock, Apple FM) simply don't log one.
            if let stats = selectedModel.lastGenerationStats {
                debug("iter \(iter) · " + stats.summaryLine, category: "Perf")
            }
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
            let argsData = try? JSONSerialization.data(
                withJSONObject: resolvedArgs, options: [.sortedKeys])
            let argsStr = argsData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            let pre = String(buffer[..<call.range.lowerBound])
            switch toolLoopGuard.evaluate(
                toolName: call.name, canonicalArguments: argsStr)
            {
            case .allow:
                break
            case .stop(let reason):
                // Keep the model's prose before the rejected call, but do not
                // append the redundant tool emission to the transcript. The
                // forced no-tools summary below can answer from results that
                // are already present without another exploratory dispatch.
                updateAssistant(pre)
                forcedSummaryReason = reason
                debug("tool-loop circuit breaker: \(reason) — forcing summary",
                      category: "Chat")
                break toolLoop
            }
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
                // Record what this fetch was about + the vetted drift
                // threads it surfaced, so the next turn's follow-up
                // resolves and the reply can offer where to go next.
                self.updateFocusAfterTool(
                    toolName: call.name, args: resolvedArgs, result: fullResult)
                // Pass-through sentinel: the tool wants its `text` emitted
                // verbatim to the user without another model pass. Used by
                // `narrate_article` so Wikipedia prose reaches TTS unaltered
                // (no paraphrase, no re-summarization, no KV-cost for the
                // model re-encoding an article it already decided to read).
                let isPassThrough = (fullResult["pass_through"] as? Bool) == true
                let passThroughText = (fullResult["text"] as? String) ?? ""
                // Track article reading position so a later "continue"
                // pages forward (article_overview / narrate_article only;
                // a no-op for every other tool).
                noteReadingState(toolName: call.name,
                                 args: resolvedArgs, result: fullResult)
                noteDiscussionAnchor(toolName: call.name, result: fullResult)
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
                activeQueryTelemetry?.recordTool(
                    name: call.name,
                    duration: toolDt,
                    usedZimKinds: telemetryZimKinds(from: fullResult))
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
                        debug(synth, category: "Assistant")
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
                    "locate",
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
        if let last = messages.last, last.role == .assistant,
           forcedSummaryReason != nil || last.text.isEmpty {
            if let forcedSummaryReason {
                debug("tool loop stopped (\(forcedSummaryReason)) — forcing a summary turn",
                      category: "Chat")
            } else {
                debug("tool loop exhausted after \(maxIters) iters — forcing a summary turn",
                      category: "Chat")
            }
            let summaryPrompt: String
            let summaryInstruction = ChatTurn(
                role: .user,
                text: "You've used your tool budget. Without calling any more "
                    + "tools, summarize what you found for the user in 1–3 "
                    + "sentences based on the tool results above."
            )
            var finalTurns = turns
            finalTurns.append(summaryInstruction)
            // Same trained-format path as iter-0 (see the note there).
            summaryPrompt = selectedModel.template.renderTranscript(
                systemPreamble: systemMessage, tools: toolDecls, turns: finalTurns
            )
            var buffer = ""
            let params = GenerationParameters(
                maxTokens: 256, temperature: 0.3, topP: 0.9,
                useModelSamplingProfile: false)
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
        let mapParams = GenerationParameters(
            maxTokens: 256, temperature: 0.2, topP: 0.9,
            useModelSamplingProfile: false)
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
            temperature: 0.3, topP: 0.9,
            useModelSamplingProfile: false
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
            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                activeQueryTelemetry?.markFirstResponse()
            }
        }
    }

    private func appendToAssistant(_ replacement: String) {
        if messages.last?.role == .assistant {
            let scrubbed = scrubReasoning(replacement)
            recordRawEmissionIfScrubbed(raw: replacement, scrubbed: scrubbed)
            messages[messages.count - 1].text = scrubbed
            if !scrubbed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                activeQueryTelemetry?.markFirstResponse()
            }
        }
    }

    private func updateAssistant(_ newText: String) {
        if messages.last?.role == .assistant {
            let scrubbed = scrubReasoning(newText)
            recordRawEmissionIfScrubbed(raw: newText, scrubbed: scrubbed)
            messages[messages.count - 1].text = scrubbed
            if !scrubbed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                activeQueryTelemetry?.markFirstResponse()
            }
        }
    }

    /// A2 (PERFORMANCE_REVIEW.md): whenever the display text diverges from
    /// the model's emission (reasoning scrub here; offer/disambiguation
    /// appendices at their own sites), keep the exact emission on the
    /// message so the next prompt rebuild byte-matches the KV mirror.
    /// Streaming pushes overwrite it, so the final push leaves the
    /// complete raw reply in place.
    private func recordRawEmissionIfScrubbed(raw: String, scrubbed: String) {
        guard messages.last?.role == .assistant else { return }
        if scrubbed != raw || messages[messages.count - 1].rawAssistantText != nil {
            messages[messages.count - 1].rawAssistantText = raw
        }
    }

    /// Convert result-local ZIM filenames into broad archive kinds before
    /// telemetry. The names themselves never leave this process.
    private func telemetryZimKinds(from result: [String: Any]) -> [String] {
        var referencedNames: Set<String> = []
        func collect(_ value: Any, depth: Int) {
            guard depth <= 5 else { return }
            if let dictionary = value as? [String: Any] {
                for (key, nested) in dictionary {
                    if key == "zim", let name = nested as? String, !name.isEmpty {
                        referencedNames.insert(name.lowercased())
                    } else {
                        collect(nested, depth: depth + 1)
                    }
                }
            } else if let array = value as? [Any] {
                for nested in array.prefix(100) {
                    collect(nested, depth: depth + 1)
                }
            }
        }
        collect(result, depth: 0)

        var kindByFilename: [String: String] = [:]
        for entry in library {
            kindByFilename[entry.url.lastPathComponent.lowercased()] = entry.kind.rawValue
        }
        return Array(Set(referencedNames.compactMap { kindByFilename[$0] })).sorted()
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
                  let data = try? entry.reader.read(path: path)?.content
            else { continue }
            // Strip only a bounded prefix: the preview needs ~400 chars of
            // cleaned lead, and stripHTML over a full multi-hundred-KB
            // article per hit was the cost on the search hot path. The
            // lossy UTF-8 decode absorbs a split trailing codepoint.
            let html = String(decoding: data.prefix(64 * 1024), as: UTF8.self)
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

    /// Capture / refresh the article reading position after an article
    /// tool runs, so a later "continue" knows what to read next. A no-op
    /// for any other tool. `article_overview` spoke an LLM summary, so
    /// "continue" starts from the top (full lead → onward, `next = 0`);
    /// a full `narrate_article` read the whole thing, so there's nothing
    /// left (`next = total`); a paged `narrate_article` reports where it
    /// stopped via `next_section_index`.
    private func noteReadingState(
        toolName: String, args: [String: Any], result: [String: Any]
    ) {
        func intVal(_ any: Any?) -> Int? {
            (any as? Int) ?? (any as? NSNumber)?.intValue
        }
        switch toolName {
        case "article_overview":
            guard let title = result["title"] as? String,
                  let outline = result["available_sections"] as? [[String: Any]],
                  !outline.isEmpty
            else { return }
            readingState = ReadingState(
                title: title, zim: result["zim"] as? String,
                total: outline.count, next: 0)
        case "narrate_article":
            guard let title = result["title"] as? String else { return }
            let total = intVal(result["total_sections"])
                ?? intVal(result["section_count"]) ?? 0
            // A paged read reports the next unread section; a whole-
            // article read leaves nothing to continue.
            let next = intVal(result["next_section_index"]) ?? total
            readingState = ReadingState(
                title: title, zim: result["zim"] as? String,
                total: total, next: next)
        default:
            break
        }
    }

    /// Read the next chunk of the article tracked in `readingState` —
    /// dispatched straight to `narrate_article(section_index:)` and
    /// emitted to the assistant bubble (and thus TTS) with no model
    /// pass. When the article is exhausted, say so and clear the state.
    @MainActor
    private func continueReadingArticle() async {
        guard let adapter, var st = readingState else { return }
        if st.next >= st.total {
            updateAssistant("That's the end of the article on \(st.title).")
            readingState = nil
            return
        }
        var args: [String: Any] = ["title": st.title, "section_index": st.next]
        if let zim = st.zim { args["zim"] = zim }
        func jsonString(_ obj: [String: Any]) -> String {
            (try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        }
        debug("continue-reading \(st.title): section \(st.next)/\(st.total)",
              category: "Router")
        do {
            let result = try await adapter.dispatch(
                tool: "narrate_article", args: args
            )
            let rawStr = jsonString(result)
            recordToolTrace(ToolCallTrace(
                name: "narrate_article",
                arguments: jsonString(args),
                result: rawStr,
                rawResult: rawStr,
                error: nil
            ))
            let text = (result["text"] as? String) ?? ""
            guard !text.isEmpty else {
                updateAssistant("That's the end of the article on \(st.title).")
                readingState = nil
                return
            }
            updateAssistant(text)
            debug("continue-reading: emitted \(text.count) chars", category: "Chat")
            if let next = (result["next_section_index"] as? Int)
                ?? (result["next_section_index"] as? NSNumber)?.intValue
            {
                st.next = next
                readingState = st
            } else {
                readingState = nil
            }
        } catch {
            debug("continue-reading dispatch failed: \(error)", category: "Tool")
            updateAssistant("Sorry — I couldn't read more of \(st.title).")
            readingState = nil
        }
    }

    // MARK: - Discussion mode ("let's discuss X")

    /// Implicit discussion entry: after an `article_overview` ("tell me
    /// about X" / "what is X"), pin X as a discussion topic — lazily, so the
    /// sections load on the first follow-up — so subsequent questions are
    /// answered from the article + corpus drift instead of routing afresh
    /// each turn. Skipped on a miss or when already discussing this topic.
    /// To switch topics, the user says "let's discuss Y" (a discuss_article
    /// intent, which exits + re-anchors).
    private func noteDiscussionAnchor(toolName: String, result: [String: Any]) {
        guard toolName == "article_overview" else { return }
        if let e = result["error"] as? String, !e.isEmpty { return }
        guard let title = result["title"] as? String, !title.isEmpty else { return }
        let topic = ArticleHeuristics.topicCore(title)
        if let ds = discussionState,
           ds.topic.caseInsensitiveCompare(topic) == .orderedSame { return }
        groundedPromptCache = nil
        discussionState = DiscussionState(
            anchorTitle: title, topic: topic,
            zim: result["zim"] as? String, sources: [],
            linkedArticleTitles: [],
            sectionEmbeddings: [:],
            lastQuestion: nil)
    }

    /// Enter discussion mode from a `discuss_article` dispatch: pin every
    /// section, build the reusable semantic section index, then invite
    /// questions. A miss falls back to the same did-you-mean reply as
    /// `article_overview` (no confabulation). Returns `true` (fast path).
    @MainActor
    private func handleDiscussEntry(
        dictArgs: [String: Any], fullResult: [String: Any]
    ) async -> Bool {
        if let err = fullResult["error"] as? String, !err.isEmpty {
            updateAssistant(IntentRouter.synthesizeArticleMissReply(
                args: dictArgs, fullResult: fullResult))
            return true
        }
        guard let title = fullResult["title"] as? String,
              let rawSecs = fullResult["sections"] as? [[String: Any]],
              !rawSecs.isEmpty
        else {
            updateAssistant("I couldn't open that article to discuss.")
            return true
        }
        let sections = rawSecs.map(Self.decodedArticleSection)
        let preparing = "Great — I’m preparing for our chat about \(title)."
        // Keep this exact prefix when preparation finishes. Voice streaming
        // can safely speak it immediately, then consume only the appended
        // "Ready" suffix instead of replaying or slicing replaced text.
        updateAssistant(preparing)
        let preparationStarted = ProcessInfo.processInfo.systemUptime
        let sources = [(title: title, sections: sections)]
        let embeddings: [String: [Float]]
        switch discussionPreparationStrategy {
        case .none:
            embeddings = [:]
        case .semanticSections:
            embeddings = await prepareDiscussionEmbeddings(sources: sources)
        }
        let preparationElapsed = ProcessInfo.processInfo.systemUptime
            - preparationStarted
        lastDiscussionPreparationStats = DiscussionPreparationStats(
            strategy: discussionPreparationStrategy,
            title: title,
            sectionCount: sections.count,
            vectorCount: embeddings.count,
            vectorBytes: embeddings.values.reduce(0) {
                $0 + ($1.count * MemoryLayout<Float>.size)
            },
            elapsedSeconds: preparationElapsed)
        groundedPromptCache = nil
        discussionState = DiscussionState(
            anchorTitle: title,
            topic: ArticleHeuristics.topicCore(title),
            zim: fullResult["zim"] as? String,
            sources: sources,
            linkedArticleTitles: Self.linkedArticleTitles(from: fullResult),
            sectionEmbeddings: embeddings,
            lastQuestion: nil)
        updateAssistant(preparing
            + "\n\nReady. What would you like to explore about \(title)?")
        debug(String(format:
            "discuss: prepared %@ · strategy=%@ · %d sections · %d semantic vectors · %.2fs",
            title, discussionPreparationStrategy.rawValue,
            sections.count, embeddings.count, preparationElapsed),
            category: "Router")
        return true
    }

    /// Embed every loaded discussion section once. The title and body entries
    /// are keyed by the exact strings `ArticleHeuristics` will later ask its
    /// embedder to score. Body embeddings include their heading and article
    /// title as context, while the map key remains the unmodified body.
    /// Cap on embedded sections per call: each section costs ~2 real
    /// transformer forward passes (5–50 ms each), so a 100-section article
    /// spent seconds in "preparing…" before the discussion could start.
    /// Sections beyond the cap simply miss their vectors, and retrieval
    /// degrades to its deterministic order for them. Document order keeps
    /// the lead + major narrative sections — the ones follow-ups actually
    /// hit — inside the cap.
    private static let maxDiscussionEmbedSections = 32

    private func prepareDiscussionEmbeddings(
        sources: [(title: String, sections: [ArticleSection])]
    ) async -> [String: [Float]] {
        var vectors: [String: [Float]] = [:]
        var embedded = 0
        for source in sources {
            for section in source.sections where !section.text.isEmpty {
                if embedded >= Self.maxDiscussionEmbedSections {
                    debug("discussion embeddings capped at \(Self.maxDiscussionEmbedSections) sections",
                          category: "Chat")
                    return vectors
                }
                embedded += 1
                if !section.title.isEmpty {
                    let key = PreparedDiscussionEmbedder.key(section.title)
                    if vectors[key] == nil,
                       let vector = await SemanticReranker.shared.embedText(
                           section.title)
                    {
                        vectors[key] = vector
                    }
                }
                let contextualBody = [source.title, section.title, section.text]
                    .filter { !$0.isEmpty }
                    .joined(separator: ". ")
                if let vector = await SemanticReranker.shared.embedText(
                    contextualBody)
                {
                    vectors[PreparedDiscussionEmbedder.key(section.text)] = vector
                }
            }
        }
        return vectors
    }

    /// Honor an explicit user-selected Wikipedia article as the authoritative
    /// source for this turn and subsequent follow-ups. Unlike
    /// `article_overview`, this loads all sections so a narrow request can
    /// retrieve evidence buried deep in the article (for example Santa Rosa,
    /// California's "20th century" section).
    @MainActor
    private func handleWikipediaSourceDirective(
        _ directive: WikipediaSourceDirective
    ) async -> Bool {
        guard let adapter else {
            updateAssistant("Wikipedia isn't ready yet.")
            return true
        }
        var args: [String: Any] = ["title": directive.title]
        guard let wikipedia = library.first(where: {
            $0.isEnabled && $0.reader.kind == .wikipedia
        }) ?? library.first(where: {
            $0.isEnabled && $0.reader.kind == .mdwiki
        }) else {
            updateAssistant(
                "I don't have an enabled offline Wikipedia library open.")
            debug("explicit Wikipedia source unavailable: no Wikipedia ZIM",
                  category: "Router")
            return true
        }
        // Supplying the exact library is the authority boundary: a source
        // correction may resolve or fail inside Wikipedia, but it can never
        // drift into a StreetZIM article or place search.
        args["zim"] = wikipedia.url.lastPathComponent
        guard let result = try? await adapter.dispatch(
            tool: "discuss_article", args: args)
        else {
            updateAssistant("I couldn't open the Wikipedia article on \(directive.title).")
            return true
        }
        let argsData = try? JSONSerialization.data(
            withJSONObject: args, options: [.sortedKeys])
        let resultData = try? JSONSerialization.data(
            withJSONObject: result, options: [.sortedKeys])
        let argsString = argsData.flatMap {
            String(data: $0, encoding: .utf8)
        } ?? "{}"
        let resultString = resultData.flatMap {
            String(data: $0, encoding: .utf8)
        } ?? "{}"
        recordToolTrace(ToolCallTrace(
            name: "discuss_article",
            arguments: argsString,
            result: resultString,
            rawResult: resultString,
            error: nil))
        if let error = result["error"] as? String, !error.isEmpty {
            updateAssistant(IntentRouter.synthesizeArticleMissReply(
                args: args, fullResult: result))
            debug("explicit Wikipedia source missed: \(error)",
                  category: "Router")
            return true
        }
        guard let title = result["title"] as? String,
              let rawSections = result["sections"] as? [[String: Any]],
              !rawSections.isEmpty
        else {
            updateAssistant("I found the Wikipedia article but couldn't read its sections.")
            return true
        }
        let sections = rawSections.map(Self.decodedArticleSection)
        let sources = [(title: title, sections: sections)]
        let embeddings: [String: [Float]]
        switch discussionPreparationStrategy {
        case .none:
            embeddings = [:]
        case .semanticSections:
            embeddings = await prepareDiscussionEmbeddings(sources: sources)
        }
        var state = DiscussionState(
            anchorTitle: title,
            topic: ArticleHeuristics.topicCore(title),
            zim: result["zim"] as? String,
            sources: sources,
            linkedArticleTitles: Self.linkedArticleTitles(from: result),
            sectionEmbeddings: embeddings,
            lastQuestion: nil)
        discussionState = state
        groundedPromptCache = nil
        focus.remember(FocusEntity(
            name: title, kind: .topic,
            zimPath: result["path"] as? String))
        debug("explicit source selected: \(title) · \(sections.count) sections",
              category: "Router")

        if let facet = directive.question, !facet.isEmpty {
            let question = "What does the article say about \(facet)?"
            state.lastQuestion = nil
            discussionState = state
            await answerWithinDiscussion(state, question: question)
        } else {
            updateAssistant(
                "Okay — I’ll use the Wikipedia article “\(title)” as the source. "
                + "What would you like me to look for?")
        }
        return true
    }

    /// Answer one follow-up within discussion mode: retrieve the lead +
    /// the section(s) most relevant to the question, then stream a single
    /// generation grounded strictly in those passages (mirrors the
    /// map-reduce reduce phase). The small model only ever sees the few
    /// passages a question needs — never the whole article.
    @MainActor
    private func answerWithinDiscussion(_ ds: DiscussionState, question: String) async {
        activeQueryTelemetry?.setRoute("discussion")
        var state = ds
        let contextualQuestion = ArticleHeuristics
            .contextualizedDiscussionQuestion(
                question, previousQuestion: state.lastQuestion)
        if contextualQuestion != question {
            debug("discussion continuation resolved: \(contextualQuestion)",
                  category: "Chat")
        }
        // Implicit-entry anchors store only the title; pull the anchor's
        // sections on the first follow-up.
        if state.sources.isEmpty, let adapter {
            var a: [String: Any] = ["title": state.anchorTitle]
            if let zim = state.zim { a["zim"] = zim }
            if let res = try? await adapter.dispatch(tool: "discuss_article", args: a),
               let raw = res["sections"] as? [[String: Any]], !raw.isEmpty {
                state.sources = [(
                    state.anchorTitle,
                    raw.map(Self.decodedArticleSection)
                )]
                state.linkedArticleTitles = Self.linkedArticleTitles(from: res)
                if discussionPreparationStrategy == .semanticSections {
                    state.sectionEmbeddings = await prepareDiscussionEmbeddings(
                        sources: state.sources)
                }
                discussionState = state
            }
        }
        guard !state.sources.isEmpty else {
            updateAssistant("What would you like to know about \(state.topic)?")
            return
        }
        // Direct-link fallback: if none of the articles in hand cover the
        // follow-up, search for a useful candidate but accept it only when an
        // article already in hand links to it. Name similarity alone cannot
        // escape the prepared topic.
        let coveredByHand = state.sources.contains {
            ArticleHeuristics.sectionsCoverQuestion(
                $0.sections, contextualQuestion, articleTitle: $0.title)
        }
        let asksAboutOpposingSides = ArticleHeuristics
            .asksAboutOpposingSides(contextualQuestion)
        if (!coveredByHand || asksAboutOpposingSides), let adapter {
            let kwList = ArticleHeuristics.questionKeywords(contextualQuestion)
            let query = (state.topic + " " + kwList.joined(separator: " "))
                .trimmingCharacters(in: .whitespaces)
            let existingTitles = Set(state.sources.map { $0.title.lowercased() })
            let preferredTitles = ArticleHeuristics.namedEventArticleCandidates(
                state.sources.flatMap(\.sections), question: contextualQuestion)
            if let pulled = await pullArticleForDiscussion(
                query: query, keywords: kwList, zim: state.zim, adapter: adapter,
                excludingTitles: existingTitles,
                preferredTitles: preferredTitles,
                allowedTitles: state.linkedArticleTitles) {
                state.sources.append((pulled.title, pulled.sections))
                state.linkedArticleTitles.formUnion(pulled.linkedTitles)
                if discussionPreparationStrategy == .semanticSections {
                    let added = await prepareDiscussionEmbeddings(sources: [
                        (title: pulled.title, sections: pulled.sections),
                    ])
                    state.sectionEmbeddings.merge(added) { _, new in new }
                    debug("discuss: indexed pulled source \(pulled.title) · \(added.count) semantic vectors",
                          category: "Chat")
                }
                debug("discuss: pulled “\(pulled.title)” for “\(question)”",
                      category: "Router")
            }
        }
        state.lastQuestion = question
        discussionState = state   // persist any pulled-in article

        let first = await generateGroundedAnswer(
            state: state, question: question,
            retrievalQuestion: contextualQuestion)
        // Reactive corpus fallback: the coverage gate is lexical, so a
        // question whose keywords merely APPEAR in the articles in hand
        // ("why did he invade Ukraine?" while holding the Crimea-annexation
        // article) skips the pull, and the model rightly answers "I don't
        // see it". Treat that answer as the coverage signal: pull the best
        // corpus article for the question and regenerate ONCE.
        if Self.looksLikeDontSee(first), let adapter {
            // FIRST: retry within the articles already in hand, excluding
            // every section the model has seen — cheaper than a corpus
            // pull and usually where the answer actually is (2026-08-02:
            // Bulgaria's NATO facts sat two sections below the Geography
            // pick). Only fall through to the corpus when the in-article
            // retry also comes up dry.
            let retried = await generateGroundedAnswer(
                state: state, question: question,
                retrievalQuestion: contextualQuestion,
                excludeTriedPassages: true)
            if !Self.looksLikeDontSee(retried) {
                debug("discuss: in-article retry answered after a don't-see",
                      category: "Router")
                state.lastQuestion = question
                discussionState = state
                return
            }
            let kwList = ArticleHeuristics.questionKeywords(contextualQuestion)
            let query = (state.topic + " " + kwList.joined(separator: " "))
                .trimmingCharacters(in: .whitespaces)
            let existingTitles = Set(state.sources.map { $0.title.lowercased() })
            let preferredTitles = ArticleHeuristics.namedEventArticleCandidates(
                state.sources.flatMap(\.sections), question: contextualQuestion)
            if let pulled = await pullArticleForDiscussion(
                query: query, keywords: kwList, zim: state.zim, adapter: adapter,
                excludingTitles: existingTitles,
                preferredTitles: preferredTitles,
                allowedTitles: state.linkedArticleTitles) {
                state.sources.append((pulled.title, pulled.sections))
                state.linkedArticleTitles.formUnion(pulled.linkedTitles)
                if discussionPreparationStrategy == .semanticSections {
                    let added = await prepareDiscussionEmbeddings(sources: [
                        (title: pulled.title, sections: pulled.sections),
                    ])
                    state.sectionEmbeddings.merge(added) { _, new in new }
                    debug("discuss: indexed retry source \(pulled.title) · \(added.count) semantic vectors",
                          category: "Chat")
                }
                discussionState = state
                debug("discuss: retry — pulled “\(pulled.title)” after a don't-see answer",
                      category: "Router")
                _ = await generateGroundedAnswer(
                    state: state, question: question,
                    retrievalQuestion: contextualQuestion)
            }
        }
    }

    /// The canned honesty reply from the grounded-discuss instruction —
    /// used as the trigger for the one-shot corpus-pull retry.
    private static func looksLikeDontSee(_ s: String) -> Bool {
        let t = s.lowercased()
        if t.contains("don't see") || t.contains("do not see")
            || t.contains("not in the passages")
            || t.contains("don't have that")
            || t.contains("isn't in what i have")
        { return true }

        // A qualified answer can give the useful period/reign and then note
        // that the evidence lacks a more exact date. That is not a retrieval
        // failure. Only treat the newer insufficiency phrases as a fallback
        // signal when they occur in the opening sentence; this still catches
        // "The evidence does not specify the sides" without discarding a
        // three-sentence answer that ends with an honest caveat.
        let opening = t.split(separator: ".", maxSplits: 1,
                              omittingEmptySubsequences: true)
            .first.map(String.init) ?? String(t.prefix(240))
        return opening.contains("does not specify")
            || opening.contains("doesn't specify")
            || opening.contains("not specified in")
            || opening.contains("not provided in the evidence")
    }

    /// Grounded answers keep their useful section follow-ups and also expose
    /// the explicit full-article narration action that older builds offered.
    /// The chip's complete prompt makes it deterministic even when the user is
    /// already inside pinned discussion mode.
    private func groundedSuggestions(
        state: DiscussionState,
        sections: [ArticleSection],
        after question: String
    ) -> [DiscoveryThread] {
        let readFull = DiscoveryThread(
            label: "Read full article",
            kind: .topic,
            source: .section,
            note: "Complete Wikipedia article",
            prompt: "Read the full article about \(state.anchorTitle) aloud")
        let contextual = ConversationThreads.contextualQuestions(
            topic: state.topic,
            sections: sections,
            after: question,
            max: 3)
        return [readFull] + contextual
    }

    /// One grounded generation over the discussion's current sources:
    /// rank sections for the question, assemble capped passages, stream
    /// the answer into the assistant bubble, and return the final text.
    @MainActor
    @discardableResult
    private func generateGroundedAnswer(
        state: DiscussionState, question: String,
        retrievalQuestion: String? = nil,
        excludeTriedPassages: Bool = false
    ) async -> String {
        preemptLlamaPromptOptimizationForGroundedTurn()
        let resolvedQuestion = retrievalQuestion ?? question
        // Anchor lead for topic context + the top-ranked sections across all
        // articles in hand. Wider budget (6) than before — llama.cpp KV is
        // cheap, and specific-fact follow-ups need the deeper section, not
        // just the lead. Each passage is capped so several fit in n_ctx.
        let defaultPassageLimit = ArticleHeuristics
            .groundedPassageLimit(for: resolvedQuestion)
        let passageLimit = longerReplies
            ? min(8, max(6, defaultPassageLimit + 2))
            : defaultPassageLimit
        let defaultCharacterLimit = ArticleHeuristics
            .groundedPassageCharacterLimit(for: resolvedQuestion)
        let passageCharacterLimit = longerReplies
            ? min(2_400, max(1_400, defaultCharacterLimit + 500))
            : defaultCharacterLimit
        // A content-free referential follow-up ("What are they?", "tell me
        // more") refers to the pinned topic, not whichever linked support
        // article happened to be retrieved most recently. Rank only the
        // anchor for that turn so an auxiliary source cannot capture it.
        let anchorSnapback = ArticleHeuristics
            .questionKeywords(resolvedQuestion).isEmpty
        let rankingSources = anchorSnapback
            ? Array(state.sources.prefix(1))
            : state.sources
        let ranked: [(article: String, section: ArticleSection)]
        if !state.sectionEmbeddings.isEmpty,
           let questionVector = await SemanticReranker.shared.embedText(
               resolvedQuestion)
        {
            var vectors = state.sectionEmbeddings
            vectors[PreparedDiscussionEmbedder.key(resolvedQuestion)] = questionVector
            let embedder = PreparedDiscussionEmbedder(vectors: vectors)
            ranked = ArticleHeuristics.rankSectionsMultiSource(
                resolvedQuestion, sources: rankingSources, embedder: embedder,
                k: max(4, passageLimit * 2))
            debug("discuss retrieval: prepared semantic section index",
                  category: "Chat")
        } else {
            ranked = ArticleHeuristics.rankSectionsMultiSource(
                resolvedQuestion, sources: rankingSources,
                k: max(4, passageLimit * 2))
        }
        var picked: [(article: String, section: ArticleSection)] = []
        if let anchor = state.sources.first,
           let lead = anchor.sections.first(where: { $0.title.isEmpty }) {
            picked.append((anchor.title, lead))
        }
        // A participant follow-up may have just pulled a dedicated event
        // article. Even after indexing that source, make its lead the next
        // passage when its title carries all of the event terms in the
        // question. Otherwise a broad country section can still win a close
        // semantic tie and hide the better Red/White/combatant evidence.
        if ArticleHeuristics.asksAboutOpposingSides(resolvedQuestion) {
            let participantTerms: Set<String> = [
                "side", "sides", "combatant", "combatants", "belligerent",
                "belligerents", "fought", "forces", "armies",
            ]
            let eventTerms = ArticleHeuristics.questionKeywords(resolvedQuestion)
                .filter { !participantTerms.contains($0) }
            if let dedicated = state.sources.dropFirst().first(where: { source in
                let title = source.title.lowercased()
                return !eventTerms.isEmpty
                    && eventTerms.allSatisfy { title.contains($0) }
            }), let lead = dedicated.sections.first(where: { $0.title.isEmpty }) {
                picked.append((dedicated.title, lead))
            }
        }
        // In-article retry: after a don't-see answer, rank AGAIN but skip
        // every section already fed to the model — the fact usually sits in
        // a section the first ranking passed over ("dealt with the West and
        // NATO" ranked Geography; Foreign relations had the answer,
        // 2026-08-02). Keys are section-granular prefixes of the cache's
        // window-granular keys.
        let excludedSectionKeys: Set<String> = excludeTriedPassages
            ? Set((groundedPromptCache?.passageKeys ?? []).map {
                String($0.split(separator: "\u{1F}").prefix(2).joined(separator: "\u{1F}"))
            })
            : []
        func sectionKey(_ p: (article: String, section: ArticleSection)) -> String {
            let t = p.section.title.lowercased()
            return p.article.lowercased() + "\u{1F}"
                + ((t.isEmpty || t == "lead") ? "lead" : t)
        }
        for r in ranked where !picked.contains(where: {
            $0.article == r.article && $0.section.title == r.section.title
        }) && !excludedSectionKeys.contains(sectionKey(r)) {
            picked.append(r)
            if picked.count >= passageLimit { break }
        }
        debug("discuss \(state.topic): passages = " + picked.map {
            "\($0.article)§\($0.section.title.isEmpty ? "lead" : $0.section.title)"
        }.joined(separator: " | "), category: "Chat")

        func passageText(_ p: (article: String, section: ArticleSection)) -> String {
            ArticleHeuristics.groundedPassageWindow(
                p.section.text,
                question: resolvedQuestion,
                maxChars: passageCharacterLimit)
        }
        func passageKey(_ p: (article: String, section: ArticleSection)) -> String {
            // Include the selected sentence window, not just the section.
            // "parents" and "school" can both live in Early life and
            // education; deduping only by heading would prevent the later
            // question from appending its distinct evidence window.
            let rawTitle = p.section.title.lowercased()
            let normalizedTitle = (rawTitle.isEmpty || rawTitle == "lead")
                ? "lead" : rawTitle
            let base = p.article.lowercased() + "\u{1F}" + normalizedTitle
            // Cache the anchor lead once. Letting every question create a
            // different lead window consumed the warm turn's one-new-passage
            // allowance before the actually relevant Parents/Education
            // section could be appended.
            if normalizedTitle == "lead" {
                return base
            }
            return base + "\u{1F}" + passageText(p)
        }
        func renderPassages(
            _ items: [(article: String, section: ArticleSection)]
        ) -> String {
            items.map { p -> String in
                let head = p.section.title.isEmpty
                    ? p.article : "\(p.article) — \(p.section.title)"
                return "## \(head)\n\(passageText(p))"
            }.joined(separator: "\n\n")
        }
        let answerStyle = longerReplies
            ? "Be detailed but natural, and don't say \"according to the passage\". For broad questions, answer in four to eight substantive sentences and cover the main context, chronology, and consequences supported by the evidence."
            : "Be conversational and informative, and don't say \"according to the passage\". Usually answer in two to four substantive sentences: give the direct answer first, then add the most useful context supported by the evidence. A truly atomic fact may use one sentence. For historical questions, include the date or period when the evidence gives it, then briefly explain what happened and why it mattered."
        let preamble = """
        You are discussing "\(state.topic)" with the user using offline Wikipedia evidence supplied throughout this conversation. Answer using ONLY that evidence. \(answerStyle) If the answer isn't in the evidence, say you don't see it in what you have on \(state.topic). Attribute carefully: the evidence may mix statements by DIFFERENT parties (\(state.topic), other governments, critics) — never put one party's words in another's mouth; if the question asks what someone said, report only THAT person's statements. For a broad school or education question, include both secondary school and university when the evidence supplies both. For "how many died", report figures explicitly labeled killed or dead for each side; a broader casualty figure may include wounded and is NOT a death count, so label it separately rather than substituting it for deaths. When casualty figures conflict, label each side or source's estimate separately; never combine killed and wounded into a total death count. When asked who fought or about opposing sides, name every opposing side supported by the evidence, not only the most recently mentioned force. Use at least two sentences for a participant question: identify the principal sides first, then add supported context such as other participants, foreign intervention, or what each side represented. Give just the answer directly — no reasoning steps, no preamble, no <think> block.
        """

        var cache: GroundedPromptCache
        if let existing = groundedPromptCache,
           existing.modelID == selectedModel.id,
           existing.topic.caseInsensitiveCompare(state.topic) == .orderedSame,
           existing.systemPreamble == preamble {
            cache = existing
        } else {
            cache = GroundedPromptCache(
                topic: state.topic,
                modelID: selectedModel.id,
                systemPreamble: preamble,
                turns: [],
                passageKeys: [])
        }

        let wasWarm = !cache.turns.isEmpty
        let lowerQuestion = resolvedQuestion.lowercased()
        let refreshLeadForSides = ArticleHeuristics
            .asksAboutOpposingSides(lowerQuestion)
        // Put genuinely unseen evidence first. Participant questions may
        // also refresh a previously cached lead because a different sentence
        // window can contain the sides, but that refresh must never consume
        // the warm turn's one-passage budget ahead of a newly pulled exact
        // event article.
        var passagesForTurn = picked.filter {
            !cache.passageKeys.contains(passageKey($0))
        }
        if refreshLeadForSides {
            passagesForTurn += picked.filter { passage in
                let title = passage.section.title.lowercased()
                let isLead = title.isEmpty || title == "lead"
                return isLead
                    && cache.passageKeys.contains(passageKey(passage))
            }
        }
        if !wasWarm {
            passagesForTurn = picked
        } else if anchorSnapback, let anchorPassage = picked.first {
            // Repeat the anchor evidence even when its cache key is already
            // present. The newest user turn then carries an explicit topic
            // reset instead of asking the model to resolve a bare pronoun
            // against a potentially newer auxiliary article.
            passagesForTurn = [anchorPassage]
        } else if passagesForTurn.count > 1 {
            // Earlier evidence remains in the append-only transcript. Add at
            // most the best unseen section per follow-up; appending all
            // newly ranked sections made a simple "How are they created?"
            // turn prefill 905 avoidable tokens even though the lead already
            // covered the answer. A genuinely new facet/article still gets
            // its highest-ranked 1,500-character passage; the don't-see retry
            // can append a pulled article if that evidence is insufficient.
            passagesForTurn = Array(passagesForTurn.prefix(1))
        }

        // Date/quantity follow-ups get the single best-matching sentence
        // from ANY section quoted verbatim as evidence. Section-level
        // ranking can miss the fact ("When did Bulgaria join NATO?" pulled
        // no new passage and the model invented 2009), and even with the
        // right section a 1-bit model paraphrases dates badly ("joined
        // NATO as a member of the OSCE") — real captures 2026-08-02.
        let keyFact: (article: String, sentence: String)? =
            ArticleHeuristics.isFactoidShaped(question)
            ? ArticleHeuristics.keyFactSentence(
                question: resolvedQuestion, sources: state.sources)
            : nil
        if let keyFact {
            debug("discuss key fact: \(keyFact.article) · “\(String(keyFact.sentence.prefix(90)))…”",
                  category: "Chat")
        }
        func makeUserTurn(
            passages: [(article: String, section: ArticleSection)]
        ) -> ChatTurn {
            var evidence: String
            if passages.isEmpty {
                evidence = "No new evidence for this turn; use the offline Wikipedia evidence already supplied earlier in the conversation."
            } else {
                evidence = "New offline Wikipedia evidence:\n\n" + renderPassages(passages)
            }
            if let keyFact {
                evidence += "\n\nKey sentence from \(keyFact.article): \"\(keyFact.sentence).\""
            }
            return ChatTurn(
                role: .user,
                text: "\(evidence)\n\nQuestion: \(resolvedQuestion)")
        }
        func renderPrompt(_ turns: [ChatTurn]) -> String {
            if selectedModel is Gemma4Provider {
                return selectedModel.template.renderTranscript(
                    systemPreamble: preamble, tools: [], turns: turns)
            }
            return selectedModel.formatTranscript(
                systemPreamble: preamble, turns: turns)
        }

        // Floor the budget at 512: the FT sometimes opens a <think> on this
        // off-distribution grounded prompt, and on a low-budget device
        // profile (e.g. the Mac CLI) that burns the allowance and truncates
        // the answer mid-sentence ("Perovskite solar cells can be built…").
        // 512 leaves room for the answer even after a short reasoning preamble.
        let params = GenerationParameters(
            maxTokens: max(effectiveMaxReplyTokens, 512),
            temperature: 0.3, topP: 0.9,
            // Bonsai's publisher-tuned sampler is materially better for
            // natural grounded conversation. Providers without a profile
            // continue to use the conservative task values above.
            useModelSamplingProfile: true)
        var candidateTurns = cache.turns + [makeUserTurn(passages: passagesForTurn)]

        // Preserve exact grounded names and count labels for the few fact
        // shapes where a sampled paraphrase is actively harmful. Bonsai is
        // still the conversational engine for every open-ended turn; this
        // is the same fast/extractive split used by the top-level factoid
        // router, now applied to follow-ups within a pinned article.
        if let extractive = ArticleHeuristics.groundedExtractiveAnswer(
            question: resolvedQuestion,
            // The extractor does not feed this text to the model, so it can
            // safely inspect the complete selected sections. Using the
            // compact model window here dropped an earlier 60–200 killed
            // estimate from the Alamo casualty section.
            passages: picked.map { $0.section.text }) {
            updateAssistant(extractive)
            cache.turns = candidateTurns
                + [ChatTurn(role: .assistant, text: extractive)]
            cache.passageKeys.formUnion(passagesForTurn.map(passageKey))
            groundedPromptCache = cache
            if let idx = messages.indices.last,
               messages[idx].role == .assistant {
                var seenSources = Set<GroundingSource>()
                messages[idx].groundingSources = picked.compactMap { p in
                    let source = GroundingSource(
                        kind: .wikipedia,
                        title: p.article,
                        section: p.section.title.isEmpty ? nil : p.section.title,
                        library: state.zim)
                    return seenSources.insert(source).inserted ? source : nil
                }
                let sections = state.sources.first?.sections
                    ?? state.sources.flatMap(\.sections)
                messages[idx].suggestions = groundedSuggestions(
                    state: state, sections: sections, after: question)
            }
            debug("grounded extractive reply: \(extractive)", category: "Chat")
            debug(extractive, category: "Assistant")
            return extractive
        }

        var prompt = renderPrompt(candidateTurns)

        // Keep enough room for the complete answer. Once a long discussion
        // approaches n_ctx, start a fresh grounded window with the passages
        // selected for THIS question. That is an intentional cache miss, but
        // avoids either overflowing the context or silently losing evidence.
        var promptTokens: Int?
        if let llama = selectedModel as? LlamaCppProvider {
            promptTokens = llama.promptTokenCount(prompt)
            // Pure appends prefill only the new suffix. The verified 16K
            // allocation supplies overflow headroom, while this 6K rolling
            // ceiling still reserves reply space and bounds attention/decode
            // cost before an intentional fresh window.
            let budget = min(6144,
                max(256, llama.contextTokens - params.maxTokens - 32))
            if let count = promptTokens, count > budget, wasWarm {
                cache.turns.removeAll()
                cache.passageKeys.removeAll()
                passagesForTurn = picked
                candidateTurns = [makeUserTurn(passages: passagesForTurn)]
                prompt = renderPrompt(candidateTurns)
                promptTokens = llama.promptTokenCount(prompt)
                debug("discuss cache: compacted at \(count) tokens (budget \(budget)); rebuilt current evidence",
                      category: "Chat")
            }
        }
        debug("discuss cache: \(wasWarm && !cache.turns.isEmpty ? "append" : "cold") · prior-turns=\(cache.turns.count) · new-passages=\(passagesForTurn.count) · prompt=\(promptTokens.map(String.init) ?? "?") tok",
              category: "Chat")
        if !passagesForTurn.isEmpty {
            debug("discuss evidence append: " + passagesForTurn.map {
                let name = $0.section.title.isEmpty ? "lead" : $0.section.title
                return "\($0.article)§\(name)(\(passageText($0).count)c)"
            }.joined(separator: " | "), category: "Chat")
        }

        var buffer = ""
        var lastUIPush = Date.distantPast
        do {
            for try await chunk in selectedModel.generate(prompt: prompt, parameters: params) {
                buffer += chunk
                let now = Date()
                if now.timeIntervalSince(lastUIPush) >= 0.1 {
                    updateAssistant(buffer); lastUIPush = now
                }
            }
            let final = stripLeakedReasoning(buffer)
            updateAssistant(final)
            // Grounded turns bypass runGenerationLoop, so emit the uniform
            // cross-runtime [Perf] row here too — the Bonsai A/B harness
            // drives exactly this path.
            if let stats = selectedModel.lastGenerationStats {
                debug("grounded · " + stats.summaryLine, category: "Perf")
            }
            // Normally store the exact raw emission: re-tokenising that
            // transcript reproduces the tokens already resident in llama.cpp,
            // so the next prompt is a strict append. A leaked reasoning marker
            // is different. Keeping it would contaminate every later turn and
            // could make the model continue the malformed pattern. Rebuild the
            // next turn from the scrubbed answer instead; sacrificing one warm
            // append is preferable to poisoning the whole discussion cache.
            let cacheAnswer: String
            if buffer.contains("<think>") || buffer.contains("</think>") {
                cacheAnswer = final
                debug("reasoning marker removed from grounded cache; next turn may rebuild KV",
                      category: "Chat")
            } else {
                cacheAnswer = buffer
            }
            cache.turns = candidateTurns
                + [ChatTurn(role: .assistant, text: cacheAnswer)]
            cache.passageKeys.formUnion(passagesForTurn.map(passageKey))
            groundedPromptCache = cache
            if let idx = messages.indices.last,
               messages[idx].role == .assistant
            {
                var seenSources = Set<GroundingSource>()
                messages[idx].groundingSources = picked.compactMap { p in
                    guard cache.passageKeys.contains(passageKey(p)) else { return nil }
                    let source = GroundingSource(
                        kind: .wikipedia,
                        title: p.article,
                        section: p.section.title.isEmpty ? nil : p.section.title,
                        library: state.zim)
                    return seenSources.insert(source).inserted ? source : nil
                }
                debug("grounding sources: "
                    + messages[idx].groundingSources.map { source in
                        [source.title, source.section]
                            .compactMap { $0 }
                            .joined(separator: "§")
                    }.joined(separator: " | "),
                      category: "Chat")
                // Suggestions should stay about the pinned subject. Pulled
                // support articles improve evidence, but their own generic
                // headings (for example Domestic policy › History) made chips
                // drift away from the conversation.
                let sections = state.sources.first?.sections
                    ?? state.sources.flatMap(\.sections)
                messages[idx].suggestions = groundedSuggestions(
                    state: state, sections: sections, after: question)
                debug("contextual suggestions: "
                    + messages[idx].suggestions.map(\.label).joined(separator: " | "),
                      category: "Chat")
            }
            // Mirror the answer into the debug log — the grounded path
            // didn't, so device logs showed WHICH passages were used but
            // never WHAT the model said, making bad answers undiagnosable
            // from a pasted log (2026-07-02).
            let trimmed = final.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                debug(trimmed, category: "Assistant")
            }
            return final
        } catch is CancellationError {
            let partial = stripLeakedReasoning(buffer)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            updateAssistant(partial.isEmpty ? "Stopped." : partial)
            debug("grounded generation stopped by user", category: "Chat")
            return partial
        } catch {
            debug("discuss generate failed: \(error)", category: "Chat")
            let msg = "Sorry — I hit an error answering that about \(state.topic)."
            updateAssistant(msg)
            return msg
        }
    }

    /// The FT model occasionally opens a `<think>` reasoning block on the
    /// off-distribution grounded-discuss prompt and forgets to close it, so
    /// the template's closed-span scrubber can't strip it. When that happens
    /// the real answer is the prose AFTER the reasoning — take the last
    /// paragraph. Closed spans are stripped first with the active template;
    /// returning the raw closed block here made the reactive fallback inspect
    /// hidden chain-of-thought and retry otherwise-good answers.
    private func stripLeakedReasoning(_ s: String) -> String {
        var scrubbed = selectedModel.template.stripReasoning(s)
        // Some llama.cpp chat templates evaluate `<think>` as part of the
        // assistant prefix, so the generated buffer contains only
        // `draft</think>final`. Template-specific scrubbers normally handle
        // this, but keep the grounded path safe for every provider family.
        if let close = scrubbed.range(of: "</think>", options: .backwards) {
            scrubbed = String(scrubbed[close.upperBound...])
        }
        guard let open = scrubbed.range(of: "<think>") else { return scrubbed }
        if scrubbed.range(of: "</think>", range: open.upperBound..<scrubbed.endIndex) != nil {
            return scrubbed
        }
        let after = String(scrubbed[open.upperBound...])
        if let lastBreak = after.range(of: "\n\n", options: .backwards) {
            let tail = String(after[lastBreak.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty { return tail }
        }
        return after.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Find an article that covers a follow-up the pinned article(s) don't,
    /// and return its sections. Reuses the `search` (best title) +
    /// `discuss_article` (all sections) dispatches — all real ZIM data.
    @MainActor
    private func pullArticleForDiscussion(
        query: String, keywords: [String], zim: String?, adapter: MCPToolAdapter,
        excludingTitles: Set<String> = [], preferredTitles: [String] = [],
        allowedTitles: Set<String>
    ) async -> (title: String, sections: [ArticleSection], linkedTitles: Set<String>)? {
        func loadArticle(
            _ title: String
        ) async -> (title: String, sections: [ArticleSection], linkedTitles: Set<String>)? {
            var args: [String: Any] = ["title": title]
            if let zim { args["zim"] = zim }
            guard let res = try? await adapter.dispatch(
                tool: "discuss_article", args: args),
                  let raw = res["sections"] as? [[String: Any]], !raw.isEmpty
            else { return nil }
            let secs = raw.map(Self.decodedArticleSection)
            guard secs.reduce(0, { $0 + $1.text.count }) >= 500 else {
                return nil
            }
            return (
                (res["title"] as? String) ?? title,
                secs,
                Self.linkedArticleTitles(from: res))
        }

        for title in preferredTitles {
            let key = title.lowercased()
            guard !excludingTitles.contains(key),
                  ArticleHeuristics.isDirectlyLinkedArticle(
                    title, allowedTitleKeys: allowedTitles)
            else { continue }
            if let article = await loadArticle(title) {
                debug("discuss: direct-link named-event pull “\(article.title)”",
                      category: "Router")
                return article
            }
        }

        guard let search = try? await adapter.dispatch(
            tool: "search", args: ["query": query, "limit": 5]) else { return nil }
        let hits = (search["hits"] as? [[String: Any]])
            ?? (search["results"] as? [[String: Any]]) ?? []
        func hitTitle(_ h: [String: Any]) -> String? {
            if let t = h["title"] as? String, !t.isEmpty { return t }
            if let p = h["path"] as? String, !p.isEmpty {
                return p.split(separator: "/").last
                    .map { $0.replacingOccurrences(of: "_", with: " ") }
            }
            return nil
        }
        let titles = hits.compactMap(hitTitle)
        // A candidate must be both directly linked AND title-relevant. The
        // old "then the rest" fallback let a semantic search hit for Raman
        // scattering answer "When were photons discovered?" and persisted
        // that drift into the next turn.
        let ordered = titles.filter {
            ArticleHeuristics.linkedExpansionTitleMatchesQuestion(
                $0, keywords: keywords)
        }
        if ordered.isEmpty, !titles.isEmpty {
            debug("discuss: declined unrelated linked search hits for \"\(query)\"",
                  category: "Router")
        }

        // Try candidates in order, returning the first with REAL content —
        // skip redirect/stub pages (real capture 2026-05-30: the search's top
        // hit "Energy efficiency of internal combustion engines" was a
        // 334-byte redirect stub, so the answer was "I don't see it"; the
        // content lives in "Internal combustion engine"). dedup by title.
        var tried = Set<String>()
        for title in ordered.prefix(5) {
            let key = title.lowercased()
            if tried.contains(key) || excludingTitles.contains(key)
                || !ArticleHeuristics.isDirectlyLinkedArticle(
                    title, allowedTitleKeys: allowedTitles) { continue }
            tried.insert(key)
            if let article = await loadArticle(title) { return article }
        }
        return nil
    }

    /// Normalize the outbound link metadata returned by `discuss_article`.
    /// Keep both visible anchor text and the destination path's final
    /// component: Wikipedia may display "the civil war" while linking to
    /// `A/Russian_Civil_War`, and the latter is the exact article identity.
    private static func linkedArticleTitles(
        from result: [String: Any]
    ) -> Set<String> {
        let links = result["linked_articles"] as? [[String: Any]] ?? []
        return ArticleHeuristics.linkedArticleTitleKeys(links)
    }

    /// The overview adapter labels Wikipedia's untitled opening passage
    /// `lead` for JSON consumers, while discussion retrieval represents that
    /// same passage with an empty heading. Normalize at the tool boundary so
    /// lead anchoring works identically for both adapter paths.
    private static func decodedArticleSection(
        _ dictionary: [String: Any]
    ) -> ArticleSection {
        let rawTitle = (dictionary["title"] as? String) ?? ""
        let title = rawTitle.caseInsensitiveCompare("lead") == .orderedSame
            ? ""
            : rawTitle
        return ArticleSection(
            title: title,
            level: (dictionary["level"] as? Int) ?? 0,
            text: (dictionary["text"] as? String) ?? "")
    }

    /// Run the dispatched tool, record the trace, and synthesize a
    /// one-line assistant caption. Returns `true` when the fast path
    /// was used successfully and the LLM should be skipped.
    @MainActor
    private func executeDirectIntent(_ intent: DirectIntent) async -> Bool {
        guard let adapter else { return false }
        activeQueryTelemetry?.setRoute("fast_path", primaryTool: intent.toolName)
        // Replace "my location" / "here" / "me" / "current location"
        // with "lat,lon" before dispatching, so the geocoder's parse-
        // as-coords shortcut picks up the user's GPS fix. Previously
        // only the LLM-driven tool-dispatch path (runGenerationLoop)
        // ran this substitution, so fast-path-dispatched
        // route_from_places / near_places with `origin:"my
        // location"` failed with `could not resolve my location`
        // (2026-04-23 gist aa8ca1dc, "Directions to San Jose
        // airport").
        let dictArgs = substituteCurrentLocation(in: intent.anyArgs)
        let argsStr: String = {
            guard let data = try? JSONSerialization.data(
                withJSONObject: dictArgs, options: [.sortedKeys]
            ), let s = String(data: data, encoding: .utf8) else { return "{}" }
            return s
        }()
        debug("fast-path dispatch \(intent.toolName)(\(argsStr))", category: "Tool")
        do {
            let dispatchStarted = ProcessInfo.processInfo.systemUptime
            let dispatchMemory = MemoryStats.physFootprintMB()
            let fullResult = try await adapter.dispatch(
                tool: intent.toolName, args: dictArgs
            )
            let dispatchDuration = ProcessInfo.processInfo.systemUptime - dispatchStarted
            activeQueryTelemetry?.recordTool(
                name: intent.toolName,
                duration: dispatchDuration,
                usedZimKinds: telemetryZimKinds(from: fullResult))
            // Record subject + drift threads for the next follow-up.
            updateFocusAfterTool(
                toolName: intent.toolName, args: dictArgs, result: fullResult)
            let resultData = (try? JSONSerialization.data(
                withJSONObject: fullResult, options: [.sortedKeys]
            )) ?? Data()
            debug(String(format:
                "fast-path %@ returned %d bytes in %.3fs · Δmem=%+.1f MB",
                intent.toolName, resultData.count,
                dispatchDuration,
                MemoryStats.physFootprintMB() - dispatchMemory),
                category: "Tool")
            let rawStr = String(data: resultData, encoding: .utf8) ?? "{}"
            recordToolTrace(ToolCallTrace(
                name: intent.toolName,
                arguments: argsStr,
                result: rawStr,
                rawResult: rawStr,
                error: nil
            ))
            if intent.toolName == "narrate_article" {
                let body = (fullResult["text"] as? String) ?? ""
                let passThrough = (fullResult["pass_through"] as? Bool) == true
                guard passThrough, !body.isEmpty else {
                    let title = (dictArgs["title"] as? String) ?? "that topic"
                    updateAssistant("I couldn't read the full article on \(title) from the loaded Wikipedia archive.")
                    debug("narrate fast path returned no body", category: "Router")
                    return true
                }
                noteReadingState(
                    toolName: "narrate_article",
                    args: dictArgs,
                    result: fullResult)
                noteDiscussionAnchor(
                    toolName: "narrate_article",
                    result: fullResult)
                updateAssistant(body)
                if let idx = messages.indices.last,
                   messages[idx].role == .assistant
                {
                    messages[idx].groundingSources = [GroundingSource(
                        kind: .wikipedia,
                        title: (fullResult["title"] as? String)
                            ?? (dictArgs["title"] as? String) ?? "Wikipedia",
                        section: nil,
                        library: fullResult["zim"] as? String)]
                }
                debug("narrate fast path: emitted \(body.count) chars; no LLM",
                      category: "Router")
                return true
            }
            if intent.toolName == "discuss_article" {
                return await handleDiscussEntry(
                    dictArgs: dictArgs, fullResult: fullResult)
            }
            let placesTools: Set<String> = [
                "near_named_place", "near_places",
                "nearby_stories", "nearby_stories_at_place",
                "locate",
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
                case "article_factoid":
                    return IntentRouter.articleFactoidResultIsUsable(fullResult)
                case "what_is_here":
                    return IntentRouter.whatIsHereResultIsUsable(fullResult)
                default:
                    // Places + routing tools have their own
                    // empty-results handling in their synth.
                    return true
                }
            }()
            if !usable {
                if intent.toolName == "article_factoid" {
                    if (fullResult["ambiguous"] as? Bool) == true {
                        let synth = IntentRouter.synthesizeArticleFactoidReply(
                            args: dictArgs, fullResult: fullResult)
                        updateAssistant(synth)
                        let rows = (fullResult["disambiguation"]
                            as? [[String: Any]]) ?? []
                        let predicate = (dictArgs["predicate"] as? String)
                            ?? "foundation"
                        let choices = rows.compactMap { row -> DiscoveryThread? in
                            guard let title = row["title"] as? String,
                                  !title.isEmpty else { return nil }
                            let prompt = predicate == "age"
                                ? "How old is \(title)?"
                                : "When was \(title) founded?"
                            return DiscoveryThread(
                                label: title,
                                kind: .topic,
                                source: .wikilink,
                                zimPath: row["path"] as? String,
                                prompt: prompt)
                        }
                        if let idx = messages.indices.last,
                           messages[idx].role == .assistant {
                            messages[idx].suggestions = choices
                        }
                        focus.setLastList(rows.compactMap { row -> FocusEntity? in
                            guard let title = row["title"] as? String,
                                  !title.isEmpty else { return nil }
                            return FocusEntity(
                                name: title, kind: .topic,
                                zimPath: row["path"] as? String)
                        })
                        pendingFactoidPredicate = predicate
                        debug("factoid ambiguity — offered "
                            + choices.map(\.label).joined(separator: " | ")
                            + "; no LLM", category: "Router")
                        return true
                    }
                    if let notes = fullResult["diagnostics"] as? [String],
                       !notes.isEmpty {
                        debug("factoid diagnostics: "
                            + notes.joined(separator: " | "), category: "Router")
                    }
                    // An incomplete "When was X?" only produces a factoid
                    // when the adapter proved X is a company. On a miss,
                    // immediately open the same Wikipedia title instead of
                    // dropping into an ungrounded model turn. Besides avoiding
                    // preknowledge, this establishes discussion focus so
                    // follow-ups such as "how many died there?" and "who were
                    // the combatants?" stay attached to the event.
                    if (dictArgs["implicit"] as? Bool) == true
                        || (dictArgs["tentative"] as? Bool) == true {
                        if let idx = messages.indices.last,
                           messages[idx].role == .assistant {
                            messages[idx].toolCalls.removeAll()
                            messages[idx].text = ""
                        }
                        if let title = dictArgs["title"] as? String,
                           !title.isEmpty {
                            debug("tentative factoid miss — opening Wikipedia topic \(title)",
                                  category: "Router")
                            return await executeDirectIntent(DirectIntent(
                                toolName: "article_overview",
                                args: ["title": .string(title)]))
                        }
                        return false
                    }
                    let synth = IntentRouter.synthesizeArticleFactoidReply(
                        args: dictArgs, fullResult: fullResult)
                    updateAssistant(synth)
                    debug("factoid evidence miss — skipping LLM",
                          category: "Router")
                    return true
                }
                // article_overview misses are deterministic — the LLM
                // has no hidden knowledge of what's in the offline ZIM,
                // and when handed a missed title it confabulates a wrong
                // entity (real capture 2026-05-29: a mis-transcribed
                // "Dutch Lithuania" → "…the Dutch Republic"). Say we
                // couldn't find it, offer the closest real titles, and
                // STOP — don't fall through to the LLM.
                if intent.toolName == "article_overview" {
                    // Voice dictation drops possessive apostrophes
                    // ("putins childhood" — real capture 2026-07-01).
                    // Before dead-ending in a did-you-mean, retry ONCE
                    // with the aggressive possessive strip; a wrong
                    // guess just re-misses into the same reply.
                    if let title = dictArgs["title"] as? String {
                        let retry = IntentRouter.stripPossessiveFacetAggressive(from: title)
                        if retry != title {
                            var retryArgs = dictArgs
                            retryArgs["title"] = retry
                            if let second = try? await adapter.dispatch(
                                tool: "article_overview", args: retryArgs),
                               IntentRouter.articleOverviewResultIsUsable(second)
                            {
                                debug("article miss — possessive retry hit “\(retry)”",
                                      category: "Router")
                                return await executeDirectIntent(DirectIntent(
                                    toolName: "article_overview",
                                    args: ["title": .string(retry)]))
                            }
                        }
                    }
                    // Descriptive-phrase rescue: "the ones Einstein
                    // predicted" is a DESCRIPTION, not a title — search
                    // for it (folding in the prior subject when the
                    // phrase is deictic) and open the top hit. Real
                    // capture 2026-07-02: "gravity waves" resolved to
                    // the fluid-dynamics article; the correction "No, I
                    // meant the ones Einstein predicted" was dispatched
                    // as a literal title and dead-ended.
                    if let title = dictArgs["title"] as? String {
                        // Content words only — deictic filler ("the ones")
                        // dragged the search to the wrong article.
                        let kws = ArticleHeuristics.questionKeywords(title)
                        var query = kws.isEmpty ? title : kws.joined(separator: " ")
                        let lower = title.lowercased()
                        if lower.hasPrefix("the one") || lower.hasPrefix("that one")
                            || lower.hasPrefix("those "),
                           let prior = focus.primaryEntity?.name {
                            query = prior + " " + query
                        }
                        if let search = try? await adapter.dispatch(
                            tool: "search", args: ["query": query, "limit": 3]),
                           let hits = search["hits"] as? [[String: Any]],
                           let topTitle = hits.first?["title"] as? String,
                           !topTitle.isEmpty,
                           topTitle.lowercased() != lower
                        {
                            if let second = try? await adapter.dispatch(
                                tool: "article_overview", args: ["title": topTitle]),
                               IntentRouter.articleOverviewResultIsUsable(second)
                            {
                                debug("article miss — search rescue “\(query)” → “\(topTitle)”",
                                      category: "Router")
                                return await executeDirectIntent(DirectIntent(
                                    toolName: "article_overview",
                                    args: ["title": .string(topTitle)]))
                            }
                        }
                    }
                    let synth = IntentRouter.synthesizeArticleMissReply(
                        args: dictArgs, fullResult: fullResult)
                    updateAssistant(synth)
                    debug("article miss — did-you-mean, skipping LLM",
                          category: "Router")
                    return true
                }
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
            } else if intent.toolName == "article_factoid" {
                noteDiscussionAnchor(toolName: "article_overview",
                                     result: fullResult)
                let synth = IntentRouter.synthesizeArticleFactoidReply(
                    args: dictArgs, fullResult: fullResult)
                updateAssistant(synth)
                debug(synth, category: "Assistant")
                let resolution = (fullResult["resolution"] as? String) ?? "unknown"
                debug("factoid fast path → grounded Wikipedia lead (\(resolution)); no LLM",
                      category: "Router")
                await appendThreadOfferIfUseful()
                return true
            } else if intent.toolName == "article_overview"
                   || intent.toolName == "compare_articles"
            {
                if intent.toolName == "article_overview" {
                    // Remember the article so "continue" can page on
                    // after the spoken summary.
                    noteReadingState(toolName: "article_overview",
                                     args: dictArgs, result: fullResult)
                    noteDiscussionAnchor(toolName: "article_overview",
                                         result: fullResult)
                }
                // Answer as a GROUNDED SINGLE-SHOT over the fetched
                // sections — the same machinery discuss mode uses — with
                // NO conversation history in the prompt. The previous
                // design injected a synthetic tool round-trip and let
                // `runGenerationLoop` summarise with the full transcript;
                // with unrelated prior turns in context the FT parroted
                // a previous question verbatim instead of summarising
                // (device + Mac captures 2026-07-02, "And tell me about
                // Donald Trump" → reply "How about his mother?").
                let question = messages.last(where: { $0.role == .user })?.text
                    ?? "Tell me about this."
                var sources: [(title: String, sections: [ArticleSection])] = []
                func sections(from dict: [String: Any]) -> [ArticleSection] {
                    ((dict["sections"] as? [[String: Any]]) ?? [])
                        .map(Self.decodedArticleSection)
                }
                if let articles = fullResult["articles"] as? [[String: Any]] {
                    for a in articles {
                        let t = (a["title"] as? String) ?? ""
                        let secs = sections(from: a)
                        if !t.isEmpty, !secs.isEmpty { sources.append((t, secs)) }
                    }
                } else {
                    let t = (fullResult["title"] as? String)
                        ?? (dictArgs["title"] as? String) ?? "the article"
                    let secs = sections(from: fullResult)
                    if !secs.isEmpty { sources.append((t, secs)) }
                }
                guard !sources.isEmpty else {
                    updateAssistant("I found the article but couldn't read its sections.")
                    return true
                }
                let anchor = sources[0].title
                let grounded = DiscussionState(
                    anchorTitle: anchor,
                    topic: ArticleHeuristics.topicCore(anchor),
                    zim: fullResult["zim"] as? String,
                    sources: sources,
                    linkedArticleTitles: [],
                    sectionEmbeddings: [:],
                    lastQuestion: nil)
                debug("fast-path \(intent.toolName) → grounded single-shot over \(sources.count) source(s)",
                      category: "Router")
                await generateGroundedAnswer(state: grounded, question: question)
                // Official ambiguity ("gravity waves" → fluid OR Einstein):
                // name the alternate meanings and register them as the
                // selectable list, so "the second one" / "the Einstein one"
                // switches without a fight (real capture 2026-07-02).
                if let alts = fullResult["disambiguation"] as? [[String: Any]] {
                    let names = alts.compactMap { $0["title"] as? String }.prefix(3)
                    if !names.isEmpty,
                       let idx = messages.indices.last,
                       messages[idx].role == .assistant,
                       !messages[idx].text.isEmpty
                    {
                        let list = names.count == 1
                            ? names[names.startIndex]
                            : names.dropLast().joined(separator: ", ")
                                + " or \(names.last!)"
                        // Keep the exact emission for prompt rebuilds — the
                        // appendix was never generated (A2).
                        if messages[idx].rawAssistantText == nil {
                            messages[idx].rawAssistantText = messages[idx].text
                        }
                        messages[idx].text +=
                            "\n\n(\"\(anchor)\" has other meanings too — say the word if you meant \(list).)"
                        focus.setLastList(
                            [FocusEntity(name: anchor, kind: .topic,
                                         zimPath: fullResult["path"] as? String)]
                            + alts.compactMap { a -> FocusEntity? in
                                guard let t = a["title"] as? String else { return nil }
                                return FocusEntity(name: t, kind: .topic,
                                                   zimPath: a["path"] as? String)
                            })
                        debug("disambiguation offered: \(names.joined(separator: " | "))",
                              category: "Router")
                    }
                }
                await appendThreadOfferIfUseful()
                return true
            } else if intent.toolName == "what_is_here" {
                let synth = IntentRouter.synthesizeWhatIsHereReply(
                    fullResult: fullResult
                )
                updateAssistant(synth.isEmpty ? "Location below." : synth)
            } else {
                updateAssistant("Results below.")
            }
            // Terminal fast-path replies (places / routing / what_is_here)
            // get the same "where next" offer the LLM path appends. The
            // article_overview / compare_articles branches returned `false`
            // above and hand off to the LLM, which offers there instead.
            await appendThreadOfferIfUseful()
            return true
        } catch {
            debug("fast-path dispatch failed: \(error)", category: "Tool")
            // Geocoding misses are deterministic — the LLM has no
            // hidden knowledge of which places are in the loaded
            // streetzim. Burning 15–20 s to have the model re-run
            // the same tool with the same args and hit the same
            // miss is pure latency cost with no better outcome.
            // On-device repro (2026-04-22 gist 007b1a69): "Bars in
            // north beach" with only a Palo Alto streetzim loaded
            // — fast-path said `could not resolve`, LLM was called
            // anyway, produced the same no-results message after
            // 20 seconds. Short-circuit with a clear error for
            // places/routing tools; keep the LLM retry for the
            // article path where a different title guess can help.
            let placesAndRouting: Set<String> = [
                "near_named_place", "near_places", "nearby_stories",
                "nearby_stories_at_place", "route_from_places",
                "plan_driving_route", "what_is_here", "locate",
            ]
            let errText = String(describing: error)
            let isGeocodeMiss = errText.contains("could not resolve")
                || errText.contains("noMatch")
                || errText.contains("no matching")
            if placesAndRouting.contains(intent.toolName), isGeocodeMiss {
                let subject = placeSubject(from: dictArgs) ?? "that place"
                updateAssistant(
                    "I can't find \"\(subject)\" in the loaded maps. "
                    + "The current streetzim may not cover that area — "
                    + "try a place within its region, or load a "
                    + "streetzim that includes it."
                )
                return true
            }
            if let idx = messages.indices.last,
               messages[idx].role == .assistant
            {
                messages[idx].text = ""
                messages[idx].toolCalls.removeAll()
            }
            return false
        }
    }

    /// Pull whichever "place" identifier a places/routing tool was
    /// called with so the fast-path error caption can quote it back
    /// to the user. Keys we've shipped over the life of the adapter:
    /// `place`, `destination`, `origin`, `query`.
    private func placeSubject(from args: [String: Any]) -> String? {
        for key in ["place", "destination", "origin", "query"] {
            if let v = args[key] as? String, !v.isEmpty { return v }
        }
        return nil
    }

    // MARK: - Discourse state updates

    /// Fold a completed tool call into the conversation focus: record the
    /// subject entity, capture any enumerated list (so "the second one"
    /// resolves next turn), and extract + rank the vetted drift threads.
    /// Pure bookkeeping — never throws, never blocks.
    private func updateFocusAfterTool(
        toolName: String, args: [String: Any], result: [String: Any]
    ) {
        func dbl(_ v: Any?) -> Double? {
            if let d = v as? Double { return d }
            if let i = v as? Int { return Double(i) }
            if let n = v as? NSNumber { return n.doubleValue }
            return nil
        }

        // A missed fetch is not a subject. Recording it would make the
        // phantom title the pronoun target for the next turn ("tell me
        // more" → re-fetch of an article that doesn't exist), and its
        // "threads" would be junk. Leave focus exactly as it was.
        if result["error"] != nil { return }

        let topicTools: Set<String> = [
            "article_overview", "article_factoid", "compare_articles",
            "get_article_section", "narrate_article",
        ]
        if topicTools.contains(toolName),
           let title = (result["title"] as? String) ?? (args["title"] as? String),
           !title.isEmpty {
            // Use one identity key for both the focus entity and the touch-
            // index so `centroid(of: focus keys)` lines up with indexed
            // vectors. Prefer the ZIM path; fall back to the title.
            let rawPath = (result["path"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            let key = rawPath ?? title
            focus.remember(FocusEntity(name: title, kind: .topic, zimPath: key))
            if let lead = Self.leadText(from: result) {
                indexText(key: key, title: title, text: lead)
            }
        }

        let placesTools: Set<String> = [
            "locate", "near_named_place", "near_places",
            "nearby_stories", "nearby_stories_at_place",
        ]
        if placesTools.contains(toolName) {
            // The named place the user searched in is the discussion anchor;
            // record it first so the enumerated list head ends up primary.
            if let place = args["place"] as? String, !place.isEmpty {
                focus.remember(FocusEntity(name: place, kind: .place))
            }
            let rows = (result["results"] as? [[String: Any]])
                ?? (result["stories"] as? [[String: Any]]) ?? []
            let list: [FocusEntity] = rows.compactMap { row in
                let name = (row["wiki_title"] as? String) ?? (row["label"] as? String)
                    ?? (row["name"] as? String) ?? (row["title"] as? String)
                guard let name, !name.isEmpty else { return nil }
                return FocusEntity(
                    name: name, kind: .place,
                    zimPath: row["wiki_path"] as? String,
                    lat: dbl(row["lat"]), lon: dbl(row["lon"]))
            }
            if !list.isEmpty { focus.setLastList(list) }
        }

        if toolName == "compare_articles", let titles = args["titles"] as? [String] {
            let list = titles.filter { !$0.isEmpty }
                .map { FocusEntity(name: $0, kind: .topic) }
            if !list.isEmpty { focus.setLastList(list) }
        }

        if toolName == "route_from_places" {
            // Prefer the RESOLVED destination — it carries the canonical name
            // plus coordinates, so a follow-up "what's near there?" / "how far
            // back?" has real coords instead of re-geocoding a bare string.
            let destResolved = result["destination_resolved"] as? [String: Any]
            let destName = (destResolved?["name"] as? String)
                ?? (args["destination"] as? String) ?? ""
            if !destName.isEmpty {
                focus.remember(FocusEntity(
                    name: destName, kind: .place,
                    lat: dbl(destResolved?["lat"]), lon: dbl(destResolved?["lon"])))
            }
        }

        if toolName == "what_is_here" {
            // "Where am I?" must not be a conversational dead end: remember the
            // place so "tell me more" re-opens its article and "what's near
            // here?" reuses its coordinates. Prefer the resolved wiki title (a
            // clean article handle) over the bare admin name.
            let placeName = (result["wiki_title"] as? String)
                ?? (result["nearest_named_place"] as? String) ?? ""
            if !placeName.isEmpty {
                let lat = dbl(result["place_lat"]) ?? dbl(result["lat"])
                let lon = dbl(result["place_lon"]) ?? dbl(result["lon"])
                focus.remember(FocusEntity(
                    name: placeName, kind: .place, lat: lat, lon: lon))
                if let summary = result["wiki_summary"] as? String, !summary.isEmpty {
                    indexText(key: placeName, title: placeName, text: summary)
                }
            }
        }

        // Persist chat-planned routes so `route_status` ("how much
        // longer?") works. Previously ONLY the Siri App Intent path
        // called `setActiveRoute` — a route planned by typing/voice in
        // the app left `activeRoute == nil` and route_status errored.
        if toolName == "route_from_places" || toolName == "plan_driving_route" {
            persistActiveRoute(args: args, result: result)
        }

        let threads = ConversationThreads.rank(
            ConversationThreads.extract(toolName: toolName, result: result),
            focus: focus, max: 4)
        // Keep the previous turn's threads when this tool contributed
        // none — a drill-in (`get_article_section` before it carried
        // `related[]`, `route_status`, a thin `search`) used to WIPE the
        // open threads, killing the "where next" offer exactly when the
        // user engaged. Stale-but-grounded beats empty.
        if !threads.isEmpty { focus.setThreads(threads) }
    }

    /// Build an `ActiveRoute` from a successful routing tool result and
    /// store it in `ZimfoContext` (fire-and-forget actor hop). Mirrors
    /// the construction in `ZimfoIntents.PlanRouteIntent.perform`.
    private func persistActiveRoute(args: [String: Any], result: [String: Any]) {
        func dbl(_ v: Any?) -> Double? {
            if let d = v as? Double { return d }
            if let i = v as? Int { return Double(i) }
            if let n = v as? NSNumber { return n.doubleValue }
            return nil
        }
        guard let polyRaw = result["polyline"] as? [[Double]],
              polyRaw.count >= 2,
              polyRaw.allSatisfy({ $0.count >= 2 })
        else { return }
        let totalDist = dbl(result["distance_m"])
            ?? dbl(result["distance_km"]).map({ $0 * 1000 }) ?? 0
        let totalDur = dbl(result["duration_s"])
            ?? dbl(result["duration_min"]).map({ $0 * 60 }) ?? 0
        guard totalDist > 0 else { return }
        var cum: [Double] = [0]
        cum.reserveCapacity(polyRaw.count)
        for i in 1..<polyRaw.count {
            let d = RouteProgress.haversineMetersApprox(
                polyRaw[i - 1][0], polyRaw[i - 1][1],
                polyRaw[i][0], polyRaw[i][1])
            cum.append(cum[i - 1] + d)
        }
        let originName = ((result["origin_resolved"] as? [String: Any])?["name"] as? String)
            ?? (args["origin"] as? String) ?? "here"
        let destName = ((result["destination_resolved"] as? [String: Any])?["name"] as? String)
            ?? (args["destination"] as? String) ?? "destination"
        let route = ActiveRoute(
            startedAt: Date(),
            origin: .init(lat: polyRaw.first![0], lon: polyRaw.first![1]),
            destination: .init(lat: polyRaw.last![0], lon: polyRaw.last![1]),
            originName: originName,
            destinationName: destName,
            zim: (result["zim"] as? String) ?? "",
            totalDurationSeconds: totalDur,
            totalDistanceMeters: totalDist,
            polyline: polyRaw.map { .init(lat: $0[0], lon: $0[1]) },
            cumulativeDistanceMeters: cum,
            turnByTurn: (result["turn_by_turn"] as? [String]) ?? []
        )
        debug("persisting active route \(originName) → \(destName) for route_status",
              category: "Router")
        Task { await ZimfoContext.shared.setActiveRoute(route) }
    }

    /// Pull the lead prose out of an article-shaped tool result for embedding.
    private static func leadText(from result: [String: Any]) -> String? {
        if let sections = result["sections"] as? [[String: Any]] {
            for s in sections {
                if let t = s["text"] as? String, !t.isEmpty { return t }
            }
        }
        for key in ["lead", "text", "summary", "preview"] {
            if let t = result[key] as? String, !t.isEmpty { return t }
        }
        return nil
    }

    /// Fire-and-forget: embed `text` and add it to the touch-index under
    /// `key`. Best-effort and off the turn's critical path — the vector is for
    /// FUTURE turns' recall, so a slow embed never blocks the reply. No-op when
    /// the embedding model is unavailable.
    private func indexText(key: String, title: String, text: String) {
        guard !key.isEmpty, !text.isEmpty else { return }
        let index = embeddingIndex
        Task {
            if await index.contains(key) { return }
            if let vec = await SemanticReranker.shared.embedText(text) {
                await index.add(key: key, title: title, vector: vec)
            }
        }
    }

    /// Local-area seed-index entry point, called from the location stream.
    /// Throttles on distance (only a new area re-seeds) then kicks off a
    /// fire-and-forget seed. No-op on jitter, while generating, or off-area.
    private func seedNearbyPlacesIfMoved(lat: Double, lon: Double) {
        let here = CLLocation(latitude: lat, longitude: lon)
        if let last = lastSeedLocation {
            let prev = CLLocation(latitude: last.lat, longitude: last.lon)
            if here.distance(from: prev) < Self.seedReseedMeters { return }
        }
        lastSeedLocation = (lat, lon)
        Task { [weak self] in
            guard let self else { return }
            // Don't compete with an active turn for ZIM/IO; the next fix or the
            // user's own near-me query will seed instead.
            if self.isGenerating { return }
            await self.seedNearbyPlaces(lat: lat, lon: lon)
        }
    }

    /// Pull wiki-backed places around `(lat,lon)` and embed each one's lead
    /// excerpt into the touch-index, keyed by ZIM path (so it lines up with
    /// the `focus` entity keys the centroid is built from). Reuses the same
    /// `near_places` dispatch the model uses, so enrichment/vetting is shared.
    /// Best-effort: a failed dispatch just leaves the index as-is.
    private func seedNearbyPlaces(lat: Double, lon: Double) async {
        let args: [String: Any] = [
            "lat": lat,
            "lon": lon,
            "radius_km": Self.seedRadiusKm,
            "limit": Self.seedMaxPlaces,
            // Filter to wiki-backed places so the whole `limit` budget goes to
            // rows we can actually embed (a bare bar/shop has no lead to index).
            "has_wiki": true,
        ]
        guard let adapter = self.adapter else { return }
        guard let result = try? await adapter.dispatch(
            tool: "near_places", args: args
        ) else { return }
        // near_places encodes its rows under "results"; each wiki-backed row
        // carries `wiki_path` plus an excerpt under `excerpt` (and a mirrored
        // `wiki_excerpt`). Read either so a field rename can't silently zero
        // the seed.
        let rows = (result["results"] as? [[String: Any]]) ?? []
        for row in rows {
            guard let path = row["wiki_path"] as? String, !path.isEmpty
            else { continue }
            let excerpt = (row["excerpt"] as? String)
                ?? (row["wiki_excerpt"] as? String) ?? ""
            guard !excerpt.isEmpty else { continue }
            let title = (row["wiki_title"] as? String) ?? path
            // indexText dedupes via the index's `contains` and embeds off the
            // critical path, so re-seeding overlapping areas is cheap.
            indexText(key: path, title: title, text: excerpt)
        }
    }

    /// Append a short "where to go next" line to the current assistant reply,
    /// drawn from the vetted drift threads. When the touch-index has the
    /// conversation's gist, the threads are first re-ranked by semantic
    /// similarity to that centroid so the offer follows the whole stroll, not
    /// just the last sentence; otherwise the deterministic source order stands.
    /// Place threads are only offered when wiki-backed (a bare POI like a bar
    /// isn't something to "hear about"). Skipped when the model already ended
    /// with its own offer, or the reply is empty.
    private func appendThreadOfferIfUseful() async {
        var offerable = focus.openThreads.filter {
            ConversationThreads.isUserFacing($0)
                && ($0.kind != .place || $0.zimPath != nil)
                && !recentlyOfferedThreadKeys.contains($0.matchKey)
        }
        guard !offerable.isEmpty else { return }
        offerable = await rerankBySimilarity(offerable)
        guard let line = ConversationThreads.offer(offerable) else { return }
        guard let idx = messages.indices.last,
              messages[idx].role == .assistant else { return }
        let text = messages[idx].text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        // Grounded article answers install question-shaped section chips
        // directly ("How was it first detected?"). Keep those instead of
        // replacing them with barely contextual wikilink labels and do not
        // append a redundant prose "Want to hear about…" line.
        if !messages[idx].suggestions.isEmpty { return }
        // Attach the vetted threads as tappable chips regardless of whether we
        // also append a prose offer below — the chips make any offer (ours or
        // the model's own phrasing) actionable with a single tap.
        messages[idx].suggestions = offerable
        let tail = text.suffix(90).lowercased()
        if tail.contains("want to") || tail.contains("would you like")
            || tail.contains("shall i") || tail.contains("tell you about")
            || tail.contains("i can tell you") {
            return
        }
        // Preserve the model's exact emission for prompt rebuilds BEFORE
        // mutating the display text — the offer line and the whitespace trim
        // below were never generated, and re-feeding them diverges the KV
        // prefix at the last assistant reply (PERFORMANCE_REVIEW.md A2).
        if messages[idx].rawAssistantText == nil {
            messages[idx].rawAssistantText = messages[idx].text
        }
        messages[idx].text = text + "\n\n" + line
        // Remember what we offered so the next turn's line is fresh.
        // `offer()` phrases the first 3 threads — mark exactly those.
        recentlyOfferedThreadKeys.append(
            contentsOf: offerable.prefix(3).map(\.matchKey))
        if recentlyOfferedThreadKeys.count > 12 {
            recentlyOfferedThreadKeys.removeFirst(
                recentlyOfferedThreadKeys.count - 12)
        }
    }

    /// A tapped suggestion chip. Clears the chips on the offering message (so a
    /// pick doesn't leave a stale offer behind) and dispatches it as a normal
    /// turn — the router re-opens a topic via `article_overview`, identical to
    /// the user typing or saying "tell me about <label>". Mirrors the
    /// `ReferenceResolver` `.thread` binding that already handles the spoken
    /// "yes" / "the war" path.
    public func selectSuggestion(_ thread: DiscoveryThread) {
        if let idx = messages.indices.last, messages[idx].role == .assistant {
            messages[idx].suggestions = []
        }
        send(thread.prompt ?? "tell me about \(thread.label)")
    }

    /// Order candidate threads by cosine similarity of each thread's label to
    /// the centroid of the articles touched this conversation. Returns the
    /// input unchanged when the index is cold or the model is unavailable.
    private func rerankBySimilarity(
        _ threads: [DiscoveryThread]
    ) async -> [DiscoveryThread] {
        let focusKeys = focus.entities.compactMap { $0.zimPath }
            .filter { !$0.isEmpty }
        guard !focusKeys.isEmpty,
              let centroid = await embeddingIndex.centroid(of: focusKeys)
        else { return threads }
        var scores: [String: Float] = [:]
        for t in threads {
            let key = (t.zimPath?.isEmpty == false) ? t.zimPath! : t.matchKey
            if let v = await SemanticReranker.shared.embedText(t.label) {
                scores[key] = VectorMath.cosine(v, centroid)
            }
        }
        guard !scores.isEmpty else { return threads }
        return ConversationThreads.orderBySimilarity(threads, scores: scores)
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
    static func trimForModel(toolName: String, result: [String: Any], articleCapKB: Int) -> [String: Any] {
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
        case "near_named_place", "near_places", "nearby_stories",
             "nearby_stories_at_place", "locate", "what_is_here":
            // The UI/map keeps the complete raw payload, but the model only
            // needs the closest handful of rows. A 25-POI StreetZIM result
            // pushed the Bonsai 27B 4K evaluation prompt to 4,228 tokens —
            // just over its phone-sized context — before the user even asked
            // the follow-up. Bound both row count and per-row prose here.
            var out = result
            let rowCap = 8
            let proseCap = 240
            func compactRows(_ rows: [[String: Any]]) -> [[String: Any]] {
                rows.prefix(rowCap).map { row in
                    var compact = row
                    for key in ["description", "summary", "wiki_summary",
                                "excerpt", "text"] {
                        if let value = compact[key] as? String,
                           value.count > proseCap
                        {
                            compact[key] = String(value.prefix(proseCap)) + "…"
                        }
                    }
                    // These fields feed maps/debugging, not prose synthesis.
                    for key in ["tags", "raw_tags", "geometry", "polyline"] {
                        compact[key] = nil
                    }
                    return compact
                }
            }
            for key in ["results", "stories", "nearby"] {
                if let rows = out[key] as? [[String: Any]] {
                    if rows.count > rowCap { out[key + "_total"] = rows.count }
                    out[key] = compactRows(rows)
                }
            }
            if let categories = out["by_category"] as? [[String: Any]],
               categories.count > 12
            {
                out["by_category_total"] = categories.count
                out["by_category"] = Array(categories.prefix(12))
            }
            return out
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
            // The composite tools pre-chew: lead + the 1–2 most
            // informative narrative sections (`pickOverview`) — or,
            // for compares, the relations-article sections that
            // mention the counterpart. Feeding whole articles is off
            // the table (on-device repro: two full 15–30 KB articles
            // jetsam'd the app mid-summary), but the previous
            // lead-ONLY trim threw the picked sections away too, so
            // "history of X" / "how have A and B gotten along" reached
            // the model as 160 words of lead and nothing else.
            //
            // Keep the lead (word-capped at 160 ≈ 200 tokens) PLUS up
            // to two picked sections (word-capped at 120 each). Worst
            // case per article ≈ 400 words ≈ 520 tokens; a two-article
            // compare ≈ 1 K tokens — an order of magnitude under the
            // raw-article payload that caused the jetsam. Word-based
            // truncation keeps boundaries clean so the model doesn't
            // see "...founded in 19".
            let leadWordCap = 160
            let sectionWordCap = 120
            let extraSectionCap = 2
            func wordCapped(_ text: String, cap: Int) -> (text: String, truncated: Bool) {
                let words = text.split(separator: " ",
                                       omittingEmptySubsequences: false)
                guard words.count > cap else { return (text, false) }
                return (words.prefix(cap).joined(separator: " ") + "…", true)
            }
            func trimSections(_ sections: [[String: Any]]) -> [[String: Any]] {
                guard let lead = sections.first else { return [] }
                var out: [[String: Any]] = []
                var trimmedLead = lead
                if let text = lead["text"] as? String {
                    let capped = wordCapped(text, cap: leadWordCap)
                    trimmedLead["text"] = capped.text
                    if capped.truncated { trimmedLead["truncated"] = true }
                }
                out.append(trimmedLead)
                for section in sections.dropFirst().prefix(extraSectionCap) {
                    var trimmed = section
                    if let text = section["text"] as? String {
                        let capped = wordCapped(text, cap: sectionWordCap)
                        trimmed["text"] = capped.text
                        if capped.truncated { trimmed["truncated"] = true }
                    }
                    out.append(trimmed)
                }
                if sections.count > out.count {
                    out[out.count - 1]["sections_dropped"] = sections.count - out.count
                }
                return out
            }
            var out = result
            if let sections = out["sections"] as? [[String: Any]] {
                out["sections"] = trimSections(sections)
            }
            if let articles = out["articles"] as? [[String: Any]] {
                out["articles"] = articles.map { a -> [String: Any] in
                    var inner = a
                    if let sections = a["sections"] as? [[String: Any]] {
                        inner["sections"] = trimSections(sections)
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
    /// Keep this schema independent of the user's installed ZIM filenames.
    /// `zim` is optional and MCPToolAdapter sanitizes invented filenames, so
    /// baking the live inventory into an enum only made the expensive static
    /// model prefix differ for every library (and made a release-seeded cache
    /// impossible to reuse).
    private func toolDeclarations(registry: MCPToolRegistry) -> [ModelToolDeclaration] {
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
                let enumValues = (raw["enum"] as? [Any])?.compactMap {
                    $0 as? String
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
