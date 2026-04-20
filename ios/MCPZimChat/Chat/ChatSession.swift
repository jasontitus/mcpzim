// SPDX-License-Identifier: MIT
//
// Top-level observable app state. Owns the list of opened ZIM readers, the
// set of available models, the current chat transcript, and a reference to
// the MCPZim tool adapter that a Gemma 4 tool loop can dispatch through.

import Foundation
import MCPZimKit
import Observation

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
    }

    @ObservationIgnored private var stateObservationTask: Task<Void, Never>?

    /// Last known current location, if the user has granted permission.
    /// Injected into the system preamble so "directions to X" can
    /// default to "from here" without the user having to name an
    /// origin. Refreshed lazily on each new turn.
    public var currentLocation: (lat: Double, lon: Double)? = nil
    @ObservationIgnored private var lastLocationFetch: Date = .distantPast

    /// Warm the expensive start-up caches off the user's hot path.
    /// Called from `RootView` at launch. Intentionally concurrent —
    /// streetzim graph parse, reranker asset load, and location fix
    /// are all independent, so there's no point serialising them.
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
    }

    /// Kick off a location fetch if we haven't had a fresh fix in the
    /// last two minutes. Non-blocking — the preamble uses whatever we
    /// last saw, so a first-query user's reply doesn't stall on GPS.
    /// The first launch also triggers the `WhenInUse` permission
    /// prompt via `CLLocationManager`, so use a generous timeout so
    /// the user has time to tap Allow.
    public func refreshLocationIfStale() {
        #if canImport(UIKit)
        guard Date().timeIntervalSince(lastLocationFetch) > 120 else { return }
        lastLocationFetch = Date()
        Task { [weak self] in
            do {
                let here = try await LocationFetcher.once(timeout: 20)
                await MainActor.run {
                    self?.currentLocation = (here.latitude, here.longitude)
                    self?.debug("location fixed: (\(here.latitude), \(here.longitude))",
                                category: "Voice")
                }
            } catch {
                await MainActor.run {
                    self?.debug("location unavailable: \(error)", category: "Voice")
                }
            }
        }
        #endif
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
        let gemma = Gemma4Provider()
        var providers: [any ModelProvider] = [gemma]
        if #available(macOS 26.0, iOS 19.0, *) {
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
        gemma.debugSink = { [weak self] msg in
            Task { @MainActor [weak self] in
                self?.debug(msg, category: "Gemma4")
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
        Task { @MainActor in await self.loadSelectedModel() }
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
    }

    public func setEnabled(_ enabled: Bool, for entryID: LibraryEntry.ID) async {
        guard let idx = library.firstIndex(where: { $0.id == entryID }) else { return }
        guard library[idx].isEnabled != enabled else { return }
        library[idx].isEnabled = enabled
        await rebuildService()
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
        debug("conversation reset", category: "Chat")
    }

    /// Classification of the most recent user turn. Set in `send()`
    /// and read by the generation loop so the system message can
    /// carry category-specific guidance (Phase 2b).
    public private(set) var lastQueryComplexity: QueryComplexity = .topical

    public func send(_ text: String) {
        // Refresh GPS if our last fix is stale — the preamble built in
        // `runGenerationLoop` injects `currentLocation`, so a recent
        // snapshot means "directions to X" queries Just Work.
        refreshLocationIfStale()
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
        messages.append(ChatMessage(role: .assistant, text: ""))
        isGenerating = true
        Task {
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
        defer { isGenerating = false }
        guard let adapter else {
            debug("No adapter — library is empty.", category: "Chat")
            appendAssistant("[No ZIMs loaded — add .zim files to the app's Documents folder, then tap Refresh Library.]")
            return
        }

        let registry = await adapter.registry
        let toolDecls = self.toolDeclarations(registry: registry)
        let complexity = self.lastQueryComplexity
        let categoryHint: String = {
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
        }()
        let locationLine: String
        if let here = currentLocation {
            locationLine = """

            The user's current location is approximately \
            (lat: \(String(format: "%.5f", here.lat)), \
            lon: \(String(format: "%.5f", here.lon))). \
            When they ask for directions WITHOUT naming an origin \
            (e.g. "directions to San Francisco", "how do I get to the \
            hospital"), call `plan_driving_route` with that origin \
            already filled in — do NOT ask them where they are.
            """
        } else {
            locationLine = ""
        }
        let systemMessage = """
        You are a helpful assistant with access to tools over locally-loaded \
        ZIM archives. Call tools immediately whenever they can answer the \
        user's question — do NOT ask the user to confirm, and do NOT ask \
        which ZIM to use (if there is a streetzim, use it for routing; \
        if there is a wikipedia, use it for general knowledge; if there \
        is an mdwiki, use it for medical questions). Pick sensible \
        defaults for optional arguments. Only respond in prose after \
        you have the tool result.\(locationLine)

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

        === This turn's classification ===
        \(categoryHint)

        === Grounding policy ===
        This app's value to the user is that answers are grounded in \
        the loaded ZIM archives — not in your training priors. So: \
        * Every factual claim in your reply should trace to a tool result \
          you have seen this turn OR an earlier turn in this conversation. \
        * If the user asks a follow-up that refers back to a prior topic \
          ("when was that?", "tell me more about it"), reuse the article(s) \
          already retrieved — don't answer from memory. \
        * Cite section / article names inline (e.g. "per 'Article' § \
          Causes…") whenever a claim isn't obviously common knowledge. \
        * If the loaded ZIMs genuinely don't cover the question, say that \
          — do not guess.
        """
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
        var turns: [ChatTurn] = messages
            .dropLast()
            .filter { !$0.text.isEmpty }
            .map { ChatTurn(role: $0.role.asChatTurnRole, text: $0.text) }

        // Up to 6 tool loops per user turn — enough for small models
        // that burn iterations exploring (small search → wrong zim →
        // retry) before landing on a useful answer. Still capped so
        // a genuinely stuck loop terminates.
        let maxIters = 6
        for iter in 0..<maxIters {
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
                prompt = Gemma4PromptTemplate.render(
                    systemMessage: systemMessage,
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
                maxTokens: DeviceProfile.current.maxReplyTokens,
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
                    if let call = Self.extractToolCall(in: buffer) {
                        appendToAssistant(buffer)
                        toolCall = call
                        break
                    }
                }
                // Flush the final tokens — the throttle may have skipped them.
                appendToAssistant(buffer)
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

            let argsData = try? JSONSerialization.data(withJSONObject: call.args)
            let argsStr = argsData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            let pre = String(buffer[..<call.range.lowerBound])
            // Always include the full `<|tool_call>…<tool_call|>` block in the
            // assistant turn we feed back — Gemma 4 was trained to see its own
            // tool-call emission in context when generating the continuation.
            let assistantTurnText = String(buffer[..<call.range.upperBound])
            let memBefore = MemoryStats.physFootprintMB()
            let toolStart = Date()
            debug("dispatching \(call.name)(\(argsStr)) — first call against a ZIM may block on graph/index load", category: "Tool")
            do {
                let fullResult = try await adapter.dispatch(tool: call.name, args: call.args)
                // Routing results carry a polyline with thousands of points
                // and a turn-by-turn list that together inflate to 50+ KB.
                // Feeding that verbatim into the next prompt turns into
                // ~30 000 tokens of context, which is expensive and useless
                // to the model (it can't navigate by lat/lons anyway). Trim
                // to a summary before re-prompting.
                var preTrim = fullResult
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
                    toolTurnText = Gemma4ToolFormat.formatToolResponse(name: call.name, payload: result)
                } else {
                    toolTurnText = resultStr
                }
                turns.append(ChatTurn(role: .tool, text: toolTurnText))
            } catch {
                let err = String(describing: error)
                debug("tool \(call.name) failed: \(err)", category: "Tool")
                recordToolTrace(ToolCallTrace(name: call.name, arguments: argsStr, result: "", error: err))
                if !assistantTurnText.isEmpty {
                    turns.append(ChatTurn(role: .assistant, text: assistantTurnText))
                }
                let errPayload: Any = ["error": err]
                let toolTurnText: String
                if selectedModel is Gemma4Provider {
                    toolTurnText = Gemma4ToolFormat.formatToolResponse(name: call.name, payload: errPayload)
                } else {
                    toolTurnText = "[error] \(err)"
                }
                turns.append(ChatTurn(role: .tool, text: toolTurnText))
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
                summaryPrompt = Gemma4PromptTemplate.render(
                    systemMessage: systemMessage, tools: toolDecls, turns: finalTurns
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
                prompt = Gemma4PromptTemplate.render(
                    systemMessage: preamble, tools: [], turns: turns
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
            maxTokens: DeviceProfile.current.maxReplyTokens,
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
            maxTokens: DeviceProfile.current.maxReplyTokens,
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
            messages[messages.count - 1].text = replacement
        }
    }

    private func updateAssistant(_ newText: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = newText
        }
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
    private func toolDeclarations(registry: MCPToolRegistry) -> [Gemma4ToolFormat.ToolDeclaration] {
        let zimNames = library.filter { $0.isEnabled }.map { $0.url.lastPathComponent }
        return registry.tools.map { tool -> Gemma4ToolFormat.ToolDeclaration in
            let schema = (try? JSONSerialization.jsonObject(with: tool.inputSchemaJSON)) as? [String: Any] ?? [:]
            let properties = schema["properties"] as? [String: Any] ?? [:]
            let required = Set((schema["required"] as? [String]) ?? [])
            let params: [Gemma4ToolFormat.ToolDeclaration.Parameter] = properties.keys.sorted().map { key in
                let raw = (properties[key] as? [String: Any]) ?? [:]
                let typeStr = ((raw["type"] as? String) ?? "string").lowercased()
                let type: Gemma4ToolFormat.ToolDeclaration.Parameter.ParamType = {
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

    /// Accept either format: Gemma 4's native `<|tool_call>call:NAME{…}<tool_call|>`
    /// (what the real model emits) or the legacy `<tool_call>{…json…}</tool_call>`
    /// (what Mock and some older prompts use). Whichever fires first wins.
    static func extractToolCall(in buffer: String) -> (range: Range<String.Index>, name: String, args: [String: Any])? {
        if let m = Gemma4ToolCallParser.firstCall(in: buffer) {
            return (m.range, m.name, m.arguments)
        }
        if let m = ChatToolCallParser.firstCall(in: buffer) {
            return (m.range, m.name, m.arguments)
        }
        return nil
    }
}
