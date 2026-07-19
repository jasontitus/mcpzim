// SPDX-License-Identifier: MIT
//
// End-to-end on-Mac harness for the fast-path pipeline.
//
// Loads Qwen 3 4B (or the caller's chosen model), opens a real
// Wikipedia ZIM, wires them through `ChatSession.forTesting` with a
// real libzim-backed adapter, and runs a series of example
// comparison / overview queries. For each query the harness prints:
//
//   - which tool the fast path picked (input quality)
//   - the resolved article titles / relations article (input quality)
//   - the trimmed tool result fed to the LLM (what it actually sees)
//   - the final streamed response (output quality)
//   - a short Pass / Fail verdict + any tuning suggestions
//
// Run it like:
//
//   MCPZimEvalCLI --probe-e2e \
//     --zim ~/Downloads/wikipedia_en_all_maxi_2025-10.zim
//
// Full model load happens once per run (the first time also has to
// download ~2.5 GB of weights via HuggingFace Hub). Each query then
// streams in 10–30 s on an M-series Mac.

import Foundation
import MCPZimKit

@MainActor
enum ProbeE2ECLI {

    struct Case: Sendable {
        let query: String
        /// Tool we expect the fast-path classifier to pick. `nil` means
        /// "no fast-path match; the case exercises the LLM loop".
        let expectedTool: String?
        /// Substrings (case-insensitive) the final LLM response should
        /// contain. Empty array = no content assertion.
        let mustContain: [String]
    }

    // Default suite — spans compare_articles (countries / cities / people
    // / teams / companies) + article_overview. Extend via --add-case.
    static let defaultCases: [Case] = [
        // Countries — expect the dedicated relations article.
        .init(query: "Compare North Korea and South Korea",
              expectedTool: "compare_articles",
              mustContain: ["korea"]),
        .init(query: "Compare France and Germany",
              expectedTool: "compare_articles",
              mustContain: ["france", "germany"]),
        // Cities — expect the side-by-side path (no relations article).
        .init(query: "Compare Tokyo and Paris",
              expectedTool: "compare_articles",
              mustContain: ["tokyo", "paris"]),
        .init(query: "Compare San Francisco and New York",
              expectedTool: "compare_articles",
              mustContain: ["francisco", "york"]),
        // Famous people.
        .init(query: "Compare Elon Musk and Jeff Bezos",
              expectedTool: "compare_articles",
              mustContain: ["musk", "bezos"]),
        .init(query: "Compare Einstein and Newton",
              expectedTool: "compare_articles",
              mustContain: ["einstein", "newton"]),
        // Football / sports.
        .init(query: "Compare Real Madrid and Barcelona",
              expectedTool: "compare_articles",
              mustContain: ["madrid", "barcelona"]),
        // Tech companies.
        .init(query: "Compare Apple and Microsoft",
              expectedTool: "compare_articles",
              mustContain: ["apple", "microsoft"]),
        // Single-article overview.
        .init(query: "Tell me about Palo Alto",
              expectedTool: "article_overview",
              mustContain: ["palo alto"]),
        .init(query: "What is aspirin",
              expectedTool: "article_overview",
              mustContain: ["aspirin"]),
    ]

    static func run(args inputArgs: [String]) async {
        var zim: String? = nil
        var modelRepo = "mlx-community/Qwen3-4B-4bit"
        var extraCases: [Case] = []
        var args = inputArgs[...]
        while let a = args.first {
            args = args.dropFirst()
            switch a {
            case "--zim":
                zim = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--model":
                if let m = args.first {
                    modelRepo = String(m)
                    args = args.dropFirst()
                }
            case "--add-case":
                // Format: "query|tool|keyword1,keyword2"
                // Example: --add-case "Compare cats and dogs|compare_articles|cats,dogs"
                if let spec = args.first {
                    args = args.dropFirst()
                    let parts = String(spec).split(separator: "|", maxSplits: 2,
                                                   omittingEmptySubsequences: false)
                    guard parts.count >= 1 else { break }
                    let q = String(parts[0])
                    let tool = parts.count > 1 && !parts[1].isEmpty
                        ? String(parts[1]) : nil
                    let keys = parts.count > 2
                        ? String(parts[2]).split(separator: ",").map {
                            $0.trimmingCharacters(in: .whitespaces).lowercased()
                          }
                        : []
                    extraCases.append(.init(
                        query: q, expectedTool: tool, mustContain: keys))
                }
            default:
                FileHandle.standardError.write(Data(
                    "probe-e2e: unknown argument \(a)\n".utf8))
                exit(2)
            }
        }
        guard let zim, !zim.isEmpty else {
            FileHandle.standardError.write(Data(
                "probe-e2e: --zim <path> required\n".utf8))
            exit(2)
        }

        let cases = defaultCases + extraCases
        print("== probe-e2e ==")
        print("zim:   \(zim)")
        print("model: \(modelRepo)")
        print("cases: \(cases.count)\n")

        // Open ZIM + build adapter.
        let url = URL(fileURLWithPath: zim)
        let fileName = url.lastPathComponent
        let reader: ZimReader
        do {
            reader = try LibzimReader(url: url)
        } catch {
            print("ZIM open failed: \(error)"); exit(3)
        }
        let service = DefaultZimService(readers: [(fileName, reader)])
        let adapter = await MCPToolAdapter(
            service: service, hasStreetzim: false
        )

        // Load the model.
        print("loading \(modelRepo)… (first run downloads ~2.5 GB)")
        let t0 = Date()
        let template: any ModelTemplate = modelRepo.lowercased().contains("qwen")
            ? QwenChatMLTemplate()
            : Gemma4Template()
        let provider = Gemma4Provider(
            id: "e2e",
            displayName: "E2E probe",
            huggingFaceRepo: modelRepo,
            template: template
        )
        do {
            try await provider.load()
        } catch {
            print("model load failed: \(error)"); exit(4)
        }
        let loadSeconds = Date().timeIntervalSince(t0)
        print(String(format: "model loaded in %.1fs\n", loadSeconds))

        let session = ChatSession.forTesting(
            providers: [provider], adapter: adapter, initialModelId: "e2e"
        )
        // The session won't touch location, but drop a harmless default
        // so `composeSystemMessage` doesn't print "permission pending".
        session.currentLocation = (lat: 37.441, lon: -122.155)

        // Run each case.
        struct Result {
            let query: String
            let tool: String
            let passed: Bool
            let suggestions: [String]
        }
        var results: [Result] = []

        for (i, c) in cases.enumerated() {
            print("─── [\(i+1)/\(cases.count)] \(c.query)")

            // 1. Intent classification (input quality).
            let intent = IntentRouter.classify(c.query,
                                               currentLocation: session.currentLocation)
            let chosenTool = intent?.toolName ?? "(no fast-path)"
            print("  fast-path tool: \(chosenTool)")
            if let expected = c.expectedTool, expected != chosenTool {
                print("  ⚠️  expected tool '\(expected)', got '\(chosenTool)'")
            }

            // 2. Dispatch just to peek at the raw tool result — lets us
            //    report article quality even when the LLM is slow.
            if let intent = intent {
                do {
                    let raw = try await adapter.dispatch(
                        tool: intent.toolName, args: intent.anyArgs
                    )
                    summariseToolResult(raw, toolName: intent.toolName)
                } catch {
                    print("  dispatch preview threw: \(error)")
                }
            }

            // 3. Run the full turn through ChatSession.
            let runT0 = Date()
            session.send(c.query)
            let deadline = Date().addingTimeInterval(180)
            while session.isGenerating, Date() < deadline {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            let elapsed = Date().timeIntervalSince(runT0)
            if session.isGenerating {
                print("  ⚠️ hit 180s deadline while still generating")
            }

            // 4. Inspect the final assistant message + output-quality check.
            let assistant = (session.messages.last { $0.role == .assistant })
            let text = assistant?.text ?? ""
            print("  response (\(text.count) chars, \(String(format: "%.1fs", elapsed))):")
            print("    " + text.replacingOccurrences(of: "\n", with: "\n    "))

            let missing = c.mustContain.filter {
                !text.lowercased().contains($0.lowercased())
            }
            let passed = missing.isEmpty
                && (c.expectedTool == nil || c.expectedTool == chosenTool)
            print("  \(passed ? "✅" : "❌") \(passed ? "pass" : "fail")")
            if !missing.isEmpty {
                print("     missing keywords: \(missing.joined(separator: ", "))")
            }

            // 5. Tuning suggestions.
            var suggestions: [String] = []
            if text.count < 60 {
                suggestions.append(
                    "response is very short (\(text.count) chars) — "
                    + "consider raising `leadWordCap` in ChatSession.trimForModel "
                    + "so the model has more context to summarise")
            }
            if text.lowercased().contains("i couldn't find")
                || text.lowercased().contains("not found") {
                suggestions.append(
                    "model reports miss — verify article paths via "
                    + "`MCPZimEvalCLI --probe-article --title <X>`")
            }
            if !suggestions.isEmpty {
                print("  suggestions:")
                for s in suggestions { print("     - \(s)") }
            }

            results.append(.init(query: c.query, tool: chosenTool,
                                  passed: passed, suggestions: suggestions))
            session.resetConversation()
            print()
        }

        // Summary.
        let pass = results.filter(\.passed).count
        print("════════════════════════════════════════")
        print("Summary: \(pass)/\(results.count) passed")
        for r in results where !r.passed {
            print("  ❌ \(r.query)  (tool=\(r.tool))")
        }
        exit(pass == results.count ? 0 : 1)
    }

    // MARK: - Helpers

    /// Print a one-screen summary of the tool's RAW output so we can
    /// see which articles / sections the fast path landed on before
    /// the LLM sees them.
    private static func summariseToolResult(
        _ result: [String: Any], toolName: String
    ) {
        if let err = result["error"] as? String {
            print("  raw error: \(err)")
            return
        }
        switch toolName {
        case "compare_articles":
            if let strategy = result["strategy"] as? String {
                print("  strategy: \(strategy)")
                if let t = result["resolved_title"] as? String {
                    print("  relations article: \(t)")
                }
            }
            if let articles = result["articles"] as? [[String: Any]] {
                print("  articles:")
                for a in articles {
                    let t = (a["title"] as? String) ?? "?"
                    let err = (a["error"] as? String) ?? ""
                    let sections = (a["sections"] as? [[String: Any]]) ?? []
                    if !err.isEmpty {
                        print("    - \(t): ERROR \(err)")
                    } else {
                        let lead = (sections.first?["text"] as? String) ?? ""
                        let words = lead.split(separator: " ").count
                        print("    - \(t): \(sections.count) sections, lead \(words) words")
                    }
                }
            }
        case "article_overview":
            let t = (result["title"] as? String) ?? "?"
            let sections = (result["sections"] as? [[String: Any]]) ?? []
            let lead = (sections.first?["text"] as? String) ?? ""
            let words = lead.split(separator: " ").count
            print("  resolved: \(t) (\(sections.count) sections, lead \(words) words)")
        case "what_is_here":
            if let place = result["nearest_named_place"] as? String {
                print("  nearest: \(place)")
            }
        default:
            break
        }
    }
}

// MARK: - Multi-turn "let's discuss X" harness
//
// Drives a whole conversation through the real ChatSession on the Mac —
// real libzim ZIM + the shipping LFM2.5 GGUF loaded from a LOCAL path (no
// download). Discussion state persists across turns (no reset), so this
// exercises the multi-article retrieval + corpus-fallback end to end and
// prints each answer. Lets us iterate on the discuss flow without a phone.
//
//   MCPZimEvalCLI --probe-discuss \
//     --zim ~/Downloads/wikipedia_en_top_nopic_2026-03.zim \
//     [--gguf <path>] [--turn "..." --turn "..."]
@MainActor
enum ProbeDiscussCLI {
    private struct QASuite: Decodable {
        let schemaVersion: Int
        let name: String
        let description: String?
        let conversations: [QAConversation]
    }

    private struct QAConversation: Decodable {
        let id: String
        let description: String?
        let tags: [String]?
        let requiresStreetzim: Bool?
        let turns: [QATurn]
    }

    private struct QATurn: Decodable {
        let user: String
        let expectedTool: String?
        let expectedSourceTitles: [String]?
        let expectedSourceSections: [String]?
        let anchorGroups: [[String]]?
        let mustNotContain: [String]?
        let mustNotSuggest: [String]?
        let expectedClarification: Bool?
        let minimumAnswerSentences: Int?
        let maxSecondsAdvisory: Double?
    }

    private struct RunTurn {
        let user: String
        let expectation: QATurn?
    }

    private struct RunConversation {
        let id: String
        let description: String?
        let turns: [RunTurn]
    }

    private struct PreparationReport: Encodable {
        let strategy: String
        let title: String
        let sectionCount: Int
        let vectorCount: Int
        let vectorBytes: Int
        let elapsedSeconds: Double
    }

    private struct TurnReport: Encodable {
        let conversationID: String
        let turn: Int
        let user: String
        let routedTool: String?
        let elapsedSeconds: Double
        let answer: String
        let groundingTitles: [String]
        let groundingSections: [String]
        let suggestions: [String]
        let passed: Bool?
        let failures: [String]
        let preparation: PreparationReport?
    }

    private struct ProbeReport: Encodable {
        let schemaVersion: Int
        let generatedAt: Date
        let zim: String
        let streetzim: String?
        let gguf: String
        let modelDisplayName: String
        let modelFileMB: Double
        let contextTokens: Int
        let kvCacheType: String
        let deviceProfile: String
        let phoneMode: Bool
        let ramBudgetMB: Double?
        let estimatedPhoneFootprintMB: Double?
        let preparationStrategy: String
        let samplingTemperature: Double?
        let samplingTopP: Double?
        let samplingTopK: Int?
        let samplingSeed: UInt32
        let passedTurns: Int
        let failedTurns: Int
        let advisoryLatencyMisses: Int
        let peakFootprintMB: Double
        let turns: [TurnReport]
    }

    static let defaultTurns = [
        "let's discuss the history of Lithuania",
        "When did they get independence from the soviets?",
        "What is the population?",
        "How have they gotten along with Poland?",
    ]

    static func run(args inputArgs: [String]) async {
        var zim: String?
        var streetzim: String?
        var gguf = "/Users/jasontitus/experiments/mcpzim/tools/fine-tune/"
            + "ft-out-lfm2.5-8b-v7full/imx/lfm2.5-8b-a1b-ft.imx.IQ3_XS.gguf"
        var turns: [String] = []
        var phoneMode = false
        var ramBudgetMB = 0.0
        var lat = 37.441, lon = -122.155   // Palo Alto default
        var suitePath: String?
        var caseFilters: [String] = []
        var listSuite = false
        var samplingTemperature: Double?
        var samplingTopP: Double?
        var samplingTopK: Int?
        var samplingSeed: UInt32 = 42
        var preparationStrategy: ChatSession.DiscussionPreparationStrategy =
            .semanticSections
        var reportJSONPath: String?
        var runtime = "llamacpp"
        // Ternary 2-bit is the stock-runnable MLX pack (affine bits=2).
        // The phone-class 1-bit pack needs PrismML-Eng/mlx-swift branch
        // `prism` — stock mlx-c rejects bits=1 at load.
        var mlxRepo = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        var args = inputArgs[...]
        while let a = args.first {
            args = args.dropFirst()
            switch a {
            case "--zim":
                zim = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--streetzim":
                streetzim = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--lat":
                if let v = args.first.flatMap({ Double($0) }) { lat = v; args = args.dropFirst() }
            case "--lon":
                if let v = args.first.flatMap({ Double($0) }) { lon = v; args = args.dropFirst() }
            case "--gguf":
                if let g = args.first { gguf = String(g); args = args.dropFirst() }
            case "--turn":
                if let t = args.first { turns.append(String(t)); args = args.dropFirst() }
            case "--suite":
                suitePath = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--case":
                if let id = args.first {
                    caseFilters.append(String(id))
                    args = args.dropFirst()
                }
            case "--list-suite":
                listSuite = true
            case "--temperature":
                if let n = args.first.flatMap({ Double($0) }) {
                    samplingTemperature = n; args = args.dropFirst()
                }
            case "--top-p":
                if let n = args.first.flatMap({ Double($0) }) {
                    samplingTopP = n; args = args.dropFirst()
                }
            case "--top-k":
                if let n = args.first.flatMap({ Int($0) }) {
                    samplingTopK = n; args = args.dropFirst()
                }
            case "--seed":
                if let n = args.first.flatMap({ UInt32($0) }) {
                    samplingSeed = n; args = args.dropFirst()
                }
            case "--prep-mode":
                guard let raw = args.first,
                      let parsed = ChatSession.DiscussionPreparationStrategy(
                        rawValue: String(raw))
                else {
                    FileHandle.standardError.write(Data(
                        "probe-discuss: --prep-mode must be none or semantic-sections\n".utf8))
                    exit(2)
                }
                preparationStrategy = parsed
                args = args.dropFirst()
            case "--report-json":
                reportJSONPath = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--phone-mode":
                phoneMode = true
            case "--ram-mb":
                if let n = args.first.flatMap({ Double($0) }) {
                    ramBudgetMB = n; args = args.dropFirst()
                }
            case "--runtime":
                // llamacpp (default) drives the --gguf path; mlx loads the
                // official MLX pack via Gemma4Provider (our generic MLX
                // provider) for the cross-runtime Bonsai A/B. Run the
                // harness once per runtime — two clean processes measure
                // more honestly than one process hosting two 27B models.
                if let r = args.first { runtime = String(r); args = args.dropFirst() }
            case "--mlx-repo":
                if let r = args.first { mlxRepo = String(r); args = args.dropFirst() }
            default:
                FileHandle.standardError.write(Data("probe-discuss: unknown arg \(a)\n".utf8))
                exit(2)
            }
        }
        guard runtime == "llamacpp" || runtime == "mlx" else {
            FileHandle.standardError.write(Data(
                "probe-discuss: --runtime must be llamacpp or mlx\n".utf8))
            exit(2)
        }

        let suite: QASuite?
        if let suitePath {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: suitePath))
                let decoder = JSONDecoder()
                decoder.keyDecodingStrategy = .convertFromSnakeCase
                suite = try decoder.decode(QASuite.self, from: data)
            } catch {
                FileHandle.standardError.write(Data(
                    "probe-discuss: could not decode suite \(suitePath): \(error)\n".utf8))
                exit(2)
            }
        } else {
            suite = nil
        }

        if listSuite {
            guard let suite else {
                FileHandle.standardError.write(Data(
                    "probe-discuss: --list-suite requires --suite <path>\n".utf8))
                exit(2)
            }
            print("== \(suite.name) (schema \(suite.schemaVersion)) ==")
            if let description = suite.description { print(description) }
            print()
            for conversation in suite.conversations {
                let street = conversation.requiresStreetzim == true ? " · StreetZIM" : ""
                let tags = (conversation.tags ?? []).joined(separator: ", ")
                print("\(conversation.id) · \(conversation.turns.count) turn(s)\(street)")
                if !tags.isEmpty { print("  tags: \(tags)") }
                if let description = conversation.description {
                    print("  \(description)")
                }
            }
            exit(0)
        }

        guard let zim, !zim.isEmpty else {
            FileHandle.standardError.write(Data("probe-discuss: --zim <path> required\n".utf8))
            exit(2)
        }
        // Phone mode: use an iPhone's device profile (article/reply budgets)
        // and track phys_footprint against a RAM ceiling, so the Mac harness
        // reflects what the phone would do before jetsam. Default ceiling =
        // 6144 MB (the iPhone 17 Pro Max process cap with increased-memory).
        if phoneMode {
            DeviceProfile.override = .balanced
            if ramBudgetMB == 0 { ramBudgetMB = 6144 }
        }

        let runConversations: [RunConversation]
        var skippedStreetCases: [String] = []
        if let suite {
            let selected: [QAConversation]
            if caseFilters.isEmpty {
                selected = suite.conversations
            } else {
                let known = Set(suite.conversations.map(\.id))
                let unknown = caseFilters.filter { !known.contains($0) }
                if !unknown.isEmpty {
                    FileHandle.standardError.write(Data(
                        "probe-discuss: unknown suite case(s): \(unknown.joined(separator: ", "))\n".utf8))
                    exit(2)
                }
                let wanted = Set(caseFilters)
                selected = suite.conversations.filter { wanted.contains($0.id) }
            }
            let runnable = selected.filter { conversation in
                if conversation.requiresStreetzim == true && streetzim == nil {
                    skippedStreetCases.append(conversation.id)
                    return false
                }
                return true
            }
            runConversations = runnable.map { conversation in
                RunConversation(
                    id: conversation.id,
                    description: conversation.description,
                    turns: conversation.turns.map {
                        RunTurn(user: $0.user, expectation: $0)
                    })
            }
        } else {
            let scenario = turns.isEmpty ? defaultTurns : turns
            runConversations = [RunConversation(
                id: "ad_hoc",
                description: nil,
                turns: scenario.map { RunTurn(user: $0, expectation: nil) })]
        }
        guard !runConversations.isEmpty else {
            print("No runnable conversations. Supply --streetzim for: "
                + skippedStreetCases.joined(separator: ", "))
            exit(0)
        }

        let totalTurns = runConversations.reduce(0) { $0 + $1.turns.count }
        print("== probe-discuss ==\nzim:   \(zim)\nstreetzim: \(streetzim ?? "(none)")\ngguf:  \(gguf)\nprep:  \(preparationStrategy.rawValue)\nconversations: \(runConversations.count)\nturns: \(totalTurns)")
        if !skippedStreetCases.isEmpty {
            print("skipped (no StreetZIM): \(skippedStreetCases.joined(separator: ", "))")
        }
        print("profile: \(DeviceProfile.current.label)"
            + (ramBudgetMB > 0 ? " · RAM budget \(Int(ramBudgetMB)) MB" : "") + "\n")

        let url = URL(fileURLWithPath: zim)
        var readers: [(String, ZimReader)] = []
        var openedLibraries: [(url: URL, reader: ZimReader)] = []
        do {
            let reader = try LibzimReader(url: url)
            readers.append((url.lastPathComponent, reader))
            openedLibraries.append((url, reader))
        }
        catch { print("ZIM open failed: \(error)"); exit(3) }
        if let streetzim, !streetzim.isEmpty {
            let surl = URL(fileURLWithPath: streetzim)
            do {
                let reader = try LibzimReader(url: surl)
                readers.append((surl.lastPathComponent, reader))
                openedLibraries.append((surl, reader))
            }
            catch { print("streetzim open failed: \(error)"); exit(3) }
        }
        let service = DefaultZimService(readers: readers)
        let adapter = await MCPToolAdapter(
            service: service, hasStreetzim: streetzim != nil)
        // GPS host state so what_is_here / distance_to / near-me flows work
        // exactly like on the phone (ZimfoContext there, a fixed fix here).
        let fixLat = lat, fixLon = lon
        await adapter.installHostStateProvider {
            HostStateSnapshot(
                activeRoute: nil,
                currentLocation: LocationSnapshot(lat: fixLat, lon: fixLon))
        }
        // Wire the SAME semantic reranker the iOS app installs, so the
        // CLI's `search` (used by discuss drift) reorders BM25 hits by
        // NLContextualEmbedding — otherwise the harness is a pessimistic
        // lower bound vs the phone (e.g. it pulled "Oxford Photovoltaics"
        // instead of "Perovskite solar cell" for "how about perovskites?").
        SemanticReranker.log = { print("    [Rerank] \($0)") }
        await adapter.installHitReranker { query, hits in
            await SemanticReranker.shared.rerank(query: query, hits: hits)
        }

        let lowerGGUF = gguf.lowercased()
        // --runtime mlx is Bonsai-only today, so the MLX path inherits the
        // Bonsai sampling recipe even when --gguf points elsewhere.
        let isBonsai = lowerGGUF.contains("bonsai") || runtime == "mlx"
        let isTernaryBonsai = isBonsai
            && (lowerGGUF.contains("ternary") || lowerGGUF.contains("q2_0"))
        let isLFM = lowerGGUF.contains("lfm")
        let template: any ModelTemplate
        if isBonsai || lowerGGUF.contains("qwen") {
            template = QwenChatMLTemplate()
        } else if isLFM {
            template = LFM25Template()
        } else {
            template = Gemma3Template()
        }
        let modelID = isTernaryBonsai
            ? ChatSession.ternaryBonsai27BModelID
            : (isBonsai ? "bonsai-27b-q1-gguf" : "local-discuss-model")
        let explicitSampling = samplingTemperature != nil
            || samplingTopP != nil || samplingTopK != nil
        let samplingProfile: GenerationSamplingProfile? = {
            if isBonsai || explicitSampling {
                return GenerationSamplingProfile(
                    temperature: samplingTemperature ?? (isBonsai ? 1.0 : 0.3),
                    topP: samplingTopP ?? (isBonsai ? 0.95 : 0.9),
                    topK: samplingTopK ?? (isBonsai ? 20 : 40))
            }
            return nil
        }()
        let provider: any ModelProvider
        let sessionModelID: String
        // Report fields that only the llama.cpp runtime pins exactly.
        // MLX has no fixed n_ctx (cache grows with the prompt) and no
        // local file path pre-download, so these stay 0/"n/a" there.
        var modelFileMB: Double = 0
        var reportContextTokens = 0
        var reportKVType = "n/a"
        if runtime == "mlx" {
            // The MLX side of the Bonsai A/B: same weights (Prism's official
            // 1-bit pack, standard MLX affine quant bits=1/group=128), same
            // ChatML template, same sampling profile — only the runtime
            // differs. Weights download via HubClient on first load
            // (~3.8 GB) into the shared HF cache.
            sessionModelID = "bonsai-27b-q1-mlx"
            provider = Gemma4Provider(
                id: sessionModelID,
                displayName: "Bonsai 27B (1-bit · MLX)",
                huggingFaceRepo: mlxRepo,
                approximateMemoryMB: 5900,
                template: QwenChatMLTemplate(),
                replyTokensFloor: 512,
                samplingProfile: samplingProfile)
            reportKVType = DeviceProfile.current.useQuantizedKVCache
                ? "mlx-q4" : "mlx-fp16"
            print("runtime: \(provider.displayName)"
                + " · repo \(mlxRepo)"
                + " · profile=\(DeviceProfile.current.label)"
                + " · phone-mode=\(phoneMode ? "on" : "off")")
        } else {
            sessionModelID = modelID
            let llamaProvider = LlamaCppProvider(
            id: modelID,
            displayName: isTernaryBonsai
                ? "Bonsai 27B Ternary (2-bit · Metal · Mac)"
                : (isBonsai
                ? "Bonsai 27B (1-bit · Metal)"
                : "Local conversation model (\((gguf as NSString).lastPathComponent))"),
            huggingFaceRepo: isBonsai
                ? (isTernaryBonsai
                    ? "prism-ml/Ternary-Bonsai-27B-gguf"
                    : "prism-ml/Bonsai-27B-gguf")
                : "sliderforthewin/lfm2.5-8b-a1b-ft-GGUF",
            ggufFilename: (gguf as NSString).lastPathComponent,
            expectedGGUFBytes: isTernaryBonsai
                ? 7_165_121_600 : (isBonsai ? 3_803_452_480 : nil),
            localGGUFPath: gguf,
            replyTokensFloor: isTernaryBonsai
                ? 1024 : (isBonsai ? 512 : 1024),
            approximateMemoryMB: isTernaryBonsai
                ? 9000 : (isBonsai ? 5500 : 4200),
            // Match the SHIPPING config (ChatSession registers the LFM FT
            // with a 32k window) — at the 8192 default the harness
            // overflowed n_ctx on turn 6 of a discuss run and reported a
            // failure the phone wouldn't have.
            contextTokens: isBonsai && !isTernaryBonsai ? 16384 : 32768,
            kvCacheType: isBonsai ? .q4_0 : .q8_0,
            samplingProfile: samplingProfile,
            samplingSeed: samplingSeed,
            template: template)
            provider = llamaProvider
            modelFileMB = {
                guard let attrs = try? FileManager.default.attributesOfItem(
                    atPath: gguf),
                      let bytes = attrs[.size] as? NSNumber
                else { return 0 }
                return bytes.doubleValue / 1_048_576
            }()
            reportContextTokens = llamaProvider.contextTokens
            reportKVType = llamaProvider.kvCacheType.rawValue
            print("runtime: \(provider.displayName)"
                + " · model \(String(format: "%.1f", modelFileMB)) MiB"
                + " · n_ctx=\(llamaProvider.contextTokens)"
                + " · kv=\(llamaProvider.kvCacheType.rawValue)"
                + " · profile=\(DeviceProfile.current.label)"
                + " · phone-mode=\(phoneMode ? "on" : "off")")
        }
        if let samplingProfile {
            print(String(format: "sampling: temp %.2f · top-p %.2f · top-k %d · seed %u",
                         samplingProfile.temperature,
                         samplingProfile.topP,
                         samplingProfile.topK,
                         samplingSeed))
        } else {
            print("sampling: task defaults")
        }
        print("loading model from local path…")
        let t0 = Date()
        do { try await provider.load() }
        catch { print("model load failed: \(error)"); exit(4) }
        print(String(format: "model loaded in %.1fs\n", Date().timeIntervalSince(t0)))

        let session = ChatSession.forTesting(
            providers: [provider], adapter: adapter, initialModelId: sessionModelID,
            discussionPreparationStrategy: preparationStrategy)
        // Match the app's two views of its open libraries. The adapter owns
        // the readers used by tools, while ChatSession.library carries source
        // authority and enablement. Explicit source directives intentionally
        // consult the latter so they cannot drift from Wikipedia to StreetZIM.
        session.library = openedLibraries.map {
            ChatSession.LibraryEntry(url: $0.url, reader: $0.reader)
        }
        session.currentLocation = (lat: fixLat, lon: fixLon)

        // Sample phys_footprint (the jetsam metric) across the run.
        let peak = PeakMem()
        let sampler = Task.detached {
            while !Task.isCancelled {
                await peak.sample()
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }

        var passedTurns = 0
        var failedTurns = 0
        var advisoryLatencyMisses = 0
        var turnReports: [TurnReport] = []
        for (conversationIndex, conversation) in runConversations.enumerated() {
            session.resetConversation()
            var reportedPreparationKey: String?
            print("╔══ [\(conversationIndex + 1)/\(runConversations.count)] \(conversation.id)")
            if let description = conversation.description { print("║ \(description)") }
            for (turnIndex, turn) in conversation.turns.enumerated() {
                print("─── [\(turnIndex + 1)/\(conversation.turns.count)] YOU: \(turn.user)")
                let directIntent = IntentRouter.classify(
                    turn.user,
                    currentLocation: session.currentLocation,
                    focus: session.focus)
                if let expectedTool = turn.expectation?.expectedTool {
                    print("    ROUTE expected=\(expectedTool) actual=\(directIntent?.toolName ?? "(none)")")
                }

                let rt = Date()
                session.send(turn.user)
                let deadline = Date().addingTimeInterval(180)
                while session.isGenerating, Date() < deadline {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                let elapsed = Date().timeIntervalSince(rt)
                let assistantMessage = session.messages.last {
                    $0.role == .assistant
                }
                let text = assistantMessage?.text ?? "(no reply)"
                let grounding = assistantMessage?.groundingSources ?? []
                let suggestions = assistantMessage?.suggestions.map(\.label) ?? []
                let groundingTitles = Array(Set(grounding.map(\.title))).sorted()
                // Preserve lead provenance in reports/assertions. A nil
                // section means the answer used the article lead; dropping it
                // made correct lead-grounded answers look ungrounded.
                let groundingSections = Array(Set(grounding.map {
                    ($0.section?.isEmpty == false) ? $0.section! : "lead"
                })).sorted()
                print("    BOT (\(String(format: "%.1fs", elapsed)), \(text.count) chars):")
                print("    " + text.replacingOccurrences(of: "\n", with: "\n    "))
                if !groundingTitles.isEmpty {
                    print("    SOURCES actual: " + groundingTitles.joined(separator: " | "))
                }
                if !groundingSections.isEmpty {
                    print("    SECTIONS actual: " + groundingSections.joined(separator: " | "))
                }
                if !suggestions.isEmpty {
                    print("    SUGGESTIONS actual: " + suggestions.joined(separator: " | "))
                }

                var preparationReport: PreparationReport?
                if let stats = session.lastDiscussionPreparationStats {
                    let key = "\(stats.title)|\(stats.elapsedSeconds)"
                    if key != reportedPreparationKey {
                        reportedPreparationKey = key
                        preparationReport = PreparationReport(
                            strategy: stats.strategy.rawValue,
                            title: stats.title,
                            sectionCount: stats.sectionCount,
                            vectorCount: stats.vectorCount,
                            vectorBytes: stats.vectorBytes,
                            elapsedSeconds: stats.elapsedSeconds)
                        print(String(format:
                            "    PREP %@ · %@ · %d sections · %d vectors · %.2f MB · %.3fs",
                            stats.title, stats.strategy.rawValue,
                            stats.sectionCount, stats.vectorCount,
                            Double(stats.vectorBytes) / (1024 * 1024),
                            stats.elapsedSeconds))
                    }
                }

                var failures: [String] = []
                var passed: Bool?
                if let expectation = turn.expectation {
                    let lower = text.lowercased()
                    if let expectedTool = expectation.expectedTool,
                       directIntent?.toolName != expectedTool {
                        failures.append(
                            "route expected \(expectedTool), got \(directIntent?.toolName ?? "none")")
                    }
                    for group in expectation.anchorGroups ?? [] {
                        if !group.contains(where: { lower.contains($0.lowercased()) }) {
                            failures.append("missing one of [\(group.joined(separator: " | "))]")
                        }
                    }
                    for forbidden in expectation.mustNotContain ?? []
                    where lower.contains(forbidden.lowercased()) {
                        failures.append("contained forbidden phrase [\(forbidden)]")
                    }
                    for forbidden in expectation.mustNotSuggest ?? []
                    where suggestions.contains(where: {
                        $0.localizedCaseInsensitiveContains(forbidden)
                    }) {
                        failures.append("suggested forbidden phrase [\(forbidden)]")
                    }
                    if expectation.expectedClarification == true,
                       !lower.contains("which") && !lower.contains("did you mean") {
                        failures.append("expected a clarification question")
                    }
                    if let minimum = expectation.minimumAnswerSentences {
                        let sentenceCount = Self.answerSentenceCount(text)
                        if sentenceCount < minimum {
                            failures.append(
                                "expected at least \(minimum) sentence(s), got \(sentenceCount)")
                        }
                    }
                    if let sources = expectation.expectedSourceTitles, !sources.isEmpty {
                        print("    GROUNDING target: \(sources.joined(separator: " | "))")
                        for expected in sources where !groundingTitles.contains(where: {
                            $0.localizedCaseInsensitiveContains(expected)
                                || expected.localizedCaseInsensitiveContains($0)
                        }) {
                            failures.append("missing grounded source [\(expected)]")
                        }
                    }
                    if let sections = expectation.expectedSourceSections, !sections.isEmpty {
                        print("    SECTIONS target: \(sections.joined(separator: " | "))")
                        let foundExpectedSection = sections.contains { expected in
                            groundingSections.contains {
                                $0.localizedCaseInsensitiveContains(expected)
                                    || expected.localizedCaseInsensitiveContains($0)
                            }
                        }
                        if !foundExpectedSection {
                            failures.append(
                                "missing any grounded section [\(sections.joined(separator: " | "))]")
                        }
                    }
                    if let maxSeconds = expectation.maxSecondsAdvisory,
                       elapsed > maxSeconds {
                        advisoryLatencyMisses += 1
                        print(String(format: "    ⚠ latency %.1fs > %.1fs advisory", elapsed, maxSeconds))
                    }
                    if session.isGenerating {
                        failures.append("hit the 180-second hard timeout")
                    }
                    if failures.isEmpty {
                        passedTurns += 1
                        passed = true
                        print("    ✅ content pass")
                    } else {
                        failedTurns += 1
                        passed = false
                        print("    ❌ content fail")
                        for failure in failures { print("       - \(failure)") }
                    }
                }
                turnReports.append(TurnReport(
                    conversationID: conversation.id,
                    turn: turnIndex + 1,
                    user: turn.user,
                    routedTool: directIntent?.toolName,
                    elapsedSeconds: elapsed,
                    answer: text,
                    groundingTitles: groundingTitles,
                    groundingSections: groundingSections,
                    suggestions: suggestions,
                    passed: passed,
                    failures: failures,
                    preparation: preparationReport))
                print()
            }
        }

        sampler.cancel()
        let peakMB = await peak.peakMB
        // Free the llama context BEFORE exit(0) — leaving it loaded trips
        // GGML_ASSERT([rsets->data count] == 0) in the Metal device's static
        // destructor at __cxa_finalize, exiting 134 after a successful run.
        await provider.unload()
        // On macOS the GGUF is mmap'd + Metal-unified, so phys_footprint
        // under-counts much of the model's resident cost; iOS jetsam would
        // see the weights too. Add the actual GGUF byte size for a useful
        // conservative phone estimate. `approximateMemoryMB` is a picker UI
        // peak estimate (weights + KV), not a weight-size figure, and adding
        // it here used to double-count runtime overhead.
        let phoneEst = peakMB + modelFileMB
        print(String(format: "── peak footprint: %.0f MB (mmap under-counts the model on macOS)", peakMB))
        if ramBudgetMB > 0 {
            print(String(format: "   phone-equivalent ≈ %.0f MB (footprint + %.0f MiB GGUF)",
                         phoneEst, modelFileMB))
            let over = phoneEst - ramBudgetMB
            print(over > 0
                ? String(format: "   ⚠️ OVER %.0f MB budget by %.0f MB — phone would jetsam", ramBudgetMB, over)
                : String(format: "   ✓ within %.0f MB budget (%.0f MB headroom)", ramBudgetMB, -over))
        }
        if suite != nil {
            print("── suite summary: \(passedTurns)/\(passedTurns + failedTurns) content turns passed"
                + " · \(advisoryLatencyMisses) latency advisory miss(es)")
        }
        if let reportJSONPath {
            let report = ProbeReport(
                schemaVersion: 2,
                generatedAt: Date(),
                zim: zim,
                streetzim: streetzim,
                gguf: gguf,
                modelDisplayName: provider.displayName,
                modelFileMB: modelFileMB,
                contextTokens: reportContextTokens,
                kvCacheType: reportKVType,
                deviceProfile: DeviceProfile.current.label,
                phoneMode: phoneMode,
                ramBudgetMB: ramBudgetMB > 0 ? ramBudgetMB : nil,
                estimatedPhoneFootprintMB: ramBudgetMB > 0 ? phoneEst : nil,
                preparationStrategy: preparationStrategy.rawValue,
                samplingTemperature: samplingProfile?.temperature,
                samplingTopP: samplingProfile?.topP,
                samplingTopK: samplingProfile?.topK,
                samplingSeed: samplingSeed,
                passedTurns: passedTurns,
                failedTurns: failedTurns,
                advisoryLatencyMisses: advisoryLatencyMisses,
                peakFootprintMB: peakMB,
                turns: turnReports)
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [
                    .prettyPrinted, .sortedKeys, .withoutEscapingSlashes,
                ]
                encoder.dateEncodingStrategy = .iso8601
                let data = try encoder.encode(report)
                try data.write(
                    to: URL(fileURLWithPath: reportJSONPath),
                    options: .atomic)
                print("── JSON report: \(reportJSONPath)")
            } catch {
                FileHandle.standardError.write(Data(
                    "probe-discuss: could not write JSON report: \(error)\n".utf8))
                exit(5)
            }
        }
        exit(failedTurns == 0 ? 0 : 1)
    }

    /// Approximate prose sentence count for conversational-depth assertions.
    /// The suite uses this only as a lower bound, so a lightweight terminal-
    /// punctuation count is preferable to coupling the evaluator to NLP APIs.
    private static func answerSentenceCount(_ text: String) -> Int {
        let matches = text.matches(of: /[.!?]+(?:\s|$)/)
        if !matches.isEmpty { return matches.count }
        return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0 : 1
    }
}

/// Tracks the peak `phys_footprint` seen during a CLI run.
actor PeakMem {
    private(set) var peakMB = 0.0
    func sample() { peakMB = max(peakMB, MemoryStats.physFootprintMB()) }
}
