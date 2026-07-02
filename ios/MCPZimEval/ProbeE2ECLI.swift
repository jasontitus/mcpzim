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
    static let defaultTurns = [
        "let's discuss the history of Lithuania",
        "When did they get independence from the soviets?",
        "What is the population?",
        "How have they gotten along with Poland?",
    ]

    static func run(args inputArgs: [String]) async {
        var zim: String?
        var gguf = "/Users/jasontitus/experiments/mcpzim/tools/fine-tune/"
            + "ft-out-lfm2.5-8b-v7full/lfm2.5-8b-a1b-ft.Q3_K_M.gguf"
        var turns: [String] = []
        var phoneMode = false
        var ramBudgetMB = 0.0
        var args = inputArgs[...]
        while let a = args.first {
            args = args.dropFirst()
            switch a {
            case "--zim":
                zim = args.first.map { String($0) }
                if !args.isEmpty { args = args.dropFirst() }
            case "--gguf":
                if let g = args.first { gguf = String(g); args = args.dropFirst() }
            case "--turn":
                if let t = args.first { turns.append(String(t)); args = args.dropFirst() }
            case "--phone-mode":
                phoneMode = true
            case "--ram-mb":
                if let n = args.first.flatMap({ Double($0) }) {
                    ramBudgetMB = n; args = args.dropFirst()
                }
            default:
                FileHandle.standardError.write(Data("probe-discuss: unknown arg \(a)\n".utf8))
                exit(2)
            }
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
        let scenario = turns.isEmpty ? defaultTurns : turns
        print("== probe-discuss ==\nzim:   \(zim)\ngguf:  \(gguf)\nturns: \(scenario.count)")
        print("profile: \(DeviceProfile.current.label)"
            + (ramBudgetMB > 0 ? " · RAM budget \(Int(ramBudgetMB)) MB" : "") + "\n")

        let url = URL(fileURLWithPath: zim)
        let reader: ZimReader
        do { reader = try LibzimReader(url: url) }
        catch { print("ZIM open failed: \(error)"); exit(3) }
        let service = DefaultZimService(readers: [(url.lastPathComponent, reader)])
        let adapter = await MCPToolAdapter(service: service, hasStreetzim: false)
        // Wire the SAME semantic reranker the iOS app installs, so the
        // CLI's `search` (used by discuss drift) reorders BM25 hits by
        // NLContextualEmbedding — otherwise the harness is a pessimistic
        // lower bound vs the phone (e.g. it pulled "Oxford Photovoltaics"
        // instead of "Perovskite solar cell" for "how about perovskites?").
        SemanticReranker.log = { print("    [Rerank] \($0)") }
        await adapter.installHitReranker { query, hits in
            await SemanticReranker.shared.rerank(query: query, hits: hits)
        }

        let provider = LlamaCppProvider(
            id: "lfm25-ft",
            displayName: "LFM2.5 8B-A1B FT (Q3_K_M)",
            huggingFaceRepo: "sliderforthewin/lfm2.5-8b-a1b-ft-GGUF",
            ggufFilename: "lfm2.5-8b-a1b-ft.Q3_K_M.gguf",
            localGGUFPath: gguf,
            replyTokensFloor: 1024,
            approximateMemoryMB: 4200,
            // Match the SHIPPING config (ChatSession registers the LFM FT
            // with a 32k window) — at the 8192 default the harness
            // overflowed n_ctx on turn 6 of a discuss run and reported a
            // failure the phone wouldn't have.
            contextTokens: 32768,
            template: LFM25Template())
        print("loading model from local path…")
        let t0 = Date()
        do { try await provider.load() }
        catch { print("model load failed: \(error)"); exit(4) }
        print(String(format: "model loaded in %.1fs\n", Date().timeIntervalSince(t0)))

        let session = ChatSession.forTesting(
            providers: [provider], adapter: adapter, initialModelId: "lfm25-ft")
        session.currentLocation = (lat: 37.441, lon: -122.155)

        // Sample phys_footprint (the jetsam metric) across the run.
        let peak = PeakMem()
        let sampler = Task.detached {
            while !Task.isCancelled {
                await peak.sample()
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }

        for (i, turn) in scenario.enumerated() {
            print("─── [\(i + 1)/\(scenario.count)] YOU: \(turn)")
            let rt = Date()
            session.send(turn)
            let deadline = Date().addingTimeInterval(180)
            while session.isGenerating, Date() < deadline {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            let text = (session.messages.last { $0.role == .assistant })?.text ?? "(no reply)"
            print("    BOT (\(String(format: "%.1fs", Date().timeIntervalSince(rt))), \(text.count) chars):")
            print("    " + text.replacingOccurrences(of: "\n", with: "\n    "))
            print()
        }

        sampler.cancel()
        let peakMB = await peak.peakMB
        // Free the llama context BEFORE exit(0) — leaving it loaded trips
        // GGML_ASSERT([rsets->data count] == 0) in the Metal device's static
        // destructor at __cxa_finalize, exiting 134 after a successful run.
        await provider.unload()
        // On macOS the GGUF is mmap'd + Metal-unified, so phys_footprint
        // under-counts the model's resident cost; iOS jetsam would see the
        // weights too. Add the model size for a phone-faithful estimate.
        let modelMB = Double(provider.approximateMemoryMB)
        let phoneEst = peakMB + modelMB
        print(String(format: "── peak footprint: %.0f MB (mmap under-counts the model on macOS)", peakMB))
        if ramBudgetMB > 0 {
            print(String(format: "   phone-equivalent ≈ %.0f MB (footprint + ~%.0f MB model weights)",
                         phoneEst, modelMB))
            let over = phoneEst - ramBudgetMB
            print(over > 0
                ? String(format: "   ⚠️ OVER %.0f MB budget by %.0f MB — phone would jetsam", ramBudgetMB, over)
                : String(format: "   ✓ within %.0f MB budget (%.0f MB headroom)", ramBudgetMB, -over))
        }
        exit(0)
    }
}

/// Tracks the peak `phys_footprint` seen during a CLI run.
actor PeakMem {
    private(set) var peakMB = 0.0
    func sample() { peakMB = max(peakMB, MemoryStats.physFootprintMB()) }
}
