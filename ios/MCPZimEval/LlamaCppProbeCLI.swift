// SPDX-License-Identifier: MIT
//
// LlamaCppProbeCLI — exercise LlamaCppProvider's Swift code path end-to-
// end from the command line so we can capture llama.cpp's stderr (which
// iOS OSLog swallows). Reproduces the iOS phone crash scenario without
// needing device install / UI interaction:
//   bars in north beach  → near_named_place fast-path + big tool result
//   which is closest to the ferry building → LLM prefill of the above
//
// Usage:
//   MCPZimEvalCLI --probe-llama [--short]
//
// The LlamaCppProvider loads the same GGUF with the same context
// params as the phone (fa=ENABLED, type_k=type_v=Q8_0, swa_full=false,
// offload_kqv=true). Any crash that happens here = the crash happens
// on Mac Metal too, which means it's NOT iOS-specific. Any clean run
// here confirms iOS Metal is the differentiator.

import Foundation
import MCPZimKit

enum LlamaCppProbeCLI {
    static func run(args: [String]) async {
        let short = args.contains("--short")
        let cacheExperiment = args.contains("--cache-experiment")
        // `--gguf PATH` points the provider at an arbitrary local GGUF (e.g.
        // the Gemma 4 E4B QAT) so we can smoke-test whether the SHIPPING
        // llama.cpp xcframework loads + decodes it — no HF download involved.
        var localGGUF: String? = nil
        if let i = args.firstIndex(of: "--gguf"), i + 1 < args.count {
            localGGUF = args[i + 1]
        }
        let isBonsai = localGGUF?
            .lowercased().contains("bonsai-27b-q1_0") ?? false
        let template: any ModelTemplate = isBonsai
            ? QwenChatMLTemplate() : Gemma3Template()

        print("[probe] creating LlamaCppProvider…")
        let provider = LlamaCppProvider(
            id: localGGUF != nil ? "local-gguf-probe" : "gemma3-4b-it-q4km-gguf",
            displayName: localGGUF.map { "Local GGUF (\(($0 as NSString).lastPathComponent))" }
                ?? "Gemma 3 4B IT (Q4_K_M · llama.cpp)",
            huggingFaceRepo: "bartowski/google_gemma-3-4b-it-GGUF",
            ggufFilename: "google_gemma-3-4b-it-Q4_K_M.gguf",
            expectedGGUFBytes: isBonsai ? 3_803_452_480 : nil,
            localGGUFPath: localGGUF,
            approximateMemoryMB: isBonsai ? 5200 : 3200,
            contextTokens: isBonsai ? 16384 : 8192,
            kvCacheType: isBonsai ? .q4_0 : .q8_0,
            template: template
        )
        if cacheExperiment {
            provider.debugSink = { print("[llama] \($0)") }
        }

        print("[probe] provider.load()…")
        do {
            try await provider.load()
        } catch {
            print("[probe] LOAD FAILED: \(error)")
            exit(1)
        }
        print("[probe] load OK")

        if cacheExperiment {
            await runCacheExperiment(provider: provider, template: template)
            await provider.unload()
            return
        }

        // Build a prompt in the same shape ChatSession would — multi-
        // turn transcript with a bars-in-north-beach tool response
        // followed by a follow-up user turn. Use Gemma3Template so
        // the markup matches what the iOS app generates.
        let systemPreamble =
            "You are a helpful offline assistant with access to a " +
            "Wikipedia + OpenStreetMap index. When the user asks about " +
            "places, call the relevant tool. Keep replies concise."

        // Synthesize a 25-bar tool response (matching phone's default
        // limit=25) with 1244 total-in-radius. ~1500 chars of JSON.
        let barsJson: String = {
            var rows: [String] = []
            let names = [
                "Vesuvio Cafe", "Specs'", "Tosca Cafe", "Mr Bing's",
                "15 Romolo", "Tony's Pizza Napoletana Bar",
                "Comstock Saloon", "Saloon", "Columbus Cafe",
                "Church Key", "Rose Pistola Bar", "Gold Dust Lounge",
                "Redwood Room", "The View Lounge", "Harry Denton's",
                "Tonga Room", "Top of the Mark", "Clock Bar",
                "Bourbon & Branch", "Rickhouse", "Swig", "Trick Dog",
                "Smuggler's Cove", "Pagan Idol", "Zombie Village",
            ]
            for (i, name) in names.enumerated() {
                let lat = 37.805 + Double(i % 5) * 0.001 - 0.002
                let lon = -122.410 + Double(i / 5) * 0.002 - 0.006
                rows.append(
                    "{\"name\":\"\(name)\",\"type\":\"poi\"," +
                    "\"subtype\":\"bar\",\"location\":\"North Beach\"," +
                    "\"lat\":\(lat),\"lon\":\(lon)," +
                    "\"distance_m\":\(200 + i * 60)}"
                )
            }
            return "[" + rows.joined(separator: ",") + "]"
        }()

        let toolResult: String =
            "{\"radius_km\":5,\"total_in_radius\":1244," +
            "\"by_category\":[{\"category\":\"bar\",\"count\":1244}]," +
            "\"results_shown\":25,\"results\":\(barsJson)," +
            "\"query\":\"north beach\"," +
            "\"resolved\":{\"name\":\"North Beach\",\"type\":\"place\"}}"

        // Assemble ChatTurns the way ChatSession does.
        let turns: [ChatTurn] = [
            .init(role: .user, text: "bars in north beach"),
            .init(
                role: .assistant,
                text: "```tool_call\n{\"function\":\"near_named_place\",\"parameters\":{\"place\":\"north beach\",\"kinds\":[\"bar\"]}}\n```"
            ),
            .init(
                role: .user,
                text: "[TOOL_RESPONSE name=near_named_place]\n\(toolResult)"
            ),
            .init(
                role: .assistant,
                text: "Found 1244 bars near north beach."
            ),
            .init(
                role: .user,
                text: "which is closest to the ferry building"
            ),
        ]

        let prompt = template.renderTranscript(
            systemPreamble: systemPreamble, tools: [], turns: turns)
        print("[probe] prompt: \(prompt.count) chars")
        if short {
            // Only prefill, don't decode — lets us test whether the
            // crash is in prefill specifically.
            print("[probe] --short: not running generate; exiting after load")
            await provider.unload()
            print("[probe] OK (load-only path didn't crash)")
            return
        }

        let params = GenerationParameters(
            maxTokens: 64, temperature: 0.3, topP: 0.9)
        print("[probe] generate(…)…")
        var chunks = 0
        var bytes = 0
        var firstText = ""
        do {
            for try await chunk in provider.generate(
                prompt: prompt, parameters: params)
            {
                chunks += 1
                bytes += chunk.utf8.count
                if firstText.count < 200 { firstText += chunk }
            }
        } catch {
            print("[probe] GENERATE FAILED: \(error)")
            exit(2)
        }
        print("[probe] OK — chunks=\(chunks) bytes=\(bytes)")
        print("[probe] first chunk sample: \(firstText.prefix(200))")
        await provider.unload()
    }

    /// Compare the two prompt shapes that matter for Bonsai's hybrid cache:
    /// an exact transcript append (recurrent + KV state is reusable) and a
    /// newly assembled standalone follow-up (diverges near the start and
    /// forces a full prefill). Set MCPZIM_BENCH_STATE_CACHE=1 alongside this
    /// flag to also time llama.cpp's explicit SSD session serialization.
    private static func runCacheExperiment(
        provider: LlamaCppProvider, template: any ModelTemplate
    ) async {
        let preamble = """
        You are discussing Vladimir Putin with the user using offline Wikipedia evidence supplied throughout this conversation. Answer using ONLY that evidence. Be concise and natural. If the answer is absent, say you do not see it. Give just the answer directly.
        """
        let sections = [
            "Vladimir Putin was born on 7 October 1952 in Leningrad. His parents were Vladimir Spiridonovich Putin and Maria Ivanovna Putina. His father served in the Soviet Navy's submarine fleet in the early 1930s and later in the NKVD destruction battalion during the Second World War. His mother was a factory worker.",
            "Putin studied at School No. 193 and later at Saint Petersburg High School 281. He entered Leningrad State University in 1970 and graduated in 1975 with a law degree. His thesis concerned the most favoured nation trading principle in international law.",
            "Putin's paternal grandfather, Spiridon Putin, worked as a personal cook for Vladimir Lenin and Joseph Stalin. Two older brothers died before Vladimir's birth. The family lived in a communal apartment in Leningrad.",
            "After university Putin joined the KGB and trained in Leningrad. He later served in Dresden. Following the collapse of East Germany he returned to Leningrad and worked in the university's international affairs section.",
            "Putin entered politics in Saint Petersburg and later moved to Moscow. He became prime minister in 1999 and acting president at the end of that year. He won the 2000 presidential election.",
            "The supplied evidence is from offline encyclopedia sections selected by retrieval. Each section retains its article and section heading so statements can be attributed accurately and follow-up questions can reuse earlier evidence.",
        ]
        // Repeat the source bundle to approximate the 1.7–1.9k-token prompt
        // measured in the real Wikipedia path on iPhone.
        let evidence = (sections + sections + sections).enumerated().map {
            "## Wikipedia section \($0.offset + 1)\n\($0.element)"
        }.joined(separator: "\n\n")
        let firstUser = "New offline Wikipedia evidence:\n\n\(evidence)\n\nQuestion: Tell me about Vladimir Putin."
        let firstTurns = [ChatTurn(role: .user, text: firstUser)]
        let firstPrompt = template.renderTranscript(
            systemPreamble: preamble, tools: [], turns: firstTurns)

        func generate(_ label: String, prompt: String) async -> String {
            let started = ProcessInfo.processInfo.systemUptime
            var reply = ""
            do {
                for try await chunk in provider.generate(
                    prompt: prompt,
                    parameters: .init(maxTokens: 48, temperature: 0.0, topP: 1.0))
                {
                    reply += chunk
                }
            } catch {
                print("[cache] \(label) FAILED: \(error)")
            }
            let elapsed = ProcessInfo.processInfo.systemUptime - started
            print(String(format:
                "[cache] %@ · prompt=%d chars · reply=%d chars · %.3fs",
                label, prompt.count, reply.count, elapsed))
            return reply
        }

        let firstReply = await generate("cold", prompt: firstPrompt)
        let appendTurns = firstTurns + [
            ChatTurn(role: .assistant, text: firstReply),
            ChatTurn(role: .user, text: "No new evidence; use the evidence already supplied.\n\nQuestion: What about his parents?"),
        ]
        let appendPrompt = template.renderTranscript(
            systemPreamble: preamble, tools: [], turns: appendTurns)
        _ = await generate("append follow-up", prompt: appendPrompt)

        let standaloneTurns = [ChatTurn(
            role: .user,
            text: "New offline Wikipedia evidence:\n\n\(evidence)\n\nQuestion: Where did he go to school?")]
        let standalonePrompt = template.renderTranscript(
            systemPreamble: preamble, tools: [], turns: standaloneTurns)
        _ = await generate("standalone follow-up", prompt: standalonePrompt)
    }
}
