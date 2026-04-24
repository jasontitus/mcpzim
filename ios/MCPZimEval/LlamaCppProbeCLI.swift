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

        print("[probe] creating LlamaCppProvider…")
        let provider = LlamaCppProvider(
            id: "gemma3-4b-it-q4km-gguf",
            displayName: "Gemma 3 4B IT (Q4_K_M · llama.cpp)",
            huggingFaceRepo: "bartowski/google_gemma-3-4b-it-GGUF",
            ggufFilename: "google_gemma-3-4b-it-Q4_K_M.gguf",
            approximateMemoryMB: 3200,
            template: Gemma3Template()
        )

        print("[probe] provider.load()…")
        do {
            try await provider.load()
        } catch {
            print("[probe] LOAD FAILED: \(error)")
            exit(1)
        }
        print("[probe] load OK")

        // Build a prompt in the same shape ChatSession would — multi-
        // turn transcript with a bars-in-north-beach tool response
        // followed by a follow-up user turn. Use Gemma3Template so
        // the markup matches what the iOS app generates.
        let template = Gemma3Template()
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
}
