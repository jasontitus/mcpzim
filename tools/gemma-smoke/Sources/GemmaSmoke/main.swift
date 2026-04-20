// SPDX-License-Identifier: MIT

import Foundation
import MCPZimKit
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - CLI arg handling

let userMessage: String = {
    let args = CommandLine.arguments.dropFirst()
    return args.isEmpty ? "tell me about baseball" : args.joined(separator: " ")
}()

print("== Gemma 4 smoke test ==")
print("user: \(userMessage)")

// MARK: - Render our template

let turns: [ChatTurn] = [ChatTurn(role: .user, text: userMessage)]
let prompt = Gemma4PromptTemplate.render(systemPreamble: "", turns: turns)
print("\n--- rendered prompt (\(prompt.count) chars) ---")
print(prompt)
print("--- end prompt ---\n")

// Also render the Gemma4PromptFormatter reference for comparison.
let reference = Gemma4PromptFormatter.userTurn(userMessage)
print("--- Gemma4PromptFormatter.userTurn(...) reference ---")
print(reference)
print("--- end reference ---\n")

if prompt == reference {
    print("OK: our template matches Gemma4PromptFormatter byte-for-byte")
} else {
    print("WARN: our template differs from Gemma4PromptFormatter — will try ours first")
}

// MARK: - Load model

await Gemma4Registration.registerIfNeeded().value

let t0 = Date()
print("\nloading container (modelId=\(Gemma4SwiftCore.verifiedModelId))…")
let container: ModelContainer
do {
    container = try await LLMModelFactory.shared.loadContainer(
        configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId)
    )
} catch {
    fputs("FATAL: loadContainer failed: \(error)\n", stderr)
    exit(1)
}
print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")

// MARK: - Tokenize + generate

let tokens = await container.encode(prompt)
print("\nencoded \(tokens.count) tokens")
print("first 20 ids: \(tokens.prefix(20).map(String.init).joined(separator: ","))")
print("last 20 ids:  \(tokens.suffix(20).map(String.init).joined(separator: ","))")

let input = LMInput(tokens: MLXArray(tokens))
let params = GenerateParameters(maxTokens: 200, temperature: 0.7, topP: 0.95)

print("\n--- streaming generation ---")
let tGen = Date()
var chunkCount = 0
var fullOutput = ""
var pending = ""
let stopMarkers = ["<turn|>", "<|turn>"]
let maxMarker = stopMarkers.map(\.count).max() ?? 0
do {
    let stream = try await container.generate(input: input, parameters: params)
    streamLoop: for await event in stream {
        guard case .chunk(let s) = event else { continue }
        chunkCount += 1
        pending += s
        fullOutput += s
        if let hit = stopMarkers.compactMap({ pending.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
            let clean = String(pending[..<hit.lowerBound])
            if !clean.isEmpty { print(clean, terminator: "") }
            print("\n--- stop marker hit: '\(stopMarkers.first(where: { pending.range(of: $0) != nil })!)' ---")
            break streamLoop
        }
        if pending.count > maxMarker {
            let flushEnd = pending.index(pending.endIndex, offsetBy: -(maxMarker - 1))
            print(String(pending[..<flushEnd]), terminator: "")
            fflush(stdout)
            pending = String(pending[flushEnd...])
        }
    }
    if !pending.isEmpty { print(pending, terminator: "") }
    print()
} catch {
    fputs("\nFATAL: generate threw: \(error)\n", stderr)
    exit(1)
}

print("\n--- stats ---")
print("chunks:     \(chunkCount)")
print("gen time:   \(String(format: "%.2f", Date().timeIntervalSince(tGen)))s")
print("output len: \(fullOutput.count) chars")
print("\n--- raw output dump (so you can see markers verbatim) ---")
print(fullOutput)
print("--- end raw output ---")
