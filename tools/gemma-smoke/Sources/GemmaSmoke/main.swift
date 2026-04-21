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

// Probe mode: run a tokenizer-stability check for KV-cache reuse.
// Triggered by `GEMMA_SMOKE_MODE=cache-probe ./GemmaSmoke`. When set,
// the tool renders two prompts where `prompt1` is a strict append
// extension of `prompt0` (as ChatSession does iter0 → iter1), encodes
// both, and reports the longest-common-prefix. If LCP < tokens0.count
// the tokenizer re-segmented across the boundary, which is what kills
// our KV-cache reuse on the iOS side.
let probeMode: Bool = ProcessInfo.processInfo.environment["GEMMA_SMOKE_MODE"] == "cache-probe"
// Prompt-layout experiment mode: simulates the iOS "directions to SF /
// then SJ" flow and reports cache hit rates for several preamble
// layouts. Triggered by `GEMMA_SMOKE_MODE=prompt-experiment`.
let experimentMode: Bool = ProcessInfo.processInfo.environment["GEMMA_SMOKE_MODE"] == "prompt-experiment"

print("== Gemma 4 smoke test ==")
if probeMode { print("MODE: cache-probe (no generation; LCP report only)") }
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

if experimentMode {
    let exp = PromptExperiment(container: container)
    try await exp.run()
    exit(0)
}

// MARK: - Tokenize + generate

let tokens = await container.encode(prompt)
print("\nencoded \(tokens.count) tokens")
print("first 20 ids: \(tokens.prefix(20).map(String.init).joined(separator: ","))")
print("last 20 ids:  \(tokens.suffix(20).map(String.init).joined(separator: ","))")

// MARK: - Probe mode: tokenizer LCP stability

if probeMode {
    // Simulate what ChatSession does across iter 0 → iter 1 of a
    // single user turn:
    //   iter 0 prompt: [system][tools][user]
    //   iter 0 generates: "<assistant text><tool_call|>"
    //   iter 1 prompt: [system][tools][user][asst emission][tool result][assistant re-open]
    // If the tokenizer's prefix is stable, LCP(iter1tokens, iter0tokens+emitted) == iter0tokens+emitted.
    // If BPE re-segments across the boundary, LCP < that and we lose cache reuse.
    let systemPreamble = "" // use empty so we isolate the re-tokenization question
    let prompt0 = Gemma4PromptTemplate.render(
        systemPreamble: systemPreamble,
        turns: [ChatTurn(role: .user, text: userMessage)]
    )
    // Simulated assistant emission — matches what Gemma typically produces
    // when it issues a tool call on iter 0. Includes the native tool-call
    // markers since those are what our cutoff logic retains.
    let assistantEmission = "I'll look that up. <|tool_call>call:search{query:\"baseball\"}<tool_call|>"
    // Simulated tool response + next-assistant re-open, as rendered by
    // Gemma4PromptTemplate when the transcript grows.
    let prompt1 = Gemma4PromptTemplate.render(
        systemPreamble: systemPreamble,
        turns: [
            ChatTurn(role: .user, text: userMessage),
            ChatTurn(role: .assistant, text: assistantEmission),
            ChatTurn(role: .tool, text: "{\"hits\":[{\"title\":\"Baseball\"}]}"),
        ]
    )
    print("\n== cache-probe ==")
    print("prompt0 length: \(prompt0.count) chars")
    print("prompt1 length: \(prompt1.count) chars")
    print("prompt0 is string prefix of prompt1: \(prompt1.hasPrefix(prompt0))")
    if !prompt1.hasPrefix(prompt0) {
        let c0 = Array(prompt0), c1 = Array(prompt1)
        var i = 0
        while i < c0.count && i < c1.count && c0[i] == c1[i] { i += 1 }
        print("  string diverges at char idx \(i)")
        let lo = max(0, i - 20), hi0 = min(c0.count, i + 20), hi1 = min(c1.count, i + 20)
        print("  p0 context: \(String(c0[lo..<hi0]).debugDescription)")
        print("  p1 context: \(String(c1[lo..<hi1]).debugDescription)")
    }
    let t0 = (await container.encode(prompt0)).map { Int32($0) }
    let t1 = (await container.encode(prompt1)).map { Int32($0) }
    print("\ntokens0: \(t0.count)")
    print("tokens1: \(t1.count)")
    // How many leading tokens of t1 match t0?
    var lcp01 = 0
    let n01 = min(t0.count, t1.count)
    while lcp01 < n01 && t0[lcp01] == t1[lcp01] { lcp01 += 1 }
    print("LCP(t0, t1) = \(lcp01)")
    print("  → if == tokens0.count (\(t0.count)), ChatSession's cachedTokens at end of iter 0 WOULD prefix-match iter 1.")
    print("  → if < tokens0.count, the BPE tokenizer re-segments text near the boundary and cache reuse dies.")
    if lcp01 < t0.count {
        let start = max(0, lcp01 - 4)
        let end0 = min(t0.count, lcp01 + 4)
        let end1 = min(t1.count, lcp01 + 4)
        print("  diverge at token idx \(lcp01):")
        print("    t0[\(start)..<\(end0)] = \(Array(t0[start..<end0]))")
        print("    t1[\(start)..<\(end1)] = \(Array(t1[start..<end1]))")
    }
    // Also: the full "cached" state at end of iter 0 in ChatSession is
    // (prompt0 tokens) + (generated tokens for assistantEmission). Check
    // whether that concatenation byte-matches the prefix of t1.
    let emitTokens = await container.encode(assistantEmission)
    let cachedLikeEndOfIter0: [Int32] = t0 + emitTokens.map { Int32($0) }
    var lcpCache = 0
    let nCache = min(cachedLikeEndOfIter0.count, t1.count)
    while lcpCache < nCache && cachedLikeEndOfIter0[lcpCache] == t1[lcpCache] { lcpCache += 1 }
    print("\nsimulating ChatSession's cachedTokens at end of iter 0:")
    print("  cached.count (= prompt0 tokens + assistantEmission tokens): \(cachedLikeEndOfIter0.count)")
    print("  LCP(cached, t1) = \(lcpCache)")
    print("  missed by: \(cachedLikeEndOfIter0.count - lcpCache) tokens")
    if lcpCache < cachedLikeEndOfIter0.count {
        let start = max(0, lcpCache - 4)
        let endC = min(cachedLikeEndOfIter0.count, lcpCache + 6)
        let end1 = min(t1.count, lcpCache + 6)
        print("  cached[\(start)..<\(endC)] = \(Array(cachedLikeEndOfIter0[start..<endC]))")
        print("  t1    [\(start)..<\(end1)] = \(Array(t1[start..<end1]))")
    }

    // BPE round-trip stability. On iOS, the KV cache stores raw token
    // IDs emitted by the SAMPLER (during streaming). When ChatSession
    // rebuilds iter 1's prompt, it re-encodes the DECODED assistant
    // text. If encode(decode(tokens)) != tokens we lose cache reuse.
    // Stress this with several realistic assistant emissions including
    // tool-call markers, URLs, numbers, emoji — the spots where BPE
    // often re-merges.
    print("\n== BPE round-trip stability ==")
    let stressSamples = [
        "I'll look that up. <|tool_call>call:search{query:\"baseball\"}<tool_call|>",
        "Sure — calling `route_from_places` with origin_lat:37.44121 origin_lon:-122.15530",
        "Here you go: https://example.com/path?q=abc&r=123. See §3.",
        "Temperature 72°F. 🔥 ok?",
    ]
    var allStable = true
    for (idx, s) in stressSamples.enumerated() {
        let enc1 = await container.encode(s)
        let tokenizer = await container.perform { ctx in ctx.tokenizer }
        let decoded = tokenizer.decode(tokens: enc1)
        let enc2 = await container.encode(decoded)
        let same = enc1 == enc2
        if !same { allStable = false }
        print("[\(idx)] len1=\(enc1.count) len2=\(enc2.count) same=\(same)  sample: \(s.prefix(40))…")
        if !same {
            // Where do they diverge?
            var k = 0
            while k < min(enc1.count, enc2.count) && enc1[k] == enc2[k] { k += 1 }
            print("    diverges at token idx \(k):")
            let s1 = max(0, k - 3), e1 = min(enc1.count, k + 5), e2 = min(enc2.count, k + 5)
            print("    enc1[\(s1)..<\(e1)] = \(Array(enc1[s1..<e1]))")
            print("    enc2[\(s1)..<\(e2)] = \(Array(enc2[s1..<e2]))")
            let d1 = tokenizer.decode(tokens: Array(enc1[s1..<e1]))
            let d2 = tokenizer.decode(tokens: Array(enc2[s1..<e2]))
            print("    decoded enc1 span: \(d1.debugDescription)")
            print("    decoded enc2 span: \(d2.debugDescription)")
        }
    }
    print("\nall round-trip stable: \(allStable)")

    // The decisive test: run an ACTUAL generation, capture raw sampler
    // token IDs in order, then compare `rawIDs` to `encode(decode(rawIDs))`.
    // If they differ — even by a single token — the iOS cache strategy
    // of storing rawIDs and comparing to encode(newPrompt) is broken.
    print("\n== real generation → re-encode test ==")
    let genPrompt = Gemma4PromptTemplate.render(
        systemPreamble: "",
        turns: [ChatTurn(role: .user, text: userMessage)]
    )
    let promptTokens = await container.encode(genPrompt)
    let lmInput = LMInput(tokens: MLXArray(promptTokens))
    let genParams2 = GenerateParameters(maxTokens: 60, temperature: 0.0, topP: 1.0, prefillStepSize: 128)
    var rawSamplerIDs: [Int32] = []
    let stream2 = try await container.perform { ctx in
        try MLXLMCommon.generateTokens(
            input: lmInput, cache: ctx.model.newCache(parameters: genParams2),
            parameters: genParams2, context: ctx
        )
    }
    for await event in stream2 {
        guard case .token(let id) = event else { continue }
        rawSamplerIDs.append(Int32(id))
    }
    let decoded3 = await container.perform { ctx in
        ctx.tokenizer.decode(tokens: rawSamplerIDs.map { Int($0) })
    }
    let reencoded3 = await container.encode(decoded3)
    print("rawSampler IDs: \(rawSamplerIDs.count)")
    print("re-encoded   :  \(reencoded3.count)")
    print("equal        :  \(rawSamplerIDs.map { Int($0) } == reencoded3)")
    if rawSamplerIDs.map({ Int($0) }) != reencoded3 {
        print("→ DRIFT CONFIRMED. The iOS KV cache stores sampler IDs; the next turn's prompt")
        print("  encodes the decoded text and gets DIFFERENT token IDs — cache never hits.")
        var k = 0
        let a = rawSamplerIDs.map { Int($0) }, b = reencoded3
        while k < min(a.count, b.count) && a[k] == b[k] { k += 1 }
        print("  diverge at token idx \(k):")
        let s1 = max(0, k - 3), e1 = min(a.count, k + 5), e2 = min(b.count, k + 5)
        print("  rawSampler[\(s1)..<\(e1)] = \(Array(a[s1..<e1]))")
        print("  reencoded [\(s1)..<\(e2)] = \(Array(b[s1..<e2]))")
    } else {
        print("→ rawSamplerIDs == encode(decode(rawSamplerIDs)). No drift for this sample.")
        print("  Bug must be elsewhere (likely cachedTokens bookkeeping around tool-call cutoff).")
    }
    print("\ndecoded text: \(decoded3.prefix(200))…")

    // Final test: prove (or disprove) that BPE preserves token
    // boundaries when appending ChatSession's template suffix after an
    // assistant emission. cachedTokens at end of iter 0 stores
    // encode(emission) up to <tool_call|>. Iter 1's prompt concatenates
    // emission + "<turn|>\n<|turn>tool\n..." — if the first byte of
    // the suffix merges with the last byte of emission, encode(iter1)
    // won't contain encode(emission) as a contiguous prefix block and
    // LCP falls short by the number of re-merged tokens.
    print("\n== emission + suffix boundary test ==")
    let emissionSuffix = "\n<turn|>\n<|turn>tool\n{\"ok\":true}<turn|>\n<|turn>model\n"
    for (idx, emission) in stressSamples.enumerated() {
        let encEm = await container.encode(emission).map { Int32($0) }
        let encEmPlusSuf = await container.encode(emission + emissionSuffix).map { Int32($0) }
        var k = 0
        while k < encEm.count && k < encEmPlusSuf.count && encEm[k] == encEmPlusSuf[k] { k += 1 }
        let boundaryOK = (k == encEm.count)
        print("[\(idx)] boundary preserved: \(boundaryOK)  (emission=\(encEm.count) tokens, matched prefix=\(k))")
        if !boundaryOK {
            print("    emission[last 4]: \(Array(encEm.suffix(4)))")
            print("    withSuf[\(encEm.count-4)..<\(encEm.count)]: \(Array(encEmPlusSuf[(encEm.count-4)..<encEm.count]))")
        }
    }
    exit(0)
}

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
