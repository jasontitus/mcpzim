// SPDX-License-Identifier: MIT
//
// Swappable local-LLM abstraction. The chat flow only ever talks to a
// `ModelProvider`, so adding a new model (other MLX-Swift variants, a
// llama.cpp binding, etc.) is a matter of writing one more conformance and
// appending it to `ChatSession.availableModels`.

import Foundation
import MCPZimKit

public struct GenerationParameters: Sendable {
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double
    public var stopSequences: [String]

    public init(
        maxTokens: Int = 512,
        temperature: Double = 0.7,
        topP: Double = 0.95,
        stopSequences: [String] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.stopSequences = stopSequences
    }
}

public enum ModelLoadState: Equatable, Sendable {
    case notLoaded
    case downloading(Double)        // progress 0...1
    case loading
    case ready
    case failed(String)

    public var isReady: Bool { self == .ready }
}

public protocol ModelProvider: AnyObject, Sendable {
    var id: String { get }
    var displayName: String { get }
    var approximateMemoryMB: Int { get }    // for the UI memory warning.
    var supportsToolCalls: Bool { get }

    /// Observable-ish state — implementations usually back this with an
    /// `@MainActor` property on a SwiftUI-visible store so the UI can react
    /// to downloads.
    func stateStream() -> AsyncStream<ModelLoadState>

    func load() async throws
    func unload() async

    /// Stream token chunks for a fully-formatted prompt. The provider applies
    /// no additional chat templating — the caller supplies the final string.
    func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<String, Error>

    /// Render a transcript into the provider's native chat template. The
    /// returned string is ready to feed into `generate(prompt:…)` and ends
    /// on the provider's "open model/assistant turn" marker so generation
    /// continues the current assistant reply.
    func formatTranscript(systemPreamble: String, turns: [ChatTurn]) -> String
}

public extension ModelProvider {
    /// Generic fallback template — `<|role|>\n…` blocks, ending on an open
    /// assistant turn. Providers with a native template should override.
    func formatTranscript(systemPreamble: String, turns: [ChatTurn]) -> String {
        var out = ""
        if !systemPreamble.isEmpty {
            out += "<|system|>\n\(systemPreamble)\n"
        }
        for t in turns {
            out += "<|\(t.role.rawValue)|>\n\(t.text)\n"
        }
        out += "<|assistant|>\n"
        return out
    }
}
