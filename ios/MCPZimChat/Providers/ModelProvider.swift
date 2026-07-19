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
    public var topK: Int
    /// Open-ended/model-native generation should normally use the model's
    /// published sampler. Grounded extraction and tool-result summarization
    /// can opt out for deterministic factual answers.
    public var useModelSamplingProfile: Bool
    public var stopSequences: [String]

    public init(
        maxTokens: Int = 512,
        temperature: Double = 0.7,
        topP: Double = 0.95,
        topK: Int = 40,
        useModelSamplingProfile: Bool = true,
        stopSequences: [String] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.useModelSamplingProfile = useModelSamplingProfile
        self.stopSequences = stopSequences
    }
}

/// A model-specific sampling recipe. Callers can continue supplying their
/// task-level defaults through `GenerationParameters`; providers only install
/// this override when a publisher recommends a materially different recipe.
public struct GenerationSamplingProfile: Sendable {
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var presencePenalty: Double

    public init(
        temperature: Double,
        topP: Double,
        topK: Int,
        presencePenalty: Double = 0
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.presencePenalty = presencePenalty
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

    /// Model-family-specific prompt + tool-call template. Drives every
    /// tool-surface detail that differs across LLM families — Gemma 4's
    /// `<|turn>` markers + custom tool-call syntax vs Qwen's ChatML +
    /// JSON tool calls, etc. The host (ChatSession / MCPToolAdapter)
    /// calls through this protocol and never touches a model-specific
    /// type directly, so adding a new family is `struct FooTemplate:
    /// ModelTemplate { … }` plus a provider that returns it here.
    var template: any ModelTemplate { get }

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

    /// Request cancellation of the active generation. Implementations should
    /// return promptly at their next safe batch/token boundary. The default is
    /// a no-op for providers whose async stream already follows Task
    /// cancellation; llama.cpp overrides it because its decode loop is a
    /// detached blocking task.
    func cancelGeneration()

    /// Render a transcript into the provider's native chat template. The
    /// returned string is ready to feed into `generate(prompt:…)` and ends
    /// on the provider's "open model/assistant turn" marker so generation
    /// continues the current assistant reply.
    func formatTranscript(systemPreamble: String, turns: [ChatTurn]) -> String
}

public extension ModelProvider {
    func cancelGeneration() {}

    /// Default: Gemma 4's template. Providers for other families
    /// (Qwen, Llama 3, …) should override. Safe default for the
    /// MockProvider / FoundationModelsProvider paths — those never
    /// exercise the text-loop tool surface, so the template is
    /// effectively unused.
    var template: any ModelTemplate { Gemma4Template() }

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
