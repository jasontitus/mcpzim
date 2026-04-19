// SPDX-License-Identifier: MIT
//
// Gemma 4 provider over Swift-Gemma4-Core
// (https://github.com/yejingyang8963-byte/Swift-gemma4-core).
//
// The Swift-Gemma4-Core dependency is *optional at compile time*: if you
// haven't added it to the project yet, the file still builds and exposes a
// provider that reports its load state as `.failed("Gemma 4 runtime not
// linked")`. Once you add the package dependency in Xcode, remove the
// `#if canImport(Gemma4SwiftCore)` fallback path and the real implementation
// takes over.

import Foundation

#if canImport(Gemma4SwiftCore) && canImport(MLXLMCommon)
import Gemma4SwiftCore
import MLXLMCommon

public final class Gemma4Provider: ModelProvider, @unchecked Sendable {
    public let id = "gemma4-4b-it-4bit"
    public let displayName = "Gemma 4 4B (4-bit)"
    public let approximateMemoryMB = 400        // per Swift-Gemma4-Core benchmarks.
    public let supportsToolCalls = true

    private let modelId = verifiedModelId       // exported by Gemma4SwiftCore.
    private var container: ModelContainer?
    private var state: ModelLoadState = .notLoaded
    private var continuations: [AsyncStream<ModelLoadState>.Continuation] = []
    private let queue = DispatchQueue(label: "gemma4.state")

    public init() {
        Gemma4Registration.registerIfNeeded()
    }

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { cont in
            queue.sync {
                cont.yield(self.state)
                self.continuations.append(cont)
            }
        }
    }

    private func set(_ s: ModelLoadState) {
        queue.sync {
            self.state = s
            self.continuations.forEach { $0.yield(s) }
        }
    }

    public func load() async throws {
        set(.downloading(0))
        do {
            // mlx-swift-lm's Hub client streams weight downloads; observing
            // progress granularly requires hooking its progress callback.
            // For a first cut we fall through to `.loading` once the download
            // finishes and `loadContainer` returns.
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: ModelConfiguration(id: modelId)
            )
            self.container = container
            set(.ready)
        } catch {
            set(.failed(String(describing: error)))
            throw error
        }
    }

    public func unload() async {
        container = nil
        set(.notLoaded)
    }

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            guard let container else {
                continuation.finish(throwing: ModelError.notLoaded)
                return
            }
            Task {
                do {
                    // MLXLMCommon's `generate` emits `.chunk(String)` events for
                    // each incremental decode step and a `.toolCall` variant we
                    // intentionally ignore here — our own `Gemma4ToolLoop` parses
                    // tool calls out of the text stream instead, to keep the
                    // same code path working for providers that don't support
                    // native tool-call events.
                    for try await event in await container.generate(
                        text: prompt,
                        parameters: GenerateParameters(
                            maxTokens: parameters.maxTokens,
                            temperature: Float(parameters.temperature),
                            topP: Float(parameters.topP)
                        )
                    ) {
                        switch event {
                        case .chunk(let s):
                            continuation.yield(s)
                        case .info, .toolCall:
                            break
                        @unknown default:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

#else

// Fallback used when the Swift-Gemma4-Core package hasn't been added yet.
// Keeps the app buildable and surfaces a clean error in the model picker.
public final class Gemma4Provider: ModelProvider, @unchecked Sendable {
    public let id = "gemma4-4b-it-4bit"
    public let displayName = "Gemma 4 4B (4-bit) — not linked"
    public let approximateMemoryMB = 400
    public let supportsToolCalls = true

    public init() {}

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { cont in
            cont.yield(.failed("Add the Swift-Gemma4-Core package dependency (see ios/README.md)."))
            cont.finish()
        }
    }

    public func load() async throws { throw ModelError.notLinked }
    public func unload() async {}

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { cont in cont.finish(throwing: ModelError.notLinked) }
    }
}

#endif

public enum ModelError: Error, CustomStringConvertible {
    case notLoaded
    case notLinked

    public var description: String {
        switch self {
        case .notLoaded: return "Model is not loaded yet."
        case .notLinked: return "Gemma 4 runtime is not linked. Add the Swift-Gemma4-Core Swift Package to this target."
        }
    }
}
