// SPDX-License-Identifier: MIT
//
// Canned-response provider for UI-only development. Lets you work on SwiftUI
// layout, tool-call rendering, and library management without loading the
// 1.5 GB Gemma weights or dealing with Metal warm-up. It even honours
// tool-call syntax so you can exercise the whole tool loop end-to-end.

import Foundation

public final class MockProvider: ModelProvider, @unchecked Sendable {
    public let id = "mock"
    public let displayName = "Mock (scripted)"
    public let approximateMemoryMB = 0
    public let supportsToolCalls = true

    private let lock = NSLock()
    private var stateContinuations: [AsyncStream<ModelLoadState>.Continuation] = []
    private var state: ModelLoadState = .notLoaded

    public init() {}

    public func stateStream() -> AsyncStream<ModelLoadState> {
        AsyncStream { cont in
            self.lock.lock()
            let s = self.state
            self.stateContinuations.append(cont)
            self.lock.unlock()
            cont.yield(s)
        }
    }

    private func set(_ new: ModelLoadState) {
        lock.lock()
        state = new
        let conts = stateContinuations
        lock.unlock()
        conts.forEach { $0.yield(new) }
    }

    public func load() async throws {
        set(.loading)
        try? await Task.sleep(nanoseconds: 300_000_000)
        set(.ready)
    }

    public func unload() async { set(.notLoaded) }

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                let reply = Self.script(for: prompt)
                for chunk in reply.chunked(into: 8) {
                    try? await Task.sleep(nanoseconds: 30_000_000)
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
        }
    }

    /// Recognise a couple of common prompts so the mock returns something
    /// sensible. For everything else, echo a plausible-sounding sentence.
    private static func script(for prompt: String) -> String {
        let p = prompt.lowercased()
        if p.contains("route") && (p.contains("boston") || p.contains("fenway") || p.contains(" to ")) {
            return """
            I'll plan that route. <tool_call>{"name":"route_from_places","arguments":{"origin":"Boston Common","destination":"Fenway Park"}}</tool_call>
            """
        }
        if p.contains("list") && p.contains("librar") {
            return """
            <tool_call>{"name":"list_libraries","arguments":{}}</tool_call>
            """
        }
        return "Okay — that request reached a mock model. Switch to Gemma 4 in the model picker for a real answer."
    }
}

private extension String {
    func chunked(into size: Int) -> [String] {
        guard size > 0 else { return [self] }
        var out: [String] = []
        var i = startIndex
        while i < endIndex {
            let end = index(i, offsetBy: size, limitedBy: endIndex) ?? endIndex
            out.append(String(self[i..<end]))
            i = end
        }
        return out
    }
}
