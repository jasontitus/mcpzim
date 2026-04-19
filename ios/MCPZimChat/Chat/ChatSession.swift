// SPDX-License-Identifier: MIT
//
// Top-level observable app state. Owns the list of opened ZIM readers, the
// set of available models, the current chat transcript, and a reference to
// the MCPZim tool adapter that a Gemma 4 tool loop can dispatch through.

import Foundation
import MCPZimKit
import Observation

@MainActor
@Observable
public final class ChatSession {
    // MARK: - Library (opened ZIMs)

    public struct LibraryEntry: Identifiable, Sendable {
        public let id = UUID()
        public let url: URL
        public let reader: ZimReader
        public var kind: ZimKind { reader.kind }
        public var displayName: String {
            reader.metadata.title.isEmpty ? url.lastPathComponent : reader.metadata.title
        }
    }

    public var library: [LibraryEntry] = []
    public var libraryError: String?

    // MARK: - Models

    public private(set) var models: [any ModelProvider]
    public var selectedModel: any ModelProvider
    public var modelState: ModelLoadState = .notLoaded

    // MARK: - Transcript

    public var messages: [ChatMessage] = []
    public var isGenerating = false
    public var lastError: String?

    // MARK: - Plumbing

    public var service: DefaultZimService?
    public var adapter: MCPToolAdapter?

    @ObservationIgnored private var stateObservationTask: Task<Void, Never>?

    public init() {
        let mock = MockProvider()
        let gemma = Gemma4Provider()
        self.models = [gemma, mock]
        self.selectedModel = mock
        startObservingSelectedModel()
    }

    private func startObservingSelectedModel() {
        stateObservationTask?.cancel()
        stateObservationTask = Task { [weak self] in
            guard let self else { return }
            for await state in self.selectedModel.stateStream() {
                self.modelState = state
            }
        }
    }

    // MARK: - Library management

    public func scanDocumentsFolder() async {
        let fm = FileManager.default
        guard let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        else { return }
        let files = (try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil))?.filter {
            $0.pathExtension.lowercased() == "zim"
        } ?? []
        await openReaders(urls: files)
    }

    public func openReaders(urls: [URL]) async {
        var opened: [LibraryEntry] = []
        for url in urls {
            do {
                let reader = try LibzimReader(url: url)
                opened.append(LibraryEntry(url: url, reader: reader))
            } catch {
                libraryError = "Could not open \(url.lastPathComponent): \(error)"
            }
        }
        library = opened
        await rebuildService()
    }

    private func rebuildService() async {
        let pairs = library.map { ($0.url.lastPathComponent, $0.reader as ZimReader) }
        let svc = DefaultZimService(readers: pairs)
        self.service = svc
        self.adapter = await MCPToolAdapter.from(service: svc)
    }

    // MARK: - Model switching

    public func select(modelId: String) async {
        guard let found = models.first(where: { $0.id == modelId }) else { return }
        // Unload the previous model — iOS memory budget is tight.
        await selectedModel.unload()
        selectedModel = found
        startObservingSelectedModel()
    }

    public func loadSelectedModel() async {
        do {
            try await selectedModel.load()
        } catch {
            lastError = String(describing: error)
        }
    }

    // MARK: - Send a user turn

    public func send(_ text: String) {
        let user = ChatMessage(role: .user, text: text)
        messages.append(user)
        messages.append(ChatMessage(role: .assistant, text: ""))
        isGenerating = true
        Task { await runGenerationLoop() }
    }

    /// Core tool-aware generation loop.
    ///
    /// The strategy is deliberately transport-agnostic: we build a plain-text
    /// transcript from the current messages, stream tokens from the selected
    /// `ModelProvider`, and watch for `<tool_call>...</tool_call>` blocks in
    /// the stream. When one is detected, the loop halts generation, calls
    /// `MCPToolAdapter.dispatch(...)`, appends a synthetic tool response to
    /// the transcript, and restarts.
    private func runGenerationLoop() async {
        defer { isGenerating = false }
        guard let adapter else {
            appendAssistant("[No ZIMs loaded — add .zim files to the app's Documents folder, then tap Refresh Library.]")
            return
        }

        // Build the transcript. Swift-Gemma4-Core's Gemma4PromptFormatter
        // applies Gemma-specific turn tokens; we keep the prompt format here
        // provider-neutral and let the provider add its own templating if
        // desired. The tool preamble comes from the adapter.
        let registry = await adapter.registry
        var transcript = Self.toolsPreamble(registry: registry)
        for m in messages where !m.text.isEmpty {
            switch m.role {
            case .system:    transcript += "<|system|>\n\(m.text)\n"
            case .user:      transcript += "<|user|>\n\(m.text)\n"
            case .assistant: transcript += "<|assistant|>\n\(m.text)\n"
            case .tool:      transcript += "<|tool|>\n\(m.text)\n"
            }
        }
        transcript += "<|assistant|>\n"

        // Up to 4 tool loops per user turn — stop runaway recursion.
        for _ in 0..<4 {
            var buffer = ""
            var toolCall: (range: Range<String.Index>, name: String, args: [String: Any])?

            do {
                for try await chunk in selectedModel.generate(prompt: transcript, parameters: .init()) {
                    buffer += chunk
                    appendToAssistant(buffer)
                    if let call = Self.extractToolCall(in: buffer) {
                        toolCall = call
                        break
                    }
                }
            } catch {
                lastError = String(describing: error)
                return
            }

            guard let call = toolCall else { return }

            // Dispatch the tool, record the trace, append synthetic tool turn.
            let argsData = try? JSONSerialization.data(withJSONObject: call.args)
            let argsStr = argsData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            do {
                let result = try await adapter.dispatch(tool: call.name, args: call.args)
                let resultData = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
                let resultStr = String(data: resultData, encoding: .utf8) ?? "{}"
                recordToolTrace(ToolCallTrace(name: call.name, arguments: argsStr, result: resultStr, error: nil))
                let pre = String(buffer[..<call.range.lowerBound])
                updateAssistant(pre)                            // truncate the <tool_call> block from the visible message.
                transcript += pre
                transcript += "<|tool|>\n" + resultStr + "\n<|assistant|>\n"
            } catch {
                let err = String(describing: error)
                recordToolTrace(ToolCallTrace(name: call.name, arguments: argsStr, result: "", error: err))
                transcript += "<|tool|>\n[error] \(err)\n<|assistant|>\n"
            }
        }
    }

    // MARK: - Transcript helpers

    private func appendAssistant(_ text: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = text
        }
    }

    private func appendToAssistant(_ replacement: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = replacement
        }
    }

    private func updateAssistant(_ newText: String) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = newText
        }
    }

    private func recordToolTrace(_ trace: ToolCallTrace) {
        if messages.last?.role == .assistant {
            messages[messages.count - 1].toolCalls.append(trace)
        }
    }

    // MARK: - Prompt formatting

    private static func toolsPreamble(registry: MCPToolRegistry) -> String {
        var lines: [String] = [
            "<|system|>",
            "You are a helpful assistant running on-device. You have access to the following tools.",
            "To call a tool, emit a single line:",
            "<tool_call>{\"name\":\"TOOL_NAME\",\"arguments\":{...}}</tool_call>",
            "and wait for the <tool_response> turn before continuing.",
            "",
            "Available tools:",
        ]
        for tool in registry.tools {
            let schema = String(data: tool.inputSchemaJSON, encoding: .utf8) ?? "{}"
            lines.append("- \(tool.name): \(tool.description)")
            lines.append("  input: \(schema)")
        }
        lines.append("")
        return lines.joined(separator: "\n")
    }

    static func extractToolCall(in buffer: String) -> (range: Range<String.Index>, name: String, args: [String: Any])? {
        guard let start = buffer.range(of: "<tool_call>"),
              let end = buffer.range(of: "</tool_call>", range: start.upperBound..<buffer.endIndex)
        else { return nil }
        let jsonRange = start.upperBound..<end.lowerBound
        let json = String(buffer[jsonRange])
        guard let data = json.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = parsed["name"] as? String
        else { return nil }
        let args = (parsed["arguments"] as? [String: Any]) ?? [:]
        return (start.lowerBound..<end.upperBound, name, args)
    }
}
