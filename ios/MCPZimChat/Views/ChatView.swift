// SPDX-License-Identifier: MIT

import SwiftUI
#if canImport(AppKit)
import AppKit
#endif
#if canImport(UIKit)
import UIKit
#endif

struct ChatView: View {
    @Environment(ChatSession.self) private var session
    @State private var draft = ""
    @State private var showVoiceChat = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Only show the status banner when there's something to
            // report (loading, error, not-yet-loaded). Once the model
            // is ready the chat gets the full vertical space.
            if needsStatusBanner {
                modelBanner
                Divider()
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if session.messages.isEmpty { emptyState }
                        ForEach(session.messages) { m in
                            MessageRow(message: m).id(m.id)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
                // Drag-scrolling the chat also dismisses the keyboard
                // so the debug pane + composer area become reachable
                // without manually tapping "Done".
                .scrollDismissesKeyboard(.immediately)
                .onChange(of: session.messages.last?.text) { _, _ in
                    if let last = session.messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
            if !showVoiceChat { composer }
            DebugPaneView()
        }
        .alert(
            "Error",
            isPresented: .init(
                get: { session.lastError != nil },
                set: { if !$0 { session.lastError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(session.lastError ?? "")
        }
        .sheet(isPresented: $showVoiceChat) {
            VoiceChatView()
                .environment(session)
                // Bottom panel — leaves the chat + map visible so the
                // user can watch the response render (and the route
                // webview with its map) while Kokoro reads it aloud.
                .presentationDetents([.height(72), .fraction(0.4), .large])
                .presentationBackgroundInteraction(.enabled(upThrough: .fraction(0.4)))
                .presentationDragIndicator(.visible)
        }
    }

    private var modelBanner: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(indicatorColor)
                .frame(width: 8, height: 8)
            Text(session.selectedModel.displayName)
                .font(.footnote)
            Spacer()
            Text(stateLabel)
                .font(.footnote)
                .foregroundStyle(.secondary)
            if canLoad {
                Button("Load") { Task { await session.loadSelectedModel() } }
                    .font(.footnote.weight(.semibold))
                    .buttonStyle(.borderedProminent)
                    .controlSize(.mini)
            }
            Button {
                session.resetConversation()
            } label: {
                Image(systemName: "arrow.counterclockwise")
            }
            .help("Start a new conversation")
            .controlSize(.mini)
            .disabled(session.messages.isEmpty || session.isGenerating)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.thinMaterial)
    }

    private var canLoad: Bool {
        switch session.modelState {
        case .notLoaded, .failed: return true
        case .downloading, .loading, .ready: return false
        }
    }

    private var needsStatusBanner: Bool {
        switch session.modelState {
        case .ready: return false
        default: return true
        }
    }

    private var indicatorColor: Color {
        switch session.modelState {
        case .ready: return .green
        case .loading, .downloading: return .yellow
        case .failed: return .red
        case .notLoaded: return .gray
        }
    }

    private var stateLabel: String {
        switch session.modelState {
        case .notLoaded: return "not loaded"
        case .loading: return "loading…"
        case .downloading(let p): return "downloading \(Int(p * 100))%"
        case .ready: return "ready"
        case .failed(let msg): return "error: \(msg)"
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Try asking:").font(.headline)
            Group {
                Text("• What's in my library?")
                Text("• Route from Boston Common to Fenway Park")
                Text("• What is aspirin used for?")
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: $draft, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...5)
                .focused($inputFocused)
                .submitLabel(.send)
                .onSubmit(send)
                // Prewarm the model session as soon as the user
                // focuses the input — Apple FM's system daemon
                // aggressively unloads the model on idle, so getting
                // the KV cache hot during the seconds before the user
                // hits send is the single biggest TTFT win we have.
                .onChange(of: inputFocused) { _, focused in
                    if focused { session.prewarmSelectedModel() }
                }
            Button {
                showVoiceChat = true
            } label: {
                Image(systemName: "mic.circle.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(.tint)
            }
            .accessibilityLabel("Voice chat")
            .disabled(session.isGenerating)
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 28))
            }
            .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty || session.isGenerating)
        }
        .padding(10)
        .background(.bar)
    }

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !session.isGenerating else { return }
        draft = ""
        session.send(text)
    }
}

private struct MessageRow: View {
    let message: ChatMessage
    @Environment(ChatSession.self) private var session
    @State private var justCopied = false

    /// Inline "Sources used" panel is a debug affordance; surface it
    /// only when the debug pane is visible too.
    private var sourcesVisible: Bool { session.showDebugPane }

    /// Article-fetching tool traces that should feed `HeroMediaView`.
    /// Any of these implies the article is load-bearing for the reply,
    /// so surfacing its hero image is useful context.
    /// Compact user-facing duration label. "answered in 4.2 s",
    /// "answered in 1 min 12 s". Italic small-caps hint lives in the
    /// caller so this function is pure.
    static func formatElapsed(_ dt: TimeInterval) -> String {
        if dt < 60 { return String(format: "answered in %.1f s", dt) }
        let total = Int(dt.rounded())
        let m = total / 60, s = total % 60
        return s == 0
            ? "answered in \(m) min"
            : "answered in \(m) min \(s) s"
    }

    static func traceHasArticle(_ trace: ToolCallTrace) -> Bool {
        let names: Set<String> = [
            "get_article", "get_article_section", "list_article_sections",
        ]
        return trace.succeeded && names.contains(trace.name)
    }

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer()
                bubble(fill: Color.accentColor.opacity(0.12))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                // Map first — feels natural for routing answers, and
                // the streaming text grows downward below it instead
                // of pushing the map around as new sentences arrive.
                ForEach(message.toolCalls) { trace in
                    if traceHasRoute(trace) {
                        RouteWebView(trace: trace)
                    } else if Self.traceHasArticle(trace) {
                        // Any tool call that named a specific
                        // article — full fetch, section pull, or
                        // section list — is a signal that the
                        // article is load-bearing for the reply.
                        // Surface its hero image / video.
                        HeroMediaView(trace: trace)
                    }
                }
                ZStack(alignment: .topTrailing) {
                    bubble(fill: Color.gray.opacity(0.15))
                    if !message.text.isEmpty {
                        copyButton
                            .padding(6)
                    }
                }
                if let elapsed = message.elapsed,
                   !message.text.isEmpty,
                   !session.isGenerating || message.id != session.messages.last?.id
                {
                    Text(Self.formatElapsed(elapsed))
                        .font(.caption2.italic())
                        .foregroundStyle(.secondary)
                        .padding(.leading, 10)
                }
                if !message.toolCalls.isEmpty, sourcesVisible {
                    SourcesSection(traces: message.toolCalls)
                }
            }
        case .tool:
            // Raw tool-result JSON is part of the debug surface; hide
            // unless the user has asked for it. The model's own reply
            // summarizes the result for the user.
            if sourcesVisible {
                bubble(fill: Color.orange.opacity(0.10))
            } else {
                EmptyView()
            }
        case .system:
            EmptyView()
        }
    }

    @ViewBuilder
    private func bubble(fill: Color) -> some View {
        Text(Self.displayText(message.text, role: message.role))
            .textSelection(.enabled)
            .padding(10)
            .background(fill, in: RoundedRectangle(cornerRadius: 12))
            .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }

    /// Strip leftover tool-call markup from the assistant's visible
    /// prose. The parser catches well-formed blocks, but during
    /// streaming we briefly see the half-emitted opener (e.g.
    /// `<|tool_call>call:search{query:<|"|>pizza`) before the closing
    /// sentinel arrives. Nuke anything from the first opener to the
    /// end of the string so the chat never flashes raw template text.
    private static func displayText(_ raw: String, role: ChatMessage.Role) -> String {
        guard role == .assistant else { return raw }
        var t = raw
        // Closed blocks (all the canonical spellings).
        let closedPatterns = [
            #"<\|tool_call\|?>[\s\S]*?<tool_call\|>"#,
            #"<tool_call>[\s\S]*?</tool_call>"#,
            #"<\|tool_response\|?>[\s\S]*?<tool_response\|>"#,
        ]
        for pat in closedPatterns {
            t = t.replacingOccurrences(of: pat, with: "", options: .regularExpression)
        }
        // Any stray opener — drop from there to end of string. During
        // streaming this hides the partially-arrived tool-call until
        // the parser finishes; after parse the opener shouldn't remain,
        // but if the model went off-format we still don't show raw
        // template text. Use a broad prefix so a 4-byte token like
        // "<|to" or "<|tool_c" in-flight also gets masked.
        if let r = t.range(of: #"<\|?tool[_a-z]*"#, options: .regularExpression) {
            t = String(t[..<r.lowerBound])
        }
        if let r = t.range(of: #"<tool[_a-z]*"#, options: .regularExpression) {
            t = String(t[..<r.lowerBound])
        }
        // Drop any lingering sentinel scraps.
        for lit in ["<tool_call|>", "<tool_response|>", "<|\"|>", "<|\""] {
            t = t.replacingOccurrences(of: lit, with: "")
        }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// One-click copy of the assistant reply. Turns into a check-mark for
    /// a second so the user sees it actually landed in the clipboard.
    private var copyButton: some View {
        Button {
            copyMessage(message.text)
            justCopied = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                justCopied = false
            }
        } label: {
            Image(systemName: justCopied ? "checkmark" : "doc.on.doc")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(4)
                .background(.regularMaterial, in: Circle())
        }
        .buttonStyle(.plain)
        .help(justCopied ? "Copied" : "Copy reply")
    }
}

private func copyMessage(_ text: String) {
    #if canImport(AppKit)
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
    #elseif canImport(UIKit)
    UIPasteboard.general.string = text
    #endif
}

private struct ToolCallRow: View {
    let trace: ToolCallTrace
    @State private var expanded = false
    @State private var justCopiedArgs = false
    @State private var justCopiedResult = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Button {
                withAnimation { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: trace.succeeded ? "wrench.and.screwdriver" : "exclamationmark.triangle")
                    Text(trace.name).font(.footnote.weight(.semibold))
                    Spacer()
                    Button {
                        copyMessage(copyPayload)
                        justCopiedResult = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                            justCopiedResult = false
                        }
                    } label: {
                        Image(systemName: justCopiedResult ? "checkmark" : "doc.on.doc")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .help(justCopiedResult ? "Copied" : "Copy full trace")
                    Image(systemName: expanded ? "chevron.up" : "chevron.down").font(.caption)
                }
                .foregroundStyle(trace.succeeded ? Color.primary : Color.red)
            }
            if expanded {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .top) {
                        Text("args: \(trace.arguments)")
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        copyChip(label: justCopiedArgs ? "✓" : "copy") {
                            copyMessage(trace.arguments)
                            justCopiedArgs = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                                justCopiedArgs = false
                            }
                        }
                    }
                    if let err = trace.error {
                        Text("error: \(err)")
                            .font(.caption.monospaced())
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                    } else if let prose = articleProse(from: trace) {
                        // For article-body tools (get_article,
                        // get_article_section) the most useful thing
                        // to show is the actual text the model saw —
                        // rendered as real prose, not JSON. This is
                        // what makes "did it come from Wikipedia?"
                        // answerable at a glance.
                        HStack(alignment: .top) {
                            Text(prose)
                                .font(.callout)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            copyChip(label: justCopiedResult ? "✓" : "copy") {
                                copyMessage(prose)
                                justCopiedResult = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                                    justCopiedResult = false
                                }
                            }
                        }
                    } else {
                        HStack(alignment: .top) {
                            Text(trace.result)
                                .font(.caption.monospaced())
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            copyChip(label: justCopiedResult ? "✓" : "copy") {
                                copyMessage(trace.result)
                                justCopiedResult = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                                    justCopiedResult = false
                                }
                            }
                        }
                    }
                }
            }
        }
        .padding(8)
        .background(Color.gray.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
    }

    /// Extract readable article prose from an article-returning
    /// tool's result JSON. Returns nil for other tools (so the
    /// caller falls back to raw monospace JSON).
    private func articleProse(from trace: ToolCallTrace) -> String? {
        guard trace.name == "get_article" || trace.name == "get_article_section" else {
            return nil
        }
        guard let data = trace.result.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = obj["text"] as? String
        else { return nil }
        // Prepend a header line so the user has article / section
        // context without hunting through metadata.
        var header: [String] = []
        if let title = obj["title"] as? String, !title.isEmpty {
            header.append(title)
        }
        if let section = obj["section"] as? String, !section.isEmpty, section.lowercased() != "lead" {
            header.append("§ \(section)")
        } else if trace.name == "get_article_section" {
            header.append("§ lead")
        }
        let prefix = header.isEmpty ? "" : header.joined(separator: " ") + "\n\n"
        return prefix + text
    }

    /// A compact name/args/result/error dump suitable for pasting into a bug
    /// report or a follow-up question.
    private var copyPayload: String {
        var out = "tool: \(trace.name)\nargs: \(trace.arguments)\n"
        if let err = trace.error {
            out += "error: \(err)\n"
        } else {
            out += "result: \(trace.result)\n"
        }
        return out
    }

    private func copyChip(label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.caption2.monospaced())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.gray.opacity(0.15), in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// Per-assistant-turn "Sources used" audit trail. Groups every
/// tool call that ran during this turn under one expandable header
/// so the user can verify what the model actually had access to
/// vs. what might have come from training priors. Defaults to
/// expanded when sources exist — the whole point is visibility.
private struct SourcesSection: View {
    let traces: [ToolCallTrace]
    @State private var expanded = true

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(traces) { trace in
                    ToolCallRow(trace: trace)
                }
            }
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.caption)
                Text("Sources used (\(traces.count))")
                    .font(.caption.weight(.semibold))
                if let hint = briefSummary {
                    Text("— \(hint)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            .foregroundStyle(.secondary)
        }
        .padding(8)
        .background(Color.blue.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    /// One-line summary of which article(s) / tool(s) were consulted,
    /// rendered next to the collapsible header so the user doesn't
    /// have to expand to know the gist.
    private var briefSummary: String? {
        let articleTools = traces.filter {
            ["get_article", "get_article_section"].contains($0.name)
        }
        if !articleTools.isEmpty {
            let paths = articleTools.compactMap { extractJSONField("path", from: $0.arguments) }
            let uniq = Array(Set(paths)).sorted()
            if !uniq.isEmpty {
                return uniq.prefix(2).joined(separator: ", ")
                    + (uniq.count > 2 ? ", …" : "")
            }
        }
        let names = Array(Set(traces.map(\.name))).sorted()
        return names.prefix(3).joined(separator: ", ")
    }

    /// Cheap extractor — the args payload is small JSON, not worth
    /// pulling in JSONDecoder for.
    private static func extractJSONField(_ field: String, from json: String) -> String? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj[field] as? String
    }
    private func extractJSONField(_ field: String, from json: String) -> String? {
        Self.extractJSONField(field, from: json)
    }
}

#Preview {
    ChatView().environment(ChatSession())
}
