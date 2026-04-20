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
            modelBanner
            Divider()
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
                .onChange(of: session.messages.last?.text) { _, _ in
                    if let last = session.messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
            composer
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
    @State private var justCopied = false

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer()
                bubble(fill: Color.accentColor.opacity(0.12))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                ZStack(alignment: .topTrailing) {
                    bubble(fill: Color.gray.opacity(0.15))
                    if !message.text.isEmpty {
                        copyButton
                            .padding(6)
                    }
                }
                ForEach(message.toolCalls) { trace in
                    ToolCallRow(trace: trace)
                    if traceHasRoute(trace) {
                        // Offline streetzim viewer (tiles/JS/CSS served from
                        // the ZIM via `zim://`). MapKit used to render here
                        // as a sanity check but it pulls Apple Maps tiles
                        // over the network, which defeats the offline story.
                        RouteWebView(trace: trace)
                    }
                }
            }
        case .tool:
            bubble(fill: Color.orange.opacity(0.10))
        case .system:
            EmptyView()
        }
    }

    @ViewBuilder
    private func bubble(fill: Color) -> some View {
        Text(message.text)
            .textSelection(.enabled)
            .padding(10)
            .background(fill, in: RoundedRectangle(cornerRadius: 12))
            .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
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

#Preview {
    ChatView().environment(ChatSession())
}
