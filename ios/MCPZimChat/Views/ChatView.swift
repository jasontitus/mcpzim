// SPDX-License-Identifier: MIT

import SwiftUI

struct ChatView: View {
    @Environment(ChatSession.self) private var session
    @State private var draft = ""
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
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.thinMaterial)
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

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer()
                bubble(fill: Color.accentColor.opacity(0.12))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                bubble(fill: Color(.secondarySystemBackground))
                ForEach(message.toolCalls) { trace in
                    ToolCallRow(trace: trace)
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
}

private struct ToolCallRow: View {
    let trace: ToolCallTrace
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Button {
                withAnimation { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: trace.succeeded ? "wrench.and.screwdriver" : "exclamationmark.triangle")
                    Text(trace.name).font(.footnote.weight(.semibold))
                    Spacer()
                    Image(systemName: expanded ? "chevron.up" : "chevron.down").font(.caption)
                }
                .foregroundStyle(trace.succeeded ? .primary : .red)
            }
            if expanded {
                VStack(alignment: .leading, spacing: 4) {
                    Text("args: \(trace.arguments)").font(.caption.monospaced())
                    if let err = trace.error {
                        Text("error: \(err)").font(.caption.monospaced()).foregroundStyle(.red)
                    } else {
                        Text(trace.result).font(.caption.monospaced())
                    }
                }
            }
        }
        .padding(8)
        .background(Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
    }
}

#Preview {
    ChatView().environment(ChatSession())
}
