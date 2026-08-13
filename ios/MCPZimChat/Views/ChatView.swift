// SPDX-License-Identifier: MIT

import SwiftUI
import MCPZimKit
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

    /// Stable id for the zero-height spacer we pin at the tail of the
    /// `LazyVStack` so `proxy.scrollTo(chatBottomAnchorId)` always
    /// lands on the bottom regardless of which row is currently last.
    private let chatBottomAnchorId = "mcpzim-chat-bottom"

    /// Snapshot of every signal that should pin the chat to the bottom.
    /// Watching only `last?.text` missed: (a) appending a fresh assistant
    /// message placeholder (text still ""), (b) new tool-call traces
    /// arriving on an existing assistant row, (c) the ThinkingIndicator
    /// appearing/disappearing, (d) `isGenerating` transitions that resize
    /// neighbouring rows.
    private struct ScrollSignal: Equatable {
        var messageCount: Int
        var lastText: String?
        var lastToolCallCount: Int?
        var isGenerating: Bool
        var showThinking: Bool
    }

    private var scrollSignal: ScrollSignal {
        ScrollSignal(
            messageCount: session.messages.count,
            lastText: session.messages.last?.text,
            lastToolCallCount: session.messages.last?.toolCalls.count,
            isGenerating: session.isGenerating,
            showThinking: showThinkingIndicator
        )
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool = true) {
        // Dispatch to the next runloop — when the caller just mutated
        // `session.messages` or toggled `showThinkingIndicator`,
        // SwiftUI hasn't measured the new layout yet, so a synchronous
        // `scrollTo` lands on the pre-change geometry. One tick later
        // the anchor has its real y-offset and the scroll hits bottom.
        DispatchQueue.main.async {
            if animated {
                withAnimation(.easeOut(duration: 0.15)) {
                    proxy.scrollTo(chatBottomAnchorId, anchor: .bottom)
                }
            } else {
                proxy.scrollTo(chatBottomAnchorId, anchor: .bottom)
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Only show the status banner when there's something to
            // report (loading, error, not-yet-loaded). Once the model
            // is ready the chat gets the full vertical space.
            if needsStatusBanner {
                modelBanner
                Divider()
            }
            // The mode persists across launches, so it has to be visible:
            // otherwise a user who once said "let's talk local" sees later
            // answers change with no clue why.
            ConversationModeBar()
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if session.messages.isEmpty { emptyState }
                        // Compute the latest-assistant / last-message ids
                        // once per pass and hand them to each row as stored
                        // `let`s. If every row read `session.messages`
                        // itself, all instantiated assistant rows would
                        // re-run `body` on each 10 Hz streaming push
                        // instead of just the streaming one.
                        let latestAssistantId = session.messages
                            .last(where: { $0.role == .assistant })?.id
                        let lastMessageId = session.messages.last?.id
                        ForEach(session.messages) { m in
                            MessageRow(
                                message: m,
                                isLatestAssistant: m.id == latestAssistantId,
                                isLastMessage: m.id == lastMessageId
                            ).id(m.id)
                        }
                        // Claude-style "thinking" indicator. Shows
                        // while the session is generating and the
                        // last assistant message has no visible
                        // text yet — either the placeholder hasn't
                        // been appended, or the raw text contains
                        // only `<tool_call>…</tool_call>` markup
                        // that gets stripped for display (mid tool
                        // round-trip). Auto-hides once real prose
                        // arrives so it doesn't clash with the
                        // streaming bubble.
                        if session.isGenerating, showThinkingIndicator {
                            ThinkingIndicator()
                                .id("thinking")
                                .transition(.opacity)
                        }
                        // Invisible anchor pinned to the bottom of the
                        // stack so `proxy.scrollTo(chatBottomAnchorId)`
                        // always lands on the absolute tail, regardless
                        // of whether the ThinkingIndicator or the last
                        // message row happens to be the final rendered
                        // child at the moment of the scroll.
                        Color.clear
                            .frame(height: 1)
                            .id(chatBottomAnchorId)
                    }
                    .animation(.easeInOut(duration: 0.2), value: session.isGenerating)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
                // Drag-scrolling the chat also dismisses the keyboard
                // so the debug pane + composer area become reachable
                // without manually tapping "Done".
                .scrollDismissesKeyboard(.immediately)
                // Multi-signal auto-scroll — any change in the signal
                // snapshot pins the view to the bottom anchor so layout
                // changes above can't backscroll the visible content.
                // One watcher on the combined snapshot (instead of five
                // stacked `onChange`s) means a tick that moves several
                // signals at once — new message + trace + indicator
                // flip — schedules a single scroll, not three. While
                // generating the scroll is non-animated: the text
                // signal fires at 10 Hz, and ten interrupted ease-out
                // animations per second forced a re-layout down to the
                // bottom anchor on every push.
                .onChange(of: scrollSignal) { _, _ in
                    scrollToBottom(proxy, animated: !session.isGenerating)
                }
                .onAppear { scrollToBottom(proxy, animated: false) }
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

    /// True when the last message either isn't an assistant turn yet, or
    /// is an assistant turn whose *rendered* text is empty — i.e. the raw
    /// buffer contains only `<tool_call>…</tool_call>` markup that
    /// `MessageRow.displayText` strips. Checking the raw `text.isEmpty`
    /// wasn't enough: a tool-call round-trip leaves raw-non-empty but
    /// display-empty, so without this we'd hide the dots and show a
    /// blank gray bubble instead.
    private var showThinkingIndicator: Bool {
        guard let last = session.messages.last else { return true }
        if last.role != .assistant { return true }
        return MessageRow.displayText(last.text, role: .assistant).isEmpty
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
                // Previously "What's in my library?" — the model read
                // "library" as a nearby POI and routed to `near_places`,
                // returning nearby library buildings. Phrase unambiguously
                // as an archive/ZIM inventory so `list_libraries` fires.
                Text("• What offline archives do I have?")
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
        VStack(spacing: 0) {
            if session.promptOptimizationState.isActive {
                promptOptimizationBanner
                    .padding(.horizontal, 10)
                    .padding(.top, 8)
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message", text: $draft, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                    .focused($inputFocused)
                    .submitLabel(.send)
                    .onSubmit(send)
                    // SwiftUI's `.onSubmit(send)` doesn't fire when
                    // `TextField(..., axis: .vertical)` is set — the return
                    // key on the software keyboard inserts a newline
                    // instead, even with `.submitLabel(.send)` advertising
                    // the blue send arrow. Compensate by watching for the
                    // newline ourselves: if `draft` ever ends with `\n`,
                    // treat it as the user tapping Send, drop the
                    // newline, and submit. Matches how iOS Messages +
                    // most chat apps behave. `send()` clears `draft` so
                    // we don't re-fire on the same keystroke.
                    .onChange(of: draft) { _, newValue in
                        if newValue.hasSuffix("\n") {
                            draft = String(newValue.dropLast())
                            let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty, !session.isGenerating else { return }
                            send()
                        }
                    }
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
                if session.isGenerating {
                    Button {
                        session.stopGeneration()
                    } label: {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 28))
                            .foregroundStyle(.red)
                    }
                    .accessibilityLabel("Stop response")
                } else {
                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 28))
                    }
                    .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(10)
        }
        .background(.bar)
    }

    private var promptOptimizationBanner: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: "bolt.horizontal.circle.fill")
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text(promptOptimizationTitle)
                        .font(.caption.weight(.semibold))
                    Text(promptOptimizationDetail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                ProgressView()
                    .controlSize(.small)
            }
            if case .building(let progress) =
                session.promptOptimizationState
            {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .accessibilityValue("\(Int(progress * 100)) percent")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color.accentColor.opacity(0.09), in: RoundedRectangle(
            cornerRadius: 10, style: .continuous))
        .accessibilityElement(children: .combine)
    }

    private var promptOptimizationTitle: String {
        if session.isGenerating {
            return "Finishing model optimization for this answer"
        }
        switch session.promptOptimizationState {
        case .checking: return "Checking saved model optimization"
        case .restoring: return "Restoring faster first replies"
        case .building: return "Optimizing first replies in the background"
        case .idle, .ready, .failed: return "Model ready"
        }
    }

    private var promptOptimizationDetail: String {
        if session.isGenerating {
            return "This one-time work will make later questions start faster."
        }
        return "You can keep typing and send whenever you’re ready."
    }

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !session.isGenerating else { return }
        draft = ""
        // Retract the keyboard after send — the reply often includes a
        // route map / hero image that covers the top 2/3 of the screen,
        // and touches on the WKWebView get captured by MapLibre before
        // `.scrollDismissesKeyboard(.immediately)` on the outer ScrollView
        // can fire. User can tap the composer to bring it back.
        inputFocused = false
        session.send(text)
    }
}

private struct MessageRow: View {
    let message: ChatMessage
    /// Precomputed in `ChatView`'s ForEach so this row never reads
    /// `session.messages` — an `@Observable` property rewritten at
    /// 10 Hz during streaming, which would re-run every instantiated
    /// row's `body` per push instead of just the streaming one.
    let isLatestAssistant: Bool
    /// True when this row is the very last message in the transcript
    /// (assistant or otherwise) — used to suppress the elapsed label
    /// on the still-streaming bubble.
    let isLastMessage: Bool
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
            // A live WKWebView + MapLibre instance costs roughly 300–500
            // MB of Metal buffers. Keeping one alive per scroll-past
            // route/places trace blows the 6 GB jetsam cap under a
            // long session. Restrict the inline webview to the newest
            // assistant message — older ones collapse to a static
            // "Open map" chip that full-screens on demand.
            VStack(alignment: .leading, spacing: 6) {
                // Map first — feels natural for routing answers, and
                // the streaming text grows downward below it instead
                // of pushing the map around as new sentences arrive.
                ForEach(message.toolCalls) { trace in
                    if TraceKindCache.hasRoute(trace) {
                        if isLatestAssistant {
                            RouteWebView(trace: trace)
                        } else {
                            MapPlaceholder(label: "Route map", systemImage: "map") {
                                // No-op in the stub; the user has no
                                // programmatic hook without rehydrating
                                // the full RouteWebView state anyway.
                            }
                        }
                    } else if TraceKindCache.hasPlaces(trace) {
                        // Nearby-tool results carry a list of geocoded
                        // places — render them as pins on the map with
                        // a coverage-radius ring so the user sees the
                        // spatial distribution, not just a prose list.
                        if isLatestAssistant {
                            PlacesWebView(trace: trace)
                        } else {
                            MapPlaceholder(label: "Places on map", systemImage: "mappin.and.ellipse") { }
                        }
                    } else if Self.traceHasArticle(trace) {
                        // Any tool call that named a specific
                        // article — full fetch, section pull, or
                        // section list — is a signal that the
                        // article is load-bearing for the reply.
                        // Surface its hero image / video — but ONLY on
                        // the newest assistant message: HeroMediaView
                        // mounts a live WKWebView per spec, and
                        // scrolling back through a session with N
                        // article traces resurrected N webviews — the
                        // same jetsam blowup the route/places guards
                        // above exist to prevent. Older messages
                        // simply collapse the hero.
                        if isLatestAssistant {
                            HeroMediaView(trace: trace)
                        }
                    }
                }
                // Skip the assistant bubble entirely while there's no
                // visible text yet. We check the *displayed* text (after
                // stripping tool-call markup) rather than the raw
                // `message.text`, because during a tool round-trip the
                // raw text is full of `<tool_call>…</tool_call>` markers
                // that `displayText` removes — which would leave us
                // drawing a full-padding gray capsule around an empty
                // Text, stacked above the ThinkingIndicator.
                let displayed = Self.displayText(message.text, role: .assistant)
                if !displayed.isEmpty {
                    ZStack(alignment: .topTrailing) {
                        bubble(fill: Color.gray.opacity(0.15), displayed: displayed)
                        copyButton.padding(6)
                    }
                }
                if !userFacingSources.isEmpty {
                    provenanceChips(userFacingSources)
                }
                // Tappable "where to go next" chips — only on the newest
                // assistant message (older offers are stale once the focus
                // has moved on). A tap dispatches the pick via ChatSession.
                if isLatestAssistant, !message.suggestions.isEmpty {
                    suggestionChips(message.suggestions)
                }
                if let elapsed = message.elapsed,
                   !displayed.isEmpty,
                   !session.isGenerating || !isLastMessage
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

    /// Horizontal row of tappable drift-thread chips under the latest reply.
    /// Each chip dispatches `ChatSession.selectSuggestion`, which re-opens the
    /// topic exactly as typing it would. Disabled mid-generation so a tap
    /// can't race an in-flight turn.
    @ViewBuilder
    private func suggestionChips(_ threads: [DiscoveryThread]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(threads, id: \.self) { thread in
                    Button {
                        session.selectSuggestion(thread)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.turn.down.right").font(.caption2)
                            Text(thread.label).lineLimit(1)
                        }
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(Capsule().fill(Color.accentColor.opacity(0.12)))
                        .overlay(Capsule().strokeBorder(Color.accentColor.opacity(0.35)))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                }
            }
            .padding(.horizontal, 10)
            .padding(.top, 2)
        }
        .disabled(session.isGenerating)
    }

    /// Grounding is part of the product answer, not a debug-only tool trace.
    /// Wikipedia responses show their article/section; map/place responses
    /// show StreetZIM. This makes model preknowledge distinguishable from
    /// offline-library evidence at a glance.
    private var userFacingSources: [GroundingSource] {
        if !message.groundingSources.isEmpty { return message.groundingSources }
        let streetTools: Set<String> = [
            "near_named_place", "near_places", "nearby_stories",
            "nearby_stories_at_place", "route_from_places",
            "plan_driving_route", "what_is_here", "locate",
        ]
        if message.toolCalls.contains(where: {
            $0.succeeded && streetTools.contains($0.name)
        }) {
            return [GroundingSource(
                kind: .streetZIM,
                title: "Offline OpenStreetMap")]
        }
        return []
    }

    @ViewBuilder
    private func provenanceChips(_ sources: [GroundingSource]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                // Deterministic per-sentence verification verdict, when the
                // grounded path computed one. "N/M matched" is the honest
                // summary; any unsupported sentence gets its own warning
                // chip so trained-data leakage is visible, not blended in.
                let attribs = message.sentenceAttributions
                let claims = attribs.filter { $0.support < 1.0 || $0.isSupported }
                if !attribs.isEmpty {
                    let unsupported = attribs.filter { !$0.isSupported && $0.support < 1.0 }
                    if unsupported.isEmpty {
                        chip(icon: "checkmark.seal.fill",
                             label: "All \(claims.count) statement\(claims.count == 1 ? "" : "s") matched sources",
                             tint: .green)
                    } else {
                        ForEach(Array(unsupported.prefix(2).enumerated()), id: \.offset) { _, a in
                            chip(icon: "exclamationmark.triangle.fill",
                                 label: "Not in sources: “\(String(a.sentence.prefix(48)))…”",
                                 tint: .orange)
                        }
                        if unsupported.count > 2 {
                            chip(icon: "exclamationmark.triangle.fill",
                                 label: "+\(unsupported.count - 2) more unverified",
                                 tint: .orange)
                        }
                    }
                }
                ForEach(Array(sources.enumerated()), id: \.offset) { _, source in
                    HStack(spacing: 4) {
                        Image(systemName: source.kind == .wikipedia
                            ? "book.closed.fill" : "map.fill")
                        Text(sourceLabel(source)).lineLimit(1)
                    }
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(Color.secondary.opacity(0.10)))
                    .accessibilityLabel("Source: \(sourceLabel(source))")
                }
            }
            .padding(.horizontal, 10)
        }
    }

    @ViewBuilder
    private func chip(icon: String, label: String, tint: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
            Text(label).lineLimit(1)
        }
        .font(.caption2.weight(.medium))
        .foregroundStyle(tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Capsule().fill(tint.opacity(0.12)))
    }

    private func sourceLabel(_ source: GroundingSource) -> String {
        if let section = source.section, !section.isEmpty {
            return "\(source.kind.rawValue) · \(source.title) › \(section)"
        }
        return "\(source.kind.rawValue) · \(source.title)"
    }

    /// `displayed` lets the assistant branch pass in the stripped text it
    /// already computed, so a single body pass doesn't run the
    /// `displayText` pipeline twice over the same string.
    @ViewBuilder
    private func bubble(fill: Color, displayed: String? = nil) -> some View {
        let displayed = displayed ?? Self.displayText(message.text, role: message.role)
        Group {
            if message.role == .assistant {
                MarkdownMessageText(source: displayed)
                    // Leave the copy affordance clear of a heading or the
                    // first line of prose in the top-right corner.
                    .padding(.trailing, 24)
            } else {
                Text(displayed)
            }
        }
            .textSelection(.enabled)
            .padding(10)
            .background(fill, in: RoundedRectangle(cornerRadius: 12))
            .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }

    /// Tiny FIFO memo over the strip pipeline: during streaming the
    /// thinking-indicator check and the row body both render the SAME
    /// growing text at ~10 Hz, so without this the full multi-regex
    /// pipeline ran two to three times per UI push on the main thread.
    ///
    /// PI review 2026-08-13 (perf #2): the key used to be
    /// `(raw.hashValue, raw.count)`. Both halves are *full traversals* of
    /// the message — `hashValue` SipHashes every byte, `count` walks every
    /// grapheme cluster — spent looking up a value that during streaming
    /// can never be present: the key fingerprints the exact text, and a
    /// push is by definition a text that just changed. So the first call
    /// of every push paid 2×O(n) to miss by construction, and only the
    /// duplicate calls *within* one push could ever hit. Key on the O(1)
    /// UTF-8 length and confirm with `==` (which short-circuits on shared
    /// storage — the repeat calls in a push pass literally the same
    /// String instance), so the key costs nothing and settled messages
    /// re-rendered on scroll hit too.
    @MainActor
    private static var displayTextMemo: [(utf8Count: Int, raw: String, value: String)] = []

    @MainActor
    fileprivate static func displayText(_ raw: String, role: ChatMessage.Role) -> String {
        guard role == .assistant else { return raw }
        let utf8Count = raw.utf8.count
        if let hit = displayTextMemo.first(where: {
            $0.utf8Count == utf8Count && $0.raw == raw
        }) {
            return hit.value
        }
        let value = AssistantMarkupStripper.displayText(raw)
        displayTextMemo.append((utf8Count, raw, value))
        if displayTextMemo.count > 16 {
            displayTextMemo.removeFirst(displayTextMemo.count - 16)
        }
        return value
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

/// Strips leftover tool-call markup from the assistant's visible prose.
/// The parser catches well-formed blocks, but during streaming we briefly
/// see the half-emitted opener (e.g.
/// `<|tool_call>call:search{query:<|"|>pizza`) before the closing sentinel
/// arrives. Nuke anything from the first opener to the end of the string
/// so the chat never flashes raw template text.
///
/// Lifted out of `MessageRow` (which is file-private, and so untestable)
/// when perf #2 of the 2026-08-13 review split this into a bounded
/// streaming path plus the unbounded pipeline it must agree with — see
/// `AssistantMarkupStripperTests`, which pins the two together.
enum AssistantMarkupStripper {
    // Compiled once — `replacingOccurrences(options: .regularExpression)`
    // and `range(of:options:.regularExpression)` recompile their ICU
    // pattern on every call, and `displayText` runs 2–3× per row per
    // streaming push. (These are already `static let`, not the per-call
    // `try? NSRegularExpression(...)` idiom, so MCPZimKit's `RegexCache`
    // has nothing to add here.)

    /// Closed blocks (all the canonical spellings) — stripped wholesale.
    private static let closedBlockRegexes: [NSRegularExpression] = [
        #"<\|tool_call\|?>[\s\S]*?<tool_call\|>"#,
        #"<tool_call>[\s\S]*?</tool_call>"#,
        #"<\|tool_response\|?>[\s\S]*?<tool_response\|>"#,
        // Reasoning the FT sometimes emits — strip closed <think>…</think>
        // so it isn't shown (it's scrubbed from the final text anyway, but
        // without this it flashes on screen mid-stream then redraws away).
        #"<think>[\s\S]*?</think>"#,
    ].map { try! NSRegularExpression(pattern: $0) }

    /// Stray openers — everything from the first match to end-of-string
    /// gets masked. Broad prefixes so a 4-byte token like "<|to" or
    /// "<|tool_c" in-flight is also hidden.
    private static let strayOpenerRegexes: [NSRegularExpression] = [
        #"<\|?tool[_a-z]*"#,
        #"<tool[_a-z]*"#,
    ].map { try! NSRegularExpression(pattern: $0) }

    /// How far past the buffer's final `<` a pass in `stripMarkup` can
    /// possibly reach. The longest fixed sentinel is `<tool_response|>`
    /// at 16 bytes; the `[\s\S]*?` spans are unbounded but always
    /// terminate on a closer that itself starts with `<`, so they cannot
    /// end later than 16 bytes past the last `<` either. 32 doubles that.
    private static let sentinelWindow = 32

    /// Streaming entry point.
    ///
    /// PI review 2026-08-13 (perf #2): `stripMarkup` rescans the *entire*
    /// accumulated reply, and `MessageRow` runs it on every 10 Hz push, so
    /// a `narrate_article` answer of tens of KB burned O(n²) bytes of ICU
    /// matching on the main thread while it streamed. Every pattern above
    /// begins with `<`, so nothing after the buffer's final `<` can take
    /// part in any of them: run the passes over that bounded head only,
    /// then carry the (usually enormous) marker-free tail through verbatim
    /// — or drop it wholesale when a pass masked to end-of-string. For the
    /// shape that actually streams — a closed `<|tool_call>…<tool_call|>`
    /// block followed by a long narration — the regex work stops scaling
    /// with the narration and only the byte scan below stays linear.
    static func displayText(_ raw: String) -> String {
        let utf8 = raw.utf8
        guard let split = headEnd(utf8) else {
            // No `<` in the buffer at all, so every pass is provably a
            // no-op and only the trim survives.
            return raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let head = stripMarkup(String(decoding: utf8[..<split], as: UTF8.self))
        guard !head.droppedTail else {
            return head.text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return (head.text + String(decoding: utf8[split...], as: UTF8.self))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// End of the region that can still hold a sentinel: the last `<` plus
    /// `sentinelWindow` bytes, walked forward off any UTF-8 continuation
    /// byte so both halves decode (and so the split can never land inside
    /// a combining sequence close enough to a sentinel to change how
    /// `range(of:)` clusters it). `nil` when the buffer holds no `<`.
    private static func headEnd(_ utf8: String.UTF8View) -> String.Index? {
        var cursor = utf8.endIndex
        var lastAngle: String.Index?
        while cursor > utf8.startIndex {
            utf8.formIndex(before: &cursor)
            if utf8[cursor] == UInt8(ascii: "<") {
                lastAngle = cursor
                break
            }
        }
        guard var end = lastAngle else { return nil }
        var slack = sentinelWindow
        while slack > 0, end < utf8.endIndex {
            utf8.formIndex(after: &end)
            slack -= 1
        }
        while end < utf8.endIndex, utf8[end] & 0xC0 == 0x80 {
            utf8.formIndex(after: &end)
        }
        return end
    }

    /// The pipeline itself, unbounded and untrimmed. `droppedTail` reports
    /// that a pass masked from some offset to end-of-string, which is how
    /// `displayText` knows the tail it held back is gone too.
    static func stripMarkup(_ raw: String) -> (text: String, droppedTail: Bool) {
        var t = raw
        var droppedTail = false
        // Closed blocks (all the canonical spellings).
        for re in closedBlockRegexes {
            let range = NSRange(t.startIndex..., in: t)
            if re.firstMatch(in: t, range: range) != nil {
                t = re.stringByReplacingMatches(in: t, range: range, withTemplate: "")
            }
        }
        // Qwen 3.x may put the opening <think> in the prompt and generate
        // only `scratchpad</think>answer`. Do not display the scratchpad or
        // the raw closer when that template-injected-opener shape appears.
        if let r = t.range(of: "</think>", options: .backwards) {
            t = String(t[r.upperBound...])
        }
        // Any stray opener — drop from there to end of string. During
        // streaming this hides the partially-arrived tool-call until
        // the parser finishes; after parse the opener shouldn't remain,
        // but if the model went off-format we still don't show raw
        // template text.
        for re in strayOpenerRegexes {
            if let m = re.firstMatch(in: t, range: NSRange(t.startIndex..., in: t)),
               let r = Range(m.range, in: t)
            {
                t = String(t[..<r.lowerBound])
                droppedTail = true
            }
        }
        // Unclosed <think> mid-stream: hide from the opener to end until the
        // closing tag arrives (then the closed pattern above strips the pair).
        if let r = t.range(of: "<think") {
            t = String(t[..<r.lowerBound])
            droppedTail = true
        }
        // Drop any lingering sentinel scraps.
        for lit in ["<tool_call|>", "<tool_response|>", "<|\"|>", "<|\""] {
            t = t.replacingOccurrences(of: lit, with: "")
        }
        return (t, droppedTail)
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

/// Route/places classification JSON-parses the trace's full `rawResult`
/// (10–100 KB for map-bearing traces). Traces are immutable after
/// creation, so classify each id once and memoize — without this every
/// `MessageRow.body` re-run during streaming re-parses the payload at
/// 10 Hz. Main-actor only: called exclusively from `body`.
@MainActor
private enum TraceKindCache {
    private static var route: [UUID: Bool] = [:]
    private static var places: [UUID: Bool] = [:]

    static func hasRoute(_ trace: ToolCallTrace) -> Bool {
        if let hit = route[trace.id] { return hit }
        let v = traceHasRoute(trace)
        trimIfNeeded(&route)
        route[trace.id] = v
        return v
    }

    static func hasPlaces(_ trace: ToolCallTrace) -> Bool {
        if let hit = places[trace.id] { return hit }
        let v = traceHasPlaces(trace)
        trimIfNeeded(&places)
        places[trace.id] = v
        return v
    }

    /// Traces are bounded per conversation but ids never repeat across
    /// resets, so cap the map. A wholesale drop on the (rare) crossing
    /// beats tracking LRU order for a dictionary of Bools.
    private static func trimIfNeeded(_ cache: inout [UUID: Bool]) {
        if cache.count >= 1024 { cache.removeAll(keepingCapacity: true) }
    }
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

/// Non-interactive stand-in shown for older-message route/places traces
/// so scrolling back through the chat doesn't resurrect heavyweight
/// WKWebView + MapLibre instances. Each live webview holds ~300–500 MB
/// of Metal buffers; stacking them across a dozen tool calls reliably
/// trips the iPhone's jetsam cap mid-generation. Freshest message still
/// shows the live map — this placeholder is only for history.
private struct MapPlaceholder: View {
    let label: String
    let systemImage: String
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 14, weight: .semibold))
                Text(label)
                    .font(.system(size: 13, weight: .semibold))
                Spacer(minLength: 0)
                Text("Open in newest message")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color.gray.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .disabled(true)
    }
}

/// Claude-style "thinking" indicator. Three dots that fade in and
/// out in sequence while the model is prefilling / sampling an empty
/// response. Mirrors the visual weight of a normal assistant bubble
/// so the layout doesn't jump when the real stream takes over.
struct ThinkingIndicator: View {
    /// Seconds per full sine cycle. Controls how fast the "walking dots"
    /// travel. 1.2 s feels energetic without being jittery.
    private let cycle: Double = 1.2

    var body: some View {
        // TimelineView re-evaluates its body on the display link so we
        // can derive each dot's opacity from the wall clock, no manual
        // @State / withAnimation dance. Using `.animation` schedule
        // lets SwiftUI pick the right redraw cadence (~60 Hz).
        TimelineView(.animation) { ctx in
            let t = ctx.date.timeIntervalSinceReferenceDate
            HStack(spacing: 6) {
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .fill(Color.secondary)
                        .frame(width: 8, height: 8)
                        .opacity(dotOpacity(index: i, at: t))
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.gray.opacity(0.15), in: Capsule())
        }
        .accessibilityLabel("Thinking")
    }

    /// Each dot gets a sine-wave opacity curve offset by 1/3 of the
    /// cycle so they "walk" left-to-right. Range [0.25, 1.0] keeps
    /// the trailing dots visible rather than blinking to invisible.
    private func dotOpacity(index i: Int, at t: Double) -> Double {
        let phase = (t / cycle).truncatingRemainder(dividingBy: 1.0)
        let offset = Double(i) / 3.0
        let wave = (sin((phase + offset) * 2 * .pi) + 1) / 2 // 0…1
        return 0.25 + 0.75 * wave
    }
}
