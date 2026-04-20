// SPDX-License-Identifier: MIT
//
// Compact voice-chat UI. Renders as a single-row bar at the bottom of
// the chat — left: state icon (mic / ellipsis / speaker / error),
// middle: live transcript or status, right: End button. All
// orchestration lives in `VoiceChatController`.

import SwiftUI

struct VoiceChatView: View {
    @Environment(ChatSession.self) private var session
    @Environment(\.dismiss) private var dismiss
    @State private var controller: VoiceChatController?

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            stateIcon
            Text(previewText)
                .font(.footnote)
                .lineLimit(2)
                .foregroundStyle(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
            endButton
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .task {
            if controller == nil {
                let c = VoiceChatController(session: session)
                controller = c
                await c.start()
            }
        }
        .onDisappear {
            controller?.stop()
        }
    }

    // MARK: - Subviews

    @ViewBuilder
    private var stateIcon: some View {
        // Tiny colored dot-sized icon — blue/yellow/green/red flags
        // the state without taking vertical space.
        Image(systemName: iconName)
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(iconColor)
            .frame(width: 20, height: 20)
            .symbolEffect(.pulse, options: .repeating, value: isThinking)
    }

    private var endButton: some View {
        Button {
            controller?.stop()
            dismiss()
        } label: {
            Image(systemName: "xmark.circle.fill")
                .font(.title2)
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("End voice chat")
    }

    // MARK: - Derived state

    private var state: VoiceChatController.State { controller?.state ?? .idle }

    private var isListening: Bool {
        if case .listening = state { return true } else { return false }
    }
    private var isThinking: Bool {
        if case .thinking = state { return true } else { return false }
    }

    private var iconName: String {
        switch state {
        case .listening: return "waveform"
        case .thinking:  return "ellipsis"
        case .speaking:  return "speaker.wave.2.fill"
        case .error:     return "exclamationmark.triangle.fill"
        case .idle:      return "mic.slash.fill"
        }
    }

    private var iconColor: Color {
        switch state {
        case .listening: return .accentColor
        case .thinking:  return .yellow
        case .speaking:  return .green
        case .error:     return .red
        case .idle:      return .gray
        }
    }

    private var statusText: String {
        switch state {
        case .idle:           return "IDLE"
        case .listening:      return "LISTENING"
        case .thinking:       return "THINKING"
        case .speaking:       return "SPEAKING"
        case .error:          return "ERROR"
        }
    }

    /// What to show in the big text field next to the icon. During
    /// listening we echo the recognized partial; in other states we
    /// show a short status or the last error.
    private var previewText: String {
        if case .error(let msg) = state { return msg }
        let live = controller?.liveTranscript ?? ""
        if isListening {
            return live.isEmpty ? "Pause when you're done." : live
        }
        if isThinking { return live.isEmpty ? "…" : live }
        if case .speaking = state { return "Playing reply…" }
        return "Tap to start."
    }
}

#Preview {
    VoiceChatView().environment(ChatSession(autoLoadOnInit: false))
}
