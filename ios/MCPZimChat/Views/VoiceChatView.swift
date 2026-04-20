// SPDX-License-Identifier: MIT
//
// Sheet that hosts the hands-free voice loop. The view itself is
// thin — all the orchestration lives in `VoiceChatController`.

import SwiftUI

struct VoiceChatView: View {
    @Environment(ChatSession.self) private var session
    @Environment(\.dismiss) private var dismiss
    @State private var controller: VoiceChatController?

    var body: some View {
        VStack(spacing: 24) {
            header
            Spacer()
            orb
            statusLabel
            transcriptPreview
            Spacer()
            controls
        }
        .padding()
        .task {
            // Build the controller lazily so it inherits the session
            // we got from the environment, then auto-start the loop.
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

    private var header: some View {
        HStack {
            Text("Voice Chat")
                .font(.headline)
            Spacer()
            Button {
                controller?.stop()
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close voice chat")
        }
    }

    @ViewBuilder
    private var orb: some View {
        let level = CGFloat(controller?.inputLevel ?? 0)
        let scale = 1.0 + (isListening ? 0.6 * level : 0)
        ZStack {
            Circle()
                .fill(orbColor.opacity(0.15))
                .frame(width: 220, height: 220)
                .scaleEffect(scale)
                .animation(.easeOut(duration: 0.1), value: level)
            Circle()
                .fill(orbColor.opacity(0.35))
                .frame(width: 140, height: 140)
                .scaleEffect(isThinking ? 0.85 + 0.15 * thinkingPulse : 1.0)
                .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: thinkingPulse)
            Image(systemName: orbIcon)
                .font(.system(size: 56, weight: .semibold))
                .foregroundStyle(orbColor)
        }
        .onAppear { thinkingPulse = 1 }
    }

    @State private var thinkingPulse: Double = 0

    private var statusLabel: some View {
        Text(statusText)
            .font(.subheadline.weight(.medium))
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
    }

    private var transcriptPreview: some View {
        Group {
            if let live = controller?.liveTranscript, !live.isEmpty {
                Text(live)
                    .font(.title3)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .lineLimit(3)
            } else {
                Text(" ").font(.title3)
            }
        }
        .frame(minHeight: 60)
    }

    private var controls: some View {
        HStack(spacing: 16) {
            Button {
                controller?.stop()
                dismiss()
            } label: {
                Label("End", systemImage: "phone.down.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .tint(.red)
            Button {
                controller?.toggle()
            } label: {
                Label(toggleLabel, systemImage: toggleIcon)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
    }

    // MARK: - Derived UI state

    private var state: VoiceChatController.State {
        controller?.state ?? .idle
    }

    private var isListening: Bool {
        if case .listening = state { return true } else { return false }
    }
    private var isThinking: Bool {
        if case .thinking = state { return true } else { return false }
    }

    private var orbIcon: String {
        switch state {
        case .listening: return "waveform"
        case .thinking:  return "ellipsis"
        case .speaking:  return "speaker.wave.2.fill"
        case .error:     return "exclamationmark.triangle.fill"
        case .idle:      return "mic.slash.fill"
        }
    }

    private var orbColor: Color {
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
        case .idle:           return "Tap the mic to start."
        case .listening:      return "Listening… pause when you're done."
        case .thinking:       return "Thinking…"
        case .speaking:       return "Speaking…"
        case .error(let msg): return msg
        }
    }

    private var toggleLabel: String {
        switch state {
        case .idle, .error: return "Start"
        default:            return "Pause"
        }
    }

    private var toggleIcon: String {
        switch state {
        case .idle, .error: return "mic.fill"
        default:            return "pause.fill"
        }
    }
}

#Preview {
    VoiceChatView().environment(ChatSession(autoLoadOnInit: false))
}
