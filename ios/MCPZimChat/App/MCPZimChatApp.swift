// SPDX-License-Identifier: MIT

import SwiftUI

@main
struct MCPZimChatApp: App {
    @State private var session = ChatSession()

    init() {
        AppTelemetry.configure()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(session)
        }
        #if os(macOS)
        .commands {
            CommandMenu("Models") {
                Button("Use Bonsai 27B (1-bit)") {
                    Task { await session.select(modelId: "bonsai-27b-q1-gguf") }
                }

                Button {
                    Task {
                        await session.select(
                            modelId: ChatSession.ternaryBonsai27BModelID)
                    }
                } label: {
                    Label(ternaryModelActionTitle, systemImage: ternaryModelActionIcon)
                }
                .disabled(isTernaryModelBusy)

                Divider()

                Menu("DSpark") {
                    Button("Unavailable on Apple Silicon in this release") { }
                        .disabled(true)
                    Button("CUDA serving path only; not installed") { }
                        .disabled(true)
                }
            }
        }
        #endif
    }

    #if os(macOS)
    private var ternaryIsSelected: Bool {
        session.selectedModel.id == ChatSession.ternaryBonsai27BModelID
    }

    private var isTernaryModelBusy: Bool {
        guard ternaryIsSelected else { return false }
        switch session.modelState {
        case .downloading, .loading: return true
        default: return false
        }
    }

    private var ternaryModelActionTitle: String {
        if ternaryIsSelected {
            switch session.modelState {
            case .downloading(let fraction):
                return "Downloading Ternary Bonsai 27B… \(Int(fraction * 100))%"
            case .loading:
                return "Loading Ternary Bonsai 27B…"
            case .ready:
                return "Ternary Bonsai 27B (Selected)"
            case .failed:
                return "Retry Ternary Bonsai 27B"
            case .notLoaded:
                return session.isTernaryBonsai27BCached
                    ? "Load Ternary Bonsai 27B (Downloaded)"
                    : "Download Ternary Bonsai 27B (7.17 GB)…"
            }
        }
        return session.isTernaryBonsai27BCached
            ? "Use Ternary Bonsai 27B (Downloaded)"
            : "Download & Use Ternary Bonsai 27B (7.17 GB)…"
    }

    private var ternaryModelActionIcon: String {
        if ternaryIsSelected, session.modelState.isReady { return "checkmark" }
        return session.isTernaryBonsai27BCached ? "internaldrive" : "arrow.down.circle"
    }
    #endif
}
