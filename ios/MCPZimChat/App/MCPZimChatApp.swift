// SPDX-License-Identifier: MIT

import SwiftUI

#if os(iOS)
/// Minimal delegate for the background download session: iOS relaunches the
/// app when a `ZimDownloadManager` transfer finishes while we're not running,
/// and hands over a completion handler to call once the session's events are
/// drained. Touching `.shared` here also recreates the URLSession so the
/// pending delegate callbacks have somewhere to land.
final class ZimfoAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        guard identifier == ZimDownloadManager.sessionIdentifier else {
            completionHandler()
            return
        }
        Task { @MainActor in
            ZimDownloadManager.shared.backgroundEventsCompletionHandler = completionHandler
        }
    }
}
#endif

@main
struct MCPZimChatApp: App {
    @State private var session = ChatSession()
    @StateObject private var swarm = ZimSwarmController()
    @Environment(\.scenePhase) private var scenePhase
    #if os(iOS)
    @UIApplicationDelegateAdaptor(ZimfoAppDelegate.self) private var appDelegate
    #endif

    init() {
        AppTelemetry.configure()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(session)
                .environmentObject(swarm)
                .task {
                    // Ship any session that finished before this launch (incl.
                    // the one before a crash). Opt-in + no-op otherwise.
                    DiagnosticsUploader.uploadFinishedLogs(archive: .shared)
                }
                .onChange(of: scenePhase) { _, phase in
                    // Backgrounding is when the current session becomes a
                    // "finished" log worth sending — fires over cellular, so a
                    // walking session lands with no interaction.
                    if phase == .background {
                        DiagnosticsUploader.uploadFinishedLogs(archive: .shared)
                    }
                    // iOS tears down Bonjour listeners/browsers during
                    // suspension; nearby devices reappear only after a
                    // rebuild.
                    if phase == .active {
                        swarm.refreshConnectivity()
                    }
                }
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
