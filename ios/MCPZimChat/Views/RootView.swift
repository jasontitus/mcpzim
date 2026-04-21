// SPDX-License-Identifier: MIT

import SwiftUI

struct RootView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        NavigationStack {
            ChatView()
                .navigationTitle("Zimfo")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        Button {
                            session.resetConversation()
                        } label: {
                            Image(systemName: "arrow.counterclockwise")
                        }
                        .accessibilityLabel("New conversation")
                        .disabled(session.messages.isEmpty || session.isGenerating)
                    }
                    ToolbarItem(placement: .primaryAction) {
                        NavigationLink { LibraryView() } label: {
                            Image(systemName: "gearshape")
                        }
                        .accessibilityLabel("Settings")
                    }
                }
                .overlay {
                    SetupOverlayView()
                }
                .task {
                    // Sandbox Documents first, then reopen any ZIMs the user
                    // previously picked from Downloads/external folders.
                    await session.scanDocumentsFolder()
                    await session.restoreExternalBookmarks()
                    // Prime location permission + snapshot early so the
                    // very first "directions to X" query has `currentLocation`
                    // baked into the system preamble.
                    LocationFetcher.requestAuthorizationIfNeeded()
                    LocationFetcher.start() // kick continuous updates ASAP
                    session.refreshLocationIfStale()
                    // Warm the routing graph + rerank model so the first
                    // real query doesn't pay their ~1.2–3 s load costs.
                    session.prewarmBackgroundCaches()
                    // One-time prompt-cache warm (or load from disk).
                    // Blocks send() until complete via setupState.
                    await session.runSetupIfNeeded()
                }
        }
    }
}

/// Blocking overlay shown while the one-time prompt-cache prewarm runs.
/// Dismisses itself once `session.setupState == .ready`.
struct SetupOverlayView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        Group {
            switch session.setupState {
            case .ready:
                EmptyView()
            case .failed(let msg):
                // Don't block forever on error — show a toast-style
                // banner and let the user proceed.
                VStack(spacing: 8) {
                    Text("Setup failed")
                        .font(.headline)
                    Text(msg)
                        .font(.caption)
                        .multilineTextAlignment(.center)
                    Button("Continue anyway") {
                        // no-op — the session remains usable, just
                        // without a pre-warmed cache.
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(20)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                .padding(24)
            case .pending, .running:
                ZStack {
                    Color.black.opacity(0.35).ignoresSafeArea()
                    VStack(spacing: 14) {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .scaleEffect(1.3)
                        Text("Setting things up…")
                            .font(.headline)
                        Text(stageText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 260)
                    }
                    .padding(24)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                }
                .transition(.opacity)
            }
        }
        .animation(.default, value: stateDescription)
    }

    private var stageText: String {
        switch session.setupState {
        case .pending: return "Waiting for the model to load."
        case .running(let stage, _): return stage
        case .ready: return ""
        case .failed: return ""
        }
    }

    private var stateDescription: String {
        switch session.setupState {
        case .pending: return "pending"
        case .running(let s, _): return "running:\(s)"
        case .ready: return "ready"
        case .failed(let m): return "failed:\(m)"
        }
    }
}

#Preview {
    RootView().environment(ChatSession())
}
