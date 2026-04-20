// SPDX-License-Identifier: MIT

import SwiftUI

struct RootView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        NavigationStack {
            ChatView()
                .navigationTitle("Zimfo")
                .navigationBarTitleDisplayMode(.inline)
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
                .task {
                    // Sandbox Documents first, then reopen any ZIMs the user
                    // previously picked from Downloads/external folders.
                    await session.scanDocumentsFolder()
                    await session.restoreExternalBookmarks()
                    // Prime location permission + snapshot early so the
                    // very first "directions to X" query has `currentLocation`
                    // baked into the system preamble.
                    LocationFetcher.requestAuthorizationIfNeeded()
                    session.refreshLocationIfStale()
                    // Warm the routing graph + rerank model so the first
                    // real query doesn't pay their ~1.2–3 s load costs.
                    session.prewarmBackgroundCaches()
                }
        }
    }
}

#Preview {
    RootView().environment(ChatSession())
}
