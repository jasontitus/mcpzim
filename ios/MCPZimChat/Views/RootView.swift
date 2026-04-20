// SPDX-License-Identifier: MIT

import SwiftUI

struct RootView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        NavigationStack {
            ChatView()
                .navigationTitle("MCPZim Chat")
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        NavigationLink { LibraryView() } label: {
                            Image(systemName: "books.vertical")
                        }
                        .accessibilityLabel("Library")
                    }
                    ToolbarItem(placement: .primaryAction) {
                        ModelPickerView()
                    }
                }
                .task {
                    // Sandbox Documents first, then reopen any ZIMs the user
                    // previously picked from Downloads/external folders.
                    await session.scanDocumentsFolder()
                    await session.restoreExternalBookmarks()
                }
        }
    }
}

#Preview {
    RootView().environment(ChatSession())
}
