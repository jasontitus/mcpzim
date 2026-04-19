// SPDX-License-Identifier: MIT

import SwiftUI

struct RootView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        NavigationStack {
            ChatView()
                .navigationTitle("MCPZim Chat")
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        NavigationLink { LibraryView() } label: {
                            Image(systemName: "books.vertical")
                        }
                        .accessibilityLabel("Library")
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        ModelPickerView()
                    }
                }
                .task { await session.scanDocumentsFolder() }
        }
    }
}

#Preview {
    RootView().environment(ChatSession())
}
