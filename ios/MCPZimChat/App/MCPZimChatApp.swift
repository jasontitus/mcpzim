// SPDX-License-Identifier: MIT

import SwiftUI

@main
struct MCPZimChatApp: App {
    @State private var session = ChatSession()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(session)
        }
    }
}
