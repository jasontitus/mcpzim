// SPDX-License-Identifier: MIT
//
// Always-visible indicator + control for which corpus ambiguous turns
// prefer. The mode PERSISTS across launches, so without something on
// screen a user who once said "let's talk local" has no way to tell why
// later answers changed — the only other surface was buried in Settings.
//
// Deliberately not tabs: tabs imply two conversation histories, and the
// whole point of one thread is that a follow-up can cross corpora ("how
// far is that from here?" right after a Wikipedia answer). This is the
// same at-a-glance awareness without splitting the transcript.

import SwiftUI
import MCPZimKit

struct ConversationModeBar: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        @Bindable var bindable = session
        VStack(spacing: 3) {
            Picker("Answer from", selection: $bindable.conversationMode) {
                Text("Auto").tag(ConversationMode.auto)
                Text("Maps").tag(ConversationMode.local)
                Text("Wikipedia").tag(ConversationMode.encyclopedia)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .accessibilityLabel("Answer ambiguous questions from")
            .accessibilityHint(
                "Auto tries maps first and falls back to Wikipedia. "
                + "Maps stays on your offline maps. Wikipedia stays on articles.")
            // Both affordances, named. The segmented control shows WHICH
            // mode is active but not that it can be tapped, and nothing
            // on screen revealed that voice works at all.
            Text(hint)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    /// Names the way OUT of wherever you are, not a generic blurb — in a
    /// pinned mode the useful phrase is the one that unpins it.
    private var hint: String {
        switch session.conversationMode {
        case .auto:
            return "Tap to pin a source · or say “let’s talk local”"
        case .local:
            return "Maps only · tap above, or say “back to normal”"
        case .encyclopedia:
            return "Articles only · tap above, or say “back to normal”"
        }
    }
}
