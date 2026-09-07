// SPDX-License-Identifier: MIT
import Foundation

/// Resolves deliberate choices against the cards on the latest reply, in
/// display order. The resulting turn is identical to tapping that card.
/// Tool-result candidates and historical place lists are not visible offers.
public enum ConversationSuggestionSelection {
    public static func prompt(
        for text: String, suggestions: [DiscoveryThread]
    ) -> String? {
        selection(for: text, suggestions: suggestions).map(action)
    }

    public static func action(_ thread: DiscoveryThread) -> String {
        thread.prompt ?? "tell me about \(thread.label)"
    }

    /// Explicit topic choices retain the archive that was actually sampled.
    public static func articleIntent(for thread: DiscoveryThread) -> DirectIntent? {
        guard thread.source == .wikilink, let zim = thread.zim, !zim.isEmpty
        else { return nil }
        return DirectIntent(toolName: "article_overview", args: [
            "title": .string(thread.label), "zim": .string(zim),
        ])
    }

    /// A current section choice navigates the source directly. A stale card,
    /// archive mismatch, missing heading, or empty section is not a license
    /// to answer from another article. Include nested sections in source
    /// order, stopping at the next peer/ancestor heading.
    public static func sections(
        for thread: DiscoveryThread, articleTitle: String, zim: String?,
        sections: [ArticleSection]
    ) -> [ArticleSection] {
        guard thread.source == .section,
              thread.articleTitle == articleTitle, thread.zim == zim,
              let heading = thread.sectionTitle, !heading.isEmpty,
              let index = sections.firstIndex(where: { $0.title == heading })
        else { return [] }
        let level = sections[index].level
        var result = [sections[index]]
        for section in sections.dropFirst(index + 1) {
            guard section.level > level else { break }
            result.append(section)
        }
        return result.filter { !SourceBoundAnswer.clean($0.text).isEmpty }
    }

    public static func voiceCue(_ threads: [DiscoveryThread]) -> String? {
        let useful = threads.filter {
            $0.label != "Read full article" && $0.label != "Another topic"
        }
        guard let first = useful.first ?? threads.first else { return nil }
        let continuation = first.source == .nearbyPlace
            ? "another nearby lead" : "more topics"
        // Speak the exact card action so custom section questions and map
        // commands round-trip through the same selection resolver as taps.
        return "You can say “\(action(first)),” “\(continuation),” or ask your own follow-up."
    }

    public static func selection(
        for text: String, suggestions: [DiscoveryThread]
    ) -> DiscoveryThread? {
        guard !suggestions.isEmpty else { return nil }
        func normalized(_ value: String) -> String {
            value.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty }.joined(separator: " ")
        }
        let query = normalized(text)
        guard !query.isEmpty else { return nil }

        // An acknowledgement is only a choice when there is one offer.
        let affirmatives: Set<String> = [
            "yes", "yes please", "yeah", "yep", "sure", "okay", "ok",
            "please do", "sounds good", "go for it",
        ]
        if suggestions.count == 1, affirmatives.contains(query) {
            return suggestions[0]
        }

        // Exact labels are safe; keyword overlap would swallow new questions
        // such as "why did the second world war start?".
        let prefixes = ["", "the ", "tell me about ", "tell me about the ",
                        "let s explore ", "yes ", "yes please "]
        let matches = suggestions.filter { thread in
            let label = normalized(thread.label)
            return query == normalized(action(thread))
                || (!label.isEmpty && prefixes.contains { query == $0 + label })
        }
        if matches.count == 1 { return matches[0] }
        if matches.count > 1 { return nil }

        // Explicit suggestion/option selectors avoid stealing "the second
        // one" from a numbered map/search result shown in the same reply.
        let ordinals = ["first", "second", "third", "fourth", "fifth", "sixth"]
        for (index, _) in suggestions.enumerated() {
            var selectors = ["suggestion \(index + 1)", "option \(index + 1)"]
            if index < ordinals.count {
                for noun in ["suggestion", "option"] {
                    selectors += ["\(ordinals[index]) \(noun)",
                                  "the \(ordinals[index]) \(noun)"]
                }
            }
            if selectors.contains(query) { return suggestions[index] }
        }
        return nil
    }
}
