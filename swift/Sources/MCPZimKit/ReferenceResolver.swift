// SPDX-License-Identifier: MIT
//
// Deterministic coreference for conversational follow-ups.
//
// `TOOL_DESIGN.md` is explicit that the on-device 4B model is BAD at
// multi-hop reasoning and at "picking the best hit vs. the first". Resolving
// what "it" / "the second one" / "the older church" refers to is exactly
// that kind of brittle reasoning — so we do it in Swift against the
// `ConversationFocus`, not in the model's head.
//
// Given the raw user turn and the current focus, `resolve` returns a
// `ResolvedReference`:
//   * `binding`        — the specific entity (or list slot) the turn refers
//                        to, or `.none` for a fresh query, or `.ambiguous`
//                        when a descriptive selector matches several items.
//   * `rewrittenQuery` — the user text with the referent substituted in
//                        ("who built it" → "who built Stanford Memorial
//                        Church"), so a downstream fetch hits the right
//                        article.
//   * `isContinuation` — whether this reads as a follow-up at all (drives
//                        whether the router should consult the focus before
//                        its stateless patterns).
//
// Pure text-in / value-out: no UI, no tool dispatch, no ZIM access.

import Foundation

public struct ResolvedReference: Equatable, Sendable {
    public enum Binding: Equatable, Sendable {
        /// Fresh query — no anaphora; the router should use its stateless
        /// patterns / the LLM as today.
        case none
        /// Bound to a specific known entity (a pronoun, an elliptical
        /// subject, or a unique descriptive selector).
        case entity(FocusEntity)
        /// Bound to an open drift thread the assistant just offered — accepted
        /// with a bare "yes"/"sure" or named ("the war it's named after").
        /// Carried as a promoted `FocusEntity` so it dispatches like any other
        /// bound subject (the host needs no new case).
        case thread(FocusEntity)
        /// Bound to a slot in `focus.lastList` ("the second one").
        case listSelection(index: Int, entity: FocusEntity)
        /// A descriptive selector ("the old one") matched multiple list
        /// items — the host should ask which, not guess.
        case ambiguous([FocusEntity])
    }

    public var binding: Binding
    public var rewrittenQuery: String
    public var isContinuation: Bool

    public init(binding: Binding, rewrittenQuery: String, isContinuation: Bool) {
        self.binding = binding
        self.rewrittenQuery = rewrittenQuery
        self.isContinuation = isContinuation
    }

    /// The single entity this reference resolves to, if unambiguous.
    public var boundEntity: FocusEntity? {
        switch binding {
        case .entity(let e): return e
        case .thread(let e): return e
        case .listSelection(_, let e): return e
        case .none, .ambiguous: return nil
        }
    }
}

public enum ReferenceResolver {

    /// Pronouns that point at a single prior subject.
    private static let singularPronouns: Set<String> = [
        "it", "its", "it's", "that", "this", "he", "him", "his",
        "she", "her", "hers", "they", "them", "their", "theirs",
        "those", "these",
    ]

    /// Words that, leading a short turn, signal "keep going on what we were
    /// just discussing" even without an explicit pronoun.
    private static let continuationOpeners: [String] = [
        "and", "but", "so", "then", "also", "plus", "ok", "okay",
        "what about", "how about", "what else", "anything else",
        "more", "tell me more", "more about", "more on", "go on",
        "keep going", "continue", "and what", "and how", "what's next",
        "whats next", "next",
    ]

    /// Bare yes-class turns that ACCEPT an offer. Deliberately excludes
    /// "go on"/"more"/"continue" (those mean "more about the CURRENT subject",
    /// handled by the elliptical path) — only turns with no standalone subject
    /// reading, so "yes"/"sure" bind to the offered thread instead of being
    /// rewritten into a nonsense "yes <subject>" query.
    private static let bareAffirmatives: Set<String> = [
        "yes", "yeah", "yep", "yup", "yes please", "ok sure", "sure",
        "ok", "okay", "okey", "ok then", "alright", "all right",
        "do it", "please do", "go for it", "sounds good", "why not",
        "let's", "lets", "let's hear it", "lets hear it",
    ]

    /// Bare elliptical question stems — a question with no subject of its own,
    /// which therefore inherits the active entity. "who built it" already has
    /// a pronoun; these are the *subjectless* ones ("how old", "when", "why").
    private static let ellipticalStems: [String] = [
        "why", "how", "when", "where", "who", "what",
        "how old", "how big", "how far", "how tall", "how long",
        "how come", "what for", "what year", "by whom",
    ]

    /// Ordinal words → zero-based list index.
    private static let ordinals: [String: Int] = [
        "first": 0, "1st": 0, "one": 0,
        "second": 1, "2nd": 1, "two": 1,
        "third": 2, "3rd": 2, "three": 2,
        "fourth": 3, "4th": 3, "four": 3,
        "fifth": 4, "5th": 4, "five": 4,
    ]

    public static func resolve(
        _ raw: String,
        focus: ConversationFocus
    ) -> ResolvedReference {
        let text = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!"))
        let lower = text.lowercased()
        let words = lower.split(whereSeparator: { $0 == " " }).map(String.init)
        let wordCount = words.count

        let opener = continuationOpeners.first { startsWithWord(lower, $0) }
        let hasPronoun = words.contains { singularPronouns.contains($0) }
        let isShort = wordCount > 0 && wordCount <= 6
        // A turn is a continuation if it's a short opener, carries a pronoun,
        // names a list slot, or is a subjectless elliptical question.
        var isContinuation = (opener != nil) || hasPronoun

        // ---- 0. Accept an offered thread ("yes", "sure") -------------------
        // After the assistant offered "want to hear about X?", a bare
        // affirmative picks the lead drift thread. Gated on an open offer so a
        // stray "ok"/"sure" with nothing offered still falls through to the
        // subject-continuation logic below.
        if !focus.openThreads.isEmpty, bareAffirmatives.contains(lower) {
            let t = focus.openThreads[0]
            return ResolvedReference(
                binding: .thread(t.asEntity(turn: focus.turn)),
                rewrittenQuery: "tell me about \(t.label)",
                isContinuation: true
            )
        }

        // ---- 1. List selection ("the second one", "the other one") --------
        if !focus.lastList.isEmpty {
            if let sel = resolveListSelection(lower, words: words, focus: focus) {
                return ResolvedReference(
                    binding: sel,
                    rewrittenQuery: rewrite(text, with: boundName(of: sel)),
                    isContinuation: true
                )
            }
        }

        // ---- 2. Descriptive selector ("the older one", "the church") ------
        // Match "the <noun>" anywhere in the turn, so "tell me about the
        // cathedral" works, not just turns that literally start with "the".
        if words.contains("the"), !focus.lastList.isEmpty {
            let matches = descriptiveMatches(words, focus: focus)
            if matches.count == 1 {
                return ResolvedReference(
                    binding: .entity(matches[0]),
                    rewrittenQuery: rewrite(text, with: matches[0].name),
                    isContinuation: true
                )
            } else if matches.count > 1 {
                return ResolvedReference(
                    binding: .ambiguous(matches),
                    rewrittenQuery: text,
                    isContinuation: true
                )
            }
            // 0 matches: fall through — "the X" may be a fresh topic.
        }

        // ---- 2a. Locative "there" → most-recent place --------------------
        // "what's near there?", "how far from there?" — anaphoric "there"
        // points at the last place we have coordinates for (a routed
        // destination, a what_is_here spot), so continuationIntent can route
        // from/around it. Gated on a locational cue so "are we there yet?"
        // doesn't bind. ("here" is deictic — left to the GPS-based what_is_here
        // / near-me paths.)
        if words.contains("there") {
            let locational = ["near", "around", "nearby", "close", "how far",
                              "distance", "how long", "walk", "drive",
                              "directions", "route", "get to"]
            if locational.contains(where: { lower.contains($0) }),
               let place = focus.mostRecent(kind: .place) {
                return ResolvedReference(
                    binding: .entity(place),
                    rewrittenQuery: text,
                    isContinuation: true
                )
            }
        }

        // ---- 2b. Named drift thread ("the war", "the bridge it crosses") --
        // When threads are open and there is NO on-screen list to pick from,
        // a content-word hit on a unique thread label binds to that thread.
        // (With a list present, selectors mean list items — handled above; a
        // populated list also makes article-style drift threads unlikely.)
        if focus.lastList.isEmpty, !focus.openThreads.isEmpty,
           let t = matchOpenThread(words: words, focus: focus) {
            return ResolvedReference(
                binding: .thread(t.asEntity(turn: focus.turn)),
                rewrittenQuery: "tell me about \(t.label)",
                isContinuation: true
            )
        }

        // ---- 3. Pronoun → primary entity ----------------------------------
        if hasPronoun, let primary = focus.primaryEntity {
            return ResolvedReference(
                binding: .entity(primary),
                rewrittenQuery: rewrite(text, with: primary.name),
                isContinuation: true
            )
        }

        // ---- 4. Subjectless elliptical question → primary entity ----------
        // "why?", "how old?", "and?" carry no subject of their own. If we have
        // an active entity and the turn is short, bind it.
        if isShort, let primary = focus.primaryEntity {
            let isElliptical = ellipticalStems.contains { startsWithWord(lower, $0) }
                || opener != nil
            // Guard: a short turn that introduces its OWN proper subject
            // ("Marie Curie?") is not elliptical — only bind when the turn is
            // an opener/stem with no trailing proper-noun subject of its own.
            if isElliptical, !introducesOwnSubject(words) {
                isContinuation = true
                return ResolvedReference(
                    binding: .entity(primary),
                    rewrittenQuery: appendSubject(text, primary.name),
                    isContinuation: true
                )
            }
        }

        return ResolvedReference(
            binding: .none,
            rewrittenQuery: text,
            isContinuation: isContinuation
        )
    }

    // MARK: - List selection

    private static func resolveListSelection(
        _ lower: String, words: [String], focus: ConversationFocus
    ) -> ResolvedReference.Binding? {
        let list = focus.lastList

        // "the other one" / "the other" — only well-defined for a 2-item list.
        if lower.contains("the other") {
            guard list.count == 2 else { return nil }
            let primaryKey = focus.primaryEntity?.matchKey
            if let other = list.first(where: { $0.matchKey != primaryKey }) {
                let idx = list.firstIndex(of: other) ?? 1
                return .listSelection(index: idx, entity: other)
            }
            return .listSelection(index: 1, entity: list[1])
        }

        // "the last one".
        if lower.contains("the last"), let last = list.last {
            return .listSelection(index: list.count - 1, entity: last)
        }

        // "number 3" / "#2".
        if let m = firstMatch(lower, pattern: #"(?:number|#)\s*(\d+)"#),
           let n = Int(m), n >= 1, n <= list.count {
            return .listSelection(index: n - 1, entity: list[n - 1])
        }

        // Ordinal word, but only when it's used as a selector ("the second
        // one", "second", "the 2nd") — require the word to be present as a
        // standalone token and the turn to read like a pick, not a fresh
        // topic that merely contains an ordinal ("first world war").
        for (word, idx) in ordinals {
            guard words.contains(word) else { continue }
            let looksLikePick = lower.contains("the \(word)")
                || lower.hasPrefix(word)
                || lower.contains("\(word) one")
            guard looksLikePick, idx < list.count else { continue }
            return .listSelection(index: idx, entity: list[idx])
        }

        return nil
    }

    // MARK: - Descriptive selection

    /// Match "the <descriptor> [one]" against the list by token overlap with
    /// each item's label. "the older church" matches items whose label
    /// contains "church"; the descriptor "older" is a hint we currently use
    /// only to disambiguate when the noun alone is unique.
    private static func descriptiveMatches(
        _ words: [String], focus: ConversationFocus
    ) -> [FocusEntity] {
        // Take the noun phrase that follows the FIRST "the" — that's the
        // descriptor ("...the cathedral", "...the older church"). Drop a
        // trailing "one"/"ones".
        guard let theIdx = words.firstIndex(of: "the") else { return [] }
        var toks = Array(words[(theIdx + 1)...])
        if toks.last == "one" || toks.last == "ones" { toks.removeLast() }
        let content = toks.filter { !stopwords.contains($0) }
        guard !content.isEmpty else { return [] }
        let matches = focus.lastList.filter { item in
            let label = item.name.lowercased()
            return content.contains { tok in
                tok.count >= 3 && label.contains(tok)
            }
        }
        return matches
    }

    private static let stopwords: Set<String> = [
        "the", "a", "an", "one", "ones", "of", "in", "on", "at",
        "older", "newer", "old", "new", "bigger", "smaller", "big",
        "small", "first", "last", "other", "that", "this",
    ]

    // MARK: - Drift-thread selection

    /// Match a turn naming an open drift thread ("the war", "tell me about the
    /// cathedral") against `focus.openThreads` by content-word overlap with
    /// each thread's label (+ optional gloss). Returns the thread only on a
    /// UNIQUE hit — ambiguity falls through. Function/question words are
    /// stripped so elliptical stems ("how old?", "how big?") never spuriously
    /// match a label.
    private static func matchOpenThread(
        words: [String], focus: ConversationFocus
    ) -> DiscoveryThread? {
        var toks = words
        if let i = toks.firstIndex(of: "the") { toks = Array(toks[(i + 1)...]) }
        if toks.last == "one" || toks.last == "ones" { toks.removeLast() }
        let functional: Set<String> = stopwords
            .union(singularPronouns)
            .union(["why", "how", "when", "where", "who", "what", "is", "are",
                    "was", "were", "did", "does", "do", "and", "but", "so",
                    "then", "about", "me", "tell", "more", "ok", "okay",
                    "yes", "sure", "please", "hear", "want", "let's", "lets"])
        let content = toks.filter { !functional.contains($0) && $0.count >= 3 }
        guard !content.isEmpty else { return nil }
        let matches = focus.openThreads.filter { thread in
            let hay = (thread.label + " " + (thread.note ?? "")).lowercased()
            return content.contains { hay.contains($0) }
        }
        return matches.count == 1 ? matches[0] : nil
    }

    // MARK: - Rewriting

    /// Replace the first pronoun token in `text` with `name`. Falls back to
    /// the original text when no pronoun is present (the caller then decides
    /// whether to append).
    private static func rewrite(_ text: String, with name: String) -> String {
        let tokens = text.split(separator: " ", omittingEmptySubsequences: false)
            .map(String.init)
        var replaced = false
        var out: [String] = []
        for tok in tokens {
            let bare = tok.lowercased()
                .trimmingCharacters(in: CharacterSet(charactersIn: ",.;:!?"))
            if !replaced, singularPronouns.contains(bare) {
                out.append(name)
                replaced = true
            } else {
                out.append(tok)
            }
        }
        if replaced { return out.joined(separator: " ") }
        return text
    }

    /// Attach `name` as the subject of a subjectless question:
    /// "how old" → "how old is <name>"; "why" → "why <name>";
    /// "more" → "tell me more about <name>".
    private static func appendSubject(_ text: String, _ name: String) -> String {
        let lower = text.lowercased()
        if lower.hasPrefix("more") || lower.contains("tell me more")
            || lower == "go on" || lower == "keep going" || lower == "continue" {
            return "tell me more about \(name)"
        }
        if lower.hasPrefix("how ") || lower == "how" {
            // "how old" → "how old is X"; keep the user's adjective.
            return "\(text) is \(name)"
        }
        if lower.hasPrefix("when") || lower.hasPrefix("where") {
            return "\(text) is \(name)"
        }
        return "\(text) \(name)"
    }

    // MARK: - Helpers

    private static func boundName(of binding: ResolvedReference.Binding) -> String {
        switch binding {
        case .entity(let e): return e.name
        case .thread(let e): return e.name
        case .listSelection(_, let e): return e.name
        case .none, .ambiguous: return ""
        }
    }

    /// Heuristic: does this short turn carry its own (capitalised, multi-char)
    /// subject, making it a fresh query rather than an elliptical follow-up?
    /// Works off the RAW words — capitalisation is the signal.
    private static func introducesOwnSubject(_ lowerWords: [String]) -> Bool {
        // Operates on lowercased words, so we can't use capitalisation. Use a
        // content heuristic instead: a noun-bearing token that isn't a
        // function/opener word and is 4+ chars suggests a real subject.
        let functional: Set<String> = singularPronouns
            .union(stopwords)
            .union(["why", "how", "when", "where", "who", "what", "is", "are",
                    "was", "were", "did", "does", "do", "and", "but", "so",
                    "then", "about", "me", "tell", "more", "ok", "okay"])
        return lowerWords.contains { !functional.contains($0) && $0.count >= 4 }
    }

    private static func startsWithWord(_ text: String, _ phrase: String) -> Bool {
        if text == phrase { return true }
        return text.hasPrefix(phrase + " ")
    }

    private static func firstMatch(_ text: String, pattern: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let m = re.firstMatch(in: text, range: range), m.numberOfRanges >= 2,
              let r = Range(m.range(at: 1), in: text) else { return nil }
        return String(text[r])
    }
}
