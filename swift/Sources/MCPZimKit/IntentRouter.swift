// SPDX-License-Identifier: MIT
//
// Fast-path intent router + one-line places-reply synthesiser.
// Both are pure text-in-text-out functions; living in MCPZimKit (not
// the iOS target) keeps them exercised by `swift test` alongside the
// rest of the kit.
//
// The iOS `ChatSession` wraps these and handles the side effects
// (tool dispatch, state mutation); nothing here touches UI, storage,
// or the model runtime.

import Foundation

/// A resolved fast-path dispatch — tool name + args. The iOS side
/// turns this into an `adapter.dispatch` call and a synthesised
/// assistant caption, all without paying the LLM prefill.
public struct DirectIntent: Equatable, Sendable {
    public let toolName: String
    public let args: [String: AnyJSONValue]

    public init(toolName: String, args: [String: AnyJSONValue]) {
        self.toolName = toolName
        self.args = args
    }

    /// Convenience for hosts that speak the native `[String: Any]`
    /// dict the `MCPToolAdapter.dispatch` signature expects.
    public var anyArgs: [String: Any] {
        args.mapValues { $0.anyValue }
    }
}

/// Small, sendable JSON scalar/container shim. Swift's `[String: Any]`
/// isn't `Sendable` or `Equatable`, which makes `DirectIntent`
/// awkward to compare in tests. This covers everything the router
/// needs to emit (strings, numbers, bool, arrays thereof).
public enum AnyJSONValue: Equatable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([AnyJSONValue])
    case object([String: AnyJSONValue])

    public var anyValue: Any {
        switch self {
        case .string(let s): return s
        case .int(let i):    return i
        case .double(let d): return d
        case .bool(let b):   return b
        case .array(let xs): return xs.map(\.anyValue)
        case .object(let o): return o.mapValues { $0.anyValue }
        }
    }
}

public enum IntentRouter {

    /// Attempt to match the raw user text against one of the fast-path
    /// patterns the LLM would otherwise solve. Returns `nil` on
    /// anything unusual — callers fall back to the LLM loop.
    ///
    /// `currentLocation` is required for the "<X> near me" pattern;
    /// all other patterns are location-independent.
    ///
    /// Default search radius is 5 km (the LLM typically picks 1–5;
    /// without context we pick the generous end so "museums in
    /// Mountain View" finds the 15 in-range instead of the 0
    /// within 1 km).
    public static func classify(
        _ raw: String,
        currentLocation: (lat: Double, lon: Double)? = nil,
        focus: ConversationFocus? = nil
    ) -> DirectIntent? {
        let text = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!"))
        if text.isEmpty { return nil }
        let lower = text.lowercased()
        let defaultRadiusKm: Double = 5

        // Context-aware fast path. When the host supplies a conversation
        // focus and this turn reads as a follow-up that BINDS to a known
        // entity ("who built it", "the second one", "tell me more"), resolve
        // the referent in Swift and fetch the RIGHT thing — instead of
        // falling through to the stateless patterns below (which are
        // context-blind and would mis-handle a pronoun) or to a confused LLM.
        if let focus, !focus.isEmpty {
            if let intent = continuationIntent(raw, focus: focus) {
                return intent
            }
        }

        // "what is here" / "where am I" → what_is_here.
        // No args — the MCP adapter fills lat/lon from the host's GPS
        // fix at dispatch time via `hostStateProvider`. Runs BEFORE
        // `<cat> near me` because "what is around me" would otherwise
        // match that pattern with kind="what is" and then fail for
        // lack of location.
        if matches(lower, pattern:
            #"^(?:what(?:'s|\s+is)\s+here|where\s+am\s+i|what(?:'s|\s+is)\s+(?:around|near)\s+(?:me|here)|what\s+do\s+you\s+see)$"#)
        {
            return DirectIntent(toolName: "what_is_here", args: [:])
        }

        // "<category> near me" / "<category> around here" — use GPS.
        // We match FIRST, then require the location. If the pattern
        // matched but we don't have a location, we must NOT fall
        // through to the generic `<X> in|near|at <Y>` pattern below
        // — it would gladly classify "me" as a place.
        if let m = match(lower, pattern: #"^(.+?)\s+(?:near|around)\s+(me|here)$"#) {
            guard let here = currentLocation else { return nil }
            let kind = singularize(m[0])
            return DirectIntent(toolName: "near_places", args: [
                "lat":       .double(here.lat),
                "lon":       .double(here.lon),
                "kinds":     .array([.string(kind)]),
                "radius_km": .double(defaultRadiusKm)
            ])
        }

        // "where is the nearest <category>" / "nearest <category>" /
        // "closest <category>" / "<category> nearby" — the single-closest
        // phrasings the "<cat> near me" pattern above misses. Resolves to
        // the same GPS-anchored near_places (results come back distance-
        // sorted, so the nearest is first). Real capture 2026-05-30: "Where
        // is the nearest coffee shop?" matched no pattern, fell to the LLM,
        // and the model asked the user for their location despite the GPS
        // preamble. Needs a fix; without one we bail to the LLM (which has
        // the coords in its preamble). The question-word guard stops
        // "what's nearby" / "how far is it" from binding as a category.
        let nearestKind: String? = {
            if let m = match(lower, pattern:
                #"^(?:(?:find|show|get|give)\s+me\s+|i\s+(?:need|want)\s+|i'?d\s+like\s+)?(?:(?:where|what|which)(?:'s|\s+is|\s+are)\s+)?(?:the\s+|my\s+|a\s+)?(?:nearest|closest|nearby)\s+(.+?)(?:\s+(?:to|near|around|from)\s+(?:me|here|us|my\s+location))?$"#)
            { return m[0] }
            if let m = match(lower, pattern:
                #"^(?:is\s+there\s+(?:a|an|any)\s+|are\s+there\s+(?:any\s+)?|any\s+|find\s+(?:me\s+)?(?:a|an|the)?\s*)?(.+?)\s+(?:nearby|near\s?by|close\s+by|around\s+here|near\s+here)$"#)
            { return m[0] }
            return nil
        }()
        if let rawKind = nearestKind {
            let firstWord = rawKind.split(separator: " ", maxSplits: 1)
                .first.map(String.init) ?? ""
            let skip: Set<String> = [
                "what", "what's", "where", "when", "why", "how", "who",
                "anything", "something", "everything", "it", "that", "this",
                "my", "your", "the",
            ]
            if !skip.contains(firstWord) {
                // A real category — anchor on GPS, or bail to the LLM if we
                // have no fix (never guess "me"/"here" as a place).
                guard let here = currentLocation else { return nil }
                let kind = singularize(rawKind)
                return DirectIntent(toolName: "near_places", args: [
                    "lat":       .double(here.lat),
                    "lon":       .double(here.lon),
                    "kinds":     .array([.string(kind)]),
                    "radius_km": .double(defaultRadiusKm)
                ])
            }
        }

        // "directions to <place>" / "route to <place>" / "navigate to <place>".
        // Checked BEFORE the `<X> in <Y>` pattern so destinations
        // containing " in " ("Library in Mountain View") still route
        // as directions, not as a places search. Polite prefixes like
        // "give me", "show me", "get me", "can you give me", and
        // leading "please" are accepted — otherwise "Give me directions
        // to SF" falls to the LLM, which sometimes emits malformed JSON
        // and silently drops the turn.
        let directionsBody = stripDirectionsPrefix(lower)
        if let m = match(directionsBody, pattern:
            #"^(?:directions?|route|navigate)\s+(?:to|for)\s+(.+)$"#)
            ?? match(directionsBody, pattern:
                #"^(?:how\s+(?:do\s+i\s+|to\s+)?get\s+to)\s+(.+)$"#)
            ?? match(directionsBody, pattern:
                #"^(?:take\s+me\s+to)\s+(.+)$"#)
        {
            let dest = m[0]
            return DirectIntent(toolName: "route_from_places", args: [
                "origin":      .string("my location"),
                "destination": .string(dest)
            ])
        }

        // "<category> in <place>" / "<category> near <place>". Named-place
        // geocoding happens inside the tool — we don't pre-resolve here.
        if let m = match(lower,
                         pattern: #"^(.+?)\s+(?:in|near|around|at)\s+(.+)$"#)
        {
            let kind = singularize(m[0])
            let place = m[1]
            // Guard: don't misclassify questions ("where can I find bars in SF")
            // or explanations ("how does rain form in clouds") as places.
            let skipLeadingWords: Set<String> = [
                "what", "where", "when", "why", "how", "who",
                "tell", "show", "find", "can", "could", "would",
                "should", "is", "are", "do", "does", "did"
            ]
            let firstWord = lower.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
            if !skipLeadingWords.contains(firstWord) {
                return DirectIntent(toolName: "near_named_place", args: [
                    "place":     .string(place),
                    "kinds":     .array([.string(kind)]),
                    "radius_km": .double(defaultRadiusKm)
                ])
            }
        }

        // "compare <A> and|vs|with|to <B>" → compare_articles. Two-entity
        // pattern; Qwen 3.5 4B was the observed culprit that dropped this
        // turn with a malformed `" "Foo"` splice (see
        // dropped-request.log Case 2). The tool handles 2–4 titles —
        // we surface two; the model never needed to run.
        if let m = match(lower, pattern:
            #"^compare\s+(.+?)\s+(?:and|vs\.?|versus|with|to)\s+(.+)$"#)
        {
            let (a, b) = expandSharedSuffix(first: m[0], second: m[1])
            return DirectIntent(toolName: "compare_articles", args: [
                "titles": .array([.string(a), .string(b)])
            ])
        }

        // "let's discuss X" / "let's talk about X" / "discuss X" / "dig
        // into X" → discuss_article: pin the article as a discussion focus
        // so follow-up questions are answered from its sections (grounded
        // single-article RAG) instead of a fresh tool route each turn.
        // Runs before article_overview so the discuss verbs win. A bare
        // pronoun subject ("discuss it") is left to the LLM for now.
        if let m = match(lower, pattern:
            #"^(?:(?:let'?s|let\s+us|can\s+we|could\s+we|i\s+(?:want|wanna|would\s+like)\s+to|i'?d\s+like\s+to)\s+)?(?:discuss|talk\s+about|chat\s+about|dig\s+into|go\s+deep\s+on)\s+(.+)$"#)
        {
            let subject = m[0].trimmingCharacters(in: .whitespaces)
            let firstWord = subject
                .split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
            let navPronouns: Set<String> = [
                "it", "this", "that", "them", "these", "those", "him", "her",
            ]
            if !subject.isEmpty, !navPronouns.contains(firstWord) {
                return DirectIntent(toolName: "discuss_article", args: [
                    "title": .string(subject)
                ])
            }
        }

        // "tell me about X" / "what is X" / "who is/was X" /
        // "give me an overview of X" → article_overview. Runs LAST
        // so that `what_is_here`, directions, `compare`, and places
        // patterns win first. Subject starting with a route/demonstrative
        // pronoun ("my", "this", "here", …) is almost always a
        // navigational query ("what is my next turn") that wants the
        // LLM, not an article — bail so the model gets it. Articles
        // that don't exist in the loaded ZIMs come back as a clean
        // "no article" miss, which is still faster than a 15 s
        // prefill + possibly-malformed tool call.
        if let m = match(lower, pattern:
            #"^(?:tell\s+me\s+(?:about|more\s+about)|what(?:'s|\s+is|\s+are)|who(?:'s|\s+is|\s+was|\s+were|\s+are)|give\s+me\s+(?:an?\s+)?overview\s+of|overview\s+of)\s+(.+)$"#)
        {
            let subject = m[0].trimmingCharacters(in: .whitespaces)
            let firstWord = subject
                .split(separator: " ", maxSplits: 1)
                .first.map(String.init) ?? ""
            let navPronouns: Set<String> = [
                "my", "our", "your", "here", "now", "next",
                "this", "that", "these", "those", "it"
            ]
            if navPronouns.contains(firstWord) { return nil }
            // Subject must have at least one content character — "what
            // is" with nothing after would match `.+` on the trailing
            // "?!." the caller stripped. Guard against that.
            if subject.isEmpty { return nil }
            return DirectIntent(toolName: "article_overview", args: [
                "title": .string(subject)
            ])
        }

        // Corrections / restatements of a mis-transcribed query: "I meant
        // X", "I was actually talking about X", "no, I said X". Voice
        // users fix an ASR mishear this way — real capture 2026-05-29:
        // "grand Duchy of Lithuania" came through as "grand Dutch
        // Lithuania", and when the user restated it correctly the turn
        // fell to the LLM, which reused its earlier wrong guess ("Dutch
        // Lithuania") instead of the corrected words. Strip the wrapper
        // and re-route the corrected text through the router; a bare
        // restated noun phrase ("…the grand duchy of lithuania") defaults
        // to an article lookup. Runs LAST so a normal "tell me about X"
        // never reaches here.
        if let corrected = stripCorrectionPrefix(lower) {
            // Re-route the corrected words (handles "I meant directions
            // to X", "I was asking about bars in SF", etc.).
            if let inner = classify(corrected, currentLocation: currentLocation) {
                return inner
            }
            // Otherwise treat the restatement as a bare article subject,
            // dropping any leading "tell me about" / "what is" lead.
            let subject = stripArticleLead(corrected)
                .trimmingCharacters(in: .whitespaces)
            let firstWord = subject
                .split(separator: " ", maxSplits: 1)
                .first.map(String.init) ?? ""
            let navPronouns: Set<String> = [
                "my", "our", "your", "here", "now", "next",
                "this", "that", "these", "those", "it",
            ]
            if !subject.isEmpty, !navPronouns.contains(firstWord) {
                return DirectIntent(toolName: "article_overview", args: [
                    "title": .string(subject)
                ])
            }
        }

        return nil
    }

    /// Strip a leading correction/restatement wrapper ("I meant …", "I
    /// was actually talking about …", "no, I said …") and return the
    /// residual, or nil when the text isn't a correction. Deliberately
    /// conservative — only fires on explicit correction verbs, so a
    /// plain "no" answer or "actually it's fine" doesn't get reshaped.
    private static func stripCorrectionPrefix(_ s: String) -> String? {
        let pattern =
            #"^(?:no[,\s]+|actually[,\s]+)?(?:i\s+(?:actually\s+)?(?:meant|mean|said|wanted)|i\s+was\s+(?:actually\s+)?(?:talking|asking)\s+about|i\s+was\s+referring\s+to|i'?m\s+(?:talking|asking)\s+about|actually[,\s]+i\s+(?:meant|mean))\s+(.+)$"#
        if let m = match(s, pattern: pattern), let r = m.first {
            return r.trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    /// Drop a leading "tell me about" / "what is" / "who was" / "overview
    /// of" / "about" lead so a restated subject indexes on the entity.
    private static func stripArticleLead(_ s: String) -> String {
        let p =
            #"^(?:tell\s+me\s+(?:about|more\s+about)|what(?:'s|\s+is|\s+are)|who(?:'s|\s+is|\s+was|\s+were|\s+are)|give\s+me\s+(?:an?\s+)?overview\s+of|overview\s+of|about)\s+"#
        return s.replacingOccurrences(
            of: p, with: "", options: .regularExpression)
    }

    /// True when the user is asking to keep reading the article currently
    /// being narrated aloud — the LITERAL "continue reading" verbs only
    /// ("continue", "keep reading", "read on", "next section", …). The
    /// host (ChatSession) acts on this only when a reading position is
    /// active, and pages the next section.
    ///
    /// Open-ended follow-ups ("tell me more", "more", "go on", "what
    /// else", "keep going") are deliberately NOT matched here — they carry
    /// conversational intent, so they fall through to `classify`'s
    /// focus-aware `continuationIntent`, which re-opens the subject and
    /// lets the drift engine offer related threads. Splitting the phrases
    /// this way keeps article paging and conversational continuation from
    /// fighting over the same words (see CONVERSATIONAL_REDESIGN.md).
    public static func isContinueReading(_ raw: String) -> Bool {
        let t = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!,"))
            .lowercased()
        if t.isEmpty { return false }
        let exact: Set<String> = [
            "continue", "continue reading", "keep reading", "read on",
            "read more", "read it", "next section", "next page",
            "resume reading",
        ]
        if exact.contains(t) { return true }
        return matches(t, pattern:
            #"^(?:please\s+|and\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?(?:please\s+)?(?:keep\s+reading|read\s+(?:me\s+)?(?:on|more)|continue(?:\s+reading)?)\b"#)
    }

    /// Resolve a conversational follow-up against the focus and turn it into
    /// a concrete tool call. Returns `nil` when the turn is NOT a binding
    /// continuation (the caller then proceeds to the stateless patterns).
    ///
    /// Bindings map to tools by the resolved entity's kind:
    ///   * a place the user references locationally ("how far is it", "near
    ///     it") → `near_places` around its coords;
    ///   * any other bound entity → `article_overview` on its name, which
    ///     returns the lead + section list so the model can answer the
    ///     specific facet ("who built it", "how old") from one fetch.
    ///
    /// Ambiguous descriptive selectors ("the old one" matching several) and
    /// pure cache-answerable follow-ups are deliberately left to the LLM loop
    /// (return `nil`) — the router only short-circuits the unambiguous picks.
    static func continuationIntent(
        _ raw: String, focus: ConversationFocus
    ) -> DirectIntent? {
        let resolved = ReferenceResolver.resolve(raw, focus: focus)
        guard resolved.isContinuation, let entity = resolved.boundEntity else {
            return nil
        }
        let lower = raw.lowercased()

        // Locational follow-up about a place we have coordinates for.
        let locational = ["near", "around", "close", "nearby",
                          "how far", "distance", "walk", "drive",
                          "directions", "route"]
        if entity.kind == .place, let lat = entity.lat, let lon = entity.lon,
           locational.contains(where: { lower.contains($0) }) {
            if lower.contains("directions") || lower.contains("route")
                || lower.contains("how do i get") {
                return DirectIntent(toolName: "route_from_places", args: [
                    "origin":      .string("my location"),
                    "destination": .string(entity.name),
                ])
            }
            return DirectIntent(toolName: "near_places", args: [
                "lat":       .double(lat),
                "lon":       .double(lon),
                "radius_km": .double(1),
            ])
        }

        // Default: re-open the subject encyclopedically. `article_overview`
        // returns the lead plus the section list, which covers the common
        // elliptical follow-ups ("who built it", "when", "how big") without a
        // second round-trip.
        return DirectIntent(toolName: "article_overview", args: [
            "title": .string(entity.name),
        ])
    }

    /// Explicit "we're done with this article" phrases that end a "let's
    /// discuss X" session. Conservative — exact matches only, so a genuine
    /// question never accidentally ends the discussion. (A topic *change*
    /// is detected separately by the host re-classifying the turn.)
    public static func isDiscussionExit(_ raw: String) -> Bool {
        let t = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!,"))
            .lowercased()
        let exits: Set<String> = [
            "stop", "stop discussing", "stop discussing it", "stop discussing this",
            "never mind", "nevermind", "done", "i'm done", "im done", "we're done",
            "that's all", "thats all", "exit", "quit", "ok done", "okay done",
            "enough", "let's stop", "lets stop", "change topic", "change the topic",
            "new topic", "something else", "let's talk about something else",
            "talk about something else", "move on", "forget it",
        ]
        return exits.contains(t)
    }

    /// Natural-English shared-suffix inference for `compare X and Y Z`.
    ///
    /// When a speaker says "compare north and south korea", what they
    /// almost always mean is "compare north korea and south korea" —
    /// "Korea" is a shared suffix the speaker dropped from the first
    /// half. The strict regex parse gives us `X = "north"`,
    /// `Y_Z = "south korea"`. If we dispatch that verbatim, the
    /// `north` article lookup misses and the reply degenerates into
    /// "Comparing north and south korea. Results below." with nothing
    /// below (real on-device capture).
    ///
    /// Heuristic: if the first title is a *single directional or
    /// ordinal word* AND the second has 2+ words, treat the last word
    /// of the second as the shared suffix and append it to the first.
    /// Conservative by design — we only reshape when the shape is an
    /// obvious shared-suffix pattern. Legit pairs like "cats and
    /// dogs" or "Apple and Google" (first is a single non-directional
    /// word) pass through untouched.
    static func expandSharedSuffix(
        first: String, second: String
    ) -> (String, String) {
        let sharedPrefixWords: Set<String> = [
            "north", "south", "east", "west",
            "northern", "southern", "eastern", "western",
            "upper", "lower",
            "old", "new", "young", "elder", "modern", "ancient",
            "first", "second", "third", "fourth", "fifth",
            "left", "right",
            "big", "little", "greater", "lesser",
        ]
        let aWords = first.split(separator: " ").map(String.init)
        let bWords = second.split(separator: " ").map(String.init)
        guard aWords.count == 1,
              bWords.count >= 2,
              sharedPrefixWords.contains(aWords[0]),
              let tail = bWords.last
        else {
            return (first, second)
        }
        // Don't double-append if the speaker was explicit enough to
        // include the suffix on both halves ("compare north korea
        // and south korea" — already aWords.count == 2, guarded
        // above).
        return (aWords[0] + " " + tail, second)
    }

    /// Turn an English plural into its singular form for the OSM
    /// `kinds` vocabulary. Handles the three common -s / -es / -ies
    /// patterns. Anything irregular (men, children, criteria…) is
    /// passed through — those are rare enough in POI categories
    /// that the OSM vocab's fuzzy match picks up the slack.
    public static func singularize(_ s: String) -> String {
        guard s.count > 3 else { return s }
        // "libraries" → "library". Must precede the generic -s rule
        // or we'd strip just the trailing s and keep a dangling "e".
        if s.hasSuffix("ies") {
            return String(s.dropLast(3)) + "y"
        }
        // "churches", "buses", "boxes", "quizzes" — two-char strip.
        for ending in ["ches", "shes", "xes", "ses", "zes"] {
            if s.hasSuffix(ending) {
                return String(s.dropLast(2))
            }
        }
        // Plain -s plural. Exclude -ss (passes, grass) which isn't
        // actually a plural.
        if s.hasSuffix("s"), !s.hasSuffix("ss") {
            return String(s.dropLast())
        }
        return s
    }

    /// Strip polite wrappers that wouldn't change the underlying
    /// directions intent. Voice input in particular produces these
    /// routinely ("Give me directions to…", "Can you show me
    /// directions to…"). We strip once, up-front, so the core
    /// patterns stay simple.
    private static func stripDirectionsPrefix(_ s: String) -> String {
        // Order matters: outer politeness first, then the "<verb> me"
        // pair. Kept as separate passes so each replacement is a
        // single anchored regex.
        var out = s
        let patterns = [
            #"^please\s+"#,
            #"^(?:can|could|would|will)\s+you\s+(?:please\s+)?"#,
            #"^(?:i\s+(?:need|want|would\s+like)|i'd\s+like)\s+"#,
            #"^(?:give|show|get|find|tell|fetch)\s+me\s+(?:(?:the|some)\s+)?"#,
        ]
        for p in patterns {
            out = out.replacingOccurrences(
                of: p, with: "", options: .regularExpression)
        }
        return out
    }

    /// Boolean "does this text match the pattern" helper — used by
    /// the `what_is_here` check where we don't care about captures,
    /// just whether the pattern fires. `match()` returns nil for
    /// capture-less patterns because `numberOfRanges < 2`, so it's
    /// unsuitable here.
    private static func matches(_ text: String, pattern: String) -> Bool {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return false
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }

    /// Light regex helper — returns only the capture groups (not the
    /// full-match range). `NSRegularExpression` verbatim with a
    /// `nil`-to-`Substring[]` adapter.
    private static func match(_ text: String, pattern: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return nil
        }
        let range = NSRange(text.startIndex..., in: text)
        guard let m = regex.firstMatch(in: text, options: [], range: range),
              m.numberOfRanges >= 2
        else { return nil }
        var out: [String] = []
        for i in 1..<m.numberOfRanges {
            if let r = Range(m.range(at: i), in: text) {
                out.append(String(text[r]))
            }
        }
        return out
    }

    // MARK: - Reply synthesis

    /// Build the one-line caption we drop into the assistant bubble
    /// when the fast path (or post-tool skip-model-reply) wants to
    /// bypass the LLM's prose. The caller passes the tool args +
    /// the full tool result; the map bubble below carries the
    /// answer, this is just the header.
    public static func synthesizePlacesReply(
        toolName: String,
        args: [String: Any],
        fullResult: [String: Any]
    ) -> String {
        // `locate` resolves a single named place to a pin; the map bubble
        // below is the real answer, so the caption just names what resolved
        // (which may differ from what was asked — e.g. "Stanford Hospital"
        // → "Stanford Health Care").
        if toolName == "locate" {
            let name = (fullResult["resolved"] as? [String: Any])?["name"] as? String
                ?? (fullResult["results"] as? [[String: Any]])?.first?["name"] as? String
                ?? (args["place"] as? String) ?? "the place"
            return "\(name) — shown on the map below."
        }
        let kind: String = {
            if let k = (args["kinds"] as? [String])?.first, !k.isEmpty { return k }
            if let k = args["kinds"] as? String, !k.isEmpty { return k }
            if let q = args["query"] as? String, !q.isEmpty { return q }
            return "places"
        }()
        let where_: String = {
            if let p = args["place"] as? String, !p.isEmpty { return p }
            if (args["lat"] as? NSNumber) != nil
               && (args["lon"] as? NSNumber) != nil { return "you" }
            if let o = args["origin"] as? String, !o.isEmpty { return o }
            return "here"
        }()
        let count: Int? = {
            if let n = (fullResult["total_in_radius"] as? NSNumber)?.intValue { return n }
            if let rs = fullResult["results"] as? [[String: Any]] { return rs.count }
            if let ss = fullResult["stories"] as? [[String: Any]] { return ss.count }
            if let n = (fullResult["count"] as? NSNumber)?.intValue { return n }
            return nil
        }()
        let radiusKm = (fullResult["radius_km"] as? NSNumber)?.doubleValue

        // Pluralise once for the display line — "bar" → "bars";
        // "museums" → "museums"; "cafes" → "cafes".
        let kindPlural = kind.hasSuffix("s") ? kind : kind + "s"

        var line: String
        if let n = count, n > 0 {
            line = "Found \(n) \(kindPlural) near \(where_)"
        } else if count == 0 {
            line = "No \(kindPlural) found near \(where_)"
        } else {
            line = "Results for \(kindPlural) near \(where_)"
        }
        if let r = radiusKm { line += " (within \(formatKm(r)))" }
        line += ". Tap a pin on the map for details"
        if count != 0 { line += ", or tap List for the full rundown" }
        line += "."
        return line
    }

    private static func formatKm(_ km: Double) -> String {
        if km < 1 { return "\(Int(km * 1000)) m" }
        if km.truncatingRemainder(dividingBy: 1) == 0 {
            return "\(Int(km)) km"
        }
        return String(format: "%.1f km", km)
    }

    // MARK: - Fast-path usability checks
    //
    // Returning `false` signals to the caller that the tool technically
    // succeeded (no exception) but didn't produce anything the user
    // will find useful — e.g. compare_articles came back with no
    // resolved articles AND no relations article, or article_overview
    // returned a miss. The caller then clears the fast-path attempt
    // and falls through to the LLM loop, which can at least try a
    // different tool / different titles / freeform answer.

    public static func compareResultIsUsable(_ fullResult: [String: Any]) -> Bool {
        if let err = fullResult["error"] as? String, !err.isEmpty { return false }
        // Dedicated relations article found — always useful.
        if (fullResult["strategy"] as? String) == "dedicated_relations_article",
           let sections = fullResult["sections"] as? [[String: Any]],
           let first = sections.first,
           !((first["text"] as? String) ?? "").isEmpty
        {
            return true
        }
        // Otherwise need at least two articles with real section text.
        let articles = (fullResult["articles"] as? [[String: Any]]) ?? []
        let good = articles.filter { a in
            if let e = a["error"] as? String, !e.isEmpty { return false }
            let sections = (a["sections"] as? [[String: Any]]) ?? []
            let text = (sections.first?["text"] as? String) ?? ""
            return !text.isEmpty
        }
        return good.count >= 2
    }

    public static func articleOverviewResultIsUsable(_ fullResult: [String: Any]) -> Bool {
        if let err = fullResult["error"] as? String, !err.isEmpty { return false }
        let sections = (fullResult["sections"] as? [[String: Any]]) ?? []
        let text = (sections.first?["text"] as? String) ?? ""
        return !text.isEmpty
    }

    public static func whatIsHereResultIsUsable(_ fullResult: [String: Any]) -> Bool {
        if let err = fullResult["error"] as? String, !err.isEmpty { return false }
        let place = (fullResult["nearest_named_place"] as? String) ?? ""
        return !place.isEmpty
    }

    // MARK: - Reply synthesis for non-places fast paths

    /// Caption for `article_overview` fast-path. Grabs the lead
    /// section's first sentence or two so the bubble carries a real
    /// answer instead of a stub — no LLM needed.
    public static func synthesizeArticleOverviewReply(
        args: [String: Any], fullResult: [String: Any]
    ) -> String {
        let title = (fullResult["title"] as? String)
            ?? (args["title"] as? String) ?? "this topic"
        if let err = fullResult["error"] as? String, !err.isEmpty {
            return "I don't have an article on “\(title)” in the loaded ZIMs."
        }
        if let sections = fullResult["sections"] as? [[String: Any]],
           let lead = sections.first,
           let text = (lead["text"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !text.isEmpty
        {
            return firstSentences(text, maxChars: 260)
        }
        return "Here's what I have on \(title)."
    }

    /// Caption for an `article_overview` MISS — the title didn't resolve
    /// in any loaded ZIM. We deliberately do NOT hand these to the LLM
    /// (it confabulates a wrong entity from the bad title); instead we
    /// say so plainly and offer the closest real titles the index found,
    /// so the user can re-ask. Often there are no suggestions (a mis-hear
    /// shares no tokens with any real title) — that's fine, the plain
    /// "couldn't find it" still beats an invented answer.
    public static func synthesizeArticleMissReply(
        args: [String: Any], fullResult: [String: Any]
    ) -> String {
        let title = (fullResult["requested_title"] as? String)
            ?? (args["title"] as? String) ?? "that"
        let suggestions = (fullResult["suggestions"] as? [String]) ?? []
        let base = "I couldn't find an article for “\(title)” in the offline Wikipedia."
        if suggestions.isEmpty {
            return base + " Try saying the name a different way."
        }
        if suggestions.count == 1 {
            return base + " Did you mean \(suggestions[0])?"
        }
        return base + " Did you mean: " + suggestions.prefix(3).joined(separator: ", ") + "?"
    }

    /// Caption for `compare_articles` fast-path. Leads with the first
    /// sentence of each article so the two subjects are actually
    /// introduced; the full side-by-side payload lands in the trace /
    /// map bubble below.
    public static func synthesizeCompareReply(
        args: [String: Any], fullResult: [String: Any]
    ) -> String {
        if let err = fullResult["error"] as? String, !err.isEmpty {
            return err
        }
        // Relations-article fast path: compare_articles for a pair
        // like (North Korea, South Korea) first probes for a
        // dedicated Wikipedia relations article. When it hits, the
        // result shape is different from the default side-by-side —
        // top-level `sections` + `strategy == "dedicated_relations_article"`
        // and no `articles` array. Render the lead of the relations
        // article as the caption.
        if let strategy = fullResult["strategy"] as? String,
           strategy == "dedicated_relations_article"
        {
            let title = (fullResult["resolved_title"] as? String) ?? "these two"
            let sections = (fullResult["sections"] as? [[String: Any]]) ?? []
            let lead = (sections.first?["text"] as? String) ?? ""
            let snippet = firstSentences(lead, maxChars: 320)
            if snippet.isEmpty {
                return "**\(title)** — see the article for details."
            }
            return "**\(title)** — \(snippet)"
        }
        let articles = (fullResult["articles"] as? [[String: Any]]) ?? []
        // Drop entries where the tool couldn't fetch the article — a
        // bad title will land here (real-device example: "north" +
        // "south korea" instead of "north korea" + "south korea"),
        // and carrying the title alone with no content produces the
        // degenerate "Comparing north and south korea. Results below."
        // bubble with nothing below.
        let good = articles.filter { a in
            if let e = a["error"] as? String, !e.isEmpty { return false }
            let sections = (a["sections"] as? [[String: Any]]) ?? []
            let text = (sections.first?["text"] as? String) ?? ""
            return !text.isEmpty
        }
        let lines: [String] = good.prefix(3).map { a in
            let t = (a["title"] as? String) ?? ""
            let sections = (a["sections"] as? [[String: Any]]) ?? []
            let text = (sections.first?["text"] as? String) ?? ""
            let snippet = firstSentences(text, maxChars: 160)
            if t.isEmpty { return snippet }
            return "**\(t)** — \(snippet)"
        }
        // Need both subjects to have content before we can usefully
        // compare — one-subject-found outputs "**South Korea** — …"
        // with no North Korea, which reads as a wiki lookup not a
        // comparison and isn't what the user asked for.
        if lines.count >= 2 {
            return lines.joined(separator: "\n\n")
        }
        let titles = (args["titles"] as? [String]) ?? []
        if titles.count >= 2 {
            // Name the titles we couldn't resolve so the user can see
            // what needs re-asking. Better than the old "Comparing X
            // and Y. Results below." which was a lie when `Results`
            // turned out to be empty.
            let failing: [String] = articles.compactMap { a in
                guard let e = a["error"] as? String, !e.isEmpty else { return nil }
                return (a["title"] as? String)
            }
            if !failing.isEmpty {
                let q = failing.map { "“\($0)”" }.joined(separator: " or ")
                return "I couldn't find articles for \(q). "
                    + "Try the full names (e.g. \"North Korea and South Korea\")."
            }
            return "I couldn't find articles matching those titles. "
                + "Try the full names on both sides."
        }
        return "I couldn't put together a comparison from that query."
    }

    /// Caption for `what_is_here` fast-path. Describes the resolved
    /// place + admin area; if the tool attached a Wikipedia summary,
    /// appends the first sentence of that.
    public static func synthesizeWhatIsHereReply(
        fullResult: [String: Any]
    ) -> String {
        if let err = fullResult["error"] as? String, !err.isEmpty {
            return err
        }
        let place = (fullResult["nearest_named_place"] as? String) ?? ""
        let area = (fullResult["admin_area"] as? String) ?? ""
        let distRaw = (fullResult["distance_m"] as? Int)
            ?? (fullResult["distance_m"] as? NSNumber)?.intValue ?? 0
        if place.isEmpty {
            return "I couldn't identify a named place near your location."
        }
        var line = "You're"
        if distRaw <= 100 {
            line += " in \(place)"
        } else if distRaw < 1000 {
            line += " \(distRaw) m from \(place)"
        } else {
            line += String(format: " %.1f km from %@",
                           Double(distRaw) / 1000.0, place)
        }
        if !area.isEmpty, area.lowercased() != place.lowercased() {
            line += " (\(area))"
        }
        line += "."
        if let summary = fullResult["wiki_summary"] as? String,
           !summary.isEmpty
        {
            line += " " + firstSentences(summary, maxChars: 200)
        }
        return line
    }

    /// Trim `text` to at most one or two complete sentences (up to
    /// `maxChars`). Keeps the ending punctuation when we cut on a
    /// sentence boundary; falls back to a hard cut + ellipsis if no
    /// boundary appears within budget.
    public static func firstSentences(_ text: String, maxChars: Int) -> String {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty { return "" }
        if t.count <= maxChars { return t }
        // Scan for the last sentence terminator inside the budget.
        let limitIdx = t.index(t.startIndex, offsetBy: maxChars)
        var lastTerm: String.Index?
        var i = t.startIndex
        while i < limitIdx {
            let c = t[i]
            if c == "." || c == "!" || c == "?" {
                let next = t.index(after: i)
                if next == t.endIndex || t[next].isWhitespace {
                    lastTerm = next
                }
            }
            i = t.index(after: i)
        }
        if let end = lastTerm {
            return String(t[..<end]).trimmingCharacters(in: .whitespaces)
        }
        return String(t[..<limitIdx]).trimmingCharacters(in: .whitespaces) + "…"
    }
}
