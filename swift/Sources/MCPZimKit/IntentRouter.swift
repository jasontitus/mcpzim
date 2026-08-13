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

/// An explicit instruction to ground the next answer in one particular
/// Wikipedia article. This is intentionally separate from ordinary intent
/// classification: the host can honor it before a pinned discussion or a
/// location-shaped fast path gets a chance to reinterpret the words.
public struct WikipediaSourceDirective: Equatable, Sendable {
    public let title: String
    public let question: String?

    public init(title: String, question: String? = nil) {
        self.title = title
        self.question = question
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
        var text = raw
            .replacingOccurrences(of: "\u{2019}", with: "'") // iOS smart quote → ASCII, so "Putin’s" title-matches
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!"))
        // Strip a leading connective — voice turns routinely open with
        // "And/So/Also/Ok" ("And tell me about Donald Trump"), which
        // defeated every ^-anchored pattern below: the turn classified
        // as nothing, discussion mode never saw the topic switch, and
        // the model confabulated Trump answers from Putin passages
        // (real capture 2026-07-02). Only strip when a real clause
        // follows, so a bare "ok"/"and?" still reads as a continuation.
        if let m = match(text.lowercased(), pattern:
            #"^(?:and|so|also|then|ok|okay|now|next|hey|oh|no|nope|wait|actually)[.,\s]+(.{4,})$"#) {
            text = String(text.suffix(m[0].count))
        }
        if text.isEmpty { return nil }
        let lower = text.lowercased()
        let defaultRadiusKm: Double = 5

        // Reading requests are semantic actions, not questions about an
        // article. Resolve them before ordinary continuation routing so a
        // pinned discussion cannot swallow "read the whole article" and
        // answer from its small evidence window instead. Explicit titles work
        // without focus; the subject-less form binds to the current topic.
        if let reading = readArticleIntent(text, focus: focus) {
            return reading
        }

        // Explicit source selection outranks every stateless interpretation.
        // Source control means "prepare this exact article", not merely open
        // its compact overview. The app host intercepts the directive before
        // ordinary routing, while other callers still get the same all-
        // sections tool semantics.
        if let directive = wikipediaSourceDirective(text, focus: focus) {
            return DirectIntent(toolName: "discuss_article", args: [
                "title": .string(directive.title),
            ])
        }

        // A comparison establishes a TWO-topic list in ConversationFocus.
        // Follow-ups such as "How many were killed in each?" do not name
        // either subject, so a small model is otherwise forced to infer both
        // the referents and the right retrieval tool. Bonsai 27B's 1-bit Mac
        // run exposed the failure mode directly: it routed the casualty turn
        // to a geographic nearby-stories tool. Resolve this narrow discourse
        // shape in Swift and align both articles on the requested section.
        // Causal/interpretive follow-ups ("what changed between the two?")
        // deliberately fall through to the model so it synthesizes from the
        // grounded comparison already present in the conversation.
        if let focus,
           let route = comparisonContinuationRoute(lower, focus: focus)
        {
            switch route {
            case .retrieve(let section):
                var args: [String: AnyJSONValue] = [
                    "titles": .array(focus.lastList.prefix(2).map {
                        .string($0.name)
                    })
                ]
                if let section { args["section"] = .string(section) }
                return DirectIntent(toolName: "compare_articles", args: args)
            case .synthesizeFromContext:
                return nil
            }
        }

        // Deterministic Wikipedia factoid: a founding-date question should
        // never pay for the model to first choose `article_overview` and then
        // pay for a second model pass to quote the lead. On the phone that
        // cost 85 s + 55 s for "When was Tesla founded?" (2026-07-15).
        // Route the narrow, objectively extractable shape to an internal
        // adapter operation. The adapter accepts an answer only when a real
        // Wikipedia lead contains both a founding synonym and a year.
        if let factoid = ageFactoidIntent(text, focus: focus)
            ?? foundingFactoidIntent(lower, focus: focus) {
            return factoid
        }

        // Pearl Harbor casualty questions are a high-confidence named-event
        // factoid, not a request about every death at the modern naval base.
        // Route straight to the event article so the extractive death-count
        // path can quote its casualty sentence. This avoids a full 12-tool
        // prefill and prevents the model from inventing a breakdown around an
        // otherwise correct total (TestFlight build 20260810203059).
        if lower.contains("pearl harbor"),
           ["died", "dead", "death", "killed", "fatalit", "casualt"]
            .contains(where: { lower.contains($0) })
        {
            return DirectIntent(toolName: "article_overview", args: [
                "title": .string("attack on pearl harbor"),
            ])
        }

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

        // "Where is <named place>?" → locate. The location engine answered
        // the real HP Garage query in 0.01 s, but the phone first spent 87 s
        // asking Bonsai to choose this tool (2026-07-15). Keep the pattern
        // deliberately named-place-shaped: nearest-category queries already
        // returned above, while encyclopedic forms such as "where is Apple
        // headquartered?" and "where is the capital of France?" fall through.
        let locateSubject: String? = {
            if let m = match(text, pattern:
                #"^where(?:'s|\s+is)\s+(.+?)(?:\s+located)?$"#) {
                return m[0]
            }
            if let m = match(text, pattern:
                #"^(?:locate|find\s+the\s+location\s+of)\s+(.+)$"#) {
                return m[0]
            }
            if let m = match(text, pattern:
                #"^show\s+me\s+(.+?)\s+on\s+(?:the\s+)?map$"#) {
                return m[0]
            }
            return nil
        }()
        if var subject = locateSubject {
            subject = subject.replacingOccurrences(
                of: #"^the\s+"#, with: "",
                options: [.regularExpression, .caseInsensitive]
            ).trimmingCharacters(in: .whitespaces)
            let lowerSubject = subject.lowercased()
            let first = lowerSubject.split(separator: " ", maxSplits: 1)
                .first.map(String.init) ?? ""
            let nonNames: Set<String> = [
                "it", "this", "that", "there", "here", "he", "she", "they",
                "my", "our", "your", "his", "her", "their", "nearest",
                "closest", "best", "article", "information", "answer",
            ]
            let encyclopedic = lowerSubject.range(
                of: #"\b(?:capital|population|history|founder|meaning|definition|birthplace)\s+of\b|\b(?:headquartered|born|founded|created|established|happen|happened|take\s+place)\b"#,
                options: .regularExpression
            ) != nil
            if !subject.isEmpty, !nonNames.contains(first), !encyclopedic {
                return DirectIntent(toolName: "locate", args: [
                    "place": .string(subject),
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

        // "how far is it to X" / "how far to X" / "how far away is X" /
        // "distance to X" / "how long to (get to/drive to) X" → route from
        // the user's location; the routing reply states the distance +
        // duration. Bare "how far is it" (no X) has no match here and falls
        // to the focus-aware continuationIntent (route to the focused place).
        if let m = match(lower, pattern:
            #"^how\s+far\s+(?:is\s+it\s+)?(?:away\s+)?(?:to|is|from\s+here\s+to)\s+(.+)$"#)
            ?? match(lower, pattern:
                #"^(?:what'?s\s+the\s+)?distance\s+(?:to|from\s+here\s+to)\s+(.+)$"#)
            ?? match(lower, pattern:
                #"^how\s+long\s+(?:does\s+it\s+take\s+)?to\s+(?:get\s+to\s+|drive\s+to\s+|walk\s+to\s+|reach\s+)?(.+)$"#)
        {
            let dest = m[0].trimmingCharacters(in: .whitespaces)
            // Reject a pronoun destination ("how far is it" → dest "it"):
            // that's a follow-up about the focused place, handled by
            // continuationIntent, not a standalone route.
            let pronoun: Set<String> = ["it", "that", "this", "there", "them", "here"]
            if !dest.isEmpty, !pronoun.contains(dest) {
                return DirectIntent(toolName: "route_from_places", args: [
                    "origin":      .string("my location"),
                    "destination": .string(dest)
                ])
            }
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
                "should", "is", "are", "do", "does", "did",
                // First-person prose here is normally a factual request,
                // not a POI category. Real TestFlight crash 2026-08-10:
                // “I am wanting to know how many people died … in World
                // War II” became a 12-word `kinds` value and launched a map.
                "i", "i'm", "im", "we", "no", "please"
            ]
            let firstWord = lower.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
            if !skipLeadingWords.contains(firstWord),
               !isConversationalKnowledgeRequest(text) {
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
                    "title": .string(canonicalArticleSubject(subject))
                ])
            }
        }

        // "how do/does X work" → article_overview(X). An explanatory
        // question about a topic — route it to a grounded overview (which
        // also opens an implicit discussion) instead of the LLM loop, which
        // over-tools into a wrong/failed get_article_section and then
        // narrates the failure (real capture 2026-05-30: "how do combustion
        // engines work?" → "…I attempted to retrieve the History section…
        // wasn't found… Let me try a broader search"). The last subject word
        // is singularised so "combustion engines" resolves to the
        // (singular-titled) article.
        if let m = match(lower, pattern:
            #"^how\s+(?:do|does|did)\s+(.+?)\s+works?$"#)
        {
            var subject = m[0]
                .replacingOccurrences(
                    of: #"^(?:a|an|the)\s+"#, with: "",
                    options: [.regularExpression, .caseInsensitive])
                .trimmingCharacters(in: .whitespaces)
            var words = subject.split(separator: " ").map(String.init)
            if let last = words.last, words.count >= 1 {
                words[words.count - 1] = singularize(last)
                subject = words.joined(separator: " ")
            }
            if !subject.isEmpty {
                return DirectIntent(toolName: "article_overview", args: [
                    "title": .string(subject)
                ])
            }
        }

        // "explain X [to me]" / "teach me about X" / "help me
        // understand X" → article_overview(X). These broad explanatory
        // openers used to miss every deterministic route and enter the
        // generic 12-tool dispatcher. On Bonsai 27B that meant prefilling
        // roughly 5,900 tokens merely to have the model choose the same
        // `article_overview` tool, adding 80–90 seconds on iPhone before
        // retrieval even began. Keep clause-shaped requests ("explain why
        // …", "help me understand how …") in the synthesis loop; a phrase
        // that names a topic can use the compact grounded overview path.
        if let subject = explanatoryOverviewSubject(lower) {
            return DirectIntent(toolName: "article_overview", args: [
                "title": .string(subject)
            ])
        }

        // Date/event factoid OPENERS name their subject and want one fact:
        // "When did Bulgaria join NATO?" as a session's first turn missed
        // every fast path and fell into the LLM tool loop — which on-device
        // costs ~15 s of prefill PER tool round-trip, wandered to
        // "Bulgaria–North Macedonia relations" first, and hit the circuit
        // breaker with no answer delivered (real capture 2026-08-02). The
        // grounded overview path answers the same shape mid-discussion in
        // ~3 s: route the opener there too — the full question rides along,
        // and the key-fact extractor quotes the dated sentence.
        if let m = match(lower, pattern:
            #"^(?:when|what\s+year|what\s+date)\s+(?:did|was|were|does|do|has|have|is)\s+(.+?)\s+(?:join|start|begin|end|becom|become|became|gain|declar|win|won|lose|lost|fall|fell|found|independen|launch|open|close|die|born|built|build|made|create|complet|establish|invent|discover|elect|crown|marri|assassinat|kill|releas|publish|form|sign|enter|leave|left|adopt)"#)
        {
            let subject = m[0].trimmingCharacters(in: .whitespaces)
            let firstWord = subject
                .split(separator: " ", maxSplits: 1)
                .first.map(String.init) ?? ""
            let factoidNavPronouns: Set<String> = [
                "my", "our", "your", "here", "now", "next",
                "this", "that", "these", "those", "it", "he", "she",
                "they", "his", "her", "their", "him", "them", "i", "we",
            ]
            if !subject.isEmpty, !factoidNavPronouns.contains(firstWord) {
                return DirectIntent(toolName: "article_overview", args: [
                    "title": .string(Self.stripPossessiveFacet(from: subject))
                ])
            }
            // Pronoun subject ("when did it join…") falls through — the
            // focus-aware continuation path owns those.
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
        // "how/what about X" reaches here only when the focus-aware
        // resolver did NOT bind — i.e. the turn names its own subject
        // ("How about Donald Trump's childhood?" mid-discussion of
        // Putin). Subject-less "what about his parents" binds upstream
        // and never lands on this pattern; the navPronouns bail below
        // covers the empty-focus case.
        if let m = match(lower, pattern:
            #"^(?:tell\s+me\s+(?:about|more\s+about)|(?:how|what)\s+about|what(?:'s|\s+is|\s+are)|who(?:'s|\s+is|\s+was|\s+were|\s+are)|give\s+me\s+(?:an?\s+)?overview\s+of|overview\s+of)\s+(.+)$"#)
        {
            // Dictation artifacts first: collapse stutter, then cut any
            // trailing interrogative clause — the clause is the QUESTION
            // (which the grounded answer keeps from the full user text),
            // not part of the title ("Tell me about the war of 1812 what
            // were the what were the causes?", device capture 2026-08-03,
            // dispatched the whole tail and search-rescued to "1812
            // Louisiana hurricane").
            let subject = strippingTrailingInterrogativeClause(
                collapseStutter(m[0].trimmingCharacters(in: .whitespaces)))
            let firstWord = subject
                .split(separator: " ", maxSplits: 1)
                .first.map(String.init) ?? ""
            let navPronouns: Set<String> = [
                "my", "our", "your", "here", "now", "next",
                "this", "that", "these", "those", "it",
                "his", "her", "their", "him", "them",
            ]
            if navPronouns.contains(firstWord) { return nil }
            // Subject must have at least one content character — "what
            // is" with nothing after would match `.+` on the trailing
            // "?!." the caller stripped. Guard against that.
            if subject.isEmpty { return nil }
            // "the most recent version of Apple TV" — the ENTITY after
            // "of" is the article title; the attribute phrase is the
            // question, not part of the title (real capture 2026-07-19:
            // "No. What is the most recent version of Apple TV?"
            // dispatched the whole phrase and missed). The grounded
            // answer keeps the full user question, so the facet
            // survives even though the title is reduced.
            // "What's IN dupont circle?" / "what's AROUND adams morgan?"
            // captured the preposition into the title, so the lookup was
            // `article_overview("in dupont circle")` — a title no ZIM
            // holds, salvaged only by search rescue. Strip it so the
            // article path at least asks for the real subject. (Whether
            // these should reach StreetZIM instead of Wikipedia is a
            // separate routing question — the conversational eval expects
            // near_named_place here; surfaced 2026-08-13 when that suite
            // ran for the first time.)
            var reducedSubject = subject
            for preposition in ["in ", "around ", "near ", "at ", "inside "]
            where reducedSubject.hasPrefix(preposition) {
                reducedSubject = String(
                    reducedSubject.dropFirst(preposition.count)
                ).trimmingCharacters(in: CharacterSet.whitespaces)
                break
            }
            if reducedSubject.isEmpty { return nil }
            if let m = match(reducedSubject, pattern: #"^((?:the|its)\s+.+?)\s+of\s+(.+)$"#),
               m.count >= 2,
               ReferenceResolver.isAttributePhrase(m[0])
            {
                reducedSubject = m[1].trimmingCharacters(in: CharacterSet.whitespaces)
            }
            // "Putin's early life" / "Vladimir Putin and his early life"
            // — the ENTITY is the article title; the possessive facet is
            // a section, not part of the title. Dispatching the raw
            // phrase misses ("no article: putin's early life", real
            // capture 2026-07-01) and dead-ends in a did-you-mean.
            // `article_overview(title: entity)` succeeds and its
            // `pickOverview` already prioritises the classic facet
            // sections ("early life", "career", …).
            let title = canonicalArticleSubject(
                Self.stripPossessiveFacet(from: reducedSubject))
            return DirectIntent(toolName: "article_overview", args: [
                "title": .string(title)
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

    /// Recognize a user's explicit Wikipedia source selection.
    ///
    /// Supported shapes include:
    /// - "Use the Wikipedia article on Santa Rosa, California."
    /// - "What does the Santa Rosa, California Wikipedia article say about
    ///   the 1906 earthquake?"
    /// - "Look up the Wikipedia article on Santa Rosa California and tell me
    ///   what it says about the 1906 earthquake."
    /// - "Use Wikipedia, not StreetZIM, for that." (binds to current focus)
    ///
    /// The returned `question` excludes the source-control wrapper so section
    /// retrieval ranks the requested facet rather than words such as "look
    /// up", "article", and "Wikipedia".
    public static func wikipediaSourceDirective(
        _ raw: String,
        focus: ConversationFocus? = nil
    ) -> WikipediaSourceDirective? {
        let text = raw
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(
                of: #"\s+"#, with: " ",
                options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!"))
        guard !text.isEmpty else { return nil }
        let lower = text.lowercased()
        let namesWikipediaArticle =
            lower.range(
                of: #"\bwikipedia\s+(?:article|page)\b|\b(?:article|page)\s+(?:on|from)\s+wikipedia\b"#,
                options: .regularExpression) != nil
        let sourceOnly = lower.range(
            of: #"\b(?:use|from|check|consult|search)\s+(?:the\s+)?(?:offline\s+)?wikipedia\b"#,
            options: .regularExpression) != nil
        guard namesWikipediaArticle || sourceOnly else { return nil }

        func cleaned(_ value: String?) -> String? {
            guard var value else { return nil }
            value = value.trimmingCharacters(
                in: CharacterSet.whitespacesAndNewlines
                    .union(CharacterSet(charactersIn: "\"'.,;:!?")))
            value = value.replacingOccurrences(
                of: #"^(?:the\s+)"#, with: "",
                options: [.regularExpression, .caseInsensitive])
            return value.isEmpty ? nil : value
        }

        // Question-first form: "What does the Santa Rosa, California
        // Wikipedia article say about the 1906 earthquake?"
        if let m = match(text, pattern:
            #"(?i)^what\s+(?:does|do)\s+(?:the\s+)?(.+?)\s+wikipedia\s+(?:article|page)\s+(?:say|says)\s+(?:about|on|regarding)\s+(.+)$"#),
           m.count >= 2,
           let title = cleaned(m[0]),
           let question = cleaned(m[1])
        {
            return WikipediaSourceDirective(title: title, question: question)
        }

        // Article-first question: "In the Wikipedia article on Santa Rosa,
        // California, what does it say about the 1906 earthquake?"
        if let m = match(text, pattern:
            #"(?i)^(?:in|from)\s+(?:the\s+)?(?:offline\s+)?wikipedia\s+(?:article|page)\s+(?:about|on|for)\s+(.+?)[,;]?\s+what\s+(?:does\s+)?(?:it|the\s+(?:article|page))\s+(?:say|says)\s+(?:about|on|regarding)\s+(.+)$"#),
           m.count >= 2,
           let title = cleaned(m[0]),
           let question = cleaned(m[1])
        {
            return WikipediaSourceDirective(title: title, question: question)
        }

        // Direct source search: "Search the Santa Rosa, California
        // Wikipedia article for the 1906 earthquake."
        if let m = match(text, pattern:
            #"(?i)^(?:please\s+)?(?:search|check|consult|look\s+in|read)\s+(?:the\s+)?(.+?)\s+wikipedia\s+(?:article|page)\s+(?:for|about|on|regarding)\s+(.+)$"#),
           m.count >= 2,
           let title = cleaned(m[0]),
           let question = cleaned(m[1])
        {
            return WikipediaSourceDirective(title: title, question: question)
        }

        // Command + question: "Look up the Wikipedia article on Santa Rosa
        // California and tell me what it says about the 1906 earthquake."
        if let m = match(text, pattern:
            #"(?i)^.*?\bwikipedia\s+(?:article|page)\s+(?:about|on|for)\s+(.+?)(?:[.;]\s*|\s+)(?:(?:and|then)\s+|i\s+then\s+want\s+you\s+to\s+)?(?:please\s+)?(?:tell\s+me|read|see|find\s+out|check)\s+.*?\bwhat\s+(?:it|the\s+(?:article|page)|that\s+(?:article|page))\s+(?:say|says)\s+(?:about|on|regarding)\s+(.+)$"#),
           m.count >= 2,
           let title = cleaned(m[0]),
           let question = cleaned(m[1])
        {
            return WikipediaSourceDirective(title: title, question: question)
        }

        // "Use the Wikipedia article on X for information about Y."
        if let m = match(text, pattern:
            #"(?i)^.*?\bwikipedia\s+(?:article|page)\s+(?:about|on|for)\s+(.+?)\s+for\s+(?:information\s+)?(?:about|on|regarding)\s+(.+)$"#),
           m.count >= 2,
           let title = cleaned(m[0]),
           let question = cleaned(m[1])
        {
            return WikipediaSourceDirective(title: title, question: question)
        }

        // Reverse title form: "Use the Santa Rosa, California Wikipedia
        // article as the source."
        if let m = match(text, pattern:
            #"(?i)^.*?\b(?:use|open|consult|check)\s+(?:only\s+)?(?:the\s+)?(.+?)\s+wikipedia\s+(?:article|page)(?:\s+as\s+(?:the\s+)?source)?$"#),
           let captured = m.first,
           let title = cleaned(captured)
        {
            return WikipediaSourceDirective(title: title)
        }

        // Source selection with an explicit article title and no facet.
        if let m = match(text, pattern:
            #"(?i)^.*?\bwikipedia\s+(?:article|page)\s+(?:about|on|for)\s+(.+)$"#),
           let captured = m.first
        {
            var title = captured
            // Stop a two-sentence instruction before its second command even
            // when the speaker omitted the exact "what it says" wording.
            title = title.replacingOccurrences(
                of: #"[.;]\s*(?:i\s+then|then|and|please|now)\b.*$"#,
                with: "", options: [.regularExpression, .caseInsensitive])
            if let title = cleaned(title) {
                return WikipediaSourceDirective(title: title)
            }
        }

        // "Use Wikipedia, not StreetZIM, for that" keeps the current article
        // but explicitly changes the source policy.
        if sourceOnly, let title = focus?.primaryEntity?.name {
            let question: String? = {
                if let m = match(text, pattern:
                    #"(?i)\b(?:for|about|regarding)\s+(.+)$"#),
                   let value = cleaned(m.first),
                   !["that", "this", "it", "the answer"].contains(
                       value.lowercased())
                {
                    return value
                }
                return nil
            }()
            return WikipediaSourceDirective(title: title, question: question)
        }
        return nil
    }

    /// Recognize a request to narrate a complete Wikipedia article. The
    /// returned intent deliberately omits `section_index`: that is the adapter
    /// contract for a full pass-through read, while "continue reading" uses
    /// the separate paged path.
    public static func readArticleIntent(
        _ raw: String,
        focus: ConversationFocus? = nil
    ) -> DirectIntent? {
        var text = raw
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!"))
        guard !text.isEmpty else { return nil }

        text = text.replacingOccurrences(
            of: #"^(?:please\s+|(?:can|could|would)\s+you\s+|i(?:'d|\s+would)\s+like\s+you\s+to\s+)+"#,
            with: "", options: [.regularExpression, .caseInsensitive])
        let pattern = #"^read\s+(?:me\s+)?(?:the\s+)?(?:(?:full|whole|entire|complete)\s+)?(?:wikipedia\s+)?article(?:\s+(?:about|on|for)\s+(.+?))?(?:\s+(?:aloud|to\s+me))?$"#
        guard let match = match(text, pattern: pattern) else { return nil }

        let explicit = match.first?
            .replacingOccurrences(
                of: #"\s+(?:aloud|to\s+me)$"#, with: "",
                options: [.regularExpression, .caseInsensitive])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let title = (explicit?.isEmpty == false)
            ? explicit
            : focus?.primaryEntity?.name
        guard let title, !title.isEmpty else { return nil }
        return DirectIntent(
            toolName: "narrate_article",
            args: ["title": .string(title)])
    }

    private enum ComparisonContinuationRoute {
        case retrieve(section: String?)
        case synthesizeFromContext
    }

    /// Recognize a subject-less follow-up to the most recently displayed
    /// two-topic comparison. The cue must explicitly refer to BOTH subjects;
    /// ordinary list selections ("the second one") remain the resolver's job.
    private static func comparisonContinuationRoute(
        _ lower: String, focus: ConversationFocus
    ) -> ComparisonContinuationRoute? {
        // Only a list that IS a compared pair. A disambiguation offer also
        // leaves two topics in `lastList`, and "how many people died on
        // each side in the war?" then retrieved compare_articles(War of
        // 1812, French invasion of Russia) — Mac replay of the 2026-08-03
        // field session.
        guard focus.lastListKind == .comparison else { return nil }
        let pair = Array(focus.lastList.prefix(2))
        guard pair.count == 2,
              pair.allSatisfy({ $0.kind == .topic })
        else { return nil }

        let pairCues = [
            " each", "each ", "both", "between the two", "the two wars",
            "those two", "which war", "which one", "compared with",
            "compared to",
        ]
        guard pairCues.contains(where: { lower.contains($0) }) else {
            return nil
        }

        // These are synthesis questions over the facts already fetched. A
        // new retrieval adds latency/context but no evidence the model lacks.
        let synthesisCues = [
            "why", "what changed", "how did that", "made ", "make ",
            "more deadly", "more important", "what explains",
        ]
        if synthesisCues.contains(where: { lower.contains($0) }) {
            return .synthesizeFromContext
        }

        let section: String?
        if ["casualt", "killed", "deaths", "fatalit"]
            .contains(where: { lower.contains($0) })
        {
            section = "Casualties"
        } else if ["combatant", "belligerent", "who fought", "which side"]
            .contains(where: { lower.contains($0) })
        {
            section = "Belligerents"
        } else if ["cause", "started", "origins"]
            .contains(where: { lower.contains($0) })
        {
            section = "Causes"
        } else {
            section = nil
        }
        return .retrieve(section: section)
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

    /// Section facets people attach to a subject with a possessive
    /// ("X's early life", "X and her career"). These are classic
    /// Wikipedia section names, never part of the article title —
    /// kept as an explicit whitelist because REAL titles legitimately
    /// contain possessives ("Hitchhiker's Guide to the Galaxy").
    private static let possessiveFacets: Set<String> = [
        "early life", "early years", "early life and education",
        "childhood", "youth", "education", "career", "later life",
        "later years", "personal life", "family", "children",
        "death", "legacy", "history", "biography", "background",
        "rise to power", "presidency", "reign", "discography",
        "filmography", "achievements", "accomplishments", "works",
        "net worth", "wife", "husband", "spouse",
    ]

    /// "putin's early life" → "putin"; "vladimir putin and his early
    /// life" → "vladimir putin". Leaves the subject untouched unless
    /// the trailing phrase is a whitelisted facet.
    static func stripPossessiveFacet(from subject: String) -> String {
        // "<entity>'s <facet>"
        if let r = subject.range(
            of: "'s ", options: [.backwards, .caseInsensitive])
        {
            let facet = String(subject[r.upperBound...]).lowercased()
                .trimmingCharacters(in: .whitespaces)
            if possessiveFacets.contains(facet) {
                return String(subject[..<r.lowerBound])
                    .trimmingCharacters(in: .whitespaces)
            }
        }
        // "<entity> and (his|her|its|their) <facet>"
        if let m = match(subject, pattern:
            #"^(.+?)\s+and\s+(?:his|her|its|their)\s+(.+)$"#),
           m.count >= 2,
           possessiveFacets.contains(
               m[1].lowercased().trimmingCharacters(in: .whitespaces))
        {
            let entityLen = m[0].trimmingCharacters(in: .whitespaces).count
            return String(subject.prefix(entityLen))
                .trimmingCharacters(in: .whitespaces)
        }
        return subject
    }

    /// Aggressive variant for the article-MISS retry: also strips the
    /// apostrophe-less possessive voice dictation produces ("putins
    /// childhood" → "putin"). Too greedy for first-pass routing ("paris
    /// history" → "pari"), so it only runs after the literal title
    /// already missed — a wrong guess there just re-misses into the
    /// same did-you-mean the user would have seen anyway.
    public static func stripPossessiveFacetAggressive(from subject: String) -> String {
        let conservative = stripPossessiveFacet(from: subject)
        if conservative != subject { return conservative }
        if let m = match(subject, pattern: #"^(.+?)s\s+(.+)$"#),
           m.count >= 2,
           possessiveFacets.contains(
               m[1].lowercased().trimmingCharacters(in: .whitespaces)),
           m[0].count >= 4
        {
            return m[0].trimmingCharacters(in: .whitespaces)
        }
        return subject
    }

    /// Drop a leading "tell me about" / "what is" / "who was" / "overview
    /// of" / "about" lead so a restated subject indexes on the entity.
    private static func stripArticleLead(_ s: String) -> String {
        let p =
            #"^(?:tell\s+me\s+(?:about|more\s+about)|what(?:'s|\s+is|\s+are)|who(?:'s|\s+is|\s+was|\s+were|\s+are)|give\s+me\s+(?:an?\s+)?overview\s+of|overview\s+of|about)\s+"#
        return s.replacingOccurrences(
            of: p, with: "", options: .regularExpression)
    }

    /// Collapse immediately-repeated word runs — dictation stutter.
    /// Voice input routinely doubles a phrase mid-turn ("what were the
    /// what were the causes", device capture 2026-08-03), and the doubled
    /// run then poisons downstream title extraction. Longest runs collapse
    /// first so "what were the what were the" loses one full trigram
    /// instead of producing "what were the were the".
    public static func collapseStutter(_ text: String) -> String {
        var tokens = text.split(separator: " ").map(String.init)
        for n in stride(from: 4, through: 1, by: -1) {
            var i = 0
            while i + 2 * n <= tokens.count {
                if Array(tokens[i ..< i + n]) == Array(tokens[i + n ..< i + 2 * n]) {
                    tokens.removeSubrange(i + n ..< i + 2 * n)
                    // Stay at i: a triple repeat needs another pass here.
                } else {
                    i += 1
                }
            }
        }
        return tokens.joined(separator: " ")
    }

    /// Auxiliaries that open an interrogative CLAUSE ("what were …",
    /// "when did …"). Deliberately excludes content verbs so titles like
    /// "The Man Who Sold the World" survive — "sold" is not in this set.
    private static let interrogativeAuxiliaries: Set<String> = [
        "is", "are", "was", "were", "did", "do", "does", "can", "could",
        "will", "would", "should", "has", "have", "had", "am",
    ]
    private static let interrogativeOpeners: Set<String> = [
        "what", "who", "whom", "when", "where", "why", "how", "which",
    ]

    /// Trim a trailing interrogative clause off an extracted article
    /// subject. "Tell me about the war of 1812 what were the causes?"
    /// captures the WHOLE tail as the title, misses the ZIM, and search-
    /// rescue lands on "1812 Louisiana hurricane" (device capture
    /// 2026-08-03). The clause cut requires interrogative + auxiliary so
    /// a title merely containing a question word ("Doctor Who") is never
    /// truncated; the caller keeps the full user text as the grounded
    /// question, so the facet the clause asked about survives.
    public static func strippingTrailingInterrogativeClause(_ subject: String) -> String {
        let tokens = subject.split(separator: " ").map(String.init)
        guard tokens.count >= 3 else { return subject }
        for i in 1 ..< (tokens.count - 1) {
            let word = tokens[i].lowercased()
                .trimmingCharacters(in: CharacterSet(charactersIn: ",;:"))
            guard interrogativeOpeners.contains(word),
                  interrogativeAuxiliaries.contains(tokens[i + 1].lowercased())
            else { continue }
            let head = tokens[..<i].joined(separator: " ")
                .trimmingCharacters(in: CharacterSet(charactersIn: " ,;:-–—"))
            return head.isEmpty ? subject : head
        }
        return subject
    }

    /// True when a stateless-parsed article `title` actually names the
    /// subject already in hand. The raw substring check misses anaphora
    /// with a leading article: "How many people died on each side in the
    /// war?" parses a title of "the war", which is not a substring of
    /// "war of 1812" — so the turn LEFT the pinned discussion, hit the
    /// ambiguity gate, and threw away a warm KV cache (device capture
    /// 2026-08-03). Article-strip the title and accept a token-subset
    /// match: "war" ⊆ {war, of, 1812} stays; "art" ⊄ {stuttgart} still
    /// leaves, because tokens — unlike substrings — respect word bounds.
    public static func titleNamesPinnedSubject(
        _ title: String, inHand: [String]
    ) -> Bool {
        let lower = title.lowercased().trimmingCharacters(in: .whitespaces)
        guard !lower.isEmpty else { return false }
        if inHand.contains(where: { $0.contains(lower) || lower.contains($0) }) {
            return true
        }
        var tokens = lower.split(separator: " ").map(String.init)
        if let first = tokens.first, ["the", "a", "an"].contains(first) {
            tokens.removeFirst()
        }
        guard !tokens.isEmpty else { return false }
        let titleTokens = Set(tokens)
        return inHand.contains { name in
            titleTokens.isSubset(of: Set(name.split(separator: " ").map(String.init)))
        }
    }

    /// Normalize a few high-confidence spoken event aliases whose literal
    /// wording otherwise resolves to a similarly named side article. Keep the
    /// list deliberately tiny: this runs before Wikipedia disambiguation and
    /// therefore must never guess on a merely similar title.
    private static func canonicalArticleSubject(_ raw: String) -> String {
        let subject = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = subject.lowercased().replacingOccurrences(
            of: #"^the\s+"#, with: "", options: .regularExpression)
        let pearlHarborAliases: Set<String> = [
            "japanese attack on pearl harbor",
            "japanese attack at pearl harbor",
            "pearl harbor attack",
            "attack at pearl harbor",
            "battle of pearl harbor",
        ]
        return pearlHarborAliases.contains(normalized)
            ? "attack on pearl harbor"
            : subject
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

        // Locational follow-up. The resolver binds pronouns to the
        // most-recent entity of ANY kind, but "how far is it?" after
        // "restaurants near the Ferry Building" then "tell me about
        // Ohlone history" means the PLACE, not the topic — so when the
        // turn is locational and the bound entity isn't a locatable
        // place, rebind to the most recent place in focus.
        //
        // Phrase split:
        //  * distance-shaped ("how far", "which way", "can I walk") →
        //    `distance_to` — a distance + compass-direction + walk-estimate
        //    answer, cheap, no routing-graph work;
        //  * travel-shaped ("directions", "how do I get", "how long",
        //    "get to") → `route_from_places` — the routed reply carries the
        //    real driving duration;
        //  * proximity-shaped ("near it", "around") → `near_places`.
        let distanceWords = ["how far", "distance", "how close",
                             "which way", "which direction",
                             "can i walk", "walkable", "walking distance",
                             "walk there", "can i drive", "drive there",
                             "how long to walk", "how long to drive"]
        let proximityWords = ["near", "around", "nearby", "close by",
                              "close to", "what's close"]
        let routeWords = ["directions", "route", "how do i get",
                          "get to", "how long"]
        let isLocational = (distanceWords + proximityWords + routeWords)
            .contains { lower.contains($0) }
        if isLocational {
            let place: FocusEntity? = (entity.kind == .place)
                ? entity
                : focus.mostRecent(kind: .place)
            if let place {
                if distanceWords.contains(where: { lower.contains($0) }) {
                    // "how far / which way is it" wants a distance +
                    // direction answer, not a POI dump around the place.
                    var args: [String: AnyJSONValue] = [
                        "place": .string(place.name),
                    ]
                    if let lat = place.lat, let lon = place.lon {
                        args["lat"] = .double(lat)
                        args["lon"] = .double(lon)
                    }
                    return DirectIntent(toolName: "distance_to", args: args)
                }
                if routeWords.contains(where: { lower.contains($0) }) {
                    return DirectIntent(toolName: "route_from_places", args: [
                        "origin":      .string("my location"),
                        "destination": .string(place.name),
                    ])
                }
                if let lat = place.lat, let lon = place.lon {
                    return DirectIntent(toolName: "near_places", args: [
                        "lat":       .double(lat),
                        "lon":       .double(lon),
                        "radius_km": .double(1),
                    ])
                }
            }
            // No locatable place in focus — fall through to the
            // encyclopedic default below (better than guessing coords).
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

    /// True when a sentence is shaped like a request for knowledge rather
    /// than a compact POI category. This is also a defense-in-depth signal for
    /// hosts with a pinned article: a correction containing a preposition must
    /// not eject the conversation into StreetZIM routing.
    public static func isConversationalKnowledgeRequest(_ raw: String) -> Bool {
        var text = raw.lowercased()
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        text = text.replacingOccurrences(
            of: #"^(?:no|nope|wait|actually|sorry)[,.:;\s]+"#,
            with: "", options: .regularExpression)
        let opener = #"^(?:what|which|who|when|where|why|how)\b|^(?:i|we)\s+(?:am\s+|are\s+)?(?:want(?:ing)?|trying|asking)\s+to\s+(?:know|learn|understand|find\s+out)\b|^(?:can|could|would)\s+you\s+(?:tell|explain|look\s+up|find\s+out)\b"#
        return text.range(of: opener, options: .regularExpression) != nil
    }

    /// Whether the speaker explicitly presents a fresh subject, rather than
    /// merely asking another question while a prepared discussion is pinned.
    /// Hosts combine this with evidence coverage: an implicit article-looking
    /// parse only leaves the topic when the current article cannot cover it.
    public static func isExplicitDiscussionTopicChange(_ raw: String) -> Bool {
        // Deictic words bind the requested facet back to the prepared topic.
        // For example, "Tell me about Buddhism there" is about Buddhism in
        // Mongolia, not a request to open an article titled "Buddhism there".
        if isDiscussionDeicticFollowUp(raw) { return false }
        let t = raw.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let pattern = #"^(?:please\s+)?(?:tell me about|talk about|let(?:'|’)s talk about|discuss|let(?:'|’)s discuss|switch(?: the)? topic to|change(?: the)? topic to|move on to)\s+\S"#
        return t.range(of: pattern, options: .regularExpression) != nil
    }

    /// Whether an utterance explicitly points back to the subject already in
    /// discourse. Keep this deliberately narrower than a bare "it": these
    /// forms are reliable topic pins without turning genuine hand-offs such
    /// as "Tell me about Theravada Buddhism" into follow-ups.
    public static func isDiscussionDeicticFollowUp(_ raw: String) -> Bool {
        let t = raw.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let pattern = #"\b(?:there|its|their)\b|\b(?:in|within|under|during|across|throughout)\s+(?:that|this)\s+(?:country|place|region|area|society|culture|government|period|era|article|topic)\b"#
        return t.range(of: pattern, options: .regularExpression) != nil
    }

    /// A conversational facet hand-off that normally inherits the prepared
    /// subject: "How about Buddhism?", "And what about Christianity?".
    /// This is deliberately only a SHAPE signal. The host still requires the
    /// pinned article to mention the requested facet before resisting a new
    /// article route, so "How about Donald Trump?" can leave an unrelated
    /// discussion while "How about Christianity?" stays within Lithuania.
    public static func isEllipticalDiscussionFollowUp(_ raw: String) -> Bool {
        let t = raw.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let pattern = #"^(?:(?:and|so|then)\s+)?(?:how|what)\s+about\b"#
        return t.range(of: pattern, options: .regularExpression) != nil
    }

    /// Whether a stateless article-looking title is actually a common facet
    /// of the subject already pinned by discussion mode. For example, the
    /// generic router can parse "Who were the combatants?" as an article
    /// named "the combatants"; the host uses this signal to keep the current
    /// topic and retrieve its combatant evidence instead.
    public static func isDiscussionFacetTitle(_ raw: String) -> Bool {
        let lowered = raw.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // A possessive pronoun explicitly points back to the subject already
        // in discourse. The remainder is open-ended ("its early history",
        // "their economic policy"), so an exact facet allow-list cannot
        // safely enumerate it. Treat these as pinned-topic facets before the
        // stateless article router can reinterpret the words as a new title.
        if ["his ", "her ", "their ", "its "].contains(where: {
            lowered.hasPrefix($0)
        }) {
            return true
        }
        let normalized = lowered
            .replacingOccurrences(
                of: #"^(?:the|his|her|their|its)\s+"#,
                with: "", options: .regularExpression)
        let facets: Set<String> = [
            "background", "causes", "combatants", "belligerents",
            "participants", "commanders", "casualties", "deaths",
            "aftermath", "outcome", "parents", "family", "school",
            "education", "career", "legacy", "effects", "sources",
            "formation", "detection", "history", "early history",
            "modern history", "culture", "economy", "religion", "wealth",
            "population", "demographics", "geography", "climate",
            "government", "politics", "government and politics",
            "languages", "language", "ethnic groups", "territory", "capital",
            "sports", "festivals", "sports and festivals",
        ]
        return facets.contains(normalized)
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
    private static func explanatoryOverviewSubject(_ lower: String) -> String? {
        let patterns = [
            #"^(?:(?:can|could|would|will)\s+you\s+)?explain\s+(.+?)(?:\s+to\s+me)?$"#,
            #"^(?:(?:can|could|would|will)\s+you\s+)?teach\s+me\s+about\s+(.+)$"#,
            #"^(?:(?:can|could|would|will)\s+you\s+)?help\s+me\s+understand\s+(.+)$"#,
        ]
        var subject: String?
        for pattern in patterns {
            if let m = match(lower, pattern: pattern) {
                subject = m[0].trimmingCharacters(in: .whitespaces)
                break
            }
        }
        guard var subject, !subject.isEmpty else { return nil }

        // These lead into a proposition rather than naming an article.
        // The general synthesis loop should reason over them instead of
        // looking up an article literally titled "why the sky is blue".
        let clauseLeads = ["why ", "how ", "whether ", "what happens "]
        if clauseLeads.contains(where: { subject.hasPrefix($0) }) {
            return nil
        }

        let navPronouns: Set<String> = [
            "it", "this", "that", "these", "those", "him", "her", "them",
            "my", "our", "your",
        ]
        let firstWord = subject
            .split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
        if navPronouns.contains(firstWord) { return nil }

        // A natural "explain the Standard Model to me" should search the
        // encyclopedia title, not a literal leading article.
        subject = subject.replacingOccurrences(
            of: #"^(?:a|an|the)\s+"#, with: "",
            options: [.regularExpression, .caseInsensitive])
        return subject.trimmingCharacters(in: .whitespaces)
    }

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
        guard let regex = RegexCache.shared.compiled(pattern, options: []) else {
            return false
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }

    /// Light regex helper — returns only the capture groups (not the
    /// full-match range). `NSRegularExpression` verbatim with a
    /// `nil`-to-`Substring[]` adapter.
    private static func match(_ text: String, pattern: String) -> [String]? {
        guard let regex = RegexCache.shared.compiled(
            pattern, options: [.caseInsensitive]
        ) else {
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

    /// "How old is X?" is usually a founding-age question for cities,
    /// institutions, and companies. Route it to the same grounded fact
    /// extractor, but mark it tentative: people and other subjects without a
    /// foundation event must fall back to the general article path rather
    /// than receiving a misleading "not found" terminal answer.
    private static func ageFactoidIntent(
        _ text: String, focus: ConversationFocus?
    ) -> DirectIntent? {
        guard let captures = match(text, pattern:
            #"^(?:how\s+(?:many\s+years\s+)?old\s+(?:is|was)|what(?:'s|\s+is)\s+the\s+age\s+of)\s+(.+)$"#),
              captures.count == 1
        else { return nil }

        var subject = captures[0].trimmingCharacters(in: .whitespaces)
        let referential = ["it", "this", "that", "the place", "the company"]
        if referential.contains(subject.lowercased()),
           let focused = focus?.primaryEntity?.name,
           !focused.isEmpty {
            subject = focused
        }
        guard subject.count >= 2, !referential.contains(subject.lowercased()) else {
            return nil
        }
        return DirectIntent(toolName: "article_factoid", args: [
            "title": .string(subject),
            "predicate": .string("age"),
            "implicit": .bool(false),
            "tentative": .bool(true),
        ])
    }

    /// Reapply a pending factoid predicate after the user chooses one item
    /// from a clarification list ("the second one", "the state", or the
    /// exact label). Without this, the list resolver would open a generic
    /// article overview and silently forget the original "How old...?".
    public static func factoidSelectionIntent(
        _ text: String, predicate: String, focus: ConversationFocus
    ) -> DirectIntent? {
        guard predicate == "age" || predicate == "foundation" else {
            return nil
        }
        let resolved = ReferenceResolver.resolve(text, focus: focus)
        guard case .listSelection(_, let entity) = resolved.binding else {
            return nil
        }
        var args: [String: AnyJSONValue] = [
            "title": .string(entity.name),
            "predicate": .string(predicate),
            "implicit": .bool(false),
        ]
        if predicate == "age" { args["tentative"] = .bool(true) }
        return DirectIntent(toolName: "article_factoid", args: args)
    }

    /// Recognise only founding-date questions whose answers can be extracted
    /// verbatim from an article lead. Broad "when" questions deliberately
    /// remain on the normal grounded path: dates of battles, deaths, terms in
    /// office, etc. need different evidence rules.
    private static func foundingFactoidIntent(
        _ lower: String, focus: ConversationFocus?
    ) -> DirectIntent? {
        let explicit = match(lower, pattern:
            #"^(?:when\s+(?:was|were)|what\s+year\s+(?:was|were)|in\s+what\s+year\s+(?:was|were))\s+(.+?)\s+(?:first\s+)?(founded|established|formed|created|incorporated)$"#)
        // Voice recognition and casual typed questions sometimes stop at
        // "When was Apple?". Treat that as a *tentative* company-founding
        // request. The adapter applies a second, strict organization-language
        // gate and falls back to normal routing on "When was the Alamo?".
        let implicit = explicit == nil
            ? match(lower, pattern: #"^when\s+was\s+(.+)$"#)
            : nil
        guard let captures = explicit ?? implicit,
              explicit != nil ? captures.count == 2 : captures.count == 1
        else { return nil }

        var subject = captures[0].trimmingCharacters(in: .whitespaces)
        let referentialSubjects: Set<String> = [
            "it", "they", "the company", "the organization",
            "the organisation", "the business", "the school", "the city",
        ]
        if referentialSubjects.contains(subject),
           let focused = focus?.primaryEntity?.name,
           !focused.isEmpty {
            subject = focused
        }
        guard subject.count >= 2, !referentialSubjects.contains(subject) else {
            return nil
        }
        return DirectIntent(toolName: "article_factoid", args: [
            "title": .string(subject),
            "predicate": .string("foundation"),
            "implicit": .bool(explicit == nil),
        ])
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

        // Lead with the CLOSEST hit by name — "the nearest coffee shop is
        // Blue Bottle, 250 m north-east" is the conversational answer;
        // "Found 186 coffee shops… tap a pin" reads like a UI caption and
        // is useless spoken aloud (real capture 2026-07-02). The map/list
        // still shows the rest.
        var nearest = ""
        if kind != "places",
           let rows = fullResult["results"] as? [[String: Any]],
           let top = rows.first,
           let name = (top["name"] as? String) ?? (top["label"] as? String) {
            var bits = "The nearest \(kind) is \(name)"
            if let d = (top["distance_m"] as? NSNumber)?.doubleValue {
                bits += d < 1000
                    ? ", \(Int(d)) m"
                    : String(format: ", %.1f km", d / 1000)
                if let dir = top["direction"] as? String { bits += " \(dir)" }
            }
            nearest = bits + ". "
        }
        var line: String
        if let n = count, n > 0 {
            if nearest.isEmpty {
                line = "Found \(n) \(kindPlural) near \(where_)"
            } else if n > 1 {
                line = nearest + "\(n - 1) more \(kindPlural) near \(where_)"
            } else {
                line = nearest + "It's the only one near \(where_)"
            }
        } else if count == 0 {
            line = "No \(kindPlural) found near \(where_)"
        } else {
            line = nearest + "Results for \(kindPlural) near \(where_)"
        }
        if let r = radiusKm { line += " (within \(formatKm(r)))" }
        line += " — they're on the map below."
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

    /// Validate the internal `article_factoid` result before the host treats
    /// it as terminal. A title resolution alone is insufficient: the result
    /// must carry the evidence sentence selected from the Wikipedia lead.
    public static func articleFactoidResultIsUsable(
        _ fullResult: [String: Any]
    ) -> Bool {
        if let err = fullResult["error"] as? String, !err.isEmpty { return false }
        let fact = (fullResult["fact"] as? String) ?? ""
        return !fact.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Render the already-grounded fact without invoking the language model.
    public static func synthesizeArticleFactoidReply(
        args: [String: Any], fullResult: [String: Any]
    ) -> String {
        if (fullResult["ambiguous"] as? Bool) == true {
            let title = (fullResult["requested_title"] as? String)
                ?? (args["title"] as? String) ?? "that subject"
            let suggestions = (fullResult["suggestions"] as? [String]) ?? []
            if !suggestions.isEmpty {
                let names = Array(suggestions.prefix(3))
                let list: String
                if names.count == 1 {
                    list = names[0]
                } else if names.count == 2 {
                    list = "\(names[0]) or \(names[1])"
                } else {
                    list = names.dropLast().joined(separator: ", ")
                        + ", or \(names.last!)"
                }
                return "Which “\(title)” did you mean — \(list)?"
            }
            return "Which “\(title)” did you mean? Please add a city, state, country, person, or other qualifier."
        }
        if let fact = fullResult["fact"] as? String,
           !fact.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return fact
        }
        let title = (fullResult["requested_title"] as? String)
            ?? (args["title"] as? String) ?? "that subject"
        return "I couldn't find a founding date for \(title) in the offline Wikipedia."
    }

    /// Locate a sentence (or adjacent sentence pair) containing both a
    /// founding synonym and a four-digit year. This is intentionally strict:
    /// no year means no deterministic answer. `title` is used only to replace
    /// an opening "It" / "The company" with an unambiguous spoken subject.
    public static func extractFoundationFact(
        from rawText: String, title: String? = nil
    ) -> String? {
        var text = ArticleHeuristics.stripCitations(rawText)
        text = text.replacingOccurrences(
            of: #"\s+"#, with: " ", options: .regularExpression
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }

        // A deceased person's lead normally opens with a parenthesized
        // birth–death range. This extractor is for the age of places and
        // organizations; do not accidentally age the person from a company,
        // government, or institution they later established.
        let opener = String(text.prefix(500))
        if opener.range(
            of: #"\([^)]*\b(?:1[0-9]{3}|20[0-9]{2})\b[^)]*[–—][^)]*\b(?:1[0-9]{3}|20[0-9]{2})\b[^)]*\)"#,
            options: .regularExpression
        ) != nil {
            return nil
        }

        let sentences = factoidSentences(text)
        guard !sentences.isEmpty else { return nil }
        let verbRegex = RegexCache.shared.compiled(
            #"\b(?:founded|established|formed|created|incorporated)\b"#,
            options: [.caseInsensitive])
        let yearRegex = RegexCache.shared.compiled(
            #"\b(?:1[0-9]{3}|20[0-9]{2})\b"#)
        func contains(_ regex: NSRegularExpression?, in string: String) -> Bool {
            guard let regex else { return false }
            return regex.firstMatch(
                in: string, range: NSRange(string.startIndex..., in: string)) != nil
        }
        func hasVerb(_ s: String) -> Bool {
            contains(verbRegex, in: s)
        }
        func hasYear(_ s: String) -> Bool {
            contains(yearRegex, in: s)
        }

        for i in sentences.indices {
            var candidate: String?
            if hasVerb(sentences[i]), hasYear(sentences[i]) {
                candidate = sentences[i]
            } else if hasVerb(sentences[i]), i + 1 < sentences.count,
                      hasYear(sentences[i + 1]) {
                candidate = sentences[i] + " " + sentences[i + 1]
            } else if hasYear(sentences[i]), i + 1 < sentences.count,
                      hasVerb(sentences[i + 1]) {
                candidate = sentences[i] + " " + sentences[i + 1]
            }
            guard var fact = candidate else { continue }
            if let title, !title.isEmpty {
                let lower = fact.lowercased()
                for prefix in ["the company ", "the organization ",
                               "the organisation ", "it "] {
                    if lower.hasPrefix(prefix) {
                        fact = title + " " + fact.dropFirst(prefix.count)
                        break
                    }
                }
            }
            if fact.count > 500 {
                fact = String(fact.prefix(500))
                    .trimmingCharacters(in: .whitespaces) + "…"
            }
            return fact
        }
        return nil
    }

    /// Ground a place's practical "age" in the start of its modern
    /// settlement when Wikipedia states that more clearly than a formal
    /// founding date. This deliberately requires a dated arrival/landing
    /// immediately tied to a settlement sentence, so an explorer merely
    /// visiting the area is not mistaken for the city's origin. Formal
    /// founding/incorporation remains the fallback.
    public static func extractPlaceOriginFact(
        from rawText: String, title: String? = nil
    ) -> String? {
        var text = ArticleHeuristics.stripCitations(rawText)
        text = text.replacingOccurrences(
            of: #"\s+"#, with: " ", options: .regularExpression
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }

        let sentences = factoidSentences(text)
        let yearRegex = RegexCache.shared.compiled(
            #"\b(?:1[0-9]{3}|20[0-9]{2})\b"#)
        let foundingRegex = RegexCache.shared.compiled(
            #"\b(?:founded|founding\s+date)\b"#, options: [.caseInsensitive])
        let statehoodRegex = RegexCache.shared.compiled(
            #"\b(?:admitted\s+to\s+(?:the\s+)?union|became\s+(?:a|the)\s+state|statehood)\b"#,
            options: [.caseInsensitive])
        let arrivalRegex = RegexCache.shared.compiled(
            #"\b(?:arrived|landed)\b"#, options: [.caseInsensitive])
        let settlementRegex = RegexCache.shared.compiled(
            #"\bsettlement\b"#, options: [.caseInsensitive])
        let namedRegex = RegexCache.shared.compiled(
            #"\b(?:named|moved|established|founded)\b"#, options: [.caseInsensitive])
        let incorporationRegex = RegexCache.shared.compiled(
            #"\b(?:incorporated|re-incorporated)\b"#, options: [.caseInsensitive])
        let placeTypeRegex = RegexCache.shared.compiled(
            #"\b(?:city|town)\b"#, options: [.caseInsensitive])
        func contains(_ regex: NSRegularExpression?, in string: String) -> Bool {
            guard let regex else { return false }
            return regex.firstMatch(
                in: string, range: NSRange(string.startIndex..., in: string)) != nil
        }
        func hasYear(_ sentence: String) -> Bool {
            contains(yearRegex, in: sentence)
        }

        // Prefer an explicit founding statement in the same sentence.
        for sentence in sentences where hasYear(sentence) {
            if contains(foundingRegex, in: sentence) {
                return sentence
            }
        }

        // For states, legal admission/statehood is the meaningful age basis;
        // an earlier treaty may describe how the territory was acquired but
        // not when the state came into existence.
        for sentence in sentences where hasYear(sentence) {
            if contains(statehoodRegex, in: sentence) {
                return sentence
            }
        }

        // Real city leads often define the modern settlement as a dated
        // settlers' arrival followed immediately by "The settlement ...".
        for i in sentences.indices where hasYear(sentences[i]) {
            guard contains(arrivalRegex, in: sentences[i]) else {
                continue
            }
            let next = i + 1 < sentences.count ? sentences[i + 1] : ""
            let context = sentences[i] + " " + next
            guard contains(settlementRegex, in: context),
                  contains(namedRegex, in: context)
            else { continue }
            return context.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // If no settlement origin is stated, a formal city/town
        // incorporation is a clear and reproducible age basis.
        for sentence in sentences where hasYear(sentence) {
            if contains(incorporationRegex, in: sentence),
               contains(placeTypeRegex, in: sentence) {
                return sentence
            }
        }
        return extractFoundationFact(from: text, title: title)
    }

    /// Lightweight sentence segmentation with abbreviation protection. The
    /// generic `. ` split breaks company names at "Inc." immediately before
    /// the founding verb, which is precisely the evidence we need to retain.
    private static func factoidSentences(_ text: String) -> [String] {
        let abbreviations: Set<String> = [
            "inc", "co", "corp", "ltd", "llc", "plc", "u.s", "u.k",
            "mr", "mrs", "ms", "dr", "prof", "st", "no", "vs",
        ]
        var out: [String] = []
        var start = text.startIndex
        var i = text.startIndex
        while i < text.endIndex {
            let c = text[i]
            guard c == "." || c == "!" || c == "?" else {
                i = text.index(after: i)
                continue
            }
            let after = text.index(after: i)
            guard after == text.endIndex || text[after].isWhitespace else {
                i = after
                continue
            }
            let prefix = text[start..<i]
            let token = prefix.split(whereSeparator: { $0.isWhitespace }).last
                .map { String($0).lowercased()
                    .trimmingCharacters(in: .punctuationCharacters) } ?? ""
            let dottedInitialism = token.contains(".")
            if c == ".", (abbreviations.contains(token)
                          || token.count == 1 || dottedInitialism) {
                i = after
                continue
            }
            let end = after
            let sentence = String(text[start..<end])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { out.append(sentence) }
            start = end
            while start < text.endIndex, text[start].isWhitespace {
                start = text.index(after: start)
            }
            i = start
        }
        if start < text.endIndex {
            let tail = String(text[start...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty { out.append(tail) }
        }
        return out
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
