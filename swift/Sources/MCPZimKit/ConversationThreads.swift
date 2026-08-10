// SPDX-License-Identifier: MIT
//
// The deterministic half of the hybrid topic-drift engine.
//
// Product decision: the model only PHRASES the "where to go next" offer; the
// *candidates* are extracted and vetted in Swift so they can never be a
// hallucinated entity that isn't in the loaded ZIMs. This file is that
// extractor:
//
//   * `WikiLinks.parse(html:)` pulls the real outbound article links from a
//     Kiwix Wikipedia article body — these are the genuine adjacencies the
//     conversation can follow ("the architect", "the war it's named after").
//   * `extract(toolName:result:)` turns any tool result dictionary into
//     `DiscoveryThread`s (POIs from a places result, section headings + links
//     from an article result).
//   * `rank(_:focus:)` dedupes against what's already been discussed and caps
//     the list so the offer stays short.
//   * `offer(_:)` is a deterministic fallback caption for the no-LLM fast
//     path; the LLM path gets the same threads via the preamble and phrases
//     its own offer.
//
// Pure value-in / value-out. Foundation-only; no UI, no ZIM access.

import Foundation

public enum ConversationThreads {

    // MARK: - Extraction from tool results

    /// Build vetted threads from a tool's raw JSON result. `toolName` selects
    /// the shape we expect; unknown tools yield no threads (safe default).
    public static func extract(
        toolName: String,
        result: [String: Any]
    ) -> [DiscoveryThread] {
        switch toolName {
        case "locate", "near_named_place", "near_places",
             "nearby_stories", "nearby_stories_at_place":
            return placesThreads(result)
        case "article_overview", "article_factoid", "get_article_section",
             "get_article_by_title", "narrate_article":
            return articleThreads(result)
        case "compare_articles":
            return compareThreads(result)
        case "what_is_here":
            return whatIsHereThreads(result)
        default:
            return []
        }
    }

    /// POIs the user was just shown become "go there / read about it" threads.
    private static func placesThreads(_ result: [String: Any]) -> [DiscoveryThread] {
        let rows = (result["results"] as? [[String: Any]])
            ?? (result["stories"] as? [[String: Any]])
            ?? []
        var out: [DiscoveryThread] = []
        for row in rows {
            let label = firstString(row, "wiki_title", "label", "name", "title")
            guard let label, !label.isEmpty else { continue }
            let lat = doubleField(row, "lat")
            let lon = doubleField(row, "lon")
            let path = row["wiki_path"] as? String
            var note: String?
            if let d = doubleField(row, "distance_m") {
                note = distanceNote(d)
            }
            out.append(DiscoveryThread(
                label: label,
                kind: .place,
                source: .nearbyPlace,
                zimPath: path,
                lat: lat, lon: lon,
                note: note
            ))
        }
        return out
    }

    /// "Where am I?" surfaces the wiki-backed places AROUND the user (the
    /// reverse-geocode's runners-up, carried under `nearby`) as "want to hear
    /// about it?" threads. The nearest place itself becomes the focus subject,
    /// so the drift is to its neighbours — not back to itself.
    private static func whatIsHereThreads(_ result: [String: Any]) -> [DiscoveryThread] {
        let rows = (result["nearby"] as? [[String: Any]]) ?? []
        var out: [DiscoveryThread] = []
        for row in rows {
            guard let label = firstString(row, "wiki_title", "label", "name", "title"),
                  !label.isEmpty else { continue }
            // `.topic`, not `.place`: these neighbours carry a wiki cross-ref
            // rather than a clean ZIM path, so the offer filter would drop a
            // path-less `.place` thread. As topics they read as "hear about
            // it" (→ article_overview) while still carrying coords.
            out.append(DiscoveryThread(
                label: label,
                kind: .topic,
                source: .nearbyPlace,
                zimPath: row["wiki_path"] as? String,
                lat: doubleField(row, "lat"),
                lon: doubleField(row, "lon"),
                note: doubleField(row, "distance_m").map(distanceNote)))
        }
        return out
    }

    /// An article result offers two kinds of drift: its outbound wikilinks
    /// (lateral moves to related subjects) and its remaining section headings
    /// (deeper moves within the subject). Links lead, since a fresh subject
    /// is what makes a conversation "move".
    private static func articleThreads(_ result: [String: Any]) -> [DiscoveryThread] {
        var out: [DiscoveryThread] = []

        // Outbound links: prefer a pre-extracted `links`/`related` array
        // (host can attach one when it has the raw HTML); otherwise parse any
        // `html` field we were handed.
        if let related = result["related"] as? [[String: Any]] {
            for r in related {
                if let label = firstString(r, "title", "label", "name"),
                   !label.isEmpty {
                    out.append(DiscoveryThread(
                        label: label, kind: .topic, source: .wikilink,
                        zimPath: r["path"] as? String ?? r["href"] as? String
                    ))
                }
            }
        } else if let links = result["links"] as? [String] {
            for l in links where !l.isEmpty {
                out.append(DiscoveryThread(
                    label: l, kind: .topic, source: .wikilink))
            }
        } else if let html = result["html"] as? String {
            for link in WikiLinks.parse(html: html, max: 8) {
                out.append(DiscoveryThread(
                    label: link.title, kind: .topic, source: .wikilink,
                    zimPath: link.path))
            }
        }

        // Section headings as "deeper" threads.
        if let sections = result["sections"] as? [[String: Any]] {
            for s in sections {
                guard let rawTitle = s["title"] as? String else { continue }
                let title = rawTitle.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !shouldSkipSection(title)
                else { continue }
                out.append(DiscoveryThread(
                    label: "What about \(title.lowercased())?",
                    kind: .topic,
                    source: .section,
                    prompt: "What about \(title.lowercased())?"))
            }
        }
        return out
    }

    // MARK: - Contextual follow-up questions

    /// Turn real article headings into natural, tappable follow-up questions.
    /// Unlike lateral wikilink offers, these keep the current discussion
    /// pinned and ask about a facet the loaded article can actually answer.
    /// The heading remains the source of truth; phrasing is deterministic so
    /// the chips can never invent an unsupported subject.
    public static func contextualQuestions(
        topic: String,
        sections: [ArticleSection],
        after question: String,
        max: Int = 3
    ) -> [DiscoveryThread] {
        guard max > 0 else { return [] }
        func termSet(_ rawTerms: [String]) -> Set<String> {
            Set(rawTerms.flatMap {
                $0.components(separatedBy: CharacterSet.alphanumerics.inverted)
            }.filter { $0.count >= 3 }.map(stem))
        }
        let topicTerms = termSet(ArticleHeuristics.questionKeywords(topic))
        var asked = termSet(ArticleHeuristics.questionKeywords(question))
        asked.subtract(topicTerms)
        // Expand only the facet the user explicitly asked about. Broad
        // retrieval synonyms are wrong here: "Tell me about the Battle of
        // the Alamo" used to expand "battle" into casualties/combatants and
        // suppress exactly the useful next questions.
        let q = question.lowercased()
        let askedFacetAliases: [([String], [String])] = [
            (["school", "university", "college", "education"], ["school", "university", "education"]),
            (["parent", "mother", "father", "family"], ["parent", "mother", "father", "family", "early life"]),
            (["died", "dead", "death", "killed", "casualt", "fatalit"], ["death", "killed", "casualty", "casualties", "fatality", "fatalities", "losses"]),
            (["combatant", "belligerent", "who fought", "which side"], ["combatant", "belligerent", "army", "forces", "troops", "defenders"]),
            (["detect", "observ"], ["detection", "observation", "experiment"]),
            (["created", "create", "formed", "formation", "produced"], ["source", "formation", "production", "binary"]),
        ]
        for (cues, aliases) in askedFacetAliases
        where cues.contains(where: { q.contains($0) }) {
            asked.formUnion(termSet(aliases))
        }
        var seen = Set<String>()
        var candidates: [(score: Int, order: Int, thread: DiscoveryThread)] = []
        let normalizedHeadings = sections.map {
            $0.title.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let topicWords = Set(topic.lowercased().components(
            separatedBy: CharacterSet.alphanumerics.inverted))
        let eventWords: Set<String> = [
            "battle", "war", "siege", "revolution", "massacre",
            "uprising", "conflict", "campaign", "attack",
        ]
        let isNamedEvent = !topicWords.isDisjoint(with: eventWords)
        let hasLifeHeading = normalizedHeadings.contains { h in
            ["early life", "childhood", "personal life", "family"]
                .contains(where: { h.contains($0) })
        }
        let hasPersonContextHeading = normalizedHeadings.contains { h in
            ["career", "presidency", "public image", "historical evaluation",
             "legacy", "relationship"]
                .contains(where: { h.contains($0) })
        }
        // Countries, cities, and institutions routinely have an Education
        // section. That alone cannot make the subject a person: Mongolia was
        // offered “Where did Mongolia go to school?” and biography-style
        // legacy wording. Require both a life/family marker and another
        // person-context marker.
        let isBiography = !isNamedEvent && hasLifeHeading
            && hasPersonContextHeading

        for (order, section) in sections.enumerated() {
            let raw = section.title.trimmingCharacters(in: .whitespacesAndNewlines)
            let lower = raw.lowercased()
            guard !shouldSkipSection(raw) else { continue }
            if isNamedEvent,
               ["nato", "foreign policy", "the west", "education",
                "personal life", "family", "public image", "poll", "ranking"]
                .contains(where: { lower.contains($0) }) {
                continue
            }
            // A bare "History" heading inside a biography is normally a
            // template/subtopic artifact and produced the contextless chip
            // "How did its history unfold?". Biography-specific facets are
            // both more useful and more natural.
            if isBiography, lower == "history" { continue }
            let headingTerms = Set(lower.components(
                separatedBy: CharacterSet.alphanumerics.inverted
            ).filter { $0.count >= 3 }.map(stem))
            // Do not immediately suggest the facet the user just asked. A
            // generic opener ("Tell me about X") has subject-name keywords
            // but no heading overlap, so its best drill-ins remain eligible.
            if !asked.isEmpty, !asked.isDisjoint(with: headingTerms) { continue }

            let priority = contextualPriority(lower, isBiography: isBiography)
            // Low-signal table-of-contents residue made the chips feel like
            // random Wikipedia navigation ("What about polls and rankings?").
            // Only offer facets for which we have deliberate conversational
            // wording. Two strong chips are better than three padded ones.
            guard priority > 20 else { continue }
            let phrased = contextualQuestion(
                topic: topic, heading: raw, isBiography: isBiography,
                isNamedEvent: isNamedEvent)
            var promptTerms = termSet(ArticleHeuristics.questionKeywords(phrased))
            promptTerms.subtract(topicTerms)
            if !asked.isEmpty, !asked.isDisjoint(with: promptTerms) { continue }
            let key = phrased.lowercased()
            guard seen.insert(key).inserted else { continue }
            candidates.append((
                score: priority,
                order: order,
                thread: DiscoveryThread(
                    label: phrased,
                    kind: .topic,
                    source: .section,
                    note: raw,
                    prompt: phrased)))
        }

        return candidates.sorted {
            if $0.score == $1.score { return $0.order < $1.order }
            return $0.score > $1.score
        }.prefix(max).map(\.thread)
    }

    private static func contextualQuestion(
        topic: String,
        heading: String,
        isBiography: Bool,
        isNamedEvent: Bool
    ) -> String {
        let h = heading.lowercased()
        let possessive = topic.hasSuffix("s") ? "\(topic)'" : "\(topic)'s"
        if h.contains("early life") || h.contains("childhood") || h == "youth" {
            return "What was \(possessive) early life like?"
        }
        if h.contains("education") || h.contains("school") {
            return isBiography
                ? "Where did \(topic) go to school?"
                : "What is education like in \(topic)?"
        }
        if h.contains("family") || h.contains("parents") {
            return "What about \(possessive) family?"
        }
        if h.contains("personal life") || h.contains("marriage") {
            return "What was \(possessive) personal life like?"
        }
        if h.contains("career") || h.contains("rise to power") {
            return "How did \(possessive) career develop?"
        }
        if h.contains("foreign policy") || h.contains("nato")
            || (isBiography && h.contains("the west"))
        {
            return "How has \(topic) dealt with the West and NATO?"
        }
        if h.contains("public image") || h.contains("approval")
            || h.contains("popularity") || h.contains("poll")
            || h.contains("ranking")
        {
            return "How is \(topic) viewed by the public?"
        }
        if h.contains("legacy") || h.contains("influence") {
            return "What is \(possessive) legacy?"
        }
        if h.contains("death") {
            return "How did \(topic) die?"
        }
        if h.contains("background") || h.contains("prelude") {
            return "What led up to it?"
        }
        if isNamedEvent && (h.contains("combatant") || h.contains("belligerent")
            || h.contains("forces") || h.contains("army") || h.contains("armies")
            || h.contains("troops") || h.contains("defender")
            || h.contains("reinforcement"))
        {
            return "Who were the combatants?"
        }
        if h.contains("casualt") || h.contains("losses") {
            return "How many people died?"
        }
        if h.contains("aftermath") || h.contains("consequences") {
            return "What happened afterward?"
        }
        if h.contains("outcome") || h.contains("result") {
            return "What was the outcome?"
        }
        if h.contains("cause") || h.contains("origin") {
            return "What caused it?"
        }
        if h.contains("detect") || h.contains("observ") || h.contains("experiment") {
            return "How was it first detected?"
        }
        if h.contains("source") || h.contains("formation") || h.contains("production") {
            return "How is it created?"
        }
        if h.contains("binar") || h.contains("compact object") {
            return "What kinds of systems create them?"
        }
        if h.contains("effect") || h.contains("impact") {
            return "What effects does it have?"
        }
        if h.contains("application") || h.contains("uses") || h == "use" {
            return "What is it used for?"
        }
        if isBiography && (h.contains("historical") || h.contains("history")) {
            return "How has \(possessive) legacy been assessed?"
        }
        if h.contains("history") || h.contains("discovery") {
            return "How did its history unfold?"
        }
        return "What about \(h)?"
    }

    /// Prefer high-value conversational facets over article-order trivia.
    /// This makes biographies lead with early life/career/family, battles
    /// with combatants/casualties/aftermath, and science with sources,
    /// effects, and detection.
    private static func contextualPriority(
        _ heading: String,
        isBiography: Bool
    ) -> Int {
        let tiers: [([String], Int)] = [
            (["early life", "education", "family", "career", "personal life"], 100),
            (["combatant", "belligerent", "army", "armies", "forces", "troops", "defender", "reinforcement", "casualt", "aftermath", "outcome", "cause"], 100),
            (["source", "formation", "production", "binar", "compact object", "detect", "observ", "effect", "application"], 100),
            (["foreign policy", "nato"], 90),
            (["background", "history", "historical", "legacy", "influence", "discovery"], 80),
            (["public image", "approval", "popularity", "poll", "ranking"], 70),
            (["politic", "foreign policy", "culture", "economy", "geography"], 60),
        ]
        if isBiography, heading.contains("the west") { return 90 }
        for (needles, score) in tiers where needles.contains(where: { heading.contains($0) }) {
            return score
        }
        return 20
    }

    private static func stem(_ word: String) -> String {
        for suffix in ["ies", "es", "s", "ed", "ing"] where word.hasSuffix(suffix) {
            let value = String(word.dropLast(suffix.count))
            if value.count >= 4 { return value }
        }
        return word
    }

    private static func compareThreads(_ result: [String: Any]) -> [DiscoveryThread] {
        var out: [DiscoveryThread] = []
        if let articles = result["articles"] as? [[String: Any]] {
            for a in articles {
                if let t = a["title"] as? String, !t.isEmpty {
                    out.append(DiscoveryThread(
                        label: t, kind: .topic, source: .relation,
                        zimPath: a["path"] as? String))
                }
            }
        }
        return out
    }

    // MARK: - Ranking

    /// Dedupe against entities already in focus (don't offer what we just
    /// talked about), drop self-references, and cap the list so the offer
    /// stays conversational (1–3 items, default ceiling 4).
    public static func rank(
        _ threads: [DiscoveryThread],
        focus: ConversationFocus,
        max: Int = 4
    ) -> [DiscoveryThread] {
        let discussed = Set(focus.entities.map(\.matchKey))
        var seen = Set<String>()
        var out: [DiscoveryThread] = []
        // Stable priority: nearby places + wikilinks (lateral moves) before
        // sections (deeper) — moving the conversation beats drilling in.
        let order: [DiscoveryThread.Source] = [
            .nearbyPlace, .wikilink, .relation, .section,
        ]
        for src in order {
            for t in threads where t.source == src {
                let key = t.matchKey
                if !isUserFacing(t) || seen.contains(key) || discussed.contains(key) {
                    continue
                }
                seen.insert(key)
                out.append(t)
                if out.count >= max { return out }
            }
        }
        return out
    }

    /// Re-order an already-ranked thread list by semantic similarity to the
    /// conversation, given a precomputed `key -> score` map (cosine to the
    /// focus centroid, produced by `EmbeddingIndex`). Threads keyed by their
    /// `zimPath` when present, else their label. Unscored threads keep their
    /// original relative order, after the scored ones. Kept sync + pure so the
    /// async embedding work stays in the host; this is just the stable sort.
    public static func orderBySimilarity(
        _ threads: [DiscoveryThread], scores: [String: Float]
    ) -> [DiscoveryThread] {
        func key(_ t: DiscoveryThread) -> String {
            if let p = t.zimPath, !p.isEmpty { return p }
            return t.matchKey
        }
        let floor = -Float.greatestFiniteMagnitude
        return threads.enumerated().sorted { a, b in
            let sa = scores[key(a.element)] ?? floor
            let sb = scores[key(b.element)] ?? floor
            if sa == sb { return a.offset < b.offset }
            return sa > sb
        }.map(\.element)
    }

    // MARK: - Deterministic offer caption (no-LLM fast path)

    /// A short, natural "where next" line for the fast path that runs no
    /// model. The LLM path instead gets these threads in the preamble and
    /// writes its own offer.
    public static func offer(_ threads: [DiscoveryThread]) -> String? {
        let picked = Array(threads.filter(isUserFacing).prefix(3))
        guard !picked.isEmpty else { return nil }
        let parts = picked.map { t -> String in
            if let n = t.note, !n.isEmpty { return "\(t.label) (\(n))" }
            return t.label
        }
        let list: String
        switch parts.count {
        case 1: list = parts[0]
        case 2: list = "\(parts[0]) or \(parts[1])"
        default:
            list = parts.dropLast().joined(separator: ", ")
                + ", or \(parts.last!)"
        }
        return "Want to hear about \(list)?"
    }

    // MARK: - Helpers

    /// Final defense before a discovery thread becomes visible. Article
    /// adapters call Wikipedia's untitled opening passage `lead`; that is an
    /// internal retrieval label, not a subject or section a person can ask
    /// about. Keep the check source-specific so a real wikilink to the
    /// article "Lead" (the chemical element) remains a valid suggestion.
    public static func isUserFacing(_ thread: DiscoveryThread) -> Bool {
        let label = thread.label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !label.isEmpty else { return false }
        guard thread.source == .section else { return true }

        if let note = thread.note, shouldSkipSection(note) { return false }
        let folded = label.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if internalSectionThreadLabels.contains(folded) { return false }
        if let prompt = thread.prompt {
            let foldedPrompt = prompt.lowercased()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if internalSectionThreadLabels.contains(foldedPrompt) { return false }
        }
        return true
    }

    private static let skipSections: Set<String> = [
        "see also", "references", "notes", "footnotes", "citations",
        "further reading", "external links", "bibliography", "sources",
    ]

    private static let internalSectionTitles: Set<String> = ["lead"]
    private static let internalSectionThreadLabels: Set<String> = [
        "lead", "what about lead", "what about lead?",
    ]

    private static func shouldSkipSection(_ raw: String) -> Bool {
        let title = raw.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return title.isEmpty
            || internalSectionTitles.contains(title)
            || skipSections.contains(title)
    }

    private static func firstString(
        _ dict: [String: Any], _ keys: String...
    ) -> String? {
        for k in keys {
            if let s = dict[k] as? String, !s.isEmpty { return s }
        }
        return nil
    }

    private static func doubleField(_ dict: [String: Any], _ key: String) -> Double? {
        if let d = dict[key] as? Double { return d }
        if let i = dict[key] as? Int { return Double(i) }
        if let n = dict[key] as? NSNumber { return n.doubleValue }
        return nil
    }

    private static func distanceNote(_ meters: Double) -> String {
        if meters < 1000 { return "\(Int(meters.rounded())) m away" }
        return String(format: "%.1f km away", meters / 1000)
    }
}

/// Minimal outbound-link extractor for Kiwix-generated Wikipedia HTML.
///
/// Kiwix article links are relative anchors to other ZIM entries — typically
/// `href="Article_Name"` or `href="./Article_Name"` / `"../A/Article_Name"`.
/// We keep only those: external (`http`, `//`), intra-page (`#`), and
/// namespaced links (`File:`, `Category:`, `Template:`, `Help:`, `Special:`,
/// `Wikipedia:`, `Portal:`, `wikt:` …) are dropped, since none of those is a
/// subject the user would want to "go to next". Links are returned in first-
/// appearance order (so lead-paragraph links — the most salient — come first),
/// deduped by title, capped at `max`.
public enum WikiLinks {

    public struct Link: Equatable, Sendable {
        public let title: String
        public let path: String
        public init(title: String, path: String) {
            self.title = title
            self.path = path
        }
    }

    private static let droppedNamespaces: [String] = [
        "file:", "image:", "category:", "template:", "help:",
        "special:", "wikipedia:", "portal:", "wikt:", "w:",
        "media:", "mediawiki:", "module:", "talk:",
    ]

    /// Boilerplate link targets that ARE real ZIM articles but make terrible
    /// "want to hear about…?" offers — drug-reference sites and infobox
    /// identifier fields. Belt-and-suspenders on top of the prose-only
    /// extraction below (real capture 2026-05-30: a medicine offered
    /// "MedlinePlus, Drugs.com, Trade names").
    private static let boilerplateTitles: Set<String> = [
        "medlineplus", "drugs.com", "trade names", "trade name",
        "pregnancy category", "route of administration",
        "routes of administration", "defined daily dose", "atc code",
        "cas number", "pubchem", "drugbank", "chemspider", "kegg", "chembl",
        "british approved name", "united states adopted name", "license data",
        "international nonproprietary name", "iso 4217", "doi", "isbn", "issn",
    ]

    public static func parse(html: String, max: Int = 8) -> [Link] {
        // Prefer prose for ordinary related-topic suggestions so infobox and
        // navigation links cannot dominate the chips.
        let prose = proseParagraphs(html)
        return parseLinks(source: prose.isEmpty ? html : prose, max: max)
    }

    /// Parse the complete article body. Disambiguation pages put their
    /// canonical choices in lists rather than prose paragraphs, so callers
    /// resolving an already-confirmed disambiguation page need this variant.
    public static func parseAll(html: String, max: Int = 8) -> [Link] {
        parseLinks(source: html, max: max)
    }

    private static func parseLinks(source: String, max: Int) -> [Link] {
        let pattern = #"<a\b[^>]*?href="([^"]*)"[^>]*>(.*?)</a>"#
        guard let re = RegexCache.shared.compiled(
            pattern,
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) else { return [] }

        let range = NSRange(source.startIndex..., in: source)
        var out: [Link] = []
        var seen = Set<String>()

        for m in re.matches(in: source, range: range) {
            guard m.numberOfRanges >= 3,
                  let hrefR = Range(m.range(at: 1), in: source),
                  let textR = Range(m.range(at: 2), in: source)
            else { continue }

            let href = String(source[hrefR])
            let rawText = String(source[textR])
            let title = decodeAndStrip(rawText)

            guard isArticleLink(href), !title.isEmpty, title.count >= 2 else {
                continue
            }
            // Skip pure-number / citation anchor text ("[1]", "12").
            if title.allSatisfy({ $0.isNumber || $0 == "[" || $0 == "]" }) {
                continue
            }
            if boilerplateTitles.contains(title.lowercased()) { continue }
            // Dedupe by DESTINATION (two anchors with different text — e.g. a
            // second "again" link — pointing at the same article are one
            // thread), falling back to the title when there's no path.
            let path = normalizePath(href)
            let key = (path.isEmpty ? title : path).lowercased()
            if seen.contains(key) { continue }
            seen.insert(key)
            out.append(Link(title: title, path: path))
            if out.count >= max { break }
        }
        return out
    }

    /// Concatenate the article's prose `<p>…</p>` blocks. These hold the
    /// body text whose links are genuine lateral topics; the infobox
    /// (`<table>`), navboxes, and references (`<ol>`) — where the
    /// boilerplate links live — are not `<p>` and so never reach the
    /// extractor.
    private static func proseParagraphs(_ html: String) -> String {
        guard let re = RegexCache.shared.compiled(
            #"<p\b[^>]*>(.*?)</p>"#,
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) else { return "" }
        let range = NSRange(html.startIndex..., in: html)
        var parts: [Substring] = []
        for m in re.matches(in: html, range: range) {
            if let r = Range(m.range(at: 1), in: html) {
                parts.append(html[r])
            }
        }
        return parts.joined(separator: " ")
    }

    private static func isArticleLink(_ href: String) -> Bool {
        let h = href.lowercased()
        if h.isEmpty || h.hasPrefix("#") { return false }
        if h.hasPrefix("http://") || h.hasPrefix("https://")
            || h.hasPrefix("//") || h.hasPrefix("mailto:")
            || h.hasPrefix("tel:") || h.hasPrefix("data:") {
            return false
        }
        // Strip leading relative markers to inspect the last path component
        // for a namespace prefix.
        let last = href.split(separator: "/").last.map(String.init) ?? href
        let lastLower = last.lowercased()
        for ns in droppedNamespaces where lastLower.hasPrefix(ns) {
            return false
        }
        return true
    }

    /// Strip Kiwix relative prefixes so the path is a clean ZIM-relative
    /// entry the host's `articleByTitle`/`get_article_section` can resolve.
    private static func normalizePath(_ href: String) -> String {
        var p = href
        while p.hasPrefix("./") { p.removeFirst(2) }
        while p.hasPrefix("../") { p.removeFirst(3) }
        return p
    }

    private static func decodeAndStrip(_ s: String) -> String {
        // Drop any nested tags (links sometimes wrap <i>/<b>/<span>).
        // Compiled once — this runs per link inside the parseLinks loop.
        var t = s
        if let re = RegexCache.shared.compiled("<[^>]+>") {
            t = re.stringByReplacingMatches(
                in: t, range: NSRange(t.startIndex..., in: t), withTemplate: "")
        }
        let entities = [
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&quot;": "\"", "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
        ]
        for (k, v) in entities { t = t.replacingOccurrences(of: k, with: v) }
        if let numeric = RegexCache.shared.compiled(#"&#(?:x[0-9a-fA-F]+|[0-9]+);"#) {
            let mutable = NSMutableString(string: t)
            let matches = numeric.matches(
                in: t, range: NSRange(t.startIndex..., in: t))
            for match in matches.reversed() {
                let entity = mutable.substring(with: match.range)
                let body = entity.dropFirst(entity.hasPrefix("&#x") ? 3 : 2).dropLast()
                let radix = entity.hasPrefix("&#x") ? 16 : 10
                guard let value = UInt32(body, radix: radix),
                      let scalar = Unicode.Scalar(value)
                else { continue }
                mutable.replaceCharacters(in: match.range, with: String(scalar))
            }
            t = mutable as String
        }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
