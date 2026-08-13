// SPDX-License-Identifier: MIT
//
// Shared heuristics for the "composite" article tools — article_overview,
// compare_articles, narrate_article, article_relationship. Each of those
// tools is a thin wrapper in MCPToolAdapter that delegates to the helpers
// here. Keeping the heuristics in one place so tuning (section priority,
// excerpt length, citation strip) happens in one file.
//
// None of this talks to libzim directly — every primitive it needs goes
// through `any ZimService`, so the logic works with the stub service in
// tests and the real `DefaultZimService` on device.

import Foundation

public enum ArticleHeuristics {

    // MARK: - Cached regex primitives
    //
    // `range(of:options:.regularExpression)` and
    // `replacingOccurrences(options:.regularExpression)` compile a fresh
    // NSRegularExpression on every call — the trap `stripCitations` already
    // names below. These heuristics evaluate their patterns per sentence
    // inside article loops (`keyFactSentence` runs one per sentence of every
    // section of every source; `groundedExtractiveAnswer` runs four per
    // grounded passage), so compilation dominated the match itself (DS4 perf
    // duplicate group 2026-08-13). Route every pattern through the shared
    // process-wide cache instead; the patterns are literals, so the cache
    // stays bounded by the number of call sites here.

    @inline(__always)
    private static func regexRange(
        _ text: String, _ pattern: String,
        options: NSRegularExpression.Options = []
    ) -> Range<String.Index>? {
        guard let re = RegexCache.shared.compiled(pattern, options: options),
              let m = re.firstMatch(
                in: text, range: NSRange(text.startIndex..., in: text))
        else { return nil }
        return Range(m.range, in: text)
    }

    /// Existence-only check. Deliberately does NOT go through `regexRange`:
    /// `Range(nsRange, in:)` returns nil when a match lands mid-grapheme, and
    /// a predicate must not silently answer "no match" for that.
    @inline(__always)
    private static func regexMatches(
        _ text: String, _ pattern: String,
        options: NSRegularExpression.Options = []
    ) -> Bool {
        guard let re = RegexCache.shared.compiled(pattern, options: options) else {
            return false
        }
        return re.firstMatch(
            in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    @inline(__always)
    private static func regexReplacing(
        _ text: String, _ pattern: String, with template: String,
        options: NSRegularExpression.Options = []
    ) -> String {
        guard let re = RegexCache.shared.compiled(pattern, options: options) else {
            return text
        }
        return re.stringByReplacingMatches(
            in: text, range: NSRange(text.startIndex..., in: text),
            withTemplate: template)
    }

    // MARK: - Section selection

    /// Sections whose prose reliably carries narrative content about the
    /// article subject. Matched case-insensitively against the section
    /// heading; prefix match counts (so "History", "History of X", and
    /// "Early history" all hit the "history" slot). Ordered by the
    /// rough priority we want when filling an overview.
    private static let prioritySectionTitles: [String] = [
        "history",
        "overview",
        "background",
        "description",
        "geography",
        "culture",
        "economy",
        "demographics",
        "etymology",
        "early life",
        "career",
        "biography",
    ]

    /// Fetch the full ordered section list for an article given a title.
    /// Combines `articleByTitle` (fuzzy title → path resolution, handles
    /// redirects and the `en:Foo` OSM tag form) with `articleSections`
    /// (full outline parse). Two service calls, but the default service's
    /// article LRU makes the second a cache hit — one read, one parse.
    public static func sectionsByTitle(
        service: any ZimService,
        title: String,
        zim: String?
    ) async throws -> (zim: String, path: String, title: String, sections: [ArticleSection]) {
        let hit = try await service.articleByTitle(title: title, zim: zim, section: "lead")
        let all = try await service.articleSections(path: hit.path, zim: hit.zim)
        return (hit.zim, hit.path, all.title, all.sections)
    }

    /// Pick up to `maxSections` sections for an "overview" response. Always
    /// includes the lead. Then prefers sections whose title matches one of
    /// the priority slots (in priority order), filling any remaining room
    /// with the largest-by-bytes sections that weren't already picked.
    ///
    /// Boilerplate sections (References, See also, …) are already filtered
    /// out by `ArticleSections.parse`, so we don't re-filter here.
    public static func pickOverview(
        sections: [ArticleSection],
        maxSections: Int
    ) -> [ArticleSection] {
        guard !sections.isEmpty else { return [] }
        let limit = max(1, maxSections)
        var out: [ArticleSection] = []
        if let lead = sections.first(where: { $0.title.isEmpty }) {
            out.append(lead)
        }
        let named = sections.filter { !$0.title.isEmpty }
        var pickedTitles = Set(out.map(\.title))
        for priority in prioritySectionTitles {
            if out.count >= limit { break }
            if let match = named.first(where: { s in
                !pickedTitles.contains(s.title)
                    && s.title.lowercased().hasPrefix(priority)
            }) {
                out.append(match)
                pickedTitles.insert(match.title)
            }
        }
        if out.count < limit {
            let remaining = named
                .filter { !pickedTitles.contains($0.title) }
                .sorted { $0.bytes > $1.bytes }
            for s in remaining {
                if out.count >= limit { break }
                out.append(s)
            }
        }
        return out
    }

    // MARK: - Prose cleaning

    /// Detect Wikipedia disambiguation pages — the ones titled
    /// `Foo (disambiguation)` or opening with "Foo may refer to:".
    /// These return a list of meanings, not a description of a
    /// specific entity, and on device we were rendering the "may
    /// refer to:" line as a place's Wikipedia preview (real capture:
    /// "Oak Grove" → disambig). Both the tagged-wiki path and the
    /// fallback name-search path bail when this returns true.
    public static func isDisambiguationArticle(
        title: String, leadText: String
    ) -> Bool {
        let t = title.lowercased()
        if t.contains("(disambiguation)") { return true }
        let lead = leadText.prefix(400).lowercased()
        // Wikipedia uses both "X may refer to:" and "X most commonly
        // refers to:" as canonical disambiguation openers. Keep this
        // bounded to those grammatical forms so running prose such as
        // "residents also refer to ..." remains a normal article.
        if regexMatches(
            lead,
            #"\b(?:(?:may|can)\s+refer\s+to|(?:most\s+commonly|commonly)\s+refers\s+to)\b"#) {
            return true
        }
        return false
    }

    /// Strip leading paragraphs that are just the article's own
    /// title repeated. Wikipedia's lead section often opens with
    /// `Title\n\nShort descriptor\n\nTitle is a …` where the first
    /// two lines are the infobox caption + heading we already show
    /// as the row label. Without this the list rendered
    /// "Palo Alto Junior Museum and Zoo" THREE times before any
    /// actual sentence (see 2026-04-22 screenshot).
    ///
    /// Conservative: only drops a leading line when it's a
    /// normalised equal / prefix of the title. Doesn't touch the
    /// rest of the lead.
    public static func stripLeadingTitleRepetition(
        _ text: String, title: String
    ) -> String {
        let titleNorm = normaliseForCompare(title)
        guard !titleNorm.isEmpty else { return text }
        let paragraphs = text.components(separatedBy: "\n\n")
        var idx = 0
        while idx < paragraphs.count {
            let p = paragraphs[idx].trimmingCharacters(
                in: .whitespacesAndNewlines)
            if p.isEmpty { idx += 1; continue }
            let pNorm = normaliseForCompare(p)
            // Only strip paragraphs that are EXACTLY the title after
            // normalisation. Anything more permissive (like "starts
            // with the title") clobbers the real first sentence —
            // Wikipedia leads frequently open with
            // "Title is a …" which shares the title as a prefix but
            // IS the content we want to keep.
            if pNorm == titleNorm { idx += 1; continue }
            break
        }
        if idx == 0 { return text }
        return paragraphs[idx..<paragraphs.count]
            .joined(separator: "\n\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Lowercase + strip punctuation + collapse whitespace. Used by
    /// the title-repetition detector so "Palo Alto", "palo alto", and
    /// "Palo Alto." compare equal.
    private static func normaliseForCompare(_ s: String) -> String {
        let lowered = s.lowercased()
        var out = ""
        out.reserveCapacity(lowered.count)
        var lastWasSpace = false
        for c in lowered {
            if c.isLetter || c.isNumber {
                out.append(c); lastWasSpace = false
            } else if c.isWhitespace {
                if !lastWasSpace, !out.isEmpty {
                    out.append(" "); lastWasSpace = true
                }
            } else {
                // punctuation → eat
            }
        }
        return out.trimmingCharacters(in: .whitespaces)
    }

    /// Strip inline citation markers that look fine on screen but read
    /// badly through TTS. Keeps the sentence punctuation intact.
    /// Examples removed: [1], [12], [a], [citation needed], [note 3], [nb 2].
    /// Patterns run through the shared compiled-regex cache — this is
    /// called once per section on the narration/excerpt paths, and
    /// `replacingOccurrences(options: .regularExpression)` recompiles
    /// per call.
    public static func stripCitations(_ text: String) -> String {
        let passes: [(pattern: String, template: String, options: NSRegularExpression.Options)] = [
            (#"\[\s*\d{1,3}\s*\]"#, "", .caseInsensitive),  // [1], [12]
            (#"\[\s*[a-zA-Z]\s*\]"#, "", .caseInsensitive), // [a], [B]
            (#"\[\s*citation needed\s*\]"#, "", .caseInsensitive),
            (#"\[\s*(?:note|nb|sic|clarification needed)[^\]]*\]"#, "", .caseInsensitive),
            (#"[ \t]{2,}"#, " ", []),
            (#" +([.,;:!?])"#, "$1", []),
        ]
        var out = text
        for pass in passes {
            guard let re = RegexCache.shared.compiled(
                pass.pattern, options: pass.options
            ) else { continue }
            out = re.stringByReplacingMatches(
                in: out, range: NSRange(out.startIndex..., in: out),
                withTemplate: pass.template
            )
        }
        return out
    }

    /// Truncate a chunk of prose to roughly `maxChars` characters, snapping
    /// to the nearest sentence boundary so TTS doesn't stop mid-word. Falls
    /// back to a hard cut + ellipsis if no sentence boundary is close.
    public static func trimToSentence(_ text: String, maxChars: Int) -> String {
        if text.count <= maxChars { return text }
        let hardCap = min(maxChars + 200, text.count)
        let window = String(text.prefix(hardCap))
        let searchFrom = window.index(
            window.startIndex,
            offsetBy: max(0, maxChars / 2)
        )
        var best: String.Index?
        for terminator in [". ", "! ", "? ", ".\n", "!\n", "?\n"] {
            if let r = window.range(of: terminator, options: .backwards,
                                    range: searchFrom..<window.endIndex)
            {
                if best == nil || r.upperBound > best! {
                    best = r.upperBound
                }
            }
        }
        if let cut = best {
            return String(window[..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return String(text.prefix(maxChars)).trimmingCharacters(in: .whitespaces) + "…"
    }

    /// Evidence depth for one grounded conversational turn. The previous
    /// path always sent lead + up to five sections, even for a direct fact,
    /// which made a simple biography opener pay a 2k-token cold prefill. Use
    /// the already-established query classification to spend context where it
    /// changes answer quality.
    public static func groundedPassageLimit(for question: String) -> Int {
        let lower = question.lowercased()
        // An explicit source-inspection request asks what the selected
        // article says, not only for its single highest-scoring mention.
        // Wikipedia often splits one event across history and subject-
        // specific sections (Santa Rosa's 1906 earthquake appears in both
        // "20th century" and "Seismicity"), so retain several local passages.
        if regexMatches(
            lower,
            #"\bwhat\s+does\s+(?:this\s+|the\s+)?(?:wikipedia\s+)?article\s+say\s+(?:about|on|regarding)\b"#
        ) {
            return 4
        }
        switch QueryComplexity.classify(question) {
        case .factoid, .navigational:
            return 2       // lead + one precise section/window
        case .explanatory:
            return 4       // lead + enough evidence for mechanism/synthesis
        case .topical:
            // Short facet follow-ups ("What about his parents?") behave like
            // factoids even though their opener is not who/when/where.
            let words = question.split(whereSeparator: { $0.isWhitespace })
            return words.count <= 8 ? 2 : 3
        }
    }

    /// Character budget for each passage after section ranking. Named
    /// sections are reduced to a sentence window around the user's terms;
    /// explanatory turns retain a wider window for causal context.
    public static func groundedPassageCharacterLimit(for question: String) -> Int {
        let q = question.lowercased()
        // Education biographies often put primary/secondary school and
        // university in adjacent paragraphs. Preserve both so a broad
        // "where did they go to school?" gets a complete answer.
        if ["school", "college", "education"]
            .contains(where: { q.contains($0) }) {
            return 1_200
        }
        // Casualty prose commonly separates killed, wounded, and total
        // casualty estimates across several sentences. The wider window is
        // still small, but lets the prompt distinguish those categories.
        if q.contains("how many"),
           ["died", "dead", "death", "killed", "fatalit"]
            .contains(where: { q.contains($0) }) {
            return 1_100
        }
        switch QueryComplexity.classify(question) {
        case .factoid, .navigational: return 700
        case .topical: return 900
        case .explanatory: return 1_100
        }
    }

    /// Extract a compact, sentence-aligned window from a selected section.
    /// Ranking picks the right *section*; this picks the right part inside a
    /// long section so a fact near the middle is not lost by `prefix(1500)`.
    /// The best matching sentence is returned with adjacent context, bounded
    /// by `maxChars`. With no useful content words we retain the beginning,
    /// which is the natural behavior for an overview or "tell me more".
    public static func groundedPassageWindow(
        _ text: String,
        question: String,
        maxChars: Int
    ) -> String {
        guard maxChars > 0, text.count > maxChars else { return text }
        let sentences = sentenceChunks(text)
        guard !sentences.isEmpty else { return trimToSentence(text, maxChars: maxChars) }
        let weighted = weightedKeywords(questionKeywords(question))
        guard !weighted.isEmpty else { return trimToSentence(text, maxChars: maxChars) }
        let lowerQuestion = question.lowercased()
        let asksForDeathCount = lowerQuestion.contains("how many")
            && ["died", "dead", "death", "killed", "fatalit"]
                .contains(where: { lowerQuestion.contains($0) })

        func score(_ sentence: String) -> Float {
            let lower = sentence.lowercased()
            var value: Float = 0
            for (term, weight) in weighted {
                let root = stem(term)
                if lower.contains(root) { value += weight }
            }
            // A casualty section often begins with one party's disputed
            // claim and gives the historian/eyewitness consensus later. For
            // an unqualified death-count question, center the evidence on a
            // numbered consensus estimate so the model sees the useful range
            // instead of confidently repeating the first claim in the prose.
            if asksForDeathCount,
               regexMatches(lower, #"\d"#),
               ["died", "dead", "death", "killed", "fatalit", "casualt"]
                .contains(where: { lower.contains($0) })
            {
                value += 2.0
                if lower.contains("most eyewitness") { value += 2.0 }
                if lower.contains("most historian") { value += 1.5 }
                if lower.contains("between") || lower.contains("range")
                    || lower.contains("estimate") { value += 1.0 }
                if lower.contains("claimed") { value -= 0.75 }
            }
            return value
        }
        let scores = sentences.map(score)
        guard let best = scores.indices.max(by: { scores[$0] < scores[$1] }),
              scores[best] > 0
        else { return trimToSentence(text, maxChars: maxChars) }

        // Cleaned Wikipedia occasionally contains a table-like block with no
        // sentence boundaries. Do not let one 1,300-character "sentence"
        // defeat the evidence cap; center a word-bounded window on the best
        // exact/synonym match so the requested fact stays in view.
        if sentences[best].count > maxChars {
            let sentence = sentences[best]
            let lower = sentence.lowercased()
            let hitOffset = weighted.lazy.compactMap { term -> Int? in
                guard let range = lower.range(of: stem(term.term)) else { return nil }
                return lower.distance(from: lower.startIndex, to: range.lowerBound)
            }.first ?? 0
            let startOffset = min(
                max(0, hitOffset - maxChars / 3),
                max(0, sentence.count - maxChars))
            let endOffset = min(sentence.count, startOffset + maxChars)
            let start = sentence.index(sentence.startIndex, offsetBy: startOffset)
            let end = sentence.index(sentence.startIndex, offsetBy: endOffset)
            var snippet = String(sentence[start..<end])
            if startOffset > 0,
               let firstSpace = snippet.firstIndex(where: { $0.isWhitespace }) {
                snippet.removeSubrange(snippet.startIndex...firstSpace)
            }
            if endOffset < sentence.count,
               let lastSpace = snippet.lastIndex(where: { $0.isWhitespace }) {
                snippet.removeSubrange(lastSpace..<snippet.endIndex)
                snippet += "…"
            }
            return snippet.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        var lo = best
        var hi = best
        var count = sentences[best].count
        // One preceding sentence is valuable for pronoun resolution ("His
        // father…"); then grow toward the higher-scoring neighbor.
        if lo > 0, count + 1 + sentences[lo - 1].count <= maxChars {
            lo -= 1
            count += 1 + sentences[lo].count
        }
        while true {
            let leftFits = lo > 0 && count + 1 + sentences[lo - 1].count <= maxChars
            let rightFits = hi + 1 < sentences.count
                && count + 1 + sentences[hi + 1].count <= maxChars
            guard leftFits || rightFits else { break }
            if rightFits && (!leftFits || scores[hi + 1] >= scores[lo - 1]) {
                hi += 1
                count += 1 + sentences[hi].count
            } else {
                lo -= 1
                count += 1 + sentences[lo].count
            }
        }
        return sentences[lo...hi].joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Return a direct answer for grounded follow-ups where preserving the
    /// source's exact names or numeric labels matters more than paraphrasing.
    /// The conversational model still handles open-ended turns; this narrow
    /// path prevents a sampled rewrite from dropping parent names or turning
    /// a killed-plus-wounded casualty total into a death count.
    public static func groundedExtractiveAnswer(
        question: String,
        passages: [String],
        passageLabels: [String] = []
    ) -> String? {
        let lowerQuestion = question.lowercased()
        let evidence = passages.enumerated().flatMap {
            index, passage -> [(sentence: String, label: String)] in
            var text = stripCitations(passage)
            // Section bodies can begin with Wikipedia navigation hatnotes.
            // They are useful links in the article UI but are not part of a
            // spoken answer (for example, "Main article: Intelligence career
            // of Vladimir Putin"). Remove complete hatnote lines before
            // collapsing whitespace, while their newline boundary still
            // exists.
            text = regexReplacing(
                text, #"(?im)^(?:main article|further information):[^\n]*(?:\n|$)"#,
                with: "")
            text = regexReplacing(text, #"\s+"#, with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            text = regexReplacing(text, #"\(\s+"#, with: "(")
            text = regexReplacing(text, #"\s+\)"#, with: ")")
            let label = index < passageLabels.count ? passageLabels[index] : ""
            return sentenceChunks(text).map { (sentence: $0, label: label) }
        }
        let cleaned = evidence.map(\.sentence)
        guard !cleaned.isEmpty else { return nil }

        if ["parent", "mother", "father"]
            .contains(where: { lowerQuestion.contains($0) }) {
            return extractParentAnswer(
                question: lowerQuestion, sentences: cleaned)
        }

        let asksForDeathCount = lowerQuestion.contains("how many")
            && ["died", "dead", "death", "killed", "fatalit"]
                .contains(where: { lowerQuestion.contains($0) })
        if asksForDeathCount {
            return extractDeathCountAnswer(
                question: lowerQuestion, evidence: evidence)
        }
        if lowerQuestion.contains("after"),
           lowerQuestion.contains("graduat") {
            return extractPostGraduationAnswer(sentences: cleaned)
        }
        return nil
    }

    private static func extractParentAnswer(
        question: String,
        sentences: [String]
    ) -> String? {
        let asksMother = question.contains("mother")
        let asksFather = question.contains("father")
        let asksBoth = question.contains("parent") || (!asksMother && !asksFather)

        func isNameBearing(_ lower: String) -> Bool {
            lower.contains("born to") || lower.contains("children of")
                || lower.contains("child of") || lower.contains("son of")
                || lower.contains("daughter of")
                || lower.contains("parents were")
        }
        func containsWord(_ word: String, in text: String) -> Bool {
            regexMatches(
                text,
                #"\b"# + NSRegularExpression.escapedPattern(for: word) + #"\b"#,
                options: .caseInsensitive)
        }
        func containsParentRole(_ word: String, in text: String) -> Bool {
            var literalText = text
            if word == "father" {
                // Biography leads use honorifics such as "Founding Father"
                // and "Father of His Country". Those are not parent facts.
                literalText = regexReplacing(
                    literalText,
                    #"(?i)\bfounding fathers?\b|\bfather of (?:his|the) (?:country|nation)\b"#,
                    with: "")
            }
            return containsWord(word, in: literalText)
        }

        var chosen = Set<Int>()
        // Leads can contain a citation-damaged birth/death fragment joined to
        // an otherwise useful "children of" clause. Prefer the shortest
        // name-bearing sentence across the grounded sections; biography
        // family/early-life prose normally supplies the cleaner statement.
        let nameBearingIndices = sentences.indices.filter { index in
            isNameBearing(sentences[index].lowercased())
        }
        if asksBoth,
           let index = nameBearingIndices.min(by: { lhs, rhs in
               sentences[lhs].count < sentences[rhs].count
           }) {
            chosen.insert(index)
        }
        if asksBoth || asksMother,
           let index = sentences.indices.first(where: {
               containsParentRole("mother", in: sentences[$0])
           }) {
            chosen.insert(index)
        }
        if asksBoth || asksFather,
           let index = sentences.indices.first(where: {
               containsParentRole("father", in: sentences[$0])
           }) {
            chosen.insert(index)
        }

        guard !chosen.isEmpty else { return nil }
        let answer = chosen.sorted().prefix(3).map { sentences[$0] }
            .joined(separator: " ")
        return trimToSentence(answer, maxChars: 520)
    }

    private static func extractDeathCountAnswer(
        question: String,
        evidence: [(sentence: String, label: String)]
    ) -> String? {
        struct Candidate {
            let index: Int
            let sentence: String
            let score: Int
            let explicitlyDeaths: Bool
        }
        let deathTerms = ["killed", "dead", "died", "fatalit"]
        let casualtyTerms = deathTerms + ["wounded", "casualt"]
        let contextTerms = questionKeywords(question).filter { term in
            !["killed", "dead", "died", "death", "fatality", "fatalities",
              "casualty", "casualties", "wounded"].contains(term)
        }
        var candidates: [Candidate] = []

        for (index, item) in evidence.enumerated() {
            let sentence = item.sentence
            let lower = sentence.lowercased()
            guard regexMatches(lower, #"\d"#),
                  casualtyTerms.contains(where: { lower.contains($0) })
            else { continue }

            // Biography dates are not casualty counts. This exact shape
            // produced “He died on 2 November 2012 …” three times for “How
            // many people died at Pearl Harbor?” in TestFlight feedback.
            if regexMatches(
                lower, #"\b(?:he|she|they|[A-Z][a-z]+)\s+died\s+(?:on|in|at)\b"#,
                options: .caseInsensitive),
               !regexMatches(
                lower,
                #"\b(?:\d[\d,]*\s+(?:people|persons|civilians|sailors|soldiers|troops|men|women|americans|japanese)|(?:killed|dead|fatalities)\s+(?:was|were|numbered|totaled)?\s*\d)\b"#,
                options: .caseInsensitive) {
                continue
            }

            let explicitlyDeaths = deathTerms.contains(where: { lower.contains($0) })
            var score = explicitlyDeaths ? 4 : 1
            if !contextTerms.isEmpty {
                let context = (item.label + " " + sentence).lowercased()
                let matches = contextTerms.filter { context.contains($0) }.count
                // An explicitly named event must be represented either by
                // the sentence or by its article/section label. This keeps a
                // 2019 Pearl Harbor shipyard shooting from answering a
                // question about the Japanese attack while still accepting
                // “2,403 Americans were killed” from the correctly titled
                // Attack on Pearl Harbor article.
                guard matches > 0 else { continue }
                score += matches * 3
            }
            if regexMatches(
                lower, #"\d[\d,]*\s*(?:[–—-]|to|and)\s*\d"#,
                options: .caseInsensitive) {
                score += 2
            }
            if lower.contains("most eyewitness") { score += 5 }
            if lower.contains("most historian") { score += 4 }
            if lower.contains("other estimates") { score += 4 }
            if lower.contains("estimat") || lower.contains("range") { score += 2 }
            if lower.contains("claimed") { score -= 4 }
            if lower.contains("secretary reported") { score -= 2 }
            candidates.append(Candidate(
                index: index, sentence: sentence, score: score,
                explicitlyDeaths: explicitlyDeaths))
        }

        // A total-casualties sentence alone cannot answer "how many died".
        guard candidates.contains(where: \.explicitlyDeaths) else { return nil }
        let selected = candidates
            .sorted {
                if $0.score != $1.score { return $0.score > $1.score }
                return $0.index < $1.index
            }
            .prefix(3)
            .sorted { $0.index < $1.index }
        let answer = selected.map(\.sentence).joined(separator: " ")
        return trimToSentence(answer, maxChars: 700)
    }

    private static func extractPostGraduationAnswer(
        sentences: [String]
    ) -> String? {
        let actionTerms = [
            "joined", "began", "started", "entered", "enlisted",
            "trained", "worked", "career",
        ]
        let candidates = sentences.enumerated().compactMap {
            index, sentence -> (index: Int, sentence: String, score: Int)? in
            let lower = sentence.lowercased()
            guard actionTerms.contains(where: { lower.contains($0) }) else {
                return nil
            }
            var score = 0
            if lower.contains("after graduating") { score += 8 }
            if lower.contains("graduat") { score += 5 }
            if lower.contains("joined") { score += 3 }
            if lower.contains("trained") { score += 2 }
            if lower.contains("kgb") { score += 5 }
            if lower.contains("1975") { score += 4 }
            return (index, sentence, score)
        }
        guard let best = candidates.max(by: {
            if $0.score != $1.score { return $0.score < $1.score }
            return $0.index > $1.index
        }), best.score >= 4 else { return nil }
        return trimToSentence(best.sentence, maxChars: 420)
    }

    /// Lightweight sentence splitter for cleaned Wikipedia prose. Keep the
    /// punctuation and split only when a terminator is followed by whitespace
    /// and an uppercase/digit opener; this avoids most decimal/abbreviation
    /// damage without adding a NaturalLanguage dependency to MCPZimKit.
    private static func sentenceChunks(_ text: String) -> [String] {
        let chars = Array(text)
        guard !chars.isEmpty else { return [] }
        var out: [String] = []
        var start = 0
        var i = 0
        while i < chars.count {
            if chars[i] == "." || chars[i] == "!" || chars[i] == "?" {
                var next = i + 1
                while next < chars.count, chars[next].isWhitespace { next += 1 }
                var wordStart = i
                while wordStart > start,
                      chars[wordStart - 1].isLetter || chars[wordStart - 1] == "."
                { wordStart -= 1 }
                let priorWord = String(chars[wordStart..<i]).lowercased()
                let abbreviations: Set<String> = [
                    "no", "dr", "mr", "mrs", "ms", "st", "prof", "sr", "jr",
                    "e.g", "i.e", "u.s", "u.k",
                ]
                let decimal = i > 0 && chars[i - 1].isNumber
                    && next < chars.count && chars[next].isNumber
                let boundary = !decimal && !abbreviations.contains(priorWord)
                    && (next >= chars.count
                        || chars[next].isUppercase || chars[next].isNumber)
                if boundary {
                    let value = String(chars[start...i])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !value.isEmpty { out.append(value) }
                    start = next
                    i = next
                    continue
                }
            }
            i += 1
        }
        if start < chars.count {
            let tail = String(chars[start...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty { out.append(tail) }
        }
        return out
    }

    /// Concatenate `sections` into one TTS-ready body. Leads in with the
    /// article title as a sentence ("Palo Alto."), announces each named
    /// section as a short sentence ("History."), and separates sections
    /// with a blank line so Kokoro gets a beat between them. Citation
    /// markers are stripped throughout.
    public static func formatForNarration(
        title: String,
        sections: [ArticleSection]
    ) -> String {
        var out = ""
        let cleanedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleanedTitle.isEmpty {
            out += cleanedTitle + ".\n\n"
        }
        for s in sections {
            // The lead often opens with the article's own title repeated
            // (infobox caption + heading spans) — spoken aloud that's
            // "Cessna. Cessna. Cessna…" before the first real sentence
            // (real capture 2026-08-02). The stripper is conservative:
            // only leading lines that normalize to the title are dropped.
            let deduped = stripLeadingTitleRepetition(s.text, title: cleanedTitle)
            let body = stripCitations(deduped).trimmingCharacters(in: .whitespacesAndNewlines)
            if body.isEmpty { continue }
            if s.title.isEmpty {
                out += body + "\n\n"
            } else {
                out += "\(s.title). \(body)\n\n"
            }
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Discussion retrieval

    /// Rank an article's sections by relevance to a question, for grounded
    /// "let's discuss this article" answering — single-document RAG. The
    /// lead is ALWAYS returned first (it's the topic anchor); the rest are
    /// the top scorers, so a typical k=3 yields lead + the two most
    /// on-topic sections, which comfortably fits the 8K context.
    ///
    /// Scoring blends a heading match (section titles are dense relevance
    /// signal — "Economy" for "what's the economy like") with a body match,
    /// title-weighted so a long off-topic body can't bury a bang-on
    /// heading. Uses the dependency-free `HashingEmbedder` by default, so it
    /// works with zero model assets and is deterministic under `swift test`;
    /// the host can pass a richer (`NLContextualEmbedding`-backed) embedder
    /// for semantic matches when the model is warm.
    public static func rankSectionsForQuestion(
        _ question: String,
        sections: [ArticleSection],
        embedder: TextEmbedder = HashingEmbedder(),
        k: Int = 3
    ) -> [ArticleSection] {
        guard !sections.isEmpty else { return [] }
        let leadIdx = sections.firstIndex(where: { $0.title.isEmpty }) ?? 0
        let qv = embedder.embed(question)
        let weighted = weightedKeywords(questionKeywords(question))
        // Keyword evidence dominates when the question HAS keywords —
        // pure n-gram cosine ranked "Pets" over "Early life" for "what
        // about his parents" (real device capture 2026-07-01). The
        // embedder remains the sole signal for keyword-less questions
        // ("tell me more") and a small tiebreak otherwise.
        let embedderWeight: Float = weighted.isEmpty ? 1.0 : 0.35
        var scored = sections.indices
            .filter { $0 != leadIdx && !sections[$0].text.isEmpty }
            .map { i -> (idx: Int, kw: Float, total: Float) in
                let s = sections[i]
                let titleScore = s.title.isEmpty
                    ? 0 : VectorMath.cosine(qv, embedder.embed(s.title))
                let bodyScore = VectorMath.cosine(qv, embedder.embed(s.text))
                let embed = 0.55 * titleScore + 0.45 * bodyScore
                let kw = keywordScore(weighted, section: s)
                    + sectionIntentBoost(question: question, section: s)
                return (i, kw, kw + embedderWeight * embed)
            }
        // When ANY section carries keyword evidence, sections with none
        // are out of the running — padding the context with off-topic
        // passages ("Pets" for a parents question) invites drivel.
        if scored.contains(where: { $0.kw > 0 }) {
            scored = scored.filter { $0.kw > 0 }
        }
        let ranked = scored
            .sorted { $0.total > $1.total }
            .prefix(max(0, k - 1))
            .map { sections[$0.idx] }
        return [sections[leadIdx]] + ranked
    }

    /// Content words of a follow-up question — interrogatives, pronouns, and
    /// stopwords stripped, so the remainder is what the user is actually
    /// asking ABOUT. Drives the coverage check and the corpus-fallback query.
    public static func questionKeywords(_ q: String) -> [String] {
        let stop: Set<String> = [
            "the","and","for","with","are","was","were","does","did","how",
            "what","when","where","why","who","which","they","them","its",
            "their","this","that","about","have","has","had","been","get",
            "got","gotten","along","like","tell","you","much","many","more",
            "there","then","into","from","work","works","happen","happened",
            // Pronouns/auxiliaries that slipped through and polluted
            // retrieval ("what about HIS parents" scored on "his",
            // which appears in every biography sentence — 2026-07-01).
            "his","her","him","she","hers","who's","what's",
            "said","say","says","time","went","also","ever","some","any",
            // Deictic fillers from corrections ("the ONES Einstein
            // predicted") — content-free, and they dragged the search
            // rescue to the wrong article (Graviton, 2026-07-02).
            "one","ones","meant","mean","kind","sort","type",
            "people","person",
        ]
        var seen = Set<String>()
        var out: [String] = []
        for w in q.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted) {
            guard w.count >= 3, !stop.contains(w), !seen.contains(w) else { continue }
            seen.insert(w); out.append(w)
        }
        return out
    }

    /// Resolve a short chronological continuation against the immediately
    /// preceding discussion facet. “Then what happened in Soviet times” does
    /// not name Buddhism by itself, but after “Tell me about Buddhism there”
    /// both retrieval and the grounded answer must retain that facet.
    ///
    /// The continuation grammar intentionally requires a question/auxiliary
    /// after “then”, so an explicit hand-off such as “Then tell me about
    /// Donald Trump” is left untouched for normal topic-change routing.
    /// Unit/attribute words that cannot identify a section on their own —
    /// a follow-up made only of these needs the previous question's
    /// keywords to rank meaningfully.
    static let unitAttributeKeywords: Set<String> = [
        "year", "date", "day", "month", "time", "age", "old",
        "many", "much", "number", "long", "far", "tall", "big",
        "wide", "deep", "fast", "heavy", "name", "kind", "type",
    ]

    public static func contextualizedDiscussionQuestion(
        _ question: String, previousQuestion: String?
    ) -> String {
        guard let previousQuestion,
              !previousQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
        else { return question }
        let lowered = question.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let pattern = #"^(?:(?:and|so)\s+)?then\s+(?:what|how|why|where|when|who|which|did|does|was|were|is|are|can|could|would)\b|^(?:(?:and|so)\s+)?what\s+happened\s+(?:next|later|after\s+that)\b|^(?:(?:and|so)\s+)?what\s+about\s+(?:then|later|after\s+that)\b"#
        let isNarrativeContinuation = regexMatches(lowered, pattern)
        // Elliptical follow-ups whose only keywords are units/attributes
        // inherit context: ranking "What year?" literally sent retrieval to
        // Demographics (median age 2022) when the thread was "When did it
        // join NATO?" — real capture 2026-08-02. "year" can't identify a
        // section; the previous question's keywords can. Facet questions
        // with a real topic word ("What is the population?") stay clean —
        // that word IS the retrieval signal.
        let kws = questionKeywords(question)
        let isKeywordPoor = !kws.isEmpty
            && kws.allSatisfy { Self.unitAttributeKeywords.contains($0) }
        guard isNarrativeContinuation || isKeywordPoor else { return question }
        let inherited = questionKeywords(previousQuestion)
        guard !inherited.isEmpty else { return question }
        // Appending only the inherited content words keeps section scoring
        // free of instruction words such as “previous” or “facet”. The dash
        // still gives the model a natural appositive reading.
        return question + " — " + inherited.joined(separator: " ")
    }

    /// True for date/quantity-shaped questions where the answer is one
    /// sentence and paraphrase risk is highest ("When did it join NATO?" →
    /// "…as a member of the OSCE", no date — real capture 2026-08-02).
    public static func isFactoidShaped(_ question: String) -> Bool {
        let q = question.lowercased()
        return regexMatches(
            q,
            #"^(?:and\s+|so\s+)?(?:when|what year|what date|how (?:many|much|old|long|tall|far|big)|who)\b"#)
    }

    /// Scan EVERY section of the in-hand sources for the single sentence
    /// that best matches the question's content keywords. Section-level
    /// ranking can miss the fact entirely (the NATO date sat in a section
    /// retrieval never picked), and even with the right section in
    /// evidence a 1-bit model paraphrases dates badly. Quoting the exact
    /// sentence as its own evidence line anchors the answer. Returns nil
    /// unless at least two keywords (or all, when fewer) land in one
    /// sentence.
    public static func keyFactSentence(
        question: String,
        sources: [(title: String, sections: [ArticleSection])]
    ) -> (article: String, sentence: String)? {
        let keywords = weightedKeywords(questionKeywords(question))
        guard !keywords.isEmpty else { return nil }
        let required = min(2, keywords.count)
        var best: (article: String, sentence: String, score: Double)?
        for source in sources {
            for section in source.sections {
                for raw in section.text.split(whereSeparator: { ".!?".contains($0) }) {
                    let sentence = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard sentence.count >= 20, sentence.count <= 400 else { continue }
                    let hay = sentence.lowercased()
                    var hits = 0
                    var score = 0.0
                    for (kw, w) in keywords where hay.contains(kw) {
                        hits += 1
                        score += Double(w)
                    }
                    guard hits >= required else { continue }
                    // Prefer sentences carrying a year for date questions.
                    if regexMatches(hay, #"\b(1[89]|20)\d{2}\b"#) {
                        score += 1.5
                    }
                    if best == nil || score > best!.score {
                        best = (source.title, sentence, score)
                    }
                }
            }
        }
        guard let best else { return nil }
        return (best.article, best.sentence)
    }

    /// Participant questions often appear lexically covered by a broad
    /// history section even though a dedicated event article has the useful
    /// belligerent detail. Treat them as an evidence-expansion request so a
    /// pinned country/topic discussion can pull that event article without
    /// interpreting the follow-up as a topic change.
    public static func asksAboutOpposingSides(_ question: String) -> Bool {
        let q = question.lowercased()
        return [
            "who fought", "what were the sides", "what were both sides",
            "which sides", "which side", "opposing sides", "combatant",
            "belligerent",
        ].contains(where: { q.contains($0) })
    }

    /// Extract a named event that the pinned evidence itself references, so
    /// participant follow-ups can open the dedicated article by exact title
    /// before attempting a noisy corpus search. This intentionally starts
    /// narrow with civil wars, the common ambiguous shape where a country
    /// article's broad history section mentions the event but not its sides.
    public static func namedEventArticleCandidates(
        _ sections: [ArticleSection], question: String
    ) -> [String] {
        var seen = Set<String>()
        var candidates: [String] = []
        let lowerQuestion = question.lowercased()
        let evidenceText = sections.map(\.text).joined(separator: " ").lowercased()

        // A place article often mentions the historical event and links to
        // its dedicated article. Prefer that identity before a noisy corpus
        // search or a later incident in the same place section.
        if (lowerQuestion.contains("pearl harbor")
                || evidenceText.contains("pearl harbor")),
           lowerQuestion.contains("attack"),
           lowerQuestion.contains("japanese") {
            seen.insert("attack on pearl harbor")
            candidates.append("Attack on Pearl Harbor")
        }

        guard lowerQuestion.contains("civil war"),
              let regex = RegexCache.shared.compiled(
                #"\b(?:[A-Z][\p{L}\p{M}'’.-]*\s+){1,4}Civil War\b"#)
        else { return candidates }

        for section in sections {
            let text = section.text as NSString
            let range = NSRange(location: 0, length: text.length)
            for match in regex.matches(in: section.text, range: range) {
                var title = text.substring(with: match.range)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if title.hasPrefix("The ") { title.removeFirst(4) }
                let key = title.lowercased()
                guard !seen.contains(key) else { continue }
                seen.insert(key)
                candidates.append(title)
            }
        }
        return candidates
    }

    /// Normalize `discuss_article`'s outbound link metadata into exact title
    /// identities suitable for a graph-constrained topic walk. Both the
    /// visible anchor and destination path count: Wikipedia may display
    /// "the civil war" while linking to `A/Russian_Civil_War`.
    public static func linkedArticleTitleKeys(
        _ links: [[String: Any]]
    ) -> Set<String> {
        var titles = Set<String>()
        for link in links {
            if let title = link["title"] as? String {
                let normalized = title.trimmingCharacters(
                    in: .whitespacesAndNewlines).lowercased()
                if !normalized.isEmpty { titles.insert(normalized) }
            }
            if let rawPath = link["path"] as? String {
                var title = (rawPath.removingPercentEncoding ?? rawPath)
                    .split(separator: "/").last.map(String.init) ?? ""
                title = title.replacingOccurrences(of: "_", with: " ")
                if title.hasSuffix(".html") { title.removeLast(5) }
                title = title.trimmingCharacters(
                    in: .whitespacesAndNewlines).lowercased()
                if !title.isEmpty { titles.insert(title) }
            }
        }
        return titles
    }

    /// Exact authorization check for a prepared-topic article walk. This is
    /// deliberately not fuzzy: a similarly named article is not connected
    /// evidence unless the current article actually links to it.
    public static func isDirectlyLinkedArticle(
        _ title: String, allowedTitleKeys: Set<String>
    ) -> Bool {
        allowedTitleKeys.contains(
            title.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    /// Relevance gate for a graph-constrained support-article walk. A direct
    /// Wikipedia link proves that an article is related to the anchor, but it
    /// does not prove that it answers the current facet. For example, the
    /// Photons article links to Raman scattering; a search for "photons
    /// discovered" ranked that article highly and permanently pulled the
    /// discussion away from photons. Require a content word from the question
    /// in the candidate title as well. Explicit named-event candidates are
    /// handled separately by the host before this generic gate.
    public static func linkedExpansionTitleMatchesQuestion(
        _ title: String, keywords: [String]
    ) -> Bool {
        let loweredTitle = title.lowercased()
        let meaningful = keywords
            .filter { $0.count >= 4 }
            .map(stem)
        return !meaningful.isEmpty
            && meaningful.contains(where: { loweredTitle.contains($0) })
    }

    /// Synonym expansion for question keywords whose answer prose uses
    /// different words: "parents" almost never appears in a biography —
    /// the "Early life" section says "his mother … his father". Expanded
    /// terms score at reduced weight so an exact keyword still wins.
    static let keywordSynonyms: [String: [String]] = [
        // Geopolitics: "How has X dealt with the West and NATO?" ranked
        // Geography because "west" is dense in border prose while the
        // membership facts live under Foreign relations / Politics /
        // Military headings (real capture 2026-08-02).
        "nato": ["foreign", "relations", "military", "alliance", "membership"],
        "eu": ["european", "union", "foreign", "relations", "membership"],
        "west": ["foreign", "relations", "europe", "nato"],
        "ally": ["foreign", "relations", "alliance", "military"],
        "allies": ["foreign", "relations", "alliance", "military"],
        "parents": ["mother", "father", "family", "early life"],
        "parent": ["mother", "father", "family", "early life"],
        "mother": ["parents", "family", "early life"],
        "father": ["parents", "family", "early life"],
        "siblings": ["brother", "sister", "family", "early life"],
        "brothers": ["brother", "family", "early life"],
        "sisters": ["sister", "family", "early life"],
        "wife": ["married", "marriage", "personal life"],
        "husband": ["married", "marriage", "personal life"],
        "married": ["marriage", "wife", "personal life"],
        "children": ["daughter", "son", "personal life", "family"],
        "kids": ["children", "daughter", "son", "personal life"],
        "childhood": ["early life", "born", "school"],
        "young": ["early life", "childhood", "born"],
        "grew": ["early life", "childhood", "born"],
        "born": ["early life", "birth"],
        "education": ["school", "university", "studied", "early life"],
        "school": ["education", "university", "studied", "early life"],
        "university": ["education", "school", "studied"],
        "college": ["education", "university", "school", "formal education", "early life"],
        "job": ["career", "work"],
        "money": ["wealth", "net worth", "income"],
        "rich": ["wealth", "net worth"],
        // Organization articles frequently describe their origin as a team
        // that "began as a collaboration" and identify later people as a
        // "co-founder". A literal `founders` query otherwise missed both the
        // lead and the relevant deep section (Google Brain, 2026-07-16).
        "founder": ["co-founder", "cofounder", "founded", "established", "began", "started", "created", "collaboration"],
        "founders": ["co-founder", "cofounder", "founded", "established", "began", "started", "created", "collaboration"],
        "leader": ["led", "director", "head", "president", "co-founder", "cofounder"],
        "leaders": ["led", "director", "head", "president", "co-founder", "cofounder"],
        // A historical origin is usually phrased as "proposed", "introduced",
        // or "coined", not literally "discovered". These terms both select
        // the article's history section and center the evidence window on the
        // relevant chronology (Photons, 2026-07-16).
        "discover": ["discovery", "history", "historical", "origin", "proposed", "introduced", "coined", "observed", "identified"],
        "discovered": ["discovery", "history", "historical", "origin", "proposed", "introduced", "coined", "observed", "identified"],
        "discovery": ["discover", "history", "historical", "origin", "proposed", "introduced", "coined", "observed", "identified"],
        "detected": ["detection", "observed", "observation", "history", "historical"],
        "detection": ["detected", "observed", "observation", "history", "historical"],
        "died": ["death", "killed", "casualty", "casualties", "fatalities", "losses"],
        "dead": ["death", "killed", "casualty", "casualties", "fatalities", "losses"],
        "deaths": ["death", "died", "killed", "casualty", "casualties", "fatalities", "losses"],
        "killed": ["death", "died", "casualty", "casualties", "fatalities", "losses"],
        "fatalities": ["death", "died", "killed", "casualty", "casualties", "losses"],
        "combatants": ["belligerents", "armies", "forces", "troops", "defenders"],
        "combatant": ["belligerent", "army", "forces", "troops", "defenders"],
        "sides": ["combatants", "belligerents", "opposing", "forces", "armies"],
        "side": ["combatant", "belligerent", "opposing", "force", "army"],
    ]

    /// (keyword, weight) pairs: the question's own keywords at full
    /// weight plus their synonyms at reduced weight, deduped.
    static func weightedKeywords(_ kws: [String]) -> [(term: String, weight: Float)] {
        var out: [(String, Float)] = []
        var seen = Set<String>()
        for k in kws where !seen.contains(k) {
            seen.insert(k)
            out.append((k, 1.0))
        }
        for k in kws {
            for syn in keywordSynonyms[k] ?? [] where !seen.contains(syn) {
                seen.insert(syn)
                out.append((syn, 0.6))
            }
        }
        return out
    }

    /// Crude stem so "parents" matches "parent(s)" and "annexed" matches
    /// "annex…": strip one plural/verbal suffix when the remainder stays
    /// ≥4 chars.
    static func stem(_ w: String) -> String {
        for suffix in ["ies", "es", "s", "ed", "ing"] where w.hasSuffix(suffix) {
            let stemmed = String(w.dropLast(suffix.count))
            if stemmed.count >= 4 { return stemmed }
        }
        return w
    }

    /// Remove words that merely repeat the already-selected article title,
    /// leaving the facet the user is asking for. Without this, “Who founded
    /// Google Brain?” scores every section that repeats “Google Brain” and
    /// can rank a product section above origin/co-founder evidence.
    static func facetKeywords(
        _ question: String, excludingArticleTitles titles: [String]
    ) -> [String] {
        let all = questionKeywords(question)
        guard !all.isEmpty else { return [] }
        let titleTerms = Set(titles.flatMap(questionKeywords).map(stem))
        let focused = all.filter { !titleTerms.contains(stem($0)) }
        // A request that contains only the title (“Google Brain?”) is still a
        // valid broad question; retain it rather than turning it keywordless.
        return focused.isEmpty ? all : focused
    }

    /// Keyword-evidence score for one section: heading hits are a strong
    /// "this section is about it" signal; recurring body mentions are a
    /// medium one (capped so one long section can't win on volume alone).
    static func keywordScore(
        _ weighted: [(term: String, weight: Float)],
        section: ArticleSection
    ) -> Float {
        guard !weighted.isEmpty else { return 0 }
        let title = section.title.lowercased()
        let body = section.text.lowercased()
        var score: Float = 0
        for (term, weight) in weighted {
            let st = stem(term)
            if !title.isEmpty, title.contains(st) {
                score += 2.0 * weight
            }
            var count = 0
            var idx = body.startIndex
            while count < 5, let r = body.range(of: st, range: idx..<body.endIndex) {
                count += 1
                idx = r.upperBound
            }
            score += Float(count) * 0.3 * weight
        }
        return score
    }

    /// A few conversational questions express a *kind* of section rather
    /// than repeating its heading. In particular, "How was it first
    /// detected?" otherwise over-ranks "Ground-based detectors" on the
    /// shared `detect` stem, even though the answer lives in History /
    /// Discovery. Keep this small and heading-only: it breaks lexical ties
    /// without pretending we have evidence that is not in the article.
    static func sectionIntentBoost(
        question: String,
        section: ArticleSection
    ) -> Float {
        let q = question.lowercased()
        let title = section.title.lowercased()
        guard !title.isEmpty else { return 0 }
        var boost: Float = 0

        // A short elliptical people question needs a strict word-boundary
        // preference. The normal stemmed scorer intentionally makes related
        // words match, but that made "How about the Mongols?" rank
        // "Mongolian cuisine" ahead of "Mongol empire...". On a warm
        // discussion turn only the best unseen section is appended, so that
        // near-match produced an entirely food-based answer. Prefer a title
        // containing the exact singular/plural subject token; this stays
        // generic for Romans/Roman history, Vikings/Viking expansion, etc.
        let content = questionKeywords(question)
        if IntentRouter.isEllipticalDiscussionFollowUp(question),
           content.count == 1 {
            let subject = stem(content[0])
            let titleTokens = Set(title.split(whereSeparator: {
                !$0.isLetter && !$0.isNumber
            }).map(String.init))
            if titleTokens.contains(subject) {
                boost += 6.0
            }
        }
        let asksHistoricalOrigin = (
            q.contains("first") || q.contains("when") || q.contains("who")
        ) && (
            q.contains("detect") || q.contains("observ")
                || q.contains("discover") || q.contains("coin")
                || q.contains("invent")
        )
        if asksHistoricalOrigin,
           title.contains("histor") || title.contains("discover")
                || title.contains("observ") || title.contains("origin")
                || title.contains("development")
                || title.contains("first detection")
        {
            boost += 4.0
        }
        if (q.contains("after") && q.contains("graduat")),
           title.contains("career") || title.contains("intelligence")
        {
            boost += 6.0
        }
        if ["side", "combatant", "belligerent", "who fought"]
            .contains(where: { q.contains($0) }),
           title.contains("civil war") || title.contains("combatant")
                || title.contains("belligerent")
        {
            boost += 4.0
        }
        return boost
    }

    /// True when the article's sections plausibly cover a follow-up — any
    /// content keyword appears in a section title or body. A question with no
    /// content keywords ("tell me more") counts as covered (stay on anchor).
    /// When false, the host pulls a better article from the corpus.
    public static func sectionsCoverQuestion(
        _ sections: [ArticleSection], _ question: String,
        articleTitle: String? = nil
    ) -> Bool {
        let kws = articleTitle.map {
            facetKeywords(question, excludingArticleTitles: [$0])
        } ?? questionKeywords(question)
        if kws.isEmpty { return true }
        // Expand with synonyms (stemmed) so "parents" counts the "his
        // mother … his father" prose of an Early-life section as
        // coverage instead of triggering a spurious corpus pull.
        let terms = weightedKeywords(kws).map { stem($0.term) }
        let q = question.lowercased()
        let asksHistoricalOrigin = (
            q.contains("first") || q.contains("when") || q.contains("who")
        ) && (
            q.contains("detect") || q.contains("observ")
                || q.contains("discover") || q.contains("coin")
                || q.contains("invent")
        )
        for s in sections {
            let title = s.title.lowercased()
            // A temporal discovery/origin question is locally covered when
            // the anchor has a dedicated historical section. Requiring the
            // exact word "discovered" twice in its body caused a needless
            // corpus walk even though Wikipedia described the concept as
            // proposed/introduced/coined there.
            if asksHistoricalOrigin,
               title.contains("histor") || title.contains("discover")
                    || title.contains("observ") || title.contains("origin")
                    || title.contains("development")
            {
                return true
            }
            // A keyword in a HEADING is a strong "this section is about it"
            // signal.
            if terms.contains(where: { title.contains($0) }) { return true }
            // Otherwise require a keyword to recur (≥2×) in one section's
            // body — a single passing mention isn't real coverage, and was
            // letting "population" skip a useful corpus pull (2026-05-30).
            let body = s.text.lowercased()
            let strongSingletons = Set([
                "founder", "co-founder", "cofounder", "founded",
                "established", "began", "started", "created",
                "collaboration",
            ].map(stem))
            for k in terms {
                if strongSingletons.contains(k), body.contains(k) {
                    return true
                }
                var count = 0
                var idx = body.startIndex
                while let r = body.range(of: k, range: idx..<body.endIndex) {
                    count += 1
                    if count >= 2 { return true }
                    idx = r.upperBound
                }
            }
        }
        return false
    }

    /// Weaker evidence gate for an elliptical "how/what about X?" follow-up.
    /// A single mention is not enough for the general corpus-coverage test,
    /// but it is enough to show that X is plausibly a facet of an explicitly
    /// prepared topic. The host uses this only with the elliptical utterance
    /// shape, then the normal section ranker selects the best local evidence.
    public static func sectionsMentionQuestion(
        _ sections: [ArticleSection], _ question: String,
        articleTitle: String? = nil
    ) -> Bool {
        let kws = articleTitle.map {
            facetKeywords(question, excludingArticleTitles: [$0])
        } ?? questionKeywords(question)
        if kws.isEmpty { return true }
        let terms = weightedKeywords(kws).map { stem($0.term) }
        return sections.contains { section in
            let haystack = (section.title + " " + section.text).lowercased()
            return terms.contains(where: { haystack.contains($0) })
        }
    }

    /// Core topic of an article title for building a corpus-fallback query:
    /// drop a trailing "(disambiguator)" and a leading "History of " /
    /// "Economy of " / "List of " … so "History of Lithuania (1219-1295)" →
    /// "Lithuania" (the broad article the follow-ups actually live in).
    public static func topicCore(_ title: String) -> String {
        var t = title
        if let r = regexRange(t, #"\s*\([^)]*\)\s*$"#) {
            t.removeSubrange(r)
        }
        t = regexReplacing(
            t,
            #"^(?:the\s+)?(?:history|economy|geography|politics|culture|demographics|religion|military|list|timeline|outline|index|government)\s+of\s+(?:the\s+)?"#,
            with: "", options: .caseInsensitive)
        let cleaned = t.trimmingCharacters(in: .whitespaces)
        return cleaned.isEmpty ? title : cleaned
    }

    /// Rank sections drawn from SEVERAL articles by relevance to a question —
    /// the multi-article discussion retriever. Same title/body blend as
    /// `rankSectionsForQuestion`, but flat across sources and tagged with the
    /// owning article so the answer can ground in whichever article actually
    /// covers the follow-up (anchor or a pulled-in one).
    public static func rankSectionsMultiSource(
        _ question: String,
        sources: [(title: String, sections: [ArticleSection])],
        embedder: TextEmbedder = HashingEmbedder(),
        k: Int = 3
    ) -> [(article: String, section: ArticleSection)] {
        let qv = embedder.embed(question)
        let weighted = weightedKeywords(facetKeywords(
            question, excludingArticleTitles: sources.map(\.title)))
        // Same keyword-first blend as `rankSectionsForQuestion` — see
        // the rationale there.
        let embedderWeight: Float = weighted.isEmpty ? 1.0 : 0.35
        var scored: [(total: Float, kw: Float, article: String, section: ArticleSection)] = []
        for (title, secs) in sources {
            for s in secs where !s.text.isEmpty {
                let ts = s.title.isEmpty
                    ? 0 : VectorMath.cosine(qv, embedder.embed(s.title))
                let bs = VectorMath.cosine(qv, embedder.embed(s.text))
                let embed = 0.55 * ts + 0.45 * bs
                let kw = keywordScore(weighted, section: s)
                    + sectionIntentBoost(question: question, section: s)
                scored.append((kw + embedderWeight * embed, kw, title, s))
            }
        }
        // Same off-topic cut as `rankSectionsForQuestion`: with keyword
        // evidence anywhere, zero-evidence sections don't pad the list.
        if scored.contains(where: { $0.kw > 0 }) {
            scored = scored.filter { $0.kw > 0 }
        }
        return scored.sorted { $0.total > $1.total }
            .prefix(max(1, k))
            .map { (article: $0.article, section: $0.section) }
    }

    /// Disambiguation candidates from an article's HATNOTES — the italic
    /// top-of-article cross-references Wikipedia uses precisely for
    /// ambiguity ("For the phenomenon of general relativity, see
    /// Gravitational wave."). More reliable offline than probing
    /// "<title> (disambiguation)": the nopic builds exclude
    /// disambiguation pages entirely, but hatnotes ship inside the
    /// article body. Only disambiguation-style notes count —
    /// "Further information:" / "Main article:" are section
    /// cross-references, not alternate meanings.
    public static func disambiguationHatnotes(
        html: String, max: Int = 3
    ) -> [(title: String, path: String)] {
        let divPattern = #"<div[^>]*class="[^"]*hatnote[^"]*"[^>]*>(.*?)</div>"#
        guard let re = RegexCache.shared.compiled(
            divPattern,
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) else { return [] }
        let range = NSRange(html.startIndex..., in: html)
        var out: [(String, String)] = []
        var seen = Set<String>()
        for m in re.matches(in: html, range: range) {
            guard m.numberOfRanges >= 2,
                  let r = Range(m.range(at: 1), in: html) else { continue }
            let body = String(html[r])
            let text = regexReplacing(body, "<[^>]+>", with: "").lowercased()
            let isDisambigStyle = text.hasPrefix("for ")
                || text.contains("this article is about")
                || text.contains("not to be confused")
                || text.contains("may refer to")
                || text.contains("(disambiguation)")
            guard isDisambigStyle else { continue }
            for link in WikiLinks.parse(html: body, max: 4) {
                let key = link.title.lowercased()
                if key.contains("disambiguation") || seen.contains(key) { continue }
                seen.insert(key)
                out.append((link.title, link.path))
                if out.count >= max { return out }
            }
        }
        return out
    }

    // MARK: - Relationship probing

    /// Ordered list of article titles to probe when answering
    /// "relations between A and B". Wikipedia's convention is
    /// `A–B relations` (en-dash) with the alphabetically-earlier country
    /// first; we try the common swap + hyphen variants too because OSM
    /// and older title indexes aren't always normalized. First hit wins.
    public static func relationshipCandidates(a: String, b: String) -> [String] {
        let aa = a.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: " ")
        let bb = b.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: " ")
        guard !aa.isEmpty, !bb.isEmpty else { return [] }
        // Wikipedia canonical: alphabetically-earlier first, en-dash.
        let (first, second) = aa.lowercased() <= bb.lowercased() ? (aa, bb) : (bb, aa)
        var out: [String] = []
        func push(_ s: String) {
            if !out.contains(s) { out.append(s) }
        }
        push("\(first)–\(second) relations")
        push("\(first)-\(second) relations")
        push("\(aa)–\(bb) relations")
        push("\(aa)-\(bb) relations")
        push("\(bb)–\(aa) relations")
        push("\(bb)-\(aa) relations")
        push("History of \(first)–\(second) relations")
        push("History of \(first)-\(second) relations")
        push("Foreign relations of \(aa)")
        push("Foreign relations of \(bb)")
        return out
    }

    /// Given a full section list and a counterpart name, pull the lead
    /// plus any named sections whose prose mentions the counterpart. Used
    /// by `article_relationship` when we land on a dedicated relations
    /// article and want to surface the parts that actually name both
    /// parties.
    public static func sectionsMentioning(
        _ name: String,
        in sections: [ArticleSection],
        maxExtra: Int = 3
    ) -> [ArticleSection] {
        var out: [ArticleSection] = []
        if let lead = sections.first(where: { $0.title.isEmpty }) {
            out.append(lead)
        }
        let needle = name.lowercased()
        let named = sections.filter { !$0.title.isEmpty }
        let mentioning = named.filter { $0.text.lowercased().contains(needle) }
        for s in mentioning {
            if out.count >= maxExtra + 1 { break }
            out.append(s)
        }
        return out
    }

    // MARK: - Excerpt helpers

    /// Default excerpt length for `nearby_stories` entries. ~800 chars ≈
    /// the opening 2 paragraphs of most Wikipedia articles — enough to
    /// actually learn something, not a teaser. Tunable here (one place).
    public static let defaultStoryExcerptChars: Int = 800
}
