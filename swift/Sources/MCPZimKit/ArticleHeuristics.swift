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
    /// (full outline parse). Two reads — libzim caches entries so the
    /// second is cheap, but if this ever shows up in profiling it's the
    /// obvious thing to collapse into one service primitive.
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
        // "X may refer to:" is the canonical disambig opener; match
        // as a word-bounded substring so we don't catch "also refer
        // to" in body prose.
        if lead.range(of: #"\bmay refer to\b"#,
                      options: .regularExpression) != nil {
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
    public static func stripCitations(_ text: String) -> String {
        let patterns = [
            #"\[\s*\d{1,3}\s*\]"#,                          // [1], [12]
            #"\[\s*[a-zA-Z]\s*\]"#,                         // [a], [B]
            #"\[\s*citation needed\s*\]"#,                  // [citation needed]
            #"\[\s*(?:note|nb|sic|clarification needed)[^\]]*\]"#,
        ]
        var out = text
        for p in patterns {
            out = out.replacingOccurrences(
                of: p, with: "", options: [.regularExpression, .caseInsensitive]
            )
        }
        out = out.replacingOccurrences(
            of: #"[ \t]{2,}"#, with: " ", options: .regularExpression
        )
        out = out.replacingOccurrences(
            of: #" +([.,;:!?])"#, with: "$1", options: .regularExpression
        )
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
            let body = stripCitations(s.text).trimmingCharacters(in: .whitespacesAndNewlines)
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
        let ranked = sections.indices
            .filter { $0 != leadIdx && !sections[$0].text.isEmpty }
            .map { i -> (Int, Float) in
                let s = sections[i]
                let titleScore = s.title.isEmpty
                    ? 0 : VectorMath.cosine(qv, embedder.embed(s.title))
                let bodyScore = VectorMath.cosine(qv, embedder.embed(s.text))
                return (i, 0.55 * titleScore + 0.45 * bodyScore)
            }
            .sorted { $0.1 > $1.1 }
            .prefix(max(0, k - 1))
            .map { sections[$0.0] }
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
            "there","into","from","work","works","happen","happened",
        ]
        var seen = Set<String>()
        var out: [String] = []
        for w in q.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted) {
            guard w.count >= 3, !stop.contains(w), !seen.contains(w) else { continue }
            seen.insert(w); out.append(w)
        }
        return out
    }

    /// True when the article's sections plausibly cover a follow-up — any
    /// content keyword appears in a section title or body. A question with no
    /// content keywords ("tell me more") counts as covered (stay on anchor).
    /// When false, the host pulls a better article from the corpus.
    public static func sectionsCoverQuestion(
        _ sections: [ArticleSection], _ question: String
    ) -> Bool {
        let kws = questionKeywords(question)
        if kws.isEmpty { return true }
        let hay = sections
            .map { ($0.title + " " + $0.text).lowercased() }
            .joined(separator: " ")
        return kws.contains { hay.contains($0) }
    }

    /// Core topic of an article title for building a corpus-fallback query:
    /// drop a trailing "(disambiguator)" and a leading "History of " /
    /// "Economy of " / "List of " … so "History of Lithuania (1219-1295)" →
    /// "Lithuania" (the broad article the follow-ups actually live in).
    public static func topicCore(_ title: String) -> String {
        var t = title
        if let r = t.range(of: #"\s*\([^)]*\)\s*$"#, options: .regularExpression) {
            t.removeSubrange(r)
        }
        t = t.replacingOccurrences(
            of: #"^(?:history|economy|geography|politics|culture|demographics|religion|military|list|timeline|outline|index|government)\s+of\s+(?:the\s+)?"#,
            with: "", options: [.regularExpression, .caseInsensitive])
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
        var scored: [(Float, String, ArticleSection)] = []
        for (title, secs) in sources {
            for s in secs where !s.text.isEmpty {
                let ts = s.title.isEmpty
                    ? 0 : VectorMath.cosine(qv, embedder.embed(s.title))
                let bs = VectorMath.cosine(qv, embedder.embed(s.text))
                scored.append((0.55 * ts + 0.45 * bs, title, s))
            }
        }
        return scored.sorted { $0.0 > $1.0 }
            .prefix(max(1, k))
            .map { (article: $0.1, section: $0.2) }
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
