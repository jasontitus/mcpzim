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
