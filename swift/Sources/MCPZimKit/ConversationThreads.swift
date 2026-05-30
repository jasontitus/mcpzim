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
        case "near_named_place", "near_places",
             "nearby_stories", "nearby_stories_at_place":
            return placesThreads(result)
        case "article_overview", "get_article_section", "narrate_article":
            return articleThreads(result)
        case "compare_articles":
            return compareThreads(result)
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
                guard let title = s["title"] as? String,
                      !title.isEmpty,
                      !skipSections.contains(title.lowercased())
                else { continue }
                out.append(DiscoveryThread(
                    label: title, kind: .topic, source: .section))
            }
        }
        return out
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
                if key.isEmpty || seen.contains(key) || discussed.contains(key) {
                    continue
                }
                seen.insert(key)
                out.append(t)
                if out.count >= max { return out }
            }
        }
        return out
    }

    // MARK: - Deterministic offer caption (no-LLM fast path)

    /// A short, natural "where next" line for the fast path that runs no
    /// model. The LLM path instead gets these threads in the preamble and
    /// writes its own offer.
    public static func offer(_ threads: [DiscoveryThread]) -> String? {
        let picked = Array(threads.prefix(3))
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

    private static let skipSections: Set<String> = [
        "see also", "references", "notes", "footnotes", "citations",
        "further reading", "external links", "bibliography", "sources",
    ]

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

    public static func parse(html: String, max: Int = 8) -> [Link] {
        let pattern = #"<a\b[^>]*?href="([^"]*)"[^>]*>(.*?)</a>"#
        guard let re = try? NSRegularExpression(
            pattern: pattern,
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) else { return [] }

        let range = NSRange(html.startIndex..., in: html)
        var out: [Link] = []
        var seen = Set<String>()

        for m in re.matches(in: html, range: range) {
            guard m.numberOfRanges >= 3,
                  let hrefR = Range(m.range(at: 1), in: html),
                  let textR = Range(m.range(at: 2), in: html)
            else { continue }

            let href = String(html[hrefR])
            let rawText = String(html[textR])
            let title = decodeAndStrip(rawText)

            guard isArticleLink(href), !title.isEmpty, title.count >= 2 else {
                continue
            }
            // Skip pure-number / citation anchor text ("[1]", "12").
            if title.allSatisfy({ $0.isNumber || $0 == "[" || $0 == "]" }) {
                continue
            }
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
        var t = s.replacingOccurrences(
            of: "<[^>]+>", with: "", options: .regularExpression)
        let entities = [
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&quot;": "\"", "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
        ]
        for (k, v) in entities { t = t.replacingOccurrences(of: k, with: v) }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
