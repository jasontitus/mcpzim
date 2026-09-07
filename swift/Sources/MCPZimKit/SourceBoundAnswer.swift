// SPDX-License-Identifier: MIT
import Foundation

/// The factual output boundary. Ranking chooses source sentences; it never
/// authorizes generated paraphrases. Every emitted excerpt has an exact
/// location in a retrieved ZIM passage. This proves provenance, not that an
/// encyclopedia's own claims are true or that retrieval found every answer.
public enum SourceBoundAnswer {
    public struct Passage: Equatable, Sendable {
        public let article: String
        public let section: String
        public let library: String?
        public let text: String
        public init(article: String, section: String = "lead", library: String? = nil, text: String) {
            self.article = article; self.section = section
            self.library = library; self.text = text
        }
    }
    public struct Excerpt: Equatable, Sendable {
        public let text: String
        public let passageIndex: Int
    }
    public struct Reply: Equatable, Sendable {
        public let excerpts: [Excerpt]
        public var hasEvidence: Bool { !excerpts.isEmpty }
        public var text: String { excerpts.map(\.text).joined(separator: "\n\n") }
    }

    /// Only whitespace and citation markers are normalized. Negation,
    /// numbers, names, modality, and word order remain untouched.
    public static func clean(_ text: String) -> String {
        ArticleHeuristics.stripCitations(text)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func sentences(_ text: String) -> [String] {
        ArticleHeuristics.sentenceChunks(clean(text))
    }

    public static func missingAnswer(topic: String? = nil) -> String {
        let subject = topic.map { " on \($0)" } ?? ""
        return "I don't see an answer to that in the ZIM passages I found\(subject). Try a more specific question or another article."
    }

    /// `sectionOverview` permits an overview fallback for a validated source
    /// navigation choice only; passages must be scoped to that section.
    /// Prefer focused facts when available (e.g. schools within Education).
    /// Free-form questions still require coverage of their factual facets.
    public static func answer(question: String, topic: String, passages: [Passage],
                              maxSentences: Int = 3, maxCharacters: Int = 1_200,
                              sectionOverview: Bool = false) -> Reply {
        if sectionOverview {
            let focused = answer(question: question, topic: topic, passages: passages,
                maxSentences: maxSentences, maxCharacters: maxCharacters)
            if focused.hasEvidence { return focused }
        }
        struct Candidate {
            let text: String
            let passage: Int
            let position: Int
            let score: Double
            let parentIdentity: Bool
        }
        let titleTerms = Set(([topic] + passages.map(\.article))
            .flatMap(ArticleHeuristics.questionKeywords).map(ArticleHeuristics.stem))
        let generic: Set<String> = ["please", "explain", "describe", "overview", "introduction",
            "brief", "detail", "details", "information", "learn", "give",
            "compare", "comparison", "difference", "differences", "must", "can", "could", "should", "would"]
        // These phrases describe the requested conversation, rather than
        // additional facts that a source sentence must literally repeat.
        let relevanceQuestion = question
            .replacingOccurrences(of: #"(?i)\bcareer develop(?:ed)?\b"#, with: "career", options: .regularExpression)
            .replacingOccurrences(of: #"(?i)\bdealt with\b"#, with: "", options: .regularExpression)
        let lower = relevanceQuestion.lowercased()
        let parentQuestion = !sectionOverview && ArticleHeuristics.questionKeywords(relevanceQuestion).contains {
            ["parent", "parents", "mother", "father"].contains($0)
                && !titleTerms.contains(ArticleHeuristics.stem($0))
        }
        let asksParentIdentity = parentQuestion && lower.range(
            of: #"^(?:(?:and|so)\s+)?who\b|\bnames?\b"#, options: .regularExpression) != nil
        let keywords = sectionOverview ? [] : ArticleHeuristics.questionKeywords(relevanceQuestion).filter {
            !titleTerms.contains(ArticleHeuristics.stem($0)) && !generic.contains($0)
                && !(parentQuestion && ["name", "names"].contains($0))
        }
        let weighted = ArticleHeuristics.weightedKeywords(keywords)
        let variants = keywords.map(evidenceTerms)
        let broad = keywords.isEmpty
        let specialized = broad ? nil : ArticleHeuristics.groundedExtractiveAnswer(
            question: question, passages: passages.map(\.text),
            passageLabels: passages.map { $0.article + " " + $0.section })
        // A heuristic may help select sentences, but even its prose must
        // resolve back to source sentences. Never copy its rewritten output.
        let preferred = Set(specialized.map(sentences) ?? [])
        var candidates: [Candidate] = []
        var seenArticles = Set<String>()
        for (index, passage) in passages.enumerated() {
            let firstPassageForArticle = seenArticles.insert(passage.article).inserted
            let heading = passage.section.lowercased()
            let headingWords = Set(ArticleHeuristics.questionKeywords(heading).map(ArticleHeuristics.stem))
            // Heading matches use literal terms plus narrow spelling aliases,
            // not broad retrieval synonyms (NATO ≠ every foreign-policy heading).
            func headingCovers(_ keyword: String) -> Bool {
                headingWords.contains(ArticleHeuristics.stem(keyword))
                    || (keyword == "west" && headingWords.contains("western"))
            }
            let headingCoverage = keywords.filter(headingCovers).count
            let headingOverview = !broad && headingCoverage == keywords.count
            for (position, sentence) in sentences(passage.text).enumerated() {
                guard sentence.count >= 8, sentence.count <= 2_400 else { continue }
                let words = Set(ArticleHeuristics.questionKeywords(sentence).map(ArticleHeuristics.stem))
                // A source can identify both parents in one clause without
                // repeating "mother" or "father". Keep that entire clause;
                // never assign either name to a role using model knowledge.
                let collectiveParentIdentity = parentQuestion && sentence.range(of:
                    #"\b(?i:parents\b[^.!?]{0,100}\b(?:were|are))\s+\p{Lu}[\p{L}'’\-]+"#,
                    options: .regularExpression) != nil
                // A section supplies the context that its sentences need not
                // repeat. Every remaining question facet still needs evidence:
                // "Early life" + "secret dossier" cannot answer "secret password".
                let covered = zip(keywords, variants).filter { keyword, terms in
                    headingCovers(keyword)
                        || (collectiveParentIdentity && ["parent", "parents", "mother", "father"].contains(keyword))
                        || terms.contains { term in
                        words.contains(ArticleHeuristics.stem(term))
                            || (term.contains(" ") && sentence.lowercased().contains(term))
                    }
                }.count
                if !broad && covered < keywords.count { continue }
                var score = weighted.reduce(0.0) { sum, term in
                    sum + (words.contains(ArticleHeuristics.stem(term.term)) ? Double(term.weight) : 0)
                }
                if preferred.contains(sentence) { score += 4 }
                var parentIdentity = false
                if parentQuestion {
                    // Prefer an actual parent-identification clause over a
                    // business story that merely mentions someone's father.
                    parentIdentity = sentence.range(of:
                        #"(?i)\b(?:parents\b[^.!?]{0,100}\b(?:were|are)|(?:mother|father)(?: was| is|,)\s+\p{L}+|born to|son of|daughter of)\b"#,
                        options: .regularExpression) != nil
                    if parentIdentity { score += 8 }
                    if asksParentIdentity && !parentIdentity { continue }
                    if !["parent", "mother", "father", "born to", "son of", "daughter of"]
                        .contains(where: { sentence.lowercased().contains($0) }) { continue }
                }
                if broad {
                    // An overview begins at the anchor lead, in source order.
                    score = (firstPassageForArticle ? 100 : 10) - Double(position) * 0.1
                } else if headingOverview {
                    // Answer the requested section in source order instead
                    // of rewarding an incidental word match elsewhere.
                    score = 1_000 - Double(position) * 0.1
                } else {
                    score += Double(headingCoverage) * 4
                    if score == 0 { continue }
                }
                candidates.append(Candidate(text: sentence, passage: index, position: position,
                    score: score, parentIdentity: parentIdentity))
            }
        }
        if parentQuestion, candidates.contains(where: \.parentIdentity) {
            candidates.removeAll { !$0.parentIdentity }
        }
        candidates.sort {
            if $0.score != $1.score { return $0.score > $1.score }
            if $0.passage != $1.passage { return $0.passage < $1.passage }
            return $0.position < $1.position
        }
        var selected: [Candidate] = []
        var seen = Set<String>()
        var characters = 0
        for candidate in candidates {
            guard selected.count < max(1, min(8, maxSentences)) else { break }
            guard seen.insert(candidate.text).inserted else { continue }
            // Keep a complete first sentence even when it exceeds the soft
            // display budget; never cut off a qualification or negation.
            if !selected.isEmpty && characters + candidate.text.count > maxCharacters { continue }
            selected.append(candidate)
            characters += candidate.text.count
        }
        // Context and attribution can change when sentences are reordered.
        // Keep each source's selected sentences in their original order.
        selected.sort { $0.passage == $1.passage ? $0.position < $1.position : $0.passage < $1.passage }
        return Reply(excerpts: selected.map { Excerpt(text: $0.text, passageIndex: $0.passage) })
    }

    /// Search aliases are intentionally broad; some cannot substitute for
    /// evidence about a named organization or a specific relative.
    private static func evidenceTerms(_ keyword: String) -> [String] {
        switch keyword {
        case "nato": return ["nato", "north atlantic treaty organization", "north atlantic treaty organisation"]
        case "west": return ["west", "western"]
        case "mother", "father": return [keyword]
        case "school", "education": return ["school", "education", "university", "college", "studied"]
        default: return ArticleHeuristics.weightedKeywords([keyword]).map(\.term)
        }
    }

    /// Accept source bodies only from known article tools. Tool errors,
    /// model summaries, search snippets, and arbitrary JSON are not evidence.
    public static func passages(toolName: String, result: [String: Any]) -> [Passage] {
        guard result["error"] == nil else { return [] }
        let library = result["zim"] as? String
        let title = result["title"] as? String ?? result["path"] as? String ?? "ZIM article"
        switch toolName {
        case "article_overview", "discuss_article":
            return ((result["sections"] as? [[String: Any]]) ?? []).compactMap { row in
                guard let text = row["text"] as? String, !text.isEmpty else { return nil }
                return Passage(article: title, section: row["title"] as? String ?? "lead",
                    library: library, text: text)
            }
        case "get_article_section", "get_article_by_title":
            guard let text = result["text"] as? String, !text.isEmpty else { return [] }
            return [Passage(article: title, section: result["section"] as? String ?? "lead",
                library: library, text: text)]
        case "get_article":
            guard let text = result["text"] as? String else { return [] }
            if (result["mimetype"] as? String)?.contains("html") == true {
                return ArticleSections.parse(html: text).map {
                    Passage(article: title, section: $0.title, library: library, text: $0.text)
                }
            }
            return [Passage(article: title, library: library, text: text)]
        case "compare_articles":
            return ((result["articles"] as? [[String: Any]]) ?? []).flatMap {
                passages(toolName: "article_overview", result: $0)
            }
        default: return []
        }
    }
}
