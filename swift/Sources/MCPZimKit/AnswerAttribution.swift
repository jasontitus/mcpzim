// SPDX-License-Identifier: MIT
//
// Per-sentence provenance for grounded answers. After generation, each
// sentence of the model's prose is aligned against the EXACT passages that
// were in the prompt, deterministically — the model gets no vote on its own
// attribution. Sentences whose content isn't covered by any passage are
// flagged, making trained-data leakage and hallucination visible in the UI
// instead of silently blending into grounded text.
//
// Scoring is lexical (stemmed content-token coverage) with numeric tokens
// weighted 3×: a date, count, or measurement the passage never contained is
// the classic hallucination tell, and plain token overlap under-penalizes a
// sentence that copies the passage's nouns but invents its numbers.

import Foundation

public struct SentenceAttribution: Equatable, Sendable {
    public let sentence: String
    /// Index into the passage list handed to `attribute` — nil when no
    /// passage reaches the support threshold.
    public let passageIndex: Int?
    /// 0…1 coverage of the sentence's (weighted) content tokens by the best
    /// passage. Kept even for unsupported sentences so the UI/logs can show
    /// "near miss" vs "out of thin air".
    public let support: Double

    public var isSupported: Bool { passageIndex != nil }

    public init(sentence: String, passageIndex: Int?, support: Double) {
        self.sentence = sentence
        self.passageIndex = passageIndex
        self.support = support
    }
}

public enum AnswerAttribution {

    public struct Passage: Sendable {
        public let article: String
        public let section: String?
        public let text: String
        public init(article: String, section: String?, text: String) {
            self.article = article
            self.section = section
            self.text = text
        }
    }

    /// Below this weighted-coverage fraction a sentence is unsupported.
    /// Grounded prose paraphrases, so full coverage never happens; 0.5 with
    /// stemming catches inventions while tolerating normal rewording (tuned
    /// against the regression fixtures in AnswerAttributionTests).
    public static let supportThreshold = 0.5

    public static func attribute(
        answer: String, passages: [Passage]
    ) -> [SentenceAttribution] {
        guard !passages.isEmpty else { return [] }
        let passageTokens: [Set<String>] = passages.map { tokenSet($0.text) }
        return splitSentences(answer).map { sentence in
            let tokens = weightedTokens(sentence)
            // ≤2 content tokens with no numerics ("Sure!", "Of course.",
            // "Happy to help."): connective prose, not a factual claim —
            // never flag it.
            let hasNumeric = tokens.contains { $0.1 > 1 }
            guard tokens.count > 2 || hasNumeric else {
                return SentenceAttribution(
                    sentence: sentence, passageIndex: nil, support: 1)
            }
            var bestIdx: Int? = nil
            var bestScore = 0.0
            for (i, ptoks) in passageTokens.enumerated() {
                var covered = 0.0
                var total = 0.0
                var hasUnmatchedNumeric = false
                for (token, weight) in tokens {
                    total += weight
                    if ptoks.contains(token) {
                        covered += weight
                    } else if weight > 1 {
                        hasUnmatchedNumeric = true
                    }
                }
                // A passage must contain every numeric claim verbatim. A high
                // noun overlap must not bless an invented date, count, rank,
                // or measurement.
                let lexicalScore = total > 0 ? covered / total : 0
                let score = hasUnmatchedNumeric
                    ? min(lexicalScore, supportThreshold.nextDown) : lexicalScore
                if score > bestScore {
                    bestScore = score
                    bestIdx = i
                }
            }
            return SentenceAttribution(
                sentence: sentence,
                passageIndex: bestScore >= supportThreshold ? bestIdx : nil,
                support: bestScore)
        }
    }

    /// One compact log line per answer, e.g.
    /// `s1→Apple TV (device)§Background 0.82 · s2→UNSUPPORTED 0.21`.
    public static func logLine(
        _ attributions: [SentenceAttribution], passages: [Passage]
    ) -> String {
        attributions.enumerated().map { i, a in
            if let p = a.passageIndex {
                let src = passages[p]
                let sec = src.section.map { "§\($0)" } ?? ""
                return String(format: "s%d→%@%@ %.2f", i + 1, src.article, sec, a.support)
            }
            return String(format: "s%d→UNSUPPORTED %.2f", i + 1, a.support)
        }.joined(separator: " · ")
    }

    // MARK: - Internals

    static func splitSentences(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        var iterator = text.makeIterator()
        var prev: Character? = nil
        while let c = iterator.next() {
            current.append(c)
            if c == "\n" || (prev != nil && ".!?".contains(prev!) && c == " ") {
                let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !t.isEmpty { out.append(t) }
                current = ""
            }
            prev = c
        }
        let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { out.append(t) }
        return out
    }

    private static let functionWords: Set<String> = [
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
        "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "it", "its", "this", "that", "these", "those",
        "which", "who", "whose", "whom", "he", "she", "his", "her", "they",
        "their", "them", "we", "you", "i", "not", "no", "also", "such",
        "has", "have", "had", "can", "may", "will", "would", "into", "than",
        "then", "when", "while", "there", "where", "other", "both", "over",
        "under", "about", "after", "before", "between", "known",
    ]

    /// Stemmed content tokens with weights: numbers 3×, words 1×.
    static func weightedTokens(_ text: String) -> [(String, Double)] {
        var out: [(String, Double)] = []
        for raw in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
            let w = String(raw)
            if w.allSatisfy(\.isNumber) {
                // Years/counts/measurements — the highest-signal tokens.
                out.append((w, 3))
                continue
            }
            guard w.count >= 3, !functionWords.contains(w) else { continue }
            out.append((ArticleHeuristics.stem(w), 1))
        }
        return out
    }

    private static func tokenSet(_ text: String) -> Set<String> {
        var set = Set<String>()
        for raw in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
            let w = String(raw)
            if w.allSatisfy(\.isNumber) {
                set.insert(w)
                continue
            }
            guard w.count >= 3 else { continue }
            set.insert(ArticleHeuristics.stem(w))
        }
        return set
    }
}
