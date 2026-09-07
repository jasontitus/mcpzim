// SPDX-License-Identifier: MIT
//
// Exact sentence provenance. A lexical overlap score is not evidence that
// a model preserved the source's meaning; only complete source sentences
// receive a match. Factual answer text is rendered by SourceBoundAnswer.

import Foundation

public struct SentenceAttribution: Equatable, Sendable {
    public let sentence: String
    /// Index into the passage list handed to `attribute`, or nil when no
    /// complete source sentence matches.
    public let passageIndex: Int?
    /// One for an exact source sentence, zero otherwise. This is not a
    /// probability or a semantic entailment score.
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

    /// Exact source sentences only. Lexical overlap cannot establish that
    /// a paraphrase preserves negation, roles, causation, or attribution.

    public static func attribute(
        answer: String, passages: [Passage]
    ) -> [SentenceAttribution] {
        let sourceSentences = passages.map { Set(SourceBoundAnswer.sentences($0.text)) }
        return SourceBoundAnswer.sentences(answer).map { sentence in
            let index = sourceSentences.firstIndex { $0.contains(sentence) }
            return SentenceAttribution(sentence: sentence, passageIndex: index,
                support: index == nil ? 0 : 1)
        }
    }

    /// One compact log line per answer, e.g.
    /// `s1→Apple TV (device)§Background 1.00 · s2→UNSUPPORTED 0.00`.
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

}
