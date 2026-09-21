import Foundation

/// A small, deterministic answer contract. Retrieval synonyms are not proof
/// of a relationship: an authorship question needs a subject, an authorship
/// assertion and an identifiable work, not an incidental occurrence of 'write'.
public struct EvidenceQuestion: Equatable, Sendable {
    public enum Relation: String, Sendable { case authoredWorks, birthplace }
    public let relation: Relation
    public let workKind: String?
    private let permitsArticleSubject: Bool

    public static func parse(_ question: String) -> EvidenceQuestion? {
        let q = question.lowercased()
        if q.range(of: #"\bwhere\b"#, options: .regularExpression) != nil,
           q.range(of: #"\bborn\b"#, options: .regularExpression) != nil {
            let relativeOrMultiple = q.range(of: #"['’]s\b|\b(?:mother|father|parents|brother|sister|son|daughter|wife|husband|and|or)\b"#, options: .regularExpression) != nil
            return .init(relation: .birthplace, workKind: nil, permitsArticleSubject: !relativeOrMultiple)
        }
        guard q.range(of: #"^\s*(?:what|which)\b"#, options: .regularExpression) != nil,
              q.range(of: #"\b(?:write|wrote|written|author|authored|pen|penned)\b"#, options: .regularExpression) != nil else { return nil }
        let kinds = ["play", "book", "novel", "poem", "song", "essay", "story"]
        let words = Set(ArticleHeuristics.questionKeywords(q).map(ArticleHeuristics.stem))
        return .init(relation: .authoredWorks, workKind: kinds.first { words.contains($0) }, permitsArticleSubject: true)
    }

    public static func lemma(_ word: String) -> String {
        switch word.lowercased() {
        case "write", "writes", "wrote", "written", "writing": return "write"
        case "authored", "authoring", "authors": return "author"
        case "penned", "penning": return "pen"
        default: return word
        }
    }

    /// Conservative English patterns. Unsupported syntax falls back to more
    /// evidence, never to model knowledge. Quotes and other people's actions
    /// cannot borrow the article subject merely by occurring in its article.
    func accepts(sentence: String, preceding: [String], article: String,
                 topic: String, section: String) -> Bool {
        let names = [topic, topic.split(separator: " ").last.map(String.init) ?? topic]
            .map(NSRegularExpression.escapedPattern(for:))
        let subject = "(?:" + names.joined(separator: "|") + ")"
        func matches(_ text: String, _ pattern: String) -> Bool {
            text.range(of: pattern, options: .regularExpression) != nil
        }
        if relation == .birthplace {
            guard permitsArticleSubject, EntityResolutionPolicy.sameTitle(article, topic) else { return false }
            // Keep the full title for source identity, but remove a trailing
            // disambiguator when matching the person's name in source prose.
            let person = topic.replacingOccurrences(of: #"\s+\([^()]+\)$"#, with: "", options: .regularExpression)
            let birthNames = [person, person.split(separator: " ").last.map(String.init) ?? person]
                .map(NSRegularExpression.escapedPattern(for:))
            let birthActor = "(?:" + birthNames.joined(separator: "|") + ")"
            // Require a named assertion: even a preceding subject sentence
            // can introduce another person whose pronoun follows it.
            guard let assertion = sentence.range(of: "(?i)^" + birthActor + #"\s+(?:was|is)\s+born\b"#,
                                                  options: .regularExpression) else { return false }
            let remainder = String(sentence[assertion.upperBound...])
            let placePattern = #"(?i:\b(?:in|at)\s+(?:the\s+)?)(?!(?i:January|February|March|April|May|June|July|August|September|October|November|December)\b)\p{Lu}[\p{L}’'-]+"#
            guard let place = remainder.range(of: placePattern, options: .regularExpression) else { return false }
            let between = String(remainder[..<place.lowerBound])
            return !matches(between, #"(?i)[;.!?]|\b(?:and|but|whose|mother|father|parents|wife|husband|brother|sister|son|daughter)\b"#)
        }
        // Pronouns are accepted only in an unbroken subject-anchored run.
        // A new named subject, quotation or unrecognized sentence breaks it.
        var anchored = false
        if EntityResolutionPolicy.sameTitle(article, topic) {
            for prior in preceding {
                if matches(prior, "(?i)^" + subject + #"\s*(?:\(|was\b|is\b|was born\b)"#) { anchored = true }
                else if !matches(prior, #"(?i)^(?:he|she|his|her)\b"#) { anchored = false }
            }
        }
        let actor = anchored ? "(?:" + subject + "|[Hh]e|[Ss]he)" : subject
        let possessive = anchored ? "(?:" + subject + "['’]s|[Hh]is|[Hh]er)" : subject + "['’]s"
        let verbs = #"(?:wrote|writes|has written|had written|authored|penned)"#
        let assertion = "(?i)^" + actor + #"\s+"# + verbs + #"\s+"#
        let list = "(?i)^" + possessive + #"\s+(?:plays|books|novels|poems|songs|essays|stories|works)\s+(?:include|included|are|were)\s+"#
        guard let range = sentence.range(of: assertion, options: .regularExpression)
                ?? sentence.range(of: list, options: .regularExpression) else { return false }
        let object = String(sentence[range.upperBound...])
        // Don't treat hypothetical/negative actions as authored works.
        guard !matches(object, #"(?i)^(?:no|not|nothing|that|about|to|for|prolifically|frequently)\b"#),
              !matches(object, #"(?i)\b(?:would|might|could|if)\b"#) else { return false }
        // An object must contain a named work. This also rejects 'many plays'
        // and 'until his death' as answers to 'what did he write?'.
        guard matches(object, #"(?:^|\s|[“\"])[A-Z][\p{L}’'-]{2,}"#) else { return false }
        if let workKind {
            let words = Set(ArticleHeuristics.questionKeywords(sentence + " " + section).map(ArticleHeuristics.stem))
            guard words.contains(workKind) else { return false }
            let explicitKinds = ["play", "book", "novel", "poem", "song", "essay", "story"]
            let objectWords = Set(ArticleHeuristics.questionKeywords(object).map(ArticleHeuristics.stem))
            if explicitKinds.contains(where: { $0 != workKind && objectWords.contains($0) }),
               !objectWords.contains(workKind) { return false }
        }
        return true
    }
}

/// Only spelling presentation differences authorize an automatic title-index
/// match. Real redirect paths are resolved separately by the ZIM reader.
public enum EntityResolutionPolicy {
    public static func sameTitle(_ a: String, _ b: String) -> Bool {
        func normalized(_ s: String) -> String {
            s.replacingOccurrences(of: "_", with: " ")
                .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
        return normalized(a) == normalized(b)
    }
}
