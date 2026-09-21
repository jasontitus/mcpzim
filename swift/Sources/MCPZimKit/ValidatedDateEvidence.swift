import Foundation

/// A small, conversation-local reuse record, populated only from a direct
/// source-bound answer. It cannot reuse background or resolve a new subject.
public struct ValidatedDateEvidence: Sendable {
    private let resolvedQuestion: String
    private let passages: [SourceBoundAnswer.Passage]

    public init?(question: String, passages: [SourceBoundAnswer.Passage], directFact: Bool) {
        guard directFact, ExplorationPlan.asksPastDate(question),
              !passages.isEmpty, passages.count <= 3,
              passages.reduce(0, { $0 + $1.text.count }) <= 3300 else { return nil }
        let text = passages.map(\.text).joined(separator: " ")
        let regex = try! NSRegularExpression(pattern: #"\b(?:1[0-9]{3}|20[0-9]{2})\b"#)
        let years = Set(regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
            .compactMap { Range($0.range, in: text).map { String(text[$0]) } })
        guard years.count == 1,
              text.range(of: #"(?i)\b(?:not|never|uncertain|unknown|disputed|might|may|could|would|if)\b|(?i)n['’]t\b"#,
                         options: .regularExpression) == nil else { return nil }
        self.resolvedQuestion = Self.key(question)
        self.passages = passages
    }

    public func answer(to followUp: String, currentQuestion: String?) -> [SourceBoundAnswer.Passage]? {
        let request = Self.key(followUp)
        guard ["what year", "which year"].contains(request),
              let currentQuestion, Self.key(currentQuestion) == resolvedQuestion else { return nil }
        return passages
    }

    private static func key(_ text: String) -> String {
        text.lowercased().replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
    }
}
