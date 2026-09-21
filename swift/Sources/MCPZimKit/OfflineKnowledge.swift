// SPDX-License-Identifier: MIT
import Foundation

/// Small, model-free content boundary shared by Siri and its tests. It never
/// calls the MCP dispatcher: a question cannot select a mutating/network tool.
public struct OfflineArticle: Codable, Equatable, Sendable {
    public let zim: String
    public let path: String
    public let title: String
    public init(zim: String, path: String, title: String) {
        self.zim = zim; self.path = path; self.title = title
    }
}

public struct OfflineKnowledge: Sendable {
    public enum Measurement: Sendable {
        case htmlBytes(Int), parsedBytes(Int), sections(Int), answerCharacters(Int)
    }
    // Wikipedia HTML includes extensive references, styling, and metadata.
    // Keep its storage budget separate from the prose/scoring budget below.
    public static let maxHTMLBytes = 8 * 1024 * 1024
    public enum Resolution: Equatable, Sendable {
        case article(OfflineArticle)
        /// Even a single fuzzy suggestion needs explicit user selection.
        case choices([OfflineArticle])
    }
    public let service: any ZimService
    public init(service: any ZimService) { self.service = service }

    public static func validatedInput(_ input: String) throws -> String {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, text.count <= 500,
              !text.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) && $0 != "\n" && $0 != "\t" })
        else { throw Failure.invalidInput }
        return text
    }

    public enum Failure: LocalizedError {
        case invalidInput, noLibrary, staleArticle, tooLarge
        public var errorDescription: String? {
            switch self {
            case .invalidInput: return "Ask a question of 500 characters or fewer."
            case .noLibrary: return "Open Zimfo and download or enable an offline archive first."
            case .staleArticle: return "That article is no longer available in the enabled library. Please choose it again."
            case .tooLarge: return "This article is too large for a quick answer. Open it in Zimfo to continue."
            }
        }
    }

    /// Infer a named subject without changing the question used for evidence.
    /// A pronoun without an explicit AppEntity never binds to old app chat.
    public static func topic(in question: String) -> String? {
        let pronouns: Set<String> = ["it", "this", "that", "these", "those", "he", "she", "they", "him", "her", "them", "its", "his", "their", "my", "our", "your", "i", "me", "we", "us", "you", "here", "there", "who", "someone", "somebody"]
        func namedSubject(_ value: String) -> String? {
            let title = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !title.isEmpty,
                  !pronouns.contains(title.lowercased().split(separator: " ").first.map(String.init) ?? "") else { return nil }
            return title
        }
        // The in-app router handles date openers but deliberately leaves many
        // "where" questions for location routing. Siri's encyclopedia action
        // needs both inverted and embedded birthplace/birthdate forms.
        let factoid = question.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!"))
            .replacingOccurrences(of: #"(?i)^(?:please[\s,]+)?(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?tell\s+me[\s,:]+"#,
                                  with: "", options: .regularExpression)
            .replacingOccurrences(of: #"(?i)^please[\s,]+"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let patterns = [
            #"(?i)^(?:where|when)\s+(?:was|were|is)\s+(.+?)\s+born(?:\s*,?\s*please)?$"#,
            #"(?i)^(?:where|when)\s+(.+?)\s+(?:was|were|is)\s+born(?:\s*,?\s*please)?$"#,
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern),
                  let match = regex.firstMatch(in: factoid, range: NSRange(factoid.startIndex..., in: factoid)),
                  let range = Range(match.range(at: 1), in: factoid) else { continue }
            let subject = String(factoid[range])
            // One returned article cannot stand for two requested people.
            guard subject.range(of: #"(?i)\s+(?:and|or)\s+"#, options: .regularExpression) == nil else { return nil }
            guard subject.range(of: #"(?i)['’]s\b|\b(?:mother|father|parents|brother|sister|son|daughter|wife|husband)\b"#,
                                options: .regularExpression) == nil else { return nil }
            return namedSubject(subject)
        }
        guard let intent = IntentRouter.classify(question, mode: .encyclopedia),
              ["article_overview", "discuss_article"].contains(intent.toolName),
              case .string(let title) = intent.args["title"] else { return nil }
        return namedSubject(title)
    }

    public static func confirmedQuestion(_ question: String, requestedTopic: String?, article: OfflineArticle) -> String {
        guard let requestedTopic, !requestedTopic.isEmpty else { return question }
        return question.replacingOccurrences(of: "(?i)\\b" + NSRegularExpression.escapedPattern(for: requestedTopic) + "\\b",
                                             with: NSRegularExpression.escapedTemplate(for: article.title), options: .regularExpression)
    }

    /// A single source document for an explicit onscreen-content request.
    /// Export all cleaned sections or refuse; never silently truncate facts.
    public static let maxDocumentBytes = 128 * 1024

    public func document(article: OfflineArticle) async throws -> Data {
        try Task.checkCancellation()
        guard !article.zim.isEmpty, article.zim.count <= 500, !article.path.isEmpty,
              article.path.count <= 2048, article.title.count <= 500 else { throw Failure.staleArticle }
        let raw = try await service.article(path: article.path, zim: article.zim)
        try Task.checkCancellation()
        guard raw.bytes <= Self.maxHTMLBytes else { throw Failure.tooLarge }
        let parsed = try await service.articleSections(path: article.path, zim: article.zim)
        try Task.checkCancellation()
        guard EntityResolutionPolicy.sameTitle(parsed.title, article.title),
              !ArticleHeuristics.isDisambiguationArticle(title: parsed.title, leadText: parsed.sections.first?.text ?? "")
        else { throw Failure.staleArticle }
        guard !parsed.sections.isEmpty else { throw Failure.staleArticle }
        guard parsed.sections.count <= 250 else { throw Failure.tooLarge }
        var data = Data("Downloaded article: \(article.title)\nArchive: \(article.zim)\nSource document, not a chat transcript. Information reflects the downloaded edition.\n".utf8)
        for section in parsed.sections {
            try Task.checkCancellation()
            let text = "\n\(section.title.isEmpty ? "Introduction" : section.title)\n\(section.text)\n"
            guard text.utf8.count <= Self.maxDocumentBytes - data.count else { throw Failure.tooLarge }
            data.append(contentsOf: text.utf8)
        }
        guard parsed.sections.contains(where: { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
        else { throw Failure.staleArticle }
        return data
    }

    public func resolve(topic: String) async throws -> Resolution {
        let topic = try Self.validatedInput(topic)
        try Task.checkCancellation()
        guard try await service.inventory().zims.contains(where: {
            [.wikipedia, .mdwiki, .streetzim].contains($0.kind)
        }) else { throw Failure.noLibrary }
        if let hit = try? await service.articleByTitle(title: topic, zim: nil, section: "lead"),
           !ArticleHeuristics.isDisambiguationArticle(title: hit.title, leadText: hit.section.text) {
            try Task.checkCancellation()
            return .article(.init(zim: hit.zim, path: hit.path, title: hit.title))
        }
        try Task.checkCancellation()
        let hits = try await service.search(query: topic, limit: 8, kind: .wikipedia)
        var seen = Set<String>()
        let candidates = hits.filter {
            !$0.title.lowercased().contains("(disambiguation)") && seen.insert($0.zim + "\n" + $0.path).inserted
        }.prefix(8)
        var choices: [OfflineArticle] = []
        for hit in candidates {
            try Task.checkCancellation()
            guard let lead = try? await service.articleSection(path: hit.path, section: "lead", zim: hit.zim),
                  !ArticleHeuristics.isDisambiguationArticle(title: lead.title, leadText: lead.section.text) else { continue }
            choices.append(.init(zim: hit.zim, path: hit.path, title: lead.title))
            if choices.count == 5 { break }
        }
        return .choices(choices)
    }

    public static func needsLiveData(_ question: String) -> Bool {
        question.range(of: #"(?i)\b(?:right now|currently|latest|today|tonight|live traffic|open now|current (?:hours|price|weather|population)|is .+ open)\b"#,
                       options: .regularExpression) != nil
    }

    public func answer(question: String, article: OfflineArticle,
                       diagnostics: (@Sendable (Measurement) async -> Void)? = nil) async throws -> String {
        let question = try Self.validatedInput(question)
        try Task.checkCancellation()
        if Self.needsLiveData(question) {
            return "Downloaded archives cannot verify live conditions or current facts. Ask for historical information, or check a current source."
        }
        // This bounds HTML parsing/scoring, not libzim decompression itself.
        let raw = try await service.article(path: article.path, zim: article.zim)
        await diagnostics?(.htmlBytes(raw.bytes))
        try Task.checkCancellation()
        guard raw.bytes <= Self.maxHTMLBytes else { throw Failure.tooLarge }
        let parsed = try await service.articleSections(path: article.path, zim: article.zim)
        try Task.checkCancellation()
        guard EntityResolutionPolicy.sameTitle(parsed.title, article.title),
              !ArticleHeuristics.isDisambiguationArticle(title: parsed.title, leadText: parsed.sections.first?.text ?? "")
        else { throw Failure.staleArticle }
        let parsedBytes = parsed.sections.reduce(0, { $0 + $1.text.utf8.count })
        await diagnostics?(.parsedBytes(parsedBytes))
        await diagnostics?(.sections(parsed.sections.count))
        try Task.checkCancellation()
        guard parsed.sections.count <= 250, parsedBytes <= 500_000 else { throw Failure.tooLarge }
        let passages = ArticleHeuristics.rankSectionsForQuestion(question, sections: parsed.sections, k: 4)
            .map { SourceBoundAnswer.Passage(article: article.title, section: $0.title, library: article.zim, text: $0.text) }
        let reply = SourceBoundAnswer.answer(question: question, topic: article.title,
                                             passages: passages, maxSentences: 2, maxCharacters: 600)
        guard reply.hasEvidence else { return SourceBoundAnswer.missingAnswer(topic: article.title) }
        await diagnostics?(.answerCharacters(reply.text.count))
        try Task.checkCancellation()
        // SourceBoundAnswer's first sentence may exceed its *soft* budget.
        // Refuse long speech rather than chopping off negation/qualifications.
        guard reply.text.count <= 900 else { throw Failure.tooLarge }
        return "From the downloaded article \(article.title): \(reply.text)"
    }
}
