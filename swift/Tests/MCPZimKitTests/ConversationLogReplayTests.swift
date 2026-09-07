import XCTest
@testable import MCPZimKit

/// Opt-in replay of local diagnostic fixtures. No logs or personal locations
/// are bundled with the app. The report compares routing and excerpt selection;
/// it does not simulate UI, ASR, model generation or missing archive passages.
final class ConversationLogReplayTests: XCTestCase {
    func testReplayLocalConversationEvidence() throws {
        guard let input = ProcessInfo.processInfo.environment["ZIMFO_REPLAY_INPUT"],
              let output = ProcessInfo.processInfo.environment["ZIMFO_REPLAY_OUTPUT"] else {
            throw XCTSkip("Set ZIMFO_REPLAY_INPUT and ZIMFO_REPLAY_OUTPUT for diagnostic replay")
        }
        let rows = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: input))) as! [[String: Any]]
        var report: [[String: Any]] = []
        for row in rows {
            let question = row["question"] as! String
            var focus = ConversationFocus()
            if let topic = row["priorTopic"] as? String {
                focus.remember(FocusEntity(name: topic, kind: .topic))
            }
            let location: (lat: Double, lon: Double)? = (row["lat"] as? Double).flatMap { lat in
                (row["lon"] as? Double).map { (lat, $0) }
            }
            let intent = IntentRouter.classify(question, currentLocation: location, focus: focus)
            var result: [String: Any] = ["id": row["id"]!, "question": question,
                                        "tool": intent?.toolName ?? "none", "args": intent?.anyArgs ?? [:]]
            if let topic = row["topic"] as? String,
               let text = row["sourceExcerpt"] as? String {
                let reply = SourceBoundAnswer.answer(question: question, topic: topic,
                    passages: [.init(article: topic, section: row["section"] as? String ?? "lead", text: text)],
                    sectionOverview: row["sectionChoice"] as? Bool ?? false)
                result["evidence"] = reply.hasEvidence
                result["answer"] = reply.text
            }
            if let topic = (row["topic"] ?? row["priorTopic"]) as? String {
                let pool = rows.filter { ($0["topic"] as? String) == topic }.compactMap { other -> SourceBoundAnswer.Passage? in
                    guard let text = other["sourceExcerpt"] as? String else { return nil }
                    return .init(article: topic, section: other["section"] as? String ?? "lead", text: text)
                }
                if !pool.isEmpty {
                    let reply = SourceBoundAnswer.answer(question: question, topic: topic, passages: pool,
                        sectionOverview: row["sectionChoice"] as? Bool ?? false)
                    result["pooledEvidence"] = reply.hasEvidence
                    result["pooledAnswer"] = reply.text
                }
            }
            report.append(result)
        }
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: output), options: .atomic)
        XCTAssertFalse(report.isEmpty)
    }
}
