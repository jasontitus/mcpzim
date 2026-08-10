import Foundation
import Testing
@testable import MLXLMCommon

private final class CountingTokenizer: Tokenizer, @unchecked Sendable {
    private(set) var largestDecode = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        largestDecode = max(largestDecode, tokenIds.count)
        return String(repeating: "a", count: tokenIds.count)
    }

    func convertTokenToId(_ token: String) -> Int? { 1 }
    func convertIdToToken(_ id: Int) -> String? { "a" }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("Streaming detokenizer")
struct StreamingDetokenizerTests {
    @Test("long streams retain output while bounding decode history")
    func boundsDecodeHistory() {
        let tokenizer = CountingTokenizer()
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var output = ""

        for _ in 0..<100 {
            detokenizer.append(token: 1)
            output += detokenizer.next() ?? ""
        }

        #expect(output == String(repeating: "a", count: 100))
        #expect(tokenizer.largestDecode <= 32)
    }
}
