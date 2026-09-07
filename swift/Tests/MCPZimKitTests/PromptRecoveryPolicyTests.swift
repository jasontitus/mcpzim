import XCTest
@testable import MCPZimKit

final class PromptRecoveryPolicyTests: XCTestCase {
    func testRetryAndEditedAnswerRetainPromptBoundary() {
        let checkpoint: [Int32] = [1, 2, 3]
        for prompt: [Int32] in [[1, 2, 3, 4], [1, 2, 3, 4, 99, 8]] {
            XCTAssertEqual(PromptRecoveryPolicy.reusableTokenCount(
                prompt: prompt, liveTokens: [1, 2, 3, 4, 5, 6],
                checkpointTokens: checkpoint), 3)
        }
    }

    func testRejectsChangedEvidenceOrMissingAttentionState() {
        let cases: [([Int32], [Int32], [Int32])] = [
            ([1, 9, 3, 4], [1, 2, 3, 4], [1, 2, 3]), // changed evidence
            ([1, 2, 3, 4], [], [1, 2, 3]),             // reset or failed prefill
            ([1, 2, 3, 4], [1, 2], [1, 2, 3]),       // incomplete attention state
            ([1, 2, 3, 4], [1, 9, 3, 4], [1, 2, 3]), // different live prefix
            ([1, 2, 3], [1, 2, 3, 4], [1, 2, 3]),    // no fresh logits
            ([1, 2], [1, 2, 3, 4], [1, 2, 3]),       // shortened prompt
            ([1, 2, 3, 4], [1, 2, 3, 4], []),        // no checkpoint
        ]
        for (prompt, live, checkpoint) in cases {
            XCTAssertNil(PromptRecoveryPolicy.reusableTokenCount(
                prompt: prompt, liveTokens: live, checkpointTokens: checkpoint))
        }
    }
}
