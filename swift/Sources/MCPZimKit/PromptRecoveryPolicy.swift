// SPDX-License-Identifier: MIT

/// A recurrent checkpoint is usable only while its full attention prefix
/// still exists and matches the incoming prompt. Leave a token to evaluate
/// because sequence snapshots do not retain output logits.
public enum PromptRecoveryPolicy {
    public static func reusableTokenCount(
        prompt: [Int32], liveTokens: [Int32], checkpointTokens: [Int32]
    ) -> Int? {
        let count = checkpointTokens.count
        guard count > 0, count < prompt.count, count <= liveTokens.count,
              prompt.prefix(count).elementsEqual(checkpointTokens),
              liveTokens.prefix(count).elementsEqual(checkpointTokens)
        else { return nil }
        return count
    }
}
