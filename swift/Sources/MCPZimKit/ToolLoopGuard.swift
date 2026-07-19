// SPDX-License-Identifier: MIT

import Foundation

/// Deterministic circuit breaker for text-model tool loops. Slow local models
/// can spend tens of seconds proposing each call, so redispatching an
/// identical call—or performing an unbounded series of slightly different
/// searches—is far more expensive than forcing one final evidence summary.
public struct ToolLoopGuard: Sendable {
    public enum Decision: Equatable, Sendable {
        case allow
        case stop(String)
    }

    private var seen = Set<String>()
    private var counts: [String: Int] = [:]
    private let perToolLimits: [String: Int]

    public init(perToolLimits: [String: Int] = ["search": 2]) {
        self.perToolLimits = perToolLimits
    }

    /// `canonicalArguments` should use stable key ordering. Exact duplicate
    /// suppression applies to every tool; `perToolLimits` additionally caps
    /// exploratory families such as search even when the query wording drifts.
    public mutating func evaluate(
        toolName: String, canonicalArguments: String
    ) -> Decision {
        let key = toolName + "\n" + canonicalArguments
        guard seen.insert(key).inserted else {
            return .stop("duplicate \(toolName) call suppressed")
        }

        let next = counts[toolName, default: 0] + 1
        counts[toolName] = next
        if let limit = perToolLimits[toolName], next > limit {
            return .stop("\(toolName) call budget exhausted (\(limit) per turn)")
        }
        return .allow
    }
}
