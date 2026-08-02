// SPDX-License-Identifier: MIT
//
// Process-wide compiled-NSRegularExpression cache.
//
// The intent router, reference resolver, and drift extractors match a small
// closed set of literal patterns on every user turn (and per sentence inside
// article loops). Compiling an NSRegularExpression costs tens of µs — far
// more than the match itself for short inputs — so the per-call
// `try? NSRegularExpression(...)` idiom burned 1-3 ms of pure compilation per
// on-device turn. Patterns are literals, so the cache is bounded by the
// number of distinct call sites.

import Foundation

final class RegexCache: @unchecked Sendable {
    static let shared = RegexCache()

    private let lock = NSLock()
    private var cache: [Key: NSRegularExpression] = [:]

    private struct Key: Hashable {
        let pattern: String
        let options: NSRegularExpression.Options.RawValue
    }

    /// Returns the compiled regex for `pattern`, compiling once per distinct
    /// (pattern, options) and nil for invalid patterns (mirroring the
    /// `try?` behaviour at the call sites). Invalid patterns are not cached;
    /// they only occur on programmer error and recompiling a bad literal is
    /// the visible-in-testing behaviour we want.
    func compiled(
        _ pattern: String,
        options: NSRegularExpression.Options = []
    ) -> NSRegularExpression? {
        let key = Key(pattern: pattern, options: options.rawValue)
        lock.lock()
        if let hit = cache[key] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        guard let compiled = try? NSRegularExpression(pattern: pattern, options: options) else {
            return nil
        }
        lock.lock()
        cache[key] = compiled
        lock.unlock()
        return compiled
    }
}
