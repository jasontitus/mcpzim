// SPDX-License-Identifier: MIT
//
// Parses `<tool_call>{...json...}</tool_call>` blocks (and a handful of
// provider-specific variants) emitted by an on-device LLM during
// streaming generation. Pulled out of `ios/.../ChatSession.swift` so
// that non-iOS hosts (a macOS CLI, Android fork, tests) can reuse the
// same grammar, and so it can be exercised by `swift test` without
// standing up an iOS test target.

import Foundation

public enum ChatToolCallParser {
    /// A successful parse. `range` covers the full wrapper + payload
    /// substring in the original buffer — callers use it to truncate the
    /// block from the user-visible transcript before appending the
    /// synthetic tool-response turn.
    public struct Match {
        public let range: Range<String.Index>
        public let name: String
        public let arguments: [String: Any]

        public init(range: Range<String.Index>, name: String, arguments: [String: Any]) {
            self.range = range
            self.name = name
            self.arguments = arguments
        }
    }

    /// Returns the first *complete* tool call in `buffer`, or `nil` if
    /// none is present yet. A half-received call (open tag but no
    /// balanced JSON, or JSON missing the `name` field) yields `nil` so
    /// the streaming loop keeps collecting tokens rather than
    /// dispatching a broken call.
    ///
    /// Accepted wrappers — all must carry a JSON object like
    /// `{"name":"T","arguments":{…}}`:
    ///   * `<tool_call>{…}</tool_call>`  — canonical generic form.
    ///   * `<|tool_call|{…}>`             — what Apple Foundation Models
    ///                                       emits in practice.
    ///   * `<|tool_call>{…}<tool_call|>`  — Gemma-style brackets around
    ///                                       a JSON payload (Gemma's
    ///                                       native DSL is handled by
    ///                                       Gemma4ToolCallParser
    ///                                       separately).
    public static func firstCall(in buffer: String) -> Match? {
        let openers = ["<tool_call>", "<|tool_call|", "<|tool_call>"]
        var best: Match?
        for opener in openers {
            if let m = findCall(in: buffer, afterOpener: opener) {
                if best == nil || m.range.lowerBound < best!.range.lowerBound {
                    best = m
                }
            }
        }
        return best
    }

    // MARK: - Internals

    private static func findCall(in buffer: String, afterOpener opener: String) -> Match? {
        guard let openRange = buffer.range(of: opener) else { return nil }
        // Find the first `{` at or after the opener's end — skip any
        // whitespace the model may have added.
        var cur = openRange.upperBound
        while cur < buffer.endIndex, buffer[cur].isWhitespace {
            cur = buffer.index(after: cur)
        }
        guard cur < buffer.endIndex, buffer[cur] == "{" else { return nil }
        guard let jsonRange = jsonObjectRange(in: buffer, from: cur) else { return nil }
        let json = String(buffer[jsonRange])
        guard let data = json.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = parsed["name"] as? String
        else { return nil }
        let args = (parsed["arguments"] as? [String: Any]) ?? [:]
        // Extend the span to include the closing sentinel on the same
        // line (if any) so callers strip the whole wrapper from the
        // visible text. Order matters: longer sentinels first so
        // `</tool_call>` wins over `>`.
        var endIdx = jsonRange.upperBound
        while endIdx < buffer.endIndex, buffer[endIdx].isWhitespace {
            endIdx = buffer.index(after: endIdx)
        }
        for closer in ["</tool_call>", "<tool_call|>", ">"] {
            if buffer[endIdx...].hasPrefix(closer) {
                endIdx = buffer.index(endIdx, offsetBy: closer.count)
                break
            }
        }
        return Match(range: openRange.lowerBound..<endIdx, name: name, arguments: args)
    }

    /// String-aware balanced-brace scan. `start` must point at `{`.
    /// Returns the range covering that object (inclusive of the closing
    /// `}`), or nil if unmatched.
    private static func jsonObjectRange(in buffer: String, from start: String.Index) -> Range<String.Index>? {
        var depth = 0
        var inString = false
        var escaping = false
        var idx = start
        while idx < buffer.endIndex {
            let ch = buffer[idx]
            if escaping {
                escaping = false
            } else if inString {
                if ch == "\\" {
                    escaping = true
                } else if ch == "\"" {
                    inString = false
                }
            } else {
                if ch == "\"" {
                    inString = true
                } else if ch == "{" {
                    depth += 1
                } else if ch == "}" {
                    depth -= 1
                    if depth == 0 {
                        return start..<buffer.index(after: idx)
                    }
                }
            }
            idx = buffer.index(after: idx)
        }
        return nil
    }
}
