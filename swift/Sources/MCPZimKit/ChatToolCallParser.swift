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

        // Strict path: balanced-brace walker + strict JSON parse.
        if let jsonRange = jsonObjectRange(in: buffer, from: cur),
           let parsed = parseObject(String(buffer[jsonRange])),
           let name = parsed["name"] as? String {
            let args = (parsed["arguments"] as? [String: Any]) ?? [:]
            var endIdx = jsonRange.upperBound
            while endIdx < buffer.endIndex, buffer[endIdx].isWhitespace {
                endIdx = buffer.index(after: endIdx)
            }
            // Order matters: longer sentinels first so `</tool_call>`
            // wins over `>`.
            var closerFound = false
            for closer in ["</tool_call>", "<tool_call|>", ">"] {
                if buffer[endIdx...].hasPrefix(closer) {
                    endIdx = buffer.index(endIdx, offsetBy: closer.count)
                    closerFound = true
                    break
                }
            }
            // A *complete* call must carry its closing marker. A balanced
            // JSON whose closer hasn't streamed in yet is still partial —
            // return nil so the streaming loop keeps collecting rather than
            // dispatching mid-stream (contract documented above).
            guard closerFound else { return nil }
            return Match(range: openRange.lowerBound..<endIdx, name: name, arguments: args)
        }

        // Repair fallback: FT'd Gemma 3 4B (gists 80daf913, 3f39d873)
        // emits JSON with `=` in place of `:` between key and value
        // object, e.g. `{"arguments={...}, "name": "X"}`. The brace
        // walker can't trust those braces (the inner `{` is inside an
        // unterminated key string), so bracket the JSON region by the
        // matching closer tag instead, then run a token-level repair.
        let closers: [String]
        switch opener {
        case "<tool_call>":   closers = ["</tool_call>"]
        case "<|tool_call>":  closers = ["<tool_call|>", "</tool_call>"]
        case "<|tool_call|":  closers = [">"]
        default:              closers = ["</tool_call>", "<tool_call|>", ">"]
        }
        for closer in closers {
            guard let cRange = buffer.range(of: closer, range: cur..<buffer.endIndex)
            else { continue }
            // Trim whitespace before the closer to keep the JSON tight.
            var jsonEnd = cRange.lowerBound
            while jsonEnd > cur {
                let prev = buffer.index(before: jsonEnd)
                if buffer[prev].isWhitespace {
                    jsonEnd = prev
                } else {
                    break
                }
            }
            let raw = String(buffer[cur..<jsonEnd])
            let repaired = repairKeyValueGarble(raw)
            if repaired == raw { continue }
            guard let parsed = parseObject(repaired),
                  let name = parsed["name"] as? String
            else { continue }
            let args = (parsed["arguments"] as? [String: Any]) ?? [:]
            return Match(
                range: openRange.lowerBound..<cRange.upperBound,
                name: name, arguments: args)
        }
        return nil
    }

    /// `replacingOccurrences(of:options:.regularExpression)` wrapper that
    /// rewrites `{"<word>=` and `,"<word>=` into `{"<word>":` and
    /// `,"<word>":`. Targets the FT'd Gemma 3 bug where the model emits
    /// `=` instead of `":` between a JSON key and a value object.
    /// The `[{,]` anchor keeps the regex from corrupting `=` characters
    /// that legitimately appear inside string values.
    private static func repairKeyValueGarble(_ source: String) -> String {
        return source.replacingOccurrences(
            of: #"([{,]\s*)"(\w+)="#,
            with: #"$1"$2":"#,
            options: .regularExpression
        )
    }

    private static func parseObject(_ json: String) -> [String: Any]? {
        guard let data = json.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
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
