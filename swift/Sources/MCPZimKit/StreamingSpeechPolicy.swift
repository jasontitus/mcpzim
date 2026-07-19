// SPDX-License-Identifier: MIT
//
// Pure policy for deciding when a growing assistant reply has enough stable
// prose to hand to streaming TTS. Kept in MCPZimKit so sentence/clause
// behavior and memory gating can be tested without AVFoundation or MLX.

import Foundation

public struct StreamingSpeechPrefix: Equatable, Sendable {
    public enum Boundary: Equatable, Sendable {
        case sentence
        case clause
        case softWrap
        case final
    }

    public let text: String
    public let consumedCharacters: Int
    public let boundary: Boundary

    public init(text: String, consumedCharacters: Int, boundary: Boundary) {
        self.text = text
        self.consumedCharacters = consumedCharacters
        self.boundary = boundary
    }
}

public enum StreamingSpeechPolicy {
    /// Return the earliest stable prefix that is useful to synthesize.
    ///
    /// While generation is active, complete sentences always win. With
    /// `allowEarlyClause`, a natural clause boundary after `minimumClause`
    /// characters can start speech sooner; prose without punctuation is
    /// softly wrapped at a word boundary once it reaches `maximumClause`.
    /// When generation is complete, the entire tail is returned.
    public static func takeSpeakablePrefix(
        _ text: String,
        generating: Bool,
        allowEarlyClause: Bool,
        minimumClause: Int = 56,
        maximumClause: Int = 112
    ) -> StreamingSpeechPrefix? {
        guard !text.isEmpty else { return nil }
        let chars = Array(text)
        let minimum = max(24, minimumClause)
        let maximum = max(minimum + 16, maximumClause)

        // A paragraph break is a semantic/prosody boundary, even when the
        // completed tail would otherwise fit in one backend window. In chat,
        // the deterministic "Want to hear about …?" offer is appended as a
        // new paragraph. Folding it into the answer's last Kokoro synthesis
        // window can smear the first word across the prior sentence. Consume
        // the two newlines but keep them out of the spoken prefix so the next
        // call begins cleanly at the next paragraph.
        if chars.count >= 2 {
            for i in 0..<(chars.count - 1)
            where chars[i] == "\n" && chars[i + 1] == "\n" {
                let prefix = String(chars[0..<i])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !prefix.isEmpty else { continue }
                return StreamingSpeechPrefix(
                    text: prefix,
                    consumedCharacters: i + 2,
                    boundary: .sentence)
            }
        }

        if !generating, !allowEarlyClause {
            return StreamingSpeechPrefix(
                text: text, consumedCharacters: text.count, boundary: .final)
        }

        // A completed short tail is already stable and should be spoken as-is.
        // Longer tails still need bounded draining for backends with small text
        // windows; otherwise their private chunkers create audible seams.
        if !generating, chars.count <= maximum {
            return StreamingSpeechPrefix(
                text: text, consumedCharacters: text.count, boundary: .final)
        }

        // Use the first complete sentence, not the last one in the buffer:
        // smaller first synthesis work gets audio onto the player sooner. Do
        // not cross the requested backend cap even when the sentence is long.
        for i in chars.indices where ".!?".contains(chars[i]) {
            // During a stream, EOF is not proof of a boundary: "3." may be
            // followed by "14" in the next token. Wait for whitespace.
            guard i + 1 < chars.count, chars[i + 1].isWhitespace else { continue }
            if chars[i] == ".", isLikelyAbbreviation(chars, endingAt: i) {
                continue
            }
            let consumed = i + 1
            if consumed <= maximum {
                return StreamingSpeechPrefix(
                    text: String(chars[0..<consumed]),
                    consumedCharacters: consumed,
                    boundary: .sentence)
            }
            break
        }

        guard allowEarlyClause else { return nil }

        // Commas and semicolons give Kokoro a natural continuation cadence.
        // Require following whitespace so thousands separators and times do
        // not become false boundaries.
        if chars.count > minimum {
            let upperBound = min(chars.count, maximum)
            for i in minimum..<upperBound {
                let c = chars[i]
                let isClausePunctuation = c == "," || c == ";" || c == ":"
                    || c == "—" || c == "–" || c == "\n"
                guard isClausePunctuation else { continue }
                let followedByWhitespace = c == "\n"
                    || (i + 1 < chars.count && chars[i + 1].isWhitespace)
                guard followedByWhitespace else { continue }
                let consumed = i + 1
                return StreamingSpeechPrefix(
                    text: String(chars[0..<consumed]),
                    consumedCharacters: consumed,
                    boundary: .clause)
            }
        }

        // Do not wait indefinitely for punctuation in a very long sentence.
        // Snap to the last whitespace inside the cap; the caller can append
        // continuation punctuation for more natural prosody without changing
        // how many source characters were consumed.
        guard chars.count >= maximum else { return nil }
        let cap = min(maximum, chars.count)
        var split: Int?
        if cap > minimum {
            for i in stride(from: cap - 1, through: minimum, by: -1)
            where chars[i].isWhitespace {
                split = i + 1
                break
            }
        }
        // A URL or another long token may contain no whitespace inside the
        // cap. A hard bound is still preferable to opaque backend chunking.
        let consumed = split ?? cap
        return StreamingSpeechPrefix(
            text: String(chars[0..<consumed]),
            consumedCharacters: consumed,
            boundary: .softWrap)
    }

    /// Whether it is safe to overlap TTS synthesis with model generation.
    /// A zero/negative available-memory value means the platform cannot
    /// provide the metric, so retain eager behavior rather than disabling it.
    public static func allowsEagerSynthesis(
        availableMemoryMB: Double,
        estimatedTTSMemoryMB: Int,
        minimumHeadroomMB: Double = 700
    ) -> Bool {
        guard availableMemoryMB > 0 else { return true }
        let required = max(
            minimumHeadroomMB,
            Double(max(0, estimatedTTSMemoryMB)) + 384)
        return availableMemoryMB >= required
    }

    /// Avoid handing TTS tiny false sentences such as "Dr." or the first
    /// initial in "V. Putin". This intentionally favors a short allowlist and
    /// single-letter initials; ambiguous prose can wait for a clause boundary.
    private static func isLikelyAbbreviation(
        _ chars: [Character],
        endingAt periodIndex: Int
    ) -> Bool {
        var start = periodIndex
        while start > 0, !chars[start - 1].isWhitespace {
            start -= 1
        }
        let token = String(chars[start...periodIndex]).lowercased()
        let common: Set<String> = [
            "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
            "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.",
            "a.m.", "p.m."
        ]
        if common.contains(token) { return true }
        return token.count == 2 && token.first?.isLetter == true
    }
}
