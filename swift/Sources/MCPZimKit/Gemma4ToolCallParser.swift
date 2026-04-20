// SPDX-License-Identifier: MIT
//
// Parses Gemma 4's native tool-call format:
//
//   <|tool_call>call:FUNCTION_NAME{k1:v1,k2:v2,…}<tool_call|>
//
// Values use the same mini-format `Gemma4ToolFormat` emits:
//   - string:  `<|"|>VALUE<|"|>`
//   - number:  decimal literal
//   - bool:    `true` / `false`
//   - null:    `null`
//   - array:   `[v1,v2,…]`
//   - object:  `{k:v,…}`
//
// Returned `arguments` is `[String: Any]` with Swift-native types
// (`String`, `Int` or `Double`, `Bool`, `NSNull`, `[Any]`, `[String: Any]`)
// so downstream code — especially `MCPToolAdapter.dispatch` — can feed it
// straight through `JSONSerialization.data(withJSONObject:)`.

import Foundation

public enum Gemma4ToolCallParser {
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

    public static func firstCall(in buffer: String) -> Match? {
        guard let openMarker = buffer.range(of: "<|tool_call>"),
              let closeMarker = buffer.range(of: "<tool_call|>", range: openMarker.upperBound..<buffer.endIndex)
        else { return nil }
        let body = String(buffer[openMarker.upperBound..<closeMarker.lowerBound])
        guard let parsed = parseCallBody(body) else { return nil }
        return Match(range: openMarker.lowerBound..<closeMarker.upperBound,
                     name: parsed.name,
                     arguments: parsed.arguments)
    }

    // MARK: - Body parsing

    private static func parseCallBody(_ body: String) -> (name: String, arguments: [String: Any])? {
        // Canonical form is `call:NAME{…args…}`, but small models occasionally
        // echo the *declaration* shape (`call:NAME({…}) -> OBJECT` or even
        // `call:NAME(args)`). Normalise before handing off to the object
        // parser: strip trailing `-> TYPE`, unwrap an outer `(…)` if one is
        // present so we're always looking at `NAME{…}`.
        guard body.hasPrefix("call:") else { return nil }
        let afterCall = body.index(body.startIndex, offsetBy: 5)
        var trimmed = String(body[afterCall...]).trimmingCharacters(in: .whitespaces)

        // Drop `-> RETURN_TYPE` if the model tacked one on.
        if let arrow = trimmed.range(of: " -> ") {
            trimmed = String(trimmed[..<arrow.lowerBound]).trimmingCharacters(in: .whitespaces)
        }

        guard let firstBracket = trimmed.firstIndex(where: { $0 == "{" || $0 == "(" }) else {
            return nil
        }
        let name = String(trimmed[trimmed.startIndex..<firstBracket])
            .trimmingCharacters(in: .whitespaces)
        let opener = trimmed[firstBracket]

        // Body between outermost braces/parens. Peel one layer of `(…)` if
        // present so we're left with `{…}` for the arg parser.
        let afterOpener = trimmed.index(after: firstBracket)
        var argsSlice = trimmed[afterOpener..<trimmed.endIndex]
        let expectedClose: Character = opener == "(" ? ")" : "}"
        if argsSlice.last == expectedClose {
            argsSlice = argsSlice.dropLast()
        }
        var argsText = String(argsSlice)
        if opener == "(" {
            // Expect the inner to be `{…}` — unwrap once more.
            let inner = argsText.trimmingCharacters(in: .whitespaces)
            if inner.hasPrefix("{") && inner.hasSuffix("}") {
                argsText = String(inner.dropFirst().dropLast())
            } else {
                argsText = inner
            }
        }

        guard let dict = parseObjectBody(argsText) else { return nil }
        return (name: name, arguments: dict)
    }

    // MARK: - Value parser (recursive-descent over Gemma mini-format)

    /// Parses an `{k:v,k:v,…}` object body (no surrounding braces). Returns
    /// nil on any malformed segment rather than throwing.
    private static func parseObjectBody(_ text: String) -> [String: Any]? {
        var scanner = Scanner(source: text)
        scanner.skipWhitespace()
        if scanner.isAtEnd { return [:] }
        var out: [String: Any] = [:]
        while !scanner.isAtEnd {
            scanner.skipWhitespace()
            guard let key = scanner.takeKey() else { return nil }
            guard scanner.take(":") else { return nil }
            guard let value = scanner.takeValue() else { return nil }
            out[key] = value
            scanner.skipWhitespace()
            if scanner.isAtEnd { break }
            guard scanner.take(",") else { return nil }
        }
        return out
    }

    private struct Scanner {
        var source: String
        var i: String.Index
        init(source: String) {
            self.source = source
            self.i = source.startIndex
        }
        var isAtEnd: Bool { i >= source.endIndex }
        mutating func skipWhitespace() {
            while i < source.endIndex, source[i].isWhitespace { i = source.index(after: i) }
        }
        mutating func take(_ literal: Character) -> Bool {
            guard i < source.endIndex, source[i] == literal else { return false }
            i = source.index(after: i)
            return true
        }
        mutating func takeLiteral(_ literal: String) -> Bool {
            guard source.distance(from: i, to: source.endIndex) >= literal.count else { return false }
            let end = source.index(i, offsetBy: literal.count)
            if source[i..<end] == Substring(literal) {
                i = end
                return true
            }
            return false
        }
        mutating func takeKey() -> String? {
            // Keys are unquoted identifiers up to `:`.
            let start = i
            while i < source.endIndex, source[i] != ":" , source[i] != "," , source[i] != "}" {
                i = source.index(after: i)
            }
            let key = String(source[start..<i]).trimmingCharacters(in: .whitespaces)
            return key.isEmpty ? nil : key
        }
        mutating func takeValue() -> Any? {
            skipWhitespace()
            if takeLiteral("<|\"|>") { return takeQuotedString() }
            if takeLiteral("true") { return true }
            if takeLiteral("false") { return false }
            if takeLiteral("null") { return NSNull() }
            if take("[") { return takeArray() }
            if take("{") { return takeObject() }
            return takeNumber()
        }
        mutating func takeQuotedString() -> String? {
            // Read until the closing `<|"|>`.
            let start = i
            while i < source.endIndex {
                if source[i] == "<" {
                    let tail = source[i..<source.endIndex]
                    if tail.hasPrefix("<|\"|>") {
                        let str = String(source[start..<i])
                        i = source.index(i, offsetBy: 5)
                        return str
                    }
                }
                i = source.index(after: i)
            }
            return nil
        }
        mutating func takeNumber() -> Any? {
            let start = i
            if i < source.endIndex, source[i] == "-" { i = source.index(after: i) }
            while i < source.endIndex, source[i].isNumber || source[i] == "." {
                i = source.index(after: i)
            }
            let slice = String(source[start..<i])
            if slice.isEmpty { return nil }
            if slice.contains(".") { return Double(slice) }
            return Int(slice) ?? Double(slice)
        }
        mutating func takeArray() -> Any? {
            var out: [Any] = []
            skipWhitespace()
            if take("]") { return out }
            while true {
                skipWhitespace()
                guard let v = takeValue() else { return nil }
                out.append(v)
                skipWhitespace()
                if take("]") { return out }
                guard take(",") else { return nil }
            }
        }
        mutating func takeObject() -> Any? {
            var out: [String: Any] = [:]
            skipWhitespace()
            if take("}") { return out }
            while true {
                skipWhitespace()
                guard let key = takeKey() else { return nil }
                guard take(":") else { return nil }
                guard let value = takeValue() else { return nil }
                out[key] = value
                skipWhitespace()
                if take("}") { return out }
                guard take(",") else { return nil }
            }
        }
    }
}
