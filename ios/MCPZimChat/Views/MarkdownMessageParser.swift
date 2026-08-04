// SPDX-License-Identifier: MIT

import Foundation

/// Block structure used by the chat renderer. Foundation's Markdown-backed
/// `AttributedString` handles inline presentation well, but flattens block
/// structure (headings, lists, tables, and code fences) into one run of text.
/// Keeping that small amount of structure here lets SwiftUI lay it out like a
/// real answer while still using Apple's parser for links and emphasis.
enum MarkdownMessageBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case unorderedItem(depth: Int, text: String)
    case orderedItem(depth: Int, number: String, text: String)
    case quote(String)
    case code(language: String?, text: String)
    case table([[String]])
    case divider
}

enum MarkdownMessageParser {
    static func parse(_ source: String) -> [MarkdownMessageBlock] {
        let normalized = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.components(separatedBy: "\n")
        var blocks: [MarkdownMessageBlock] = []
        var paragraph: [String] = []
        var index = 0

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            blocks.append(.paragraph(paragraph.joined(separator: "\n")))
            paragraph.removeAll(keepingCapacity: true)
        }

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                flushParagraph()
                index += 1
                continue
            }

            if let fence = fenceStart(in: trimmed) {
                flushParagraph()
                var codeLines: [String] = []
                index += 1
                while index < lines.count {
                    let candidate = lines[index]
                        .trimmingCharacters(in: .whitespaces)
                    if candidate.hasPrefix(fence.marker) {
                        index += 1
                        break
                    }
                    codeLines.append(lines[index])
                    index += 1
                }
                blocks.append(.code(
                    language: fence.language,
                    text: codeLines.joined(separator: "\n")))
                continue
            }

            if index + 1 < lines.count,
               line.contains("|"),
               isTableSeparator(lines[index + 1])
            {
                flushParagraph()
                var rows = [tableCells(line)]
                index += 2 // Skip the Markdown alignment/separator row.
                while index < lines.count {
                    let row = lines[index]
                    guard !row.trimmingCharacters(in: .whitespaces).isEmpty,
                          row.contains("|")
                    else { break }
                    rows.append(tableCells(row))
                    index += 1
                }
                blocks.append(.table(rows))
                continue
            }

            if let heading = heading(in: trimmed) {
                flushParagraph()
                blocks.append(.heading(level: heading.level, text: heading.text))
                index += 1
                continue
            }

            if let item = unorderedItem(in: line) {
                flushParagraph()
                blocks.append(.unorderedItem(depth: item.depth, text: item.text))
                index += 1
                continue
            }

            if let item = orderedItem(in: line) {
                flushParagraph()
                blocks.append(.orderedItem(
                    depth: item.depth, number: item.number, text: item.text))
                index += 1
                continue
            }

            if trimmed.hasPrefix(">") {
                flushParagraph()
                let quote = trimmed.dropFirst()
                    .trimmingCharacters(in: .whitespaces)
                blocks.append(.quote(quote))
                index += 1
                continue
            }

            if isDivider(trimmed) {
                flushParagraph()
                blocks.append(.divider)
                index += 1
                continue
            }

            paragraph.append(line)
            index += 1
        }

        flushParagraph()
        return blocks
    }

    private static func fenceStart(
        in line: String
    ) -> (marker: String, language: String?)? {
        let marker: String
        if line.hasPrefix("```") {
            marker = "```"
        } else if line.hasPrefix("~~~") {
            marker = "~~~"
        } else {
            return nil
        }
        let language = line.dropFirst(marker.count)
            .trimmingCharacters(in: .whitespaces)
        return (marker, language.isEmpty ? nil : language)
    }

    private static func heading(in line: String) -> (level: Int, text: String)? {
        let level = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(level), line.count > level else { return nil }
        let rest = line.dropFirst(level)
        guard rest.first?.isWhitespace == true else { return nil }
        return (level, rest.trimmingCharacters(in: .whitespaces))
    }

    private static func indentedContent(_ line: String) -> (depth: Int, text: String) {
        var consumed = 0
        var spaces = 0
        for character in line {
            if character == " " {
                spaces += 1
            } else if character == "\t" {
                spaces += 2
            } else {
                break
            }
            consumed += 1
        }
        return (min(spaces / 2, 4), String(line.dropFirst(consumed)))
    }

    private static func unorderedItem(in line: String) -> (depth: Int, text: String)? {
        let indented = indentedContent(line)
        guard indented.text.count >= 2 else { return nil }
        let marker = indented.text.first
        guard marker == "-" || marker == "*" || marker == "+" else { return nil }
        let afterMarker = indented.text.dropFirst()
        guard afterMarker.first?.isWhitespace == true else { return nil }
        return (
            indented.depth,
            afterMarker.trimmingCharacters(in: .whitespaces))
    }

    private static func orderedItem(
        in line: String
    ) -> (depth: Int, number: String, text: String)? {
        let indented = indentedContent(line)
        let characters = Array(indented.text)
        let digitCount = characters.prefix(while: { $0.isNumber }).count
        guard digitCount > 0,
              characters.count > digitCount + 1,
              characters[digitCount] == "." || characters[digitCount] == ")",
              characters[digitCount + 1].isWhitespace
        else { return nil }
        return (
            indented.depth,
            String(characters.prefix(digitCount)),
            String(characters.dropFirst(digitCount + 2))
                .trimmingCharacters(in: .whitespaces))
    }

    private static func isDivider(_ line: String) -> Bool {
        let compact = line.filter { !$0.isWhitespace }
        guard compact.count >= 3, let marker = compact.first,
              marker == "-" || marker == "*" || marker == "_"
        else { return false }
        return compact.allSatisfy { $0 == marker }
    }

    private static func tableCells(_ line: String) -> [String] {
        var body = line.trimmingCharacters(in: .whitespaces)
        if body.hasPrefix("|") { body.removeFirst() }
        if body.hasSuffix("|") { body.removeLast() }
        return body.split(separator: "|", omittingEmptySubsequences: false).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let cells = tableCells(line)
        guard !cells.isEmpty else { return false }
        return cells.allSatisfy { cell in
            let body = cell.filter { $0 != ":" && !$0.isWhitespace }
            // Models occasionally emit two-dash table delimiters even though
            // CommonMark examples normally use three. Accept both so a
            // nearly-correct streamed table is still presented as a table.
            return body.count >= 2 && body.allSatisfy { $0 == "-" }
        }
    }
}
