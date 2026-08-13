// SPDX-License-Identifier: MIT

import Foundation
import SwiftUI

/// Streaming-safe rich renderer for assistant prose. A half-finished Markdown
/// token simply falls back to partially parsed text and is re-rendered on the
/// next streamed chunk; no separate WebView or JavaScript renderer is needed.
struct MarkdownMessageText: View {
    let source: String

    private var blocks: [MarkdownMessageBlock] {
        MarkdownBlockCache.blocks(for: source)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownMessageBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            InlineMarkdownText(source: text)
                .font(headingFont(level))
                .padding(.top, level <= 2 ? 3 : 1)
        case .paragraph(let text):
            InlineMarkdownText(source: text)
        case .unorderedItem(let depth, let text):
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                Text("•")
                    .frame(width: 14, alignment: .trailing)
                InlineMarkdownText(source: text)
            }
            .padding(.leading, CGFloat(depth * 16))
            .accessibilityElement(children: .combine)
        case .orderedItem(let depth, let number, let text):
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                Text("\(number).")
                    .frame(minWidth: 18, alignment: .trailing)
                InlineMarkdownText(source: text)
            }
            .padding(.leading, CGFloat(depth * 16))
            .accessibilityElement(children: .combine)
        case .quote(let text):
            HStack(alignment: .top, spacing: 8) {
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(Color.secondary.opacity(0.55))
                    .frame(width: 3)
                InlineMarkdownText(source: text)
                    .italic()
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 2)
        case .code(let language, let text):
            VStack(alignment: .leading, spacing: 4) {
                if let language {
                    Text(language)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                ScrollView(.horizontal, showsIndicators: true) {
                    Text(text.isEmpty ? " " : text)
                        .font(.system(.callout, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                }
                .background(Color.primary.opacity(0.07))
                .clipShape(RoundedRectangle(cornerRadius: 7))
            }
        case .table(let rows):
            MarkdownMessageTable(rows: rows)
        case .divider:
            Divider().padding(.vertical, 2)
        }
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .title2.weight(.bold)
        case 2: return .title3.weight(.bold)
        default: return .headline
        }
    }
}

private struct InlineMarkdownText: View {
    let source: String

    var body: some View {
        Text(InlineMarkdownCache.attributed(for: source))
    }
}

/// PI review 2026-08-13 (perf #2): `blocks` is a computed property, so the
/// whole message was re-tokenized on every `body` evaluation — and SwiftUI
/// evaluates `body` at least once per 10 Hz streaming push, plus again for
/// every settled message that scrolls back into view. Key on the O(1)
/// UTF-8 length and confirm with `==`; hashing the source for a dictionary
/// key would itself be the full pass we are trying to avoid.
@MainActor
private enum MarkdownBlockCache {
    private static var entries:
        [(utf8Count: Int, source: String, blocks: [MarkdownMessageBlock])] = []
    private static let limit = 8

    static func blocks(for source: String) -> [MarkdownMessageBlock] {
        let utf8Count = source.utf8.count
        if let hit = entries.first(where: {
            $0.utf8Count == utf8Count && $0.source == source
        }) {
            return hit.blocks
        }
        let parsed = MarkdownMessageParser.parse(source)
        entries.append((utf8Count, source, parsed))
        if entries.count > limit {
            entries.removeFirst(entries.count - limit)
        }
        return parsed
    }
}

/// PI review 2026-08-13 (perf #2): `attributed` was a computed property, so
/// every `body` evaluation re-ran Foundation's Markdown parser over *every*
/// block of the message — at streaming cadence, on the main thread, and a
/// `MarkdownMessageTable` multiplies it by rows×columns. A streaming push
/// only ever changes the last block, so keying on the block's own text
/// turns a push into one parse instead of N: this is where "proportional
/// to what changed" actually lands for the render half of the finding.
@MainActor
private enum InlineMarkdownCache {
    private static var cache: [String: AttributedString] = [:]
    private static var order: [String] = []
    private static var cachedBytes = 0
    /// Generous enough that one body pass over a long reply (or a wide
    /// table) never evicts entries it is about to ask for again.
    private static let entryLimit = 512
    /// Bounding bytes as well as entries matters because a streaming reply
    /// inserts one dead partial-tail block per push, and that tail is the
    /// whole message when the model answers in a single long paragraph.
    private static let byteLimit = 512 * 1024

    static func attributed(for source: String) -> AttributedString {
        if let hit = cache[source] { return hit }
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible)
        let value = (try? AttributedString(markdown: source, options: options))
            ?? AttributedString(source)
        cache[source] = value
        order.append(source)
        cachedBytes += source.utf8.count
        var drop = 0
        while order.count - drop > 1,
              order.count - drop > entryLimit || cachedBytes > byteLimit
        {
            cachedBytes -= order[drop].utf8.count
            cache.removeValue(forKey: order[drop])
            drop += 1
        }
        if drop > 0 { order.removeFirst(drop) }
        return value
    }
}

private struct MarkdownMessageTable: View {
    let rows: [[String]]

    private var columnCount: Int {
        rows.map(\.count).max() ?? 0
    }

    var body: some View {
        if columnCount > 0 {
            ScrollView(.horizontal, showsIndicators: true) {
                Grid(
                    alignment: .leadingFirstTextBaseline,
                    horizontalSpacing: 14,
                    verticalSpacing: 6
                ) {
                    ForEach(rows.indices, id: \.self) { rowIndex in
                        GridRow {
                            ForEach(0..<columnCount, id: \.self) { columnIndex in
                                InlineMarkdownText(source: cell(
                                    row: rowIndex, column: columnIndex))
                                    .font(rowIndex == 0
                                        ? .caption.weight(.semibold)
                                        : .caption)
                            }
                        }
                        if rowIndex == 0, rows.count > 1 {
                            Divider().gridCellColumns(columnCount)
                        }
                    }
                }
                .padding(8)
            }
            .background(Color.primary.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 7))
        }
    }

    private func cell(row: Int, column: Int) -> String {
        guard rows.indices.contains(row), rows[row].indices.contains(column) else {
            return ""
        }
        return rows[row][column]
    }
}
