// SPDX-License-Identifier: MIT

import Foundation
import SwiftUI

/// Streaming-safe rich renderer for assistant prose. A half-finished Markdown
/// token simply falls back to partially parsed text and is re-rendered on the
/// next streamed chunk; no separate WebView or JavaScript renderer is needed.
struct MarkdownMessageText: View {
    let source: String

    private var blocks: [MarkdownMessageBlock] {
        MarkdownMessageParser.parse(source)
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

    private var attributed: AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible)
        return (try? AttributedString(markdown: source, options: options))
            ?? AttributedString(source)
    }

    var body: some View {
        Text(attributed)
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
