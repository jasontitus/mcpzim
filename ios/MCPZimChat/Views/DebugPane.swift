// SPDX-License-Identifier: MIT
//
// In-window step-by-step log of what's happening under the hood: LLM load
// progress, per-iteration `generate` timings, tool-call dispatch, streaming
// progress, errors. Session appends entries via `session.debug(…)`; this
// view just renders them.

import SwiftUI
#if canImport(AppKit)
import AppKit
#endif
#if canImport(UIKit)
import UIKit
#endif

struct DebugPaneView: View {
    @Environment(ChatSession.self) private var session
    @State private var justCopied = false
    @State private var lastReportHash: String? = nil

    private var dateFormatter: DateFormatter {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss.SSS"
        return df
    }

    var body: some View {
        // When the master toggle is off we want the chat to reach all
        // the way to the bottom — collapse to EmptyView rather than
        // keeping the header strip visible.
        if !session.showDebugPane {
            EmptyView()
        } else {
            visibleBody
        }
    }

    private var visibleBody: some View {
        VStack(spacing: 0) {
            header
            if session.showDebugPane {
                Divider()
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(session.debugEntries) { entry in
                                HStack(alignment: .top, spacing: 6) {
                                    Text(dateFormatter.string(from: entry.timestamp))
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                    Text("[\(entry.category)]")
                                        .font(.caption2.monospaced().weight(.semibold))
                                        .foregroundStyle(.blue)
                                    Text(entry.message)
                                        .font(.caption2.monospaced())
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .id(entry.id)
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                    }
                    .frame(height: 180)
                    .background(Color.black.opacity(0.04))
                    .onChange(of: session.debugEntries.count) { _, _ in
                        if let last = session.debugEntries.last {
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                }
            }
        }
    }

    private func copyAll() {
        let text = session.debugEntries
            .map { "\(dateFormatter.string(from: $0.timestamp)) [\($0.category)] \($0.message)" }
            .joined(separator: "\n")
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = text
        #endif
    }

    private var header: some View {
        @Bindable var bindableSession = session
        return HStack {
            Button {
                bindableSession.showDebugPane.toggle()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: session.showDebugPane ? "chevron.down" : "chevron.up")
                    Text("Debug")
                }
                .font(.caption.weight(.semibold))
            }
            .buttonStyle(.plain)
            Text("\(session.debugEntries.count) entries")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            if !session.debugEntries.isEmpty {
                Button {
                    copyAll()
                    withAnimation(.easeOut(duration: 0.15)) {
                        justCopied = true
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        withAnimation(.easeIn(duration: 0.2)) {
                            justCopied = false
                        }
                    }
                } label: {
                    Image(systemName: justCopied ? "checkmark" : "doc.on.doc")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(4)
                        .background(.regularMaterial, in: Circle())
                        .symbolEffect(.bounce, value: justCopied)
                }
                .buttonStyle(.plain)
                .help(justCopied ? "Copied" : "Copy all debug entries")
                Button {
                    // Emit the bundled messages + debug log via
                    // os.Logger so `ios/scripts/mcp-report.sh` on
                    // Mac can pick it out of the idevicesyslog
                    // buffer. Returns a short hash the UI flashes
                    // so the user can tell me which one to fetch.
                    let hash = session.emitDebugReport()
                    lastReportHash = hash
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                        if lastReportHash == hash { lastReportHash = nil }
                    }
                } label: {
                    if let h = lastReportHash {
                        Text("Sent · \(h)")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.green)
                    } else {
                        HStack(spacing: 3) {
                            Image(systemName: "paperplane")
                            Text("Report")
                        }
                        .font(.caption)
                    }
                }
                .buttonStyle(.plain)
                .help("Send a debug report over syslog. Reassemble on Mac with "
                      + "ios/scripts/mcp-report.sh latest.")
                Button("Clear") { session.debugEntries.removeAll() }
                    .font(.caption)
                    .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(.thinMaterial)
    }
}
