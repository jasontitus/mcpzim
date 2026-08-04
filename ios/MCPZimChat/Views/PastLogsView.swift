// SPDX-License-Identifier: MIT
//
// Browser for `LogArchive` — the persistent debug-log files each
// launch writes under `Documents/debug-logs/`. Two views:
//
//   - `PastLogsView`: list of every persisted run, newest first,
//     with date + file size. Tap to inspect; swipe to delete.
//   - `LogDetailView`: scrollable text of one file with a Share
//     button (AirDrop → Mac `~/Downloads`, Mail, Save to Files,
//     copy). Cuts the "copy from debug pane → switch apps → paste"
//     dance down to a single tap.

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct PastLogsView: View {
    @State private var files: [LogArchive.LogFileInfo] = []

    /// Shared formatters: building a DateFormatter (and ByteCountFormatter)
    /// per row per body pass was the classic per-row-formatter jank; the
    /// row metadata now also comes prefetched from `allFileInfos()` instead
    /// of two extra `resourceValues` stats per row.
    private static let dateFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateStyle = .short
        df.timeStyle = .medium
        return df
    }()
    private static let sizeFormatter = ByteCountFormatter()

    var body: some View {
        List {
            if files.isEmpty {
                Text("No past logs yet. Log files are written on each launch.")
                    .foregroundStyle(.secondary)
                    .font(.footnote)
            }
            ForEach(files, id: \.url) { info in
                NavigationLink {
                    LogDetailView(url: info.url)
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(info.url.lastPathComponent)
                            .font(.body.monospaced())
                            .lineLimit(1)
                        Text("\(Self.dateFormatter.string(from: info.modified)) · \(Self.sizeFormatter.string(fromByteCount: info.sizeBytes))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .swipeActions(edge: .trailing) {
                    Button(role: .destructive) {
                        LogArchive.shared.delete(info.url)
                        reload()
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        }
        .navigationTitle("Past logs")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(role: .destructive) {
                    LogArchive.shared.deleteAll()
                    reload()
                } label: {
                    Label("Clear all", systemImage: "trash")
                }
                .disabled(files.count <= 1)
            }
        }
        .onAppear(perform: reload)
    }

    private func reload() {
        files = LogArchive.shared.allFileInfos()
    }
}

struct LogDetailView: View {
    let url: URL
    @State private var logText: String = ""
    @State private var showShare = false

    var body: some View {
        ScrollView {
            Text(logText)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
        }
        .navigationTitle(url.lastPathComponent)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showShare = true
                } label: {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
            }
        }
        .onAppear {
            // Session logs can run to many MB; `read` does a whole-file
            // `String(contentsOf:)`. Load it off the main thread so opening
            // a large log doesn't hang the UI.
            let target = url
            Task {
                let text = await Task.detached(priority: .userInitiated) {
                    LogArchive.shared.read(target)
                }.value
                logText = text
            }
        }
        #if os(iOS)
        .sheet(isPresented: $showShare) {
            ShareSheet(items: [url])
        }
        #endif
    }
}

#if os(iOS)
/// Thin SwiftUI wrapper around `UIActivityViewController`. Sharing a
/// file URL (rather than raw text) means AirDrop lands the log as a
/// `.log` file in the Mac's `~/Downloads` — which is exactly the
/// "get the log to my laptop" shortcut we wanted.
private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
#endif
