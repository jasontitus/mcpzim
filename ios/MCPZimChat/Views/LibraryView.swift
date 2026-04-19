// SPDX-License-Identifier: MIT

import SwiftUI
import UniformTypeIdentifiers

struct LibraryView: View {
    @Environment(ChatSession.self) private var session
    @State private var showImporter = false

    var body: some View {
        List {
            Section("Loaded ZIMs") {
                if session.library.isEmpty {
                    Text("No ZIMs loaded. Drop `.zim` files into this app's Documents folder (via Finder while the device is mounted, or via Files.app), then tap Refresh.")
                        .foregroundStyle(.secondary)
                        .font(.footnote)
                } else {
                    ForEach(session.library) { entry in
                        LibraryRow(entry: entry)
                    }
                }
            }
            Section("Aggregate capabilities") {
                let caps = session.adapter == nil ? [] : registryCapabilities()
                if caps.isEmpty {
                    Text("—").foregroundStyle(.secondary)
                } else {
                    ForEach(caps, id: \.self) { cap in
                        Label(cap, systemImage: "checkmark.seal")
                    }
                }
            }
        }
        .navigationTitle("Library")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Add ZIM") { showImporter = true }
            }
            ToolbarItem(placement: .topBarLeading) {
                Button("Refresh") {
                    Task { await session.scanDocumentsFolder() }
                }
            }
        }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [UTType(filenameExtension: "zim") ?? .data],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                Task { await session.openReaders(urls: urls) }
            }
        }
    }

    private func registryCapabilities() -> [String] {
        // Cheap: re-inventory via the service the adapter wraps. Since
        // DefaultZimService is an actor, a sync read requires a Task; for
        // display only we stash on the session and compute on refresh.
        session.library.flatMap { entry -> [String] in
            switch entry.kind {
            case .wikipedia: return ["general_knowledge"]
            case .mdwiki: return ["medical"]
            case .streetzim: return ["maps", "plan_route", "geocode"]
            case .generic: return []
            }
        }.reduce(into: [String]()) { acc, x in if !acc.contains(x) { acc.append(x) } }
        + ["search", "get_article"]
    }
}

private struct LibraryRow: View {
    let entry: ChatSession.LibraryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: icon)
                Text(entry.displayName).font(.headline)
            }
            Text("\(entry.kind.rawValue.capitalized) · \(entry.reader.metadata.articleCount) entries")
                .font(.caption)
                .foregroundStyle(.secondary)
            if !entry.reader.metadata.description.isEmpty {
                Text(entry.reader.metadata.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }

    private var icon: String {
        switch entry.kind {
        case .wikipedia: return "globe"
        case .mdwiki: return "cross.case"
        case .streetzim: return "map"
        case .generic: return "doc"
        }
    }
}

#Preview {
    NavigationStack { LibraryView() }.environment(ChatSession())
}
