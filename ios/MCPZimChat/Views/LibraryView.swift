// SPDX-License-Identifier: MIT

import SwiftUI
import UniformTypeIdentifiers

struct LibraryView: View {
    @Environment(ChatSession.self) private var session
    @State private var showImporter = false
    @State private var pendingDelete: ChatSession.LibraryEntry?

    // Enabled count → used in the header ("3 of 5 enabled") so the user
    // can tell at a glance how many ZIMs the model will actually see.
    private var enabledCount: Int { session.library.filter { $0.isEnabled }.count }

    var body: some View {
        List {
            Section {
                if session.library.isEmpty {
                    Text("No ZIMs loaded. Drop `.zim` files into this app's Documents folder (via Finder while the device is mounted, or via Files.app), then tap Refresh.")
                        .foregroundStyle(.secondary)
                        .font(.footnote)
                } else {
                    ForEach(session.library) { entry in
                        LibraryRow(
                            entry: entry,
                            onToggle: { enabled in
                                Task { await session.setEnabled(enabled, for: entry.id) }
                            }
                        )
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            Button(role: .destructive) {
                                pendingDelete = entry
                            } label: {
                                Label(entry.isInSandboxDocuments ? "Delete" : "Unlink",
                                      systemImage: "trash")
                            }
                        }
                        .contextMenu {
                            Button(role: .destructive) {
                                pendingDelete = entry
                            } label: {
                                Label(entry.isInSandboxDocuments ? "Move to Trash" : "Unlink",
                                      systemImage: "trash")
                            }
                        }
                    }
                }
            } header: {
                HStack {
                    Text("Loaded ZIMs")
                    Spacer()
                    if !session.library.isEmpty {
                        Text("\(enabledCount) of \(session.library.count) enabled")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Section {
                HStack {
                    Text("Article cap")
                    Spacer()
                    Text("\(session.articleCapKB) KB")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Stepper(
                    value: Binding(
                        get: { session.articleCapKB },
                        set: { session.articleCapKB = $0 }
                    ),
                    in: 4...96,
                    step: 4
                ) { EmptyView() }
                    .labelsHidden()
                Text("How much of a `get_article` body to feed the model. Bigger = more complete context but slower first-token and a larger memory spike on stream open.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                HStack {
                    Text("Device tier")
                    Spacer()
                    Text(DeviceProfile.current.label)
                        .foregroundStyle(.secondary)
                }
                Text("Defaults for article cap, reply length, and MLX cache scale to available memory. Override the article cap above if you want more / less than your tier's default.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } header: {
                Text("Generation")
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
            ToolbarItem(placement: .primaryAction) {
                Button("Add ZIM") { showImporter = true }
            }
            ToolbarItem(placement: .navigation) {
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
                // Append — don't replace the sandbox-scanned library.
                Task { await session.addReaders(urls: urls) }
            }
        }
        .confirmationDialog(
            pendingDelete.map {
                $0.isInSandboxDocuments
                    ? "Move \"\($0.displayName)\" to the Trash?"
                    : "Remove \"\($0.displayName)\" from the library?"
            } ?? "",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            ),
            presenting: pendingDelete
        ) { entry in
            Button(entry.isInSandboxDocuments ? "Move to Trash" : "Unlink",
                   role: .destructive) {
                Task { await session.removeEntry(entry.id) }
            }
            Button("Cancel", role: .cancel) {}
        } message: { entry in
            Text(entry.isInSandboxDocuments
                 ? "The file will be in the Trash and can be restored from there."
                 : "The file stays where it is; this app just forgets the bookmark.")
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
    let onToggle: (Bool) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Toggle(
                "Enabled",
                isOn: Binding(get: { entry.isEnabled }, set: { onToggle($0) })
            )
            .labelsHidden()
            .toggleStyle(.switch)
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
            .opacity(entry.isEnabled ? 1.0 : 0.5)
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
