// SPDX-License-Identifier: MIT

import SwiftUI

/// Pick-and-go in-app downloads: Wikipedia editions (with and without
/// pictures) and StreetZIM offline maps. Check any number of archives, tap
/// one button, and the background downloader takes it from there — transfers
/// keep going when the app is in the background, and the device stays awake
/// while the app is frontmost with downloads running. Finished files load
/// into the library automatically.
struct DownloadCatalogView: View {
    @ObservedObject private var downloads = ZimDownloadManager.shared

    @State private var wikipedia = WikipediaZimCatalog.fallback
    @State private var maps = StreetZimCatalog.fallback
    @State private var catalogNotice: String?
    @State private var didLoadCatalogs = false
    @State private var selection: Set<String> = []
    @State private var searchText = ""
    @State private var showSpaceWarning = false

    var body: some View {
        List {
            if !downloads.items.isEmpty {
                downloadsSection
            }
            wikipediaSection(title: "Wikipedia — no pictures",
                             footer: "Full article text at a fraction of the size. Best for phones and limited storage.",
                             images: false)
            wikipediaSection(title: "Wikipedia — with pictures",
                             footer: "The same articles including images. Bigger downloads; nicest reading experience.",
                             images: true)
            mapSections
        }
        .navigationTitle("Download Library")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .searchable(text: $searchText, prompt: "Search maps and editions")
        .safeAreaInset(edge: .bottom) {
            if !selection.isEmpty {
                goBar
            }
        }
        .alert("Not enough free space", isPresented: $showSpaceWarning) {
            Button("Download Anyway") { startSelectedDownloads() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The selected archives (\(selectionSizeLabel)) look larger than this device's available storage (\(availableSpaceLabel)). You can free up space, choose smaller editions, or continue anyway.")
        }
        .task {
            guard !didLoadCatalogs else { return }
            didLoadCatalogs = true
            await refreshCatalogs()
        }
        .refreshable { await refreshCatalogs() }
    }

    // MARK: Catalog loading

    private func refreshCatalogs() async {
        async let wikipediaResult = try? WikipediaZimCatalog.load()
        async let mapsResult = try? StreetZimCatalog.load()
        let (loadedWikipedia, loadedMaps) = await (wikipediaResult, mapsResult)
        if let loadedWikipedia { wikipedia = loadedWikipedia }
        if let loadedMaps { maps = loadedMaps }
        catalogNotice = (loadedWikipedia == nil || loadedMaps == nil)
            ? "Couldn't refresh the online catalog — showing the list bundled with this version."
            : nil
    }

    // MARK: Sections

    @ViewBuilder
    private var downloadsSection: some View {
        Section("Downloading now") {
            ForEach(downloads.items) { item in
                HTTPDownloadRow(item: item, downloads: downloads)
            }
            Text("Downloads continue while the app is in the background. With the app open, the device stays awake until they finish.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func wikipediaSection(title: String, footer: String, images: Bool) -> some View {
        let items = filtered(wikipedia.filter { $0.kind == .wikipedia(images: images) })
        if !items.isEmpty {
            Section {
                ForEach(items) { item in
                    CatalogRow(item: item,
                               isSelected: selection.contains(item.id),
                               status: status(of: item)) {
                        toggle(item)
                    }
                }
                if images == false, let catalogNotice {
                    Text(catalogNotice)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text(title)
            } footer: {
                Text(footer)
            }
        }
    }

    @ViewBuilder
    private var mapSections: some View {
        // Preserve the catalog's tier ordering (continents first, then
        // regions, countries, states/cities).
        let visible = filtered(maps)
        let tiers = orderedTiers(of: visible)
        ForEach(tiers, id: \.self) { tier in
            Section {
                ForEach(visible.filter { ($0.tier ?? "") == tier }) { item in
                    CatalogRow(item: item,
                               isSelected: selection.contains(item.id),
                               status: status(of: item)) {
                        toggle(item)
                    }
                }
            } header: {
                Text(tier.isEmpty ? "Offline street maps" : "Maps · \(tier)")
            } footer: {
                if tier == tiers.last {
                    Text("StreetZIM maps include routing and directions, rich place info, and Wikipedia links. Pick the smallest area that covers where you need directions.")
                }
            }
        }
    }

    private var goBar: some View {
        VStack(spacing: 6) {
            Button {
                if exceedsFreeSpace {
                    showSpaceWarning = true
                } else {
                    startSelectedDownloads()
                }
            } label: {
                Text("Download \(selection.count) archive\(selection.count == 1 ? "" : "s") · \(selectionSizeLabel)")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            Text("\(availableSpaceLabel) free on this device")
                .font(.caption)
                .foregroundStyle(exceedsFreeSpace ? .red : .secondary)
        }
        .padding()
        .background(.bar)
    }

    // MARK: Selection & downloads

    private var allItems: [ZimCatalogItem] { wikipedia + maps }

    private var selectedItems: [ZimCatalogItem] {
        allItems.filter { selection.contains($0.id) }
    }

    private var selectionBytes: Int64 {
        selectedItems.reduce(0) { $0 + $1.sizeBytes }
    }

    private var selectionSizeLabel: String {
        ByteCountFormatter.string(fromByteCount: selectionBytes, countStyle: .file)
    }

    private var availableSpaceLabel: String {
        guard let bytes = ZimDownloadManager.availableLibraryBytes() else { return "unknown space" }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private var exceedsFreeSpace: Bool {
        guard let free = ZimDownloadManager.availableLibraryBytes() else { return false }
        return selectionBytes > free
    }

    private func toggle(_ item: ZimCatalogItem) {
        if selection.contains(item.id) { selection.remove(item.id) }
        else { selection.insert(item.id) }
    }

    private func startSelectedDownloads() {
        for item in selectedItems {
            downloads.download(item)
        }
        selection.removeAll()
    }

    private func status(of item: ZimCatalogItem) -> CatalogRow.Status {
        if downloads.isInFlight(id: item.id) { return .inFlight }
        if ZimDownloadManager.alreadyInLibrary(filename: item.filename) { return .inLibrary }
        return .selectable
    }

    private func filtered(_ items: [ZimCatalogItem]) -> [ZimCatalogItem] {
        guard !searchText.isEmpty else { return items }
        return items.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
                || $0.detail.localizedCaseInsensitiveContains(searchText)
        }
    }

    private func orderedTiers(of items: [ZimCatalogItem]) -> [String] {
        var seen = Set<String>()
        var tiers: [String] = []
        for item in items {
            let tier = item.tier ?? ""
            if seen.insert(tier).inserted { tiers.append(tier) }
        }
        return tiers
    }
}

// MARK: - Rows

private struct CatalogRow: View {
    enum Status {
        case selectable
        case inFlight
        case inLibrary
    }

    let item: ZimCatalogItem
    let isSelected: Bool
    let status: Status
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(alignment: .top, spacing: 12) {
                leadingIcon
                    .font(.title3)
                    .frame(width: 26)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(item.title)
                            .font(.headline)
                        if item.recommended {
                            Text("Recommended")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.blue.opacity(0.14), in: Capsule())
                        }
                    }
                    if !item.detail.isEmpty {
                        Text(item.detail)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Text(trailingCaption)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(status == .inLibrary ? .green : .primary)
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 2)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(status != .selectable)
    }

    @ViewBuilder
    private var leadingIcon: some View {
        switch status {
        case .inLibrary:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .inFlight:
            Image(systemName: "arrow.down.circle")
                .foregroundStyle(.secondary)
        case .selectable:
            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(isSelected ? Color.accentColor : .secondary)
        }
    }

    private var trailingCaption: String {
        switch status {
        case .inLibrary: return "In your library"
        case .inFlight: return "Downloading — see above"
        case .selectable: return item.sizeLabel
        }
    }
}

private struct HTTPDownloadRow: View {
    let item: ZimDownloadManager.Item
    @ObservedObject var downloads: ZimDownloadManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(item.title)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                Spacer()
                Menu {
                    actions
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .foregroundStyle(.secondary)
                }
                .fixedSize()
            }
            if item.state != .finished {
                ProgressView(value: item.fractionComplete)
                    .tint(tint)
            }
            HStack {
                statusText
                Spacer()
                if item.state == .downloading, item.bytesPerSecond > 0 {
                    Label(SwarmFormat.rate(item.bytesPerSecond), systemImage: "arrow.down")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }

    private var tint: Color {
        switch item.state {
        case .paused: return .gray
        case .failed: return .red
        default: return .accentColor
        }
    }

    @ViewBuilder
    private var statusText: some View {
        switch item.state {
        case .downloading, .paused:
            Text("\(SwarmFormat.bytes(item.receivedBytes)) of \(SwarmFormat.bytes(item.expectedBytes))\(item.state == .paused ? " · Paused" : "")")
        case .failed(let message):
            Text(message)
                .foregroundStyle(.red)
                .lineLimit(2)
        case .finished:
            Label("In your library", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        }
    }

    @ViewBuilder
    private var actions: some View {
        switch item.state {
        case .downloading:
            Button {
                downloads.pause(id: item.id)
            } label: {
                Label("Pause", systemImage: "pause.fill")
            }
            Button(role: .destructive) {
                downloads.cancel(id: item.id)
            } label: {
                Label("Cancel", systemImage: "trash")
            }
        case .paused:
            Button {
                downloads.resume(id: item.id)
            } label: {
                Label("Resume", systemImage: "play.fill")
            }
            Button(role: .destructive) {
                downloads.cancel(id: item.id)
            } label: {
                Label("Cancel", systemImage: "trash")
            }
        case .failed:
            Button {
                downloads.resume(id: item.id)
            } label: {
                Label("Retry", systemImage: "arrow.clockwise")
            }
            Button(role: .destructive) {
                downloads.cancel(id: item.id)
            } label: {
                Label("Dismiss", systemImage: "xmark")
            }
        case .finished:
            Button {
                downloads.cancel(id: item.id)
            } label: {
                Label("Dismiss", systemImage: "xmark")
            }
        }
    }
}
