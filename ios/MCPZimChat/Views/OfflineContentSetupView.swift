// SPDX-License-Identifier: MIT

import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// A human-facing download choice. Users never need to know the archive's
/// filename; the filename remains available for the system download itself.
struct WikipediaArchiveChoice: Identifiable, Equatable, Sendable {
    enum Edition: String, Hashable, Sendable {
        case top
        case simple
        case complete
    }

    let edition: Edition
    let filename: String
    let sizeLabel: String

    var id: Edition { edition }
    var downloadURL: URL {
        WikipediaArchiveCatalog.indexURL.appendingPathComponent(filename)
    }

    var title: String {
        switch edition {
        case .top: return "Popular English Wikipedia"
        case .simple: return "Simple English Wikipedia"
        case .complete: return "Complete English Wikipedia"
        }
    }

    var detail: String {
        switch edition {
        case .top:
            return "The most useful articles, with full text and no pictures. Best starting point for a phone."
        case .simple:
            return "Every Simple English article, with full text and no pictures. The smallest complete encyclopedia."
        case .complete:
            return "Every English Wikipedia article, with full text and no pictures. Requires substantial free storage."
        }
    }
}

/// Resolves the newest official no-picture Wikipedia files at runtime. Kiwix
/// republishes these archives regularly, so shipping a dated URL in the app
/// would eventually strand new users on an old or removed file.
enum WikipediaArchiveCatalog {
    static let indexURL = URL(string: "https://download.kiwix.org/zim/wikipedia/")!

    static let fallbackChoices: [WikipediaArchiveChoice] = [
        .init(edition: .top,
              filename: "wikipedia_en_top_nopic_2026-06.zim",
              sizeLabel: "2.1 GB"),
        .init(edition: .simple,
              filename: "wikipedia_en_simple_all_nopic_2026-05.zim",
              sizeLabel: "937 MB"),
        .init(edition: .complete,
              filename: "wikipedia_en_all_nopic_2026-06.zim",
              sizeLabel: "49 GB"),
    ]

    static func load() async throws -> [WikipediaArchiveChoice] {
        var request = URLRequest(url: indexURL)
        request.timeoutInterval = 30
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        guard let html = String(data: data, encoding: .utf8) else {
            throw URLError(.cannotDecodeContentData)
        }
        let parsed = parse(html: html)
        guard !parsed.isEmpty else { throw URLError(.cannotParseResponse) }
        return parsed
    }

    /// Pure parser kept internal so the Mac unit target can pin the directory
    /// format without making network requests.
    static func parse(html: String) -> [WikipediaArchiveChoice] {
        let pattern = #"(wikipedia_en_(all|top|simple_all)_nopic_(\d{4}-\d{2})\.zim)</a>\s+[0-9-]+\s+[0-9:]+\s+([0-9.]+\s*[KMGTP])"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }
        let range = NSRange(html.startIndex..., in: html)
        var newest: [WikipediaArchiveChoice.Edition: WikipediaArchiveChoice] = [:]

        for match in regex.matches(in: html, range: range) {
            guard match.numberOfRanges == 5,
                  let filenameRange = Range(match.range(at: 1), in: html),
                  let familyRange = Range(match.range(at: 2), in: html),
                  let sizeRange = Range(match.range(at: 4), in: html)
            else { continue }

            let filename = String(html[filenameRange])
            let edition: WikipediaArchiveChoice.Edition
            switch String(html[familyRange]) {
            case "top": edition = .top
            case "simple_all": edition = .simple
            case "all": edition = .complete
            default: continue
            }
            let choice = WikipediaArchiveChoice(
                edition: edition,
                filename: filename,
                sizeLabel: expandedSize(String(html[sizeRange]))
            )
            if newest[edition].map({ $0.filename < filename }) ?? true {
                newest[edition] = choice
            }
        }

        return [.top, .simple, .complete].compactMap { newest[$0] }
    }

    private static func expandedSize(_ compact: String) -> String {
        let trimmed = compact.replacingOccurrences(of: " ", with: "")
        guard let suffix = trimmed.last else { return compact }
        let number = trimmed.dropLast()
        let unit: String
        switch suffix {
        case "K": unit = "KB"
        case "M": unit = "MB"
        case "G": unit = "GB"
        case "T": unit = "TB"
        case "P": unit = "PB"
        default: return compact
        }
        return "\(number) \(unit)"
    }
}

/// First-run and settings-accessible guide for assembling an offline library.
/// Large archives deliberately download through the system browser: Safari's
/// download manager is much better at multi-gigabyte background transfers than
/// a foreground app session. Zimfo registers the `.zim` document type, so the
/// completed download can be opened directly or selected below.
struct OfflineContentSetupView: View {
    @Environment(ChatSession.self) private var session
    @Environment(\.dismiss) private var dismiss
    @State private var showImporter = false
    @State private var wikipediaChoices = WikipediaArchiveCatalog.fallbackChoices
    @State private var catalogNotice: String?

    private let zimType = UTType(filenameExtension: "zim") ?? .data
    private let streetZIMURL = URL(string: "https://streetzim.web.app/")!

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Build your offline library", systemImage: "square.stack.3d.up")
                            .font(.title2.weight(.semibold))
                        Text("Choose knowledge and maps in plain language. After the downloads finish, add the files to Zimfo once; everything then works without internet access.")
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 6)
                }

                Section("AI model") {
                    HStack {
                        ModelPickerView()
                        Spacer()
                        Label(modelStatus, systemImage: modelStatusIcon)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Text("The selected model downloads automatically on first use, resumes interrupted downloads, and stays on this device afterward.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section {
                    ForEach(wikipediaChoices) { choice in
                        Link(destination: choice.downloadURL) {
                            HStack(alignment: .top, spacing: 12) {
                                Image(systemName: "globe.americas.fill")
                                    .font(.title3)
                                    .frame(width: 24)
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(choice.title)
                                            .font(.headline)
                                        if choice.edition == .top {
                                            Text("Recommended")
                                                .font(.caption2.weight(.semibold))
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 2)
                                                .background(.blue.opacity(0.14), in: Capsule())
                                        }
                                    }
                                    Text(choice.detail)
                                        .font(.footnote)
                                        .foregroundStyle(.secondary)
                                    Text("Download \(choice.sizeLabel)")
                                        .font(.caption.weight(.semibold))
                                }
                                Spacer()
                                Image(systemName: "arrow.up.right")
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    if let catalogNotice {
                        Text(catalogNotice)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("1. Choose Wikipedia")
                } footer: {
                    Text("No-picture editions keep full article text while using far less storage than editions with images.")
                }

                Section {
                    Link(destination: streetZIMURL) {
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: "map.fill")
                                .font(.title3)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Choose a state, region, or country")
                                    .font(.headline)
                                Text("The StreetZIM catalog shows coverage, download size, and a plain-language description for each offline map.")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Image(systemName: "arrow.up.right")
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                } header: {
                    Text("2. Choose offline maps")
                } footer: {
                    Text("Pick the smallest area that covers where you need directions. A state or metro map is faster to download and uses less storage than a continent.")
                }

                Section {
                    Button {
                        showImporter = true
                    } label: {
                        Label("Add downloaded library or map", systemImage: "folder.badge.plus")
                    }
                    if !session.library.isEmpty {
                        Label("\(session.library.count) offline file\(session.library.count == 1 ? "" : "s") ready",
                              systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                } header: {
                    Text("3. Add the downloads")
                } footer: {
                    Text("When a browser download finishes, return here and select it. You can also open a downloaded .zim file from Files and choose Zimfo.")
                }
            }
            .navigationTitle("Offline Setup")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(session.library.isEmpty ? "Not now" : "Done") {
                        dismiss()
                    }
                }
            }
            .fileImporter(
                isPresented: $showImporter,
                allowedContentTypes: [zimType],
                allowsMultipleSelection: true
            ) { result in
                guard case .success(let urls) = result else { return }
                Task { await session.addReaders(urls: urls) }
            }
            .task {
                do {
                    wikipediaChoices = try await WikipediaArchiveCatalog.load()
                    catalogNotice = nil
                } catch {
                    // The fallback links are pinned to the latest files at
                    // build time, so setup remains useful when the index is
                    // temporarily unavailable.
                    catalogNotice = "Using the latest catalog bundled with this version."
                }
            }
        }
    }

    private var modelStatus: String {
        switch session.modelState {
        case .notLoaded: return "Not loaded"
        case .loading: return "Loading"
        case .downloading(let fraction): return "\(Int(fraction * 100))%"
        case .ready: return "Ready"
        case .failed: return "Needs attention"
        }
    }

    private var modelStatusIcon: String {
        switch session.modelState {
        case .ready: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .notLoaded: return "circle"
        case .loading, .downloading: return "arrow.down.circle"
        }
    }
}

#Preview {
    OfflineContentSetupView().environment(ChatSession())
}
