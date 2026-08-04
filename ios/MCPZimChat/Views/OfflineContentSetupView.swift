// SPDX-License-Identifier: MIT

import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// First-run and settings-accessible hub for assembling an offline library.
/// Two paths, fastest first:
///
///   1. **Copy from a friend nearby** — the person who recommended Zimfo
///      shares their Wikipedia/StreetZIM files directly over peer-to-peer
///      Wi-Fi (LocalSwarm engine); minutes, no internet.
///   2. **Download from the internet** — pick Wikipedia editions (with or
///      without pictures) and StreetZIM maps from an in-app catalog; the
///      background downloader finishes them even if the user leaves.
///
/// Either way, finished files land in the app's Documents folder and load
/// into the library automatically — no manual file wrangling. A manual
/// importer remains for archives downloaded elsewhere.
struct OfflineContentSetupView: View {
    @Environment(ChatSession.self) private var session
    @EnvironmentObject private var swarm: ZimSwarmController
    @ObservedObject private var downloads = ZimDownloadManager.shared
    @Environment(\.dismiss) private var dismiss
    @State private var showImporter = false

    private let zimType = UTType(filenameExtension: "zim") ?? .data

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Build your offline library", systemImage: "square.stack.3d.up")
                            .font(.title2.weight(.semibold))
                        Text("Zimfo answers from Wikipedia and offline street maps stored on this device. Get them from a friend in minutes, or download them here — then everything works with no internet at all.")
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 6)
                }

                Section {
                    NavigationLink {
                        NearbyShareView()
                    } label: {
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: "person.2.wave.2.fill")
                                .font(.title3)
                                .foregroundStyle(.tint)
                                .frame(width: 26)
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Copy from a friend nearby")
                                    .font(.headline)
                                Text("Did someone with Zimfo recommend the app? Their Wikipedia and maps copy straight to this device over Wi-Fi — fast, free, and no internet needed.")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                if let summary = swarm.lastImportSummary {
                                    Label(summary, systemImage: "checkmark.circle.fill")
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.green)
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                } header: {
                    Text("Fastest with a friend")
                } footer: {
                    Text("Also how you pay it forward later: share your library with the next person who installs Zimfo.")
                }

                Section {
                    NavigationLink {
                        DownloadCatalogView()
                    } label: {
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: "arrow.down.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.tint)
                                .frame(width: 26)
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Download Wikipedia & maps")
                                    .font(.headline)
                                Text("Wikipedia with or without pictures, plus StreetZIM street maps for your state, country, or continent. Pick any number and go — downloads continue in the background.")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                if activeDownloadCount > 0 {
                                    Label("\(activeDownloadCount) download\(activeDownloadCount == 1 ? "" : "s") running",
                                          systemImage: "arrow.down")
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.blue)
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                } header: {
                    Text("Download on this device")
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
                    Button {
                        showImporter = true
                    } label: {
                        Label("Add a file downloaded elsewhere", systemImage: "folder.badge.plus")
                    }
                    if !session.library.isEmpty {
                        Label("\(session.library.count) offline file\(session.library.count == 1 ? "" : "s") ready",
                              systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                } header: {
                    Text("Already have a ZIM file?")
                } footer: {
                    Text("You can also open any downloaded .zim file from the Files app and choose Zimfo.")
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
        }
    }

    private var activeDownloadCount: Int {
        downloads.items.filter { $0.state == .downloading }.count
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
    OfflineContentSetupView()
        .environment(ChatSession())
        .environmentObject(ZimSwarmController())
}
