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
            Section("Model") {
                ModelPickerView()
                Text(modelStateDescription)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
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

            VoiceModelSection()

            Section("Debug") {
                @Bindable var bindable = session
                Toggle(isOn: $bindable.showDebugPane) {
                    Text("Show debug pane")
                }
                Text("When on, a log strip appears below the chat showing tool dispatches, per-turn memory, and model timing. Turn off for a clean UI.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                NavigationLink {
                    PastLogsView()
                } label: {
                    Label("Past logs", systemImage: "doc.text.magnifyingglass")
                }
                Text("Each launch writes a timestamped log to disk so you can read it (and Share / AirDrop it) after a crash, even if the debug pane cleared.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("Routing") {
                @Bindable var bindable = session
                Toggle(isOn: $bindable.routingSkipModelReply) {
                    Text("Fast routing replies")
                }
                Text("When on, routing questions (\"directions to X\") skip the model's final summary turn and render the distance / duration / first few steps directly from the tool result. Saves about 5 s per route query. Reply wording is more mechanical.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("Generation") {
                @Bindable var bindable = session
                Toggle(isOn: $bindable.longerReplies) {
                    Text("Longer replies")
                }
                Text("Doubles the per-turn token budget (\(DeviceProfile.current.maxReplyTokens) → \(DeviceProfile.current.maxReplyTokens * 2) tokens) so the model can finish longer answers without clipping. Costs extra generation time and KV-cache memory; with 4-bit KV quantization on, the memory tax is ~4× cheaper than it used to be.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("Providers") {
                @Bindable var bindable = session
                Toggle(isOn: $bindable.enableAppleFMBinding) {
                    Text("Apple Foundation Models")
                }
                Text("Adds Apple's on-device Foundation Models to the model picker. Off saves the framework load (~10–30 MB) + per-provider tool schemas; on exposes two extra picker entries (text-loop and native-tools). Takes effect on next app launch.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
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

    private var modelStateDescription: String {
        switch session.modelState {
        case .notLoaded:          return "Not loaded. Pick a model above."
        case .loading:            return "Loading weights…"
        case .downloading(let p): return "Downloading weights… \(Int(p * 100))%"
        case .ready:              return "Ready."
        case .failed(let m):      return "Failed: \(m)"
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

/// Kokoro voice asset + picker. Shows current backend, disk
/// footprint, a one-click downloader for the ~360 MB assets, and a
/// voice picker. When the assets aren't downloaded, voice chat
/// still works — it just uses `AVSpeechSynthesizer` (system voice)
/// until Kokoro is ready.
private struct VoiceModelSection: View {
    @State private var downloader = KokoroDownloader()
    @State private var selectedVoice: String = KokoroVoicePreference.current
    @State private var isDownloaded: Bool = KokoroAssets.isDownloaded

    var body: some View {
        Section {
            HStack {
                Text("Backend")
                Spacer()
                Text(isDownloaded ? "Kokoro v1.0 (neural, on-device)" : "System voice (AVSpeechSynthesizer)")
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            HStack {
                Text("Size on disk")
                Spacer()
                Text(formatBytes(KokoroAssets.currentBytesOnDisk)
                     + " / " + formatBytes(KokoroAssets.totalExpectedBytes))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            switch downloader.state {
            case .idle, .finished, .failed:
                if isDownloaded {
                    Picker("Voice", selection: $selectedVoice) {
                        ForEach(KokoroVoicePreference.available, id: \.self) { v in
                            Text(v).tag(v)
                        }
                    }
                    .onChange(of: selectedVoice) { _, new in
                        KokoroVoicePreference.current = new
                    }
                    Button(role: .destructive) {
                        try? KokoroAssets.deleteAll()
                        isDownloaded = KokoroAssets.isDownloaded
                    } label: {
                        Label("Remove Kokoro voice", systemImage: "trash")
                    }
                } else {
                    Button {
                        Task {
                            await downloader.downloadIfNeeded()
                            isDownloaded = KokoroAssets.isDownloaded
                        }
                    } label: {
                        Label("Download Kokoro voice (~\(formatBytes(KokoroAssets.totalExpectedBytes)))",
                              systemImage: "arrow.down.circle")
                    }
                    if case .failed(let msg) = downloader.state {
                        Text("Last attempt failed: \(msg)")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            case .downloading(let name, let written, let total, let overall):
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: overall)
                    Text("Downloading \(name) — \(formatBytes(written)) / \(formatBytes(total))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Button(role: .destructive) {
                    downloader.cancel()
                } label: {
                    Label("Cancel download", systemImage: "xmark.circle")
                }
            }
            Text("Kokoro v1.0 is an 82M-param neural TTS running on Apple MLX. Without it, voice chat uses your system voice — always available, lower quality. Model from `mlx-community/Kokoro-82M-bf16`; voices from the KokoroTestApp pack. Apache-2.0 licensed.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        } header: {
            Text("Voice chat")
        }
    }

    private func formatBytes(_ b: Int64) -> String {
        let f = ByteCountFormatter()
        f.allowedUnits = [.useMB, .useGB]
        f.countStyle = .file
        return f.string(fromByteCount: b)
    }
}

/// Persisted voice-name preference. `TTSFactory.makeBest` reads
/// this on each build of the voice chat.
public enum KokoroVoicePreference {
    private static let key = "kokoro.voice"
    static let available: [String] = [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ]
    public static var current: String {
        get { UserDefaults.standard.string(forKey: key) ?? "af_heart" }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }
}

#Preview {
    NavigationStack { LibraryView() }.environment(ChatSession())
}
