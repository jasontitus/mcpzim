// SPDX-License-Identifier: MIT

import SwiftUI
import UniformTypeIdentifiers
import MCPZimKit

struct LibraryView: View {
    @Environment(ChatSession.self) private var session
    @ObservedObject private var downloads = ZimDownloadManager.shared
    @State private var showImporter = false
    @State private var showOfflineSetup = false
    @State private var pendingDelete: ChatSession.LibraryEntry?

    // Enabled count → used in the header ("3 of 5 enabled") so the user
    // can tell at a glance how many ZIMs the model will actually see.
    private var enabledCount: Int { session.library.filter { $0.isEnabled }.count }

    var body: some View {
        List {
            Section("Model") {
                ModelPickerView()
                // Wrap in TimelineView so the "Ns elapsed" tail of the
                // download status refreshes every second even when the
                // Hub's `fractionCompleted` is silent — the UI stays
                // honest about the download being alive. The TimelineView
                // re-evaluates its body on each 1-second tick.
                TimelineView(.periodic(from: .now, by: 1.0)) { _ in
                    HStack(spacing: 8) {
                        // Render an indeterminate spinner whenever the
                        // model is actively fetching or loading — the
                        // Hub's progress callback for a ~2.5 GB
                        // safetensors can sit silent at 1% for minutes
                        // while bytes flow (swift-huggingface coalesces
                        // the per-file NSProgress updates very
                        // coarsely), so the spinner is the most
                        // reliable "the app isn't frozen" signal.
                        switch session.modelState {
                        case .downloading, .waitingForNetwork, .loading:
                            ProgressView()
                                .progressViewStyle(.circular)
                                .controlSize(.small)
                        default:
                            EmptyView()
                        }
                        Text(modelStateDescription)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                if let item = modelDownloadItem {
                    ModelDownloadRow(item: item, downloads: downloads) {
                        Task { await session.select(modelId: session.selectedModel.id) }
                    }
                }
                Text("The selected model downloads automatically the first time, resumes interrupted downloads, and runs entirely on this device afterward.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Section("Get offline content") {
                NavigationLink {
                    NearbyShareView()
                } label: {
                    Label("Nearby sharing", systemImage: "person.2.wave.2.fill")
                }
                NavigationLink {
                    DownloadCatalogView()
                } label: {
                    Label("Download Wikipedia & maps", systemImage: "arrow.down.circle")
                }
                Button {
                    showOfflineSetup = true
                } label: {
                    Label("Guided setup", systemImage: "square.stack.3d.up")
                }
                Text("Copy a friend's library directly over Wi-Fi, or download Wikipedia (with or without pictures) and StreetZIM street maps in the background.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Section {
                if session.library.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("No offline content yet")
                            .font(.headline)
                        Text("Use the guided setup above, or add a library or map you already downloaded. You do not need to manage the app's Documents folder manually.")
                            .foregroundStyle(.secondary)
                            .font(.footnote)
                    }
                    .padding(.vertical, 4)
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
            // MARK: Behavior — everything that changes HOW the model
            // answers. Reply length, routing shortcut, article budget,
            // voice output, provider surface. Kept together so a
            // user tweaking "why is it so slow / chatty / quiet"
            // doesn't have to jump between three panes.
            Section("Behavior") {
                @Bindable var bindable = session
                ConversationModePicker(session: session)
                Toggle(isOn: $bindable.longerReplies) {
                    Text("Longer replies")
                }
                Text("Includes more ZIM excerpts in discussion answers and gives tool planning more time to finish.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Text("Route answers use distances and directions from the offline map.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
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
                Text("Defaults for article cap, reply length, and MLX cache scale to available memory. Override above if you want more / less than your tier's default.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            // Voice is its own composite — keep the sub-view but fold
            // it under Behavior visually by rendering it right after.
            // Its `Section` header (inside the view) will read "Voice".
            VoiceModelSection()

            Section("Providers") {
                @Bindable var bindable = session
                Toggle(isOn: $bindable.enableAppleFMBinding) {
                    Text("Apple Foundation Models")
                }
                Text("Adds Apple's on-device Foundation Models to the model picker. Off saves the framework load (~10–30 MB) + per-provider tool schemas; on exposes two extra picker entries (text-loop and native-tools). Takes effect on next app launch.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            // MARK: Debug — one place for the pane toggle, the Past-logs
            // archive, the Report button's GitHub PAT, and any future
            // diagnostic plumbing. The user asked for "all debug stuff
            // together" — this is it.
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

                Toggle(isOn: Binding(
                    get: { UserDefaults.standard.bool(forKey: DiagnosticsUploader.optInKey) },
                    set: { DiagnosticsUploader.setEnabled($0) }
                )) {
                    Text("Share debug logs for analysis")
                }
                Text("When on, each finished session's log — your questions, the article titles retrieved, and GPS coordinates — uploads to the developer's private store so conversations can be reviewed for quality. Off by default; everything stays on device until you turn this on.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                #if DEBUG
                VStack(alignment: .leading, spacing: 6) {
                    Text("GitHub PAT (gist scope)")
                        .font(.subheadline.weight(.semibold))
                    SecureField("ghp_…", text: Binding(
                        get: { DebugReportConfig.githubToken ?? "" },
                        set: { DebugReportConfig.githubToken = $0 }
                    ))
                    .textContentType(.password)
                    .autocorrectionDisabled()
                    #if os(iOS)
                    // `.textInputAutocapitalization` is iOS / catalyst /
                    // visionOS / tvOS / watchOS only — it does NOT exist
                    // on native macOS, and including it unconditionally
                    // causes the Mac target's SwiftUI type-checker to bail
                    // with "unable to type-check this expression in
                    // reasonable time". `.autocorrectionDisabled()` is
                    // cross-platform and already covers the important
                    // part on Mac (no autocorrect on a PAT).
                    .textInputAutocapitalization(.never)
                    #endif
                    .font(.footnote.monospaced())
                    Text("When set, the debug-pane Report button uploads the "
                         + "session as a secret gist so it can be fetched from "
                         + "anywhere via `ios/scripts/mcp-report.sh cloud`. "
                         + "Unset → syslog-only (requires `mcp-logs.sh` "
                         + "running on Mac).")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    if let token = DebugReportConfig.githubToken, !token.isEmpty {
                        Text("✓ token set (\(token.count) chars)")
                            .font(.footnote)
                            .foregroundStyle(.green)
                    }
                }
                .padding(.top, 4)
                #endif
            }

            Section("About & privacy") {
                LabeledContent("Version", value: Self.appVersionString)
                Link(destination: URL(string: "https://tiltastech-zimfo.web.app/privacy")!) {
                    Label("Privacy policy", systemImage: "hand.raised")
                }
                Link(destination: URL(string: "https://tiltastech-zimfo.web.app/support")!) {
                    Label("Support", systemImage: "questionmark.circle")
                }
                Link(destination: URL(string: "https://tiltastech-zimfo.web.app/licenses")!) {
                    Label("Licenses & attribution", systemImage: "doc.text")
                }
                Text("Questions, transcripts, article titles, ZIM filenames, and GPS coordinates stay on this device unless you choose to share debug logs or a report. Zimfo also sends limited usage and performance analytics; see the policy for details.")
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
                Button("Add File") { showImporter = true }
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
        .sheet(isPresented: $showOfflineSetup) {
            OfflineContentSetupView()
                .environment(session)
        }
        .alert(
            "Could not add offline content",
            isPresented: Binding(
                get: { session.libraryError != nil },
                set: { if !$0 { session.libraryError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { session.libraryError = nil }
        } message: {
            Text(session.libraryError ?? "Unknown error")
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
        case .waitingForNetwork:  return "Waiting for network…"
        case .downloading(let p):
            let pct = "\(Int(p * 100))%"
            let start = session.downloadStartedAt
            let elapsed = start.map { Int(Date().timeIntervalSince($0)) } ?? 0
            let elapsedStr = "\(elapsed)s"
            // `swift-huggingface`'s NSProgress stays near 0 for a long
            // stretch on big .safetensors downloads (the parent
            // Progress tree coalesces very coarsely), so after ~20s at
            // ≤1% we replace the stuck percent with a "fetching…" hint
            // and lean on the elapsed counter + spinner to show life.
            // When the Hub finally emits a jump past 5% we go back to
            // the numeric display.
            let percentLooksStuck = (Int(p * 100) <= 1) && elapsed > 20
            if percentLooksStuck {
                return "Fetching \(session.selectedModel.displayName)… \(elapsedStr)"
            }
            return "Downloading weights… \(pct) · \(elapsedStr)"
        case .ready:              return "Ready."
        case .failed(let m):      return "Failed: \(m)"
        }
    }

    /// The selected model's live background transfer, if any. Only surfaced
    /// while it's actually downloading / waiting / paused / failed — a
    /// finished download is already reflected in the "Ready." status.
    private var modelDownloadItem: ZimDownloadManager.Item? {
        downloads.items.first {
            $0.kind == .model && $0.id == session.selectedModel.id
                && $0.state != .finished
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

    /// "1.0 (20260802151334)" — marketing version + the datestamped build
    /// number the TestFlight pipeline stamps, so a tester's settings page
    /// identifies the exact upload. Dev builds show the project default.
    static var appVersionString: String {
        let short = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
        let build = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        return "\(short) (\(build))"
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

/// On-device TTS engine, voice, and asset controls. Supertonic is the default
/// iPhone backend; Kokoro remains selectable for listening comparisons.
private struct VoiceModelSection: View {
    @State private var downloader = KokoroDownloader()
    @State private var selectedBackend = TTSBackendPreference.current
    @State private var selectedVoice: String = KokoroVoicePreference.current
    @State private var isDownloaded: Bool = KokoroAssets.isDownloaded
    @State private var supertonicBytesOnDisk: Int64?
    @State private var kokoroBytesOnDisk: Int64?
    #if canImport(FluidAudio)
    @State private var selectedSupertonicVoice = SupertonicVoicePreference.current
    #endif

    var body: some View {
        Section {
            Picker("Engine", selection: $selectedBackend) {
                ForEach(TTSBackendPreference.allCases, id: \.self) { backend in
                    Text(backend.displayName).tag(backend)
                }
            }
            .onChange(of: selectedBackend) { _, newValue in
                TTSBackendPreference.current = newValue
            }
            backendControls
        } header: {
            Text("Voice chat")
        } footer: {
            Text("Engine and voice changes apply the next time voice chat starts.")
        }
        .task(id: selectedBackend) {
            await refreshDiskUsage()
        }
    }

    @ViewBuilder
    private var backendControls: some View {
        if selectedBackend == .supertonic {
            #if canImport(FluidAudio)
            Picker("Voice", selection: $selectedSupertonicVoice) {
                ForEach(SupertonicVoicePreference.available, id: \.self) { voice in
                    Text(voice).tag(voice)
                }
            }
            .onChange(of: selectedSupertonicVoice) { _, newValue in
                SupertonicVoicePreference.current = newValue
            }
            HStack {
                Text("Size on disk")
                Spacer()
                Text(supertonicBytesOnDisk.map(formatBytes) ?? "Calculating…")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Text("Supertonic 3 uses fixed-shape INT8 Core ML models, with the repeated VectorEstimator running primarily on the Neural Engine. Assets download on the first voice session and remain entirely on-device afterward.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            #else
            Text("Supertonic 3 is not linked in this build.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            #endif
        } else if selectedBackend == .kokoro {
            kokoroControls
        } else {
            Text("Apple's compact system voice needs no model download and uses the least memory, with lower voice quality than the neural engines.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var kokoroControls: some View {
        HStack {
            Text("Size on disk")
            Spacer()
            Text((kokoroBytesOnDisk.map(formatBytes) ?? "Calculating…")
                 + " / " + formatBytes(KokoroAssets.totalExpectedBytes))
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
        if isDownloaded {
            Label("Kokoro ready offline", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        }
        switch downloader.state {
        case .idle, .finished, .failed:
            if isDownloaded {
                Picker("Voice", selection: $selectedVoice) {
                    ForEach(KokoroVoicePreference.available, id: \.self) { voice in
                        Text(voice).tag(voice)
                    }
                }
                .onChange(of: selectedVoice) { _, newValue in
                    KokoroVoicePreference.current = newValue
                }
                Button(role: .destructive) {
                    try? KokoroAssets.deleteAll()
                    isDownloaded = KokoroAssets.isDownloaded
                    kokoroBytesOnDisk = 0
                } label: {
                    Label("Remove Kokoro voice", systemImage: "trash")
                }
            } else {
                Button {
                    Task {
                        await downloader.downloadIfNeeded()
                        isDownloaded = KokoroAssets.isDownloaded
                        await refreshDiskUsage()
                    }
                } label: {
                    Label("Download Kokoro voice (~\(formatBytes(KokoroAssets.totalExpectedBytes)))",
                          systemImage: "arrow.down.circle")
                }
                if case .failed(let message) = downloader.state {
                    Text("Last attempt failed: \(message)")
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
        Text("Kokoro offers a richer voice and uses more memory. If memory gets tight, voice chat uses Supertonic; the active engine appears in the voice screen. Downloaded voices work offline.")
            .font(.footnote)
            .foregroundStyle(.secondary)
    }

    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter
    }()

    private func formatBytes(_ bytes: Int64) -> String {
        Self.byteFormatter.string(fromByteCount: bytes)
    }

    @MainActor
    private func refreshDiskUsage() async {
        #if canImport(FluidAudio)
        let sizes = await Task.detached(priority: .utility) {
            (Supertonic3Assets.currentBytesOnDisk, KokoroAssets.currentBytesOnDisk)
        }.value
        supertonicBytesOnDisk = sizes.0
        kokoroBytesOnDisk = sizes.1
        #else
        kokoroBytesOnDisk = await Task.detached(priority: .utility) {
            KokoroAssets.currentBytesOnDisk
        }.value
        #endif
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

/// Pulled out of the Behavior section: inlining the picker plus its
/// footnote pushed that Section's body past the SwiftUI type-checker's
/// budget ("unable to type-check this expression in reasonable time",
/// 2026-08-13). Its own view keeps each body small enough to infer.
private struct ConversationModePicker: View {
    let session: ChatSession

    var body: some View {
        @Bindable var bindable = session
        Picker("Answer from", selection: $bindable.conversationMode) {
            Text("Maps or Wikipedia").tag(ConversationMode.auto)
            Text("Maps first").tag(ConversationMode.local)
            Text("Wikipedia only").tag(ConversationMode.encyclopedia)
        }
        Text(Self.explanation)
            .font(.footnote)
            .foregroundStyle(.secondary)
    }

    private static let explanation = """
        Only affects questions that could mean either — "what's in \
        Georgetown?" is its cafés or its history. "Maps or Wikipedia" \
        guesses, then falls back to the other when the guess misses. \
        Switchable by voice: say "let's talk local" or "back to normal".
        """
}

/// Live control row for the selected model's background download — the same
/// bytes / progress / state / Pause / Resume / Retry surface the ZIM catalog
/// downloader offers, so a multi-GB GGUF pull is never an opaque percent.
private struct ModelDownloadRow: View {
    let item: ZimDownloadManager.Item
    @ObservedObject var downloads: ZimDownloadManager
    /// Re-arms the provider (re-selecting the model re-enters `load()`), so a
    /// failed pull starts over with a fresh attempt and a fresh await.
    let onRetry: () -> Void

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
        case .downloading:
            Text("\(SwarmFormat.bytes(item.receivedBytes)) of \(SwarmFormat.bytes(item.expectedBytes))")
        case .waitingForNetwork:
            Label("Waiting for network…", systemImage: "wifi.slash")
        case .paused:
            Text("\(SwarmFormat.bytes(item.receivedBytes)) of \(SwarmFormat.bytes(item.expectedBytes)) · Paused")
        case .failed(let message):
            Text(message)
                .foregroundStyle(.red)
                .lineLimit(2)
        case .finished:
            Label("Ready to use", systemImage: "checkmark.circle.fill")
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
        case .waitingForNetwork:
            // The OS auto-resumes when connectivity returns. "Retry" tears the
            // parked task down and re-probes immediately.
            Button {
                downloads.resume(id: item.id)
            } label: {
                Label("Retry now", systemImage: "arrow.clockwise")
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
                onRetry()
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
