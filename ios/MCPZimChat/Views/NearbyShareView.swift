// SPDX-License-Identifier: MIT

import LocalSwarmEngine
import SwiftUI

/// Nearby sharing: seed this device's ZIM library to friends over
/// AWDL / peer-to-peer Wi-Fi, and copy a nearby friend's Wikipedia and
/// StreetZIM files straight into this library — no router or internet
/// required. This is the "friend just installed Zimfo" bootstrap path.
struct NearbyShareView: View {
    @EnvironmentObject private var controller: ZimSwarmController

    var body: some View {
        // Inner split so SwiftUI observes the engine's @Published state
        // directly (the controller does not republish SwarmManager changes).
        NearbyShareContent(controller: controller, manager: controller.manager)
    }
}

private struct NearbyShareContent: View {
    @ObservedObject var controller: ZimSwarmController
    @ObservedObject var manager: SwarmManager

    @State private var loadingSwarmID: String?
    @State private var selectionRoute: SwarmSelectionRoute?
    @State private var pinPromptSwarm: DiscoveredSwarm?
    @State private var enteredPin = ""

    private var downloadsAndReceives: [TransferStatus] {
        manager.transfers.filter { $0.role != .seeding }
    }
    private var seedingStatus: TransferStatus? {
        manager.transfers.first { $0.role == .seeding }
    }

    var body: some View {
        List {
            shareSection
            nearbySection
            if !manager.pendingReceives.isEmpty || !downloadsAndReceives.isEmpty
                || controller.lastImportSummary != nil {
                transfersSection
            }
        }
        .navigationTitle("Nearby Sharing")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .onAppear { controller.beginBrowsing() }
        .onDisappear { controller.endBrowsing() }
        .sheet(item: $selectionRoute) { route in
            SwarmFileSelectionSheet(manager: manager,
                                    manifest: route.manifest,
                                    swarm: route.swarm,
                                    pin: route.pin)
        }
        .alert("Protected share", isPresented: Binding(
            get: { pinPromptSwarm != nil },
            set: { if !$0 { pinPromptSwarm = nil } })) {
            SecureField("PIN", text: $enteredPin)
            Button("Cancel", role: .cancel) { pinPromptSwarm = nil }
            Button("Get") {
                if let swarm = pinPromptSwarm {
                    let pin = enteredPin
                    pinPromptSwarm = nil
                    Task { await openSwarm(swarm, pin: pin) }
                }
            }
        } message: {
            Text("This share needs a PIN. Enter the one shown on the sharing device.")
        }
        .alert("Nearby sharing", isPresented: Binding(
            get: { manager.lastError != nil },
            set: { if !$0 { manager.clearError() } })) {
            Button("OK") { manager.clearError() }
        } message: {
            Text(manager.lastError ?? "")
        }
    }

    // MARK: Share my library

    @ViewBuilder
    private var shareSection: some View {
        Section {
            Toggle(isOn: Binding(
                get: { controller.isSharingLibrary },
                set: { controller.setSharing($0) })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Share my library")
                    Text(shareSubtitle)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .disabled(!controller.isSharingLibrary && !controller.hasShareableFiles)

            if let modelSize = shareableModelSizeLabel {
                Toggle(isOn: Binding(
                    get: { controller.includeModelInShare },
                    set: { controller.setIncludeModel($0) })) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Include AI model (\(modelSize))")
                        Text("Lets your friend chat entirely offline — no model download needed.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            ForEach(manager.hostPreparations) { preparation in
                HStack(spacing: 12) {
                    ProgressView(value: preparation.fraction)
                    Text("Preparing \(preparation.name) · \(Int(preparation.fraction * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            if controller.isSharingLibrary, let hosted = manager.hostedManifests.first {
                HStack {
                    Label("\(hosted.files.count) file\(hosted.files.count == 1 ? "" : "s") · \(SwarmFormat.bytes(hosted.totalBytes))",
                          systemImage: "dot.radiowaves.left.and.right")
                    Spacer()
                    if let seeding = seedingStatus, seeding.bytesPerSecond > 0 {
                        Label(SwarmFormat.rate(seeding.bytesPerSecond),
                              systemImage: "arrow.up")
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.footnote)
            }
        } header: {
            Text("Give")
        } footer: {
            Text("Keep both devices on this screen with Wi-Fi turned on — no network or internet is needed. Devices stay awake while a transfer is running.")
        }
    }

    private var shareSubtitle: String {
        if controller.isSharingLibrary {
            return "Visible to nearby devices as “\(ZimSwarmController.defaultShareName)”"
        }
        if !controller.hasShareableFiles {
            return "Nothing to share yet — add Wikipedia or a map first"
        }
        return "Let a friend copy your Wikipedia, maps, and AI model"
    }

    /// Size label for the selected model's shareable file, nil when the
    /// model isn't shareable (not downloaded, or not a single-file GGUF).
    private var shareableModelSizeLabel: String? {
        guard let url = controller.shareableModelFiles().first,
              let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64)
                  .flatMap({ $0 })
        else { return nil }
        return SwarmFormat.bytes(size)
    }

    // MARK: Nearby libraries

    @ViewBuilder
    private var nearbySection: some View {
        Section {
            if manager.discoveredSwarms.isEmpty {
                HStack(spacing: 12) {
                    ProgressView()
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Looking for nearby libraries…")
                        Text("Ask your friend to open Zimfo and turn on “Share my library”. Shares from the LocalSwarm app appear here too.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 4)
            } else {
                ForEach(manager.discoveredSwarms) { swarm in
                    Button {
                        if swarm.locked {
                            enteredPin = ""
                            pinPromptSwarm = swarm
                        } else {
                            Task { await openSwarm(swarm) }
                        }
                    } label: {
                        HStack(spacing: 12) {
                            Image(systemName: swarm.locked
                                  ? "lock.fill"
                                  : "antenna.radiowaves.left.and.right")
                                .font(.title3)
                                .foregroundStyle(.tint)
                                .frame(width: 26)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(swarm.name)
                                    .font(.headline)
                                    .lineLimit(1)
                                Text("\(SwarmFormat.bytes(swarm.totalBytes)) · \(swarm.sourceCount) device\(swarm.sourceCount == 1 ? "" : "s") nearby")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if loadingSwarmID == swarm.id {
                                ProgressView()
                            } else {
                                Image(systemName: "chevron.right")
                                    .font(.footnote)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        } header: {
            Text("Get")
        } footer: {
            Text("Tap a nearby library to choose which files to copy. Files are verified chunk-by-chunk and land in your library automatically.")
        }
    }

    // MARK: Transfers

    @ViewBuilder
    private var transfersSection: some View {
        Section("Transfers") {
            ForEach(manager.pendingReceives) { pending in
                HStack(spacing: 12) {
                    ProgressView()
                    Text("Connecting to \(pending.name)…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            ForEach(downloadsAndReceives) { status in
                SwarmTransferRow(manager: manager, status: status)
            }
            if let summary = controller.lastImportSummary {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(summary)
                        Text("Ready to use in chat — and to share onward.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } icon: {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                .font(.callout)
            }
        }
    }

    private func openSwarm(_ swarm: DiscoveredSwarm, pin: String? = nil) async {
        loadingSwarmID = swarm.id
        let manifest = await manager.fetchManifest(for: swarm, pin: pin)
        loadingSwarmID = nil
        if let manifest {
            selectionRoute = SwarmSelectionRoute(manifest: manifest, swarm: swarm, pin: pin)
        }
    }
}

private struct SwarmSelectionRoute: Identifiable {
    let manifest: SwarmManifest
    let swarm: DiscoveredSwarm
    var pin: String?
    var id: String { manifest.swarmID }
}

/// Pick which of the friend's files to copy. Everything is preselected —
/// the bootstrap case is "give me what you have".
private struct SwarmFileSelectionSheet: View {
    @ObservedObject var manager: SwarmManager
    @Environment(\.dismiss) private var dismiss

    let manifest: SwarmManifest
    let swarm: DiscoveredSwarm
    var pin: String?

    @State private var selected: Set<String>

    init(manager: SwarmManager, manifest: SwarmManifest,
         swarm: DiscoveredSwarm, pin: String?) {
        self.manager = manager
        self.manifest = manifest
        self.swarm = swarm
        self.pin = pin
        _selected = State(initialValue: Set(manifest.files.map(\.id)))
    }

    private var selectedFiles: [SwarmFile] {
        manifest.files.filter { selected.contains($0.id) }
    }
    private var selectedBytes: Int64 {
        selectedFiles.reduce(0) { $0 + $1.sizeBytes }
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(manifest.files) { file in
                        Button {
                            if selected.contains(file.id) { selected.remove(file.id) }
                            else { selected.insert(file.id) }
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: selected.contains(file.id)
                                      ? "checkmark.circle.fill" : "circle")
                                    .font(.title3)
                                    .foregroundStyle(selected.contains(file.id)
                                                     ? Color.accentColor : .secondary)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(file.path)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                    HStack(spacing: 6) {
                                        Text(SwarmFormat.bytes(file.sizeBytes))
                                        if file.path.lowercased().hasSuffix(".gguf") {
                                            Text("· AI model — installs automatically")
                                        } else if !file.path.lowercased().hasSuffix(".zim") {
                                            Text("· not a ZIM — saved but not loaded")
                                        }
                                    }
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                } header: {
                    Text("\(manifest.files.count) file\(manifest.files.count == 1 ? "" : "s") · \(SwarmFormat.bytes(manifest.totalBytes))")
                } footer: {
                    Text("Copies from \(swarm.sourceCount) nearby device\(swarm.sourceCount == 1 ? "" : "s") in parallel. Interrupted copies resume where they left off.")
                }
            }
            .navigationTitle(manifest.name)
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .safeAreaInset(edge: .bottom) {
                Button {
                    manager.startDownload(manifest: manifest,
                                          selecting: selectedFiles,
                                          from: swarm,
                                          pin: pin)
                    dismiss()
                } label: {
                    Text(selected.isEmpty
                         ? "Select at least one file"
                         : "Copy \(selectedFiles.count) file\(selectedFiles.count == 1 ? "" : "s") · \(SwarmFormat.bytes(selectedBytes))")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(selected.isEmpty)
                .padding()
                .background(.bar)
            }
        }
    }
}

/// One live swarm transfer row with pause/resume/cancel.
private struct SwarmTransferRow: View {
    @ObservedObject var manager: SwarmManager
    let status: TransferStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(status.name)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Menu {
                    actions
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .foregroundStyle(.secondary)
                }
                .fixedSize()
            }
            ProgressView(value: status.fractionComplete)
                .tint(status.role == .paused ? .gray : .accentColor)
            HStack {
                Text("\(SwarmFormat.bytes(status.completedBytes)) of \(SwarmFormat.bytes(status.totalBytes))")
                Spacer()
                switch status.role {
                case .paused:
                    Text("Paused")
                case .complete:
                    Text("Importing…")
                default:
                    if status.bytesPerSecond > 0 {
                        Label(SwarmFormat.rate(status.bytesPerSecond), systemImage: "arrow.down")
                    }
                    Label("\(status.connectedPeers)", systemImage: "person.2.fill")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var actions: some View {
        switch status.role {
        case .downloading:
            Button {
                manager.pauseDownload(swarmID: status.swarmID)
            } label: {
                Label("Pause", systemImage: "pause.fill")
            }
            Button(role: .destructive) {
                manager.cancelDownload(swarmID: status.swarmID)
            } label: {
                Label("Cancel & delete partial data", systemImage: "trash")
            }
        case .paused:
            Button {
                manager.resumeDownload(swarmID: status.swarmID)
            } label: {
                Label("Resume", systemImage: "play.fill")
            }
            Button(role: .destructive) {
                manager.cancelDownload(swarmID: status.swarmID)
            } label: {
                Label("Cancel & delete partial data", systemImage: "trash")
            }
        case .complete, .seeding:
            EmptyView()
        }
    }
}

/// Tiny formatting helpers for swarm UI (the engine keeps its own private).
enum SwarmFormat {
    static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }

    static func rate(_ bytesPerSecond: Double) -> String {
        bytes(Int64(bytesPerSecond)) + "/s"
    }
}
