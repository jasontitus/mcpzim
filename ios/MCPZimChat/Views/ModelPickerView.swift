// SPDX-License-Identifier: MIT

import SwiftUI

/// Model chooser. Lists every provider with its download size and resident
/// footprint, sorts the capacity-recommended one to the top, and flags models
/// that likely won't fit this device so a user doesn't pick one that OOMs at
/// load. Picking a model auto-downloads + auto-loads it via `session.select`.
struct ModelPickerView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        Menu {
            Picker("Model", selection: Binding(
                get: { session.selectedModel.id },
                set: { id in Task { await session.select(modelId: id) } }
            )) {
                ForEach(orderedModels, id: \.id) { model in
                    Label(label(for: model), systemImage: icon(for: model))
                        .tag(model.id)
                }
            }
            // Picking a model auto-loads it via `session.select(...)`,
            // so no explicit "Load" action here. The ChatView header
            // still surfaces a retry button when a load fails.
        } label: {
            HStack(spacing: 4) {
                Text(session.selectedModel.displayName)
                    .lineLimit(1)
                Image(systemName: "chevron.down")
            }
            .font(.footnote)
        }
    }

    /// Recommended (capacity-safe) model first, then the known-good llama.cpp
    /// models (Bonsai / LFM / tuned Gemma), then everything else (labeled
    /// "Experimental" — MLX snapshots, Apple FM, mock). Too-big choices are
    /// pushed down and flagged.
    private var orderedModels: [any ModelProvider] {
        let recommended = ModelCatalog.recommendedModelID()
        return session.models.sorted { a, b in
            let ra = a.id == recommended
            let rb = b.id == recommended
            if ra != rb { return ra }
            let ka = ModelCatalog.knownGoodModelIDs.contains(a.id)
            let kb = ModelCatalog.knownGoodModelIDs.contains(b.id)
            if ka != kb { return ka }
            let fa = ModelCatalog.modelFitsDevice(a.approximateMemoryMB)
            let fb = ModelCatalog.modelFitsDevice(b.approximateMemoryMB)
            if fa != fb { return fa }
            return a.approximateMemoryMB < b.approximateMemoryMB
        }
    }

    private func label(for model: any ModelProvider) -> String {
        var parts = [model.displayName]
        if let size = downloadSizeLabel(model) { parts.append(size) }
        parts.append("~\(model.approximateMemoryMB) MB")
        var text = parts.joined(separator: " · ")
        if model.id == ModelCatalog.recommendedModelID() {
            text += " · Recommended"
        } else if ModelCatalog.knownGoodModelIDs.contains(model.id) {
            text += " · Works well"
        } else {
            text += " · Experimental"
        }
        if !ModelCatalog.modelFitsDevice(model.approximateMemoryMB) {
            text += " · Too large for this device"
        }
        return text
    }

    private func downloadSizeLabel(_ model: any ModelProvider) -> String? {
        guard let llama = model as? LlamaCppProvider,
              let bytes = llama.expectedGGUFBytes, bytes > 0 else { return nil }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private func icon(for model: any ModelProvider) -> String {
        if model is MockProvider { return "hammer" }
        return "cpu"
    }
}

#Preview {
    ModelPickerView().environment(ChatSession())
}
