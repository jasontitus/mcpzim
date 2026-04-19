// SPDX-License-Identifier: MIT

import SwiftUI

struct ModelPickerView: View {
    @Environment(ChatSession.self) private var session

    var body: some View {
        Menu {
            Picker("Model", selection: Binding(
                get: { session.selectedModel.id },
                set: { id in Task { await session.select(modelId: id) } }
            )) {
                ForEach(session.models, id: \.id) { model in
                    Label("\(model.displayName) (~\(model.approximateMemoryMB) MB)",
                          systemImage: icon(for: model))
                        .tag(model.id)
                }
            }
            Divider()
            Button("Load model") {
                Task { await session.loadSelectedModel() }
            }
            .disabled(session.modelState == .ready || session.modelState == .loading)
        } label: {
            HStack(spacing: 4) {
                Text(session.selectedModel.displayName)
                    .lineLimit(1)
                Image(systemName: "chevron.down")
            }
            .font(.footnote)
        }
    }

    private func icon(for model: any ModelProvider) -> String {
        if model is MockProvider { return "hammer" }
        return "cpu"
    }
}

#Preview {
    ModelPickerView().environment(ChatSession())
}
