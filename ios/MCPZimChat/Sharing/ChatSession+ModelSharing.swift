// SPDX-License-Identifier: MIT

import Foundation

/// Nearby Sharing's model hooks. Only single-file GGUF models
/// (`LlamaCppProvider`) are shareable today — the shipping fine-tuned
/// LFM2.5 and every other llama.cpp model qualify. MLX models are
/// multi-file HuggingFace snapshots and stay download-only for now.
extension ChatSession {
    /// The model files worth offering a friend: every llama.cpp GGUF this
    /// device has fully downloaded (byte-validated), not just the one
    /// currently selected. Like the ZIM library, the friend gets the whole
    /// working set — and if they have no model yet, the receive path adopts
    /// the first one and selects it.
    public func shareableModelFiles() -> [URL] {
        models.compactMap { model -> URL? in
            guard let provider = model as? LlamaCppProvider else { return nil }
            return provider.shareableGGUFURL
        }
    }

    /// Offer a received file to every registered model provider. Returns
    /// true when one claims it — the file has then been moved into that
    /// provider's cache slot, exactly where its own downloader would have
    /// put it. If this device has no working model yet, the adopted model is
    /// selected and loaded immediately — **provided it plausibly fits this
    /// device** — so a fresh install that just received ZIMs + model from a
    /// friend is ready to chat with zero internet. A shared model that's too
    /// big for this phone is still adopted, but isn't auto-selected (the
    /// capacity guard would reject it and strand a smaller fitting model
    /// that was adopted later in the same batch).
    public func importSharedModelFile(at url: URL) async -> Bool {
        for model in models {
            guard let provider = model as? LlamaCppProvider,
                  provider.adoptSharedGGUF(at: url) else { continue }
            debug("adopted shared model \(url.lastPathComponent) → \(provider.id)",
                  category: "Library")
            switch modelState {
            case .ready:
                // Already have a working model; leave the selection alone.
                break
            case .notLoaded, .failed:
                // No working model on this device — switch to the one that
                // just arrived if it fits, so the friend-bootstrap leaves a
                // fresh install ready to chat. Fire-and-forget so the
                // caller's import loop isn't stalled behind a multi-second
                // model load.
                if ModelCatalog.modelFitsDevice(provider.approximateMemoryMB) {
                    let modelId = provider.id
                    Task { [weak self] in
                        await self?.select(modelId: modelId)
                    }
                }
            case .loading, .downloading, .waitingForNetwork:
                break
            }
            return true
        }
        return false
    }
}
