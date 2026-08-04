// SPDX-License-Identifier: MIT

import Foundation

/// Nearby Sharing's model hooks. Only single-file GGUF models
/// (`LlamaCppProvider`) are shareable today — the shipping fine-tuned
/// LFM2.5 and every other llama.cpp model qualify. MLX models are
/// multi-file HuggingFace snapshots and stay download-only for now.
extension ChatSession {
    /// The model file worth offering a friend: the *selected* model's
    /// byte-validated GGUF — i.e. the model this device actually chats
    /// with, so the recipient ends up with a setup known to work.
    public func shareableModelFiles() -> [URL] {
        guard let provider = selectedModel as? LlamaCppProvider,
              let url = provider.shareableGGUFURL else { return [] }
        return [url]
    }

    /// Offer a received file to every registered model provider. Returns
    /// true when one claims it — the file has then been moved into that
    /// provider's cache slot, exactly where its own downloader would have
    /// put it. If this device has no working model yet, the adopted one is
    /// selected and loaded immediately: a fresh install that just received
    /// ZIMs + model from a friend is ready to chat with zero internet.
    public func importSharedModelFile(at url: URL) async -> Bool {
        for model in models {
            guard let provider = model as? LlamaCppProvider,
                  provider.adoptSharedGGUF(at: url) else { continue }
            debug("adopted shared model \(url.lastPathComponent) → \(provider.id)",
                  category: "Library")
            switch modelState {
            case .notLoaded, .failed:
                // No working model on this device — switch to (or retry)
                // the one that just arrived. Fire-and-forget so the caller's
                // import loop isn't stalled behind a multi-second model
                // load; an in-flight load/download and a ready model are
                // left alone.
                let modelId = provider.id
                Task { [weak self] in
                    await self?.select(modelId: modelId)
                }
            case .loading, .downloading, .ready:
                break
            }
            return true
        }
        return false
    }
}
