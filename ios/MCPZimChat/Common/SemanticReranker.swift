// SPDX-License-Identifier: MIT
//
// Semantic reranker for MCP `search` hits. Uses Apple's
// `NLContextualEmbedding` (available macOS 14+ / iOS 17+, ANE-
// accelerated, ~100 MB on-device model that ships with the OS) to
// reorder a BM25 candidate list by cosine similarity to the query.
//
// BM25 keyword overlap alone sends the wrong article to the top for
// plenty of natural-language queries. Example from our own usage:
// "quantum computing's effect on encryption" → BM25 picks
// `Crypto-shredding` at #1 (because "encryption" is dense) while the
// actually-correct answer is `Post-quantum cryptography` at #5.
// A sentence-level embedding comparison moves the right article up.
//
// The reranker is best-effort: if the framework's assets haven't
// downloaded yet, or the query language isn't supported, we fall
// through to the original BM25 order instead of erroring.

import Foundation
import NaturalLanguage
import MCPZimKit

actor SemanticReranker {
    /// Singleton so the embedding model loads once per process.
    static let shared = SemanticReranker()

    private var loadAttempted = false
    private var embedding: NLContextualEmbedding?
    /// In-memory cache of per-hit embeddings keyed by `zim:path`.
    /// Snippets only change when the ZIM rebuilds so this is safe to
    /// keep for the app's lifetime.
    private var cache: [String: [Double]] = [:]

    private init() {}

    /// Ensure the embedding model is loaded. If the framework needs
    /// to fetch assets we kick that off but don't block reranker
    /// calls on it — the next call after assets land will use it.
    func loadIfNeeded() async {
        if loadAttempted { return }
        loadAttempted = true
        guard let candidate = NLContextualEmbedding(language: .english) else {
            return
        }
        if !candidate.hasAvailableAssets {
            // Fire the instance-level `requestAssets()` helper —
            // returns an `NLContextualEmbedding.AssetsResult` that
            // we don't need; we just need the side-effect of the
            // download.
            do {
                _ = try await candidate.requestAssets()
            } catch {
                return
            }
        }
        do {
            try candidate.load()
            self.embedding = candidate
        } catch {
            // Load failed — reranker stays disabled, caller gets
            // BM25 order back unchanged.
        }
    }

    /// Rerank `hits` by semantic proximity to `query`. Returns the
    /// hits unchanged when the embedding model isn't available.
    func rerank(query: String, hits: [SearchHitResult]) async -> [SearchHitResult] {
        await loadIfNeeded()
        guard let embedding else { return hits }
        guard let queryVec = embed(query, with: embedding) else { return hits }
        struct Scored { let hit: SearchHitResult; let score: Double }
        var scored: [Scored] = []
        for hit in hits {
            let key = "\(hit.zim):\(hit.path)"
            let vec: [Double]
            if let cached = cache[key] {
                vec = cached
            } else {
                // Embed on `title + snippet` — snippet alone can be
                // too long, title alone too sparse. Capping keeps
                // embedding cost bounded regardless of ZIM quirks.
                let text = [hit.title, hit.snippet]
                    .filter { !$0.isEmpty }
                    .joined(separator: ". ")
                let capped = String(text.prefix(512))
                guard let fresh = embed(capped, with: embedding) else {
                    scored.append(Scored(hit: hit, score: -.infinity))
                    continue
                }
                cache[key] = fresh
                vec = fresh
            }
            scored.append(Scored(hit: hit, score: cosine(queryVec, vec)))
        }
        // Stable sort on score DESC — ties keep BM25 order.
        let ranked = scored.enumerated().sorted { a, b in
            if a.element.score != b.element.score { return a.element.score > b.element.score }
            return a.offset < b.offset
        }
        return ranked.map { $0.element.hit }
    }

    // MARK: - Embedding

    /// Mean-pool the token-level vectors `NLContextualEmbedding`
    /// returns into one sentence embedding.
    nonisolated private func embed(_ text: String, with embedding: NLContextualEmbedding) -> [Double]? {
        guard !text.isEmpty else { return nil }
        guard let result = try? embedding.embeddingResult(for: text, language: .english) else {
            return nil
        }
        var summed: [Double] = []
        var count = 0
        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vec, _ in
            if summed.isEmpty {
                summed = vec
            } else if summed.count == vec.count {
                for i in 0..<summed.count { summed[i] += vec[i] }
            }
            count += 1
            return true
        }
        guard count > 0 else { return nil }
        return summed.map { $0 / Double(count) }
    }

    nonisolated private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = normA.squareRoot() * normB.squareRoot()
        return denom == 0 ? 0 : dot / denom
    }
}
