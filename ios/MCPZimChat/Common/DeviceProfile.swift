// SPDX-License-Identifier: MIT
//
// Device-tier tuning knobs. The app runs across a wide range of
// hardware — 6 GB iPhones, 8 GB iPhone Pros, 12 GB future Pros,
// 8/16 GB iPads, 16+ GB Macs — and the "safe" prompt/response/cache
// budgets differ by an order of magnitude. We read
// `ProcessInfo.physicalMemory` once at startup, slot the device into
// a tier, and look up tuning constants from there. Everything stays
// adjustable via UI (Library → Generation) — the tier only sets
// the defaults.

import Foundation

public struct DeviceProfile: Sendable {
    /// How much of a `get_article` response we feed the model per
    /// turn. Bigger = more complete context but slower first-token
    /// and larger KV-cache reservation by MLX.
    public let articleCapKB: Int

    /// Max reply length. Also drives MLX's KV-cache pre-reservation
    /// on stream open — every +128 tokens costs ~25 MB of cache
    /// headroom, which matters on 6 GB devices.
    public let maxReplyTokens: Int

    /// Cap for MLX's per-device GPU/ANE buffer pool. Tight caps keep
    /// steady-state RSS low at a small throughput cost; generous
    /// caps let hot reuse coalesce but risk jetsam on phones.
    public let mlxCacheLimitMB: Int

    /// Whether to ask MLX for a 4-bit groupwise `QuantizedKVCache`
    /// instead of the FP16 `StandardKVCache`. ~4× smaller KV, at a
    /// small (<1% MMLU) quality tax and slightly slower prefill.
    /// On phones the KV is the binding memory constraint; on Macs
    /// we have RAM to spare, so we stay FP16 for max quality.
    public let useQuantizedKVCache: Bool

    public let label: String
}

public extension DeviceProfile {
    /// CLI/test override — when set, `current` returns this instead of the
    /// auto-detected profile, so the Mac harness can run "as a phone"
    /// (phone reply/article budgets + a jetsam memory-budget check).
    nonisolated(unsafe) static var override: DeviceProfile?

    /// The profile in effect — the override if one was set, else the
    /// auto-detected tier for this process.
    static var current: DeviceProfile { override ?? resolved }

    /// Auto-detected for the current process. Macs get the most generous
    /// defaults; iPhones scale down by physical-memory tier.
    private static let resolved: DeviceProfile = {
        #if os(macOS)
        return .mac
        #else
        let bytes = ProcessInfo.processInfo.physicalMemory
        let gb = Double(bytes) / (1024.0 * 1024.0 * 1024.0)
        // iPhone memory tiers (approximate, rounded):
        //   4 GB: iPhone SE 3 / 12 mini / older
        //   6 GB: iPhone 13–16 base / 15 Plus / 16 Plus
        //   8 GB: iPhone 15 Pro / 16 Pro
        //   12+ GB: iPhone 17 Pro / iPad Pro M-series
        switch gb {
        case ..<5:    return .tight    // 4 GB — older phones
        case ..<7:    return .snug     // 6 GB — base iPhone
        case ..<10:   return .balanced // 8 GB — Pro iPhone / M-series iPad
        default:      return .generous // 12+ GB
        }
        #endif
    }()

    /// 4 GB phones — aggressively conservative. Wikipedia drill-ins
    /// still work but the body the model sees is just the lead.
    static let tight = DeviceProfile(
        articleCapKB: 6, maxReplyTokens: 256, mlxCacheLimitMB: 256,
        useQuantizedKVCache: true,    // Re-enabled 2026-04-21 after plumbing a quantized donor tuple through `Gemma4Attention.sharedKV` — the FP16 donor-dequant transients that previously caused mid-Kokoro jetsams are gone.
        label: "tight (≈4 GB)"
    )
    /// 6 GB base iPhones — the default "mobile" target.
    static let snug = DeviceProfile(
        articleCapKB: 12, maxReplyTokens: 384, mlxCacheLimitMB: 384,
        useQuantizedKVCache: true,    // Re-enabled 2026-04-21 after plumbing a quantized donor tuple through `Gemma4Attention.sharedKV` — the FP16 donor-dequant transients that previously caused mid-Kokoro jetsams are gone.
        label: "snug (≈6 GB)"
    )
    /// 8 GB Pro iPhones & M-series iPads.
    static let balanced = DeviceProfile(
        articleCapKB: 16, maxReplyTokens: 320, mlxCacheLimitMB: 384,
        useQuantizedKVCache: true,    // Re-enabled 2026-04-21 after plumbing a quantized donor tuple through `Gemma4Attention.sharedKV` — the FP16 donor-dequant transients that previously caused mid-Kokoro jetsams are gone.
        label: "balanced (≈8 GB)"
    )
    /// 12 GB+ iPhones / iPads. On iPhone 17 Pro Max the *process* cap
    /// with `increased-memory-limit` is only 6144 MB regardless of how
    /// much physical RAM the device ships with, so the cache cap here
    /// is only slightly larger than `balanced`. Reply tokens kept
    /// short — a 500-token reply takes ~60 s for Kokoro to speak,
    /// which kills the voice-chat flow.
    static let generous = DeviceProfile(
        articleCapKB: 24, maxReplyTokens: 320, mlxCacheLimitMB: 448,
        useQuantizedKVCache: true,    // Re-enabled 2026-04-21 after plumbing a quantized donor tuple through `Gemma4Attention.sharedKV` — the FP16 donor-dequant transients that previously caused mid-Kokoro jetsams are gone.
        label: "generous (≈12 GB)"
    )
    /// Development machines — not trying to avoid jetsam on macOS;
    /// the OS swaps and we only care about responsiveness.
    static let mac = DeviceProfile(
        articleCapKB: 24, maxReplyTokens: 512, mlxCacheLimitMB: 512,
        useQuantizedKVCache: false,
        label: "macOS"
    )
}

/// Plain, non-actor model-selection rules — deliberately NOT on the
/// `@MainActor @Observable` `ChatSession` (a `nonisolated` static on an
/// `@Observable` class is a Swift 5.9 sharp edge that can trip a runtime
/// isolation assertion on launch). Kept in one place so the default pick,
/// the load guard, and the picker all agree.
enum ModelCatalog {
    /// The llama.cpp GGUF models known to run well on-device. The model
    /// picker puts these first and labels the rest "Experimental" so a user
    /// doesn't pick a model that doesn't behave (e.g. the MLX snapshots that
    /// are download-only or blocked on iOS).
    static let knownGoodModelIDs: Set<String> = [
        "bonsai-27b-q1-gguf",
        "lfm2.5-8b-a1b-q3km-gguf-ft",
        "gemma3-4b-it-q4km-gguf-ft",
    ]

    /// The model to default to on a fresh install. The user-verified,
    /// on-device proven model is LFM2.5-8B, so it is the default wherever
    /// the budget holds it; the smaller Gemma 3 4B FT is only the fallback
    /// for a tight (≈4 GB) phone where LFM2.5 can't fit. A bigger model is
    /// always selectable from the picker.
    static func recommendedModelID() -> String {
        if modelFitsDevice(3700) {  // LFM2.5-8B
            return "lfm2.5-8b-a1b-q3km-gguf-ft"
        }
        return "gemma3-4b-it-q4km-gguf-ft"
    }

    /// Whether a model's advertised resident footprint plausibly fits this
    /// device's process budget. Used to pick a safe default and to flag the
    /// picker entry as a risk.
    ///
    /// iOS caps a process at ~6 GB (6144 MB with `increased-memory-limit`)
    /// regardless of physical RAM, and the advertised peak underestimates
    /// the load-time spike. 2026-09-06 device crash: Bonsai-27B (5.5 GB
    /// advertised, ~3.8 GB weights) passed the 5.5 GB budget on an 8 GB+
    /// phone and was jetsamed because the ~1.8 GB semantic reranker +
    /// speech recognizer were loading at the SAME time as the model. We
    /// now defer those to `postSetupPrewarm` (after the model is resident),
    /// so the competing footprint is gone; Bonsai fits an 8 GB+ phone
    /// again. Budget stays 5.5 GB — Bonsai (5.5) is allowed on 8 GB+
    /// devices, rejected on snug (where `min(gb−2.5,·)` is < 5.5); LFM2.5
    /// (3.7) and Gemma 3 4B FT (3.2) fit everywhere. macOS swaps.
    static func modelFitsDevice(_ approximateMemoryMB: Int) -> Bool {
        #if os(macOS)
        return true
        #else
        let gb = Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000
        let budgetGB = min(gb - 2.5, 5.5)
        return Double(approximateMemoryMB) / 1000.0 <= budgetGB
        #endif
    }
}
