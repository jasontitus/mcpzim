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

    public let label: String
}

public extension DeviceProfile {
    /// Resolved for the current process. Macs get the most generous
    /// defaults; iPhones scale down by physical-memory tier.
    static let current: DeviceProfile = {
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
        label: "tight (≈4 GB)"
    )
    /// 6 GB base iPhones — the default "mobile" target.
    static let snug = DeviceProfile(
        articleCapKB: 12, maxReplyTokens: 384, mlxCacheLimitMB: 384,
        label: "snug (≈6 GB)"
    )
    /// 8 GB Pro iPhones & M-series iPads.
    static let balanced = DeviceProfile(
        articleCapKB: 16, maxReplyTokens: 512, mlxCacheLimitMB: 512,
        label: "balanced (≈8 GB)"
    )
    /// 12 GB+ iPhones / iPads.
    static let generous = DeviceProfile(
        articleCapKB: 24, maxReplyTokens: 512, mlxCacheLimitMB: 640,
        label: "generous (≈12 GB)"
    )
    /// Development machines — not trying to avoid jetsam on macOS;
    /// the OS swaps and we only care about responsiveness.
    static let mac = DeviceProfile(
        articleCapKB: 24, maxReplyTokens: 512, mlxCacheLimitMB: 512,
        label: "macOS"
    )
}
