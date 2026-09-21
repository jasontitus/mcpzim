import Foundation

/// Removes discontinuities at the edges of independently synthesized chunks.
/// Only the outer 5 ms are tapered; duration and interior speech are unchanged.
public enum SpeechPCMEdges {
    public static func taper(_ samples: UnsafeMutableBufferPointer<Float>, sampleRate: Int) {
        guard sampleRate > 0, !samples.isEmpty else { return }
        if samples.count == 1 { samples[0] = 0; return }
        let count = min(max(2, sampleRate / 200), samples.count / 2)
        for index in 0..<count {
            let gain: Float = count == 1 ? 0
                : Float(0.5 - 0.5 * cos(Double.pi * Double(index) / Double(count - 1)))
            samples[index] *= gain
            samples[samples.count - 1 - index] *= gain
        }
    }
}
