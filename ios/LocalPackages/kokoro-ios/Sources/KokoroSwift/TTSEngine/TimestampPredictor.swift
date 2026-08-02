//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXUtilsLibrary

class TimestampPredictor {
  private init() {}
  
  static func preditTimestamps(tokens: [MToken], predictionDuration: MLXArray) {
    /*
     Multiply by 600 to go from pred_dur frames to sample_rate 24000.
     Equivalent to dividing pred_dur frames by 40 to get timestamp in seconds.
     We will count nice round half-frames, so the divisor is 80.
    */
    guard tokens.count > 0, predictionDuration.shape[0] >= 3 else {
      // We expect at least 3: <bos>, token, <eos>
      return
    }
              
    let magicDivisor: Float = 80.0

    // ONE bulk host sync for the whole duration vector — the per-token
    // `.item()` and slice+`sum().item()` calls each forced a GPU→CPU
    // evaluation of the full duration graph, once per token, scaling
    // stalls with utterance length. All arithmetic below runs on the
    // native array.
    let dur = predictionDuration.asType(.float32).asArray(Float.self)
    let n = dur.count

    // We track 2 counts, measured in half-frames: (left, right)
    // This way we can cut space characters in half
    // TO_DO: Is -3 an appropriate offset?
    var left: Float = 0
    var right: Float = 2 * max(0, dur[0] - 3)
    left = right

    // Updates:
    // left = right + (2 * token_dur) + space_dur
    // right = left + space_dur
    var i = 1
    for t in tokens {
      guard i < n - 1 else {
        break
      }

      if t.phonemes == nil {
        if !t.whitespace.isEmpty {
          i += 1
          left = right + dur[i]
          right = left + dur[i]
          i += 1
        }
        continue
      }

      let j = i + t.phonemes!.count
      if j >= n {
        break
      }

      t.start_ts = Double(left / magicDivisor)
      let tokenDuration: Float = dur[i..<j].reduce(0, +)
      let spaceDuration: Float = t.whitespace.isEmpty ? 0.0 : dur[j]
      left = right + (2.0 * tokenDuration) + spaceDuration
      t.end_ts = Double(left / magicDivisor)
      right = left + spaceDuration
      i = j + (t.whitespace.isEmpty ? 0 : 1)
    }
  }
}
