//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Conv1d with weight normalization
class ConvWeighted: Module {
  var weightG: MLXArray
  var weightV: MLXArray
  var bias: MLXArray?
  /// Normalized conv weight, computed ONCE at init: weightV/weightG never
  /// change after construction in this inference-only port, but the full
  /// L2-norm reduction + normalize/scale used to be rebuilt on EVERY
  /// forward call — per audio frame for every conv in the decoder/
  /// generator hot path. The bias is likewise pre-reshaped to its
  /// broadcast form instead of re-reshaped per call.
  private let normalizedWeight: MLXArray

  let stride: Int
  let padding: Int
  let dilation: Int
  let outputPadding: Int
  let groups: Int

  init(
    weightG: MLXArray,
    weightV: MLXArray,
    bias: MLXArray?,
    stride: Int = 1,
    padding: Int = 1,
    dilation: Int = 1,
    outputPadding: Int = 0,
    groups: Int = 1
  ) {
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.outputPadding = outputPadding
    self.groups = groups

    self.weightG = weightG
    self.weightV = weightV
    self.bias = bias?.reshaped([1, 1, -1])
    self.normalizedWeight = Self.weightNorm(weightV: weightV, weightG: weightG, dim: 0)

    super.init()
  }

  private static func computeNorm(
    x: MLXArray,
    p: Int,
    dim: [Int]? = nil,
    keepdim: Bool = false
  ) -> MLXArray {
    guard p == 1 || p == 2 else {
      fatalError("Only p-norms with p of 1 or 2 are supported")
    }

    let dimensions: [Int]
    if let dim = dim {
      dimensions = dim
    } else {
      dimensions = Array(0 ..< x.ndim)
    }

    if p == 1 {
      // L1 norm
      return MLX.sum(MLX.abs(x), axes: dimensions, keepDims: keepdim)
    } else {
      // L2 norm
      return MLX.sqrt(MLX.sum(x * x, axes: dimensions, keepDims: keepdim))
    }
  }

  private static func weightNorm(
    weightV: MLXArray,
    weightG: MLXArray,
    dim: Int? = nil
  ) -> MLXArray {
    let rank = weightV.shape.count

    var axes: [Int]

    if let dim = dim {
      var adjustedDim = dim
      if dim < 0 {
        adjustedDim += rank
      }

      axes = Array(0 ..< rank)
      if adjustedDim != -1 {
        axes.removeAll(where: { $0 == adjustedDim })
      }
    } else {
      axes = Array(0 ..< rank)
    }

    let normV = computeNorm(x: weightV, p: 2, dim: axes, keepdim: true)

    // Add epsilon for numerical stability
    let normalizedWeight = weightV / (normV + 1e-7)
    return normalizedWeight * weightG
  }

  public func callAsFunction(_ x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
    let weight = normalizedWeight

    func applyConv(x: MLXArray, weightToUse: MLXArray) -> MLXArray {
      let result = conv(
        x,
        weightToUse,
        self.stride,
        padding,
        dilation,
        groups,
        .default
      )

      if let bias = bias {
        return result + bias
      }
      return result
    }

    if x.shape.last == weight.shape.last || groups > 1 {
      return applyConv(x: x, weightToUse: weight)
    } else {
      return applyConv(x: x, weightToUse: weight.transposed())
    }
  }
  
  public func callAsFunction(_ x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
    let weight = normalizedWeight

    func applyConv(x: MLXArray, weightToUse: MLXArray) -> MLXArray {
      let result = conv(
        x,
        weightToUse,
        self.stride,
        padding,
        dilation,
        outputPadding,
        groups,
        .default
      )

      if let bias = bias {
        return result + bias
      }
      return result
    }

    if x.shape.last == weight.shape.last || groups > 1 {
      return applyConv(x: x, weightToUse: weight)
    } else {
      return applyConv(x: x, weightToUse: weight.transposed())
    }
  }
}
