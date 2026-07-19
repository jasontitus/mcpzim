// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "KokoroSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15)
  ],
  products: [
    .library(
      name: "KokoroSwift",
      type: .dynamic,
      targets: ["KokoroSwift"]
    ),
  ],
  dependencies: [
    // Upstream pins exact 0.30.2 which conflicts with
    // Swift-gemma4-core's 0.31+ requirement. Loosen to a range —
    // CastCircle does the same in its vendored copy and the call
    // sites are compatible from 0.29 onward.
    // Same-identity requirement: every mlx-swift consumer in the workspace
    // must point at the SAME url. Tracks mlx-swift-lm's Prism fork pin
    // (1-bit quant kernels for Bonsai) — revert both together.
    .package(url: "https://github.com/PrismML-Eng/mlx-swift", branch: "prism"),
    // .package(url: "https://github.com/mlalma/eSpeakNGSwift", from: "1.0.1"),
    .package(url: "https://github.com/mlalma/MisakiSwift", from: "1.0.4"),
    .package(url: "https://github.com/mlalma/MLXUtilsLibrary.git", from: "0.0.6")
  ],
  targets: [
    .target(
      name: "KokoroSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        // .product(name: "eSpeakNGLib", package: "eSpeakNGSwift"),
        .product(name: "MisakiSwift", package: "MisakiSwift"),
        .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary")
      ],
      resources: [
       .copy("../../Resources/")
      ]
    ),
    .testTarget(
      name: "KokoroSwiftTests",
      dependencies: ["KokoroSwift"]
    ),
  ]
)
