// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LocalSwarmEngine",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "LocalSwarmEngine", targets: ["LocalSwarmEngine"]),
    ],
    targets: [
        .target(name: "LocalSwarmEngine"),
        .testTarget(name: "LocalSwarmEngineTests", dependencies: ["LocalSwarmEngine"]),
    ]
)
