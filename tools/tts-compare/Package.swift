// swift-tools-version: 5.9
import Foundation
import PackageDescription

// Pin the experimental checkout outside the app. The app's vendor is the default.
let fluidPath = ProcessInfo.processInfo.environment["TTS_FLUIDAUDIO_PATH"]
    ?? "../../ios/LocalPackages/FluidAudio"
let hasInflect = FileManager.default.fileExists(
    atPath: fluidPath + "/Sources/FluidAudio/TTS/Inflect/InflectManager.swift")
let kokoroPath = ProcessInfo.processInfo.environment["TTS_KOKORO_PIPELINE_PATH"]
var dependencies: [Package.Dependency] = [
    .package(name: "FluidAudio", path: fluidPath), .package(path: "../../swift")]
var products: [Target.Dependency] = [
    .product(name: "FluidAudio", package: "FluidAudio"),
    .product(name: "MCPZimKit", package: "swift")]
var settings: [SwiftSetting] = hasInflect ? [.define("HAS_INFLECT")] : []
if let kokoroPath {
    dependencies.append(.package(name: "KokoroPipeline", path: kokoroPath))
    products.append(.product(name: "KokoroPipeline", package: "KokoroPipeline"))
    settings.append(.define("HAS_KOKORO_PIPELINE"))
}
let package = Package(
    name: "ZimfoTTSCompare",
    platforms: [.macOS(.v14)],
    dependencies: dependencies,
    targets: [.executableTarget(
        name: "ZimfoTTSCompare",
        dependencies: products,
        swiftSettings: settings)])
