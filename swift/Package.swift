// swift-tools-version: 5.9
// SPDX-License-Identifier: MIT
//
// MCPZimKit — Swift port of mcpzim's routing + geocoder + MCP adapter for iOS
// and macOS. Lets an iOS chat app (e.g. one built around Swift-Gemma4-Core or
// Google's AI Edge Gallery with LiteRT-LM) expose ZIM-backed tools to its
// on-device LLM either as an in-process Swift API or as a fully compliant MCP
// server reachable over localhost / LAN.

import PackageDescription

let package = Package(
    name: "MCPZimKit",
    platforms: [
        // Matches Swift-Gemma4-Core and the current Kiwix-iOS target.
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        // Pure-Swift algorithms + protocols + JSON-RPC tool adapter. No
        // external dependencies so it drops cleanly into any iOS app.
        .library(name: "MCPZimKit", targets: ["MCPZimKit"]),
    ],
    targets: [
        .target(name: "MCPZimKit", dependencies: []),
        .testTarget(
            name: "MCPZimKitTests",
            dependencies: ["MCPZimKit"]
        ),
    ]
)
