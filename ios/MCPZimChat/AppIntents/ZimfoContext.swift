// SPDX-License-Identifier: MIT
//
// Persistent context shared between the app's UI and the Siri / Shortcuts
// App Intents. Stored as a small JSON file in Application Support so it
// survives app termination — Siri may dispatch an intent seconds or hours
// after the route was planned, and we want "how much longer" to still work.

import Foundation

/// Serializable snapshot of an active driving route, enough to compute
/// remaining time + distance + the nearest next turn without needing the
/// original graph.
public struct ActiveRoute: Codable, Sendable {
    public struct Coordinate: Codable, Sendable {
        public let lat: Double
        public let lon: Double
        public init(lat: Double, lon: Double) {
            self.lat = lat; self.lon = lon
        }
    }

    public let startedAt: Date
    public let origin: Coordinate
    public let destination: Coordinate
    public let originName: String
    public let destinationName: String
    public let zim: String
    public let totalDurationSeconds: Double
    public let totalDistanceMeters: Double
    /// Polyline points `[Coordinate]`. Same data the tool returned; we
    /// keep it so follow-up "how much longer" can snap the user's current
    /// lat/lon onto the line and compute a progress percentage.
    public let polyline: [Coordinate]
    /// Parallel to `polyline`; the cumulative path distance *up to* each
    /// index. `cumulativeDistanceMeters.last == totalDistanceMeters`.
    public let cumulativeDistanceMeters: [Double]
    public let turnByTurn: [String]

    public init(
        startedAt: Date,
        origin: Coordinate, destination: Coordinate,
        originName: String, destinationName: String,
        zim: String,
        totalDurationSeconds: Double, totalDistanceMeters: Double,
        polyline: [Coordinate],
        cumulativeDistanceMeters: [Double],
        turnByTurn: [String]
    ) {
        self.startedAt = startedAt
        self.origin = origin
        self.destination = destination
        self.originName = originName
        self.destinationName = destinationName
        self.zim = zim
        self.totalDurationSeconds = totalDurationSeconds
        self.totalDistanceMeters = totalDistanceMeters
        self.polyline = polyline
        self.cumulativeDistanceMeters = cumulativeDistanceMeters
        self.turnByTurn = turnByTurn
    }
}

public actor ZimfoContext {
    public static let shared = ZimfoContext()

    private let storeURL: URL
    private var _activeRoute: ActiveRoute?
    private var _lastLocation: ActiveRoute.Coordinate?

    public init(storeURL: URL = ZimfoContext.defaultStoreURL()) {
        self.storeURL = storeURL
        let loaded = Self.load(from: storeURL)
        self._activeRoute = loaded?.activeRoute
        self._lastLocation = loaded?.lastLocation
    }

    public var activeRoute: ActiveRoute? { _activeRoute }
    public var lastLocation: ActiveRoute.Coordinate? { _lastLocation }

    /// Monotonic change counter for `activeRoute`. Lets extensions (which
    /// cannot add stored properties) memoize shapes derived from the
    /// route — e.g. the MCP `RouteSnapshot` with its polyline copy —
    /// without this file importing their frameworks.
    public private(set) var routeVersion: UInt64 = 0

    public func setActiveRoute(_ route: ActiveRoute) {
        _activeRoute = route
        routeVersion &+= 1
        persistRoute()
    }

    public func clearActiveRoute() {
        _activeRoute = nil
        routeVersion &+= 1
        persistRoute()
    }

    public func updateLastLocation(_ coord: ActiveRoute.Coordinate) {
        _lastLocation = coord
        persistLocation()
    }

    // MARK: - Disk

    private struct Snapshot: Codable {
        var activeRoute: ActiveRoute?
        var lastLocation: ActiveRoute.Coordinate?
    }

    public static func defaultStoreURL() -> URL {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask,
                                   appropriateFor: nil,
                                   create: true))
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = support.appendingPathComponent("Zimfo", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("context.json")
    }

    private static func load(from url: URL) -> Snapshot? {
        var snap: Snapshot?
        if let data = try? Data(contentsOf: url) {
            snap = try? JSONDecoder().decode(Snapshot.self, from: data)
        }
        // The coordinate-only sidecar (see `persistLocation`) is written on
        // every location update, so when present it is at least as fresh as
        // the combined snapshot's copy. Old installs simply lack the file.
        if let data = try? Data(contentsOf: Self.locationURL(for: url)),
           let coord = try? JSONDecoder().decode(ActiveRoute.Coordinate.self, from: data)
        {
            var merged = snap ?? Snapshot(activeRoute: nil, lastLocation: nil)
            merged.lastLocation = coord
            snap = merged
        }
        return snap
    }

    private static func locationURL(for storeURL: URL) -> URL {
        storeURL.deletingPathExtension().appendingPathExtension("location.json")
    }

    /// Route writes re-encode the full snapshot (polyline can be thousands
    /// of points) but only happen on set/clear. Location updates land on
    /// every "how much longer?" / "what's around here?" intent, so they go
    /// to a tiny sidecar file instead of re-serialising the whole route
    /// for a one-field change.
    private func persistRoute() {
        let snap = Snapshot(activeRoute: _activeRoute, lastLocation: _lastLocation)
        guard let data = try? JSONEncoder().encode(snap) else { return }
        try? data.write(to: storeURL, options: [.atomic])
    }

    private func persistLocation() {
        guard let coord = _lastLocation,
              let data = try? JSONEncoder().encode(coord) else { return }
        try? data.write(to: Self.locationURL(for: storeURL), options: [.atomic])
    }
}

// MARK: - Along-route progress math

public enum RouteProgress {
    /// Snap `current` to the nearest point on the polyline and return
    /// the remaining distance + duration + the name of the next
    /// turn-by-turn leg.
    public static func remaining(
        for route: ActiveRoute,
        current: ActiveRoute.Coordinate
    ) -> (remainingMeters: Double, remainingSeconds: Double) {
        guard route.polyline.count >= 2 else {
            return (route.totalDistanceMeters, route.totalDurationSeconds)
        }

        // Find nearest polyline vertex (good enough for car-scale routes —
        // a finer snap-to-segment approximation barely moves the answer).
        // Argmin only, so equirectangular squared distance with the fixed
        // origin's cos hoisted — no per-vertex trig (mirrors
        // RouteSnapshot.remaining in MCPZimKit).
        var bestIdx = 0
        var bestD = Double.infinity
        let cosLat = cos(current.lat * .pi / 180)
        for (i, p) in route.polyline.enumerated() {
            let dLat = p.lat - current.lat
            let dLon = (p.lon - current.lon) * cosLat
            let d = dLat * dLat + dLon * dLon
            if d < bestD { bestD = d; bestIdx = i }
        }
        let covered = bestIdx < route.cumulativeDistanceMeters.count
            ? route.cumulativeDistanceMeters[bestIdx] : 0
        let remainingMeters = max(0, route.totalDistanceMeters - covered)
        let fraction = route.totalDistanceMeters > 0
            ? remainingMeters / route.totalDistanceMeters : 0
        let remainingSeconds = route.totalDurationSeconds * fraction
        return (remainingMeters, remainingSeconds)
    }

    /// Lightweight haversine — kept local so this file doesn't pull the
    /// full `MCPZimKit.haversineMeters` dependency (Intents target in the
    /// future may not link MCPZimKit).
    public static func haversineMetersApprox(
        _ lat1: Double, _ lon1: Double, _ lat2: Double, _ lon2: Double
    ) -> Double {
        let R = 6_371_000.0
        let rlat1 = lat1 * .pi / 180
        let rlat2 = lat2 * .pi / 180
        let dlat = (lat2 - lat1) * .pi / 180
        let dlon = (lon2 - lon1) * .pi / 180
        let a = sin(dlat / 2) * sin(dlat / 2)
            + cos(rlat1) * cos(rlat2) * sin(dlon / 2) * sin(dlon / 2)
        return 2 * R * asin(min(1.0, sqrt(a)))
    }
}
