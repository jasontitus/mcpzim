// SPDX-License-Identifier: MIT
//
// Bridge between the iOS-side `ZimfoContext` (persistent route + GPS state
// that survives Siri intent dispatch) and the MCPZimKit `HostStateSnapshot`
// the tool adapter consumes. Keeping this in the iOS target means MCPZimKit
// stays framework-free and ZimfoContext doesn't need to import MCPZimKit.

import Foundation
import MCPZimKit

/// Memo for the ActiveRoute → RouteSnapshot conversion: `mcpSnapshot()`
/// runs on every tool dispatch and the polyline copy is O(points), while
/// the route only changes on set/clear (tracked by
/// `ZimfoContext.routeVersion`). Lives outside the actor because
/// extensions cannot add stored properties; the lock covers the
/// multiple-instance (test) case.
private final class RouteSnapshotMemo: @unchecked Sendable {
    static let shared = RouteSnapshotMemo()
    private let lock = NSLock()
    private var owner: ObjectIdentifier?
    private var version: UInt64 = 0
    private var route: RouteSnapshot?

    func lookup(owner o: ObjectIdentifier, version v: UInt64) -> (hit: Bool, route: RouteSnapshot?) {
        lock.lock(); defer { lock.unlock() }
        guard owner == o, version == v else { return (false, nil) }
        return (true, route)
    }

    func store(owner o: ObjectIdentifier, version v: UInt64, route r: RouteSnapshot?) {
        lock.lock(); defer { lock.unlock() }
        owner = o; version = v; route = r
    }
}

extension ZimfoContext {
    /// Read the current route + GPS state and convert to the MCP-adapter
    /// shape. Called from tool dispatch (through the actor's executor), so
    /// picking up a stale pointer here is bounded by the actor's queue.
    public func mcpSnapshot() -> HostStateSnapshot {
        let key = ObjectIdentifier(self)
        let route: RouteSnapshot?
        let memo = RouteSnapshotMemo.shared.lookup(owner: key, version: routeVersion)
        if memo.hit {
            route = memo.route
        } else {
            route = activeRoute.map { r -> RouteSnapshot in
                RouteSnapshot(
                    origin: .init(lat: r.origin.lat, lon: r.origin.lon),
                    destination: .init(lat: r.destination.lat, lon: r.destination.lon),
                    originName: r.originName,
                    destinationName: r.destinationName,
                    totalDistanceMeters: r.totalDistanceMeters,
                    totalDurationSeconds: r.totalDurationSeconds,
                    polyline: r.polyline.map { .init(lat: $0.lat, lon: $0.lon) },
                    cumulativeDistanceMeters: r.cumulativeDistanceMeters,
                    turnByTurn: r.turnByTurn
                )
            }
            RouteSnapshotMemo.shared.store(owner: key, version: routeVersion, route: route)
        }
        let location = lastLocation.map { c -> LocationSnapshot in
            LocationSnapshot(lat: c.lat, lon: c.lon)
        }
        return HostStateSnapshot(activeRoute: route, currentLocation: location)
    }
}
