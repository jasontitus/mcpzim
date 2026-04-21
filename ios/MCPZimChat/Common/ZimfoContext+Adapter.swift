// SPDX-License-Identifier: MIT
//
// Bridge between the iOS-side `ZimfoContext` (persistent route + GPS state
// that survives Siri intent dispatch) and the MCPZimKit `HostStateSnapshot`
// the tool adapter consumes. Keeping this in the iOS target means MCPZimKit
// stays framework-free and ZimfoContext doesn't need to import MCPZimKit.

import Foundation
import MCPZimKit

extension ZimfoContext {
    /// Read the current route + GPS state and convert to the MCP-adapter
    /// shape. Called from tool dispatch (through the actor's executor), so
    /// picking up a stale pointer here is bounded by the actor's queue.
    public func mcpSnapshot() -> HostStateSnapshot {
        let route = activeRoute.map { r -> RouteSnapshot in
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
        let location = lastLocation.map { c -> LocationSnapshot in
            LocationSnapshot(lat: c.lat, lon: c.lon)
        }
        return HostStateSnapshot(activeRoute: route, currentLocation: location)
    }
}
