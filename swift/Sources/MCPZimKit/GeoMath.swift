// SPDX-License-Identifier: MIT
//
// Small geodesy helpers for conversational answers: "it's 200 m north-east
// of you" needs a bearing, not just a distance. Pure math — no UI, no ZIM,
// no CoreLocation, exercised by `swift test`.

import Foundation

public enum GeoMath {

    /// Great-circle distance in metres (haversine).
    public static func haversineMeters(
        _ lat1: Double, _ lon1: Double,
        _ lat2: Double, _ lon2: Double
    ) -> Double {
        let r = 6_371_000.0
        let dLat = (lat2 - lat1) * .pi / 180
        let dLon = (lon2 - lon1) * .pi / 180
        let a = sin(dLat / 2) * sin(dLat / 2)
            + cos(lat1 * .pi / 180) * cos(lat2 * .pi / 180)
            * sin(dLon / 2) * sin(dLon / 2)
        return r * 2 * atan2(sqrt(a), sqrt(1 - a))
    }

    /// Initial great-circle bearing from point 1 to point 2, in degrees
    /// clockwise from true north, normalised to [0, 360).
    public static func bearingDegrees(
        fromLat lat1: Double, fromLon lon1: Double,
        toLat lat2: Double, toLon lon2: Double
    ) -> Double {
        let phi1 = lat1 * .pi / 180
        let phi2 = lat2 * .pi / 180
        let dLon = (lon2 - lon1) * .pi / 180
        let y = sin(dLon) * cos(phi2)
        let x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dLon)
        let deg = atan2(y, x) * 180 / .pi
        return (deg + 360).truncatingRemainder(dividingBy: 360)
    }

    /// 8-wind compass point as a full lowercase word ("north",
    /// "north-east", …) — reads naturally in prose and TTS, unlike "NNE".
    public static func compassPoint(degrees: Double) -> String {
        let names = [
            "north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west",
        ]
        let normalized = (degrees.truncatingRemainder(dividingBy: 360) + 360)
            .truncatingRemainder(dividingBy: 360)
        let idx = Int((normalized + 22.5) / 45) % 8
        return names[idx]
    }

    /// Compass point for the bearing from point 1 to point 2.
    public static func compassPoint(
        fromLat lat1: Double, fromLon lon1: Double,
        toLat lat2: Double, toLon lon2: Double
    ) -> String {
        compassPoint(degrees: bearingDegrees(
            fromLat: lat1, fromLon: lon1, toLat: lat2, toLon: lon2))
    }
}
