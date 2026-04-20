// SPDX-License-Identifier: MIT
//
// One-shot CoreLocation fetch wired for async/await. Intents only ever
// need the current fix; no streaming, no background updates. Requests
// `WhenInUse` authorization on first call — if the user denies, the
// intent falls back gracefully.

import Foundation
import CoreLocation

final class LocationFetcher: NSObject, CLLocationManagerDelegate, @unchecked Sendable {
    private let manager = CLLocationManager()
    private var continuation: CheckedContinuation<CLLocationCoordinate2D, Error>?

    /// Tests (or "offline dev" flows) can set this to bypass the real
    /// `CLLocationManager` and return a canned coordinate. When nil,
    /// `once()` talks to CoreLocation normally.
    nonisolated(unsafe) static var overrideForTesting: (@Sendable () async throws -> CLLocationCoordinate2D)?

    static func once(timeout: TimeInterval = 8) async throws -> CLLocationCoordinate2D {
        if let override = overrideForTesting {
            return try await override()
        }
        return try await withThrowingTaskGroup(of: CLLocationCoordinate2D.self) { group in
            let fetcher = LocationFetcher()
            group.addTask {
                try await fetcher.requestOnce()
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw LocationError.timeout
            }
            guard let first = try await group.next() else {
                throw LocationError.noResult
            }
            group.cancelAll()
            return first
        }
    }

    enum LocationError: Error {
        case denied, timeout, noResult
    }

    private func requestOnce() async throws -> CLLocationCoordinate2D {
        try await withCheckedThrowingContinuation { c in
            self.continuation = c
            DispatchQueue.main.async {
                self.manager.delegate = self
                self.manager.desiredAccuracy = kCLLocationAccuracyBest
                let status = self.manager.authorizationStatus
                switch status {
                case .notDetermined:
                    self.manager.requestWhenInUseAuthorization()
                    // locationManagerDidChangeAuthorization will re-trigger.
                case .denied, .restricted:
                    self.continuation?.resume(throwing: LocationError.denied)
                    self.continuation = nil
                case .authorizedAlways, .authorizedWhenInUse:
                    self.manager.requestLocation()
                @unknown default:
                    self.manager.requestLocation()
                }
            }
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            manager.requestLocation()
        case .denied, .restricted:
            continuation?.resume(throwing: LocationError.denied)
            continuation = nil
        default: break
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        continuation?.resume(returning: loc.coordinate)
        continuation = nil
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        continuation?.resume(throwing: error)
        continuation = nil
    }
}
