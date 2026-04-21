// SPDX-License-Identifier: MIT
//
// Long-lived CoreLocation singleton. The first `CLLocationManager`
// instance you create pays a 1–30 s handshake with the system
// location service before the delegate starts firing. To match the
// Google-Maps "blue dot is already here" experience, we keep one
// manager alive for the app lifetime, start `startUpdatingLocation`
// as soon as permission is granted, and just hand out the latest
// cached fix whenever someone calls `LocationFetcher.once()`.
//
// A call that arrives before the first fix lands waits on a
// continuation that `didUpdateLocations` resumes — no new manager,
// no new handshake, no timeout burning 30 s.

import Foundation
import CoreLocation
import OSLog

private let locLog = Logger(subsystem: "org.mcpzim.MCPZimChat", category: "Location")

final class LocationFetcher: NSObject, CLLocationManagerDelegate, @unchecked Sendable {

    // MARK: - Shared instance

    static let shared = LocationFetcher()

    private let manager = CLLocationManager()
    /// Latest fix (if any). Updated on every delegate callback.
    private var latest: CLLocation?
    /// Callers waiting on the next fix; drained when one arrives.
    private var waiters: [CheckedContinuation<CLLocationCoordinate2D, Error>] = []
    /// Subscribers notified on every new fix. ChatSession registers
    /// one at launch so `session.currentLocation` tracks the singleton
    /// without any polling / timeout. `@MainActor` guarantees the
    /// subscriber doesn't see stale state.
    nonisolated(unsafe) private var subscribers: [(CLLocationCoordinate2D) -> Void] = []

    /// Tests (or "offline dev" flows) can set this to bypass the real
    /// `CLLocationManager` and return a canned coordinate.
    nonisolated(unsafe) static var overrideForTesting: (@Sendable () async throws -> CLLocationCoordinate2D)?

    override private init() {
        super.init()
        // Configure once — these carry across app lifetime.
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        manager.pausesLocationUpdatesAutomatically = true
        manager.activityType = .other
        locLog.notice("LocationFetcher.shared created")
    }

    // MARK: - Public API

    /// Call at launch so the manager starts feeding location the
    /// moment permission is granted. Safe to call multiple times.
    static func start() {
        DispatchQueue.main.async {
            shared.startIfAuthorized()
        }
    }

    /// Subscribe to every new CL fix. The subscriber is called on the
    /// main actor via `Task { @MainActor in ... }`. There's no
    /// unsubscribe because callers are expected to live for the app
    /// lifetime (ChatSession).
    static func subscribe(_ cb: @escaping (CLLocationCoordinate2D) -> Void) {
        DispatchQueue.main.async { shared.subscribers.append(cb) }
    }

    /// Trigger the `WhenInUse` permission prompt if we haven't yet.
    /// Call at launch so the dialog shows up while the user is
    /// looking, not later when they've already asked for directions.
    static func requestAuthorizationIfNeeded() {
        DispatchQueue.main.async {
            let s = shared.manager.authorizationStatus
            locLog.notice("requestAuthorizationIfNeeded: status=\(s.rawValue, privacy: .public)")
            if s == .notDetermined {
                #if os(iOS)
                shared.manager.requestWhenInUseAuthorization()
                #else
                shared.manager.requestAlwaysAuthorization()
                #endif
            } else if Self.isAuthorized(s) {
                shared.manager.startUpdatingLocation()
            }
        }
    }

    /// Cross-platform authorization check. `.authorizedWhenInUse`
    /// doesn't exist on macOS; `.authorizedAlways` covers both.
    private static func isAuthorized(_ s: CLAuthorizationStatus) -> Bool {
        #if os(iOS)
        return s == .authorizedWhenInUse || s == .authorizedAlways
        #else
        return s == .authorizedAlways
        #endif
    }

    /// Return the current location. Fast path: if the singleton has
    /// any fix newer than `maxAge` seconds, return it immediately.
    /// Otherwise wait for the next delegate callback (up to
    /// `timeout`). Because the manager is always running, the wait
    /// is typically <1 s — not the 20–30 s cold-start handshake of a
    /// per-call manager.
    static func once(timeout: TimeInterval = 15, maxAge: TimeInterval = 120) async throws -> CLLocationCoordinate2D {
        if let override = overrideForTesting {
            return try await override()
        }
        return try await shared.latestOrWait(timeout: timeout, maxAge: maxAge)
    }

    // MARK: - Internals

    private func startIfAuthorized() {
        let s = manager.authorizationStatus
        if Self.isAuthorized(s) {
            locLog.notice("startIfAuthorized: starting updates (status=\(s.rawValue, privacy: .public))")
            manager.startUpdatingLocation()
        } else {
            locLog.notice("startIfAuthorized: not yet authorized (status=\(s.rawValue, privacy: .public))")
        }
    }

    private func latestOrWait(timeout: TimeInterval, maxAge: TimeInterval) async throws -> CLLocationCoordinate2D {
        // Check status + cached fix on the main queue so delegate
        // callbacks can't race us.
        let snapshot: CLLocation? = await MainActor.run { [weak self] in
            guard let self else { return nil }
            self.startIfAuthorized()
            if let loc = self.latest,
               Date().timeIntervalSince(loc.timestamp) < maxAge {
                return loc
            }
            return nil
        }
        if let loc = snapshot {
            locLog.notice("once: cached fix (\(Int(Date().timeIntervalSince(loc.timestamp)))s old)")
            return loc.coordinate
        }
        // No fresh fix yet — enqueue a waiter + race against a
        // timeout. The delegate resumes waiters on the next fix.
        return try await withThrowingTaskGroup(of: CLLocationCoordinate2D.self) { group in
            group.addTask { [weak self] in
                try await withCheckedThrowingContinuation { c in
                    Task { @MainActor [weak self] in
                        self?.waiters.append(c)
                    }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw LocationError.timeout
            }
            defer {
                // Make sure any leftover waiters are drained with an
                // error so their continuations don't leak.
                Task { @MainActor [weak self] in self?.failPendingWaiters(.timeout) }
                group.cancelAll()
            }
            if let first = try await group.next() {
                return first
            }
            throw LocationError.noResult
        }
    }

    @MainActor
    private func failPendingWaiters(_ err: LocationError) {
        let toFail = waiters
        waiters.removeAll()
        for c in toFail { c.resume(throwing: err) }
    }

    enum LocationError: Error {
        case denied, timeout, noResult
    }

    // MARK: - CLLocationManagerDelegate

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let s = manager.authorizationStatus
        locLog.notice("didChangeAuthorization: status=\(s.rawValue, privacy: .public)")
        if Self.isAuthorized(s) {
            manager.startUpdatingLocation()
        } else if s == .denied || s == .restricted {
            Task { @MainActor in self.failPendingWaiters(.denied) }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        locLog.notice("didUpdateLocations: (\(loc.coordinate.latitude), \(loc.coordinate.longitude))")
        Task { @MainActor in
            self.latest = loc
            let waiters = self.waiters
            self.waiters.removeAll()
            for c in waiters { c.resume(returning: loc.coordinate) }
            // Push to every ChatSession / intent subscriber so their
            // `currentLocation` state stays fresh without polling.
            for cb in self.subscribers { cb(loc.coordinate) }
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        locLog.notice("didFailWithError: \(String(describing: error), privacy: .public)")
        // Keep the manager running — a transient failure shouldn't
        // break every future call. Just surface to pending waiters
        // so they don't sit on the timeout.
        Task { @MainActor in self.failPendingWaiters(.timeout) }
    }
}
