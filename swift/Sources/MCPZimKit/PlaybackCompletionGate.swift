import Foundation

/// Exactly-once completion for an audio callback which may never arrive.
/// Does not equate a deadline or cancellation with successful playback.
public final class PlaybackCompletionGate: @unchecked Sendable {
    public enum Outcome: Sendable, Equatable { case played, cancelled, timedOut, interrupted }
    private let lock = NSLock()
    private var outcome: Outcome?
    private var continuation: CheckedContinuation<Outcome, Never>?

    public init() {}

    @discardableResult
    public func finish(_ value: Outcome) -> Bool {
        let result: (Bool, CheckedContinuation<Outcome, Never>?) = lock.withLock {
            guard outcome == nil else { return (false, nil) }
            outcome = value
            let waiter = continuation
            continuation = nil
            return (true, waiter)
        }
        result.1?.resume(returning: value)
        return result.0
    }

    /// One waiter per gate. Completion before registration is supported.
    public func wait(timeout: TimeInterval) async -> Outcome {
        let seconds = timeout.isFinite ? max(0, min(timeout, 3600)) : 0
        let timer = Task {
            do { try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) }
            catch { return }
            finish(.timedOut)
        }
        defer { timer.cancel() }
        return await withTaskCancellationHandler {
            await withCheckedContinuation { waiter in
                let completed: Outcome? = lock.withLock {
                    if let outcome { return outcome }
                    precondition(continuation == nil, "Playback gate supports one waiter")
                    continuation = waiter
                    return nil
                }
                if let completed { waiter.resume(returning: completed) }
            }
        } onCancel: { self.finish(.cancelled) }
    }
}
