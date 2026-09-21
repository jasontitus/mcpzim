import XCTest
@testable import MCPZimKit

final class PlaybackCompletionGateTests: XCTestCase {
    func testStopBeforeRegistrationAndLateCallback() async {
        let gate = PlaybackCompletionGate()
        XCTAssertTrue(gate.finish(.cancelled))
        let outcome = await gate.wait(timeout: 1)
        XCTAssertEqual(outcome, .cancelled)
        XCTAssertFalse(gate.finish(.played))
    }
    func testLostCallbackTimesOutWithoutClaimingPlayback() async {
        let gate = PlaybackCompletionGate()
        let outcome = await gate.wait(timeout: 0.01)
        XCTAssertEqual(outcome, .timedOut)
        XCTAssertFalse(gate.finish(.played))
    }
    func testRouteChangeCannotBeOverwrittenByLatePlaybackCallback() async {
        let gate = PlaybackCompletionGate()
        XCTAssertTrue(gate.finish(.interrupted))
        XCTAssertFalse(gate.finish(.played))
        let result = await gate.wait(timeout: 1)
        XCTAssertEqual(result, .interrupted)
    }
    func testTaskCancellationReleasesWaiter() async {
        let gate = PlaybackCompletionGate()
        let task = Task { await gate.wait(timeout: 60) }
        task.cancel()
        let outcome = await task.value
        XCTAssertEqual(outcome, .cancelled)
    }
    func testPlaybackBeforeWaitAndCompetingStop() async {
        let gate = PlaybackCompletionGate()
        XCTAssertTrue(gate.finish(.played))
        XCTAssertFalse(gate.finish(.cancelled))
        let outcome = await gate.wait(timeout: 0)
        XCTAssertEqual(outcome, .played)
    }
}
