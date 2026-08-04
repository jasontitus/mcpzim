import XCTest
@testable import LocalSwarmEngine

final class AuthTokenTests: XCTestCase {
    /// Must match peer-go's AuthToken byte-for-byte (Android<->iOS interop).
    func testConformanceVector() {
        XCTAssertEqual(swarmAuthToken(swarmID: "testswarm", pin: "1234"),
                       "70101e2f9abdc67fef0581d6b5cd69945d95311c8037c565cc2df77a5424e3c6")
    }

    func testConstantTimeEquals() {
        let a = swarmAuthToken(swarmID: "s", pin: "9")
        XCTAssertTrue(a.utf8CStringConstantTimeEquals(a))
        XCTAssertFalse(a.utf8CStringConstantTimeEquals(swarmAuthToken(swarmID: "s", pin: "8")))
        XCTAssertFalse(a.utf8CStringConstantTimeEquals("short"))
    }
}
