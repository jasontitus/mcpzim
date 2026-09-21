import XCTest
@testable import MCPZimKit

final class ModelInputCaptureTests: XCTestCase {
    private func path() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    }

    private func record(_ capture: ModelInputCapture, prompt: String = "shared prefix\n新しい質問") throws {
        try capture.record(modelID: "source", runtime: "test", prompt: prompt,
                           tokenIDs: [1, 42, 99], sampler: ["temperature": 0],
                           stopSequences: ["<end>"])
    }

    func testDisabledDoesNotRequireContextOrWrite() throws {
        let capture = ModelInputCapture()
        try record(capture)
        XCTAssertEqual(capture.recordedCount, 0)
    }

    func testExactFullPromptsAndTokensAcrossFollowupAndCompletedManifest() throws {
        let directory = path()
        defer { try? FileManager.default.removeItem(at: directory) }
        let capture = ModelInputCapture()
        try capture.start(directory: directory, metadata: ["model": "test"])
        capture.setCase(conversation: "../unsafe-filename", turn: 0)
        try record(capture)
        capture.setCase(conversation: "../unsafe-filename", turn: 1)
        try record(capture, prompt: "shared prefix\n新しい質問\nfollow-up")
        let first = try JSONDecoder().decode(ModelInputCapture.Record.self, from:
            Data(contentsOf: directory.appendingPathComponent("invocation-000000.json")))
        let second = try JSONDecoder().decode(ModelInputCapture.Record.self, from:
            Data(contentsOf: directory.appendingPathComponent("invocation-000001.json")))
        XCTAssertEqual(first.prompt, "shared prefix\n新しい質問")
        XCTAssertEqual(first.tokenIDs, [1, 42, 99])
        XCTAssertEqual(first.stopSequences, ["<end>"])
        XCTAssertEqual(second.prompt, first.prompt + "\nfollow-up")
        XCTAssertNotEqual(first.promptSHA256, second.promptSHA256)
        XCTAssertEqual(second.turn, 1)
        XCTAssertEqual(second.ordinal, 1)
        try capture.finish()
        let manifest = try JSONDecoder().decode([String: String].self, from:
            Data(contentsOf: directory.appendingPathComponent("capture-run.json")))
        XCTAssertEqual(manifest["status"], "completed")
        XCTAssertEqual(manifest["invocationCount"], "2")
    }

    func testExistingDirectoryRejectedWithoutOverwriting() throws {
        let directory = path()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        XCTAssertThrowsError(try ModelInputCapture().start(directory: directory, metadata: [:]))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
    }

    func testCaughtRecordingFailureCannotBecomeSuccessfulCapture() throws {
        let directory = path()
        defer { try? FileManager.default.removeItem(at: directory) }
        let capture = ModelInputCapture()
        try capture.start(directory: directory, metadata: [:])
        XCTAssertThrowsError(try record(capture)) // missing case association
        capture.setCase(conversation: "later", turn: 0)
        XCTAssertThrowsError(try record(capture))
        XCTAssertThrowsError(try capture.finish())
        XCTAssertEqual(capture.recordedCount, 0)
    }

    func testDiskWriteFailureRemainsIncomplete() throws {
        let directory = path()
        defer { try? FileManager.default.removeItem(at: directory) }
        let capture = ModelInputCapture()
        try capture.start(directory: directory, metadata: [:])
        capture.setCase(conversation: "case", turn: 0)
        // Block the target path deterministically; chmod is unreliable as root.
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("invocation-000000.json"),
            withIntermediateDirectories: false)
        XCTAssertThrowsError(try record(capture))
        XCTAssertThrowsError(try capture.checkHealth())
        XCTAssertThrowsError(try capture.finish())
    }
}
