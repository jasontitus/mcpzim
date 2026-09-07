// SPDX-License-Identifier: MIT

import XCTest
import CryptoKit
@testable import MCPZimChatMac

/// Pins the model-download safety rails added on top of the shared background
/// downloader:
///   * a `TaskLabel` round-trips the model-only fields (kind, SHA-256, cache
///     destination) so a relaunch restores the transfer exactly;
///   * the SHA-256 helper returns the same digest CryptoKit computes;
///   * the authoritative cache gate rejects a wrong size and a wrong
///     checksum, and only accepts a file that matches both — the "never load
///     incomplete weights" guarantee;
///   * the storage gate's message names the shortfall.
final class ModelDownloadTests: XCTestCase {

    private var stagingDir: URL!

    override func setUpWithError() throws {
        stagingDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-download-tests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: stagingDir,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: stagingDir)
    }

    private func write(_ name: String, _ bytes: [UInt8]) throws -> URL {
        let url = stagingDir.appendingPathComponent(name)
        try Data(bytes).write(to: url)
        return url
    }

    // MARK: - TaskLabel persistence

    func testTaskLabelRoundTripsModelFields() throws {
        let label = TaskLabel(
            id: "bonsai-27b-q1-gguf",
            title: "Bonsai 27B (1-bit · Metal)",
            urlString: "https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-Q1_0.gguf",
            expectedBytes: 3_803_452_480,
            kind: "model",
            sha256: "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0",
            destPath: "/tmp/caches/huggingface/hub/models--prism-ml--Bonsai-27B-gguf/snapshots/main/Bonsai-27B-Q1_0.gguf")

        let decoded = TaskLabel(taskDescription: label.encoded)
        XCTAssertNotNil(decoded)
        XCTAssertEqual(decoded?.id, label.id)
        XCTAssertEqual(decoded?.title, label.title)
        XCTAssertEqual(decoded?.urlString, label.urlString)
        XCTAssertEqual(decoded?.expectedBytes, label.expectedBytes)
        XCTAssertEqual(decoded?.kind, "model")
        XCTAssertEqual(decoded?.sha256, label.sha256)
        XCTAssertEqual(decoded?.destPath, label.destPath)
    }

    func testTaskLabelDecodesLegacyLabelWithoutKind() {
        // A label written before model support has no kind/sha256/destPath.
        // It must still decode and default to the ZIM path (kind == nil).
        let legacy = TaskLabel(id: "wikipedia-en-nopic",
                               title: "Wikipedia EN",
                               urlString: "https://example.org/en.zim",
                               expectedBytes: 1_000)
        let decoded = TaskLabel(taskDescription: legacy.encoded)
        XCTAssertNotNil(decoded)
        XCTAssertNil(decoded?.kind)
        XCTAssertNil(decoded?.sha256)
        XCTAssertNil(decoded?.destPath)
    }

    // MARK: - SHA-256 helper

    func testSHA256HexMatchesCryptoKit() throws {
        let payload = Array("zimfo-model-download".utf8)
        let url = try write("sample.gguf", payload)
        let got = try ZimDownloadManager.sha256Hex(of: url)
        let expected = SHA256.hash(data: Data(payload))
            .map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(got, expected)
    }

    // MARK: - Cache validity gate ("never load incomplete weights")

    func testCachedGGUFIsValidAcceptsMatchingSizeAndChecksum() async throws {
        let payload = Array("0123456789abcdef".utf8)
        let url = try write("valid.gguf", payload)
        let digest = try ZimDownloadManager.sha256Hex(of: url)
        let ok = try await LlamaCppProvider.cachedGGUFIsValid(
            url, expectedBytes: Int64(payload.count), sha256: digest)
        XCTAssertTrue(ok)
    }

    func testCachedGGUFIsValidRejectsWrongSize() async throws {
        let payload = Array("0123456789abcdef".utf8)
        let url = try write("wrongsize.gguf", payload)
        let digest = try ZimDownloadManager.sha256Hex(of: url)
        let ok = try await LlamaCppProvider.cachedGGUFIsValid(
            url, expectedBytes: Int64(payload.count + 1), sha256: digest)
        XCTAssertFalse(ok)
    }

    func testCachedGGUFIsValidRejectsWrongChecksum() async throws {
        let payload = Array("0123456789abcdef".utf8)
        let url = try write("wrongsha.gguf", payload)
        // Same size, different content → the digest must differ.
        let otherDigest = "0000000000000000000000000000000000000000000000000000000000000000"
        let ok = try await LlamaCppProvider.cachedGGUFIsValid(
            url, expectedBytes: Int64(payload.count), sha256: otherDigest)
        XCTAssertFalse(ok)
    }

    func testCachedVerdictIncludesExpectedChecksum() async throws {
        let payload = Array("same file, different expected hash".utf8)
        let url = try write("memo.gguf", payload)
        let digest = SHA256.hash(data: Data(payload)).map { String(format: "%02x", $0) }.joined()
        let good = try await LlamaCppProvider.cachedGGUFIsValid(
            url, expectedBytes: Int64(payload.count), sha256: digest)
        let bad = try await LlamaCppProvider.cachedGGUFIsValid(
            url, expectedBytes: Int64(payload.count), sha256: String(repeating: "0", count: 64))
        XCTAssertTrue(good)
        XCTAssertFalse(bad, "A cached success must not bypass a changed expected digest")
    }

    @MainActor
    func testRestoredModelDestinationIsFileURL() {
        let destination = stagingDir.appendingPathComponent("model with spaces.gguf")
        let manager = ZimDownloadManager(restoringDownloads: false)
        let label = TaskLabel(id: "restore-model", title: "Model",
                              urlString: "https://example.org/model.gguf", expectedBytes: 100,
                              kind: "model", destPath: destination.path)
        manager.adoptRestoredTasks([(label: label, taskID: 9001,
                                    received: 10, expected: 100, live: true)])
        XCTAssertEqual(manager.items.first?.destination, destination)
        XCTAssertEqual(manager.items.first?.destination?.isFileURL, true)
    }

    // MARK: - Storage gate message

    func testInsufficientStorageMessageNamesShortfall() {
        let error = ZimDownloadError.insufficientStorage(
            needed: 4_000_000_000, available: 1_000_000_000,
            title: "Bonsai 27B (1-bit · Metal)")
        let message = error.errorDescription ?? ""
        XCTAssertTrue(message.contains("Bonsai 27B"))
        XCTAssertTrue(message.contains("4 GB") || message.contains("3.7 GB")
                      || message.contains("3.72 GB"))
        XCTAssertTrue(message.contains("Free up space"))
    }

    // MARK: - Capacity-aware default

    func testRecommendedModelIDIsLFMFirst() {
        // The user-verified on-device model is LFM2.5-8B, so it's the
        // default wherever the budget holds it (snug, balanced, generous).
        DeviceProfile.override = .snug
        XCTAssertEqual(ModelCatalog.recommendedModelID(),
                       "lfm2.5-8b-a1b-q3km-gguf-ft")

        // Balanced keeps LFM too.
        DeviceProfile.override = .balanced
        XCTAssertEqual(ModelCatalog.recommendedModelID(),
                       "lfm2.5-8b-a1b-q3km-gguf-ft")

        // On macOS `modelFitsDevice` always returns true (the OS swaps),
        // so the tight-tier fallback can't be exercised here — pin the
        // contract that LFM2.5 is the default on every tier where the
        // budget holds it.
        DeviceProfile.override = .generous
        XCTAssertEqual(ModelCatalog.recommendedModelID(),
                       "lfm2.5-8b-a1b-q3km-gguf-ft")

        DeviceProfile.override = nil
    }
}
