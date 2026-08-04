// SPDX-License-Identifier: MIT

import XCTest
@testable import MCPZimChatMac

/// Pins the Nearby Sharing model hand-off: a received GGUF is adopted into
/// the provider's HuggingFace-style cache slot only when the filename
/// matches the provider's pinned file and the byte count validates, and a
/// successfully adopted model immediately becomes shareable onward.
final class ModelSharingTests: XCTestCase {
    private let testRepo = "test/model-sharing-tests"
    private let testFilename = "model-sharing-test.gguf"

    private func makeProvider(expectedBytes: Int64?) -> LlamaCppProvider {
        LlamaCppProvider(id: "test-model-sharing",
                         displayName: "Model Sharing Test",
                         huggingFaceRepo: testRepo,
                         ggufFilename: testFilename,
                         expectedGGUFBytes: expectedBytes)
    }

    /// Mirrors the provider's cache layout:
    /// `<caches>/huggingface/hub/models--<repo-slug>/snapshots/main/<file>`.
    private var cacheRepoDir: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory,
                                              in: .userDomainMask).first!
        return caches
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--test--model-sharing-tests")
    }

    private var cacheSlot: URL {
        cacheRepoDir
            .appendingPathComponent("snapshots")
            .appendingPathComponent("main")
            .appendingPathComponent(testFilename)
    }

    private var stagingDir: URL!

    override func setUpWithError() throws {
        stagingDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-sharing-tests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: stagingDir,
                                                withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: cacheRepoDir)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: stagingDir)
        try? FileManager.default.removeItem(at: cacheRepoDir)
    }

    private func stage(_ filename: String, byteCount: Int) throws -> URL {
        let url = stagingDir.appendingPathComponent(filename)
        try Data(repeating: 0x5A, count: byteCount).write(to: url)
        return url
    }

    func testAdoptMovesValidatedFileIntoCacheSlot() throws {
        let provider = makeProvider(expectedBytes: 8)
        XCTAssertNil(provider.shareableGGUFURL, "nothing cached yet")
        XCTAssertFalse(provider.hasCompleteCachedGGUF)

        let received = try stage(testFilename, byteCount: 8)
        XCTAssertTrue(provider.adoptSharedGGUF(at: received))

        XCTAssertFalse(FileManager.default.fileExists(atPath: received.path),
                       "adoption must move, not copy")
        XCTAssertTrue(FileManager.default.fileExists(atPath: cacheSlot.path),
                      "file must land in the provider's own cache slot")
        XCTAssertTrue(provider.hasCompleteCachedGGUF,
                      "the provider's downloader must now see it as cached")
        XCTAssertEqual(provider.shareableGGUFURL?.lastPathComponent, testFilename,
                       "an adopted model is immediately shareable onward")
    }

    func testAdoptRejectsWrongFilenameAndLeavesFileAlone() throws {
        let provider = makeProvider(expectedBytes: 8)
        let received = try stage("some-other-model.gguf", byteCount: 8)
        XCTAssertFalse(provider.adoptSharedGGUF(at: received))
        XCTAssertTrue(FileManager.default.fileExists(atPath: received.path),
                      "a rejected file must not be consumed")
        XCTAssertFalse(FileManager.default.fileExists(atPath: cacheSlot.path))
    }

    func testAdoptRejectsWrongByteCount() throws {
        let provider = makeProvider(expectedBytes: 8)
        let received = try stage(testFilename, byteCount: 5)
        XCTAssertFalse(provider.adoptSharedGGUF(at: received),
                       "a same-named file from a different release must be rejected")
        XCTAssertTrue(FileManager.default.fileExists(atPath: received.path))
    }

    func testAdoptReplacesTruncatedEarlierDownload() throws {
        let provider = makeProvider(expectedBytes: 8)
        try FileManager.default.createDirectory(
            at: cacheSlot.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data(repeating: 0x00, count: 3).write(to: cacheSlot) // truncated
        XCTAssertFalse(provider.hasCompleteCachedGGUF)

        let received = try stage(testFilename, byteCount: 8)
        XCTAssertTrue(provider.adoptSharedGGUF(at: received))
        XCTAssertTrue(provider.hasCompleteCachedGGUF)
        let size = try FileManager.default.attributesOfItem(atPath: cacheSlot.path)[.size] as? Int64
        XCTAssertEqual(size, 8)
    }

    func testAdoptDropsDuplicateWhenGoodCopyExists() throws {
        let provider = makeProvider(expectedBytes: 8)
        try FileManager.default.createDirectory(
            at: cacheSlot.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data(repeating: 0x11, count: 8).write(to: cacheSlot) // already good

        let received = try stage(testFilename, byteCount: 8)
        XCTAssertTrue(provider.adoptSharedGGUF(at: received),
                      "an already-satisfied adopt still reports success")
        XCTAssertFalse(FileManager.default.fileExists(atPath: received.path),
                       "the duplicate must be cleaned up")
        let data = try Data(contentsOf: cacheSlot)
        XCTAssertEqual(data.first, 0x11, "the existing good copy must be kept")
    }

    func testUnpinnedProviderAdoptsOnFilenameAloneAfterChunkVerifiedTransfer() throws {
        // No pinned byte count (expectedGGUFBytes == nil): the swarm's
        // per-chunk SHA-256 already guaranteed integrity end-to-end, so a
        // filename match is sufficient.
        let provider = makeProvider(expectedBytes: nil)
        let received = try stage(testFilename, byteCount: 3)
        XCTAssertTrue(provider.adoptSharedGGUF(at: received))
        XCTAssertTrue(FileManager.default.fileExists(atPath: cacheSlot.path))
    }

    // MARK: Voice-model routing (Nearby Sharing directory shares)

    func testVoiceDestinationRoutesKnownTreesOnly() {
        let kokoro = ZimSwarmController.voiceModelDestination(
            forRelativePath: "kokoro_mlx/voices.npz")
        XCTAssertEqual(kokoro?.lastPathComponent, "voices.npz")
        XCTAssertEqual(kokoro?.deletingLastPathComponent().lastPathComponent, "kokoro_mlx")

        let supertonic = ZimSwarmController.voiceModelDestination(
            forRelativePath: "supertonic_3/supertonic-3-coreml/model.mlmodelc/weights.bin")
        XCTAssertEqual(supertonic?.lastPathComponent, "weights.bin")
        XCTAssertTrue(supertonic?.path.contains("/supertonic_3/supertonic-3-coreml/") == true,
                      "nested Core ML bundle layout must be preserved")

        XCTAssertNil(ZimSwarmController.voiceModelDestination(forRelativePath: "somepack/file.bin"),
                     "unknown trees never write into Application Support")
        XCTAssertNil(ZimSwarmController.voiceModelDestination(forRelativePath: "kokoro_mlx"),
                     "a bare top-level name is not a voice file")
        XCTAssertNil(ZimSwarmController.voiceModelDestination(forRelativePath: "wikipedia.zim"))
    }

    func testVoiceDestinationHandlesUnprefixedFolderSwarmPaths() {
        // A share consisting of only one voice folder arrives as a folder
        // swarm with unprefixed paths (the engine's Go-conformant form).
        let weights = ZimSwarmController.voiceModelDestination(
            forRelativePath: "kokoro-v1_0.safetensors")
        XCTAssertEqual(weights?.deletingLastPathComponent().lastPathComponent, "kokoro_mlx")

        let bundle = ZimSwarmController.voiceModelDestination(
            forRelativePath: "supertonic-3-coreml/model.mlmodelc/coremldata.bin")
        XCTAssertTrue(bundle?.path.contains("/supertonic_3/supertonic-3-coreml/") == true)

        XCTAssertNil(ZimSwarmController.voiceModelDestination(
            forRelativePath: "nested/kokoro-v1_0.safetensors"),
            "the bare-filename form only matches at the top level")
        XCTAssertNil(ZimSwarmController.voiceModelDestination(
            forRelativePath: "supertonic-3-coreml"),
            "a bare bundle-root name is not a voice file")
    }

    func testRelativePathComputation() {
        let root = URL(fileURLWithPath: "/tmp/stage/swarm1", isDirectory: true)
        XCTAssertEqual(
            ZimSwarmController.relativePath(
                of: URL(fileURLWithPath: "/tmp/stage/swarm1/kokoro_mlx/voices.npz"),
                under: root),
            "kokoro_mlx/voices.npz")
        XCTAssertEqual(
            ZimSwarmController.relativePath(
                of: URL(fileURLWithPath: "/tmp/stage/swarm1/file.zim"), under: root),
            "file.zim")
        XCTAssertNil(
            ZimSwarmController.relativePath(
                of: URL(fileURLWithPath: "/tmp/stage/other/x.bin"), under: root),
            "files outside the swarm's staging folder resolve to nil")
    }
}
