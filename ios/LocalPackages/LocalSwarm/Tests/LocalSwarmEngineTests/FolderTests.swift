import XCTest
import Foundation
@testable import LocalSwarmEngine

/// Folder (multi-file) shares on the Apple side: building a manifest from a
/// directory, the cross-platform id vector shared with the Go suite, a full
/// tree round-trip through the ChunkStore, and path-traversal refusal.
final class FolderTests: XCTestCase {

    private func makeDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeTree(_ root: URL, _ files: [String: Data]) throws {
        let fm = FileManager.default
        for (rel, data) in files {
            let url = root.appendingPathComponent(rel)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
    }

    /// The identical fixed tree is built by the Go suite (TestFolderConformanceVector);
    /// both assert this same id, so per-file chunking, path form, or ordering can't
    /// drift between an iPhone and a Linux/Android seeder without a test failing.
    func testFolderConformanceVectorMatchesGo() throws {
        let root = try makeDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeTree(root, [
            "a.txt": Data("localswarm\n".utf8),
            "docs/readme.md": Data("# hello\n".utf8),
            "docs/img/pixel.bin": Data([0, 1, 2, 3, 4]),
            "empty": Data(),
        ])
        let sources = try Chunker.folderSources(root)
        let (manifest, _) = try Chunker.buildManifest(name: root.lastPathComponent, sources: sources)
        XCTAssertEqual(manifest.swarmID,
                       "a9b250dea4b2399f2326b9afbf62b0a06ad74b087cc18bfae75d08e4216b9599",
                       "folder swarm id drifted from the Go seeder")
        XCTAssertEqual(manifest.files.count, 4)
        XCTAssertEqual(manifest.chunkCount, 3)
        XCTAssertEqual(manifest.totalBytes, 24)
        XCTAssertNoThrow(try manifest.validate())
    }

    func testFolderRoundTripPreservesTree() throws {
        let src = try makeDir()
        defer { try? FileManager.default.removeItem(at: src) }
        var big = Data(count: Chunker.defaultChunkSize * 2 + 321) // 2 full + 1 short chunk
        for i in 0..<big.count { big[i] = UInt8((i * 7 + 3) % 251) }
        try writeTree(src, [
            "readme.txt": Data("hi".utf8),
            "media/clip.bin": big,
            "media/art/logo.bin": Data([9, 9, 9, 9]),
            "notes/empty.txt": Data(),
        ])
        let sources = try Chunker.folderSources(src)
        let (manifest, ordered) = try Chunker.buildManifest(name: "lib", sources: sources)
        try manifest.validate()

        let seed = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)
        let dest = try makeDir()
        defer { try? FileManager.default.removeItem(at: dest) }
        let recv = try ChunkStore.forDownloading(manifest: manifest, directory: dest)
        for i in 0..<manifest.chunkCount {
            _ = try recv.writeChunk(i, data: try seed.readChunk(i))
        }
        XCTAssertTrue(recv.isComplete)

        // Every file rebuilt byte-for-byte at its relative path, subdirs and all.
        for file in manifest.files {
            let want = try Data(contentsOf: src.appendingPathComponent(file.path))
            let have = try Data(contentsOf: dest.appendingPathComponent(file.path))
            XCTAssertEqual(have, want, "content mismatch for \(file.path)")
        }
    }

    /// A hostile manifest can carry any path (content addressing is computed over
    /// whatever paths it picks), so both validation and the store must refuse "..".
    /// Mixed shares — plain files alongside whole directories in one swarm
    /// (a host app seeding "library files + model folders"). Directories are
    /// expanded with their folder name as the path prefix; the single-folder
    /// share stays unprefixed (covered by the Go conformance vector above).
    func testMixedSourcesExpandDirectoriesWithPrefix() throws {
        let root = try makeDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeTree(root, [
            "library.zim": Data("zim".utf8),
            "model.gguf": Data("gguf".utf8),
            "kokoro_mlx/kokoro-v1_0.safetensors": Data("weights".utf8),
            "kokoro_mlx/voices.npz": Data("voices".utf8),
            "supertonic_3/coreml/model.bin": Data("coreml".utf8),
        ])
        let urls = [
            root.appendingPathComponent("library.zim"),
            root.appendingPathComponent("model.gguf"),
            root.appendingPathComponent("kokoro_mlx", isDirectory: true),
            root.appendingPathComponent("supertonic_3", isDirectory: true),
        ]
        let sources = try Chunker.mixedSources(urls)
        XCTAssertEqual(sources.map(\.path),
                       ["library.zim",
                        "model.gguf",
                        "kokoro_mlx/kokoro-v1_0.safetensors",
                        "kokoro_mlx/voices.npz",
                        "supertonic_3/coreml/model.bin"],
                       "files stay flat; each directory expands under its own name")

        let (manifest, _) = try Chunker.buildManifest(name: "mixed", sources: sources)
        XCTAssertNoThrow(try manifest.validate())
        XCTAssertEqual(manifest.files.count, 5)
    }

    func testMixedSourcesThrowsForMissingEntry() throws {
        let root = try makeDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let missing = root.appendingPathComponent("gone.zim")
        XCTAssertThrowsError(try Chunker.mixedSources([missing])) { error in
            guard case Chunker.ChunkerError.unreadable = error else {
                return XCTFail("expected .unreadable, got \(error)")
            }
        }
    }

    func testDownloadRejectsTraversalPath() throws {
        let data = Data([1, 2, 3, 4])
        let file = SwarmFile(path: "../escape.bin", sizeBytes: 4, startChunkIndex: 0, endChunkIndex: 0)
        let hashes = [Hashing.sha256Hex(data)]
        let id = Chunker.contentAddressedID(files: [file], hashes: hashes, chunkSize: Chunker.defaultChunkSize)
        let manifest = SwarmManifest(protocolVersion: 1, swarmID: id, name: "evil", totalBytes: 4,
                                     chunkSizeBytes: Chunker.defaultChunkSize, chunkHashAlgo: "sha256",
                                     files: [file], chunkHashes: hashes)
        XCTAssertThrowsError(try manifest.validate(), "validate must reject a traversal path")

        let dest = try makeDir()
        defer { try? FileManager.default.removeItem(at: dest) }
        let parent = dest.deletingLastPathComponent()
        XCTAssertThrowsError(try ChunkStore.forDownloading(manifest: manifest, directory: dest),
                             "the store must refuse a path that escapes its directory")
        XCTAssertFalse(FileManager.default.fileExists(atPath: parent.appendingPathComponent("escape.bin").path),
                       "traversal must not have written outside the destination")
    }
}
