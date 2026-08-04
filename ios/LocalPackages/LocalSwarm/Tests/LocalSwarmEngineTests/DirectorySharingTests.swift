import XCTest
@testable import LocalSwarmEngine

/// Directory shares: `hostFiles(at:)` accepts directory URLs, expanding them
/// into `<dirname>/<subpath>` manifest items whose layout the receiving
/// `ChunkStore` recreates. (Validation has always accepted safe nested
/// relative paths; these tests pin the sending side and the round trip.)
final class DirectorySharingTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dir-sharing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    @discardableResult
    private func makeFile(_ relative: String, bytes: Int) throws -> URL {
        let url = root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        var data = Data(capacity: bytes)
        for i in 0..<bytes { data.append(UInt8((i &+ relative.utf8.count) & 0xFF)) }
        try data.write(to: url)
        return url
    }

    func testExpandForSharingWalksDirectoriesAndKeepsFilesFlat() throws {
        let flat = try makeFile("flat.bin", bytes: 64)
        try makeFile("voicepack/model.safetensors", bytes: 128)
        try makeFile("voicepack/styles/f1.json", bytes: 32)
        try makeFile("voicepack/.hidden", bytes: 8) // must be skipped
        let dir = root.appendingPathComponent("voicepack", isDirectory: true)

        let items = SwarmManager.expandForSharing([flat, dir])
        XCTAssertEqual(items.map(\.relativePath),
                       ["flat.bin",
                        "voicepack/model.safetensors",
                        "voicepack/styles/f1.json"],
                       "directory entries sorted by relative path; hidden files skipped")
    }

    func testExpandedOrderIsDeterministicRegardlessOfEnumeration() throws {
        try makeFile("pack/b.bin", bytes: 16)
        try makeFile("pack/a/z.bin", bytes: 16)
        try makeFile("pack/a/a.bin", bytes: 16)
        let dir = root.appendingPathComponent("pack", isDirectory: true)

        let first = SwarmManager.expandForSharing([dir]).map(\.relativePath)
        let second = SwarmManager.expandForSharing([dir]).map(\.relativePath)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first, first.sorted(),
                       "sorted expansion → identical content-addressed swarmID on every host")
    }

    func testManifestFromDirectoryItemsValidatesAndRoundTrips() throws {
        try makeFile("pack/model.bin", bytes: 5000)
        try makeFile("pack/styles/f1.json", bytes: 700)
        let dir = root.appendingPathComponent("pack", isDirectory: true)

        let items = SwarmManager.expandForSharing([dir])
        let (manifest, ordered) = try Chunker.buildManifest(name: "pack", items: items,
                                                            chunkSize: 4096)
        XCTAssertNoThrow(try manifest.validate(),
                         "nested relative paths must pass untrusted-manifest validation")
        XCTAssertEqual(manifest.files.map(\.path),
                       ["pack/model.bin", "pack/styles/f1.json"])

        // Seed from the originals, download into a fresh directory, and check
        // the tree is recreated byte-for-byte.
        let seeder = ChunkStore.forSeeding(manifest: manifest, sourceURLs: ordered)
        let downloadDir = root.appendingPathComponent("received", isDirectory: true)
        let receiver = try ChunkStore.forDownloading(manifest: manifest, directory: downloadDir)
        for index in 0..<manifest.chunkCount {
            let data = try seeder.readChunk(index)
            XCTAssertTrue(try receiver.writeChunk(index, data: data))
        }
        XCTAssertTrue(receiver.isComplete)
        for (file, source) in zip(manifest.files, ordered) {
            let received = downloadDir.appendingPathComponent(file.path)
            XCTAssertEqual(try Data(contentsOf: received), try Data(contentsOf: source),
                           "\(file.path) must arrive intact in its subdirectory")
        }
    }

    func testManifestCacheDistinguishesLayouts() throws {
        ManifestCache.clear()
        defer { ManifestCache.clear() }
        let file = try makeFile("pack/model.bin", bytes: 4096)

        // Same file shared flat vs. as part of its directory: different
        // relative paths, so the cache must never alias one to the other.
        let flatItems = [ShareItem(url: file)]
        let dirItems = [ShareItem(url: file, relativePath: "pack/model.bin")]

        let flat = try Chunker.buildManifest(name: "n", items: flatItems)
        ManifestCache.store(manifest: flat.manifest, ordered: flat.orderedURLs,
                            name: "n", items: flatItems)

        XCTAssertNotNil(ManifestCache.lookup(name: "n", items: flatItems))
        XCTAssertNil(ManifestCache.lookup(name: "n", items: dirItems),
                     "layout is part of the cache identity")

        let nested = try Chunker.buildManifest(name: "n", items: dirItems)
        XCTAssertNotEqual(flat.manifest.swarmID, nested.manifest.swarmID,
                          "relative path feeds the content address")
    }
}
