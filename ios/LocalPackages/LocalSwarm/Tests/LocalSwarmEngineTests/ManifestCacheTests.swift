import XCTest
@testable import LocalSwarmEngine

final class ManifestCacheTests: XCTestCase {

    override func setUp() {
        super.setUp()
        ManifestCache.clear()
    }

    override func tearDown() {
        ManifestCache.clear()
        super.tearDown()
    }

    private func makeFile(_ name: String, bytes: Int, in dir: URL) throws -> URL {
        let url = dir.appendingPathComponent(name)
        try Data((0..<bytes).map { UInt8(($0 * 31 + 7) % 251) }).write(to: url)
        return url
    }

    func testHitReturnsIdenticalManifestWithoutRehash() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = try makeFile("a.bin", bytes: 3 * 1024 * 1024 + 123, in: dir)

        let (manifest, ordered) = try Chunker.buildManifest(name: "a.bin", fileURLs: [file])
        ManifestCache.store(manifest: manifest, ordered: ordered, name: "a.bin", urls: [file])

        let hit = ManifestCache.lookup(name: "a.bin", urls: [file])
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.0, manifest, "cached manifest must be byte-identical")
        XCTAssertEqual(hit?.1, ordered)
    }

    func testContentChangeMisses() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = try makeFile("a.bin", bytes: 1024 * 1024, in: dir)

        let (manifest, ordered) = try Chunker.buildManifest(name: "a.bin", fileURLs: [file])
        ManifestCache.store(manifest: manifest, ordered: ordered, name: "a.bin", urls: [file])

        // Change size (and content) — the key must change.
        try Data(repeating: 7, count: 2 * 1024 * 1024).write(to: file)
        XCTAssertNil(ManifestCache.lookup(name: "a.bin", urls: [file]))
    }

    func testMtimeChangeMisses() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = try makeFile("a.bin", bytes: 512 * 1024, in: dir)

        let (manifest, ordered) = try Chunker.buildManifest(name: "a.bin", fileURLs: [file])
        ManifestCache.store(manifest: manifest, ordered: ordered, name: "a.bin", urls: [file])

        // Same size, new mtime — treated as changed (cheap and conservative).
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(120)], ofItemAtPath: file.path)
        XCTAssertNil(ManifestCache.lookup(name: "a.bin", urls: [file]))
    }

    func testMissingSourceFileMisses() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = try makeFile("a.bin", bytes: 256 * 1024, in: dir)

        let (manifest, ordered) = try Chunker.buildManifest(name: "a.bin", fileURLs: [file])
        ManifestCache.store(manifest: manifest, ordered: ordered, name: "a.bin", urls: [file])

        try FileManager.default.removeItem(at: file)
        XCTAssertNil(ManifestCache.lookup(name: "a.bin", urls: [file]))
    }

    func testLayoutIsPartOfCacheIdentity() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = try makeFile("model.bin", bytes: 4096, in: dir)

        // The same file shared flat vs. inside a folder: different relative
        // paths, so the cache must never alias one to the other.
        let flatItems = [ShareItem(url: file)]
        let dirItems = [ShareItem(url: file, relativePath: "styles/model.bin")]

        let flat = try Chunker.buildManifest(name: "n", sources: [(url: file, path: "model.bin")])
        ManifestCache.store(manifest: flat.manifest, ordered: flat.orderedURLs,
                            name: "n", items: flatItems)

        XCTAssertNotNil(ManifestCache.lookup(name: "n", items: flatItems))
        XCTAssertNil(ManifestCache.lookup(name: "n", items: dirItems),
                     "layout is part of the cache identity")

        let nested = try Chunker.buildManifest(name: "n", sources: [(url: file, path: "styles/model.bin")])
        XCTAssertNotEqual(flat.manifest.swarmID, nested.manifest.swarmID,
                          "relative path feeds the content address")
    }

    func testCachePrunesOldEntriesToConfiguredLimit() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = try makeFile("model.bin", bytes: 4096, in: dir)
        let (manifest, ordered) = try Chunker.buildManifest(name: "model", fileURLs: [file])

        for index in 0..<(ManifestCache.maxEntries + 5) {
            ManifestCache.store(
                manifest: manifest,
                ordered: ordered,
                name: "model-\(index)",
                urls: [file]
            )
        }

        XCTAssertEqual(ManifestCache.entryCount, ManifestCache.maxEntries)
    }
}
