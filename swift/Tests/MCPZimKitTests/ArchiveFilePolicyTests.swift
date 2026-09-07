import XCTest
@testable import MCPZimKit

final class ArchiveFilePolicyTests: XCTestCase {
    func testSharedAssetTraversalRejected() {
        for path in ["../x", "kokoro_mlx/../../x", "/kokoro_mlx/x", "kokoro_mlx//x", "kokoro_mlx/./x", "kokoro_mlx/..", "kokoro_mlx\\x", ""] {
            XCTAssertFalse(ArchiveFilePolicy.isSafeRelativePath(path), path)
        }
        XCTAssertTrue(ArchiveFilePolicy.isSafeRelativePath("kokoro_mlx/voices.npz"))
        XCTAssertTrue(ArchiveFilePolicy.isSafeRelativePath("supertonic_3/a.mlmodelc/weights/weight.bin"))
    }
    func testSiblingAndSymlinkAreNotOwnedDocuments() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let docs = root.appendingPathComponent("Documents")
        let outside = root.appendingPathComponent("Documents-extra")
        try FileManager.default.createDirectory(at: docs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        let link = docs.appendingPathComponent("external")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)
        XCTAssertTrue(ArchiveFilePolicy.contains(docs.appendingPathComponent("a.zim"), in: docs))
        XCTAssertFalse(ArchiveFilePolicy.contains(outside.appendingPathComponent("a.zim"), in: docs))
        XCTAssertFalse(ArchiveFilePolicy.contains(link.appendingPathComponent("a.zim"), in: docs))
        XCTAssertFalse(ArchiveFilePolicy.contains(docs.appendingPathComponent("../a.zim"), in: docs))
    }

    func testEnablementIdentitySurvivesContainerRelocation() {
        let first = URL(fileURLWithPath: "/app/one/Documents")
        let second = URL(fileURLWithPath: "/app/two/Documents")
        XCTAssertEqual(ArchiveFilePolicy.identity(first.appendingPathComponent("a.zim"), documents: first),
                       ArchiveFilePolicy.identity(second.appendingPathComponent("a.zim"), documents: second))
        XCTAssertNotEqual(ArchiveFilePolicy.identity(URL(fileURLWithPath: "/external/a.zim"), documents: first),
                          ArchiveFilePolicy.identity(first.appendingPathComponent("a.zim"), documents: first))
    }

    func testHTTPErrorWrongFormatAndTruncatedDownloadsFail() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let header = Data([0x5a, 0x49, 0x4d, 0x04] + Array(repeating: 0, count: 76))
        try header.write(to: file)
        for code: Int? in [nil, 404, 500, 204] {
            XCTAssertThrowsError(try ArchiveFilePolicy.validateDownload(file, statusCode: code, filename: "a.zim"))
        }
        XCTAssertNoThrow(try ArchiveFilePolicy.validateDownload(file, statusCode: 200, filename: "a.zim"))
        XCTAssertNoThrow(try ArchiveFilePolicy.validateDownload(file, statusCode: 206, filename: "a.zim"))
        XCTAssertThrowsError(try ArchiveFilePolicy.validateDownload(file, statusCode: 200, filename: "../a.zim"))
        try Data("<html>Unavailable</html>".utf8).write(to: file)
        XCTAssertThrowsError(try ArchiveFilePolicy.validateDownload(file, statusCode: 200, filename: "a.zim"))
        try header.prefix(4).write(to: file)
        XCTAssertThrowsError(try ArchiveFilePolicy.validateDownload(file, statusCode: 200, filename: "a.zim"))
    }

    func testAtomicCommitFailurePreservesExistingArchive() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let old = dir.appendingPathComponent("a.zim")
        let staged = dir.appendingPathComponent("new.partial")
        try Data("old".utf8).write(to: old)
        XCTAssertThrowsError(try ArchiveFilePolicy.commit(staged: staged, to: old))
        XCTAssertEqual(try Data(contentsOf: old), Data("old".utf8))
        try Data("new".utf8).write(to: staged)
        try ArchiveFilePolicy.commit(staged: staged, to: old)
        XCTAssertEqual(try Data(contentsOf: old), Data("new".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.path))
    }

    func testMediaAttributeCannotInjectHTML() {
        let raw = "zim://a/x\" onerror=\"alert(1)<script>&"
        let escaped = ZimWebPolicy.escapeAttribute(raw)
        XCTAssertFalse(escaped.contains("\""))
        XCTAssertFalse(escaped.contains("<"))
        XCTAssertTrue(escaped.contains("&amp;"))
    }
}
