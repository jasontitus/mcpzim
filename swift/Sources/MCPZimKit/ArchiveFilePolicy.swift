// SPDX-License-Identifier: MIT
import Foundation
import Darwin

/// File-system boundaries shared by archive import, downloads and App Intents.
public enum ArchiveFilePolicy {
    public static let disabledArchivesKey = "library.disabledArchives"

    public static func isSafeRelativePath(_ path: String) -> Bool {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return !parts.isEmpty && !path.contains("\\") && !path.contains("\0")
            && parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    public static func contains(_ file: URL, in directory: URL) -> Bool {
        let parent = canonicalPath(directory)
        let path = canonicalPath(file)
        return path.hasPrefix(parent.hasSuffix("/") ? parent : parent + "/")
    }

    private static func canonicalPath(_ url: URL) -> String {
        var cursor = url.standardizedFileURL
        var missing: [String] = []
        while cursor.path != "/", !FileManager.default.fileExists(atPath: cursor.path) {
            missing.append(cursor.lastPathComponent)
            cursor.deleteLastPathComponent()
        }
        cursor = cursor.resolvingSymlinksInPath()
        for component in missing.reversed() { cursor.appendPathComponent(component) }
        return cursor.standardizedFileURL.path
    }

    /// Documents paths change when iOS relocates an app container. External
    /// archives keep their full identity so equal filenames do not collide.
    public static func identity(_ file: URL, documents: URL?) -> String {
        if let documents, contains(file, in: documents) {
            return "documents/" + file.lastPathComponent
        }
        return file.standardizedFileURL.absoluteString
    }

    public static func validateDownload(_ file: URL, statusCode: Int?, filename: String) throws {
        guard statusCode == 200 || statusCode == 206,
              !filename.contains("/"), !filename.contains("\\"),
              filename.lowercased().hasSuffix(".zim") else { throw URLError(.badServerResponse) }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        let header = try handle.read(upToCount: 80) ?? Data()
        // openZIM header magic. This rejects HTTP error bodies and truncation;
        // libzim still validates the archive's tables when the reader opens.
        guard header.count == 80, header.prefix(4) == Data([0x5a, 0x49, 0x4d, 0x04])
        else { throw CocoaError(.fileReadCorruptFile) }
    }

    /// A staged file already on the destination volume is committed in one
    /// rename. Failure leaves the previous archive intact; never delete it first.
    public static func commit(staged: URL, to destination: URL) throws {
        guard staged.standardizedFileURL != destination.standardizedFileURL else { return }
        let result = staged.withUnsafeFileSystemRepresentation { source in
            destination.withUnsafeFileSystemRepresentation { target in
                Darwin.rename(source!, target!)
            }
        }
        guard result == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }
}
