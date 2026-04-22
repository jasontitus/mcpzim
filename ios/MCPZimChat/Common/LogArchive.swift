// SPDX-License-Identifier: MIT
//
// Persistent rolling log archive. Every launch starts a new file in
// `Documents/debug-logs/YYYY-MM-DD_HH-mm-ss.log` and streams each
// `ChatSession.debug(...)` line into it. Crucially, the file is
// written synchronously on each append — so when iOS jetsams or the
// app crashes, the log up to the last line is already on disk for
// post-mortem inspection.
//
// LibraryView shows a "Past logs" list that reads this directory,
// and each row shares via `UIActivityViewController` so the user can
// AirDrop the file straight to their Mac (landing as a `.log` in
// `~/Downloads`) without the paste dance. Older files are pruned to
// keep the total footprint bounded.

import Foundation

public final class LogArchive: @unchecked Sendable {
    public static let shared = LogArchive()

    private let queue = DispatchQueue(label: "org.mcpzim.LogArchive")
    private var currentURL: URL?
    private var handle: FileHandle?
    private let maxFiles = 20

    private init() {
        startNewSession()
    }

    // MARK: - Writing

    public func startNewSession() {
        queue.sync {
            try? handle?.close()
            handle = nil

            guard let dir = Self.logsDirectory() else { return }
            try? FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true
            )

            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.timeZone = TimeZone.current
            df.dateFormat = "yyyy-MM-dd_HH-mm-ss"
            let name = df.string(from: Date()) + ".log"
            let url = dir.appendingPathComponent(name)

            FileManager.default.createFile(atPath: url.path, contents: nil)
            currentURL = url
            handle = try? FileHandle(forWritingTo: url)

            pruneOldFilesLocked(keeping: maxFiles)
        }
    }

    public func append(_ line: String) {
        queue.sync {
            guard let handle, let data = (line + "\n").data(using: .utf8) else { return }
            try? handle.write(contentsOf: data)
        }
    }

    // MARK: - Reading

    public func currentFileURL() -> URL? {
        queue.sync { currentURL }
    }

    /// All persisted log files, newest first.
    public func allFiles() -> [URL] {
        guard let dir = Self.logsDirectory(),
              let entries = try? FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
              )
        else { return [] }
        return entries
            .filter { $0.pathExtension == "log" }
            .sorted { lhs, rhs in
                let l = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let r = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return l > r
            }
    }

    public func read(_ url: URL) -> String {
        (try? String(contentsOf: url, encoding: .utf8)) ?? ""
    }

    public func fileSize(_ url: URL) -> Int64 {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).flatMap { Int64($0) } ?? 0
    }

    public func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    public func delete(_ url: URL) {
        queue.sync { try? FileManager.default.removeItem(at: url) }
    }

    public func deleteAll() {
        queue.sync {
            try? handle?.close()
            handle = nil
            guard let dir = Self.logsDirectory() else { return }
            try? FileManager.default.removeItem(at: dir)
        }
        startNewSession()
    }

    // MARK: - Helpers

    private static func logsDirectory() -> URL? {
        guard let docs = try? FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil, create: true
        ) else { return nil }
        return docs.appendingPathComponent("debug-logs", isDirectory: true)
    }

    private func pruneOldFilesLocked(keeping cap: Int) {
        guard let dir = Self.logsDirectory(),
              let entries = try? FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
              )
        else { return }
        let logs = entries
            .filter { $0.pathExtension == "log" }
            .sorted { lhs, rhs in
                let l = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let r = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return l > r
            }
        guard logs.count > cap else { return }
        for stale in logs.suffix(from: cap) {
            try? FileManager.default.removeItem(at: stale)
        }
    }
}
