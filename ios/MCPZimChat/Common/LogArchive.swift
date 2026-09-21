// SPDX-License-Identifier: MIT
//
// Persistent rolling log archive. Every launch starts a new file in
// `Documents/debug-logs/YYYY-MM-DD_HH-mm-ss.log` and streams each
// `ChatSession.debug(...)` line into it. Appends are enqueued on a
// serial queue (ordering preserved) and written asynchronously — the
// caller never blocks on flash I/O. Once `write(2)` returns, the
// kernel owns the bytes, so a jetsam / crash still leaves the log on
// disk up to the last drained line for post-mortem inspection.
//
// LibraryView shows a "Past logs" list that reads this directory,
// and each row shares via `UIActivityViewController` so the user can
// AirDrop the file straight to their Mac (landing as a `.log` in
// `~/Downloads`) without the paste dance. Older files are pruned to
// keep the total footprint bounded.

import Foundation
import Observation
import OSLog

public final class LogArchive: @unchecked Sendable {
    public static let shared = LogArchive()

    private let queue = DispatchQueue(label: "org.mcpzim.LogArchive")
    private var currentURL: URL?
    private var handle: FileHandle?
    private let maxFiles = 20
    private let directoryOverride: URL?

    init(directory: URL? = nil) {
        directoryOverride = directory
        startNewSession()
    }

    // MARK: - Writing

    public func startNewSession() {
        queue.sync {
            try? handle?.close()
            handle = nil

            guard let dir = logsDirectory() else { return }
            try? FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true
            )

            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.timeZone = TimeZone.current
            df.dateFormat = "yyyy-MM-dd_HH-mm-ss"
            // Rapid cold launches must not truncate the previous session.
            let name = df.string(from: Date()) + "_" + UUID().uuidString.prefix(8) + ".log"
            let url = dir.appendingPathComponent(name)

            FileManager.default.createFile(atPath: url.path, contents: nil)
            currentURL = url
            handle = try? FileHandle(forWritingTo: url)

            // First line of every session log identifies the exact build —
            // pasted logs and crash-tail forensics are only comparable
            // across sessions when the build they came from is knowable.
            let info = Bundle.main
            let short = info.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
            let build = info.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
            if let data = "=== Zimfo \(short) (\(build)) ===\n".data(using: .utf8) {
                try? handle?.write(contentsOf: data)
            }

            pruneOldFilesLocked(keeping: maxFiles)
        }
    }

    public func append(_ line: String) {
        // Async on the same serial queue: ordering is preserved, but
        // the caller (often the main thread, mid-generation) no longer
        // blocks on a per-line flash write. Durability comes from the
        // kernel holding the data after `write(2)`, not from making
        // the caller wait for it.
        queue.async { [self] in
            guard let handle, let data = (line + "\n").data(using: .utf8) else { return }
            try? handle.write(contentsOf: data)
        }
    }

    /// Synchronous append. `append`'s async enqueue means a hard
    /// `abort(3)` inside llama.cpp loses the trailing lines (they were
    /// still queued). The startup trace must be on disk BEFORE the
    /// step it describes runs, so a SIGABRT / OOM-kill leaves the exact
    /// stage in the log for the next launch's `previousSessionUncleanTail`
    /// to surface.
    public func appendSync(_ line: String) {
        queue.sync { [self] in
            guard let handle, let data = (line + "\n").data(using: .utf8) else { return }
            try? handle.write(contentsOf: data)
        }
    }

    /// Startup crash breadcrumb. Writes (synchronously) to the
    /// persistent session log so a hard crash still leaves the exact
    /// stage on disk. Only compiled into the launch path — callers must
    /// not spam this at per-turn frequency.
    public func trace(_ line: String) {
        appendSync("[trace] " + line)
    }

    // MARK: - Unclean-exit detection

    /// Post-mortem breadcrumb: did the PREVIOUS session's log end in the
    /// middle of foreground work? Two on-device llama.cpp deaths
    /// (2026-07-02: one mid-`generate`, one mid-model-load) produced NO
    /// system crash report of any kind — this tail is the only evidence.
    /// Returns the last line of the previous run when it doesn't look
    /// like a normal background/terminate, so the host can surface
    /// "previous session died at: …" at every launch.
    ///
    /// Heuristic: iOS killing a BACKGROUNDED app is routine, so a tail
    /// containing "backgrounded" (the KV-drop line) or "terminating"
    /// counts as clean; anything else means we died while active.
    public func previousSessionUncleanTail(maxLines: Int = 3) -> String? {
        let files = allFiles()
        // files[0] is the session we just opened; [1] is the previous run.
        guard files.count >= 2 else { return nil }
        let prev = files[1]
        // Bounded tail read: a long session's log can run to many MB and
        // this fires on every launch — seek to the last 16 KB instead of
        // loading the whole file to inspect 3 lines. A mid-codepoint
        // start only garbles the first (discarded) line.
        guard let fh = try? FileHandle(forReadingFrom: prev) else { return nil }
        defer { try? fh.close() }
        let size = (try? fh.seekToEnd()) ?? 0
        let tailCap: UInt64 = 16 * 1024
        try? fh.seek(toOffset: size > tailCap ? size - tailCap : 0)
        let data = (try? fh.readToEnd()) ?? Data()
        let content = String(decoding: data, as: UTF8.self)
        let lines = content.split(separator: "\n", omittingEmptySubsequences: true)
        guard let last = lines.last else {
            return "\(prev.lastPathComponent): empty log — died before first line"
        }
        let l = last.lowercased()
        if l.contains("backgrounded") || l.contains("terminating") {
            return nil
        }
        let tail = lines.suffix(maxLines).joined(separator: "\n")
        return "\(prev.lastPathComponent):\n\(tail)"
    }

    // MARK: - Reading

    public func currentFileURL() -> URL? {
        queue.sync { currentURL }
    }

    /// One row of the past-logs list: URL plus the metadata the directory
    /// enumeration already prefetched, so views don't re-stat per row.
    public struct LogFileInfo: Sendable {
        public let url: URL
        public let modified: Date
        public let sizeBytes: Int64
    }

    /// All persisted log files with metadata, newest first. The
    /// `resourceValues` reads below are served from the enumeration's
    /// prefetched cache — one stat pass for the whole directory.
    public func allFileInfos() -> [LogFileInfo] {
        guard let dir = logsDirectory(),
              let entries = try? FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
              )
        else { return [] }
        return entries
            .filter { $0.pathExtension == "log" }
            .map { url -> LogFileInfo in
                let rv = try? url.resourceValues(
                    forKeys: [.contentModificationDateKey, .fileSizeKey])
                return LogFileInfo(
                    url: url,
                    modified: rv?.contentModificationDate ?? .distantPast,
                    sizeBytes: Int64(rv?.fileSize ?? 0)
                )
            }
            .sorted { $0.modified > $1.modified }
    }

    /// All persisted log files, newest first.
    public func allFiles() -> [URL] {
        allFileInfos().map(\.url)
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
            guard let dir = logsDirectory() else { return }
            try? FileManager.default.removeItem(at: dir)
        }
        startNewSession()
    }

    // MARK: - Helpers

    private func logsDirectory() -> URL? {
        if let directoryOverride { return directoryOverride }
        guard let docs = try? FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil, create: true
        ) else { return nil }
        return docs.appendingPathComponent("debug-logs", isDirectory: true)
    }

    private func pruneOldFilesLocked(keeping cap: Int) {
        guard let dir = logsDirectory(),
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

/// App Intents execute without ChatSession. Keep their diagnostics durable
/// and independently observable, including when the app starts in background.
/// Only lifecycle metadata belongs here: never question/title/location text.
@MainActor @Observable
final class SiriDiagnostics {
    static let shared = SiriDiagnostics()
    private static let logger = Logger(subsystem: "org.mcpzim.MCPZimChat", category: "Siri")
    private(set) var entries: [ChatSession.DebugEntry] = []
    @ObservationIgnored private let sink: (String) -> Void

    init(sink: @escaping (String) -> Void = { LogArchive.shared.appendSync($0) }) {
        self.sink = sink
    }

    func clear() { entries.removeAll() }

    func record(_ message: String) {
        let now = Date()
        entries.append(.init(timestamp: now, category: "Siri", message: message))
        if entries.count > 200 { entries.removeFirst(entries.count - 200) }
        sink("\(now.ISO8601Format()) [Siri] \(message)")
        Self.logger.notice("\(message, privacy: .public)")
    }
}

@MainActor
final class SiriInvocation {
    let id = UUID().uuidString
    private let action: String
    private let diagnostics: SiriDiagnostics
    private var awaitingParameter = false
    private var outcome = "returned"
    private let started = ProcessInfo.processInfo.systemUptime

    init(_ action: String, diagnostics: SiriDiagnostics? = nil) {
        self.action = action
        self.diagnostics = diagnostics ?? .shared
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        let os = ProcessInfo.processInfo.operatingSystemVersion
        emit("begin build=\(build) os=\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)")
    }

    func stage(_ name: String, count: Int? = nil) {
        awaitingParameter = name.hasPrefix("awaiting_")
        emit(name + (count.map { " count=\($0)" } ?? ""))
    }

    func failed(_ error: Error) {
        // Parameter requests may throw control-flow errors to resume in Siri.
        // Do not label those as retrieval failures or log localized input.
        outcome = error is CancellationError ? "cancelled" : (awaitingParameter ? "parameter_exit" : "failed")
        let typeName = String(String(reflecting: type(of: error)).prefix(120))
        emit("\(outcome) error_type=\(typeName) code=\((error as NSError).code)")
    }

    func end() {
        let ms = Int(max(0, ProcessInfo.processInfo.systemUptime - started) * 1000)
        emit("end outcome=\(outcome) elapsed_ms=\(ms)")
    }

    private func emit(_ event: String) {
        diagnostics.record("id=\(id) action=\(action) \(event)")
    }
}
