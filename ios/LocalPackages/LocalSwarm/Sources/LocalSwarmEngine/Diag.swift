import Foundation

/// Diagnostics sink. When enabled, writes each line to stderr *and* appends it to
/// `<Caches>/localswarm-diag.log` with a timestamp.
///
/// Verbose diagnostics (file names, swarm IDs, rates) are sensitive metadata and
/// can grow unbounded, so they are **off in release** and on only in debug builds.
/// The host app can flip `DiagLog.enabled` at runtime. Logs live in Caches (not
/// Documents) so they aren't exposed via file sharing / backups.
enum DiagLog {
#if DEBUG
    nonisolated(unsafe) static var enabled = true
#else
    nonisolated(unsafe) static var enabled = false
#endif

    private static let queue = DispatchQueue(label: "localswarm.diag")
    nonisolated(unsafe) private static var handle: FileHandle?
    nonisolated(unsafe) private static var opened = false
    /// Rotate (truncate to empty) once the log passes this size, so diagnostics
    /// can't grow without bound. `written` is tracked on `queue`.
    private static let maxBytes = 8 << 20
    nonisolated(unsafe) private static var written = 0

    /// `DateFormatter` is not thread-safe; only ever touched on `queue`.
    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    static let fileURL: URL? = {
        guard let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else { return nil }
        return dir.appendingPathComponent("localswarm-diag.log")
    }()

    /// Snapshot of the on-disk log for in-app display. Read on `queue` so it
    /// never races the writer. May lag the very newest line by one async hop.
    static func snapshot() -> String {
        queue.sync {
            guard let url = fileURL, let data = try? Data(contentsOf: url) else { return "" }
            return String(decoding: data, as: UTF8.self)
        }
    }

    /// Wipe the log — both the on-disk file and the write cursor.
    static func clear() {
        queue.sync {
            written = 0
            if let handle {
                try? handle.truncate(atOffset: 0)
                try? handle.seek(toOffset: 0)
            } else if let url = fileURL {
                try? Data().write(to: url)
            }
        }
    }

    static func write(_ message: String) {
        guard enabled else { return }
        let now = Date()
        // Format + emit on the serial queue so the formatter stays single-threaded
        // and stderr/file writes never interleave.
        queue.async {
            let data = Data("[\(stamp.string(from: now))] [swarm] \(message)\n".utf8)
            FileHandle.standardError.write(data)
            if !opened {
                opened = true
                if let url = fileURL {
                    if !FileManager.default.fileExists(atPath: url.path) {
                        FileManager.default.createFile(atPath: url.path, contents: nil)
                    }
                    handle = try? FileHandle(forWritingTo: url)
                    written = Int((try? handle?.seekToEnd()) ?? 0)
                }
            }
            if let handle {
                _ = try? handle.write(contentsOf: data)
                written += data.count
                if written > maxBytes {
                    try? handle.truncate(atOffset: 0)
                    _ = try? handle.seek(toOffset: 0)
                    written = 0
                }
            }
        }
    }
}

/// Writes a diagnostic line to stderr and the on-disk diag log (debug builds).
func swarmDiag(_ message: String) {
    DiagLog.write(message)
}

/// Public diagnostics entry point so the host app's UI can log into the same file.
public func swarmLog(_ message: String) {
    DiagLog.write(message)
}

/// Public facade over the diagnostics log so the host app can show it in a Debug
/// screen (read it, clear it, toggle capture) without reaching into the engine.
public enum SwarmDiagnostics {
    /// Whether verbose diagnostics are being captured. On by default in Debug
    /// builds; the Debug screen can flip it on in Release for field debugging.
    public static var enabled: Bool {
        get { DiagLog.enabled }
        set { DiagLog.enabled = newValue }
    }

    /// Current contents of the on-disk diagnostics log (oldest first).
    public static func snapshot() -> String { DiagLog.snapshot() }

    /// Clear the diagnostics log.
    public static func clear() { DiagLog.clear() }
}
