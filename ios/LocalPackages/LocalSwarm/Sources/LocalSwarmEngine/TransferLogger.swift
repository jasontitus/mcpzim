import Foundation
import os

/// Records transfer throughput to a CSV file (for offline analysis) and to the
/// unified log. Like `DiagLog`, this is verbose metadata (file names, swarm IDs,
/// rates), so it is gated on `DiagLog.enabled` (debug-only by default) and the
/// CSV lives in Caches — not Documents — so it isn't exposed via backups/sharing.
///
/// Pull from a device with:
///   `xcrun devicectl device copy from --device <UDID> \
///       --domain-type appDataContainer --domain-identifier <bundle-id> \
///       --source Library/Caches/LocalSwarmLogs/transfers.csv --destination .`
public final class TransferLogger: @unchecked Sendable {
    public static let shared = TransferLogger()

    private let queue = DispatchQueue(label: "com.localswarm.log")
    private let log = Logger(subsystem: "com.localswarm", category: "throughput")
    private var handle: FileHandle?
    private var openAttempted = false
    public private(set) var fileURL: URL?

    private static let header =
        "iso_time,event,transport,role,swarm_id,name,completed_bytes,total_bytes,bytes_per_sec,mbps,peers,elapsed_sec\n"

    private init() {
        let base = (try? FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        fileURL = base.appendingPathComponent("LocalSwarmLogs/transfers.csv")
    }

    public func record(event: String,
                       status: TransferStatus,
                       transport: Transport,
                       elapsed: TimeInterval) {
        guard DiagLog.enabled else { return }
        let mbps = status.bytesPerSecond / 1_000_000
        log.info("""
        \(event, privacy: .public) \(transport.rawValue, privacy: .public) \
        \(status.role.rawValue, privacy: .public) "\(status.name, privacy: .private)" \
        \(String(format: "%.1f", mbps), privacy: .public) MB/s \
        \(status.completedBytes, privacy: .public)/\(status.totalBytes, privacy: .public)B \
        peers=\(status.connectedPeers, privacy: .public) t=\(String(format: "%.1f", elapsed), privacy: .public)s
        """)

        let line = "\(Self.iso())," +
            "\(Self.csv(event)),\(transport.rawValue),\(status.role.rawValue)," +
            "\(Self.csv(status.swarmID)),\(Self.csv(status.name))," +
            "\(status.completedBytes),\(status.totalBytes)," +
            "\(Int(status.bytesPerSecond)),\(String(format: "%.2f", mbps))," +
            "\(status.connectedPeers),\(String(format: "%.2f", elapsed))\n"
        let data = Data(line.utf8)
        queue.async { [weak self] in
            guard let self = self else { return }
            self.openIfNeeded()
            _ = try? self.handle?.write(contentsOf: data)
        }
    }

    /// Lazily creates the CSV on first enabled write (so nothing is written when
    /// diagnostics are disabled). Runs on `queue`.
    private func openIfNeeded() {
        guard !openAttempted, let url = fileURL else { return }
        openAttempted = true
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            try? Self.header.write(to: url, atomically: true, encoding: .utf8)
        }
        handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
    }

    /// RFC-4180 CSV field escaping.
    private static func csv(_ field: String) -> String {
        guard field.contains(",") || field.contains("\"") || field.contains("\n") || field.contains("\r") else {
            return field
        }
        return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    private static func iso() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }
}
