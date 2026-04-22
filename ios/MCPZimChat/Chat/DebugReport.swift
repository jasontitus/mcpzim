// SPDX-License-Identifier: MIT
//
// "Send Debug Report" packaging + transport.
//
// Problem: users hit bad results on device and we have no durable,
// after-the-fact way to see what the model / tools / trims were
// doing at the time. The iOS debug pane shows the live log but
// evaporates as soon as the app restarts, and the chat transcript
// alone doesn't explain *why* a response went sideways.
//
// Solution: a single tap in the debug pane packages the current
// session's messages + the live debug log into a JSON blob and
// emits it through `os.Logger` as a base64-encoded stream of
// chunked lines the Mac-side `ios/scripts/mcp-report.sh` picks up
// from the `idevicesyslog` buffer that `mcp-logs.sh` already runs.
// Zero new transport, works over USB whenever the Mac has the
// streamer attached. After emission we clear the debug log so the
// next bad query has a clean report.
//
// The chunks land as lines like:
//
//   [DebugReport BEGIN hash=AB12 total=42 size=18392 time=…]
//   [DebugReport seq=1/42 hash=AB12] <base64 chunk>
//   …
//   [DebugReport END hash=AB12]
//
// Each chunk stays under 500 printable chars so `os.Logger`'s
// per-line size limit (~1 KB formatted) doesn't truncate. If a
// chunk is dropped the Mac reassembler surfaces the gap.

import Foundation
import OSLog
#if canImport(CryptoKit)
import CryptoKit
#endif

private let reportLog = Logger(
    subsystem: "org.mcpzim.MCPZimChat",
    category: "DebugReport"
)

/// Gist-upload configuration lives in UserDefaults so the user can
/// paste their PAT once on device and forget about it. When unset,
/// `emitDebugReport()` falls back to syslog-only transport and the
/// Mac-side pickup works only if `mcp-logs.sh` is streaming.
public enum DebugReportConfig {
    private static let tokenKey = "debug.report.githubToken"

    /// Short-lived GitHub PAT with `gist` scope. Stored in
    /// UserDefaults — personal-dev only; don't ship the app
    /// publicly with this in place.
    public static var githubToken: String? {
        get { UserDefaults.standard.string(forKey: tokenKey) }
        set {
            let defaults = UserDefaults.standard
            if let v = newValue, !v.isEmpty {
                defaults.set(v, forKey: tokenKey)
            } else {
                defaults.removeObject(forKey: tokenKey)
            }
        }
    }

    /// Marker included in the gist description so the Mac picker can
    /// filter the user's gists down to just our reports.
    public static let gistMarker = "mcpzim-debug-report"
}

/// On-disk / on-wire model. Deliberately flat + Codable so the Mac
/// reassembler doesn't need any of our Swift types to parse it —
/// `jq` + a browser work fine.
public struct SerializedDebugReport: Codable, Sendable {
    public let generatedAt: Date
    public let appBuild: String
    public let deviceTier: String
    public let selectedModelId: String
    public let messages: [Entry]
    public let debugEntries: [LogEntry]

    public struct Entry: Codable, Sendable {
        public let role: String
        public let text: String
        public let toolCalls: [Tool]
        public let startedAt: Date?
        public let finishedAt: Date?
        public struct Tool: Codable, Sendable {
            public let name: String
            public let arguments: String
            public let resultPreview: String
            public let error: String?
        }
    }

    public struct LogEntry: Codable, Sendable {
        public let timestamp: Date
        public let category: String
        public let message: String
    }
}

@MainActor
extension ChatSession {

    /// Serialize the session's current messages + debug log and emit
    /// the report in chunked lines via `os.Logger`. After emission,
    /// `debugEntries` is cleared so the next query produces a clean
    /// report. Returns the hash the caller can surface in a toast so
    /// the user can confirm to me which report to pull ("crappy
    /// results, hash AB12").
    @discardableResult
    public func emitDebugReport() -> String {
        let report = SerializedDebugReport(
            generatedAt: Date(),
            appBuild: Bundle.main.infoDictionary?[
                "CFBundleVersion"] as? String ?? "?",
            deviceTier: DeviceProfile.current.label,
            selectedModelId: selectedModel.id,
            messages: messages.map { m in
                SerializedDebugReport.Entry(
                    role: m.role.rawValue,
                    text: m.text,
                    toolCalls: m.toolCalls.map { t in
                        .init(
                            name: t.name,
                            arguments: t.arguments,
                            resultPreview: String(t.result.prefix(240)),
                            error: t.error
                        )
                    },
                    startedAt: m.startedAt,
                    finishedAt: m.finishedAt
                )
            },
            debugEntries: debugEntries.map { e in
                SerializedDebugReport.LogEntry(
                    timestamp: e.timestamp,
                    category: e.category,
                    message: e.message
                )
            }
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let json = (try? encoder.encode(report)) ?? Data()
        let b64 = json.base64EncodedString()
        let hash = shortHash(json)

        let chunkSize = 500
        let total = (b64.count + chunkSize - 1) / chunkSize
        let timestamp = ISO8601DateFormatter().string(from: Date())
        // Logger's OSLogMessage interpolation can't be concatenated
        // with `+`, so build the lines as plain Strings and pass
        // them through as a single `%{public}@` format specifier.
        let beginLine =
            "[DebugReport BEGIN hash=\(hash) total=\(total) "
            + "size=\(json.count) time=\(timestamp)]"
        reportLog.notice("\(beginLine, privacy: .public)")
        var idx = 0
        var start = b64.startIndex
        while start < b64.endIndex, idx < total {
            let end = b64.index(start, offsetBy: chunkSize,
                                limitedBy: b64.endIndex) ?? b64.endIndex
            let chunk = String(b64[start..<end])
            idx += 1
            let seqLine =
                "[DebugReport seq=\(idx)/\(total) hash=\(hash)] \(chunk)"
            reportLog.notice("\(seqLine, privacy: .public)")
            start = end
        }
        let endLine = "[DebugReport END hash=\(hash) chunks=\(idx)]"
        reportLog.notice("\(endLine, privacy: .public)")

        // If a GitHub PAT is configured, POST the same JSON as a
        // gist so the Mac can pull it from anywhere (not just when
        // `mcp-logs.sh` is running). The syslog transport stays as
        // the fallback for offline / missing-token scenarios.
        let gistTask: Task<String?, Never>? = {
            guard let token = DebugReportConfig.githubToken,
                  !token.isEmpty, !json.isEmpty
            else { return nil }
            return Task.detached { [hash] in
                await ChatSession.uploadAsGist(json: json, hash: hash, token: token)
            }
        }()
        if gistTask != nil {
            Task { [weak self] in
                if let url = await gistTask?.value {
                    await MainActor.run {
                        self?.debug("gist: \(url)", category: "DebugReport")
                    }
                }
            }
        }

        // Clear the debug log so the next bad query produces a clean
        // report. We keep the messages themselves — the user may want
        // to retry within the same conversation.
        debugEntries.removeAll()
        return hash
    }

    /// POST the serialized report to gist.github.com. Returns the
    /// gist's html_url on success, nil on any error (logged but not
    /// surfaced — the syslog transport is always in effect too).
    ///
    /// Gists expose a multi-file shape; we store the report under
    /// `report.json` so `gh gist view --filename report.json` works
    /// without guessing.
    nonisolated private static func uploadAsGist(
        json: Data, hash: String, token: String
    ) async -> String? {
        guard let content = String(data: json, encoding: .utf8) else { return nil }
        struct GistFile: Encodable {
            let content: String
        }
        struct GistBody: Encodable {
            let description: String
            let `public`: Bool
            let files: [String: GistFile]
        }
        let body = GistBody(
            description: "\(DebugReportConfig.gistMarker) \(hash)",
            public: false,   // secret (unlisted) — still URL-shareable
            files: ["report.json": GistFile(content: content)]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        guard let payload = try? encoder.encode(body) else { return nil }
        var req = URLRequest(url: URL(string: "https://api.github.com/gists")!)
        req.httpMethod = "POST"
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        req.setValue("MCPZimChat/1 DebugReport", forHTTPHeaderField: "User-Agent")
        req.httpBody = payload
        do {
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode)
            else {
                let body = String(data: data, encoding: .utf8) ?? "?"
                let msg = "gist upload failed: \(body)"
                reportLog.error("\(msg, privacy: .public)")
                return nil
            }
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let url = obj["html_url"] as? String
            {
                let line = "[DebugReport GIST hash=\(hash) url=\(url)]"
                reportLog.notice("\(line, privacy: .public)")
                return url
            }
            return nil
        } catch {
            reportLog.error("gist upload threw: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    private func shortHash(_ data: Data) -> String {
        #if canImport(CryptoKit)
        let digest = SHA256.hash(data: data)
        return digest.prefix(4)
            .map { String(format: "%02X", $0) }
            .joined()
        #else
        // Fallback — sum of bytes. Collisions are fine for a
        // disambiguator; we just need a tag.
        var sum: UInt32 = 0
        for b in data { sum &+= UInt32(b) }
        return String(format: "%08X", sum)
        #endif
    }
}
