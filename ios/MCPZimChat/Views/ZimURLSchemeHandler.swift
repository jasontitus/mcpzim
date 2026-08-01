// SPDX-License-Identifier: MIT
//
// Bridges `zim://<filename>/<entry-path>` URLs to the host app's open
// `LibzimReader`s, so a WKWebView can browse a ZIM as if it were a
// plain http site. No HTTP server, no network — every asset (HTML, CSS,
// JS, PNG tiles) resolves to an in-process libzim read.

import Foundation
import WebKit
import MCPZimKit

final class ZimURLSchemeHandler: NSObject, WKURLSchemeHandler, @unchecked Sendable {
    /// `(zimFilename) -> ZimReader?`. Evaluated once per request so the
    /// set of open readers can change at runtime (add / remove ZIMs
    /// without rebuilding the WebView).
    typealias Lookup = @Sendable (String) -> (any ZimReader)?
    /// Callback for surface-level diagnostics — wired to `session.debug`
    /// so 404s / failed loads appear in the in-app debug pane rather
    /// than buffered stdout.
    typealias Log = @Sendable (String) -> Void

    static let scheme = "zim"

    private let lookup: Lookup
    private let log: Log

    /// libzim reads block on cluster decompression (tens of ms for a
    /// large entry). `start` arrives on the main thread — right when a
    /// mid-generation map open fans out dozens of tile/glyph requests —
    /// so the lookup+read runs here and only the `WKURLSchemeTask`
    /// callbacks hop back to main (WebKit requires task methods on the
    /// thread `start` arrived on). Concurrent: `ZimReader`
    /// implementations are documented thread-safe, so parallel tile
    /// decompresses are fine.
    private let readQueue = DispatchQueue(
        label: "org.mcpzim.ZimURLSchemeHandler.read",
        qos: .utility,
        attributes: .concurrent
    )

    /// Tasks whose `stop` arrived while their read was still in flight.
    /// WebKit raises if any `WKURLSchemeTask` method is called after
    /// `webView(_:stop:)`, so the completion re-checks (and consumes)
    /// membership before touching the task. Both `stop` and the
    /// completion run on the main thread, so plain mutation is safe.
    /// Keyed by `ObjectIdentifier` so finished tasks aren't retained.
    private var stoppedTasks = Set<ObjectIdentifier>()

    /// Aggregated GET counter — a single map open issues ~40 resource
    /// requests, and logging each one costs a debug-pane line plus a
    /// log-file append exactly while tiles decode. Log the first
    /// request and every 25th after that. Main-thread only (`start`
    /// always arrives there). 404s and font aliases still log per-hit —
    /// they're rare and high-signal.
    private var requestCount = 0

    init(lookup: @escaping Lookup, log: @escaping Log = { _ in }) {
        self.lookup = lookup
        self.log = log
    }

    private enum ReadOutcome {
        case success(HTTPURLResponse, Data)
        case failure(Error)
    }

    func webView(_ webView: WKWebView, start urlSchemeTask: WKURLSchemeTask) {
        guard let url = urlSchemeTask.request.url else {
            urlSchemeTask.didFailWithError(NSError(
                domain: "ZimURLSchemeHandler", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "missing URL"]))
            return
        }
        // `zim://<filename>/<path>` → host = filename, path = leading slash.
        let zimName = url.host ?? ""
        let entryPath = String(url.path.dropFirst()) // drop leading `/`

        guard let reader = lookup(zimName) else {
            log("404 ZIM '\(zimName)' not loaded (asked for '\(entryPath)')")
            urlSchemeTask.didFailWithError(NSError(
                domain: "ZimURLSchemeHandler", code: 404,
                userInfo: [NSLocalizedDescriptionKey: "ZIM '\(zimName)' not loaded"]))
            return
        }
        requestCount += 1
        if requestCount == 1 || requestCount.isMultiple(of: 25) {
            log("GET #\(requestCount) \(entryPath) from \(zimName)")
        }

        let log = self.log
        readQueue.async { [weak self] in
            let outcome = Self.performRead(
                reader: reader, zimName: zimName, entryPath: entryPath,
                url: url, log: log)
            DispatchQueue.main.async {
                guard let self else { return }
                // Stopped while the read was in flight — WebKit forbids
                // any further task callbacks. Consume the marker so the
                // set stays bounded (one completion per start).
                if self.stoppedTasks.remove(ObjectIdentifier(urlSchemeTask)) != nil {
                    return
                }
                switch outcome {
                case .success(let response, let data):
                    urlSchemeTask.didReceive(response)
                    urlSchemeTask.didReceive(data)
                    urlSchemeTask.didFinish()
                case .failure(let error):
                    urlSchemeTask.didFailWithError(error)
                }
            }
        }
    }

    private static func performRead(
        reader: any ZimReader,
        zimName: String,
        entryPath: String,
        url: URL,
        log: Log
    ) -> ReadOutcome {
        do {
            var entry = try reader.read(path: entryPath)
            // Font-glyph fallback. MapLibre requests glyphs at the
            // fontstack name the style declares — often `"Open Sans
            // Bold"` with spaces. streetzim (intentionally, see
            // `create_osm_zim.py:3381`) stores fonts under their
            // space-stripped names (`fonts/OpenSansBold/...`). So
            // the space-name request 404s and every codepoint falls
            // back to local rendering — ugly-looking map labels and
            // a flood of `Unable to load glyph range` WebView
            // warnings (real capture 2026-04-23 gist b49387f4, 38
            // entries from a single map view).
            //
            // If the literal lookup misses and the path is a fonts
            // entry with spaces, try the space-stripped variant
            // once. That's all streetzim-generated ZIMs actually
            // expect.
            if entry == nil,
               entryPath.hasPrefix("fonts/"),
               entryPath.contains(" ")
            {
                let stripped = entryPath.replacingOccurrences(of: " ", with: "")
                if let hit = try reader.read(path: stripped) {
                    entry = hit
                    log("fonts alias '\(entryPath)' → '\(stripped)'")
                }
            }
            guard let entry else {
                log("404 entry '\(entryPath)' not in '\(zimName)'")
                return .failure(NSError(
                    domain: "ZimURLSchemeHandler", code: 404,
                    userInfo: [NSLocalizedDescriptionKey: "entry '\(entryPath)' not in '\(zimName)'"]))
            }
            let mime = entry.mimetype.isEmpty ? "application/octet-stream" : entry.mimetype
            let response = HTTPURLResponse(
                url: url,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": mime,
                    "Content-Length": "\(entry.content.count)",
                    "Cache-Control": "public, max-age=86400",
                    // Some viewers fetch cross-resource JSON from sibling
                    // paths in the same ZIM; all go through this handler
                    // so same-origin, but set CORS loosely to be safe.
                    "Access-Control-Allow-Origin": "*",
                ]
            )!
            return .success(response, entry.content)
        } catch {
            return .failure(error)
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
        // The read for this task may still be in flight on `readQueue`;
        // record the stop (main thread, same as the completion hop) so
        // the completion never touches the task afterwards.
        stoppedTasks.insert(ObjectIdentifier(urlSchemeTask))
    }
}
