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

    init(lookup: @escaping Lookup, log: @escaping Log = { _ in }) {
        self.lookup = lookup
        self.log = log
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
        log("GET \(entryPath) from \(zimName)")

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
                urlSchemeTask.didFailWithError(NSError(
                    domain: "ZimURLSchemeHandler", code: 404,
                    userInfo: [NSLocalizedDescriptionKey: "entry '\(entryPath)' not in '\(zimName)'"]))
                return
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
            urlSchemeTask.didReceive(response)
            urlSchemeTask.didReceive(entry.content)
            urlSchemeTask.didFinish()
        } catch {
            urlSchemeTask.didFailWithError(error)
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
        // No per-task cancellation state to unwind; libzim reads are
        // synchronous on the caller's queue.
    }
}
