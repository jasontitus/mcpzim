// SPDX-License-Identifier: MIT
//
// Thumbnail-sized hero image / video extracted from a `get_article`
// tool trace, rendered inline above the assistant reply. Same
// expand-to-fullscreen / X-to-dismiss UX as `RouteWebView`.
//
// We don't parse the HTML: we just scan for the first `<img src=...>`
// (or `<video src=…>`) and hand the src off to a tiny WKWebView that
// loads it through the existing `zim://` scheme handler. Keeps the
// path offline and avoids the cost of pulling in SwiftSoup just for
// an `img` attribute grab.

import SwiftUI
import WebKit

@MainActor
struct HeroMediaView: View {
    let trace: ToolCallTrace
    @Environment(ChatSession.self) private var session
    @State private var presentFullscreen = false

    var body: some View {
        if let spec = resolveSpec() {
            SmallWebMedia(spec: spec, session: session)
                .frame(height: 220)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(alignment: .topTrailing) {
                    Button {
                        presentFullscreen = true
                    } label: {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.system(size: 14, weight: .semibold))
                            .padding(8)
                            .background(.thinMaterial, in: Circle())
                    }
                    .accessibilityLabel("Expand media")
                    .padding(8)
                }
                .padding(.top, 4)
                .fullScreenCover(isPresented: $presentFullscreen) {
                    FullscreenMedia(spec: spec, session: session) {
                        presentFullscreen = false
                    }
                }
        }
    }

    struct Spec: Equatable {
        let zimName: String
        /// Absolute `zim://` URL to the media asset.
        let mediaURL: String
        /// Optional caption extracted from the surrounding `<figure>`.
        let caption: String
        /// True if we think the asset is a `<video>` rather than an image.
        let isVideo: Bool
    }

    private func resolveSpec() -> Spec? {
        // Surface hero media for any article-fetching tool call
        // (including `get_article_section` which is the common path
        // when the model asks for the lead section). We parse the
        // RAW article HTML from the ZIM — the `text` field in
        // `get_article_section` is already HTML-stripped, so we
        // sidestep it by reading the article directly via the
        // library's ZIM reader.
        let articleTools: Set<String> = [
            "get_article", "get_article_section", "list_article_sections",
        ]
        guard articleTools.contains(trace.name) else { return nil }

        // The article path — from the tool call arguments — and the
        // zim filename — from the tool result — together identify
        // the ZIM entry we want to re-read.
        guard let argData = trace.arguments.data(using: .utf8),
              let argJSON = try? JSONSerialization.jsonObject(with: argData) as? [String: Any],
              let path = argJSON["path"] as? String, !path.isEmpty,
              let resData = trace.rawResult.data(using: .utf8),
              let resJSON = try? JSONSerialization.jsonObject(with: resData) as? [String: Any],
              let zim = resJSON["zim"] as? String, !zim.isEmpty,
              let entry = session.library.first(where: {
                  $0.url.lastPathComponent == zim && $0.isEnabled
              })
        else { return nil }

        guard let raw = try? entry.reader.read(path: path),
              let html = String(data: raw.content, encoding: .utf8)
        else { return nil }
        guard let (src, caption, isVideo) = Self.firstMedia(in: html) else { return nil }

        let base = "zim://\(zim)/"
        let absolute: String
        if src.hasPrefix("http://") || src.hasPrefix("https://") || src.hasPrefix("zim://") {
            absolute = src
        } else if src.hasPrefix("/") {
            absolute = base + String(src.dropFirst())
        } else {
            if let slash = path.lastIndex(of: "/") {
                absolute = base + path[..<slash] + "/" + src
            } else {
                absolute = base + src
            }
        }
        return Spec(zimName: zim, mediaURL: absolute,
                    caption: caption, isVideo: isVideo)
    }

    /// Very small regex scan for the first `<img>` or `<video>` tag.
    /// Returns the src, an optional caption (nearest `<figcaption>`),
    /// and whether the tag was a video.
    private static func firstMedia(in html: String) -> (src: String, caption: String, isVideo: Bool)? {
        // Try video first (less common, higher signal value).
        if let videoSrc = extractSrc(from: html, tag: "video") {
            return (videoSrc, extractCaption(from: html) ?? "", true)
        }
        if let imgSrc = extractSrc(from: html, tag: "img") {
            return (imgSrc, extractCaption(from: html) ?? "", false)
        }
        return nil
    }

    private static func extractSrc(from html: String, tag: String) -> String? {
        // Accept both `src="…"` and Wikipedia/Kiwix's lazy-load
        // variants (`data-src`, `srcset`). We scan tag-by-tag so we
        // can reject 1-pixel spacer images that Wikipedia uses for
        // layout.
        let tagPattern = #"<\#(tag)\b[^>]*>"#
        guard let re = try? NSRegularExpression(pattern: tagPattern) else { return nil }
        let ns = html as NSString
        let matches = re.matches(in: html, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            let openTag = ns.substring(with: m.range)
            let src = firstAttribute(in: openTag, names: ["src", "data-src"])
                ?? firstSrcsetURL(in: openTag)
            guard let s = src else { continue }
            // Skip 1×1 transparent placeholders.
            if s.contains("base64,") { continue }
            if Self.isLikelySpacer(openTag) { continue }
            return s
        }
        return nil
    }

    private static func firstAttribute(in tag: String, names: [String]) -> String? {
        for name in names {
            let pattern = #"\b\#(name)\s*=\s*["']([^"']+)["']"#
            if let r = tag.range(of: pattern, options: .regularExpression) {
                let part = String(tag[r])
                if let q = part.firstIndex(where: { $0 == "\"" || $0 == "'" }) {
                    let after = part.index(after: q)
                    if let end = part[after...].firstIndex(where: { $0 == "\"" || $0 == "'" }) {
                        return String(part[after..<end])
                    }
                }
            }
        }
        return nil
    }

    private static func firstSrcsetURL(in tag: String) -> String? {
        guard let r = tag.range(of: #"\bsrcset\s*=\s*["']([^"']+)["']"#, options: .regularExpression)
        else { return nil }
        let attr = String(tag[r])
        guard let q = attr.firstIndex(where: { $0 == "\"" || $0 == "'" }) else { return nil }
        let after = attr.index(after: q)
        guard let end = attr[after...].firstIndex(where: { $0 == "\"" || $0 == "'" }) else { return nil }
        let list = String(attr[after..<end])
        // srcset values look like "foo.jpg 1x, foo-2x.jpg 2x" — take the first URL.
        let first = list.split(separator: ",").first ?? Substring(list)
        return String(first.trimmingCharacters(in: .whitespaces).split(separator: " ").first ?? "")
    }

    /// Reject 1-pixel transparent spacers Wikipedia uses for layout
    /// (they have `width="1"` or `height="1"` or a `spacer` class).
    private static func isLikelySpacer(_ tag: String) -> Bool {
        if tag.range(of: #"\bwidth\s*=\s*["']?1["']?"#, options: .regularExpression) != nil { return true }
        if tag.range(of: #"\bheight\s*=\s*["']?1["']?"#, options: .regularExpression) != nil { return true }
        if tag.contains("spacer") { return true }
        return false
    }

    private static func extractCaption(from html: String) -> String? {
        let pattern = #"<figcaption[^>]*>([\s\S]*?)</figcaption>"#
        guard let r = html.range(of: pattern, options: .regularExpression) else { return nil }
        var inner = String(html[r])
        inner = inner.replacingOccurrences(of: #"<[^>]+>"#, with: "", options: .regularExpression)
        let trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

#if os(macOS)
import AppKit
private struct SmallWebMedia: NSViewRepresentable {
    let spec: HeroMediaView.Spec
    let session: ChatSession
    func makeNSView(context: Context) -> WKWebView { makeMediaWebView(spec: spec, session: session) }
    func updateNSView(_ nsView: WKWebView, context: Context) {}
}
#else
import UIKit
private struct SmallWebMedia: UIViewRepresentable {
    let spec: HeroMediaView.Spec
    let session: ChatSession
    func makeUIView(context: Context) -> WKWebView { makeMediaWebView(spec: spec, session: session) }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
}
#endif

@MainActor
private func makeMediaWebView(spec: HeroMediaView.Spec, session: ChatSession) -> WKWebView {
    let config = WKWebViewConfiguration()
    let handler = ZimURLSchemeHandler(
        lookup: { zimName in
            session.library.first { $0.url.lastPathComponent == zimName }?.reader
        },
        log: { msg in
            Task { @MainActor in session.debug(msg, category: "zim://") }
        }
    )
    config.setURLSchemeHandler(handler, forURLScheme: "zim")
    let webView = WKWebView(frame: .zero, configuration: config)
    #if canImport(UIKit)
    webView.isOpaque = false
    webView.backgroundColor = .clear
    #endif
    let html = mediaHTML(spec: spec)
    // Base URL matters so relative refs inside the HTML resolve
    // against the ZIM, not about:blank.
    let base = URL(string: "zim://\(spec.zimName)/")
    webView.loadHTMLString(html, baseURL: base)
    return webView
}

private func mediaHTML(spec: HeroMediaView.Spec) -> String {
    let body: String
    if spec.isVideo {
        body = """
        <video src="\(spec.mediaURL)" controls playsinline
               style="width:100%;height:100%;object-fit:contain;background:#000;"></video>
        """
    } else {
        body = """
        <img src="\(spec.mediaURL)"
             style="width:100%;height:100%;object-fit:cover;background:#eee;"/>
        """
    }
    return """
    <!doctype html>
    <html><head><meta name="viewport" content="width=device-width, initial-scale=1">
    <style>html,body{margin:0;padding:0;height:100%;background:#000;}</style>
    </head><body>\(body)</body></html>
    """
}

private struct FullscreenMedia: View {
    let spec: HeroMediaView.Spec
    let session: ChatSession
    let onDismiss: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.edgesIgnoringSafeArea(.all)
            SmallWebMedia(spec: spec, session: session)
                .edgesIgnoringSafeArea(.all)
            Button {
                onDismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.primary)
                    .padding(12)
                    .background(.thinMaterial, in: Circle())
            }
            .accessibilityLabel("Close media")
            .padding(.top, 12)
            .padding(.trailing, 12)
            if !spec.caption.isEmpty {
                VStack {
                    Spacer()
                    Text(spec.caption)
                        .font(.footnote)
                        .foregroundStyle(.white)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.55))
                }
                .edgesIgnoringSafeArea(.bottom)
            }
        }
    }
}
