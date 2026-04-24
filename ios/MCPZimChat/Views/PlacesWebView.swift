// SPDX-License-Identifier: MIT
//
// Map display for "nearby" tool calls — `near_named_place`, `near_places`,
// `nearby_stories`, and `nearby_stories_at_place`. Parses the list of
// places (`results` or `stories`) out of the tool's raw payload and
// renders them as pins + a coverage-radius ring on top of the embedded
// streetzim viewer via its `window.streetzimRouting.showPlaces` +
// `showRadiusRing` hooks.
//
// Shares the same `zim://` scheme handler and JS-bridge infrastructure
// that `RouteWebView` uses — just a different injection payload.

import SwiftUI
import WebKit
import MCPZimKit

/// `true` if the trace is one of the four "nearby" tools AND it
/// returned at least one geocoded place. Other traces should not
/// light up the places map (e.g. `what_is_here` returns a single
/// named place but we already have `show_map` for that).
func traceHasPlaces(_ trace: ToolCallTrace) -> Bool {
    guard trace.succeeded else { return false }
    let placesTools: Set<String> = [
        "near_named_place", "near_places",
        "nearby_stories", "nearby_stories_at_place",
    ]
    guard placesTools.contains(trace.name) else { return false }
    return !parsePlaces(from: trace).places.isEmpty
}

/// Thin adapter — the real `parsePlacesJSON` + `PlacesPayload` type
/// live in `MCPZimKit` so they can be exercised by `swift test`.
/// Here we just route `trace.rawResult` through and return the kit
/// type verbatim. Also converts the kit payload's `origin`
/// `(lat, lon)?` tuple into the iOS-side tuple form.
func parsePlaces(from trace: ToolCallTrace) -> PlacesPayload {
    return parsePlacesJSON(rawResult: trace.rawResult)
}

/// Tap target for the "Read Wikipedia article" affordance — popped
/// into a `.sheet(item:)` so the WKWebView hosting the article only
/// renders while the sheet is up. `Identifiable` so SwiftUI can
/// drive the presentation from the optional binding.
struct ArticleSheetIntent: Identifiable, Equatable {
    let id = UUID()
    let zim: String       // e.g. "wikipedia_en_all_maxi_2025-10.zim"
    let path: String      // e.g. "A/Palo_Alto_History_Museum"
    let title: String     // display label for the navigation bar
}

struct PlacesWebView: View {
    let trace: ToolCallTrace
    @Environment(ChatSession.self) private var session
    @State private var presentFullscreen: Bool = false
    @State private var presentList: Bool = false
    @State private var articleSheet: ArticleSheetIntent? = nil
    /// Which place the list-view row tap pushed focus onto. We use a
    /// `UUID` stamp (vs a plain `Int`) so re-tapping the same row
    /// counts as a fresh focus intent — SwiftUI's `.onChange` diff
    /// ignores reassignments to the same value.
    @State private var focus: FocusIntent? = nil

    /// Ties a target place index to a stamp so repeat selections of
    /// the same row still round-trip through `updateUIView`.
    struct FocusIntent: Equatable {
        let idx: Int
        let stamp: UUID
    }

    /// Resolve which Wikipedia ZIM to load the article from. Takes
    /// the first `.wikipedia` in the library so mdwiki + enwiki
    /// coexistence stays deterministic, with en-wiki preferred by
    /// load order.
    private var wikipediaZimName: String {
        session.library
            .first(where: { $0.isEnabled && $0.reader.kind == .wikipedia })?
            .url.lastPathComponent ?? "wikipedia"
    }

    private var payload: PlacesPayload { parsePlaces(from: trace) }

    private var userLocation: (lat: Double, lon: Double)? { session.currentLocation }

    var body: some View {
        if let spec = resolveSpec(userLocation: userLocation) {
            PlacesWebContainer(spec: spec, session: session, payload: payload, focus: focus)
                .frame(height: 340)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(alignment: .bottomLeading) {
                    Button {
                        presentFullscreen = true
                    } label: {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.system(size: 14, weight: .semibold))
                            .padding(8)
                            .background(.thinMaterial, in: Circle())
                    }
                    .accessibilityLabel("Expand map")
                    .padding(8)
                }
                .overlay(alignment: .bottomTrailing) {
                    // Mirror `RouteWebView`'s Directions button — a one-
                    // tap reveal of the places as a scannable list,
                    // since the map gives the spatial read and this
                    // gives the textual one. Only shown when there's
                    // actually a list to read.
                    if !payload.places.isEmpty {
                        Button {
                            presentList = true
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "list.bullet")
                                Text("List (\(payload.places.count))")
                            }
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Color.accentColor, in: Capsule())
                            .shadow(color: .black.opacity(0.25), radius: 4, y: 2)
                        }
                        .accessibilityLabel("List all \(payload.places.count) places")
                        .padding(12)
                    }
                }
                .padding(.top, 4)
                .onAppear {
                    if session.currentLocation == nil {
                        session.refreshLocationIfStale()
                    }
                }
                #if os(iOS)
                .fullScreenCover(isPresented: $presentFullscreen) {
                    PlacesFullscreen(spec: spec, session: session, payload: payload) {
                        presentFullscreen = false
                    }
                }
                #else
                .sheet(isPresented: $presentFullscreen) {
                    PlacesFullscreen(spec: spec, session: session, payload: payload) {
                        presentFullscreen = false
                    }
                }
                #endif
                .sheet(isPresented: $presentList) {
                    PlacesListView(
                        payload: payload,
                        onSelect: { idx in
                            focus = FocusIntent(idx: idx, stamp: UUID())
                            presentList = false
                        },
                        onReadArticle: { place in
                            presentList = false
                            if let path = place.wikiPath {
                                articleSheet = ArticleSheetIntent(
                                    zim: wikipediaZimName,
                                    path: path,
                                    title: place.wikiTitle ?? place.label
                                )
                            }
                        },
                        onDirections: { place in
                            // Close the list first so the route bubble
                            // that `triggerDirectionsToCoord` appends
                            // is what the user sees next.
                            presentList = false
                            session.triggerDirectionsToCoord(
                                name: place.label,
                                lat: place.lat,
                                lon: place.lon
                            )
                        },
                        onDismiss: { presentList = false }
                    )
                }
                .sheet(item: $articleSheet) { intent in
                    ArticleSheetView(intent: intent, session: session) {
                        articleSheet = nil
                        // Clear session-level intent so the pop from
                        // popup-driven path doesn't re-fire on the
                        // next view update.
                        session.articleSheetIntent = nil
                    }
                }
                // Popup-button path: when the in-webview "Read article"
                // posts through the JS bridge, it ends up setting
                // `session.articleSheetIntent`. Mirror that into this
                // view's local `articleSheet` @State so the same
                // sheet surface handles both list-tap + popup-tap.
                .onChange(of: session.articleSheetIntent) { _, new in
                    guard let new else { return }
                    articleSheet = ArticleSheetIntent(
                        zim: new.zim, path: new.path, title: new.title
                    )
                }
        }
    }

    private func resolveSpec(userLocation: (lat: Double, lon: Double)?) -> PlacesSpec? {
        // Pick the streetzim ZIM that covers the bounding box of the
        // returned places. Prefer the one the trace was dispatched
        // against (its result's `zim` field), fall back to the first
        // enabled streetzim in the library.
        var zimFromResult: String? = nil
        if let data = trace.rawResult.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let z = json["zim"] as? String,
           session.library.contains(where: { $0.url.lastPathComponent == z && $0.isEnabled })
        {
            zimFromResult = z
        }
        let entry: ChatSession.LibraryEntry
        if let name = zimFromResult,
           let e = session.library.first(where: { $0.url.lastPathComponent == name && $0.isEnabled })
        {
            entry = e
        } else if let fallback = session.library.first(where: {
            $0.isEnabled && $0.reader.kind == .streetzim
        }) {
            entry = fallback
        } else {
            return nil
        }
        return PlacesSpec(
            zimName: entry.url.lastPathComponent,
            mainPath: "index.html",
            userLocation: userLocation
        )
    }
}

/// Separate spec type so PlacesWebView can live independently from
/// RouteWebView's `Spec` (different content, same URL scheme handler).
struct PlacesSpec: Equatable {
    let zimName: String
    let mainPath: String
    let userLocation: (lat: Double, lon: Double)?

    static func == (a: PlacesSpec, b: PlacesSpec) -> Bool {
        a.zimName == b.zimName
            && a.mainPath == b.mainPath
            && a.userLocation?.lat == b.userLocation?.lat
            && a.userLocation?.lon == b.userLocation?.lon
    }
}

/// Scrollable list of places — textual view of whatever the tool
/// returned, with name + kind/distance and (where available) a
/// Wikipedia-lead excerpt from the stories variant. Mirrors
/// `RouteWebView.DirectionsListView`'s UX so the two map-backed
/// bubble types have parallel affordances.
private struct PlacesListView: View {
    let payload: PlacesPayload
    let onSelect: (Int) -> Void
    let onReadArticle: (PlacesPayload.Place) -> Void
    let onDirections: (PlacesPayload.Place) -> Void
    let onDismiss: () -> Void

    var body: some View {
        NavigationStack {
            List {
                ForEach(Array(payload.places.enumerated()), id: \.offset) { idx, p in
                    HStack(alignment: .top, spacing: 10) {
                        // Wikipedia glyph on the LEFT so it sits in
                        // the natural "read more" position before the
                        // title — taps open a native sheet hosting a
                        // WKWebView of the article (not a chat turn).
                        // A grey placeholder keeps list alignment
                        // consistent when some rows have wiki paths
                        // and others don't.
                        if p.wikiPath != nil {
                            Button {
                                onReadArticle(p)
                            } label: {
                                Image(systemName: "book.circle.fill")
                                    .font(.title3)
                                    .foregroundStyle(Color.accentColor)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Read Wikipedia article")
                        } else {
                            Image(systemName: "book.circle")
                                .font(.title3)
                                .foregroundStyle(Color.secondary.opacity(0.3))
                                .accessibilityHidden(true)
                        }
                        Button {
                            onSelect(idx)
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(alignment: .firstTextBaseline, spacing: 8) {
                                    Text("\(idx + 1).")
                                        .font(.system(.footnote, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                    Text(p.label)
                                        .font(.body.weight(.semibold))
                                        .foregroundStyle(.primary)
                                    Spacer(minLength: 0)
                                    Image(systemName: "mappin.and.ellipse")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                if !p.description.isEmpty {
                                    Text(p.description)
                                        .font(.footnote)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 2)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        // Per-row Directions affordance — mirrors the
                        // ↪ Directions button in the map-pin popup so
                        // the user has the same one-tap route launch
                        // from either surface (textual list vs. pin
                        // hover).
                        Button {
                            onDirections(p)
                        } label: {
                            Image(systemName: "arrow.triangle.turn.up.right.circle.fill")
                                .font(.title3)
                                .foregroundStyle(Color.accentColor)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Directions to \(p.label)")
                    }
                }
            }
            .listStyle(.plain)
            .navigationTitle("\(payload.places.count) places")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done", action: onDismiss)
                }
                #else
                ToolbarItem(placement: .primaryAction) {
                    Button("Done", action: onDismiss)
                }
                #endif
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }
}

/// Native sheet that renders a Wikipedia article from the loaded
/// ZIM via `WKWebView`. Uses the existing `ZimURLSchemeHandler`
/// (same mechanism as route/places maps) so the article content,
/// CSS, and linked images all resolve against on-device bytes with
/// zero network.
struct ArticleSheetView: View {
    let intent: ArticleSheetIntent
    let session: ChatSession
    let onDismiss: () -> Void

    var body: some View {
        NavigationStack {
            ArticleWebContainer(intent: intent, session: session)
                .edgesIgnoringSafeArea(.bottom)
                .navigationTitle(intent.title)
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    #if os(iOS)
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done", action: onDismiss)
                    }
                    #else
                    ToolbarItem(placement: .primaryAction) {
                        Button("Done", action: onDismiss)
                    }
                    #endif
                }
        }
    }
}

#if os(macOS)
import AppKit
private struct ArticleWebContainer: NSViewRepresentable {
    let intent: ArticleSheetIntent
    let session: ChatSession
    func makeNSView(context: Context) -> WKWebView { makeArticleWebView(intent, session) }
    func updateNSView(_ view: WKWebView, context: Context) {}
}
#else
import UIKit
private struct ArticleWebContainer: UIViewRepresentable {
    let intent: ArticleSheetIntent
    let session: ChatSession
    func makeUIView(context: Context) -> WKWebView { makeArticleWebView(intent, session) }
    func updateUIView(_ view: WKWebView, context: Context) {}
}
#endif

@MainActor
private func makeArticleWebView(
    _ intent: ArticleSheetIntent,
    _ session: ChatSession
) -> WKWebView {
    let config = WKWebViewConfiguration()
    let handler = ZimURLSchemeHandler(
        lookup: { zimName in
            session.library.first { $0.url.lastPathComponent == zimName }?.reader
        },
        log: { msg in
            Task { @MainActor in session.debug(msg, category: "zim://") }
        }
    )
    config.setURLSchemeHandler(handler, forURLScheme: ZimURLSchemeHandler.scheme)
    if #available(macOS 13.3, iOS 16.4, *) {
        config.preferences.isElementFullscreenEnabled = true
    }
    let webView = WKWebView(frame: .zero, configuration: config)
    if #available(macOS 13.3, iOS 16.4, *) {
        webView.isInspectable = true
    }
    // Construct the URL as a pre-encoded string rather than through
    // URLComponents. When the ZIM filename contains a character
    // illegal in a URL host (a space, in a real capture:
    // `osm-silicon-valley-2026-04-22 3.zim`), `URLComponents.host = …`
    // silently makes `.url` return nil and the webView never loads —
    // the user sees a blank map bubble. Percent-encoding the host
    // ourselves keeps the URL valid; the scheme handler on the
    // receiving side reads `url.host` which decodes back to the
    // original filename for the library lookup.
    let encodedHost = intent.zim.addingPercentEncoding(
        withAllowedCharacters: .urlHostAllowed) ?? intent.zim
    let trimmedPath = intent.path.hasPrefix("/")
        ? String(intent.path.dropFirst())
        : intent.path
    let encodedPath = trimmedPath.addingPercentEncoding(
        withAllowedCharacters: .urlPathAllowed) ?? trimmedPath
    let urlString = "\(ZimURLSchemeHandler.scheme)://\(encodedHost)/\(encodedPath)"
    if let url = URL(string: urlString) {
        webView.load(URLRequest(url: url))
    }
    return webView
}

private struct PlacesFullscreen: View {
    let spec: PlacesSpec
    let session: ChatSession
    let payload: PlacesPayload
    let onDismiss: () -> Void

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            PlacesWebContainer(spec: spec, session: session, payload: payload)
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
            .accessibilityLabel("Close map")
            .padding(.bottom, 24)
            .padding(.leading, 16)
        }
    }
}

#if os(macOS)
import AppKit
private struct PlacesWebContainer: NSViewRepresentable {
    let spec: PlacesSpec
    let session: ChatSession
    let payload: PlacesPayload
    var focus: PlacesWebView.FocusIntent? = nil

    func makeNSView(context: Context) -> WKWebView {
        makePlacesWebView(spec: spec, session: session, payload: payload)
    }
    func updateNSView(_ nsView: WKWebView, context: Context) {
        reloadPlacesIfNeeded(nsView, spec: spec, payload: payload)
        applyFocusIfChanged(nsView, focus: focus, payload: payload)
    }
}
#else
import UIKit
private struct PlacesWebContainer: UIViewRepresentable {
    let spec: PlacesSpec
    let session: ChatSession
    let payload: PlacesPayload
    var focus: PlacesWebView.FocusIntent? = nil

    func makeUIView(context: Context) -> WKWebView {
        makePlacesWebView(spec: spec, session: session, payload: payload)
    }
    func updateUIView(_ uiView: WKWebView, context: Context) {
        reloadPlacesIfNeeded(uiView, spec: spec, payload: payload)
        applyFocusIfChanged(uiView, focus: focus, payload: payload)
    }
    // SwiftUI invokes this when the representable is removed from the
    // view hierarchy — i.e. when `isLatestAssistant` in ChatView flips
    // from true→false because a newer assistant turn appended. The
    // *Swift* side of the WKWebView deallocs on ARC release, but its
    // WebContent process (where MapLibre + the ~25 vector tiles live)
    // can linger for many seconds, holding ~500–800 MB that stacks on
    // top of the next turn's LLM prefill and trips the iPhone 6 GB
    // jetsam cap (2026-04-23: confirmed on Jazzman 17 — Qwen 3 4B
    // entering prefill at 5322 MB vs 2600 MB on Mac eval, delta is
    // the map WebContent process).
    //
    // Force the WebContent process to drop its heap immediately by
    // (a) stopping any in-flight loads,
    // (b) clearing the viewport's associated coordinator so MapLibre
    //     has nothing live to draw,
    // (c) navigating to `about:blank` — this tells WebKit the page
    //     and its render buffers can be released now, not on the
    //     next GC.
    static func dismantleUIView(_ uiView: WKWebView, coordinator: ()) {
        uiView.stopLoading()
        uiView.navigationDelegate = nil
        uiView.uiDelegate = nil
        uiView.load(URLRequest(url: URL(string: "about:blank")!))
    }
}
#endif

/// Zoom the map to the chosen place's coordinates and open a popup
/// against the captured MapLibre instance. Only fires when the focus
/// stamp differs from the last applied one — `updateUIView` runs on
/// every SwiftUI tick (location updates etc.), and we don't want to
/// slam easeTo each time.
@MainActor
private func applyFocusIfChanged(
    _ webView: WKWebView,
    focus: PlacesWebView.FocusIntent?,
    payload: PlacesPayload
) {
    guard let focus else { return }
    guard let coordinator = objc_getAssociatedObject(
        webView, &placesCoordinatorKey
    ) as? PlacesWebCoordinator else { return }
    if coordinator.lastFocusStamp == focus.stamp { return }
    coordinator.lastFocusStamp = focus.stamp
    guard focus.idx >= 0, focus.idx < payload.places.count else { return }
    let p = payload.places[focus.idx]
    // Escape the label / description for safe embedding in the JS
    // HTML — the streetzim viewer's popup takes raw innerHTML.
    let safeLabel = p.label
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
        .replacingOccurrences(of: "<", with: "&lt;")
    let safeDesc = p.description
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
        .replacingOccurrences(of: "<", with: "&lt;")
    let safeWikiPath = (p.wikiPath ?? "")
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
    let js = """
    (function() {
      function waitForMap(cb, tries) {
        tries = tries || 0;
        var m = window.__mcpzimMap;
        if (m && typeof m.addSource === 'function' && m.loaded && m.loaded()) cb(m);
        else if (tries < 120) setTimeout(function() { waitForMap(cb, tries + 1); }, 100);
      }
      waitForMap(function(m) {
        try {
          var coords = [\(p.lon), \(p.lat)];
          m.easeTo({ center: coords, zoom: 16, duration: 600 });
          var label = "\(safeLabel)";
          var desc = "\(safeDesc)";
          var wikiPath = "\(safeWikiPath)";
          // Prefer the shared popup builder defined in injectJS —
          // it emits the same label + description + Directions /
          // Wikipedia / Share buttons as pin-tap popups.
          var html = (typeof window.buildMcpzimPopupHTML === 'function')
            ? window.buildMcpzimPopupHTML(label, desc, coords, wikiPath)
            : ('<div style="font-family:var(--ui-font,system-ui);max-width:260px;">'
                + '<div style="font-weight:600;font-size:14px;margin-bottom:4px;">' + label + '</div>'
                + (desc ? '<div style="font-size:12px;color:#555;line-height:1.35;">' + desc + '</div>' : '')
                + '</div>');
          if (window.__mcpzimPopup && typeof window.__mcpzimPopup.remove === 'function') {
            try { window.__mcpzimPopup.remove(); } catch (e) {}
          }
          window.__mcpzimPopup = new maplibregl.Popup({ offset: 14, maxWidth: '280px' })
            .setLngLat(coords).setHTML(html).addTo(m);
        } catch (e) { console.error('mcpzim focus place failed', e); }
      });
    })();
    """
    webView.evaluateJavaScript(js) { _, _ in }
}

@MainActor
private final class PlacesWebCoordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
    var pendingInjection: String?
    var log: ((String) -> Void)?
    /// Last `FocusIntent.stamp` we pushed a zoom/popup for. Prevents
    /// `updateUIView` from re-firing the focus JS on every SwiftUI
    /// tick when nothing has actually changed.
    var lastFocusStamp: UUID? = nil
    /// Route-a-pin callback — wired by `makePlacesWebView` to
    /// `ChatSession.triggerDirectionsToCoord(name:lat:lon:)`. The
    /// coordinator holds it weakly-by-closure so the session's
    /// lifetime governs; when the chat view goes away the closure
    /// goes with it.
    var requestDirections: ((_ name: String, _ lat: Double, _ lon: Double) -> Void)?
    /// Read-the-Wikipedia-article callback — wired to
    /// `ChatSession.triggerArticleRead(title:path:)`.
    var requestArticle: ((_ title: String, _ path: String) -> Void)?

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        log?("places page loaded: \(webView.url?.absoluteString ?? "?")")
        guard let js = pendingInjection else { return }
        webView.evaluateJavaScript(js) { [weak self] _, error in
            if let error {
                self?.log?("places overlay JS failed: \(error.localizedDescription)")
            }
        }
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "mcpzim",
              let payload = message.body as? [String: Any] else { return }
        // Popup action dispatch — the pin popup's Directions/Share
        // icons post `{action: "directions" | "share", ...}` messages
        // back here so we can hand them off to native system handlers.
        if let action = payload["action"] as? String {
            handlePopupAction(action, payload: payload)
            return
        }
        let level = (payload["level"] as? String) ?? "log"
        let args = (payload["args"] as? [String]) ?? []
        log?("js.\(level) \(args.joined(separator: " "))")
    }

    private func handlePopupAction(_ action: String, payload: [String: Any]) {
        let name = (payload["name"] as? String) ?? ""
        let lat = (payload["lat"] as? NSNumber)?.doubleValue
        let lon = (payload["lon"] as? NSNumber)?.doubleValue
        switch action {
        case "directions":
            // Dispatch our OWN routing — `plan_driving_route` against
            // the streetzim graph — so the new turn shows up as a
            // route bubble in chat with the Drive/Walk/Bike pills,
            // not an external Apple Maps hand-off. The callback is
            // wired by `makePlacesWebView` to the session's
            // `triggerDirectionsToCoord`; falls back to silent if
            // the session has gone away.
            guard let lat, let lon else { return }
            requestDirections?(name, lat, lon)
        case "article":
            let path = (payload["path"] as? String) ?? ""
            guard !path.isEmpty else { return }
            requestArticle?(name, path)
        case "share":
            guard let lat, let lon else { return }
            // Build a shareable Apple Maps URL — recipients get a
            // tappable pin that opens in their default maps app,
            // not a raw "37.44, -122.15" blob. OSM place results
            // don't carry street addresses except for the `addr`
            // kind (the tool suppresses the non-`addr` `location`
            // field because of admin-boundary mislabels from the
            // streetzim generator), so name + coord link is the
            // most human-readable payload we have.
            var comps = URLComponents(string: "https://maps.apple.com/")!
            var query: [URLQueryItem] = [
                .init(name: "ll", value: "\(lat),\(lon)")
            ]
            if !name.isEmpty {
                query.append(.init(name: "q", value: name))
            }
            comps.queryItems = query
            var items: [Any] = []
            if !name.isEmpty { items.append(name) }
            if let url = comps.url { items.append(url) }
            if items.isEmpty {
                items.append(String(format: "%.5f, %.5f", lat, lon))
            }
            #if canImport(UIKit)
            presentShareSheet(items: items)
            #endif
        default:
            log?("popup action: unknown \(action)")
        }
    }

    #if canImport(UIKit)
    private func presentShareSheet(items: [Any]) {
        // Find the top-most presented view controller on the key
        // window to host the activity sheet from.
        guard let scene = UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene }).first,
              let root = scene.keyWindow?.rootViewController
        else { return }
        var top = root
        while let presented = top.presentedViewController { top = presented }
        let sheet = UIActivityViewController(
            activityItems: items, applicationActivities: nil
        )
        // iPad — anchor the popover to the centre of the top VC.
        if let pop = sheet.popoverPresentationController {
            pop.sourceView = top.view
            pop.sourceRect = CGRect(
                x: top.view.bounds.midX, y: top.view.bounds.midY,
                width: 1, height: 1
            )
            pop.permittedArrowDirections = []
        }
        top.present(sheet, animated: true)
    }
    #endif
}

@MainActor
private func makePlacesWebView(
    spec: PlacesSpec, session: ChatSession, payload: PlacesPayload
) -> WKWebView {
    let config = WKWebViewConfiguration()
    let handler = ZimURLSchemeHandler(
        lookup: { zimName in
            session.library.first { $0.url.lastPathComponent == zimName }?.reader
        },
        log: { msg in
            Task { @MainActor in session.debug(msg, category: "zim://") }
        }
    )
    config.setURLSchemeHandler(handler, forURLScheme: ZimURLSchemeHandler.scheme)

    let coordinator = PlacesWebCoordinator()
    coordinator.log = { msg in
        Task { @MainActor in session.debug(msg, category: "WebView") }
    }
    coordinator.requestDirections = { [weak session] name, lat, lon in
        Task { @MainActor in
            session?.triggerDirectionsToCoord(name: name, lat: lat, lon: lon)
        }
    }
    coordinator.requestArticle = { [weak session] title, path in
        Task { @MainActor in
            // Present a native sheet hosting a WKWebView of the
            // article from the Wikipedia ZIM. Goes through the
            // session so the outer PlacesWebView / ChatView
            // observes the intent and mounts `ArticleSheetView`.
            session?.presentArticleSheet(title: title, path: path)
        }
    }
    let userContent = WKUserContentController()
    userContent.add(coordinator, name: "mcpzim")
    let captureJS = """
    (function(){
      function post(level, args) {
        try {
          var serial = Array.from(args).map(function(a) {
            if (typeof a === 'string') return a;
            try { return JSON.stringify(a); } catch (e) { return String(a); }
          });
          window.webkit.messageHandlers.mcpzim.postMessage({level: level, args: serial});
        } catch (e) {}
      }
      ['log','info','warn','error'].forEach(function(level) {
        var orig = console[level];
        console[level] = function() {
          post(level, arguments);
          if (orig) orig.apply(console, arguments);
        };
      });
      window.addEventListener('error', function(e) {
        post('error', ['window.error:', (e.message || '?')]);
      });
      // Capture the MapLibre instance the viewer creates so downstream
      // injection can wait on it without parsing the viewer source.
      function wrapMaplibre(lib) {
        if (!lib || !lib.Map || lib.Map.__mcpzimPatched) return;
        var Orig = lib.Map;
        function Patched(opts) {
          var m = new Orig(opts);
          window.__mcpzimMap = m;
          return m;
        }
        Patched.prototype = Orig.prototype;
        Patched.__mcpzimPatched = true;
        for (var k in Orig) { if (Orig.hasOwnProperty(k)) Patched[k] = Orig[k]; }
        lib.Map = Patched;
      }
      var _maplibregl;
      try {
        Object.defineProperty(window, 'maplibregl', {
          configurable: true,
          get: function() { return _maplibregl; },
          set: function(v) { _maplibregl = v; wrapMaplibre(v); }
        });
      } catch (e) {}
    })();
    """
    let script = WKUserScript(source: captureJS, injectionTime: .atDocumentStart, forMainFrameOnly: false)
    userContent.addUserScript(script)
    config.userContentController = userContent
    if #available(macOS 13.3, iOS 16.4, *) {
        config.preferences.isElementFullscreenEnabled = true
    }
    let webView = WKWebView(frame: .zero, configuration: config)
    if #available(macOS 13.3, iOS 16.4, *) {
        webView.isInspectable = true
    }
    webView.navigationDelegate = coordinator
    objc_setAssociatedObject(webView, &placesCoordinatorKey, coordinator, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    loadPlacesSpec(webView, spec: spec, payload: payload)
    return webView
}

private var placesCoordinatorKey: UInt8 = 0

@MainActor
private func reloadPlacesIfNeeded(
    _ webView: WKWebView, spec: PlacesSpec, payload: PlacesPayload
) {
    // Comparison uses `url.host` (decodes percent-escaped back to the
    // original filename) so a ZIM whose name needed encoding at load
    // time still compares equal across re-renders. Previously the
    // expected string was built un-encoded while `webView.url?
    // .absoluteString` came back percent-encoded — they never matched
    // and we reloaded the whole map on every GPS tick.
    let currentHost = webView.url?.host ?? ""
    let currentPath = webView.url?.path.trimmingCharacters(
        in: CharacterSet(charactersIn: "/")) ?? ""
    if currentHost != spec.zimName || currentPath != spec.mainPath {
        loadPlacesSpec(webView, spec: spec, payload: payload)
        return
    }
    // URL matched — only refresh the "you are here" dot from the
    // latest spec.userLocation. Do NOT re-run the full pendingInjection:
    // it calls `showPlaces({fitBounds: true})` which re-frames the map
    // on every invocation, and `updateUIView` fires on every GPS tick.
    // That was showing up as a constant zoom-in every few seconds.
    if let here = spec.userLocation {
        let js = placesDotOnlyJS(lat: here.lat, lon: here.lon)
        webView.evaluateJavaScript(js) { _, _ in }
    }
}

/// Dot-only update for PlacesWebView — mirror of RouteWebView's
/// `userDotOnlyJS`, just namespaced against `mcpzim-me` source/layers
/// so IDs collide idempotently (first invocation adds, subsequent calls
/// only update the coordinate via setData).
@MainActor
private func placesDotOnlyJS(lat: Double, lon: Double) -> String {
    return """
    (function() {
      function waitForMap(cb, tries) {
        tries = tries || 0;
        var m = window.__mcpzimMap;
        if (m && typeof m.addSource === 'function' && m.loaded && m.loaded()) {
          cb(m);
        } else if (tries < 120) {
          setTimeout(function() { waitForMap(cb, tries + 1); }, 100);
        }
      }
      waitForMap(function(m) {
        try {
          var me = [\(lon), \(lat)];
          if (m.getSource('mcpzim-me')) {
            m.getSource('mcpzim-me').setData({
              type: 'Feature', geometry: { type: 'Point', coordinates: me }
            });
          } else {
            m.addSource('mcpzim-me', {
              type: 'geojson',
              data: { type: 'Feature', geometry: { type: 'Point', coordinates: me } }
            });
            m.addLayer({
              id: 'mcpzim-me-halo', type: 'circle', source: 'mcpzim-me',
              paint: { 'circle-radius': 14, 'circle-color': '#2563eb',
                       'circle-opacity': 0.18, 'circle-stroke-width': 0 }
            });
            m.addLayer({
              id: 'mcpzim-me-dot', type: 'circle', source: 'mcpzim-me',
              paint: { 'circle-radius': 7, 'circle-color': '#2563eb',
                       'circle-stroke-color': '#ffffff',
                       'circle-stroke-width': 2 }
            });
          }
        } catch (e) { console.error('mcpzim places me-dot update failed', e); }
      });
    })();
    """
}

@MainActor
private func loadPlacesSpec(
    _ webView: WKWebView, spec: PlacesSpec, payload: PlacesPayload
) {
    // Encode the places + optional radius ring as a single JSON blob
    // for the injected JS to consume. Keeps the interpolated string
    // small regardless of how many places the tool returned.
    var placeObjs: [[String: Any]] = []
    for p in payload.places {
        var row: [String: Any] = [
            "lat": p.lat, "lon": p.lon,
            "label": p.label, "description": p.description
        ]
        if let wp = p.wikiPath { row["wikiPath"] = wp }
        if let wt = p.wikiTitle { row["wikiTitle"] = wt }
        if let ws = p.website   { row["website"]  = ws }
        if let ph = p.phone     { row["phone"]    = ph }
        if let br = p.brand     { row["brand"]    = br }
        placeObjs.append(row)
    }
    let placesJSON: String = {
        guard let data = try? JSONSerialization.data(
            withJSONObject: placeObjs, options: []
        ) else { return "[]" }
        return String(data: data, encoding: .utf8) ?? "[]"
    }()
    let ringJS: String
    if let origin = payload.origin, let radiusKm = payload.radiusKm {
        ringJS = """
              try {
                window.streetzimRouting.showRadiusRing({
                  lat: \(origin.lat), lon: \(origin.lon), radiusKm: \(radiusKm)
                });
              } catch (e) { console.error("mcpzim places: showRadiusRing threw", e); }
        """
    } else {
        ringJS = ""
    }
    let userDotJS: String
    if let here = spec.userLocation {
        userDotJS = """
              // "You are here" dot so the user's own location is visible
              // alongside the returned pins — same style as the routing
              // view.
              var me = [\(here.lon), \(here.lat)];
              if (m.getSource('mcpzim-me')) {
                m.getSource('mcpzim-me').setData({
                  type: 'Feature', geometry: { type: 'Point', coordinates: me }
                });
              } else {
                m.addSource('mcpzim-me', {
                  type: 'geojson',
                  data: { type: 'Feature', geometry: { type: 'Point', coordinates: me } }
                });
                m.addLayer({
                  id: 'mcpzim-me-halo', type: 'circle', source: 'mcpzim-me',
                  paint: { 'circle-radius': 14, 'circle-color': '#2563eb',
                           'circle-opacity': 0.18, 'circle-stroke-width': 0 }
                });
                m.addLayer({
                  id: 'mcpzim-me-dot', type: 'circle', source: 'mcpzim-me',
                  paint: { 'circle-radius': 7, 'circle-color': '#2563eb',
                           'circle-stroke-color': '#ffffff',
                           'circle-stroke-width': 2 }
                });
              }
        """
    } else {
        userDotJS = ""
    }

    let injectJS = """
    (function() {
      var placesData = \(placesJSON);
      // Popup action helpers — routed through the iOS `mcpzim`
      // message handler so Directions opens Apple Maps + Share
      // launches the native UIActivityViewController. Exposed on
      // window so the popup's inline `onclick` can call them.
      window.mcpzimPopupAction = function(action, name, lat, lon) {
        try {
          window.webkit.messageHandlers.mcpzim.postMessage({
            action: action, name: name, lat: lat, lon: lon
          });
        } catch (e) {}
        return false;
      };
      // Popup HTML builder — small label + description row plus the
      // two icon buttons (➤ Directions, ⤴ Share). `esc` escapes
      // angle brackets so the label/description can't close the div
      // prematurely.
      window.buildMcpzimPopupHTML = function(label, description, coords, wikiPath, extras) {
        function esc(s) { return String(s || "").replace(/</g, "&lt;").replace(/"/g, "&quot;"); }
        function escAttr(s) { return String(s || "").replace(/"/g, "&quot;"); }
        var lon = coords[0], lat = coords[1];
        var safeLabel = esc(label);
        var safeDesc = esc(description);
        var jsLabel = String(label || "").replace(/\\\\/g, "\\\\\\\\").replace(/'/g, "\\\\'");
        var jsPath = String(wikiPath || "").replace(/\\\\/g, "\\\\\\\\").replace(/'/g, "\\\\'");
        var website = (extras && extras.website) || "";
        var phone   = (extras && extras.phone)   || "";
        var brand   = (extras && extras.brand)   || "";
        var html = '<div style="font-family:var(--ui-font,system-ui);max-width:280px;">'
          + '<div style="font-weight:600;font-size:14px;margin-bottom:2px;">' + safeLabel + '</div>';
        if (brand && brand !== label) {
          html += '<div style="font-size:11px;color:#6b7280;font-style:italic;margin-bottom:4px;">'
            + esc(brand) + '</div>';
        }
        if (description) {
          html += '<div style="font-size:12px;color:#555;line-height:1.35;margin-bottom:6px;">'
            + safeDesc + '</div>';
        }
        html += '<div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;">';
        html += '<button type="button" onclick="return window.mcpzimPopupAction(\\'directions\\', \\'' + jsLabel + '\\', ' + lat + ', ' + lon + ');"'
          + ' style="background:#2563eb;color:#fff;padding:6px 10px;border-radius:6px;border:0;font-size:12px;font-weight:600;cursor:pointer;">'
          + '↪ Directions</button>';
        // Website button — opens the external URL in the default
        // browser. Overture's places theme supplies this via the
        // `ws` field (renamed from `w` to avoid colliding with the
        // Wikipedia tag); plumbed through PlacesPayload.Place.website
        // to here.
        if (website) {
          html += '<a href="' + escAttr(website) + '" target="_blank" rel="noopener noreferrer"'
            + ' style="background:#0ea5e9;color:#fff;padding:6px 10px;border-radius:6px;text-decoration:none;font-size:12px;font-weight:600;">'
            + '🌐 Website</a>';
        }
        // Phone button — `tel:` URI triggers the dialer when tapped
        // in a WKWebView.
        if (phone) {
          html += '<a href="tel:' + escAttr(phone) + '"'
            + ' style="background:#22c55e;color:#fff;padding:6px 10px;border-radius:6px;text-decoration:none;font-size:12px;font-weight:600;">'
            + '📞 Call</a>';
        }
        if (wikiPath) {
          // "Read article" dispatches `get_article_section(lead)`
          // via the native bridge — new chat turn with the article's
          // hero image + lead prose.
          html += '<button type="button" onclick="return window.mcpzimPopupArticle(\\'' + jsLabel + '\\', \\'' + jsPath + '\\');"'
            + ' style="background:#16a34a;color:#fff;padding:6px 10px;border-radius:6px;border:0;font-size:12px;font-weight:600;cursor:pointer;">'
            + '📖 Wikipedia</button>';
        }
        html += '<button type="button" onclick="return window.mcpzimPopupAction(\\'share\\', \\'' + jsLabel + '\\', ' + lat + ', ' + lon + ');"'
          + ' style="background:#e5e7eb;color:#111;padding:6px 10px;border-radius:6px;border:0;font-size:12px;font-weight:600;cursor:pointer;">'
          + '⤴ Share</button>';
        html += '</div></div>';
        return html;
      };
      window.mcpzimPopupArticle = function(title, path) {
        try {
          window.webkit.messageHandlers.mcpzim.postMessage({
            action: 'article', name: title, path: path
          });
        } catch (e) {}
        return false;
      };
      function waitForMap(cb, tries) {
        tries = tries || 0;
        var m = window.__mcpzimMap;
        if (m && typeof m.addSource === "function" && m.loaded && m.loaded()) cb(m);
        else if (tries < 300) setTimeout(function() { waitForMap(cb, tries + 1); }, 100);
      }

      // Direct-on-map pin renderer — used either via the
      // `window.streetzimRouting.showPlaces` hook (newer ZIMs) or, as
      // a fallback, against the captured `window.__mcpzimMap`
      // instance when the hook isn't present (older ZIMs built
      // before the hook surface landed). Bigger circle-radius +
      // visible stroke + zoom-interpolated size so pins stay
      // legible even at the wide zoom level fit-bounds picks when
      // the search results span a region.
      function drawPinsDirect(m, places) {
        var srcId = "mcpzim-places";
        var dotId = "mcpzim-places-dots";
        var labelId = "mcpzim-places-labels";
        var features = places.map(function(p, i) {
          return {
            type: "Feature",
            geometry: { type: "Point", coordinates: [p.lon, p.lat] },
            properties: {
              idx: i,
              label: p.label || "",
              description: p.description || "",
              wikiPath: p.wikiPath || "",
              website: p.website || "",
              phone: p.phone || "",
              brand: p.brand || ""
            }
          };
        }).filter(function(f) {
          return isFinite(f.geometry.coordinates[0])
              && isFinite(f.geometry.coordinates[1]);
        });
        if (features.length === 0) return;
        var fc = { type: "FeatureCollection", features: features };
        if (m.getSource(srcId)) {
          m.getSource(srcId).setData(fc);
        } else {
          m.addSource(srcId, { type: "geojson", data: fc });
          m.addLayer({
            id: dotId, type: "circle", source: srcId,
            paint: {
              "circle-radius": [
                "interpolate", ["linear"], ["zoom"],
                8, 6, 12, 10, 16, 14
              ],
              "circle-color": "#e11d48",
              "circle-stroke-color": "#ffffff",
              "circle-stroke-width": 2,
              "circle-opacity": 0.95
            }
          });
          m.addLayer({
            id: labelId, type: "symbol", source: srcId,
            layout: {
              "text-field": ["get", "label"],
              "text-font": ["Open Sans Bold"],
              "text-size": 11,
              "text-offset": [0, 1.4],
              "text-anchor": "top",
              "text-optional": true,
              "text-max-width": 10
            },
            paint: {
              "text-color": "#111827",
              "text-halo-color": "#ffffff",
              "text-halo-width": 1.5
            }
          });
          m.on("click", dotId, function(e) {
            var f = e.features && e.features[0]; if (!f) return;
            var coords = f.geometry.coordinates;
            var labelRaw = String(f.properties.label || "");
            var descRaw = String(f.properties.description || "");
            var wikiPath = String(f.properties.wikiPath || "");
            var extras = {
              website: String(f.properties.website || ""),
              phone: String(f.properties.phone || ""),
              brand: String(f.properties.brand || "")
            };
            var html = buildMcpzimPopupHTML(labelRaw, descRaw, coords, wikiPath, extras);
            var popup = new maplibregl.Popup({ offset: 14 })
              .setLngLat(coords).setHTML(html).addTo(m);
            window.__mcpzimPopup = popup;
          });
        }
        // Fit-bounds. Single pin → z=15. Multi-pin → fitBounds with
        // padding, with a generous maxZoom so tight clusters (a few
        // bars on one block) actually zoom in instead of hitting a
        // z=14 ceiling and stranding the pins as a tiny red clump in
        // the middle of a whole-neighbourhood view.
        if (features.length === 1) {
          m.easeTo({
            center: features[0].geometry.coordinates,
            zoom: 15, duration: 600
          });
        } else {
          var b = new maplibregl.LngLatBounds(
            features[0].geometry.coordinates,
            features[0].geometry.coordinates
          );
          features.forEach(function(f) { b.extend(f.geometry.coordinates); });
          m.fitBounds(b, { padding: 80, duration: 600, maxZoom: 17 });
        }
        console.info("mcpzim places direct: drew " + features.length + " pins");
      }

      function drawRingDirect(m, opts) {
        if (!opts || typeof opts.lat !== "number"
            || typeof opts.lon !== "number"
            || typeof opts.radiusKm !== "number" || opts.radiusKm <= 0) return;
        var srcId = "mcpzim-radius";
        var fillId = "mcpzim-radius-fill";
        var lineId = "mcpzim-radius-line";
        var kmPerDegLat = 111.32;
        var kmPerDegLon = 111.32 * Math.cos(opts.lat * Math.PI / 180);
        var ring = [];
        for (var i = 0; i <= 64; i++) {
          var theta = (i / 64) * 2 * Math.PI;
          ring.push([
            opts.lon + (opts.radiusKm / kmPerDegLon) * Math.cos(theta),
            opts.lat + (opts.radiusKm / kmPerDegLat) * Math.sin(theta)
          ]);
        }
        var poly = { type: "Feature",
          geometry: { type: "Polygon", coordinates: [ring] }, properties: {} };
        if (m.getSource(srcId)) {
          m.getSource(srcId).setData(poly);
        } else {
          m.addSource(srcId, { type: "geojson", data: poly });
          m.addLayer({ id: fillId, type: "fill", source: srcId,
            paint: { "fill-color": "#2563eb", "fill-opacity": 0.08 }});
          m.addLayer({ id: lineId, type: "line", source: srcId,
            paint: { "line-color": "#2563eb", "line-width": 2,
                     "line-opacity": 0.5, "line-dasharray": [2, 2] }});
        }
      }

      // Single path: wait for the map, then EITHER use the hook
      // (newer ZIMs) OR call the direct renderers (older ZIMs whose
      // viewer was built before the hook landed). Using the hook is
      // preferred because it also hides viewer chrome — but the
      // direct path still produces pins + a ring, which is the main
      // user-visible thing.
      waitForMap(function(m) {
        \(userDotJS)
        // Always render pins via drawPinsDirect so the click-to-popup
        // goes through `buildMcpzimPopupHTML` — which has the ↪
        // Directions / 📖 Wikipedia / ⤴ Share buttons. streetzim's
        // `showPlaces` hook renders its OWN popup (just label +
        // description, no Directions), which is what the user was
        // seeing before. The hook's extra benefit was chrome-hiding;
        // we still get that via setChromeVisibility directly.
        try {
          if (typeof window.streetzimRouting !== "undefined"
              && window.streetzimRouting !== null
              && typeof window.streetzimRouting.setChromeVisibility === "function")
          {
            window.streetzimRouting.setChromeVisibility({
              search: false, controls: false, panel: false
            });
          }
        } catch (e) { /* older ZIM / hook shape change — drawPins works regardless */ }
        drawPinsDirect(m, placesData);
        \(ringJS)
      });
    })();
    """
    if let coordinator = objc_getAssociatedObject(webView, &placesCoordinatorKey) as? PlacesWebCoordinator {
        coordinator.pendingInjection = injectJS
    }
    // Build the URL as a pre-encoded string. `URLComponents.host` with
    // an illegal character (space) silently makes `.url` return nil;
    // real capture was `osm-silicon-valley-2026-04-22 3.zim` where the
    // trailing " 3" is a space. The old URLComponents path hit the nil
    // branch and the `webView.load` skipped, leaving the user with a
    // blank places map under a valid caption. Encoding the host +
    // path ourselves keeps the URL valid; the scheme handler reads
    // `url.host` which decodes back to the original filename.
    let encodedHost = spec.zimName.addingPercentEncoding(
        withAllowedCharacters: .urlHostAllowed) ?? spec.zimName
    let encodedPath = spec.mainPath.addingPercentEncoding(
        withAllowedCharacters: .urlPathAllowed) ?? spec.mainPath
    let urlString = "\(ZimURLSchemeHandler.scheme)://\(encodedHost)/\(encodedPath)"
    guard let url = URL(string: urlString) else { return }
    webView.load(URLRequest(url: url))
}
