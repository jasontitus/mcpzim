// SPDX-License-Identifier: MIT
//
// Embeds the streetzim's own HTML viewer in a WKWebView so users get the
// native zoom controls, layer toggles, satellite / OSM basemap switcher
// and whatever else the streetzim ships — instead of our simpler MapKit
// overlay. Resources resolve via `ZimURLSchemeHandler` over a custom
// `zim://` scheme, so everything stays offline.

import SwiftUI
import WebKit
import MCPZimKit

/// `true` if this trace is from a routing tool and has a polyline
/// worth rendering — OR from `show_map`, which carries a single-point
/// "polyline" so the same streetzim viewer centres on that place.
func traceHasRoute(_ trace: ToolCallTrace) -> Bool {
    guard trace.succeeded else { return false }
    let routingTools: Set<String> = ["plan_driving_route", "route_from_places", "show_map"]
    guard routingTools.contains(trace.name) else { return false }
    guard let data = trace.rawResult.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          // Only the untrimmed payload has polyline as `[[lat, lon], …]`;
          // the model-facing one summarises it to `{points, first, last}`.
          let poly = json["polyline"] as? [[Double]], !poly.isEmpty
    else { return false }
    // Routes need at least 2 points; show_map is happy with 1.
    if trace.name == "show_map" { return poly.count >= 1 }
    return poly.count >= 2
}

struct RouteWebView: View {
    /// The raw tool trace that produced this route (carries the full
    /// polyline and the streetzim filename when available).
    let trace: ToolCallTrace

    @Environment(ChatSession.self) private var session
    @State private var presentDirections: Bool = false
    /// Presenting the fullscreen map + its optional drive-mode kickoff
    /// via a single `Identifiable` item. Using
    /// `.fullScreenCover(item:)` guarantees atomic state: the cover
    /// only presents when the item transitions nil→non-nil, and the
    /// same item instance carries the mode, so there's no way
    /// SwiftUI could build the cover before `pendingDriveMode` was
    /// written (which happens with separate `Bool` + `String?`
    /// `@State` vars because the two writes aren't guaranteed to be
    /// observed atomically by the `.fullScreenCover(isPresented:)`
    /// modifier).
    struct FullscreenIntent: Identifiable {
        /// nil → plain expand (no drive mode), non-nil → enter drive /
        /// walk / bike mode via the streetzim viewer hook.
        let mode: String?
        /// Stable id per intent so SwiftUI doesn't reuse the same cover
        /// across mode changes. (Any fresh intent = fresh webview.)
        let id = UUID()
    }
    @State private var fullscreenIntent: FullscreenIntent? = nil

    /// Use the session-level location snapshot directly — ChatSession
    /// already fetches it at launch and on each send, so we get a
    /// free blue dot on every map without each view re-requesting GPS.
    private var userLocation: (lat: Double, lon: Double)? {
        session.currentLocation
    }

    /// Origin / destination parsed out of the tool result. Set when the
    /// router emits `{"origin": {"lat": …}, "destination": {"lat": …}}`;
    /// falls back to the polyline's first + last points when the routing
    /// tool didn't carry explicit origin/dest fields (e.g. `show_map`
    /// single-point traces have no endpoints to drive from).
    private var routeEndpoints: (origin: (lat: Double, lon: Double),
                                 dest: (lat: Double, lon: Double))? {
        guard let data = trace.rawResult.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if let o = json["origin"] as? [String: Any],
           let d = json["destination"] as? [String: Any],
           let oLat = (o["lat"] as? NSNumber)?.doubleValue,
           let oLon = (o["lon"] as? NSNumber)?.doubleValue,
           let dLat = (d["lat"] as? NSNumber)?.doubleValue,
           let dLon = (d["lon"] as? NSNumber)?.doubleValue
        {
            return ((oLat, oLon), (dLat, dLon))
        }
        if let poly = json["polyline"] as? [[Double]],
           poly.count >= 2,
           let first = poly.first, first.count >= 2,
           let last = poly.last, last.count >= 2
        {
            return ((first[0], first[1]), (last[0], last[1]))
        }
        return nil
    }

    /// Only show Drive/Walk/Bike for traces that have real origin+dest —
    /// `show_map` one-point pins don't have a route to enter drive mode on.
    private var supportsDriveMode: Bool {
        trace.name != "show_map" && routeEndpoints != nil
    }

    /// Full turn-by-turn list pulled from the tool's untrimmed result.
    /// Empty for `show_map` or when the route tool didn't return any
    /// instructions.
    private var turnByTurn: [String] {
        guard let data = trace.rawResult.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let turns = json["turn_by_turn"] as? [String]
        else { return [] }
        return turns
    }

    var body: some View {
        if let spec = resolveSpec(userLocation: userLocation) {
            VStack(alignment: .leading, spacing: 6) {
                WebViewContainer(spec: spec, session: session)
                    .frame(height: 480)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(alignment: .bottomLeading) {
                        // Placed bottom-left so it doesn't collide with
                        // MapLibre's top-right zoom (+/-) controls.
                        Button {
                            fullscreenIntent = FullscreenIntent(mode: nil)
                        } label: {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .font(.system(size: 14, weight: .semibold))
                                .padding(8)
                                .background(.thinMaterial, in: Circle())
                        }
                        .accessibilityLabel("Expand map")
                        .padding(8)
                    }
                    .padding(.top, 4)
                if supportsDriveMode { driveModeRow }
            }
                .onAppear {
                    // Nudge the session if we still don't have a fix.
                    if session.currentLocation == nil {
                        session.refreshLocationIfStale()
                    }
                }
                #if os(iOS)
                .fullScreenCover(item: $fullscreenIntent) { intent in
                    FullscreenMap(
                        spec: spec,
                        session: session,
                        initialDriveMode: intent.mode,
                        endpoints: routeEndpoints
                    ) {
                        fullscreenIntent = nil
                    }
                }
                #else
                .sheet(item: $fullscreenIntent) { intent in
                    FullscreenMap(
                        spec: spec,
                        session: session,
                        initialDriveMode: intent.mode,
                        endpoints: routeEndpoints
                    ) {
                        fullscreenIntent = nil
                    }
                }
                #endif
                .sheet(isPresented: $presentDirections) {
                    DirectionsListView(steps: turnByTurn) {
                        presentDirections = false
                    }
                }
        }
    }

    /// Drive / Walk / Bike + Directions buttons. Each of the three mode
    /// buttons pops the fullscreen viewer and injects JS that calls
    /// `window.streetzimRouting.setOrigin/setDest/clickMode` so the
    /// viewer's own drive-mode HUD takes over. Directions opens our
    /// turn-by-turn sheet.
    @ViewBuilder private var driveModeRow: some View {
        HStack(spacing: 8) {
            modePillButton(title: "Drive", mode: "drive", systemImage: "car.fill")
            modePillButton(title: "Walk",  mode: "walk",  systemImage: "figure.walk")
            modePillButton(title: "Bike",  mode: "bike",  systemImage: "bicycle")
            Spacer(minLength: 0)
            if !turnByTurn.isEmpty {
                Button {
                    presentDirections = true
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "list.bullet")
                        Text("Directions")
                    }
                    .font(.system(size: 13, weight: .semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(.thinMaterial, in: Capsule())
                }
                .accessibilityLabel("Full directions")
            }
        }
        .padding(.horizontal, 2)
    }

    @ViewBuilder
    private func modePillButton(title: String, mode: String, systemImage: String) -> some View {
        Button {
            fullscreenIntent = FullscreenIntent(mode: mode)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: systemImage)
                Text(title)
            }
            .font(.system(size: 13, weight: .semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(Color.accentColor.opacity(0.18), in: Capsule())
        }
        .accessibilityLabel("\(title) mode")
    }

    struct Spec: Equatable {
        let zimName: String
        let mainPath: String
        /// Polyline as GeoJSON coordinates (lon,lat order — MapLibre).
        /// Injected via JS after the viewer loads, used to both draw the
        /// line and call `fitBounds` so the map frames the route.
        let geoJSONCoords: String
        /// Current user location (lat, lon) if available. Rendered as
        /// a blue GeolocateControl-style dot on top of the route.
        let userLocation: (lat: Double, lon: Double)?

        static func == (a: Spec, b: Spec) -> Bool {
            a.zimName == b.zimName
                && a.mainPath == b.mainPath
                && a.geoJSONCoords == b.geoJSONCoords
                && a.userLocation?.lat == b.userLocation?.lat
                && a.userLocation?.lon == b.userLocation?.lon
        }
    }

    private func resolveSpec(userLocation: (lat: Double, lon: Double)?) -> Spec? {
        // 1. Which streetzim? First prefer the one the tool call was
        //    actually dispatched against (from `trace.arguments`) —
        //    otherwise multi-streetzim setups (baltics + DC + SV) would
        //    always render the viewer for the first-loaded ZIM regardless
        //    of what was routed.
        var zimFromArgs: String? = nil
        if let argData = trace.arguments.data(using: .utf8),
           let argJSON = try? JSONSerialization.jsonObject(with: argData) as? [String: Any],
           let z = argJSON["zim"] as? String,
           session.library.contains(where: { $0.url.lastPathComponent == z && $0.isEnabled })
        {
            zimFromArgs = z
        }
        // Then fall back to whichever streetzim's name appears in the
        // tool RESULT — our `routeFromPlaces` fallback may have tried
        // several and picked a different one than the model's `zim` arg.
        var zimFromResult: String? = nil
        if let resData = trace.rawResult.data(using: .utf8),
           let resJSON = try? JSONSerialization.jsonObject(with: resData) as? [String: Any],
           let z = resJSON["zim"] as? String,
           session.library.contains(where: { $0.url.lastPathComponent == z && $0.isEnabled })
        {
            zimFromResult = z
        }
        let pickedName = zimFromResult ?? zimFromArgs

        let entry: ChatSession.LibraryEntry
        if let name = pickedName,
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
        let zimName = entry.url.lastPathComponent

        // 2. Viewer entry path. Hardcoded to `index.html` — libzim's
        //    `readMainPage()` returns the redirect stub path ("mainPage")
        //    rather than the resolved target, which 404s through our
        //    scheme handler. The streetzim viewer lives at
        //    `index.html` at the ZIM root by convention.
        let mainPath = "index.html"

        // 3. Parse the polyline out of the tool's untrimmed payload.
        // `show_map` traces have a single-point polyline — that's
        // fine, `frameRoute` centres+zooms and we draw a pin instead
        // of a line.
        guard let data = trace.rawResult.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = json["polyline"] as? [[Double]], !raw.isEmpty
        else { return nil }
        // Cross-Bay Area routes can hit ~1500 polyline points. MapLibre
        // pre-renders them at each zoom level, and the in-process GL
        // buffers stacked on top of Gemma + Kokoro's Metal pools push us
        // past the 6144 MB jetsam cap. Subsample to ≤ 400 points —
        // plenty of detail for the zoomed-out overview route line.
        let downsampled = Self.downsample(raw, target: 400)
        // Polyline as a flat JS array of [lon, lat] pairs for MapLibre.
        let geoJSONCoords = "[" + downsampled.compactMap { pair -> String? in
            guard pair.count >= 2 else { return nil }
            return String(format: "[%.6f,%.6f]", pair[1], pair[0])
        }.joined(separator: ",") + "]"

        return Spec(
            zimName: zimName,
            mainPath: mainPath,
            geoJSONCoords: geoJSONCoords,
            userLocation: userLocation
        )
    }

    /// Uniform stride downsample (keeps first & last). Good enough for
    /// an overview route line — the polyline from the A* reconstruction
    /// already drops per-edge geometry, so inter-point spacing is
    /// roughly uniform-in-edges. Douglas-Peucker would be sharper but
    /// isn't worth the extra code for the zoom levels people actually
    /// see in a chat bubble.
    private static func downsample(_ pts: [[Double]], target: Int) -> [[Double]] {
        guard pts.count > target, target >= 2 else { return pts }
        let last = pts.count - 1
        let stride = Double(last) / Double(target - 1)
        var out: [[Double]] = []
        out.reserveCapacity(target)
        for i in 0..<target {
            let idx = min(last, Int((Double(i) * stride).rounded()))
            out.append(pts[idx])
        }
        if out.last! != pts.last! { out[out.count - 1] = pts.last! }
        return out
    }
}

/// Scrollable list of turn-by-turn directions pulled from the route
/// tool result. Presented as a sheet so the user can step through the
/// full list without losing the map underneath.
private struct DirectionsListView: View {
    let steps: [String]
    let onDismiss: () -> Void

    var body: some View {
        NavigationStack {
            List {
                ForEach(Array(steps.enumerated()), id: \.offset) { idx, step in
                    HStack(alignment: .top, spacing: 10) {
                        Text("\(idx + 1).")
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 32, alignment: .trailing)
                        Text(step)
                            .font(.body)
                    }
                    .padding(.vertical, 2)
                }
            }
            .listStyle(.plain)
            .navigationTitle("Turn by Turn")
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

/// Fullscreen wrapper around the same `WebViewContainer` used inline.
/// Presented via `fullScreenCover` from `RouteWebView` — an `X` button
/// top-right dismisses. Matches the inline map (same spec, same JS
/// injection) so the blue dot and route line stay consistent.
private struct FullscreenMap: View {
    let spec: RouteWebView.Spec
    let session: ChatSession
    let initialDriveMode: String?
    let endpoints: (origin: (lat: Double, lon: Double),
                    dest: (lat: Double, lon: Double))?
    let onDismiss: () -> Void

    var body: some View {
        // Bottom-leading X — the streetzim viewer's drive-mode HUD sits
        // top-10px across the full width with `position: absolute`, so
        // a top-leading iOS close button would sit right on top of the
        // turn arrow + distance, hiding the HUD entirely (and making
        // drive mode look like "just the route map"). Bottom-left stays
        // clear of both the HUD and MapLibre's zoom controls (top-right).
        ZStack(alignment: .bottomLeading) {
            WebViewContainer(
                spec: spec,
                session: session,
                initialDriveMode: initialDriveMode,
                endpoints: endpoints
            )
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
private struct WebViewContainer: NSViewRepresentable {
    let spec: RouteWebView.Spec
    let session: ChatSession
    var initialDriveMode: String? = nil
    var endpoints: (origin: (lat: Double, lon: Double),
                    dest: (lat: Double, lon: Double))? = nil

    func makeNSView(context: Context) -> WKWebView {
        makeWebView(
            spec: spec, session: session,
            initialDriveMode: initialDriveMode, endpoints: endpoints
        )
    }
    func updateNSView(_ nsView: WKWebView, context: Context) {
        reloadIfNeeded(
            nsView, spec: spec,
            initialDriveMode: initialDriveMode, endpoints: endpoints
        )
    }
}
#else
import UIKit
private struct WebViewContainer: UIViewRepresentable {
    let spec: RouteWebView.Spec
    let session: ChatSession
    var initialDriveMode: String? = nil
    var endpoints: (origin: (lat: Double, lon: Double),
                    dest: (lat: Double, lon: Double))? = nil

    func makeUIView(context: Context) -> WKWebView {
        makeWebView(
            spec: spec, session: session,
            initialDriveMode: initialDriveMode, endpoints: endpoints
        )
    }
    func updateUIView(_ uiView: WKWebView, context: Context) {
        reloadIfNeeded(
            uiView, spec: spec,
            initialDriveMode: initialDriveMode, endpoints: endpoints
        )
    }
}
#endif

@MainActor
private final class RouteWebCoordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
    var pendingInjection: String?
    /// Forwarded to `session.debug` so everything shows in the app's
    /// Debug pane — user doesn't need to open Safari's Web Inspector.
    var log: ((String) -> Void)?

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        log?("page loaded: \(webView.url?.absoluteString ?? "?")")
        guard let js = pendingInjection else { return }
        webView.evaluateJavaScript(js) { [weak self] _, error in
            if let error {
                self?.log?("overlay JS failed: \(error.localizedDescription)")
            }
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        log?("page failed: \(error.localizedDescription)")
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        log?("page provisional failed: \(error.localizedDescription)")
    }

    // MARK: - JS bridge

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "mcpzim",
              let payload = message.body as? [String: Any] else { return }
        let level = (payload["level"] as? String) ?? "log"
        let args = (payload["args"] as? [String]) ?? []
        log?("js.\(level) \(args.joined(separator: " "))")
    }
}

@MainActor
private func makeWebView(
    spec: RouteWebView.Spec,
    session: ChatSession,
    initialDriveMode: String? = nil,
    endpoints: (origin: (lat: Double, lon: Double),
                dest: (lat: Double, lon: Double))? = nil
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

    // JS-side console + error capture — forwards everything the page
    // logs (including MapLibre errors, failed fetches) into our Swift
    // log sink so it lands in the in-app Debug pane. No Safari Web
    // Inspector required.
    let coordinator = RouteWebCoordinator()
    coordinator.log = { msg in
        Task { @MainActor in session.debug(msg, category: "WebView") }
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
        post('error', ['window.error:', (e.message || '?'),
                       'at', (e.filename || '?') + ':' + (e.lineno || 0)]);
      });
      window.addEventListener('unhandledrejection', function(e) {
        var reason = e && e.reason;
        post('error', ['unhandledrejection:',
          reason && reason.stack ? reason.stack : (reason && reason.message ? reason.message : String(reason))]);
      });
      var _fetch = window.fetch;
      window.fetch = function(input, init) {
        var url = typeof input === 'string' ? input : (input && input.url) || '?';
        return _fetch.apply(this, arguments).catch(function(err) {
          post('error', ['fetch failed:', url, err && err.message]);
          throw err;
        });
      };
      // The streetzim viewer keeps its MapLibre instance in a local `var map`
      // inside `fetchConfig().then(...)`, so it's invisible from the host
      // scope. Catch it via two paths:
      // (1) Setter-based interception — the moment the viewer assigns
      //     `window.maplibregl = ...`, we patch `.Map` to capture every
      //     instance via `window.__mcpzimMap`. This is the fast path.
      //     Polling alone races with fast loads where the page assigns
      //     maplibregl AND calls `new maplibregl.Map()` in one synchronous
      //     run before our 25 ms timer fires.
      // (2) DOM-scan fallback — if MapLibre is loaded via ES-module or
      //     otherwise bypasses `window.maplibregl`, walk the DOM for
      //     `.maplibregl-canvas` whose parent carries an `_map` reference.
      function wrapMaplibre(lib) {
        if (!lib || !lib.Map || lib.Map.__mcpzimPatched) return;
        var Orig = lib.Map;
        function Patched(opts) {
          var m = new Orig(opts);
          window.__mcpzimMap = m;
          try { post('info', ['mcpzim captured MapLibre instance']); } catch (e) {}
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
      } catch (e) {
        // Some browsers disallow redefining window globals; fall back
        // to polling + DOM scan below.
      }
      function pollOrProbe(tries) {
        tries = tries || 0;
        if (typeof _maplibregl !== 'undefined' && _maplibregl && _maplibregl.Map) {
          wrapMaplibre(_maplibregl);
        } else if (typeof maplibregl !== 'undefined' && maplibregl && maplibregl.Map) {
          wrapMaplibre(maplibregl);
        }
        if (!window.__mcpzimMap) {
          // DOM fallback: find a maplibregl canvas and read the private
          // `_map` back-pointer MapLibre keeps on the container node.
          var canvas = document.querySelector('.maplibregl-canvas');
          if (canvas) {
            var container = canvas.parentNode && canvas.parentNode.parentNode;
            var m = container && (container._map || container.__map || container._maplibreMap);
            if (m && typeof m.getSource === 'function') {
              window.__mcpzimMap = m;
              try { post('info', ['mcpzim captured MapLibre via DOM scan']); } catch (e) {}
            }
          }
        }
        if (!window.__mcpzimMap && tries < 300) {
          setTimeout(function() { pollOrProbe(tries + 1); }, 50);
        }
      }
      pollOrProbe();
      post('info', ['mcpzim console bridge active']);
    })();
    """
    let script = WKUserScript(source: captureJS, injectionTime: .atDocumentStart, forMainFrameOnly: false)
    userContent.addUserScript(script)
    config.userContentController = userContent
    // Safari Web Inspector is still enabled as a secondary option.
    if #available(macOS 13.3, iOS 16.4, *) {
        config.preferences.isElementFullscreenEnabled = true
    }
    let webView = WKWebView(frame: .zero, configuration: config)
    if #available(macOS 13.3, iOS 16.4, *) {
        webView.isInspectable = true
    }
    webView.navigationDelegate = coordinator
    objc_setAssociatedObject(webView, &coordinatorKey, coordinator, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    loadSpec(
        webView, spec: spec,
        initialDriveMode: initialDriveMode, endpoints: endpoints
    )
    return webView
}

private var coordinatorKey: UInt8 = 0

@MainActor
private func reloadIfNeeded(
    _ webView: WKWebView,
    spec: RouteWebView.Spec,
    initialDriveMode: String? = nil,
    endpoints: (origin: (lat: Double, lon: Double),
                dest: (lat: Double, lon: Double))? = nil
) {
    // Compare decoded host/path against the spec so a ZIM name
    // that needed percent-encoding at load time still matches
    // across re-renders. Previously the raw-string build didn't
    // match the WebView's encoded absoluteString — the view would
    // hard-reload on every GPS tick.
    let currentHost = webView.url?.host ?? ""
    let currentPath = webView.url?.path.trimmingCharacters(
        in: CharacterSet(charactersIn: "/")) ?? ""
    if currentHost != spec.zimName || currentPath != spec.mainPath {
        loadSpec(
            webView, spec: spec,
            initialDriveMode: initialDriveMode, endpoints: endpoints
        )
        return
    }
    // URL matched — map is already loaded in this webview. Only
    // refresh the "you are here" dot from the new spec.userLocation;
    // do NOT re-evaluate the full pendingInjection. That injection
    // contains `frameRoute(m)` which calls `map.fitBounds` and drive
    // mode's `easeTo` — re-running it on every SwiftUI
    // `updateUIView` (which fires on every GPS tick, ~2 Hz) made the
    // map re-zoom itself every few seconds, visibly distracting.
    if let here = spec.userLocation {
        let js = userDotOnlyJS(lat: here.lat, lon: here.lon)
        webView.evaluateJavaScript(js) { _, _ in }
    }
}

/// Minimal JS to add/update the "you are here" marker on an already-loaded
/// map. Used by `updateUIView` to push the dot after the initial overlay
/// has already fired. Mirrors the layer definitions in `loadSpec` so the
/// IDs collide (first call creates; subsequent calls just setData).
@MainActor
private func userDotOnlyJS(lat: Double, lon: Double) -> String {
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
        } catch (e) { console.error('mcpzim me-dot update failed', e); }
      });
    })();
    """
}

@MainActor
private func loadSpec(
    _ webView: WKWebView,
    spec: RouteWebView.Spec,
    initialDriveMode: String? = nil,
    endpoints: (origin: (lat: Double, lon: Double),
                dest: (lat: Double, lon: Double))? = nil
) {
    // Stage the overlay JS before kicking off navigation so it fires on
    // `didFinish`. Waits for the viewer's MapLibre instance (the global
    // `map`) to exist, then adds a GeoJSON source + line layer.
    let userDotJS: String
    if let here = spec.userLocation {
        userDotJS = """
              // Blue "you are here" dot using two stacked circle layers.
              // Stays under the route line so the path is still readable.
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
    // Drive-mode kickoff — only if the user tapped Drive/Walk/Bike AND
    // the route has real origin/destination coords. Calls into the
    // `window.streetzimRouting` hook exposed by the streetzim viewer
    // (resources/viewer/index.html) to drive the viewer's own drive-mode
    // HUD. No-op if the hook isn't present (older ZIM without the patch)
    // or the graph hasn't finished loading within the timeout.
    // Always log what loadSpec received so we can tell from iOS syslog
    // whether a missing drive-mode kickoff is because initialDriveMode
    // arrived nil, endpoints arrived nil, or something further inside
    // the JS bailed. Logs via console.info so the WebView bridge picks
    // it up alongside the other drive-mode diagnostics.
    let modeDesc = initialDriveMode ?? "(nil)"
    let endpointsDesc: String = endpoints == nil ? "(nil)" : "set"
    let loadSpecLogJS = """
          console.info("mcpzim loadSpec: initialDriveMode=\(modeDesc) endpoints=\(endpointsDesc)");
    """
    let driveModeJS: String
    if let mode = initialDriveMode, let ep = endpoints {
        driveModeJS = """
              (function() {
                var mode = "\(mode)";
                var o = [\(ep.origin.lat), \(ep.origin.lon)];
                var d = [\(ep.dest.lat), \(ep.dest.lon)];
                // One-shot guard. SwiftUI's `updateUIView` re-runs the
                // whole injection on every session-state tick (location
                // updates, streaming chat etc.), and the viewer's mode
                // button *toggles* — re-firing would flip the drive
                // HUD on/off every few seconds. Mark the window so
                // only the first kickoff counts.
                if (window.__mcpzimDriveKicked === mode) {
                  return;
                }
                window.__mcpzimDriveKicked = mode;
                console.info("mcpzim drive-mode: kickoff mode=" + mode
                  + " origin=" + o.join(",") + " dest=" + d.join(","));
                // Fail-fast if the ZIM's viewer doesn't have the
                // streetzimRouting hook (older build — needs a rebuild
                // of the ZIM after streetzim@226d511). Without this
                // we'd silently poll for 30 s before logging the
                // timeout, long after the user has given up.
                function hookPresent() {
                  return typeof window.streetzimRouting !== "undefined"
                      && window.streetzimRouting !== null;
                }
                function hookCheck(tries) {
                  tries = tries || 0;
                  if (hookPresent()) return pickPath();
                  if (tries > 20) {  // 2 s is plenty for script setup
                    console.warn("mcpzim drive-mode: window.streetzimRouting "
                      + "missing after 2 s — this ZIM was built before the "
                      + "streetzim hook landed. Rebuild the ZIM with the "
                      + "latest resources/viewer/index.html.");
                    return;
                  }
                  setTimeout(function() { hookCheck(tries + 1); }, 100);
                }
                function pickPath() {
                  // Prefer the atomic hook when the ZIM has it —
                  // everything (loadGraph, setOrigin/setDest, await
                  // route, driveMode.enter + UI sync) happens inside
                  // the viewer's scope, no cross-bridge polling.
                  if (typeof window.streetzimRouting.enterDriveMode === "function") {
                    console.info("mcpzim drive-mode: using atomic enterDriveMode hook");
                    window.streetzimRouting.enterDriveMode(mode, o[0], o[1], d[0], d[1])
                      .then(function(r) {
                        console.info("mcpzim drive-mode: entered via atomic hook, alreadyActive=" + r.alreadyActive);
                      })
                      .catch(function(e) {
                        console.error("mcpzim drive-mode: atomic hook rejected:", e && e.message ? e.message : String(e));
                      });
                    return;
                  }
                  // Fallback for ZIMs built before the atomic hook —
                  // orchestrate setOrigin/setDest/click ourselves.
                  proceed();
                }
                function waitFor(test, cb, label, tries) {
                  tries = tries || 0;
                  try { if (test()) return cb(); } catch (e) {}
                  if (tries > 300) {
                    console.warn("mcpzim drive-mode: timeout waiting for " + label
                      + " after 30 s");
                    return;
                  }
                  setTimeout(function() { waitFor(test, cb, label, tries + 1); }, 100);
                }
                function proceed() {
                  console.info("mcpzim drive-mode: hook present; "
                    + "graphReady=" + window.streetzimRouting.graphReady);
                  // Viewer lazy-loads the routing graph on first user
                  // interaction with the routing panel (toggle click).
                  // When we're entering from a chat bubble the panel
                  // was never opened, so kick off the fetch ourselves
                  // before we poll graphReady. Calling twice is safe —
                  // the viewer's own loadGraph is a no-op when a fetch
                  // is already in flight.
                  if (!window.streetzimRouting.graphReady) {
                    if (typeof window.streetzimRouting.loadGraph === "function") {
                      console.info("mcpzim drive-mode: graph not loaded, calling loadGraph()");
                      try { window.streetzimRouting.loadGraph(); }
                      catch (e) { console.error("mcpzim drive-mode: loadGraph threw", e); }
                    } else {
                      // Older ZIM (hook present but no loadGraph method).
                      // Trigger the fetch via the existing UI path — click
                      // the route-toggle button to open the routing panel,
                      // which internally calls loadGraph(). The panel is
                      // auto-hidden once driveMode.enter fires, so the user
                      // sees it flash briefly at worst.
                      var toggle = document.getElementById("route-toggle");
                      if (toggle) {
                        console.info("mcpzim drive-mode: graph not loaded, clicking route-toggle");
                        toggle.click();
                      } else {
                        console.warn("mcpzim drive-mode: no loadGraph + no #route-toggle — can't trigger fetch");
                      }
                    }
                  }
                  waitFor(
                    function() { return window.streetzimRouting.graphReady; },
                    function() {
                      console.info("mcpzim drive-mode: graph ready, calling setOrigin/setDest");
                      try {
                        window.streetzimRouting.setOrigin(o[0], o[1], "Start");
                        window.streetzimRouting.setDest(d[0], d[1], "Destination");
                      } catch (e) {
                        console.error("mcpzim drive-mode: setOrigin/setDest threw", e);
                        return;
                      }
                      waitFor(
                        function() { return window.streetzimRouting.hasRoute; },
                        function() {
                          try {
                            window.streetzimRouting.clickMode(mode);
                            console.info("mcpzim drive-mode: clickMode fired for " + mode);
                          } catch (e) {
                            console.error("mcpzim drive-mode: clickMode threw", e);
                            return;
                          }
                          // Verify drive mode actually activated — the
                          // viewer's click handler bails silently if the
                          // mode is already active (toggle) or if
                          // lastRoute just became null. Log the HUD's
                          // visibility 250 ms after the click so we can
                          // tell the difference in the iOS syslog.
                          setTimeout(function() {
                            var hud = document.getElementById('drive-hud');
                            var visible = !!(hud && hud.classList.contains('visible'));
                            console.info("mcpzim drive-mode: 250ms post-click, hudVisible=" + visible);
                            if (!visible) {
                              console.warn("mcpzim drive-mode: HUD not visible — retrying click once");
                              try { window.streetzimRouting.clickMode(mode); } catch (e) {}
                              setTimeout(function() {
                                var h2 = document.getElementById('drive-hud');
                                var v2 = !!(h2 && h2.classList.contains('visible'));
                                console.info("mcpzim drive-mode: post-retry hudVisible=" + v2);
                              }, 200);
                            }
                          }, 250);
                        },
                        "hasRoute"
                      );
                    },
                    "graphReady"
                  );
                }
                hookCheck();
              })();
        """
    } else {
        driveModeJS = ""
    }
    let injectJS = """
    (function() {
      \(loadSpecLogJS)
      var coords = \(spec.geoJSONCoords);
      function isMapReady() {
        var m = window.__mcpzimMap;
        return m && typeof m.on === 'function' && typeof m.addSource === 'function';
      }
      function waitForMap(cb, tries) {
        tries = tries || 0;
        if (isMapReady()) {
          var m = window.__mcpzimMap;
          if (m.loaded && m.loaded()) cb(m);
          else m.on('load', function() { cb(m); });
        } else if (tries < 300) {
          setTimeout(function() { waitForMap(cb, tries + 1); }, 100);
        } else {
          console.error('mcpzim: __mcpzimMap not ready after 30s, giving up');
        }
      }
      function frameRoute(m) {
        if (coords.length === 1) {
          m.setCenter(coords[0]);
          m.setZoom(14);
          return;
        }
        var b = coords.reduce(function(b, c) { return b.extend(c); },
                              new maplibregl.LngLatBounds(coords[0], coords[0]));
        m.fitBounds(b, { padding: 40, duration: 0 });
      }
      function addRouteIfMissing(m) {
        if (coords.length >= 2) {
          if (!m.getSource('mcpzim-route')) {
            m.addSource('mcpzim-route', {
              type: 'geojson',
              data: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords } }
            });
          }
          if (!m.getLayer('mcpzim-route-line')) {
            m.addLayer({
              id: 'mcpzim-route-line',
              type: 'line',
              source: 'mcpzim-route',
              layout: { 'line-join': 'round', 'line-cap': 'round' },
              paint: { 'line-color': '#2563eb', 'line-width': 4, 'line-opacity': 0.9 }
            });
          }
        } else if (coords.length === 1) {
          if (!m.getSource('mcpzim-pin')) {
            m.addSource('mcpzim-pin', {
              type: 'geojson',
              data: { type: 'Feature', geometry: { type: 'Point', coordinates: coords[0] } }
            });
          }
          if (!m.getLayer('mcpzim-pin-dot')) {
            m.addLayer({
              id: 'mcpzim-pin-dot', type: 'circle', source: 'mcpzim-pin',
              paint: { 'circle-radius': 8, 'circle-color': '#e11d48',
                       'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2 }
            });
          }
        }
      }
      waitForMap(function(m) {
        try {
          if (coords.length >= 2) {
            if (m.getSource('mcpzim-route')) {
              m.getSource('mcpzim-route').setData({
                type: 'Feature', geometry: { type: 'LineString', coordinates: coords }
              });
            }
            addRouteIfMissing(m);
          } else if (coords.length === 1) {
            // `show_map` — just drop a pin at the single point.
            if (m.getSource('mcpzim-pin')) {
              m.getSource('mcpzim-pin').setData({
                type: 'Feature', geometry: { type: 'Point', coordinates: coords[0] }
              });
            } else {
              m.addSource('mcpzim-pin', {
                type: 'geojson',
                data: { type: 'Feature', geometry: { type: 'Point', coordinates: coords[0] } }
              });
              m.addLayer({
                id: 'mcpzim-pin-dot', type: 'circle', source: 'mcpzim-pin',
                paint: { 'circle-radius': 8, 'circle-color': '#e11d48',
                         'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2 }
              });
            }
          }
          \(userDotJS)
          frameRoute(m);
          \(driveModeJS)
          // MapLibre wipes layers whenever the style reloads — which
          // happens on zoom crossings and vector-tile style changes.
          // Re-add our overlay on every styledata/sourcedata so the
          // route line survives interaction.
          m.on('styledata', function() { addRouteIfMissing(m); });
          m.on('sourcedata', function() { addRouteIfMissing(m); });
          console.info('mcpzim route drawn (' + coords.length + ' points)');
        } catch (e) { console.error('mcpzim overlay failed', e); }
      });
    })();
    """
    if let coordinator = objc_getAssociatedObject(webView, &coordinatorKey) as? RouteWebCoordinator {
        coordinator.pendingInjection = injectJS
    }
    // No URL fragment: the viewer's baseUrl computation (lastIndexOf('/'))
    // misreads `#map=z/lat/lon` as part of the path, so it ends up fetching
    // `index.html` where `map-config.json` should be and then fails JSON
    // parse. Instead we let the page load cleanly and fit-bounds the route
    // ourselves from the overlay JS.
    // Build the URL as a pre-encoded string — `URLComponents.host`
    // rejects illegal characters (spaces, in `osm-silicon-valley-
    // 2026-04-22 3.zim`) by silently making `.url` return nil,
    // which then blanks the route-map bubble. Same workaround as
    // the places path; the scheme handler decodes url.host back to
    // the original filename for the library lookup.
    let encodedHost = spec.zimName.addingPercentEncoding(
        withAllowedCharacters: .urlHostAllowed) ?? spec.zimName
    let encodedPath = spec.mainPath.addingPercentEncoding(
        withAllowedCharacters: .urlPathAllowed) ?? spec.mainPath
    let urlString = "\(ZimURLSchemeHandler.scheme)://\(encodedHost)/\(encodedPath)"
    guard let url = URL(string: urlString) else { return }
    webView.load(URLRequest(url: url))
}
