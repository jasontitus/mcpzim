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
    @State private var presentFullscreen: Bool = false
    @State private var presentDirections: Bool = false

    /// Use the session-level location snapshot directly — ChatSession
    /// already fetches it at launch and on each send, so we get a
    /// free blue dot on every map without each view re-requesting GPS.
    private var userLocation: (lat: Double, lon: Double)? {
        session.currentLocation
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
            WebViewContainer(spec: spec, session: session)
                .frame(height: 360)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(alignment: .bottomLeading) {
                    // Placed bottom-left so it doesn't collide with
                    // MapLibre's top-right zoom (+/-) controls.
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
                    // Directions list — only when the route tool returned
                    // non-empty turn_by_turn. For `show_map` this stays
                    // hidden since there's nothing to step through.
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
                        .padding(8)
                    }
                }
                .padding(.top, 4)
                .onAppear {
                    // Nudge the session if we still don't have a fix.
                    if session.currentLocation == nil {
                        session.refreshLocationIfStale()
                    }
                }
                #if os(iOS)
                .fullScreenCover(isPresented: $presentFullscreen) {
                    FullscreenMap(spec: spec, session: session) {
                        presentFullscreen = false
                    }
                }
                #else
                .sheet(isPresented: $presentFullscreen) {
                    FullscreenMap(spec: spec, session: session) {
                        presentFullscreen = false
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
    let onDismiss: () -> Void

    var body: some View {
        ZStack(alignment: .topLeading) {
            WebViewContainer(spec: spec, session: session)
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
            .padding(.top, 12)
            .padding(.leading, 12)
        }
    }
}

#if os(macOS)
import AppKit
private struct WebViewContainer: NSViewRepresentable {
    let spec: RouteWebView.Spec
    let session: ChatSession

    func makeNSView(context: Context) -> WKWebView {
        makeWebView(spec: spec, session: session)
    }
    func updateNSView(_ nsView: WKWebView, context: Context) {
        reloadIfNeeded(nsView, spec: spec)
    }
}
#else
import UIKit
private struct WebViewContainer: UIViewRepresentable {
    let spec: RouteWebView.Spec
    let session: ChatSession

    func makeUIView(context: Context) -> WKWebView {
        makeWebView(spec: spec, session: session)
    }
    func updateUIView(_ uiView: WKWebView, context: Context) {
        reloadIfNeeded(uiView, spec: spec)
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
private func makeWebView(spec: RouteWebView.Spec, session: ChatSession) -> WKWebView {
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
    loadSpec(webView, spec: spec)
    return webView
}

private var coordinatorKey: UInt8 = 0

@MainActor
private func reloadIfNeeded(_ webView: WKWebView, spec: RouteWebView.Spec) {
    let expected = "zim://\(spec.zimName)/\(spec.mainPath)"
    if webView.url?.absoluteString != expected {
        loadSpec(webView, spec: spec)
        return
    }
    // URL matched — map is already loading/loaded in this webview.
    // Re-run the full overlay JS so the route line gets drawn (the
    // previous coordinator's `didFinish` may not have seen the
    // spec we care about in fullscreen-cover scenarios). The JS is
    // idempotent: it uses `getSource`/`getLayer` guards before
    // adding, and `setData` for existing layers.
    if let coordinator = objc_getAssociatedObject(webView, &coordinatorKey) as? RouteWebCoordinator,
       let js = coordinator.pendingInjection
    {
        webView.evaluateJavaScript(js) { _, _ in }
    }
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
private func loadSpec(_ webView: WKWebView, spec: RouteWebView.Spec) {
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
    let injectJS = """
    (function() {
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
    var components = URLComponents()
    components.scheme = ZimURLSchemeHandler.scheme
    components.host = spec.zimName
    components.path = "/" + spec.mainPath
    guard let url = components.url else { return }
    webView.load(URLRequest(url: url))
}
