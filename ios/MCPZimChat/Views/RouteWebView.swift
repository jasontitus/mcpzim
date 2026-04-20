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

/// `true` if this trace is from a routing tool and has a polyline worth rendering.
func traceHasRoute(_ trace: ToolCallTrace) -> Bool {
    guard trace.succeeded,
          trace.name == "plan_driving_route" || trace.name == "route_from_places"
    else { return false }
    guard let data = trace.rawResult.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          // Only the untrimmed payload has polyline as `[[lat, lon], …]`;
          // the model-facing one summarises it to `{points, first, last}`.
          let poly = json["polyline"] as? [[Double]], poly.count >= 2
    else { return false }
    return true
}

struct RouteWebView: View {
    /// The raw tool trace that produced this route (carries the full
    /// polyline and the streetzim filename when available).
    let trace: ToolCallTrace

    @Environment(ChatSession.self) private var session

    var body: some View {
        if let spec = resolveSpec() {
            WebViewContainer(spec: spec, session: session)
                .frame(height: 360)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding(.top, 4)
        }
    }

    struct Spec: Equatable {
        let zimName: String
        let mainPath: String
        /// Polyline as GeoJSON coordinates (lon,lat order — MapLibre).
        /// Injected via JS after the viewer loads, used to both draw the
        /// line and call `fitBounds` so the map frames the route.
        let geoJSONCoords: String
    }

    private func resolveSpec() -> Spec? {
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
        guard let data = trace.rawResult.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = json["polyline"] as? [[Double]], raw.count >= 2
        else { return nil }
        // Polyline as a flat JS array of [lon, lat] pairs for MapLibre.
        let geoJSONCoords = "[" + raw.compactMap { pair -> String? in
            guard pair.count >= 2 else { return nil }
            return String(format: "[%.6f,%.6f]", pair[1], pair[0])
        }.joined(separator: ",") + "]"

        return Spec(
            zimName: zimName,
            mainPath: mainPath,
            geoJSONCoords: geoJSONCoords
        )
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
      // scope. Patch `maplibregl.Map`'s constructor the moment the library
      // is parsed — every new Map also lands on `window.__mcpzimMap`, which
      // our overlay polls for.
      function patchMaplibre() {
        if (typeof maplibregl === 'undefined' || !maplibregl.Map) {
          return setTimeout(patchMaplibre, 25);
        }
        if (maplibregl.Map.__mcpzimPatched) return;
        var Orig = maplibregl.Map;
        function Patched(opts) {
          var m = new Orig(opts);
          window.__mcpzimMap = m;
          try { post('info', ['mcpzim captured MapLibre instance']); } catch (e) {}
          return m;
        }
        Patched.prototype = Orig.prototype;
        Patched.__mcpzimPatched = true;
        for (var k in Orig) { if (Orig.hasOwnProperty(k)) Patched[k] = Orig[k]; }
        maplibregl.Map = Patched;
      }
      patchMaplibre();
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
    }
}

@MainActor
private func loadSpec(_ webView: WKWebView, spec: RouteWebView.Spec) {
    // Stage the overlay JS before kicking off navigation so it fires on
    // `didFinish`. Waits for the viewer's MapLibre instance (the global
    // `map`) to exist, then adds a GeoJSON source + line layer.
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
        } else if (tries < 120) {
          setTimeout(function() { waitForMap(cb, tries + 1); }, 100);
        } else {
          console.error('mcpzim: __mcpzimMap not ready after 12s, giving up');
        }
      }
      function frameRoute(m) {
        var b = coords.reduce(function(b, c) { return b.extend(c); },
                              new maplibregl.LngLatBounds(coords[0], coords[0]));
        m.fitBounds(b, { padding: 40, duration: 0 });
      }
      waitForMap(function(m) {
        try {
          if (m.getSource('mcpzim-route')) {
            m.getSource('mcpzim-route').setData({
              type: 'Feature', geometry: { type: 'LineString', coordinates: coords }
            });
          } else {
            m.addSource('mcpzim-route', {
              type: 'geojson',
              data: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords } }
            });
            m.addLayer({
              id: 'mcpzim-route-line',
              type: 'line',
              source: 'mcpzim-route',
              layout: { 'line-join': 'round', 'line-cap': 'round' },
              paint: { 'line-color': '#2563eb', 'line-width': 4, 'line-opacity': 0.9 }
            });
          }
          frameRoute(m);
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
