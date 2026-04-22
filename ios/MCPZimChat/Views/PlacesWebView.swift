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

/// Extracted payload from a tool trace. `origin` is the search centre the
/// tool operated on (user's coords for `near_places` / `nearby_stories`,
/// the geocoded place for the `*_at_place` / `_named_place` variants);
/// used as the centre of the radius ring.
struct PlacesPayload {
    struct Place: Hashable {
        let lat: Double
        let lon: Double
        let label: String
        let description: String
    }
    let places: [Place]
    let origin: (lat: Double, lon: Double)?
    let radiusKm: Double?
}

/// Read a tool trace's raw JSON and pull out:
///   • The places array (`results` for near_*, `stories` for nearby_stories_*)
///   • The coverage centre: prefer `resolved.{lat,lon}`, fall back to `origin`
///   • The `radius_km` arg the tool was called with
func parsePlaces(from trace: ToolCallTrace) -> PlacesPayload {
    guard let data = trace.rawResult.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return PlacesPayload(places: [], origin: nil, radiusKm: nil) }

    // near_places / near_named_place emit `results`, which carry
    // `name`, `type`, `lat`, `lon`, `distance_m`, optional `subtype`
    // and `location`. nearby_stories / nearby_stories_at_place emit
    // `stories`, which carry `place_name`, `wiki_title`, `excerpt`,
    // `lat`, `lon`, `distance_m`. Handle both.
    var places: [PlacesPayload.Place] = []

    if let results = json["results"] as? [[String: Any]] {
        for r in results {
            guard let lat = (r["lat"] as? NSNumber)?.doubleValue,
                  let lon = (r["lon"] as? NSNumber)?.doubleValue
            else { continue }
            let name = (r["name"] as? String) ?? "(unnamed)"
            let subtype = (r["subtype"] as? String) ?? ""
            let kind = (r["type"] as? String) ?? ""
            let distanceM = (r["distance_m"] as? NSNumber)?.intValue
            var desc: [String] = []
            if !subtype.isEmpty { desc.append(subtype) }
            else if !kind.isEmpty { desc.append(kind) }
            if let d = distanceM { desc.append(formatDistanceMeters(d)) }
            places.append(.init(
                lat: lat, lon: lon, label: name,
                description: desc.joined(separator: " · ")
            ))
        }
    }
    if let stories = json["stories"] as? [[String: Any]] {
        for s in stories {
            guard let lat = (s["lat"] as? NSNumber)?.doubleValue,
                  let lon = (s["lon"] as? NSNumber)?.doubleValue
            else { continue }
            let place = (s["place_name"] as? String) ?? "(unnamed)"
            // Take the first ~120 chars of the excerpt as the popup
            // description. Full text lives in the chat bubble above.
            let excerpt = (s["excerpt"] as? String) ?? ""
            let preview = excerpt.count > 140
                ? String(excerpt.prefix(140)) + "…"
                : excerpt
            places.append(.init(
                lat: lat, lon: lon, label: place, description: preview
            ))
        }
    }

    // Origin: prefer `resolved.{lat,lon}` (the geocoded target for the
    // `*_at_place` / `_named_place` variants), fall back to `origin`
    // (the user's coords passed in for the plain near_places /
    // nearby_stories variants).
    var origin: (lat: Double, lon: Double)? = nil
    if let resolved = json["resolved"] as? [String: Any],
       let lat = (resolved["lat"] as? NSNumber)?.doubleValue,
       let lon = (resolved["lon"] as? NSNumber)?.doubleValue
    {
        origin = (lat, lon)
    } else if let o = json["origin"] as? [String: Any],
              let lat = (o["lat"] as? NSNumber)?.doubleValue,
              let lon = (o["lon"] as? NSNumber)?.doubleValue
    {
        origin = (lat, lon)
    }

    let radius = (json["radius_km"] as? NSNumber)?.doubleValue

    return PlacesPayload(places: places, origin: origin, radiusKm: radius)
}

private func formatDistanceMeters(_ m: Int) -> String {
    if m < 1000 { return "\(m) m" }
    let km = Double(m) / 1000.0
    return String(format: "%.1f km", km)
}

struct PlacesWebView: View {
    let trace: ToolCallTrace
    @Environment(ChatSession.self) private var session
    @State private var presentFullscreen: Bool = false

    private var payload: PlacesPayload { parsePlaces(from: trace) }

    private var userLocation: (lat: Double, lon: Double)? { session.currentLocation }

    var body: some View {
        if let spec = resolveSpec(userLocation: userLocation) {
            PlacesWebContainer(spec: spec, session: session, payload: payload)
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

    func makeNSView(context: Context) -> WKWebView {
        makePlacesWebView(spec: spec, session: session, payload: payload)
    }
    func updateNSView(_ nsView: WKWebView, context: Context) {
        reloadPlacesIfNeeded(nsView, spec: spec, payload: payload)
    }
}
#else
import UIKit
private struct PlacesWebContainer: UIViewRepresentable {
    let spec: PlacesSpec
    let session: ChatSession
    let payload: PlacesPayload

    func makeUIView(context: Context) -> WKWebView {
        makePlacesWebView(spec: spec, session: session, payload: payload)
    }
    func updateUIView(_ uiView: WKWebView, context: Context) {
        reloadPlacesIfNeeded(uiView, spec: spec, payload: payload)
    }
}
#endif

@MainActor
private final class PlacesWebCoordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
    var pendingInjection: String?
    var log: ((String) -> Void)?

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
        let level = (payload["level"] as? String) ?? "log"
        let args = (payload["args"] as? [String]) ?? []
        log?("js.\(level) \(args.joined(separator: " "))")
    }
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
    let expected = "zim://\(spec.zimName)/\(spec.mainPath)"
    if webView.url?.absoluteString != expected {
        loadPlacesSpec(webView, spec: spec, payload: payload)
        return
    }
    if let coordinator = objc_getAssociatedObject(webView, &placesCoordinatorKey) as? PlacesWebCoordinator,
       let js = coordinator.pendingInjection
    {
        webView.evaluateJavaScript(js) { _, _ in }
    }
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
        placeObjs.append([
            "lat": p.lat, "lon": p.lon,
            "label": p.label, "description": p.description
        ])
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
      function waitForHook(cb, tries) {
        tries = tries || 0;
        if (typeof window.streetzimRouting !== "undefined"
            && window.streetzimRouting !== null
            && typeof window.streetzimRouting.showPlaces === "function")
        { return cb(); }
        if (tries > 100) {
          console.warn("mcpzim places: streetzimRouting.showPlaces missing after 10 s");
          return;
        }
        setTimeout(function() { waitForHook(cb, tries + 1); }, 100);
      }
      function waitForMap(cb, tries) {
        tries = tries || 0;
        var m = window.__mcpzimMap;
        if (m && typeof m.addSource === "function" && m.loaded && m.loaded()) cb(m);
        else if (tries < 300) setTimeout(function() { waitForMap(cb, tries + 1); }, 100);
      }
      waitForMap(function(m) {
        \(userDotJS)
      });
      waitForHook(function() {
        try {
          // Hide the viewer's own chrome — our chat-bubble container
          // owns the UI for this context.
          if (typeof window.streetzimRouting.setChromeVisibility === "function") {
            window.streetzimRouting.setChromeVisibility({
              search: false, controls: false, panel: false
            });
          }
          window.streetzimRouting.showPlaces(placesData, { fitBounds: true, padding: 50 });
          console.info("mcpzim places: dropped " + placesData.length + " pins");
        } catch (e) {
          console.error("mcpzim places: showPlaces threw", e);
        }
        \(ringJS)
      });
    })();
    """
    if let coordinator = objc_getAssociatedObject(webView, &placesCoordinatorKey) as? PlacesWebCoordinator {
        coordinator.pendingInjection = injectJS
    }
    var components = URLComponents()
    components.scheme = ZimURLSchemeHandler.scheme
    components.host = spec.zimName
    components.path = "/" + spec.mainPath
    guard let url = components.url else { return }
    webView.load(URLRequest(url: url))
}
