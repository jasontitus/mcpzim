// SPDX-License-Identifier: MIT
//
// Fast-path intent router + one-line places-reply synthesiser.
// Both are pure text-in-text-out functions; living in MCPZimKit (not
// the iOS target) keeps them exercised by `swift test` alongside the
// rest of the kit.
//
// The iOS `ChatSession` wraps these and handles the side effects
// (tool dispatch, state mutation); nothing here touches UI, storage,
// or the model runtime.

import Foundation

/// A resolved fast-path dispatch — tool name + args. The iOS side
/// turns this into an `adapter.dispatch` call and a synthesised
/// assistant caption, all without paying the LLM prefill.
public struct DirectIntent: Equatable, Sendable {
    public let toolName: String
    public let args: [String: AnyJSONValue]

    public init(toolName: String, args: [String: AnyJSONValue]) {
        self.toolName = toolName
        self.args = args
    }

    /// Convenience for hosts that speak the native `[String: Any]`
    /// dict the `MCPToolAdapter.dispatch` signature expects.
    public var anyArgs: [String: Any] {
        args.mapValues { $0.anyValue }
    }
}

/// Small, sendable JSON scalar/container shim. Swift's `[String: Any]`
/// isn't `Sendable` or `Equatable`, which makes `DirectIntent`
/// awkward to compare in tests. This covers everything the router
/// needs to emit (strings, numbers, bool, arrays thereof).
public enum AnyJSONValue: Equatable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([AnyJSONValue])
    case object([String: AnyJSONValue])

    public var anyValue: Any {
        switch self {
        case .string(let s): return s
        case .int(let i):    return i
        case .double(let d): return d
        case .bool(let b):   return b
        case .array(let xs): return xs.map(\.anyValue)
        case .object(let o): return o.mapValues { $0.anyValue }
        }
    }
}

public enum IntentRouter {

    /// Attempt to match the raw user text against one of the fast-path
    /// patterns the LLM would otherwise solve. Returns `nil` on
    /// anything unusual — callers fall back to the LLM loop.
    ///
    /// `currentLocation` is required for the "<X> near me" pattern;
    /// all other patterns are location-independent.
    ///
    /// Default search radius is 5 km (the LLM typically picks 1–5;
    /// without context we pick the generous end so "museums in
    /// Mountain View" finds the 15 in-range instead of the 0
    /// within 1 km).
    public static func classify(
        _ raw: String,
        currentLocation: (lat: Double, lon: Double)? = nil
    ) -> DirectIntent? {
        let text = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?.!"))
        if text.isEmpty { return nil }
        let lower = text.lowercased()
        let defaultRadiusKm: Double = 5

        // "<category> near me" / "<category> around here" — use GPS.
        // We match FIRST, then require the location. If the pattern
        // matched but we don't have a location, we must NOT fall
        // through to the generic `<X> in|near|at <Y>` pattern below
        // — it would gladly classify "me" as a place.
        if let m = match(lower, pattern: #"^(.+?)\s+(?:near|around)\s+(me|here)$"#) {
            guard let here = currentLocation else { return nil }
            let kind = singularize(m[0])
            return DirectIntent(toolName: "near_places", args: [
                "lat":       .double(here.lat),
                "lon":       .double(here.lon),
                "kinds":     .array([.string(kind)]),
                "radius_km": .double(defaultRadiusKm)
            ])
        }

        // "directions to <place>" / "route to <place>" / "navigate to <place>".
        // Checked BEFORE the `<X> in <Y>` pattern so destinations
        // containing " in " ("Library in Mountain View") still route
        // as directions, not as a places search. Polite prefixes like
        // "give me", "show me", "get me", "can you give me", and
        // leading "please" are accepted — otherwise "Give me directions
        // to SF" falls to the LLM, which sometimes emits malformed JSON
        // and silently drops the turn.
        let directionsBody = stripDirectionsPrefix(lower)
        if let m = match(directionsBody, pattern:
            #"^(?:directions?|route|navigate)\s+(?:to|for)\s+(.+)$"#)
            ?? match(directionsBody, pattern:
                #"^(?:how\s+(?:do\s+i\s+|to\s+)?get\s+to)\s+(.+)$"#)
            ?? match(directionsBody, pattern:
                #"^(?:take\s+me\s+to)\s+(.+)$"#)
        {
            let dest = m[0]
            return DirectIntent(toolName: "route_from_places", args: [
                "origin":      .string("my location"),
                "destination": .string(dest)
            ])
        }

        // "<category> in <place>" / "<category> near <place>". Named-place
        // geocoding happens inside the tool — we don't pre-resolve here.
        if let m = match(lower,
                         pattern: #"^(.+?)\s+(?:in|near|around|at)\s+(.+)$"#)
        {
            let kind = singularize(m[0])
            let place = m[1]
            // Guard: don't misclassify questions ("where can I find bars in SF")
            // or explanations ("how does rain form in clouds") as places.
            let skipLeadingWords: Set<String> = [
                "what", "where", "when", "why", "how", "who",
                "tell", "show", "find", "can", "could", "would",
                "should", "is", "are", "do", "does", "did"
            ]
            let firstWord = lower.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
            if !skipLeadingWords.contains(firstWord) {
                return DirectIntent(toolName: "near_named_place", args: [
                    "place":     .string(place),
                    "kinds":     .array([.string(kind)]),
                    "radius_km": .double(defaultRadiusKm)
                ])
            }
        }

        return nil
    }

    /// Turn an English plural into its singular form for the OSM
    /// `kinds` vocabulary. Handles the three common -s / -es / -ies
    /// patterns. Anything irregular (men, children, criteria…) is
    /// passed through — those are rare enough in POI categories
    /// that the OSM vocab's fuzzy match picks up the slack.
    public static func singularize(_ s: String) -> String {
        guard s.count > 3 else { return s }
        // "libraries" → "library". Must precede the generic -s rule
        // or we'd strip just the trailing s and keep a dangling "e".
        if s.hasSuffix("ies") {
            return String(s.dropLast(3)) + "y"
        }
        // "churches", "buses", "boxes", "quizzes" — two-char strip.
        for ending in ["ches", "shes", "xes", "ses", "zes"] {
            if s.hasSuffix(ending) {
                return String(s.dropLast(2))
            }
        }
        // Plain -s plural. Exclude -ss (passes, grass) which isn't
        // actually a plural.
        if s.hasSuffix("s"), !s.hasSuffix("ss") {
            return String(s.dropLast())
        }
        return s
    }

    /// Strip polite wrappers that wouldn't change the underlying
    /// directions intent. Voice input in particular produces these
    /// routinely ("Give me directions to…", "Can you show me
    /// directions to…"). We strip once, up-front, so the core
    /// patterns stay simple.
    private static func stripDirectionsPrefix(_ s: String) -> String {
        // Order matters: outer politeness first, then the "<verb> me"
        // pair. Kept as separate passes so each replacement is a
        // single anchored regex.
        var out = s
        let patterns = [
            #"^please\s+"#,
            #"^(?:can|could|would|will)\s+you\s+(?:please\s+)?"#,
            #"^(?:i\s+(?:need|want|would\s+like)|i'd\s+like)\s+"#,
            #"^(?:give|show|get|find|tell|fetch)\s+me\s+(?:(?:the|some)\s+)?"#,
        ]
        for p in patterns {
            out = out.replacingOccurrences(
                of: p, with: "", options: .regularExpression)
        }
        return out
    }

    /// Light regex helper — returns only the capture groups (not the
    /// full-match range). `NSRegularExpression` verbatim with a
    /// `nil`-to-`Substring[]` adapter.
    private static func match(_ text: String, pattern: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return nil
        }
        let range = NSRange(text.startIndex..., in: text)
        guard let m = regex.firstMatch(in: text, options: [], range: range),
              m.numberOfRanges >= 2
        else { return nil }
        var out: [String] = []
        for i in 1..<m.numberOfRanges {
            if let r = Range(m.range(at: i), in: text) {
                out.append(String(text[r]))
            }
        }
        return out
    }

    // MARK: - Reply synthesis

    /// Build the one-line caption we drop into the assistant bubble
    /// when the fast path (or post-tool skip-model-reply) wants to
    /// bypass the LLM's prose. The caller passes the tool args +
    /// the full tool result; the map bubble below carries the
    /// answer, this is just the header.
    public static func synthesizePlacesReply(
        toolName: String,
        args: [String: Any],
        fullResult: [String: Any]
    ) -> String {
        let kind: String = {
            if let k = (args["kinds"] as? [String])?.first, !k.isEmpty { return k }
            if let k = args["kinds"] as? String, !k.isEmpty { return k }
            if let q = args["query"] as? String, !q.isEmpty { return q }
            return "places"
        }()
        let where_: String = {
            if let p = args["place"] as? String, !p.isEmpty { return p }
            if (args["lat"] as? NSNumber) != nil
               && (args["lon"] as? NSNumber) != nil { return "you" }
            if let o = args["origin"] as? String, !o.isEmpty { return o }
            return "here"
        }()
        let count: Int? = {
            if let n = (fullResult["total_in_radius"] as? NSNumber)?.intValue { return n }
            if let rs = fullResult["results"] as? [[String: Any]] { return rs.count }
            if let ss = fullResult["stories"] as? [[String: Any]] { return ss.count }
            if let n = (fullResult["count"] as? NSNumber)?.intValue { return n }
            return nil
        }()
        let radiusKm = (fullResult["radius_km"] as? NSNumber)?.doubleValue

        // Pluralise once for the display line — "bar" → "bars";
        // "museums" → "museums"; "cafes" → "cafes".
        let kindPlural = kind.hasSuffix("s") ? kind : kind + "s"

        var line: String
        if let n = count, n > 0 {
            line = "Found \(n) \(kindPlural) near \(where_)"
        } else if count == 0 {
            line = "No \(kindPlural) found near \(where_)"
        } else {
            line = "Results for \(kindPlural) near \(where_)"
        }
        if let r = radiusKm { line += " (within \(formatKm(r)))" }
        line += ". Tap a pin on the map for details"
        if count != 0 { line += ", or tap List for the full rundown" }
        line += "."
        return line
    }

    private static func formatKm(_ km: Double) -> String {
        if km < 1 { return "\(Int(km * 1000)) m" }
        if km.truncatingRemainder(dividingBy: 1) == 0 {
            return "\(Int(km)) km"
        }
        return String(format: "%.1f km", km)
    }
}
