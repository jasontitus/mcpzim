// SPDX-License-Identifier: MIT
//
// Cheap heuristic classifier that slots a user turn into one of four
// buckets. Purpose for now is observability — ChatSession logs the
// classification into the debug pane so we can calibrate before
// wiring it into the retrieval depth decision (Phase 2b) and
// map-reduce synthesis path (Phase 2c).
//
// Deliberately keyword-based, not model-backed:
//   * Runs in microseconds, no token cost.
//   * Deterministic + easy to unit-test.
//   * Caller can always override / ignore.

import Foundation

public enum QueryComplexity: String, Sendable, Equatable {
    /// Routing / "what's around X" / local-geography questions —
    /// these should go through streetzim tools, not Wikipedia.
    case navigational
    /// Short, single-fact lookup — one Wikipedia lead section or
    /// infobox is plenty. "Who was Marie Curie?", "When was the War
    /// of 1812?"
    case factoid
    /// "Tell me about X" / "what is X" — benefits from 2–3 sections
    /// of a single article.
    case topical
    /// "Explain how X works" / "compare X and Y" — needs multi-
    /// article or multi-section synthesis. Phase 2c will route this
    /// through a map-reduce summary pass.
    case explanatory
}

public extension QueryComplexity {
    static func classify(_ text: String) -> QueryComplexity {
        let lower = text.lowercased()
        let trimmed = lower.trimmingCharacters(in: .whitespacesAndNewlines)

        // Navigational first — strongest signal and also the one
        // where misclassifying wastes the most effort (spinning up
        // Wikipedia retrieval for "how do I get to X").
        let navigationalSignals = [
            "route", "directions", "how do i get", "how do i drive",
            "get to ", "get from ", "near here", "near me",
            "what's around", "whats around", "what is around",
            "what's in ", "whats in ", "what is in ",
            "what's nearby", "whats nearby", "what is nearby",
            "nearest ", "closest ", "how long until",
            "how long does it take",
            // Common OSM-category drill-ins we handle via near_places.
            "bars near", "bars around", "bars in ",
            "cafes near", "cafes around", "cafes in ",
            "restaurants near", "restaurants around", "restaurants in ",
            "museums near", "museums around", "museums in ",
            "coffee near", "coffee in ",
        ]
        if navigationalSignals.contains(where: { lower.contains($0) }) {
            return .navigational
        }

        // Explanatory signals: multi-step reasoning or multi-entity
        // synthesis, typically answerable only by reading across
        // sections or articles.
        let explanatorySignals = [
            "explain ", "explain how", "explain why",
            "how does ", "how do ",
            "why does ", "why do ", "why did ", "why is ", "why are ", "why was ",
            "compare ", "contrast ", "difference between",
            "describe how", "describe why",
            "what caused", "what led to", "how did ",
            "what's the difference", "whats the difference",
        ]
        if explanatorySignals.contains(where: { lower.contains($0) }) {
            return .explanatory
        }

        // Factoid: short query with a direct "who/when/where/what-
        // year/what-date" opener.
        if trimmed.count < 70 {
            let factoidStarters = [
                "who ", "who was", "who is ", "who were",
                "when ", "when was", "when did", "when is",
                "where ", "where is ", "where was ",
                "what year", "what date", "how many ", "how much ",
            ]
            if factoidStarters.contains(where: { trimmed.hasPrefix($0) }) {
                return .factoid
            }
        }

        // Everything else — "tell me about X", "what is X", open-
        // ended informational — falls into topical.
        return .topical
    }
}
