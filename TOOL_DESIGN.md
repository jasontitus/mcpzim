# Zimfo tool design

## Mission and scope

Zimfo answers two kinds of questions on a phone, fully offline:

1. **Places.** "Where am I?", "What's around here?", "Directions to X", "How much longer?", "Is there a post office nearby?", "Tell me something interesting about where I am."
2. **Wikipedia.** "Tell me about X.", "How is X different from Y?", "What's the history of X?", "Give me an overview of Z.", "How have A and B gotten along?"

Everything else is out of scope. A question that doesn't map cleanly onto one of those two buckets should surface as a "this app isn't the right tool" result, not a confident-but-wrong answer from the model.

## Why tools carry the thinking

The model is a 4-bit-quantized 4B parameter Gemma running on a memory-constrained phone. It's good at:

- Mapping a natural-language question onto a tool it knows exists.
- Phrasing a fluent answer from a structured payload.
- Following small, concrete instructions ("if the user asked for X in a named city, call `near_named_place`").

It is **bad** at:

- Remembering OSM kind vocabularies ("is a post office `amenity=post_office` or `office=post_office`?").
- Knowing coordinates of places ("is San Francisco at 37.77, -122.41 or 37.44, -122.15?" — it'll default to the value baked into the system preamble, which is the user's coord).
- Multi-hop reasoning over tool chains ("list sections, find the 'History' one, fetch that one").
- Picking *the best* search hit vs. the first.
- Comparing two encyclopedia entries section-by-section.

So the rule for this codebase: **if a question shape requires more than one tool call to answer reliably, package it as a composite tool.** Model orchestrates, tools retrieve, tools *also* pre-chew when the model would otherwise stitch results together badly.

## Design principles

1. **One tool per question shape, not per primitive operation.** A user never asks for "list sections" in isolation — that's a pre-step to a section fetch. Ship `article_overview` that does both.
2. **Deterministic everywhere possible.** Geocoding, routing, synonym expansion, section selection, snippet extraction all live in Swift. The model never invents coordinates or synonym candidates.
3. **Tools reject instead of soldiering on.** `near_places(origin="San Francisco")` returns a clear error that tells the model to call `near_named_place` instead. Silent wrong answers are worse than a retry loop.
4. **Preamble carries tool-selection rules, not tool prose.** The tool's own description (passed in the Gemma-4 tool schema) explains what it does. The system preamble maps natural-language question shapes onto tool names.
5. **Shape the args so the model can't screw it up.** `origin: string` accepts a place name, a "lat,lon" coord, or the sentinel `"my location"`. Numeric-only args go to separate tools (`plan_driving_route`) that explicitly gate on prior resolution.
6. **Composites delete ambiguity.** "Tell me something interesting about where I am" is one call (`nearby_stories`), not a chain of `near_places(has_wiki=true)` + N×`get_article(title=, section="Lead")`.

## Tool catalog — current

These ship today in `swift/Sources/MCPZimKit/MCPToolAdapter.swift`:

| Tool | Purpose | Arguments | Notes |
|---|---|---|---|
| `near_places` | Proximity POI search around raw lat/lon | `lat: Double, lon: Double, radius_km?: Double, kinds?: [String], has_wiki?: Bool, limit?: Int, zim?: String` | Rejects when `lat==0 && lon==0`. Use `near_named_place` for place names. |
| `near_named_place` | Proximity POI search around a named place | `place: String, kinds?: [String], radius_km?: Double, limit?: Int` | Geocodes `place` internally. Returns resolved coord + results. |
| `route_from_places` | Driving route from place name to place name | `origin: String, destination: String, zim?: String` | Accepts `"my location"` (substituted host-side) or any geocodable name. |
| `plan_driving_route` | Driving route from raw coord to raw coord | `origin_lat, origin_lon, destination_lat, destination_lon` | Only when BOTH endpoints came from a prior tool result. |
| `search` | Full-text ZIM search | `query: String, zim?: String, limit?: Int` | Fast-path for free-text lookup across any ZIM. |
| `get_article` | Full article by title | `title: String, zim?: String` | Default prose + sections. |
| `get_article_section` | Specific section by name | `title: String, section: String, zim?: String` | When a section is known. |
| `list_article_sections` | Section map for an article | `title: String, zim?: String` | Use to pick a target section before fetching. |
| `get_article_by_title` | Direct-path title lookup | `title: String, zim?: String, section?: String` | Skips full-text search — for exact titles like `"HP Garage"`. |
| `show_map` | Open the map view for a place | `place: String` | UI side-effect. |
| `get_main_page` / `list_libraries` / `zim_info` | ZIM housekeeping | various | Rarely useful to the model; exposed for completeness. |

## Tool catalog — proposed additions

These would each subsume a 2–4-call chain the model currently bungles.

### 1. `nearby_stories(lat, lon, radius_km?: Double = 2, max_stories?: Int = 5)`

**Purpose.** "Tell me something interesting about where I am." / "What's the history of this area?"

**Composite of.** `near_places(has_wiki=true, limit=N)` → for each hit, `get_article_section(section="Lead")` → return `[(place_name, wiki_title, ~180-char excerpt, lat, lon), ...]`.

**Rationale.** The model currently emits `near_places(has_wiki=true)`, sees a list of titles, and either (a) gives up and recites the list, or (b) picks one title arbitrarily and fires `get_article`. Packaging it guarantees we return story-ready snippets on the first call.

**Model hint.** "Use `nearby_stories(lat=…, lon=…)` when the user asks about the area around them in a story / interest / history framing. The tool returns short excerpts ready for narration; you just read them out."

### 2. `nearby_stories_at_place(place: String, radius_km?: Double = 2, max_stories?: Int = 5)`

**Purpose.** Same as above but for a different, named place. "Tell me interesting stories from Palo Alto."

**Composite of.** Geocode(place) → `nearby_stories(lat, lon, ...)`.

**Model hint.** Same family as `near_named_place` — use when the query binds to a named place, not the user's current position.

### 3. `route_status()`

**Purpose.** "How much longer?" / "What's my next turn?" / "Am I there yet?"

**Args.** None — the host tracks the currently-active route in ChatSession state (last `route_from_places` / `plan_driving_route` result).

**Returns.** `{ remaining_km, remaining_minutes, next_step, total_steps, progress_pct }` computed from current GPS + the stored polyline.

**Rationale.** A stateful tool the model shouldn't have to reason about. "How much longer" today forces the model to re-derive from conversation context (badly). The host already has the polyline; this call is a pure lookup.

### 4. `article_overview(title: String, max_sections?: Int = 5, zim?: String)`

**Purpose.** "Tell me about X" with appropriate depth.

**Composite of.** `list_article_sections(title)` → pick lead + the N biggest/most-relevant non-boilerplate sections → concatenated result.

**Rationale.** Saves the model from issuing list-sections + N section fetches. Tool picks based on section length heuristics (skip "References", "External links", "See also"; prefer sections named "History", "Overview", "Background", or the first few major H2s).

### 5. `compare_articles(titles: [String], section?: String, zim?: String)`

**Purpose.** "How is A different from B?" / "How has A gotten along with B?"

**Args.** `titles`: 2–4 article titles. `section`: optional — if specified, returns that section from each. If omitted, returns each article's lead + first 2 major sections, side-by-side.

**Returns.** `[{title, sections: [{name, text}]}]` aligned by role so the model can narrate the comparison.

**Rationale.** The model currently fetches each article separately, drops most of the content between calls (context window pressure), and synthesizes incoherently. Batch it.

### 6. `article_relationship(a: String, b: String, zim?: String)`

**Purpose.** "How has the US gotten along with Iran?" / "Relations between A and B?"

**Strategy.**
1. Probe for a dedicated relations article: `"Foreign relations of A"`, `"A–B relations"`, `"History of A–B relations"` (try each via `get_article_by_title` with fast failure).
2. If found, return the article's lead + any section mentioning `b`.
3. If not, fall back to `get_article_section(a, "Foreign relations")` + same for `b`, returning both.

**Rationale.** Pure retrieval heuristic the model gets wrong 80% of the time. Bake it.

### 7. `geocode(place: String)`

**Purpose.** Standalone resolution — "what are the coords of Adams Morgan?".

**Returns.** `{ resolved_name, lat, lon, country?, region?, confidence }`.

**Rationale.** Implicitly lives inside `route_from_places` and `near_named_place`, but exposing it independently lets the model chain when it wants explicit control ("find coords for X, then plan_driving_route from here"). More importantly — it surfaces WHAT the geocoder resolved to, so the model can confirm ambiguous names ("Springfield" → "Springfield, IL vs MO").

### 8. `what_is_here(lat?, lon?)`

**Purpose.** "Where am I?" / "What neighborhood is this?"

**Args.** Optional lat/lon (defaults to current GPS).

**Returns.** `{ nearest_named_place, neighborhood?, admin_area, wiki_summary?: string }`. The summary is the lead paragraph of the neighborhood's Wikipedia article when one exists.

**Rationale.** Today "where am I" fires `near_places` with no `kinds` and the model reads out random POIs. A reverse-geocode + neighborhood-article pairing is the actual right answer.

## Tool-selection decision tree (for the preamble)

```
Q: Does the question reference the user's current position?
├─ "here" / "me" / "nearby" / "around here"
│   ├─ generic: nearby_stories(lat, lon)
│   ├─ typed ("nearest post office"): near_places(lat, lon, kinds=[…])
│   ├─ route: route_from_places(origin="my location", destination=<name>)
│   └─ status ("how much longer"): route_status()
│
├─ A named place ("San Francisco", "Berkeley")
│   ├─ typed ("restaurants in SF"): near_named_place(place=…, kinds=[…])
│   ├─ story ("interesting stories from Palo Alto"): nearby_stories_at_place(place=…)
│   ├─ directions: route_from_places(origin=…, destination=…)
│   └─ "tell me about <place>": article_overview(title=<place>)
│
└─ A topic (no location anchor)
    ├─ overview: article_overview(title=<topic>)
    ├─ comparison: compare_articles(titles=[a, b])
    ├─ relationship: article_relationship(a, b)
    ├─ specific section: get_article_section(title=, section=)
    └─ free-text: search(query=)
```

The preamble should carry ONLY this tree as prose — no tool descriptions, since those travel with the tool schema. The tree is what the model needs at inference time to pick the right call.

## Failure modes the tools should defend against

1. **Model passes a place name to a coord-only tool.** Return `{"error": "near_places requires numeric lat+lon. For a named place, call near_named_place(place=…) instead."}`. We do this today for `near_places`; add the same rejection to `plan_driving_route`.

2. **Model passes `"my location"` / synonyms where not expected.** The host's `substituteCurrentLocation` resolves these in `route_from_places` origin/destination. Don't let these leak into tools that don't handle them (reject with a clear error).

3. **Model pins `zim=` to the wrong ZIM.** A `near_places` call against `wikipedia_en_all.zim` is meaningless — it needs the streetzim. Today we strip obviously-wrong pins host-side; going forward, each tool should specify in its description which ZIM types it accepts.

4. **Model loops on an empty result.** `near_places` with tight `radius_km` or an unknown `kind` returns `[]`. The model often retries with the same args. The tool response should include `suggested_next: "widen radius_km to X"` or `suggested_next: "check kinds vocabulary"` to break the loop.

5. **Model hallucinates a section name.** `get_article_section(title="Lithuania", section="Medieval History")` when the actual section is "Middle Ages". The tool should fuzzy-match and return the best match with a note, or return the section list for retry.

## Non-goals

- **No search beyond ZIMs.** No web fetch, no LLM-assisted query expansion. Offline is the product.
- **No calendar / reminders / messaging tools.** Those belong to Apple's own assistants; the app's pitch is places + Wikipedia, not general-purpose voice agent.
- **No user-writeable tools.** All tools are read-only. The app doesn't mutate anything offline.

## Implementation order (when we ship these)

1. `route_status()` — biggest UX win for voice-while-driving, purely local.
2. `nearby_stories(lat, lon)` — the "interesting stories" query is currently broken end-to-end.
3. `article_overview(title)` — cuts the most tokens across the "tell me about X" class.
4. Preamble decision-tree refactor — depends on the three above existing.
5. `compare_articles` + `article_relationship` — both ride on overview + section-picking heuristics.
6. `geocode` + `what_is_here` — round out the surface area, low urgency.

## References

- Current tool wiring: `swift/Sources/MCPZimKit/MCPToolAdapter.swift`.
- Preamble construction: `ios/MCPZimChat/Chat/ChatSession.swift::composeSystemMessage`.
- ZIM tool backend: `swift/Sources/MCPZimKit/ZimService.swift`.
- Geocoder: `swift/Sources/MCPZimKit/Geocoder.swift`.
