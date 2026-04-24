# How MCPZimChat consumes the streetzim ZIM

Reference for the streetzim-generator developer so data format decisions are
aware of the client contract on the other end. Everything below is offline,
single-process, pure Swift (no HTTP, no runtime network).

## Entries we read

| Entry path | Consumer | Purpose |
| --- | --- | --- |
| `routing-data/graph.bin` | `SZRGGraph.parse` (`swift/Sources/MCPZimKit/SZRGGraph.swift`) | CSR routing graph: node `lat`/`lon`, adjacency offsets, edge arrays (`target`, `distMeters`, `speedKmh`, `geomIdx`, `nameIdx`), name table, per-edge polyline blob. Zero-copy parsed against `Data.withUnsafeBytes`. Supports SZRG v2 / v3 / v4 (inline geoms) and v5 (split — see next row). |
| `routing-data/graph.bin` (**no geoms**) | same | We currently parse with `decodeGeoms: false` (`SZRGGraph.parse(_:geomsData:decodeGeoms:)`): skips the polyline blob entirely. Saves ~600 MB on country-scale graphs. Route polylines fall back to node-sequence lat/lons. If we ever wire per-edge curvature into the UI we'll flip this back on — for v5 split ZIMs that means also passing `routing-data/graph-geoms.bin` as `geomsData`. |
| `routing-data/graph-geoms.bin` (**v5 only**) | `SZRGGraph.parseSZGMGeoms` | SZGM v1 companion carrying the geom_offsets + zigzag-varint polyline blob that v5 hoists out of graph.bin. Only opened when the caller asks for decoded geometries. |
| `routing-data/graph-chunk-manifest.json` + `routing-data/graph-chunk-NNNN.bin` | `SZRGChunked.reassembleChunked` | Byte-range split of graph.bin for continent-scale ZIMs. Manifest validates schema (== 1), each chunk's declared byte count, total bytes, and sha256 of the reassembled buffer. Same layout applies to `graph-geoms-chunk-manifest.json` + `graph-geoms-chunk-NNNN.bin` for the v5 companion. `ZimService.loadGraph` tries the single-file primary first and only falls back to the chunked layout when the primary entry is missing, so old ZIMs keep working unchanged. |
| `routing-data/graph.bin` or `routing-data/graph-chunk-manifest.json` presence | `classifyZim` (`ZimReader.swift`) | Either triggers `ZimKind.streetzim` classification. |
| `search-data/manifest.json` | `ZimService.loadManifest` | `{"chunks": {prefix → count}}`. Used to skip prefixes that have no records when geocoding, and to enumerate chunks for `near_places`. |
| `search-data/<prefix>.json` | `ZimService.loadChunk` | Array of place records: `[{name, type, subtype, location, lat, lon}, …]`. `type` ≈ OSM top-level key (e.g. "place"), `subtype` ≈ more specific (e.g. "city"). |
| `map-config.json` (presence only) | `classifyZim` | Another classification signal for `streetzim`. We don't read contents. |

## What the tools do with that data

| Tool | streetzim pieces it touches |
| --- | --- |
| `plan_driving_route(origin_lat, origin_lon, dest_lat, dest_lon, zim?)` | `graph.bin` → `nearestNode()` on origin/dest → `aStar()` → returns distance, duration, polyline (from node seq), `roads` list grouped by `edgeNameIdx`, and a simple `turn_by_turn` array of `"<road> for X km (~Y min)"` strings. |
| `geocode(query, limit, kinds?, zim?)` | `manifest.json` + one `<prefix>.json` chunk. Ranks by `Geocoder.rank(…)` (name match quality, type priority). |
| `route_from_places(origin, destination, zim?)` | `geocode` twice, then `plan_driving_route`. If `zim` is missing or not a loaded filename, falls through every loaded streetzim and picks the first one where both endpoints resolve — so a DC query finds the DC streetzim even with a baltics one also loaded. |
| `near_places(lat, lon, radius_km, kinds?, limit, zim?)` | Scans **every** `search-data/*.json` chunk, filters by haversine distance, optionally filters by `type`/`subtype` (e.g. `kinds=["restaurant","cafe"]`), sorts by distance. **This is linear in the total number of POIs** — a category index (see "asks" below) would turn it into a single file read. |
| `list_libraries` | ZIM metadata (`Name`, `Title`, `Description`, `Language`, `Creator`, `Publisher`, `Date`, `Tags`, `Counter`) + the classification bit from above. |

## Model-facing trimming (what the LLM actually sees)

The raw route payload is 50+ KB for country-scale trips. We trim it inside
`ChatSession.trimForModel(...)` before handing to Gemma 4:

- `polyline` → `{points: N, first: [lat,lon], last: [lat,lon]}` (the UI still
  gets the full array from the untrimmed tool trace — used by the MapKit/
  WKWebView route overlay).
- `turn_by_turn` → first 20 items + `turn_by_turn_total` count.
- `roads` → same, first 20 + `roads_total`.
- Distance reformatted to the host locale (`"104.8 mi"` on US, `"168.6 km"`
  elsewhere); duration reformatted to `"2h 32m"`.
- For `search`: cap at top 10 hits; for `get_article`: cap at 24 KB of text.

So anything you pack into a record beyond `name/type/subtype/location/lat/lon`
we'll happily forward or use, but short, skim-friendly strings serve Gemma 4
better than raw HTML or long metadata blobs.

## Classification signals (so the tool-picker on the LLM side knows what to call)

`classifyZim` checks, in order:

1. `routing-data/graph.bin` or `routing-data/graph.json` entry present → streetzim.
2. `map-config.json` present → streetzim.
3. Name/filename prefix `"streetzim"` / `"street_"` → streetzim.
4. Tag `"streetzim"` → streetzim.

Any of those will flip the ZIM into the streetzim tool set. Currently they're
mutually equivalent; if one is cheaper to emit than the others, that's fine.

## Asks that would unlock new offline-only capabilities

Prioritized by ROI for the MCP agent, not necessarily by your cost:

1. **`streetzim-meta.json`** — single small file with:
   - bbox (minLat, minLon, maxLat, maxLon),
   - display name for the covered region,
   - generator date,
   - `hasRouting`, `hasSatellite`, `hasWikidata` flags,
   - POI count per category,
   - geom graph size (nodes, edges), address count.

   Lets us surface a `zim_info` tool so the agent can reason "does this ZIM
   cover my query area?" before calling anything expensive. We'd also use
   `hasSatellite` to decide whether to expose the satellite toggle in the
   WKWebView.

2. **Wikipedia/Wikidata cross-refs on POI records** — see the dedicated
   "Cross-ZIM linking" section below for the full field spec.

3. **Category index** — `category-index/amenity.json`,
   `category-index/tourism.json`, etc. Same record shape you already emit
   (`{name, lat, lon, subtype, location}`), pre-grouped by OSM top-level key.
   `near_places` today is O(N) in total POIs; with this it's one file read
   per category. Biggest latency win for broad queries like *"top museums
   near me"* on country-scale ZIMs.

Lower priority:

4. **Boundary polygons** (`boundaries/*.geojson`) — polygon-in-polygon for
   *"what neighborhood is this?"*. Already expressible via vector tiles if
   we render a WKWebView; lower urgency on the native tool path.

5. **Coarse LLM-friendly elevation grid** — we can stick with your vector
   tiles for visual; only worth exposing as a tool if we get questions like
   *"is this route hilly?"*.

## Cross-ZIM linking (streetzim ↔ Wikipedia) — data spec

The high-value agent flow we want to unlock is:

> User: *"tell me about this neighborhood"*
> Agent: calls `near_places(lat, lon, radius)` → gets N POIs each carrying
> a Wikipedia article path → calls `get_article` on each against the
> locally-loaded Wikipedia ZIM → synthesises a summary across all of them.

This works entirely offline (streetzim ZIM + Wikipedia ZIM on the same
device), but needs the streetzim generator to surface the linkage already
present in the OSM source. Concretely:

### Fields needed on POI records (in `search-data/<prefix>.json` chunks)

Add the following **optional** string fields alongside the existing
`name / type / subtype / location / lat / lon`:

| Field | Source | Example | Notes |
| --- | --- | --- | --- |
| `wikipedia` | OSM tag `wikipedia` | `"en:Lincoln_Memorial"` | Keep the `<lang>:<Article_Title>` form — mirrors OSM exactly, no normalization needed. |
| `wikidata` | OSM tag `wikidata` | `"Q162458"` | Future-proof for switching Wikipedia languages / editions. |

Same thing applies to rows in **category-index files** (when/if those ship)
so we don't lose the cross-ref in the faster path.

The build-time cost is trivial — `osmium tags-filter` already produces
tag-dict output, so adding two more columns is the 4-line change mentioned
in the team's feedback.

### Fields useful at the ZIM-meta level (`streetzim-meta.json`)

| Flag | Purpose |
| --- | --- |
| `hasWikidata` | Already on the team's do-soon list. Lets us conditionally emit the chaining instruction in the LLM tool preamble. |
| `wikiCrossRefCount` | Optional. How many POIs carry a `wikipedia` tag. If small (say <50), the agent can budget one `get_article` per hit without blowing context. |
| `wikipediaLang` | e.g. `"en"`. Saves the agent from parsing `wikipedia` strings to guess the ZIM edition. |

### Path format conventions we'll rely on

- `wikipedia` uses the raw OSM format (`<lang>:<Title>`). We'll split once
  at the first `:`, URL-unescape the right-hand side, and map to our
  Wikipedia ZIM path (`A/<Title>`) on our end.
- Don't normalise underscores vs. spaces — the ZIM's own URL table will
  do the right thing when we ask `get_article`.
- If multiple Wikipedia tags exist (`wikipedia=en:X;de:Y`), emit them
  all in the `wikipedia` field as a `;`-separated list so we can pick the
  one whose edition we have locally. Multi-language users will care.

### Edge cases / non-goals

- **No need to validate** that the article exists on HF, Wikipedia, or
  anywhere else. We check existence locally via `ZimReader.read(path:)`
  when the agent is about to cite it. A dead link is a non-event.
- **No need to dedupe** wiki paths across POIs. The agent will, because
  it's cheap.
- **Don't embed article text** in streetzim — the whole point is that the
  Wikipedia ZIM is a separate artifact; duplicating would double the
  on-disk footprint.

### How we'll consume it

Once those fields are present:

1. `MCPToolAdapter.encodePlace` starts forwarding `wikipedia` / `wikidata`
   to the model — one-line change, no new schema.
2. `near_places` trimForModel surfaces `wiki_paths: [string]` (unique,
   preserved order) in the top-level of the result so Gemma sees "these
   articles are available" in one glance.
3. System-turn prompt fragment (appended only when `hasWikidata`):
   > *"If a result includes a `wikipedia` field, you may call `get_article`
   > with `path: "A/" + title_after_colon` to read the full article and
   > synthesise."*
4. Optional: a higher-level `summarize_area(lat, lon, radius_km)` tool
   that hides the fan-out. If your model is too small to orchestrate the
   chain reliably, this single tool does it server-side (client-side in
   our case, but same idea). We won't implement it until we've watched
   Gemma try the raw chain a few times.

## Current route rendering

SwiftUI MapKit polyline + origin/destination markers, pulled from the
untrimmed tool payload in `ToolCallTrace.rawResult`. We're about to swap
this for a `WKWebView` that embeds your viewer's zoom, layer toggles and
(when available) satellite tiles, so users get parity with what a browser
on the ZIM gets. The WKWebView will use a custom `WKURLSchemeHandler` that
serves ZIM entries via `LibzimReader.read(path:)` — no HTTP server needed.
Any additional tile or style endpoint your viewer reaches at runtime
(`/api/tiles/…`, etc.) needs to resolve to a ZIM entry path for this to
work offline.
