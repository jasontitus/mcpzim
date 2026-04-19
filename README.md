# mcpzim

An [MCP](https://modelcontextprotocol.io) server that makes a group of offline
[ZIM](https://wiki.openzim.org/) files available to local LLM agents. Point it
at a directory of ZIMs and the server will:

- Inventory what's there (`list_libraries`) and advertise aggregate
  capabilities (general knowledge, medical knowledge, maps/routing) based on
  what's loaded.
- Expose search and article retrieval across every ZIM (`search`,
  `get_article`, `get_main_page`).
- When it detects a [streetzim](https://github.com/jasontitus/streetzim) ZIM
  built with `--routing`, it additionally exposes `plan_driving_route`,
  `geocode`, and `route_from_places` so a local agent can ask *"give me a
  driving route from A to B"* and get street-by-street directions, distance,
  and an estimated time.

The design principle is **opportunistic capability**: start with one Wikipedia
ZIM and you get a Wikipedia server. Drop in `mdwiki_en_all_*.zim` and it also
answers medical questions. Drop in a streetzim ZIM and it can also plan driving
routes for the area the ZIM covers. Tools appear only when the underlying data
is present, so the agent's tool list never lies about what the server can do.

## Install

Requires Python 3.10+.

```sh
pip install mcpzim               # once published
# or, from a checkout:
pip install -e .
```

`libzim` is a native wheel; prebuilt wheels exist for macOS (x86_64/arm64),
Linux (x86_64/aarch64, glibc and musl) and Windows x64. On other platforms pip
will build from source and you'll need a C++ toolchain.

## Run

Drop your ZIM files into one directory and run:

```sh
export ZIM_DIR=~/zims
mcpzim                           # stdio transport (Claude Desktop / Code)
mcpzim ~/zims/wikipedia.zim ~/zims/streetzim_ma.zim   # explicit paths
mcpzim --transport streamable-http --host 0.0.0.0 --port 8765  # LAN
```

Add to `~/.config/claude-desktop/claude_desktop_config.json` (or the
equivalent for your MCP client):

```json
{
  "mcpServers": {
    "mcpzim": {
      "command": "mcpzim",
      "env": { "ZIM_DIR": "/Users/me/zims" }
    }
  }
}
```

## Tools

Always available:

| Tool | What it does |
| --- | --- |
| `list_libraries` | Inventory: list every ZIM with kind, title, language, and the aggregate capabilities (`general_knowledge`, `medical`, `maps`, ...). Call this first. |
| `search` | Full-text search across every ZIM (uses libzim's Xapian index when present, falls back to title-prefix suggestions). Accepts an optional `kind` filter. |
| `get_article` | Fetch an entry by path; HTML is stripped of navbox / infobox / script cruft so the LLM sees clean text. |
| `get_main_page` | Main page of one ZIM, or of every loaded ZIM. |

Only present when a streetzim ZIM with routing data is loaded:

| Tool | What it does |
| --- | --- |
| `plan_driving_route` | A* over the streetzim routing graph. Input: two `lat/lon` pairs. Output: total distance, duration, polyline, and a road-segment list coalesced by street name. |
| `geocode` | Resolve a place/address string to coordinates using streetzim's prefix-chunked search index. |
| `route_from_places` | Convenience: geocode both endpoints then plan a route. |

Cost/heuristic in the router match streetzim's JS viewer exactly:
`edge_cost = distance_m / (speed_kmh / 3.6)` and `heuristic =
haversine / (100/3.6)`, so results are identical to what the in-browser
viewer would produce.

## Supported ZIMs

Type detection runs at scan time and uses a combination of filename prefix, the
ZIM's `Name` / `Tags` / `Creator` / `Publisher` metadata, and signature entries
inside the archive. Out of the box:

- **Wikipedia** — any `wikipedia_*.zim` (`Creator: Wikipedia`, tagged
  `wikipedia`).
- **mdwiki** — `mdwiki_*.zim` from the WikiProjectMed Foundation
  (tagged `mdwiki` / `medical`).
- **streetzim** — detected by the presence of `routing-data/graph.bin` or
  `map-config.json` inside the archive.
- **generic** — anything else (a `*.zim` still gets served; only the
  `ZimKind.GENERIC` default toolset applies).

## Example session

```
> list_libraries
{"zims": [
   {"path": ".../wikipedia_en_all_nopic_2026-03.zim", "kind": "wikipedia", ...},
   {"path": ".../mdwiki_en_all_2026-03.zim", "kind": "mdwiki", ...},
   {"path": ".../streetzim_ma.zim", "kind": "streetzim", "has_routing": true, ...}
 ],
 "by_kind": {"wikipedia": 1, "mdwiki": 1, "streetzim": 1},
 "capabilities": ["encyclopedia", "general_knowledge", "geocode",
                  "get_article", "list_libraries", "maps", "medical",
                  "plan_route", "search"]}

> route_from_places {"origin": "Boston Common", "destination": "Fenway Park"}
{"origin_resolved": {"name": "Boston Common", "lat": 42.3554, "lon": -71.0655, ...},
 "destination_resolved": {"name": "Fenway Park", "lat": 42.3467, "lon": -71.0972, ...},
 "distance_km": 3.27, "duration_min": 9.4,
 "roads": [
    {"name": "Beacon Street", "distance_m": 412.3, "duration_s": 44.0},
    ...
 ],
 "turn_by_turn": ["Beacon Street for 0.41 km (~0.7 min)", ...],
 "polyline": [[42.3554, -71.0655], ...]}
```

## Mobile

Concrete paths that actually work, matched to the on-device LLM hosts people
are shipping in 2026:

| Platform | LLM host | Path | Status |
| --- | --- | --- | --- |
| Desktop | Claude Desktop / Code, any MCP client | This Python server via `stdio` or `streamable-http` | **Works today** |
| Android | [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) (Gemma 4 + LiteRT-LM, Apache 2.0) | Small Kotlin fork — add a `@Tool fun callMcp(...)` that talks JSON-RPC/HTTP to this Python server | **See [`mobile/android/README.md`](mobile/android/README.md)** — ~80 lines of Kotlin + one SKILL.md |
| Android (fully offline) | same | Run `mcpzim` under Termux on the same device | Works; Termux has to build `libzim` from source (`pkg install python clang cmake`) |
| iOS | [Swift-Gemma4-Core](https://github.com/yejingyang8963-byte/Swift-gemma4-core) (MIT, iOS 17+) | Link **[`swift/MCPZimKit`](swift/)** — pure-Swift port of the routing graph parser, A*, geocoder + a transport-agnostic MCP tool adapter. Host app supplies a `ZimReader` backed by `CoreKiwix.xcframework` from the Kiwix project. | **See [`swift/README.md`](swift/README.md)** |
| iOS | Google AI Edge Gallery | Not possible — the iOS app is closed-source. | Waiting on Google |

The short version: on Android the open-source Agent Chat host already knows how
to call a tool, so a short Kotlin patch makes it speak to this Python server.
On iOS, the LLM host has no tool-calling layer yet, so the companion Swift
package ships (a) the same algorithms in pure Swift for in-process use, and (b)
a transport-agnostic MCP adapter you can plug into the official
[`modelcontextprotocol/swift-sdk`](https://github.com/modelcontextprotocol/swift-sdk)
when you want the model to call tools over LAN.

`swift/MCPZimKit`'s SZRG v2 parser, A* router, and prefix geocoder are
line-for-line ports of the Python implementations; the Python test suite and
the Swift test suite in `swift/Tests/MCPZimKitTests/` cover the same cases, so
if both green, you know the two agree.

## Development

```sh
pip install -e '.[dev]'
pytest
```

Tests do not require any real ZIM files; the routing tests build a tiny SZRG v2
graph in-memory using `mcpzim.routing.encode_graph_v2`, and the library tests
exercise the classifier directly.

## License

MIT.
