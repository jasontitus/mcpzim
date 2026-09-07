# Coffee → museum location continuity

The shared Firebase session `2026-09-05_09-12-17.log` contains the reported
failure at 09:16:26–09:16:46. The coffee request dispatches `near_places`
with GPS coordinates and a 5 km radius. “Museum?” then misses deterministic
routing, invokes Bonsai, and dispatches `near_places(place: "San Francisco",
kinds: ["museum"], radius_km: 5)`. The user did not request San Francisco.
The map accurately renders the results of a search around the wrong center.

The failure is an invented tool argument, even though the POIs themselves
come from StreetZIM. Exact Wikipedia excerpt enforcement does not prevent
this class of error. The model spends approximately 5.2 seconds choosing
that incorrect call.

## Repairs

- Persist the successful place search's actual center and radius separately
  from GPS and the first result. Keep its archive and resolved center label.
- Resolve category follow-ups such as “Museum?”, “Any good museums?”, and
  “What about parks?” using that search area before entity/pronoun routing.
  Chained category requests update the same context without model inference.
- Explicit locations and “near me” retain their normal precedence. A named
  city search followed by another category keeps the named city's center;
  “near me” returns to the current GPS fix.
- Normalize the coffee request wrapper to `coffee shop`; the old router
  passed `where is a good coffee shop` as a category. Results are described
  by distance, without inventing ratings or quality claims.
- Log the resolved search area and the farthest returned point's independently
  calculated distance. These diagnostics distinguish wrong query centers
  from incorrectly filtered results.
- Narrow museum/gallery requests within the shared tourism chip. Museum
  subtype evidence or an unambiguous museum name on a coarse tourism record
  is required; landmarks, galleries, and a cafe named “Museum Cafe” cannot
  all be counted as museums. Gallery queries similarly keep gallery evidence.
- Route the category definition “What is a museum?” to `Museum`, preserving
  explicit “Tell me about A Museum” as a request for that proper-name article.

The short-category shortcut is limited to the current/next conversation turn
after a place search. A reset or an intervening unrelated turn cannot silently
reuse old geography. Successful empty results retain the area; tool errors,
missing centers, nonfinite/out-of-range coordinates, and invalid radii cannot
seed a continuation. The stored area comes from the actual tool result,
never from the first POI row.

## Adversarial review and verification

Eight regression tests cover changed GPS, a distracting first-result location,
explicit city overrides, named-center labels, empty results,
stale/reset context, invalid metadata, and encyclopedia/venue/ordinal wording.
The first run exposed an overly broad wrapper that captured “What is a
museum?”; that definition question now stays on the Wikipedia route. The
phone replay then exposed a second definition error: the router included “a”
in the title and opened the artist `A Museum`. The replay's unexpectedly
dense museum results also prompted inspection of the shared tourism filter.
A real-service fixture proved that it counted fountains, plaques, galleries,
and a museum cafe as museums. All four resulting assertions failed before
the subtype/title repairs and pass afterward.

Full MCPZimKit suite: **596 tests passed**. The existing ZIM-only
factual-answer boundary remains covered by the full suite. Raw shared logs
remain outside tracked files. No audio or distribution upload is used.

## Final phone replay

The signed iOS Debug and macOS Release builds pass their signature gates.
The iPhone replay completes all **11 turns** with zero model-generation
calls. All nine sets of place results remain inside their specified radius;
five category transitions preserve the exact prior search center and radius.
Explicit San Francisco and “near me” requests change the center as intended.
The definition turn dispatches `article_overview(title: "museum")`.

| Turn | Completed tool result | Returned entries | Farthest entry |
|---|---:|---:|---:|
| Coffee near me | 136 ms | 25 | 758 m |
| Museum? | **45 ms** | 14 | 4988 m |
| Parks? | 45 ms | 2 | 4754 m |
| Cafes in San Francisco | 99 ms | 25 | 426 m |
| Museum? (same San Francisco center) | 110 ms | 25 | 2714 m |
| Museums near me | 18 ms | 14 | 4988 m |
| Any good museums? | 18 ms | 14 | 4988 m |
| What about museums? (after returning to coffee) | 12 ms | 14 | 4988 m |

The original “Museum?” took 5639 ms from the user log entry to the completed
tool result and chose an unrequested city. These are single-run observations,
not a statistical inference benchmark; startup and display refresh are not
included. Distances and classification use archive records. This verifies
query-area continuity and filtering, not the real-world accuracy of every
archive coordinate; duplicate source entries can still appear in the list.
Local result samples include Palo Alto Museum and Museum of American Heritage.

[Machine-readable results](benchmarks/place-continuity-2026-09-05.json)
omit the user's precise location. Installed `MCPZimChat.debug.dylib` UUID:
`E301BC00-F031-32C8-AAD0-2B19391BCDFF`; SHA-256:
`c6d74843cafedc314f99bfe0d150628aa9217557f3adfa1bca532e03fae34b8a`.
The phone was restored to a normal launch with the autorun environment
removed and passed the 45-second health watch with no new crash report.
