# Location questions: implementation and adversarial review

## Changes

Specific bar/pub/nightclub and coffee/cafe requests now filter shared category chips using recorded venue types. Retail alcohol, production breweries, plain bakeries, and misleading business names no longer become bars or cafes merely because they share a chip. Explicit bakery requests retain bakery results. This is deterministic retrieval from StreetZIM, without generated recommendations or quality claims.

Named search areas prefer an exact settlement name before an exact venue name. Exact area lookup continues past the general geocoder's 200-candidate stopping point, retaining exact-name candidates rather than unrelated substring matches. Partial names no longer silently establish a nearby-search center. Existing bounded shard decoding and raw-byte caching remain in use.

## Adversarial findings and verification

Self-review exposed a failure that the initial replay assertions missed: Salinas coffee resolved to Salinas Drive near Ventura. The archive also has shops and a station named Salinas. A fixture now puts 250 misleading streets and an exact-name shop ahead of the actual city, verifies the resolved city coordinates, and rejects partial street substitution. The replay assertions now reject Salinas Drive and require the National Steinbeck Center on the museum follow-up.

Category tests cover retail alcohol, brewery, coffee equipment shop, tea room, bakery, misleading names, pub-only requests, mixed bar/pub requests, coffee aliases, and legitimate recorded venues.

Actual California archive (2026-05-29) replay with local Bonsai Q1:

- Bar within 5 km: no results; widening to 10 km retains the original center and yields seven drinking venues, with The Pit Bar nearest.
- Show The Pit Bar: uses the listed result's coordinates, without Wikipedia substitution.
- Coffee in Salinas: resolves the city, yields 25 results within 5 km.
- Museum follow-up: retains Salinas; National Steinbeck Center, First Mayor's House, Monterey & Salinas Valley Railroad Museum, and Boronda Adobe History Center.

Five replay turns passed and their actual result centers were manually inspected. Named Salinas lookup took about 1.2 seconds on the Mac; other turns approximately 0.1 seconds. These are desktop measurements, not phone latency or memory guarantees.

Swift suite: 648 tests executed, one skipped, zero failures (647 runnable tests passed).

## Limits

Offline tags may omit valid businesses or be outdated; stricter categories favor accurate labels over speculative inclusion. Equal-name settlements can still require disambiguation; exact spelling alone does not establish the user's intended region. General geocoding outside named nearby-area searches retains its existing substring behavior. Missing names may require scanning multiple bounded shards and be slower than successful exact hits. This turn does not install or upload a release.
