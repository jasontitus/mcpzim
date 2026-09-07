# Carmel Valley remote-session diagnosis

Read the newest shared Firebase Storage session available September 6:
`5393c74a/2026-09-05_17-59-00.log`, uploaded September 6 at 18:13:51 UTC.
It contains the September 5 Carmel Valley article conversation and the 18:04
coffee search from the user's screenshot. No later conversation session was
present in the bucket at inspection time.

## Confirmed failures

1. “Tell me about Carmel valley” resolved to **Carmel Valley AVA**. The history
   follow-up stayed on that viticultural-area article. All returned factual
   prose was extracted from that ZIM article; this was wrong-entity selection,
   not unsupported generated prose. The overview adapter contains a policy
   that silently chooses the first substantial link from disambiguation pages.
   That is a plausible mechanism, but the log does not record the original
   resolved page before the substitution, so the exact resolution step is not
   independently established by this session.
2. Coffee searched the reported phone location, with the California archive,
   the `chip-cafes` index, and a hardcoded 5 km radius. It returned zero rows in
   75 ms. There was no invented distant search center in this recorded request.
3. The zero-result caption incorrectly promised a map. The map gate requires
   geocoded results, so no map rendered. The caption repair is already local
   from the preceding investigation but is not in build 20260905190339.

## Verification against the actual StreetZIM

Read `category-index/chip-cafes.json` from the local copy of the same archive,
`osm-california-2026-05-29.zim`. Independently calculated great-circle distances
from the session's reported coordinates across all 27,899 café-index records.
There are **zero records within 5 km**. The nearest records are:

| Archive record | Straight-line distance |
| --- | ---: |
| Carmel Valley Creamery Co. | 7.42 km |
| Wild Goose Bakery Cafe | 7.84 km |
| Fro N Joe | 8.06 km |
| Corkscrew Cafe | 8.36 km |

These are archive records, not current opening-hours, quality, or route-distance
recommendations. This verifies the zero-result count against the reported
location; it does not independently validate the phone's GPS accuracy.
Raw local extraction: `/private/tmp/zimfo-carmel-cafes.json`.

## Implications

The previous coverage/unreadable-index hypotheses are not supported as the
cause of this screenshot. The app needs a useful bounded wider-search option
after an empty default-radius query, with the widened radius clearly stated
and the same center retained. An explicit user radius must remain authoritative.
Increasing the radius to 10 km would expose four records here without searching
another city or generating recommendations from model memory.

Ambiguous place articles need a visible selection or evidence-based local
disambiguation rather than silently adopting the first substantial article.
Neither wider-search behavior nor article disambiguation has been changed by
this diagnosis alone. The user's latest on-device session may still be active
and therefore not yet uploaded.

## Subsequent repair and release validation

Implemented after the user requested fixes and a TestFlight release:

- An empty nearby search offers “search wider” and names the next radius.
  The follow-up doubles the radius, capped at 100 km, while preserving the
  resolved center, category, and archive. It never expands an explicit radius
  until the user requests widening. Stale or failed searches cannot seed it.
- Disambiguation pages return up to six readable, matching ZIM article choices
  rather than silently adopting the first long article. Unrelated outbound
  links are excluded. Missing readable choices request a more specific name.
- Both direct routing and model-dispatched overview calls stop for that same
  clarification. Old article discussion state is cleared, and selection cards
  retain the source archive and article path. No model prose chooses a winner.
- The earlier false-map-caption, coverage-error, and unreadable-index repairs
  are included. Empty-result replies avoid unrelated topic offers.

Adversarial self-review covered wrong-entity selection despite grounded prose,
unrelated disambiguation links, no-readable-choice behavior, radius limits,
stale/error context, and both app dispatch paths. Regression fixtures include
the four actual Carmel Valley cafe records: zero at 5 km, four at 10 km.
The full Swift suite passed 613 tests, and the signed iOS Debug build compiled.
TestFlight upload evidence is recorded separately in the release log.

## TestFlight release

Zimfo **1.0 (20260906183438)** was uploaded on 2026-09-06 using
`ios/scripts/testflight-upload.sh`. The exact distribution IPA passed
`verify-app-signature.sh` for `com.tiltastech.zimfo`, team `A6G8H8NGAM`.
Xcode reported `Upload succeeded` and `EXPORT SUCCEEDED`; the script reported
`upload submitted`, then confirmed `IN_BETA_TESTING` and verified assignment
to `InternalTesters`. The upload command exited successfully.

Local evidence: `/private/tmp/zimfo-carmel-testflight.log` and
`/private/tmp/zimfo-carmel-signature.log`. The retained archive is
`ios/build-testflight/1.0-20260906183438/Zimfo.xcarchive`.
This release was compile- and regression-tested on the Mac; remote phone
behavior still needs confirmation with this TestFlight build.
