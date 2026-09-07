# Missing map / empty nearby results

The supplied screenshot says “No coffee shops found near you (within 5 km)”
and then promises a map below, but no map appears. The reply synthesizer
unconditionally appended that promise, including for zero results. The map
render gate correctly requires at least one geocoded place.

Corrections:

- Empty search captions identify the loaded offline maps as their scope and
  suggest a wider area or different category, without promising nonexistent pins.
- A search center outside the selected archive's advertised bounding box
  raises an explicit coverage error instead of returning successful zero results.
- An advertised category index that cannot be read raises an index error.
  Previously the service silently skipped it and could report zero or partial
  counts as if the requested search had succeeded.
- Both errors produce a deterministic user-facing answer and invalidate the
  previous nearby-search area, avoiding an unrelated location on a follow-up.
- Error payloads passed directly to the caption synthesizer retain their error
  instead of being converted into a map promise.

Adversarial self-review checked genuine empty results inside coverage, missing
coverage, missing advertised chip files, stale search context, and the renderer.
The trace classification cache was inspected: traces are immutable, so a cached
unfinished result is not supported as an explanation here. Existing nonempty
nearby-place regressions remain passing. Full Swift suite: 607 tests passed.

The latest shared phone log available during this investigation was uploaded at
00:59 UTC September 6 and contains earlier voice activity, not the screenshot's
18:04 coffee search. Thus neither missing coverage nor an unreadable index is
confirmed as the cause of this particular zero-result search. Its session must
become a completed session (relaunch, then background the app) before exact
query coordinates, selected archive,
index path, and returned count can be correlated. Do not claim this repair has
restored missing coffee-shop records without that evidence.

These changes are local pending further diagnosis/distribution; the previously
uploaded TestFlight build 20260905190339 does not contain them.
