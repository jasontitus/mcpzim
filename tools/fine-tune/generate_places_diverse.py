"""Generate near_places trajectories with strict grounding.

The single-turn generate.py was producing examples that all looked
roughly like "find 25 bars in San Carlos" — same kind, similar place
type, always-25 count, templated reply. The student LoRA-fine-tuned
on that memorised the surface phrasing and the response-count number,
which surfaced as a regression where it would emit "I found 25 bars
in San Carlos" even when the tool_response said otherwise.

This generator decouples the response from the prompt's surface form:

  1. Varied kinds: restaurants, cafes, parks, museums, hotels,
     gas_stations, pharmacies, libraries, breweries, bookstores,
     gyms, ramen shops, dive bars, etc. — not just bars.
  2. Varied locations: SF neighbourhoods, Bay Area suburbs, US
     cities, small towns, international — not just San Carlos.
  3. Varied result counts sampled from a wide bucket distribution
     (0, 1, 2-5, 6-12, 13-25, 26-50, 51-100, 100+).
  4. Grounded responses: teacher is forced to cite ≥2 names from
     tool_response in the reply, and a post-generation validator
     REJECTS outputs that don't actually do that.
  5. ~8% counterfactual zero-hit examples — teacher generates a
     plausible "no matches" reply, teaching the student that empty
     results aren't a reason to hallucinate.

Output: eval-matched preamble format (same as generate_chains.py),
written to a separate file so callers can mix it in with the rest.
"""
import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke")

from openai import AsyncOpenAI


# ----------------------------------------------------------------------
# Pools — broader than the single-shot generator's defaults so the
# (kind, place) cross-product sees a lot of distinct training points.
# ----------------------------------------------------------------------
KINDS = [
    # Food + drink
    "restaurants", "cafes", "coffee shops", "bars", "breweries",
    "wine bars", "cocktail bars", "dive bars", "sports bars",
    "tiki bars", "ramen shops", "sushi spots", "taco trucks",
    "pizza places", "Thai restaurants", "Italian restaurants",
    "Mexican restaurants", "Korean BBQ", "vegan restaurants",
    "ice cream shops", "bakeries", "donut shops", "boba shops",
    # Parks + outdoor
    "parks", "playgrounds", "dog parks", "skate parks",
    "tennis courts", "basketball courts", "pickleball courts",
    "hiking trails", "trailheads",
    # Cultural + civic
    "museums", "art galleries", "libraries", "post offices",
    "town halls", "concert halls", "movie theaters",
    "comedy clubs", "music venues", "jazz clubs",
    # Health + services
    "gyms", "yoga studios", "climbing gyms", "barber shops",
    "hair salons", "nail salons", "tattoo parlors",
    "pharmacies", "hospitals", "urgent care clinics",
    "veterinary clinics", "dentists",
    # Shopping
    "grocery stores", "farmers markets", "bookstores",
    "record stores", "vintage shops", "thrift stores",
    "comic shops", "bike shops", "running stores",
    # Travel
    "hotels", "hostels", "bed and breakfasts",
    "gas stations", "EV charging stations",
    "ATMs", "banks",
    # Worship
    "churches", "temples", "mosques", "synagogues",
]

PLACES = [
    # SF neighbourhoods
    "the Mission", "North Beach", "the Castro", "the Marina",
    "Fisherman's Wharf", "SoMa", "Hayes Valley", "the Sunset",
    "the Richmond", "Nob Hill", "Russian Hill", "Chinatown",
    "the Tenderloin", "Pacific Heights", "Cole Valley",
    "Bernal Heights", "Glen Park", "Potrero Hill",
    # Bay Area suburbs
    "San Carlos", "Palo Alto", "Mountain View", "Redwood City",
    "Menlo Park", "Los Altos", "Cupertino", "Sunnyvale",
    "Berkeley", "Oakland", "Emeryville", "Albany", "Alameda",
    "Walnut Creek", "Lafayette", "Fremont", "San Jose",
    "Santa Clara", "Campbell", "Saratoga", "Los Gatos",
    # West Coast
    "Seattle", "Portland", "Los Angeles", "Santa Monica",
    "Long Beach", "Pasadena", "San Diego", "Sacramento",
    # US elsewhere
    "New York", "Brooklyn", "Queens", "Boston", "Cambridge",
    "Chicago", "Austin", "Houston", "Dallas", "Atlanta",
    "New Orleans", "Nashville", "Asheville", "Miami",
    "Washington DC", "Philadelphia", "Pittsburgh",
    "Denver", "Boulder", "Salt Lake City",
    # Small towns / less common
    "Half Moon Bay", "Pacifica", "Carmel", "Monterey",
    "Healdsburg", "Mendocino", "Truckee", "Bend",
    "Bozeman", "Marfa", "Taos", "Burlington",
    "Provincetown", "Galena", "Door County",
    # International
    "London", "Paris", "Berlin", "Rome", "Barcelona",
    "Amsterdam", "Vienna", "Prague", "Lisbon", "Stockholm",
    "Tokyo", "Osaka", "Seoul", "Hong Kong", "Singapore",
    "Bangkok", "Mumbai", "Delhi", "Cairo", "Cape Town",
    "Buenos Aires", "Rio de Janeiro", "Mexico City",
    "Reykjavik", "Helsinki", "Copenhagen",
]


# Bucketed count distribution. Wide range from zero-hit through 100+
# so the student sees every realistic result-set size.
COUNT_BUCKETS: list[tuple[int, tuple[int, int]]] = [
    (8,  (0, 0)),       # ~8% zero-hit counterfactual
    (4,  (1, 1)),
    (18, (2, 5)),
    (22, (6, 12)),
    (22, (13, 25)),
    (14, (26, 50)),
    (8,  (51, 100)),
    (4,  (101, 200)),
]


def sample_count() -> int:
    weights = [w for w, _ in COUNT_BUCKETS]
    bucket = random.choices(COUNT_BUCKETS, weights=weights, k=1)[0]
    lo, hi = bucket[1]
    return random.randint(lo, hi) if hi > lo else lo


# ----------------------------------------------------------------------
# Teacher prompt — strict grounding contract.
# ----------------------------------------------------------------------
TEACHER_SYS = """\
Generate ONE near_places training example. JSON with three keys.

Rules (trainer rejects violations):
- tool_call.function: `near_places` or `near_named_place`.
- tool_response.results length == target_count exactly. For
  target_count=0 emit `{"results":[], "total_in_radius":0, "by_category":{}}`.
- Each result has name/type/subtype/lat/lon/distance_m. Names must
  be realistic for the kind+place combo (real or invented).
- assistant_reply: 1-3 sentences. State the exact count. Mention >=2
  specific names from results (1 if count==1; 0 if zero-hit). No
  fixed templates ("I found 25 X in Y") — phrase naturally each time.
  No emoji, no markdown.
- Zero-hit replies: plainly say none found, suggest broadening
  radius / nearby place / related kind. Don't invent results.
- Stay grounded — no hours, ratings, prices the tool didn't provide.
"""


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_call": {
            "type": "object",
            "properties": {
                "function": {"type": "string"},
                "parameters": {"type": "object"},
            },
            "required": ["function", "parameters"],
        },
        "tool_response": {"type": "object"},
        "assistant_reply": {"type": "string"},
    },
    "required": ["tool_call", "tool_response", "assistant_reply"],
}


# ----------------------------------------------------------------------
# Query phrasing variety. Don't always start with the same template.
# ----------------------------------------------------------------------
QUERY_TEMPLATES = [
    "{kind} in {place}",
    "{kind} near {place}",
    "show me {kind} in {place}",
    "any good {kind} in {place}",
    "find {kind} around {place}",
    "what {kind} are in {place}",
    "{kind} close to {place}",
    "where can I find {kind} in {place}",
    "I'm looking for {kind} in {place}",
    "best {kind} in {place}",
    "list {kind} in {place}",
    "{kind} - {place}",
]


LOC_POOL = [
    (37.8050, -122.4100), (37.5124, -122.2606), (37.7793, -122.4193),
    (37.4419, -122.1430), (37.8716, -122.2727), (37.3382, -121.8863),
    (40.7128, -74.0060),  (47.6062, -122.3321), (30.2672, -97.7431),
    (51.5074, -0.1278),   (35.6762, 139.6503),  (48.8566, 2.3522),
]


def _eval_preamble() -> str:
    try:
        from eval import SYSTEM_PREAMBLE, _build_tool_block
    except Exception:
        return (
            "You are a helpful assistant running on a phone in offline "
            "mode. You have access to an offline Wikipedia + "
            "OpenStreetMap index via the tools listed below. Prefer "
            "tools over guessing. Keep replies concise."
        )
    pre = SYSTEM_PREAMBLE + "\n" + _build_tool_block()
    if random.random() < 0.5:
        lat, lon = random.choice(LOC_POOL)
        pre += f"\n\ncurrentLocation: lat={lat} lon={lon}"
    return pre


# ----------------------------------------------------------------------
# Validator — the heart of the diversity batch. Reject ungrounded
# outputs so the JSONL only contains rule-following examples.
# ----------------------------------------------------------------------
_FAIL_LOG_PATH: Optional[Path] = None


def _log_fail(query: str, reason: str, raw: str = "", extra: Optional[dict] = None) -> None:
    if _FAIL_LOG_PATH is None:
        return
    rec = {"query": query, "reason": reason, "raw": raw[:1500]}
    if extra:
        rec.update(extra)
    try:
        with _FAIL_LOG_PATH.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def validate_grounded(query: str, target_count: int, obj: dict) -> Optional[str]:
    """Return None if grounded, else a string reason for rejection."""
    tc = obj.get("tool_call") or {}
    tr = obj.get("tool_response") or {}
    reply = (obj.get("assistant_reply") or "").strip()

    fn = tc.get("function") or ""
    if fn not in ("near_places", "near_named_place"):
        return f"wrong_function: {fn}"

    results = tr.get("results")
    if not isinstance(results, list):
        return "tool_response.results not a list"
    if len(results) != target_count:
        return f"count_mismatch: got {len(results)} need {target_count}"

    if not reply:
        return "empty_reply"
    if len(reply) < 20 or len(reply) > 600:
        return f"reply_length: {len(reply)}"

    if target_count == 0:
        # Zero-hit: reply should acknowledge no matches, NOT invent any.
        # Cheap heuristic: must mention a "no/none/empty/0/zero" word.
        low = reply.lower()
        zero_words = ["no ", "none", "didn't find", "did not find",
                      "couldn't find", "could not find", "0 ", "zero",
                      "empty", "nothing"]
        if not any(w in low for w in zero_words):
            return "zero_hit_reply_did_not_acknowledge_empty"
        return None

    # Non-zero: reply must mention ≥1 result name (≥2 if count ≥ 2).
    names = [
        (r.get("name") or "").strip().lower()
        for r in results
        if isinstance(r, dict)
    ]
    names = [n for n in names if n]
    needed = 2 if target_count >= 2 else 1
    low = reply.lower()
    hits = sum(1 for n in names if n and n in low)
    if hits < needed:
        return f"ungrounded: needed {needed} name mentions, found {hits}"

    return None


def trajectory_to_messages(query: str, obj: dict) -> dict[str, Any]:
    preamble = _eval_preamble()
    tc = json.dumps(obj["tool_call"], ensure_ascii=False)
    tr = json.dumps(obj["tool_response"], ensure_ascii=False)
    reply = obj["assistant_reply"].strip()
    return {
        "messages": [
            {"role": "user", "content":
                preamble + "\n\nUser query:\n" + query},
            {"role": "assistant", "content":
                f"```tool_call\n{tc}\n```"},
            {"role": "user", "content":
                "[TOOL_RESPONSE]\n" + tr},
            {"role": "assistant", "content": reply},
        ]
    }


# ----------------------------------------------------------------------
# Async generation.
# ----------------------------------------------------------------------
async def generate_one(
    client: AsyncOpenAI, model: str,
    kind: str, place: str, target_count: int, query: str,
    temperature: float, max_tokens: int,
) -> Optional[dict[str, Any]]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "places_trajectory", "strict": True, "schema": SCHEMA,
        },
    }
    user_msg = (
        f"kind: {kind}\nplace: {place}\ntarget_count: {target_count}\n"
        f"user_query: {query!r}\n\nProduce the JSON now."
    )
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format,
            messages=[
                {"role": "system", "content": TEACHER_SYS},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        _log_fail(query, "api_error", "", {"err": str(e)})
        return None
    raw = resp.choices[0].message.content or ""
    if not raw:
        _log_fail(query, "empty_response")
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        _log_fail(query, "json_decode", raw)
        return None
    reason = validate_grounded(query, target_count, obj)
    if reason is not None:
        _log_fail(query, reason, raw, {
            "kind": kind, "place": place, "target_count": target_count,
        })
        return None
    return obj


async def run(client: AsyncOpenAI, model: str, n: int, concurrency: int,
              temperature: float, max_tokens: int, out_path: Path) -> None:
    done = 0
    if out_path.exists():
        with out_path.open() as fh:
            done = sum(1 for _ in fh)
        print(f"resuming: {done} examples already on disk")

    seeds: list[tuple[str, str, int, str]] = []
    for _ in range(n):
        kind = random.choice(KINDS)
        place = random.choice(PLACES)
        target_count = sample_count()
        query = random.choice(QUERY_TEMPLATES).format(kind=kind, place=place)
        seeds.append((kind, place, target_count, query))
    seeds = seeds[done:]
    if not seeds:
        print("already done")
        return

    sem = asyncio.Semaphore(concurrency)
    out_fh = out_path.open("a")
    try:
        t0 = time.perf_counter()
        written = 0
        failed = 0

        async def worker(seed: tuple[str, str, int, str]) -> None:
            nonlocal written, failed
            kind, place, target_count, query = seed
            async with sem:
                obj = await generate_one(
                    client, model, kind, place, target_count, query,
                    temperature, max_tokens
                )
            if obj is None:
                failed += 1
                return
            out_fh.write(
                json.dumps(trajectory_to_messages(query, obj),
                           ensure_ascii=False) + "\n")
            out_fh.flush()
            written += 1
            if written % 10 == 0:
                dt = time.perf_counter() - t0
                rate = written / dt
                eta = (len(seeds) - written) / max(rate, 0.01)
                print(f"  {done+written}/{done+len(seeds)} · "
                      f"{rate:.2f}/s · ETA {eta:.0f}s · fails={failed}")

        await asyncio.gather(*[worker(s) for s in seeds])
    finally:
        out_fh.close()
    dt = time.perf_counter() - t0
    print(f"done: {written} written · {failed} rejected · {dt:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--api-key",  default="lm-studio")
    ap.add_argument("--model",    default="gemma-3-27b-it")
    ap.add_argument("--n",        type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--max-tokens",  type=int, default=1800,
                    help="A 100-result tool_response with realistic "
                         "names + distances can run ~1400 tokens; 1800 "
                         "covers it. Requires the LM Studio teacher to "
                         "be loaded with context >= ~3k (we use 9606).")
    ap.add_argument("--out", default="train_places_diverse.jsonl")
    ap.add_argument("--fail-log", default=None)
    ap.add_argument("--seed", type=int, default=22222)
    args = ap.parse_args()

    global _FAIL_LOG_PATH
    if args.fail_log:
        _FAIL_LOG_PATH = Path(args.fail_log)

    random.seed(args.seed)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    asyncio.run(run(client, args.model, args.n, args.concurrency,
                    args.temperature, args.max_tokens, Path(args.out)))


if __name__ == "__main__":
    main()
