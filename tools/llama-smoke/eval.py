"""Chat-scenario harness for llama.cpp.

Replays EvalHarness.swift's bars_sc_caltrain_chain scenario (and any
future 2-3 turn chat scenarios) against a GGUF model via llama-cpp-python,
dispatches tool calls against a minimal stub fixture, and scores:

- toolsCalledAny / toolsNotCalled (subset match on parsed tool calls)
- responseIncludesAny (case-insensitive substring on final assistant text)
- Peak RSS + samples ≥5/6/7 GB

The goal is apples-to-apples comparison with the Swift EvalHarness on
Mac. The scenarios here must match ios/MCPZimEval/EvalHarness.swift
byte-for-byte on the user-turn text, and the fixture payload size
matches what StubZimService produces (25 bars).

Usage:
  .venv/bin/python eval.py --repo bartowski/google_gemma-3-4b-it-GGUF \\
    --file google_gemma-3-4b-it-Q4_K_M.gguf \\
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn
"""

import argparse
import ast
import json
import os
import re
import resource
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import psutil
from huggingface_hub import hf_hub_download


# LFM2.5's baked Jinja chat template uses HF Transformers-only
# `{% generation %}…{% endgeneration %}` marker tags. Plain Jinja2
# (bundled with llama-cpp-python) rejects them with a SyntaxError at
# Llama() construction time, even when we'll never call the chat API.
# Register a no-op extension so the template parses; we use a hand-
# rolled prompt builder for LFM2 anyway.
def _install_generation_noop_extension():
    from jinja2 import nodes
    from jinja2.environment import Environment
    from jinja2.ext import Extension

    class _GenerationNoop(Extension):
        tags = {"generation"}
        def parse(self, parser):
            next(parser.stream)  # consume the 'generation' tag token
            # Return the inner body as-is so any rendered output is
            # preserved. We don't use the chat path for LFM2 anyway,
            # but a sane fallback keeps llama-cpp-python happy at
            # construction time.
            return parser.parse_statements(
                ("name:endgeneration",), drop_needle=True
            )

    orig_init = Environment.__init__
    if getattr(orig_init, "_lfm_patched", False):
        return
    def patched(self, *args, **kwargs):
        extensions = list(kwargs.get("extensions", ()))
        extensions.append(_GenerationNoop)
        kwargs["extensions"] = extensions
        return orig_init(self, *args, **kwargs)
    patched._lfm_patched = True
    Environment.__init__ = patched

_install_generation_noop_extension()


# ------------------------------------------------------------------
# Fixture: the same 25 synthetic bars EvalHarness.swift's
# addBarsInSanCarlosFixture produces. Keep the list in sync.
# ------------------------------------------------------------------
BAR_NAMES = [
    "Highlands Sports Bar & Grill", "The Office Bar",
    "Orchid Room", "Sneakers American Grill",
    "Devil's Canyon Brewing", "Broadway Pub",
    "The Laurel Room", "Nini's Coffee",
    "Town Irish Pub", "Scratch Sports Bar",
    "Refuge Belgian", "Little Sky Bakery",
    "Cuban Kitchen", "Bianchini's Market",
    "Taverna", "Telefèric Barcelona",
    "Palmetto Superfoods", "Boichik Bagels",
    "Cyclismo Cafe", "Milagros",
    "Village Pub", "The Crystal Lounge",
    "Old Pro", "Dutch Goose",
    "Rose & Crown"
]


def bars_heavy_fixture(n: int = 500) -> dict[str, Any]:
    """Mirror the "bars in north beach" phone scenario — OSM returned
    ~1200 bars. Tool response structure matches MCPToolAdapter's
    `near_named_place` encoder. Scaled to `n` synthetic entries so we
    can stress-test llama.cpp with realistic payload sizes (25 is the
    default `limit`; 500-1000 emulates what hit the phone). Each row
    is ~240 bytes → 500 rows ≈ 120 KB JSON ≈ 30-50k tokens depending
    on tokenizer."""
    bars = []
    for i in range(n):
        lat = 37.8050 + (i % 20) * 0.0008 - 0.008
        lon = -122.4100 + (i // 20) * 0.0012 - 0.02
        name_pool = BAR_NAMES + [
            f"Bar {i}", f"Pub {i}", f"Tavern {i}", f"Lounge {i}",
            f"Cocktail Room {i}", f"Wine Bar {i}",
        ]
        bars.append({
            "name": name_pool[i % len(name_pool)],
            "type": "poi", "subtype": "bar",
            "location": "North Beach",
            "lat": lat, "lon": lon,
            "distance_m": 200 + i * 3,
        })
    return {
        "radius_km": 5,
        "total_in_radius": 1244,
        "by_category": [{"category": "bar", "count": 1244}],
        "results_shown": len(bars),
        "results": bars,
        "usage_note":
            "`results` holds the top-N nearest places across ALL categories. "
            "A bucket in `by_category` may have members NOT present in "
            "`results`. NEVER invent names for buckets whose members are "
            "not in `results`.",
        "query": "north beach",
        "resolved": {
            "name": "North Beach", "type": "place", "subtype": "neighbourhood",
            "location": "San Francisco, California",
            "lat": 37.8050, "lon": -122.4100,
        },
    }


def bars_sc_fixture() -> dict[str, Any]:
    """Mirror the Swift StubZimService output for the bars near San Carlos
    call. Coordinates spread the same way as addBarsInSanCarlosFixture."""
    bars = []
    for i, name in enumerate(BAR_NAMES):
        lat_off = (i % 5) * 0.004 - 0.008
        lon_off = (i // 5) * 0.006 - 0.012
        bars.append({
            "name": name, "type": "poi",
            "subtype": "bar", "location": "San Carlos",
            "lat": 37.5124 + lat_off, "lon": -122.2606 + lon_off,
            "distance_m": 200 + i * 150,
        })
    return {
        "radius_km": 5,
        "total_in_radius": len(bars),
        "by_category": [{"category": "bar", "count": len(bars)}],
        "results_shown": len(bars),
        "results": bars,
        "usage_note":
            "`results` holds the top-N nearest places across ALL categories. "
            "A bucket in `by_category` may have members NOT present in "
            "`results`. To list items in a specific bucket, call this "
            "tool again with `kinds=[\"<category>\"]`. NEVER invent "
            "names for buckets whose members are not in `results`.",
        "query": "san carlos",
        "resolved": {
            "name": "San Carlos", "type": "place", "subtype": "city",
            "location": "California, USA",
            "lat": 37.5124, "lon": -122.2606,
        },
    }


# ------------------------------------------------------------------
# Scenarios. Match EvalHarness.swift user-turn text verbatim.
# ------------------------------------------------------------------
_PA_STORIES_RESPONSE = {
    "radius_km": 3, "total_in_radius": 3,
    "by_category": [{"category": "tourism", "count": 2},
                     {"category": "historic", "count": 1}],
    "results_shown": 3,
    "results": [
        {"name": "Hewlett-Packard Garage", "type": "poi",
         "subtype": "historic", "location": "Palo Alto",
         "lat": 37.4424, "lon": -122.1491, "distance_m": 120,
         "wiki": "en:HP Garage",
         "excerpt":
             "The HP Garage is a private museum at 367 Addison Avenue "
             "in Palo Alto — widely regarded as the birthplace of "
             "Silicon Valley."},
        {"name": "Stanford University", "type": "poi",
         "subtype": "tourism", "location": "Palo Alto",
         "lat": 37.4275, "lon": -122.1697, "distance_m": 1800,
         "wiki": "en:Stanford University"},
        {"name": "Palo Alto Train Station", "type": "poi",
         "subtype": "tourism", "location": "Palo Alto",
         "lat": 37.4433, "lon": -122.1649, "distance_m": 920,
         "wiki": "en:Palo Alto station"},
    ],
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "bars_nb_ferry_heavy": {
        "system_location": {"lat": 37.8050, "lon": -122.4100},
        "turns": [
            {
                "user": "Bars in North Beach",
                "tools_called_any": ["near_places", "near_named_place"],
                "response_includes_any": ["bar", "north beach", "found"],
            },
            {
                "user": "Which is closest to the Ferry Building?",
                "response_includes_any":
                    ["ferry", "building", "close", "near", "bar"],
            },
        ],
        # Override tool fixture to return a HEAVY payload (500 bars).
        # This is the "bars in north beach → which is closest to ferry"
        # repro the user saw on-device that instantly crashed llama.cpp
        # inside llama_decode mid-prefill.
        "heavy_bars": 500,
    },
    "bars_sc_caltrain_chain": {
        "system_location": {"lat": 37.5124, "lon": -122.2606},
        "turns": [
            {
                "user": "Bars in San Carlos",
                "tools_called_any": ["near_places", "near_named_place"],
                "response_includes_any": ["bar", "san carlos", "tavern", "found"],
            },
            {
                "user": "Which is closest to Caltrain?",
                "response_includes_any": ["caltrain", "station", "close",
                                           "near", "bar"],
            },
        ],
    },
    "restaurants_in_sf": {
        "turns": [{
            "user": "Are there any good restaurants in San Francisco?",
            "tools_called_any": ["near_places", "near_named_place"],
            "response_includes_any":
                ["souvla", "zuni", "nopa", "restaurant"],
        }],
    },
    "nearby_stories_palo_alto": {
        "turns": [{
            "user": "Tell me some interesting stories from Palo Alto.",
            "tools_called_any": ["nearby_stories", "nearby_stories_at_place"],
            "response_includes_any":
                ["hewlett", "packard", "garage", "stanford", "palo alto"],
        }],
    },
    "tell_me_about_palo_alto": {
        "turns": [{
            "user": "Tell me about Palo Alto.",
            "tools_called_any": ["article_overview"],
            "tools_not_called": ["search", "get_article"],
            "response_includes_any":
                ["palo alto", "silicon valley", "stanford", "santa clara"],
        }],
    },
    "compare_musk_bezos": {
        "turns": [{
            "user": "How is Elon Musk different from Jeff Bezos?",
            "tools_called_any": ["compare_articles"],
            "response_includes_any":
                ["musk", "bezos", "tesla", "amazon", "spacex", "blue origin"],
        }],
    },
    "relations_us_iran": {
        "turns": [{
            "user": "How have the United States and Iran gotten along historically?",
            "tools_called_any": ["compare_articles", "article_relationship"],
            "response_includes_any":
                ["iran", "united states", "relations", "1979",
                 "revolution", "sanctions"],
        }],
    },
    "narrate_hp_garage": {
        "turns": [{
            "user": "Please read me the full article about the HP Garage.",
            "tools_called_any": ["narrate_article"],
            "response_includes_any":
                ["hewlett", "packard", "addison avenue",
                 "birthplace of silicon valley"],
        }],
    },
    "what_is_here_in_sf": {
        "system_location": {"lat": 37.7793, "lon": -122.4193},
        "turns": [{
            "user": "Where am I?",
            "tools_called_any": ["what_is_here"],
            "response_includes_any":
                ["san francisco", "civic center", "california"],
        }],
    },
    "grav_waves_chain": {
        "turns": [
            {
                "user": "What are gravitational waves?",
                "tools_called_any": ["article_overview", "search"],
                "response_includes_any":
                    ["general relativity", "einstein", "spacetime",
                     "massive", "accelerating"],
            },
            {
                "user": "How were they first detected?",
                "tools_called_any": ["article_overview", "search",
                                     "get_article_section"],
                "response_includes_any":
                    ["ligo", "2015", "binary", "black hole", "merger"],
            },
            {
                "user": "Who got the Nobel Prize for it?",
                "response_includes_any":
                    ["weiss", "thorne", "barish", "2017", "nobel"],
                "tools_not_called": ["article_overview", "search"],
            },
        ],
    },
    "wwi_vs_wwii_chain": {
        "turns": [
            {
                "user": "Compare World War I and World War II — causes and scale.",
                "tools_called_any": ["compare_articles", "article_overview"],
                "response_includes_any":
                    ["1914", "1939", "alliance", "axis", "trench"],
            },
            {
                "user": "How many total casualties each?",
                "tools_called_any": ["get_article_section", "article_overview",
                                     "search"],
                "response_includes_any":
                    ["million", "casualt", "civilian", "deaths"],
            },
            {
                "user": "What changed between the two that made WWII so much more deadly?",
                "response_includes_any":
                    ["industrial", "air", "civilian", "bomb",
                     "strategic", "total"],
                "tools_not_called": ["compare_articles", "article_overview"],
            },
        ],
    },
    "french_revolution_chain": {
        "turns": [
            {
                "user": "How did the French Revolution unfold?",
                "tools_called_any": ["article_overview", "search"],
                "response_includes_any":
                    ["1789", "bastille", "louis", "estates"],
            },
            {
                "user": "What role did Robespierre play?",
                "tools_called_any": ["article_overview",
                                     "get_article_section", "search"],
                "response_includes_any":
                    ["robespierre", "terror", "committee", "executed"],
            },
            {
                "user": "So what ended it?",
                "response_includes_any":
                    ["thermidor", "coup", "napoleon", "directory"],
                "tools_not_called": ["article_overview", "search"],
            },
        ],
    },
    "crispr_chain": {
        "turns": [
            {
                "user": "Explain how CRISPR-Cas9 works.",
                "tools_called_any": ["article_overview", "search"],
                "response_includes_any":
                    ["guide rna", "cas9", "cut", "dna"],
            },
            {
                "user": "What does the guide RNA actually bind to?",
                "tools_called_any": ["get_article_section",
                                     "article_overview", "search"],
                "response_includes_any":
                    ["complementary", "base-pair", "pam", "spacer"],
            },
            {
                "user": "So why do off-target effects happen?",
                "response_includes_any":
                    ["similar", "mismatch", "partial", "imperfect",
                     "binding"],
                "tools_not_called": ["article_overview", "search"],
            },
        ],
    },
    # Multi-turn extended-context chain. Mirrors
    # EvalHarness.swift::sky_is_blue_chain — opener / expand / clarify.
    # Third turn is the sharpest signal: the model must answer WITHOUT
    # calling a tool, reasoning from the prior turns' article sections.
    # Fixture dispatches each article_overview → a canned Rayleigh /
    # sunset wavelength excerpt so content is the same across runs.
    "sky_is_blue_chain": {
        "turns": [
            {
                "user": "Why is the sky blue?",
                "tools_called_any": ["article_overview", "search"],
                "response_includes_any": ["rayleigh", "scatter",
                                           "wavelength", "shorter"],
            },
            {
                "user": "So why are sunsets red then?",
                "tools_called_any": ["article_overview", "search"],
                "response_includes_any": ["longer", "wavelength",
                                           "atmosphere", "path"],
            },
            {
                "user": "Wait, what controls which wavelength wins?",
                # Clarify turn — expect reasoning from prior sections,
                # not a third tool call.
                "response_includes_any": ["path", "travel", "far",
                                           "fourth", "inverse",
                                           "distance", "thickness"],
                "tools_not_called": ["article_overview", "search"],
            },
        ],
    },
}


# ------------------------------------------------------------------
# System preamble + tool declarations.
# A compact analogue of the Swift ChatSession.toolsPreamble — gives
# the model the same shape of instructions we use on-device.
# Kept intentionally tight (~1.5 kB) so the difference with the Swift
# eval's 7729-char preamble is explicit: if llama.cpp performs well
# here, we may be over-feeding the Swift harness.
# ------------------------------------------------------------------
SYSTEM_PREAMBLE = (
    "You are a helpful assistant running on a phone in offline mode.\n"
    "You have access to an offline Wikipedia + OpenStreetMap index via "
    "the tools listed below. Prefer tools over guessing. When the user "
    "asks about places, call `near_places` or `near_named_place` and "
    "summarize results. Keep replies concise.\n"
    "\n"
    "When the user's location is known, it is provided as "
    "`currentLocation` in the tool `origin` field.\n"
)


def _tool(name: str, desc: str, **props: dict) -> dict:
    required = [k for k, v in props.items() if v.pop("required", False)]
    return {
        "type": "function",
        "function": {
            "name": name, "description": desc,
            "parameters": {
                "type": "object", "properties": props, "required": required,
            },
        },
    }


_ARR_STR = {"type": "array", "items": {"type": "string"}}

TOOLS_SCHEMA_EXTRA: list[dict[str, Any]] = [
    _tool("nearby_stories",
          "Find interesting articles/places around a named location or "
          "lat/lon. Use when the user asks for 'stories' or 'interesting "
          "things' near somewhere.",
          place={"type": "string"}, lat={"type": "number"},
          lon={"type": "number"}, kinds=_ARR_STR),
    _tool("article_overview",
          "Fetch a Wikipedia-style article with its lead + available "
          "section titles. Use for 'tell me about X' questions.",
          title={"type": "string", "required": True}),
    _tool("get_article_section",
          "Fetch a specific named section of a Wikipedia article.",
          title={"type": "string", "required": True},
          section={"type": "string", "required": True}),
    _tool("compare_articles",
          "Compare two Wikipedia-style topics side-by-side. Use for "
          "'how is X different from Y' or 'US and Y relations' queries.",
          titles=_ARR_STR),
    _tool("narrate_article",
          "Read the full body of a specific Wikipedia article verbatim.",
          title={"type": "string", "required": True}),
    _tool("what_is_here",
          "Describe the place at the user's current lat/lon.",
          lat={"type": "number"}, lon={"type": "number"}),
    _tool("search",
          "Free-text Wikipedia search; returns ranked hits.",
          query={"type": "string", "required": True}),
]

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "near_named_place",
            "description":
                "Find points of interest near a named place. The place is "
                "a free-text location (city or landmark name) which the tool "
                "geocodes internally. Returns up to 25 results sorted by "
                "distance, plus a breakdown of category counts in the radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {"type": "string"},
                    "radius_km": {"type": "number"},
                    "kinds": {"type": "array",
                               "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
                "required": ["place"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "near_places",
            "description":
                "Find points of interest around a lat/lon coordinate OR a "
                "named place. Pass `place` for free-text, OR `lat`+`lon` for "
                "coordinates. Use this for 'what's around X' or "
                "'X near me' queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "place": {"type": "string"},
                    "radius_km": {"type": "number"},
                    "kinds": {"type": "array",
                               "items": {"type": "string"}},
                },
            },
        },
    },
] + TOOLS_SCHEMA_EXTRA


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


@dataclass
class MemoryProbe:
    samples: list[float] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)

    def start(self, interval_s: float = 0.1):
        def loop():
            proc = psutil.Process()
            while not self.stop.is_set():
                self.samples.append(proc.memory_info().rss / (1024 * 1024))
                time.sleep(interval_s)
        threading.Thread(target=loop, daemon=True).start()

    def peak(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def count_ge(self, mb: float) -> int:
        return sum(1 for s in self.samples if s >= mb)


# ------------------------------------------------------------------
# Tool-call extraction. llama-cpp-python's chat interface reports
# tool calls in `choices[0].message.tool_calls` when the model uses
# the OpenAI-compatible format. For models that emit ad-hoc
# `<tool_call>…</tool_call>` or markdown fences, fall back to a
# regex scan so we don't miss invocations.
# ------------------------------------------------------------------
TOOL_FENCE_RE = re.compile(
    r"(?:```(?:tool_call|tool|json)?\s*|<tool_call>|<\|tool_call\|>)"
    r"\s*(?P<rest>.*?)(?:```|</tool_call>|<\|/tool_call\|>|\Z)",
    re.DOTALL,
)

# LFM2.5 emits Pythonic tool calls:
#   <|tool_call_start|>[fn(arg="val", k=1), fn2(...)]<|tool_call_end|>
# Body is a Python list of Call expressions, not JSON. Parsed with `ast`.
LFM_TOOL_RE = re.compile(
    r"<\|tool_call_start\|>\s*(?P<body>.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _parse_list_of_calls(body: str) -> list[dict]:
    """Parse a `[fn(k=v), fn(k=v)]` Python expression into
    [{name, args}, ...]. Returns [] if anything fails to parse.
    Also accepts a single bare `fn(k=v)` expression."""
    out: list[dict] = []
    try:
        tree = ast.parse(body, mode="eval")
    except SyntaxError:
        return out
    nodes: list[ast.expr] = []
    if isinstance(tree.body, ast.List):
        nodes = list(tree.body.elts)
    elif isinstance(tree.body, ast.Call):
        nodes = [tree.body]
    for call in nodes:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        args: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                args[kw.arg] = ast.unparse(kw.value)
        out.append({"name": call.func.id, "args": args})
    return out


def _extract_pythonic_calls(content: str) -> list[dict]:
    """Parse LFM2.5-style tool calls. Two forms observed in the wild:

    1. `<|tool_call_start|>[fn(k=v, ...)]<|tool_call_end|>` — when the
       special tokens are surfaced.
    2. Bare `[fn(k=v, ...)]` at end of content (after `</think>`) — when
       llama.cpp consumes the special tokens internally."""
    out: list[dict] = []
    # Form 1: explicit markers.
    for m in LFM_TOOL_RE.finditer(content):
        body = m.group("body").strip()
        if body:
            out.extend(_parse_list_of_calls(body))
    if out:
        return out
    # Form 2: bare `[fn(...)]` after </think>. Strip the think block
    # first so a `[something]` inside thinking text doesn't false-fire.
    tail = content.split("</think>")[-1] if "</think>" in content else content
    tail = tail.strip()
    # Scan forward from the FIRST `[` and bracket-balance (track string
    # boundaries) to find the matching outer `]`. `rfind` would land
    # inside e.g. `kinds=['bar']`.
    start = tail.find("[")
    if start >= 0:
        depth = 0
        in_str: Optional[str] = None
        esc = False
        for i in range(start, len(tail)):
            ch = tail[i]
            if in_str is not None:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    out.extend(_parse_list_of_calls(tail[start:i + 1]))
                    break
    return out


def _balanced_json(src: str, start: int) -> Optional[tuple[dict, int]]:
    """Parse a JSON object at src[start], tracking nested braces so a
    nested `parameters: {...}` doesn't terminate the scan prematurely.
    Returns (parsed_obj, end_index) or None if parse fails."""
    if start >= len(src) or src[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(src)):
        ch = src[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = src[start:i + 1]
                try:
                    return json.loads(chunk), i + 1
                except Exception:
                    return None
    return None


def extract_tool_calls(content: str, openai_calls: Optional[list]) -> list[dict]:
    """Return list of {name, args} for every tool invocation we can find."""
    calls: list[dict] = []
    if openai_calls:
        for c in openai_calls:
            fn = c.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            calls.append({"name": fn.get("name", ""), "args": args})
        if calls:
            return calls
    if not content:
        return []
    pythonic = _extract_pythonic_calls(content)
    if pythonic:
        return pythonic
    # Scan for tool fences (markdown, <tool_call>) and brace-balance
    # the JSON body inside.
    for m in TOOL_FENCE_RE.finditer(content):
        rest = m.group("rest")
        brace = rest.find("{")
        if brace < 0:
            continue
        parsed = _balanced_json(rest, brace)
        if parsed is None:
            continue
        obj, _end = parsed
        if "function" in obj and "parameters" in obj:
            calls.append({"name": obj["function"],
                          "args": obj["parameters"]})
        elif "name" in obj:
            calls.append({"name": obj["name"],
                          "args": obj.get("arguments",
                                          obj.get("parameters", {}))})
    # Also scan for bare top-level JSON without a fence (some Gemma
    # variants emit that).
    if not calls:
        brace = content.find("{")
        while brace >= 0:
            parsed = _balanced_json(content, brace)
            if parsed is None:
                brace = content.find("{", brace + 1)
                continue
            obj, end = parsed
            if isinstance(obj, dict) and "function" in obj:
                calls.append({"name": obj["function"],
                              "args": obj.get("parameters", {})})
            brace = content.find("{", end)
    return calls


# ------------------------------------------------------------------
# Tool dispatcher — just the fixtures we need for the bars scenario.
# ------------------------------------------------------------------
ARTICLE_FIXTURES: dict[str, dict[str, Any]] = {
    "rayleigh scattering": {
        "title": "Rayleigh scattering",
        "lead":
            "Rayleigh scattering describes how light interacts with "
            "particles smaller than its wavelength. Blue light has a "
            "shorter wavelength and is scattered more strongly, which "
            "is why the daytime sky appears blue. Intensity scales "
            "with 1/λ^4 — the inverse fourth-power law.",
        "available_sections": [
            "Discovery", "Mechanism", "Atmospheric optics", "Sunsets"
        ],
    },
    "diffuse sky radiation": {
        "title": "Diffuse sky radiation",
        "lead":
            "At sunrise and sunset the Sun's light travels a longer "
            "path through the atmosphere, scattering short blue "
            "wavelengths out of the line of sight so the remaining "
            "light is dominated by longer red and orange wavelengths.",
        "available_sections": ["Causes", "Observations"],
    },
    "sunset": {
        "title": "Sunset",
        "lead":
            "Sunsets appear red because the Sun's rays traverse more "
            "atmosphere at low angles, preferentially scattering out "
            "short wavelengths (blue). The remaining light, dominated "
            "by longer wavelengths, gives sunsets their red/orange hue.",
        "available_sections": ["Physics", "Duration", "Colors"],
    },
    "mie scattering": {
        "title": "Mie scattering",
        "lead":
            "Mie scattering applies when particles are comparable to "
            "or larger than the wavelength of light — e.g. dust and "
            "aerosols. It scatters longer wavelengths more uniformly, "
            "contributing to the reddening of sunsets on hazy days.",
        "available_sections": ["Derivation", "Applications"],
    },
}


def dispatch_tool(name: str, args: dict) -> dict:
    if name in ("article_overview",):
        title = (args.get("title") or "").lower().strip()
        # Loose matching — model might ask for "Rayleigh scattering",
        # "rayleigh_scattering", "Rayleigh's scattering" etc.
        for key, art in ARTICLE_FIXTURES.items():
            if key in title or title in key:
                return {
                    "title": art["title"],
                    "lead": art["lead"],
                    "available_sections": art["available_sections"],
                }
        return {"error": f"no fixture for article={title!r}"}
    if name in ("search",):
        query = (args.get("query") or "").lower()
        hits = []
        for key, art in ARTICLE_FIXTURES.items():
            if any(tok in key for tok in query.split()):
                hits.append({"title": art["title"], "path": key.replace(" ", "_")})
        return {"query": query, "hits": hits}
    if name in ("nearby_stories", "nearby_stories_at_place"):
        place = (args.get("place") or "").lower()
        if "palo alto" in place or not place:
            return _PA_STORIES_RESPONSE
        return _PA_STORIES_RESPONSE  # fallback — keywords still match
    if name in ("compare_articles", "article_relationship"):
        titles = args.get("titles") or [args.get("a"), args.get("b")]
        a, b = (titles + [None, None])[:2]
        a_lc = (a or "").lower()
        b_lc = (b or "").lower()
        if "musk" in a_lc or "musk" in b_lc:
            return {
                "a": {"title": "Elon Musk",
                       "lead": "Elon Musk founded SpaceX and runs Tesla."},
                "b": {"title": "Jeff Bezos",
                       "lead": "Jeff Bezos founded Amazon and Blue Origin."},
                "comparison":
                    "Musk leads Tesla electric vehicles and SpaceX "
                    "rocketry; Bezos built Amazon and the Blue Origin "
                    "space company.",
            }
        if "iran" in a_lc or "iran" in b_lc:
            return {
                "a": {"title": "United States"}, "b": {"title": "Iran"},
                "comparison":
                    "Relations between the United States and Iran "
                    "soured after the 1979 Iranian Revolution. "
                    "Decades of sanctions followed.",
            }
        if "world war" in a_lc or "world war" in b_lc:
            return {
                "comparison":
                    "World War I began in 1914 with trench warfare "
                    "among European alliance systems; WWII began in "
                    "1939 with Axis aggression and was waged across "
                    "industrial air, naval, and civilian targets.",
            }
        return {"comparison": f"No comparison fixture for {a!r} vs {b!r}."}
    if name == "narrate_article":
        title = (args.get("title") or "").lower()
        if "hp garage" in title or "hewlett" in title or "packard" in title:
            return {
                "article_body":
                    "The HP Garage is the private garage at 367 "
                    "Addison Avenue in Palo Alto, California where "
                    "Bill Hewlett and Dave Packard worked in 1939. "
                    "It's widely considered the birthplace of "
                    "Silicon Valley.",
            }
        return {"article_body": f"(no narration fixture for {title!r})"}
    if name == "what_is_here":
        lat = args.get("lat") or 0
        lon = args.get("lon") or 0
        if abs(lat - 37.7793) < 0.01 and abs(lon - (-122.4193)) < 0.01:
            return {
                "place":
                    "San Francisco City Hall, in the Civic Center "
                    "neighborhood of San Francisco, California.",
            }
        return {"place": f"Unknown location at ({lat}, {lon})"}
    if name == "get_article_section":
        # Canned section data — strings contain enough keywords from
        # each chain's scenario to let pass/fail land based on model
        # reasoning, not on fixture gap.
        title = (args.get("title") or "").lower()
        if "gravitational" in title or "ligo" in title:
            return {"section_body":
                "LIGO detected the first gravitational wave on "
                "September 14, 2015, from a binary black hole "
                "merger. The 2017 Nobel Prize in Physics went to "
                "Rainer Weiss, Kip Thorne, and Barry Barish."}
        if "wwi" in title or "world war" in title:
            return {"section_body":
                "Total WWI casualties are estimated around 17 "
                "million deaths, civilian and military. WWII is "
                "estimated at over 70 million deaths, with heavy "
                "civilian losses from strategic bombing."}
        if "french revolution" in title or "robespierre" in title:
            return {"section_body":
                "The Reign of Terror (1793-94) was led by the "
                "Committee of Public Safety under Maximilien "
                "Robespierre, who was himself executed at the "
                "Thermidorian Reaction. Napoleon's rise ended the "
                "Directory era."}
        if "crispr" in title or "cas9" in title:
            return {"section_body":
                "CRISPR-Cas9 uses a guide RNA that base-pairs with "
                "a target DNA sequence adjacent to a PAM motif. The "
                "guide's 20-nt spacer must be complementary. "
                "Off-target effects occur when partial or imperfect "
                "complementary binding happens at similar sites."}
        if "rayleigh" in title or "scatter" in title or "sunset" in title:
            return {"section_body":
                "Rayleigh scattering intensity scales as 1/λ^4; "
                "shorter wavelengths scatter more, which wins for "
                "blue during the day. At sunset the light path "
                "through the atmosphere is longer, so short "
                "wavelengths are scattered out before reaching the "
                "observer."}
        return {"section_body": f"(no section fixture for {title!r})"}
    if name in ("near_places", "near_named_place"):
        place = (args.get("place") or "").lower()
        # Heavy-payload variant — the "north beach" phone repro.
        # Returns 500 synthetic bars to stress llama.cpp with a big
        # tool-response JSON in the multi-turn prompt.
        if "north beach" in place:
            return bars_heavy_fixture(500)
        if "ferry" in place or "embarcadero" in place:
            # Turn 2 follow-up: "closest to ferry building". Reuse the
            # big fixture but rerank by distance to the Ferry Building
            # (37.7956, -122.3933) so the model has real numbers.
            fx = bars_heavy_fixture(500)
            ferry_lat, ferry_lon = 37.7956, -122.3933
            def dist_m(p):
                dy = (p["lat"] - ferry_lat) * 111_000
                dx = (p["lon"] - ferry_lon) * 88_000
                return (dx * dx + dy * dy) ** 0.5
            reranked = sorted(fx["results"], key=dist_m)
            for r in reranked:
                r["distance_m"] = int(dist_m(r))
            fx["results"] = reranked[:25]
            fx["results_shown"] = len(fx["results"])
            fx["resolved"] = {
                "name": "Ferry Building", "type": "poi",
                "subtype": "landmark", "location": "San Francisco",
                "lat": ferry_lat, "lon": ferry_lon,
            }
            fx["query"] = "ferry building"
            return fx
        kinds = args.get("kinds") or []
        # "Bars in San Carlos" / follow-ups that still scope to SC
        if "san carlos" in place or place == "sc":
            return bars_sc_fixture()
        # "Bars near Caltrain" / follow-ups the model phrases by rerouting
        # to the transit station. Return the SAME bar list but re-sort by
        # distance to San Carlos Caltrain station (~37.5073, -122.2601)
        # so the model can actually answer "which is closest" using real
        # numeric data. Swift EvalHarness mocks this in-process too.
        if "caltrain" in place:
            fx = bars_sc_fixture()
            caltrain_lat, caltrain_lon = 37.5073, -122.2601
            # Crude equirectangular distance in meters — fine at this
            # latitude since we're comparing nearby points.
            def dist_m(p):
                dy = (p["lat"] - caltrain_lat) * 111_000
                dx = (p["lon"] - caltrain_lon) * 88_000
                return (dx * dx + dy * dy) ** 0.5
            reranked = sorted(fx["results"], key=dist_m)
            for r in reranked:
                r["distance_m"] = int(dist_m(r))
            fx["results"] = reranked[:10]
            fx["results_shown"] = len(fx["results"])
            fx["resolved"] = {
                "name": "San Carlos Caltrain Station", "type": "poi",
                "subtype": "railway_station", "location": "San Carlos, CA",
                "lat": caltrain_lat, "lon": caltrain_lon,
            }
            fx["query"] = "caltrain"
            return fx
        # Lat/lon fallback — if the model passed only coords, still return
        # the SC bar fixture so the turn has *something* to work with.
        if args.get("lat") and args.get("lon"):
            return bars_sc_fixture()
        return {"error": f"no fixture for place={place!r}"}
    return {"error": f"unknown tool {name!r}"}


def _build_tool_block(tool_format: str = "json") -> str:
    """Render TOOLS_SCHEMA into a compact text block the model can
    consume inside the first user turn. Gemma 3's chat template
    rejects `system` and `tool` roles, so we fold both into user
    messages and instruct the model to emit tool calls in its
    native format.

    `tool_format`:
      - "json": fenced code block `{"function": ..., "parameters": ...}` —
        Gemma 3, Qwen, generic models.
      - "pythonic": `<|tool_call_start|>[fn(k=v, ...)]<|tool_call_end|>` —
        LFM2 / LFM2.5 native, trained format."""
    lines = ["You have these tools (call one when needed):"]
    for t in TOOLS_SCHEMA:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        arg_strs = []
        for k, v in params.items():
            star = "*" if k in required else ""
            arg_strs.append(f"{k}{star}:{v.get('type','string')}")
        lines.append(f"- {fn['name']}({', '.join(arg_strs)}) — {fn['description']}")
    lines.append("")
    if tool_format == "pythonic":
        lines.append(
            "When a tool can answer the user, your FIRST action must be a "
            "tool call in this exact form (no surrounding prose, no fences):\n"
            "<|tool_call_start|>[tool_name(arg=value, arg2='string value')]<|tool_call_end|>\n"
            "Use single quotes for string arguments. After you get the tool "
            "output (as a user message starting with `[TOOL_RESPONSE]`), "
            "answer the user in concise natural prose."
        )
    else:
        lines.append(
            "To call a tool, respond with ONLY a code fence like:\n"
            "```tool_call\n"
            "{\"function\":\"<name>\",\"parameters\":{...}}\n"
            "```\n"
            "After you get the tool output (as a subsequent user message "
            "starting with `[TOOL_RESPONSE]`), answer the user in natural "
            "prose. Keep replies concise."
        )
    return "\n".join(lines)


def _lfm2_render(messages: list[dict[str, str]],
                 system_text: str = "",
                 tools_schema: Optional[list[dict]] = None,
                 date_str: str = "2026-05-28") -> str:
    """Hand-rolled prompt builder matching LFM2.5's native training
    format byte-exactly. llama-cpp-python's Jinja path mis-renders
    the GGUF-baked template because the `tojson` filter returns Markup,
    which forces autoescape on the concatenated `<|im_start|>` markers
    and turns them into `&lt;|im_start|&gt;`. We bypass that here.

    The format Liquid trained on:
      <|im_start|>system
      [optional system text]

      Today's date: YYYY-MM-DD

      List of tools: [{...}, ...]<|im_end|>
      <|im_start|>user
      ...<|im_end|>
      <|im_start|>assistant
    """
    sys_block = system_text.strip()
    if tools_schema:
        # LFM expects OpenAI-shaped tool dicts. Pass through as-is.
        if sys_block:
            sys_block += "\n\n"
        sys_block += f"Today's date: {date_str}\n\n"
        sys_block += "List of tools: " + json.dumps(tools_schema)
    parts: list[str] = []
    if sys_block:
        parts.append(f"<|im_start|>system\n{sys_block}<|im_end|>\n")
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def run_scenario(llm, scenario_name: str, probe: MemoryProbe,
                 max_turn_tokens: int = 2048,
                 tool_format: str = "json") -> dict[str, Any]:
    """Drive the scenario through llama.cpp's chat API, round-tripping
    tool calls via the fixture. Folds system+tool messages into user
    turns to sidestep Gemma 3's strict user/assistant alternation."""
    sc = SCENARIOS[scenario_name]
    preamble_block = SYSTEM_PREAMBLE + "\n" + _build_tool_block(tool_format)
    loc = sc.get("system_location")
    if loc:
        preamble_block += (
            f"\n\ncurrentLocation: lat={loc['lat']} lon={loc['lon']}"
        )
    messages: list[dict[str, str]] = []
    per_turn: list[dict[str, Any]] = []
    use_lfm_path = tool_format == "pythonic"
    # LFM2.5 gets tools via the dedicated `tools_schema` arg of
    # `_lfm2_render`, but it still needs the SYSTEM_PREAMBLE prose
    # (tool-selection hints like "use near_places for places") so it
    # doesn't reach for `search` on every geo query.
    lfm_system = SYSTEM_PREAMBLE.rstrip()
    if loc := sc.get("system_location"):
        lfm_system += f"\n\ncurrentLocation: lat={loc['lat']} lon={loc['lon']}"
    for turn_idx, turn in enumerate(sc["turns"]):
        # Fold preamble into the first user turn for Gemma (no system role).
        # LFM2.5 supports a system role natively, so it goes there instead.
        user_text = turn["user"]
        if turn_idx == 0 and not use_lfm_path:
            user_text = preamble_block + "\n\nUser query:\n" + user_text
        messages.append({"role": "user", "content": user_text})
        final_content = ""
        tool_calls_seen: list[str] = []
        # 4 was the historical default; bump to 8 for exploration-heavy
        # models like Qwen 3.6 27B that often spend several tool calls
        # diagnosing fixture errors before settling on a final response.
        # Override via TOOL_ITER_BUDGET=N for one-off tests.
        max_iters = int(os.environ.get("TOOL_ITER_BUDGET", "8"))
        for iter_ in range(max_iters):
            t_iter = time.perf_counter()
            if use_lfm_path:
                prompt = _lfm2_render(
                    messages,
                    system_text=lfm_system,
                    tools_schema=TOOLS_SCHEMA,
                )
                resp = llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_turn_tokens,
                    temperature=0.2,
                    top_p=0.8,
                    repeat_penalty=1.05,
                    stop=["<|im_end|>"],
                )
                content = resp["choices"][0].get("text") or ""
                openai_calls = None
            else:
                resp = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_turn_tokens,
                    temperature=0.3,
                )
                msg = resp["choices"][0]["message"]
                content = msg.get("content") or ""
                openai_calls = msg.get("tool_calls")
            dt = time.perf_counter() - t_iter
            tcalls = extract_tool_calls(content, openai_calls)
            per_turn_iter = {
                "turn": turn_idx, "iter": iter_,
                "t_s": round(dt, 2),
                "tool_calls": [c["name"] for c in tcalls],
                "content_preview": content[:200],
            }
            if not tcalls:
                final_content = content
                per_turn.append(per_turn_iter)
                break
            messages.append({"role": "assistant", "content": content})
            tool_outputs = []
            for c in tcalls:
                tool_calls_seen.append(c["name"])
                payload = dispatch_tool(c["name"], c["args"])
                tool_outputs.append(
                    f"[TOOL_RESPONSE name={c['name']}]\n{json.dumps(payload)}"
                )
            messages.append({
                "role": "user",
                "content": "\n\n".join(tool_outputs),
            })
            per_turn.append(per_turn_iter)
        expected_tools = turn.get("tools_called_any", [])
        tool_ok = (not expected_tools) or any(
            t in tool_calls_seen for t in expected_tools
        )
        forbidden = turn.get("tools_not_called", [])
        tool_ok = tool_ok and not any(
            t in tool_calls_seen for t in forbidden
        )
        kw_any = turn.get("response_includes_any", [])
        resp_ok = (not kw_any) or any(
            kw.lower() in final_content.lower() for kw in kw_any
        )
        per_turn.append({
            "turn": turn_idx, "final_content": final_content[:300],
            "tool_calls_seen": tool_calls_seen,
            "tools_ok": tool_ok, "response_ok": resp_ok,
        })
        # If the model just ended with assistant text, SwiftUI-style
        # chat requires an assistant message in the transcript. Gemma
        # will already have that as the last message via the llm reply.
        messages.append({"role": "assistant", "content": final_content})
    passed = all(
        t.get("tools_ok", True) and t.get("response_ok", True)
        for t in per_turn if "final_content" in t
    )
    return {
        "scenario": scenario_name,
        "passed": passed,
        "turns": per_turn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="HF repo id. Omit if using --local-path.")
    ap.add_argument("--file", default=None,
                    help="GGUF filename inside the HF repo.")
    ap.add_argument("--local-path", default=None,
                    help="Local GGUF path. When set, bypasses HF download.")
    ap.add_argument("--scenario", default="bars_sc_caltrain_chain",
                    choices=list(SCENARIOS.keys()))
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--cache-type-k", default="f16")
    ap.add_argument("--cache-type-v", default="f16")
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--swa-full", choices=["true", "false", "default"],
                    default="default",
                    help="iSWA cache mode. 'false' enables rotation-based "
                         "pruning (PR #13194/#21513); 'default' uses the "
                         "model's own setting (usually full).")
    ap.add_argument("--tool-format", choices=["json", "pythonic"],
                    default="json",
                    help="Format the tool block + parser expects. "
                         "'pythonic' = LFM2/LFM2.5 native "
                         "<|tool_call_start|>[fn(...)]<|tool_call_end|>.")
    args = ap.parse_args()

    from llama_cpp import Llama

    if args.local_path:
        gguf_path = args.local_path
        model_label = args.local_path
    else:
        if not (args.repo and args.file):
            sys.exit("--repo/--file or --local-path required")
        gguf_path = hf_hub_download(repo_id=args.repo, filename=args.file)
        model_label = f"{args.repo}/{args.file}"
    print(f"eval: scenario={args.scenario}")
    print(f"       model={model_label}")
    print(f"       cache_k={args.cache_type_k} cache_v={args.cache_type_v} "
          f"flash_attn={args.flash_attn}")

    probe = MemoryProbe()
    probe.start()

    swa_full_arg: Optional[bool] = None
    if args.swa_full == "true":
        swa_full_arg = True
    elif args.swa_full == "false":
        swa_full_arg = False
    # LFM2.5's baked Jinja chat template uses HF-only `{% generation %}`
    # tags that crash the bundled Jinja at Llama() construction. We
    # build the prompt by hand for `--tool-format pythonic` anyway,
    # so force a benign format here to dodge the parse.
    llama_kwargs: dict[str, Any] = {}
    if args.tool_format == "pythonic":
        llama_kwargs["chat_format"] = "chatml"
    t_load = time.perf_counter()
    chat_template_kwargs = {}
    chat_template_path = os.environ.get("CHAT_TEMPLATE")
    if chat_template_path:
        with open(chat_template_path) as fh:
            chat_template_kwargs["chat_template"] = fh.read()
        print(f"       overriding chat template from {chat_template_path}")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        type_k=_kv_type(args.cache_type_k),
        type_v=_kv_type(args.cache_type_v),
        flash_attn=args.flash_attn,
        swa_full=swa_full_arg,
        verbose=False,
        **llama_kwargs,
        **chat_template_kwargs,
    )
    print(f"       load: {time.perf_counter()-t_load:.2f}s · "
          f"rss: {rss_mb():.0f} MB")

    t = time.perf_counter()
    result = run_scenario(llm, args.scenario, probe,
                          tool_format=args.tool_format)
    wall_s = time.perf_counter() - t
    probe.stop.set()

    peak = max(probe.peak(), rss_mb())
    print()
    print(f"RESULT scenario={args.scenario} passed={result['passed']} "
          f"wall_s={wall_s:.1f}")
    print(f"RESULT peak_mb={peak:.0f} "
          f"ge5gb={probe.count_ge(5000)} ge6gb={probe.count_ge(6000)} "
          f"ge7gb={probe.count_ge(7000)} samples={len(probe.samples)}")
    for t in result["turns"]:
        print(f"  {t}")


def _kv_type(name: str) -> int:
    return {
        "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3,
        "q5_0": 6, "q5_1": 7, "q8_0": 8, "iq4_nl": 20,
    }[name]


if __name__ == "__main__":
    main()
