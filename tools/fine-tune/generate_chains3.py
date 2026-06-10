"""Generate 3-turn chain training trajectories.

Our existing `generate_chains.py` only produces 2-turn chains
(article_overview → get_article_section → reply). But the failing
eval scenarios (`grav_waves_chain`, `wwi_vs_wwii_chain`,
`french_revolution_chain`, `crispr_chain`) all follow this 3-turn
pattern:

  Turn 1: broad question        → article_overview / compare_articles
  Turn 2: drill-down question   → get_article_section
  Turn 3: follow-up question    → NO TOOL CALL. Synthesize from prior
                                    context (see `tools_not_called`).

That third turn is the missing pattern. The model has never been
trained to RECOGNISE that a follow-up is answerable from the prior
two tool responses; under our current FT it keeps reaching for
another tool and trips `tools_not_called` on every chain.

This generator targets that gap directly. Each row produces an
11-message dialogue (system+user+(tool_call,tool_resp,reply) × 2
+ user+plain_reply) that mirrors the eval shape byte-for-byte
(SYSTEM_PREAMBLE + _build_tool_block() folded into the first user
turn, JSON-fenced tool calls — matches `train_v4_combined.jsonl`).

Supports three chain templates:

  template=article — Turn 1 article_overview, Turn 2
    get_article_section, Turn 3 conceptual follow-up.

  template=compare — Turn 1 compare_articles(A,B), Turn 2
    get_article_section(A, ...), Turn 3 cross-cutting synthesis.

  template=places  — Turn 1 near_named_place, Turn 2
    article_overview about one returned place, Turn 3 follow-up
    synthesised from the two responses.

Usage:
  .venv/bin/python generate_chains3.py \\
      --base-url http://192.168.68.104:1234/v1 --model gemma-3-27b-it \\
      --n 800 --concurrency 4 --out train_chains3.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Reuse generator infra + shared topic/people pools + tool block.
sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke")
sys.path.insert(0, str(Path(__file__).parent))

from openai import AsyncOpenAI

from eval import SYSTEM_PREAMBLE, _build_tool_block
from generate import TOPICS, PEOPLE, _log_fail


LOC_POOL = [
    (37.8050, -122.4100),  # North Beach SF
    (37.5124, -122.2606),  # San Carlos
    (37.7793, -122.4193),  # Civic Center SF
    (37.4419, -122.1430),  # Palo Alto
    (37.8716, -122.2727),  # Berkeley
    (37.3382, -121.8863),  # San Jose
    (37.7609, -122.4350),  # Marina SF
    (37.6879, -122.4702),  # Pacifica
]

PLACES_FOR_NEAR = [
    "San Francisco", "Palo Alto", "Berkeley", "San Jose",
    "Mountain View", "Sausalito", "Oakland", "Santa Cruz",
    "Adams Morgan", "Brooklyn", "Manhattan", "Cambridge",
    "Austin", "Seattle", "Portland", "Boston",
]

NEAR_KINDS = [
    ["restaurant"], ["bar"], ["cafe"], ["museum"],
    ["park"], ["bookshop"], ["bakery"], ["pub"],
    ["library"], ["theatre"], ["restaurant", "cafe"],
]


# ----------------------------------------------------------------------
# Teacher system prompts — one per template.
# ----------------------------------------------------------------------
TEACHER_SYS_ARTICLE = """\
You are a teacher generating multi-turn training data for a small
on-device assistant that calls tools. Produce ONE JSON object
modelling a 3-turn dialogue:

  Turn 1: user asks a broad question about TOPIC. Assistant calls
    article_overview(title=TOPIC). The synthetic tool_response_1
    has {title, lead, available_sections (4-8 plausible names)}.
    Assistant writes a 3-5 sentence reply_1 covering key points.

  Turn 2: user asks a more specific drill-down question. Assistant
    calls get_article_section(title=TOPIC, section=<chosen section>).
    The synthetic tool_response_2 has {section_body} (2-4 sentences).
    Assistant writes reply_2 (2-4 sentences) drawing only from
    section_body.

  Turn 3: user asks a CONCEPTUAL follow-up that is answerable from
    the combined context of reply_1 + reply_2 — DO NOT make a tool
    call here. Assistant writes reply_3 (2-4 sentences) synthesising
    the prior context. This turn must not require fresh data.

Examples of valid Turn-3 follow-ups (because they're synthesis,
not new knowledge):
  - "So what made it possible / what changed afterwards?"
  - "Why does that matter today?"
  - "How are those two ideas connected?"
  - "Who benefited / who was hurt?"

JSON keys: tool_call_1, tool_response_1, reply_1, followup_1,
tool_call_2, tool_response_2, reply_2, followup_2, reply_3.

CRITICAL — `title` field rules (used in both tool_calls and tool_response_1):
- PLAIN canonical Wikipedia title. Examples: "Industrial Revolution",
  "Immune system", "World War II", "CRISPR", "Photosynthesis".
- DO NOT decorate with prepositions, dates, or "A History of...".
- Same title in tool_call_1, tool_call_2, tool_response_1.title.

Keep invented lead/section_body factually plausible. No fabricated
dates or names that contradict common knowledge."""


TEACHER_SYS_COMPARE = """\
You are a teacher generating multi-turn training data. Produce ONE
JSON object modelling a 3-turn dialogue where the assistant compares
two topics:

  Turn 1: user asks "Compare A and B" (or "How are A and B different").
    Assistant calls compare_articles(titles=[A, B]). The synthetic
    tool_response_1 has {comparison_summary} (one paragraph contrasting
    the two). Assistant writes reply_1 (3-5 sentences).

  Turn 2: user asks a specific drill-down about ONE of the two.
    Assistant calls get_article_section(title=A_or_B, section=...).
    The synthetic tool_response_2 has {section_body}. Assistant writes
    reply_2 (2-4 sentences) drawing only from section_body.

  Turn 3: user asks a cross-cutting question (e.g., "What changed
    between the two that caused X?", "Which had more impact?").
    Assistant writes reply_3 (2-4 sentences) DIRECTLY — NO tool call.
    Synthesise from the prior tool responses + replies.

JSON keys: topic_a, topic_b, tool_call_1, tool_response_1, reply_1,
followup_1, tool_call_2, tool_response_2, reply_2, followup_2, reply_3.

CRITICAL — title rules same as the article template (plain canonical
Wikipedia titles, no decoration, same title in tool_call_2 and
tool_response_2)."""


TEACHER_SYS_PLACES = """\
You are a teacher generating multi-turn training data for an on-device
assistant. Produce ONE JSON object modelling a 3-turn dialogue:

  Turn 1: user asks for points of interest near PLACE. Assistant
    calls near_named_place(place=PLACE, kinds=KINDS). The synthetic
    tool_response_1 has {resolved: {name, lat, lon}, results: [4-6
    nearby places with {name, type, subtype, distance_m, wiki?}]}.
    Assistant writes reply_1 (3-5 sentences) summarising the top
    results.

  Turn 2: user asks a follow-up about ONE of the results (e.g.,
    "Tell me more about Stanford"). Assistant calls
    article_overview(title=<chosen result name>). The synthetic
    tool_response_2 has {title, lead, available_sections}.
    Assistant writes reply_2 (3-5 sentences).

  Turn 3: user asks a synthesis question (e.g., "So why are there
    so many of these around X?", "Is one of these older than the
    others?"). Assistant writes reply_3 (2-4 sentences) DIRECTLY —
    NO tool call.

JSON keys: place, kinds, tool_call_1, tool_response_1, reply_1,
followup_1, tool_call_2, tool_response_2, reply_2, followup_2,
reply_3.

CRITICAL: tool_response_1.resolved.{name,lat,lon} must be plausible
for PLACE. tool_response_2.title must equal the result name used in
the user follow-up. Plain canonical names."""


# Targets the `sky_is_blue_chain` failure mode: a "why X / so why Y /
# what controls Z" explainer where BOTH the first two turns look up
# *related but distinct* concepts (NOT a section drill-down), and the
# third turn synthesises the governing principle with NO tool call.
TEACHER_SYS_EXPLAINER = """\
You are a teacher generating multi-turn training data for a small
on-device assistant. Produce ONE JSON object modelling a 3-turn
"explainer" dialogue about a science/nature phenomenon:

  Turn 1: user asks "Why is X?" about a phenomenon (e.g. "Why is the
    sky blue?"). Assistant calls article_overview(title=CONCEPT_A)
    where CONCEPT_A is the governing phenomenon (e.g. "Rayleigh
    scattering"). tool_response_1 has {title, lead, available_sections}.
    Assistant writes reply_1 (2-4 sentences) explaining the cause,
    naming the key mechanism and quantities.

  Turn 2: user asks a RELATED "so why Y?" question about a sibling
    phenomenon (e.g. "So why are sunsets red then?"). Assistant calls
    article_overview(title=CONCEPT_B) — a DIFFERENT but related
    article (e.g. "Diffuse sky radiation" or "Sunset"). NOT a section
    drill-down — a fresh overview/search of a related concept.
    tool_response_2 has {title, lead, available_sections}. Assistant
    writes reply_2 (2-4 sentences).

  Turn 3: user asks the GOVERNING-PRINCIPLE question ("Wait, what
    actually controls which one wins?" / "what's the underlying
    rule?"). Assistant writes reply_3 (2-4 sentences) synthesising
    turns 1+2 into the quantitative principle — NO tool call. Name
    the concrete governing factor (e.g. wavelength dependence, the
    inverse-fourth-power law, path length through the atmosphere).

JSON keys: tool_call_1, tool_response_1, reply_1, followup_1,
tool_call_2, tool_response_2, reply_2, followup_2, reply_3.

CRITICAL:
- tool_call_1.function == "article_overview", tool_call_2.function ==
  "article_overview" (BOTH overviews — turn 2 is a fresh related
  concept, NOT get_article_section).
- Plain canonical Wikipedia titles ("Rayleigh scattering", "Sunset").
- reply_3 must contain the concrete governing quantity (wavelength,
  inverse-fourth-power, path length, distance, etc.) — this is the
  whole point of the example."""


# Targets the `french_revolution_chain` failure: a 3-turn HISTORY chain
# whose final turn must be answered from cached context with NO tool call.
# Same JSON schema as the explainer (2 tool calls + 3 replies).
TEACHER_SYS_HISTORY = """\
You are a teacher generating multi-turn training data for a small
on-device assistant. Produce ONE JSON object modelling a 3-turn
"history" dialogue about a historical event:

  Turn 1: user asks how the event unfolded (e.g. "How did the French
    Revolution unfold?"). Assistant calls article_overview(title=EVENT)
    with the event's canonical Wikipedia title. tool_response_1 has
    {title, lead, available_sections}. Assistant writes reply_1
    (2-4 sentences) narrating the arc — name at least one concrete
    year/date and the key actors or phases.

  Turn 2 (followup_1): user asks what role a SPECIFIC PERSON played
    (pick a genuinely central figure, e.g. "What role did Robespierre
    play?"). Assistant calls article_overview(title=PERSON) — a fresh
    related article, NOT a section drill-down. tool_response_2 has
    {title, lead, available_sections}. Assistant writes reply_2
    (2-4 sentences) on the person's role and fate.

  Turn 3 (followup_2): user asks a SHORT elliptical conclusion
    question ("So what ended it?" / "How did it finally end?" /
    "What stopped it?"). Assistant writes reply_3 (2-4 sentences)
    synthesising turns 1+2 from MEMORY — NO tool call. Name the
    concrete ending: the coup, treaty, surrender, collapse, or
    transition that closed the event, with its year.

JSON keys: tool_call_1, tool_response_1, reply_1, followup_1,
tool_call_2, tool_response_2, reply_2, followup_2, reply_3.

CRITICAL:
- tool_call_1.function == "article_overview", tool_call_2.function ==
  "article_overview". Turn 3 has NO tool call — that is the whole
  point of the example.
- Plain canonical Wikipedia titles ("French Revolution",
  "Maximilien Robespierre").
- followup_2 must be SHORT and elliptical (≤ 6 words, pronoun "it").
- reply_3 must name the concrete ending event AND a year."""


# Targets the `narrate_hp_garage` failure: a SINGLE-turn request to
# read a full article verbatim → narrate_article (NOT article_overview).
TEACHER_SYS_NARRATE = """\
You are a teacher generating single-turn training data for a small
on-device assistant. Produce ONE JSON object modelling a 1-turn
dialogue:

  Turn 1: user asks to READ THE FULL ARTICLE / read it aloud / narrate
    it (e.g. "Please read me the full article about the HP Garage",
    "Read me the whole Wikipedia entry on X", "Narrate the article
    about X to me"). Assistant calls narrate_article(title=TOPIC) —
    NOT article_overview (the user wants the WHOLE body, not a
    summary). tool_response_1 has {title, body} where body is 4-8
    sentences of the article's full prose. Assistant writes reply_1
    that reads the body back conversationally (3-6 sentences),
    surfacing the concrete facts (names, places, dates).

JSON keys: tool_call_1, tool_response_1, reply_1.

CRITICAL:
- tool_call_1.function MUST be "narrate_article" with {title: TOPIC}.
- Plain canonical Wikipedia title.
- reply_1 must include the concrete facts from body (proper nouns,
  addresses, dates) — not a vague paraphrase."""


# ----------------------------------------------------------------------
# Schemas — strict JSON schemas the teacher conforms to.
# ----------------------------------------------------------------------
_CALL_PROPS = {
    "function":   {"type": "string"},
    "parameters": {"type": "object"},
}
_CALL_SCHEMA = {
    "type": "object",
    "properties": _CALL_PROPS,
    "required": ["function", "parameters"],
}

SCHEMA_ARTICLE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_call_1":     _CALL_SCHEMA,
        "tool_response_1": {"type": "object"},
        "reply_1":         {"type": "string"},
        "followup_1":      {"type": "string"},
        "tool_call_2":     _CALL_SCHEMA,
        "tool_response_2": {"type": "object"},
        "reply_2":         {"type": "string"},
        "followup_2":      {"type": "string"},
        "reply_3":         {"type": "string"},
    },
    "required": ["tool_call_1", "tool_response_1", "reply_1", "followup_1",
                 "tool_call_2", "tool_response_2", "reply_2", "followup_2",
                 "reply_3"],
}

SCHEMA_COMPARE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic_a":         {"type": "string"},
        "topic_b":         {"type": "string"},
        "tool_call_1":     _CALL_SCHEMA,
        "tool_response_1": {"type": "object"},
        "reply_1":         {"type": "string"},
        "followup_1":      {"type": "string"},
        "tool_call_2":     _CALL_SCHEMA,
        "tool_response_2": {"type": "object"},
        "reply_2":         {"type": "string"},
        "followup_2":      {"type": "string"},
        "reply_3":         {"type": "string"},
    },
    "required": ["topic_a", "topic_b",
                 "tool_call_1", "tool_response_1", "reply_1", "followup_1",
                 "tool_call_2", "tool_response_2", "reply_2", "followup_2",
                 "reply_3"],
}

SCHEMA_PLACES: dict[str, Any] = {
    "type": "object",
    "properties": {
        "place":           {"type": "string"},
        "kinds":           {"type": "array", "items": {"type": "string"}},
        "tool_call_1":     _CALL_SCHEMA,
        "tool_response_1": {"type": "object"},
        "reply_1":         {"type": "string"},
        "followup_1":      {"type": "string"},
        "tool_call_2":     _CALL_SCHEMA,
        "tool_response_2": {"type": "object"},
        "reply_2":         {"type": "string"},
        "followup_2":      {"type": "string"},
        "reply_3":         {"type": "string"},
    },
    "required": ["place", "kinds",
                 "tool_call_1", "tool_response_1", "reply_1", "followup_1",
                 "tool_call_2", "tool_response_2", "reply_2", "followup_2",
                 "reply_3"],
}

# Explainer chains share the article 9-key chain shape — both turns are
# tool calls + a final synthesis. Renders through chain_to_messages.
SCHEMA_EXPLAINER = SCHEMA_ARTICLE

# Narrate is a single-turn 3-key shape (one tool call + reply).
SCHEMA_NARRATE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_call_1":     _CALL_SCHEMA,
        "tool_response_1": {"type": "object"},
        "reply_1":         {"type": "string"},
    },
    "required": ["tool_call_1", "tool_response_1", "reply_1"],
}


# ----------------------------------------------------------------------
# Seed query templates.
# ----------------------------------------------------------------------
ARTICLE_OPENERS = [
    "what is {topic}",
    "tell me about {topic}",
    "explain {topic}",
    "what are {topic}",          # works for plural/compound topics
    "how does {topic} work",
    "what happened with {topic}",
    "give me an overview of {topic}",
    "how did {topic} unfold",
    "who was {person}",
]

COMPARE_OPENERS = [
    "Compare {a} and {b}",
    "How are {a} and {b} different",
    "What's the difference between {a} and {b}",
    "{a} vs {b}",
    "Compare {a} and {b} — causes and scale",
    "How do {a} and {b} compare",
]

PLACES_OPENERS = [
    "What are some interesting {kind}s in {place}",
    "{kind}s near {place}",
    "Top {kind}s in {place}",
    "Show me {kind}s in {place}",
    "Any good {kind}s near {place}",
]

# History-event chains — targets the `french_revolution_chain` grid miss
# (2026-06-09, the v7-full model's ONLY failing scenario). Shape: how-did-it-
# unfold → role-of-person → elliptical "so what ended it?" answered from
# cached context with NO tool call. EXPLAINER_OPENERS is all physics/nature,
# so the FT never saw this pattern on historical narratives.
HISTORY_OPENERS = [
    "How did the {event} unfold?",
    "What happened during the {event}?",
    "Walk me through the {event}.",
    "How did the {event} start and play out?",
    "Tell me how the {event} developed.",
]

HISTORY_EVENTS = [
    "French Revolution", "Russian Revolution", "American Civil War",
    "fall of the Western Roman Empire", "Meiji Restoration",
    "Protestant Reformation", "Norman conquest of England",
    "Haitian Revolution", "unification of Germany", "Cuban Missile Crisis",
    "fall of the Berlin Wall", "partition of India", "Manhattan Project",
    "Spanish Civil War", "Glorious Revolution", "Black Death",
    "Suez Crisis", "Apollo program", "English Civil War",
    "Mexican Revolution", "Reconstruction era", "Hundred Years' War",
    "Bronze Age collapse", "Taiping Rebellion",
]

# Explainer: "why is X?" science/nature phenomena. The teacher picks the
# governing article; we just seed the natural-language opener.
EXPLAINER_OPENERS = [
    "Why is the sky blue?",
    "Why is the ocean blue?",
    "Why do leaves change color in autumn?",
    "Why is the sky blue?",
    "Why does ice float?",
    "Why is blood red?",
    "Why do we see a rainbow?",
    "Why is grass green?",
    "Why does the moon look bigger near the horizon?",
    "Why do stars twinkle?",
    "Why does metal feel colder than wood?",
    "Why is the sunset red?",
    "Why does the sky turn orange at dusk?",
    "Why do we get auroras?",
    "Why is snow white?",
    "Why does sound travel faster in water?",
    "Why do clouds float?",
    "Why does a prism split light?",
    "Why is the deep sea dark?",
    "Why does the sky go pink sometimes?",
    "Why do hot things glow red then white?",
    "Why is the daytime sky bright even away from the sun?",
]

# Narrate: "read me the full article" single-turn requests.
NARRATE_OPENERS = [
    "Please read me the full article about {topic}.",
    "Read me the whole Wikipedia entry on {topic}.",
    "Narrate the article about {topic} to me.",
    "Read the full {topic} article aloud.",
    "Can you read me everything about {topic}?",
    "Read me the complete article on {topic}.",
    "Give me the full text of the {topic} article.",
    "Read the whole thing about {topic} to me.",
    "I want you to read the entire article about {topic}.",
    "Read me the full Wikipedia page for {topic}.",
]

# Narrate topics — mix of places/landmarks (like HP Garage) + people +
# concrete nouns that have a readable article body.
NARRATE_TOPICS = [
    "the HP Garage", "the Golden Gate Bridge", "the Eiffel Tower",
    "Stonehenge", "the Great Wall of China", "the Statue of Liberty",
    "Machu Picchu", "the Colosseum", "Mount Fuji", "Niagara Falls",
    "the Hoover Dam", "the Sydney Opera House", "Big Ben",
    "the Brooklyn Bridge", "Alcatraz", "the Hollywood Sign",
    "Stanford University", "the Apollo 11 mission", "the Titanic",
    "the Wright Flyer", "the Rosetta Stone", "the Hubble Space Telescope",
    "the Berlin Wall", "Pompeii", "the Lincoln Memorial",
    "Times Square", "the Empire State Building", "Yellowstone",
    "the Panama Canal", "the Transcontinental Railroad",
]


def _explainer_query() -> str:
    return random.choice(EXPLAINER_OPENERS)


def _history_query() -> str:
    return random.choice(HISTORY_OPENERS).format(
        event=random.choice(HISTORY_EVENTS))


def _narrate_query() -> str:
    return random.choice(NARRATE_OPENERS).format(
        topic=random.choice(NARRATE_TOPICS))


def _article_query() -> str:
    t = random.choice(ARTICLE_OPENERS)
    if "{person}" in t:
        return t.format(person=random.choice(PEOPLE))
    return t.format(topic=random.choice(TOPICS))


# Topic pairs that make sense to compare (curated, no random nonsense).
COMPARE_PAIRS = [
    ("World War I", "World War II"),
    ("the French Revolution", "the American Revolution"),
    ("Roman Empire", "Byzantine Empire"),
    ("Newton", "Einstein"),
    ("Apollo program", "Space Shuttle"),
    ("Darwin", "Mendel"),
    ("Tesla", "Edison"),
    ("Renaissance", "Enlightenment"),
    ("Ming Dynasty", "Qing Dynasty"),
    ("Athens", "Sparta"),
    ("CRISPR", "TALEN"),
    ("DNA", "RNA"),
    ("Mars", "Venus"),
    ("the Cold War", "the Space Race"),
    ("Buddhism", "Hinduism"),
    ("Marx", "Adam Smith"),
    ("Beethoven", "Mozart"),
    ("Galileo", "Copernicus"),
    ("Picasso", "Matisse"),
    ("the Industrial Revolution", "the Information Revolution"),
]


def _compare_query() -> tuple[str, str, str]:
    a, b = random.choice(COMPARE_PAIRS)
    if random.random() < 0.5:
        a, b = b, a
    t = random.choice(COMPARE_OPENERS)
    return a, b, t.format(a=a, b=b)


def _places_query() -> tuple[str, list[str], str]:
    place = random.choice(PLACES_FOR_NEAR)
    kinds = random.choice(NEAR_KINDS)
    t = random.choice(PLACES_OPENERS)
    return place, kinds, t.format(place=place, kind=kinds[0])


# ----------------------------------------------------------------------
# Render to eval-matched trajectory (same format as v4_combined).
# ----------------------------------------------------------------------
def _eval_preamble() -> str:
    pre = SYSTEM_PREAMBLE + "\n" + _build_tool_block()
    if random.random() < 0.5:
        lat, lon = random.choice(LOC_POOL)
        pre += f"\n\ncurrentLocation: lat={lat} lon={lon}"
    return pre


def _coerce_str(v: Any) -> str:
    """Coerce a teacher-emitted field to a plain string.

    Some teachers (e.g. OMLX `gemma-4-26b-a4b-it-4bit`) wrap text in
    a `{type, text}` or `{content}` dict instead of a plain string.
    Pull the obvious text out; otherwise stringify."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        for k in ("text", "content", "value", "response"):
            if isinstance(v.get(k), str):
                return v[k].strip()
        return json.dumps(v, ensure_ascii=False).strip()
    if v is None:
        return ""
    return str(v).strip()


def chain_to_messages(query: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Render the 3-turn chain into a 10-message student trajectory.

    Wire format: same as `generate_chains.py.chain_to_messages` but
    with an extra (user, assistant) pair at the end where the
    assistant emits NO `tool_call` fence — pure synthesis."""
    preamble = _eval_preamble()
    tc1 = json.dumps(obj["tool_call_1"], ensure_ascii=False)
    tr1 = json.dumps(obj["tool_response_1"], ensure_ascii=False)
    tc2 = json.dumps(obj["tool_call_2"], ensure_ascii=False)
    tr2 = json.dumps(obj["tool_response_2"], ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content":
                preamble + "\n\nUser query:\n" + query},
            {"role": "assistant", "content":
                f"```tool_call\n{tc1}\n```"},
            {"role": "user", "content":
                "[TOOL_RESPONSE]\n" + tr1},
            {"role": "assistant", "content": _coerce_str(obj["reply_1"])},
            {"role": "user", "content": _coerce_str(obj["followup_1"])},
            {"role": "assistant", "content":
                f"```tool_call\n{tc2}\n```"},
            {"role": "user", "content":
                "[TOOL_RESPONSE]\n" + tr2},
            {"role": "assistant", "content": _coerce_str(obj["reply_2"])},
            {"role": "user", "content": _coerce_str(obj["followup_2"])},
            {"role": "assistant", "content": _coerce_str(obj["reply_3"])},
        ]
    }


def narrate_to_messages(query: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Render a single-turn narrate trajectory into a 4-message student
    trajectory (user → tool_call → tool_response → reply)."""
    preamble = _eval_preamble()
    tc1 = json.dumps(obj["tool_call_1"], ensure_ascii=False)
    tr1 = json.dumps(obj["tool_response_1"], ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content":
                preamble + "\n\nUser query:\n" + query},
            {"role": "assistant", "content":
                f"```tool_call\n{tc1}\n```"},
            {"role": "user", "content":
                "[TOOL_RESPONSE]\n" + tr1},
            {"role": "assistant", "content": _coerce_str(obj["reply_1"])},
        ]
    }


# ----------------------------------------------------------------------
# Async teacher call (one row per call).
# ----------------------------------------------------------------------
# Each entry: (teacher_system, schema, sample_query, renderer).
TEMPLATES = {
    "article":   (TEACHER_SYS_ARTICLE,   SCHEMA_ARTICLE,   _article_query,            chain_to_messages),
    "compare":   (TEACHER_SYS_COMPARE,   SCHEMA_COMPARE,   lambda: _compare_query()[2], chain_to_messages),
    "places":    (TEACHER_SYS_PLACES,    SCHEMA_PLACES,    lambda: _places_query()[2],  chain_to_messages),
    "explainer": (TEACHER_SYS_EXPLAINER, SCHEMA_EXPLAINER, _explainer_query,          chain_to_messages),
    "history":   (TEACHER_SYS_HISTORY,   SCHEMA_EXPLAINER, _history_query,            chain_to_messages),
    "narrate":   (TEACHER_SYS_NARRATE,   SCHEMA_NARRATE,   _narrate_query,            narrate_to_messages),
}


def _tool_for(template: str) -> dict[str, Any]:
    """Wrap the per-template schema as an OpenAI tool definition.

    DS4 silently ignores `response_format={type:json_schema}` (the
    server doesn't grammar-constrain against the schema, so we get
    valid-shaped JSON with empty/null fields). Tool calling is DS4's
    actually-grammar-constrained path: it renders our tool schema into
    its native DSML format, the model emits a DSML call, and the
    server maps it back to an OpenAI tool_call we can read.
    """
    _, schema, _, _ = TEMPLATES[template]
    return {
        "type": "function",
        "function": {
            "name": f"emit_chain3_{template}",
            "description": (
                f"Emit ONE training trajectory for a 3-turn "
                f"`{template}` chain. Fill every required field."
            ),
            "parameters": schema,
        },
    }


def _strip_fences(s: str) -> str:
    s = s.strip()
    for f in ("```json", "```JSON", "```"):
        if s.startswith(f):
            s = s[len(f):].lstrip()
            break
    if s.endswith("```"):
        s = s[:-3].rstrip()
    return s


async def generate_one(
    client: AsyncOpenAI, model: str, template: str, query: str,
    temperature: float, max_tokens: int,
    use_tools: bool = True,
) -> Optional[dict[str, Any]]:
    teacher_sys, _, _, _ = TEMPLATES[template]

    if use_tools:
        # Grammar-constrained tool-call path (DS4 + most OpenAI-compat
        # servers). Best fidelity; works on DS4 with
        # `extra_body={"thinking":{"type":"disabled"}}` to skip its
        # reasoning block.
        tool = _tool_for(template)
        try:
            resp = await client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": teacher_sys},
                    {"role": "user",
                     "content":
                        f"Student query: {query}\n\nProduce the trajectory "
                        f"now by calling `emit_chain3_{template}` exactly "
                        f"once. Fill every required field."},
                ],
                tools=[tool],
                tool_choice={"type": "function",
                             "function": {"name": f"emit_chain3_{template}"}},
                extra_body={"thinking": {"type": "disabled"}},
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return {"_error": "no tool_calls in response",
                        "_query": query, "_template": template,
                        "_raw": (msg.content or "")[:300]}
            args_raw = msg.tool_calls[0].function.arguments
            return json.loads(args_raw)
        except Exception as e:
            return {"_error": str(e), "_query": query, "_template": template}

    # Plain-JSON path. Used for LM Studio + Gemma 4 (LM Studio's
    # tools= pass-through misrenders Gemma 4's chat template and
    # returns empty content). Gemma 4 follows JSON-only instructions
    # well; we just strip any stray ```json``` fences.
    sys_plain = (
        teacher_sys
        + "\n\nOutput ONE JSON object. NO prose, NO markdown fences. "
        "Begin with `{` and end with `}`."
    )
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys_plain},
                {"role": "user",
                 "content":
                    f"Student query: {query}\n\nProduce the JSON now."},
            ],
        )
        raw = resp.choices[0].message.content or ""
        return json.loads(_strip_fences(raw))
    except Exception as e:
        return {"_error": str(e), "_query": query, "_template": template,
                "_raw": (raw if 'raw' in dir() else "")[:300]}


# ----------------------------------------------------------------------
# Validators — catch teacher mistakes before they pollute training.
# ----------------------------------------------------------------------
def _normalize_call(call: Any) -> dict[str, Any]:
    """Coerce a teacher-emitted tool-call object to our canonical
    `{function: str, parameters: dict}` shape. Teachers vary:
    - DS4: {function, parameters} (our canonical form)
    - Gemma 4 LM Studio: {function, arguments}
    - Gemma 4 OMLX: {name, arguments}
    - Some emit it as a JSON-encoded *string* instead of an object
    - Some wrap `arguments` itself as a JSON-encoded string"""
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except json.JSONDecodeError:
            return {}
    if not isinstance(call, dict):
        return {}
    if "name" in call and "function" not in call:
        call["function"] = call.pop("name")
    if "method" in call and "function" not in call:
        call["function"] = call.pop("method")
    if "arguments" in call and "parameters" not in call:
        call["parameters"] = call.pop("arguments")
    if "args" in call and "parameters" not in call:
        call["parameters"] = call.pop("args")
    params = call.get("parameters")
    if isinstance(params, str):
        try:
            call["parameters"] = json.loads(params)
        except json.JSONDecodeError:
            pass
    return call


def validate_row(obj: dict[str, Any], template: str) -> Optional[str]:
    """Return an error string if the row is malformed, else None."""
    if "_error" in obj:
        return obj["_error"]

    # narrate is a single-turn 3-key shape; validate + return early.
    if template == "narrate":
        for k in ("tool_call_1", "tool_response_1", "reply_1"):
            if k not in obj:
                return f"missing required key: {k}"
        tc1 = _normalize_call(obj.get("tool_call_1", {}))
        obj["tool_call_1"] = tc1
        if tc1.get("function") != "narrate_article":
            return f"tool_call_1 fn={tc1.get('function')!r}, expected narrate_article"
        if not tc1.get("parameters", {}).get("title"):
            return "narrate_article missing title"
        if "```tool_call" in obj.get("reply_1", ""):
            return "reply_1 contains a tool_call fence"
        return None

    # All other templates use the 9-key chain shape.
    required = ("tool_call_1", "tool_response_1", "reply_1", "followup_1",
                "tool_call_2", "tool_response_2", "reply_2", "followup_2",
                "reply_3")
    missing = [k for k in required if k not in obj]
    if missing:
        return f"missing required keys: {missing}"
    tc1 = _normalize_call(obj.get("tool_call_1", {}))
    tc2 = _normalize_call(obj.get("tool_call_2", {}))
    obj["tool_call_1"] = tc1
    obj["tool_call_2"] = tc2
    fn1 = tc1.get("function")
    fn2 = tc2.get("function")
    if template == "article":
        if fn1 != "article_overview":
            return f"tool_call_1 fn={fn1!r}, expected article_overview"
        if fn2 != "get_article_section":
            return f"tool_call_2 fn={fn2!r}, expected get_article_section"
        title1 = tc1.get("parameters", {}).get("title")
        title2 = tc2.get("parameters", {}).get("title")
        if not title1 or not title2 or title1 != title2:
            return f"titles mismatch: tc1={title1!r} tc2={title2!r}"
    elif template == "compare":
        if fn1 != "compare_articles":
            return f"tool_call_1 fn={fn1!r}, expected compare_articles"
        if fn2 != "get_article_section":
            return f"tool_call_2 fn={fn2!r}, expected get_article_section"
        titles = tc1.get("parameters", {}).get("titles", [])
        if not isinstance(titles, list) or len(titles) < 2:
            return f"compare titles invalid: {titles!r}"
    elif template == "places":
        if fn1 != "near_named_place":
            return f"tool_call_1 fn={fn1!r}, expected near_named_place"
        if fn2 != "article_overview":
            return f"tool_call_2 fn={fn2!r}, expected article_overview"
    elif template == "explainer":
        # Both turns are fresh overviews/searches of related concepts —
        # NOT a section drill-down. Accept article_overview or search.
        if fn1 not in ("article_overview", "search"):
            return f"tool_call_1 fn={fn1!r}, expected article_overview/search"
        if fn2 not in ("article_overview", "search"):
            return f"tool_call_2 fn={fn2!r}, expected article_overview/search"
    # Reply 3 sanity: must NOT contain a tool_call fence
    r3 = obj.get("reply_3", "")
    if "```tool_call" in r3 or "<tool_call>" in r3:
        return "reply_3 contains a tool_call fence (this turn must be pure synthesis)"
    return None


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
async def run(args):
    out_path = Path(args.out)
    fail_path = Path(args.fail_log) if args.fail_log else None
    existing = 0
    if out_path.exists():
        with open(out_path) as fh:
            existing = sum(1 for _ in fh if _.strip())
        print(f"[resume] {existing} rows already in {out_path}", file=sys.stderr)

    client = AsyncOpenAI(base_url=args.base_url,
                         api_key=args.api_key or "lm-studio")
    # `--templates a:0.5,b:0.3` overrides the default mix. Bare names
    # (no `:weight`) get equal share of the remaining mass.
    if args.templates:
        weights = {}
        bare = []
        for part in args.templates.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                name, w = part.split(":", 1)
                weights[name.strip()] = float(w)
            else:
                bare.append(part)
        if bare:
            rem = max(0.0, 1.0 - sum(weights.values()))
            for name in bare:
                weights[name] = rem / len(bare)
        for name in weights:
            if name not in TEMPLATES:
                raise SystemExit(f"unknown template: {name!r} "
                                 f"(have {list(TEMPLATES)})")
    else:
        weights = {"article": 0.5, "compare": 0.3, "places": 0.2}
    template_counts: dict[str, int] = {k: 0 for k in weights}
    total_targets = {
        t: int(round(args.n * w)) for t, w in weights.items()
    }

    todo = max(0, args.n - existing)
    if todo == 0:
        print(f"[done] target {args.n} already met", file=sys.stderr)
        return
    print(f"[start] generating {todo} more rows (target {args.n}) "
          f"by template: {total_targets}", file=sys.stderr)
    print("[cache-aware] DS4 keeps a single live KV checkpoint and "
          "auto-dedups identical byte prefixes; switching templates "
          "invalidates it. Processing one template at a time at "
          f"concurrency={args.concurrency} to keep the system+tool "
          "prefix warm across requests.", file=sys.stderr)

    out_fh = open(out_path, "a")
    fail_fh = open(fail_path, "a") if fail_path else None
    t0 = time.time()
    n_ok = n_fail = 0

    async def emit_one(template: str) -> tuple[bool, str]:
        _, _, sample_query, renderer = TEMPLATES[template]
        query = sample_query()
        obj = await generate_one(
            client, args.model, template, query,
            args.temperature, args.max_tokens,
            use_tools=not args.no_tools)
        err = validate_row(obj, template) if obj else "no response"
        if err:
            if fail_fh:
                fail_fh.write(json.dumps(
                    {"template": template, "query": query,
                     "error": err, "obj": obj}) + "\n")
                fail_fh.flush()
            return False, err
        row = renderer(query, obj)
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        return True, ""

    sem = asyncio.Semaphore(args.concurrency)

    async def one_with_sem(template: str) -> tuple[bool, str]:
        async with sem:
            return await emit_one(template)

    # Process templates serially so DS4's cached system+tool prefix
    # stays valid across consecutive requests of the same shape.
    # Within a template, run with the configured concurrency.
    for template in weights:
        target = total_targets[template]
        if target == 0:
            continue
        print(f"[template={template}] target={target} (warming KV cache "
              f"with first request, then reusing prefix)", file=sys.stderr)
        tasks = [asyncio.create_task(one_with_sem(template))
                 for _ in range(target)]
        for fut in asyncio.as_completed(tasks):
            ok, err = await fut
            if ok:
                n_ok += 1
                template_counts[template] += 1
            else:
                n_fail += 1
            if (n_ok + n_fail) % 10 == 0:
                dt = time.time() - t0
                rate = (n_ok + n_fail) / max(dt, 1)
                print(
                    f"  ok={n_ok} fail={n_fail} "
                    f"({n_ok / max(n_ok + n_fail, 1):.0%} ok) "
                    f"rate={rate:.2f}/s "
                    f"templates={template_counts}", file=sys.stderr)
    print(f"[done] ok={n_ok} fail={n_fail} templates={template_counts}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1",
                    help="OpenAI-compatible endpoint")
    ap.add_argument("--api-key", default="",
                    help="API key (LM Studio default works empty)")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n", type=int, default=800,
                    help="Target row count (resumable; counts existing rows)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--out", default="train_chains3.jsonl")
    ap.add_argument("--fail-log", default="train_chains3_fails.jsonl")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-tools", action="store_true",
                    help="Skip OpenAI tools= API. Use a plain JSON-only "
                         "system instruction. Needed for LM Studio + "
                         "Gemma 4 (tools= pass-through returns empty).")
    ap.add_argument("--templates", default="",
                    help="Comma-sep template mix, e.g. "
                         "'explainer:0.5,narrate:0.25,article:0.25'. "
                         "Bare names split the remaining mass equally. "
                         "Default: article:0.5,compare:0.3,places:0.2.")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
