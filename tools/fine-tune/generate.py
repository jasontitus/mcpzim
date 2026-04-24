"""Generate fine-tune training data for Gemma 3 4B by asking a local
LM Studio server (OpenAI-compatible API) to produce (user, assistant)
pairs that follow our tool-calling conventions.

Design: single-shot generation per example. The teacher is given the
tool schemas + one sample query + a canned fixture sketch, and it
produces the full assistant trajectory (tool_call + final response).
We never run the actual tools — the teacher invents plausible tool
results from a fixture schema we hand it, and the student learns to
match the FORM of those responses.

Setup:
  1. Start LM Studio, load Qwen 3.6 27B (or any instruction-tuned
     model >= 14B). Enable "Local Server" on port 1234.
  2. uv pip install openai tqdm
  3. python generate.py --n 100 --concurrency 8 --out train.jsonl

Output is mlx_lm.lora-compatible JSONL — one JSON object per line
with a `messages` array.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    print("pip install openai", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------
# Tool schemas — must match MCPToolAdapter.swift output shape so the
# student learns the EXACT format our Swift code will feed it at
# inference time.
# ----------------------------------------------------------------------
TOOLS_DESC = """
Tools (pick ONE):
- near_places(lat, lon, place?, radius_km?, kinds?)
- near_named_place(place, radius_km?, kinds?)
- nearby_stories(lat?, lon?, place?, kinds?)
- article_overview(title)
- get_article_section(title, section)
- compare_articles(titles)
- narrate_article(title)
- what_is_here(lat, lon)
- route_from_places(origin, destination)
- search(query)

Realistic tool_response shapes:
- near_*: {results:[{name,type,subtype,lat,lon,distance_m}], total_in_radius, by_category}
- nearby_stories: results[*] also have wiki/excerpt/wiki_path
- article_overview: {title, lead, available_sections}
- get_article_section: {section_body}
- compare_articles: {a, b, comparison}
- narrate_article: {article_body}
- what_is_here: {place}
- route_from_places: {summary, steps}
- search: {hits:[{title, snippet}]}
""".strip()


SYSTEM_PROMPT = f"""\
You are a teacher generating training data for a smaller on-device
assistant on a phone with offline Wikipedia + OpenStreetMap.

Emit ONE JSON object: {{tool_call, tool_response, assistant_reply}}.

- tool_call: {{"function":"<name>","parameters":{{...}}}}
- tool_response: invent realistic data for that tool. near_* results
  should have 3-8 entries; total_in_radius 10-2000.
- assistant_reply: 1-3 sentences for place/directions/current-place,
  3-6 sentences for learn/discuss/compare topics. Use only facts
  from tool_response.

{TOOLS_DESC}
"""


# JSON schema enforced by LM Studio / llama.cpp via response_format.
# Guarantees every response is parseable — no more fence extraction,
# no more parser rejections.
TRAJECTORY_SCHEMA: dict[str, Any] = {
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
# Seed queries. Covers the same pattern space our 13 eval scenarios
# exercise, plus natural paraphrases you'd hear on a phone.
# ----------------------------------------------------------------------
# Seeds organised by use-case so we can balance the training
# distribution. Each category also carries a suggested follow-up
# pattern used by the multi-turn generator — the pattern is a list
# of {0}-style templates where the assistant's prior reply is the
# implicit context.
SEED_CATEGORIES: dict[str, dict[str, Any]] = {
    "directions": {
        "weight": 1.0,
        "openers": [
            "directions to {place}",
            "how do I get to {place}",
            "route to {place}",
            "navigate to {place}",
            "take me to {place}",
            "how far is {place} from here",
            "which way to {place}",
            "I want to drive to {place}",
            "fastest way to {place}",
            "directions to the nearest {poi}",
        ],
        "followups": [
            "how long will it take",
            "any alternate route",
            "is there traffic",
            "can I walk instead",
            "what's the distance",
            "any stops worth making along the way",
        ],
    },
    "exploration": {
        "weight": 1.5,     # primary use case — weight heavier
        "openers": [
            "what's near me",
            "anything interesting around here",
            "show me what's nearby",
            "{poi} near me",
            "good {poi} near me",
            "cool spots around {place}",
            "what's fun in {place}",
            "recommend something to do in {place}",
            "places to visit in {place}",
            "best {poi} in {place}",
            "hidden gems in {place}",
        ],
        "followups": [
            "which is closest to me",
            "which is closest to {landmark}",
            "which has the best reviews",
            "are any open now",
            "show me only the {subtype} ones",
            "how far is the first one",
            "what about walking distance only",
        ],
    },
    "learn_topic": {
        "weight": 1.5,
        "openers": [
            "tell me about {topic}",
            "what is {topic}",
            "history of {topic}",
            "summarize {topic} for me",
            "who was {person}",
            "explain {topic}",
            "what happened in {topic}",
            "give me a quick intro to {topic}",
            "what's the deal with {topic}",
            "teach me about {topic}",
        ],
        "followups": [
            "tell me more",
            "what caused it",
            "why did that happen",
            "who were the key people",
            "what came next",
            "what's the significance",
            "how does it work",
            "when did it happen",
        ],
    },
    "discuss_aspect": {
        # "Follow-up on a specific aspect" — higher complexity, needs
        # multi-turn context. Weighted heavy because this is what
        # usually breaks small models.
        "weight": 1.5,
        "openers": [
            "tell me about {topic}",
            "explain {topic}",
            "what is {topic}",
        ],
        "followups": [
            "how does the {aspect} part work",
            "why did the {aspect} happen",
            "what role did {person} play in it",
            "how is it different from {topic_b}",
            "what are the main criticisms",
            "what's the scientific mechanism behind it",
            "so what do {aspect} imply",
            "what's the simplest analogy for this",
        ],
    },
    "current_place": {
        "weight": 1.0,
        "openers": [
            "where am I",
            "what's this neighborhood",
            "tell me about where I am",
            "what's interesting right here",
            "what's the history of this area",
            "any stories from here",
            "any stories from around this spot",
            "any wiki stories nearby",
            "any historical stories here",
            "what wiki-linked places are nearby",
            "famous landmarks around me",
            "what's {place} known for",
            "tell me something cool about my location",
        ],
        "followups": [
            "which of those is closest",
            "tell me more about the first one",
            "any more details on {landmark}",
            "is any of this walking distance",
            "which is the most historic",
        ],
    },
    "compare": {
        "weight": 1.0,
        "openers": [
            "how is {topic_a} different from {topic_b}",
            "{topic_a} vs {topic_b}",
            "compare {topic_a} and {topic_b}",
            "what's the difference between {topic_a} and {topic_b}",
            "{topic_a} or {topic_b} — which is bigger",
            "how does {topic_a} compare to {topic_b}",
        ],
        "followups": [
            "which one is older",
            "which is more widely used",
            "which had more impact",
            "in what way are they similar",
            "who prefers {topic_a}",
        ],
    },
    "narrate": {
        "weight": 0.5,
        "openers": [
            "read me the article on {topic}",
            "read aloud the {topic} article",
            "narrate {topic} for me",
            "tell me the full story of {topic}",
            "read the full {topic} wiki article to me",
            "give me the full wiki text on {topic}",
            "read out the whole {topic} article",
            "play the full article on {topic}",
        ],
        "followups": [
            "just give me the key points instead",
            "skip to the later sections",
            "read the next section",
        ],
    },
    "wiki_search": {
        "weight": 0.5,
        "openers": [
            "search wikipedia for {topic}",
            "find wiki articles about {topic}",
            "look up {topic}",
            "any wikipedia entries for {topic}",
            "what wiki articles mention {person}",
            "search for articles on {topic}",
        ],
        "followups": [
            "pick the most relevant one",
            "show more results",
        ],
    },
}

# Derived flat lists (for single-shot sampler compat).
SEED_QUERIES = [
    q for cat in SEED_CATEGORIES.values() for q in cat["openers"]
]

ASPECTS = [
    "physics", "politics", "chronology", "culture", "economics",
    "military", "biology", "chemistry", "technology", "social",
    "legal", "architectural", "artistic", "philosophical",
]

LANDMARKS = [
    "Caltrain", "the Ferry Building", "Transamerica Pyramid",
    "City Hall", "Coit Tower", "the Embarcadero", "Dolores Park",
    "Union Square", "Fisherman's Wharf", "Stanford",
    "Castro theater", "Painted Ladies", "HP Garage",
]

POIS = [
    "bars", "coffee shops", "restaurants", "bookstores",
    "parks", "museums", "gyms", "grocery stores", "pharmacies",
    "gas stations", "hotels", "theaters", "galleries",
]

PLACES = [
    # SF neighbourhoods + Bay Area cities — primary use case
    "San Carlos", "Palo Alto", "Mountain View", "North Beach",
    "the Mission", "Berkeley", "Oakland", "Sausalito",
    "Fisherman's Wharf", "the Castro", "SoMa", "Hayes Valley",
    "Chinatown", "the Marina", "Presidio", "Richmond",
    "Sunset", "Nob Hill", "Russian Hill", "Pacific Heights",
    "Los Altos", "Menlo Park", "Redwood City", "San Mateo",
    "Cupertino", "Sunnyvale", "Santa Clara", "Los Gatos",
    "Tiburon", "Mill Valley", "Tiburon Belvedere", "Alameda",
    "Emeryville", "Albany", "El Cerrito", "Richmond East Bay",
    "Walnut Creek", "Lafayette", "Orinda", "Moraga",
    "Piedmont", "Hayward", "San Leandro", "Pleasanton",
    "Livermore", "Fremont", "Milpitas", "San Jose",
    "Campbell", "Saratoga", "Monte Sereno", "Cambrian Park",
    "Willow Glen", "Japantown", "Telegraph Hill", "Jackson Square",
    "Financial District", "Cole Valley", "NoPa", "Lower Haight",
    "Duboce Triangle", "Dogpatch", "Potrero Hill", "Bayview",
    "Mission Bay", "Tenderloin", "Mid-Market", "Cow Hollow",
    # Broader US / world (for directions + topics)
    "San Francisco", "New York", "Boston", "Chicago",
    "Los Angeles", "Seattle", "Portland", "Austin",
    "Washington DC", "Philadelphia", "Miami", "New Orleans",
    "London", "Paris", "Berlin", "Rome", "Barcelona",
    "Amsterdam", "Vienna", "Prague", "Tokyo", "Seoul",
    "Hong Kong", "Singapore", "Mumbai", "Cape Town",
]

TOPICS = [
    # Science & nature
    "Rayleigh scattering", "gravitational waves", "CRISPR-Cas9",
    "black holes", "photosynthesis", "plate tectonics",
    "quantum entanglement", "DNA replication", "evolution",
    "the greenhouse effect", "dark matter", "string theory",
    "the human genome", "antibiotics", "mRNA vaccines",
    "neutron stars", "supernovae", "the Big Bang",
    "cosmic inflation", "the periodic table",
    "exoplanets", "JWST", "LIGO",
    "the Higgs boson", "the standard model",
    "general relativity", "special relativity",
    "the theory of evolution", "natural selection",
    "the immune system", "the blood-brain barrier",
    "mitosis", "meiosis", "protein folding",
    "AI alignment", "transformers", "attention mechanisms",
    # History & culture
    "the French Revolution", "World War I", "World War II",
    "the Apollo program", "the Roman Empire", "the Renaissance",
    "the Silk Road", "the Industrial Revolution", "the Cold War",
    "the Age of Exploration", "the Enlightenment",
    "the Byzantine Empire", "the Ottoman Empire",
    "the Ming Dynasty", "the Meiji Restoration",
    "the Haitian Revolution", "the Russian Revolution",
    "the American Civil War", "the Cuban Missile Crisis",
    "the Space Race", "the civil rights movement",
    "the suffragette movement", "the moon landing",
    "the fall of Rome", "the Black Death",
    "the Crusades", "the Reformation", "the Magna Carta",
    "Ancient Greece", "Ancient Egypt", "the Aztec Empire",
    "the Inca Empire", "the Mongol Empire",
    "Vikings", "the Knights Templar",
    # Tech + companies
    "Silicon Valley", "the HP Garage", "Stanford University",
    "the Apollo Guidance Computer", "Xerox PARC",
    "ARPANET", "the IBM PC", "Linux",
    # Places / landmarks as topics
    "Palo Alto", "the Golden Gate Bridge",
    "the Statue of Liberty", "the Eiffel Tower",
    "Machu Picchu", "Stonehenge", "the Great Wall",
    "the pyramids of Giza", "the Colosseum",
    "Angkor Wat", "Petra", "Chichen Itza",
    "the Acropolis", "Mount Fuji", "Yosemite",
]

PEOPLE = [
    "Marie Curie", "Albert Einstein", "Rosa Parks", "Alan Turing",
    "Ada Lovelace", "Richard Feynman", "Elon Musk", "Jeff Bezos",
    "Steve Jobs", "Satya Nadella", "Lin-Manuel Miranda",
    "Grace Hopper", "Claude Shannon", "Vint Cerf",
    "Tim Berners-Lee", "Larry Page", "Sergey Brin",
    "Bill Gates", "Warren Buffett", "Charlie Munger",
    "Isaac Newton", "Galileo Galilei", "Nikola Tesla",
    "Thomas Edison", "Leonardo da Vinci", "Michelangelo",
    "Shakespeare", "Jane Austen", "Virginia Woolf",
    "Toni Morrison", "Maya Angelou", "Frida Kahlo",
    "Katherine Johnson", "Mae Jemison", "Sally Ride",
    "Neil Armstrong", "Buzz Aldrin", "Yuri Gagarin",
    "Carl Sagan", "Stephen Hawking", "Brian Greene",
    "Rachel Carson", "Jane Goodall", "David Attenborough",
    "Nelson Mandela", "Mahatma Gandhi", "Martin Luther King Jr.",
    "Malala Yousafzai", "Ruth Bader Ginsburg", "Thurgood Marshall",
    "Eleanor Roosevelt", "Julia Child", "Frederick Douglass",
    "Harriet Tubman", "Sojourner Truth", "Ida B. Wells",
    "Mao Zedong", "Winston Churchill", "Charles de Gaulle",
    "Napoleon", "Cleopatra", "Julius Caesar",
    "Alexander the Great", "Confucius", "Laozi",
    "Muhammad Ali", "Pelé", "Serena Williams",
    "Miles Davis", "Aretha Franklin", "Prince",
]

ASPECTS = [
    "physics", "politics", "chronology", "culture", "economics",
    "military", "biology", "chemistry", "technology", "social",
    "legal", "architectural", "artistic", "philosophical",
    "religious", "linguistic", "demographic", "educational",
    "environmental", "diplomatic", "scientific", "literary",
    "musical", "cinematic", "culinary", "medical",
    "psychological", "neurological", "ecological", "geological",
    "astronomical", "mathematical", "computational", "logical",
    "cryptographic", "statistical", "evolutionary", "genetic",
    "morphological", "cellular", "quantum", "thermodynamic",
    "electromagnetic", "optical", "acoustic", "mechanical",
    "ethical", "moral", "aesthetic", "epistemological",
    "metaphysical", "ontological", "sociological", "anthropological",
    "archaeological", "historiographical", "pedagogical",
    "urban", "rural", "colonial", "post-colonial",
    "revolutionary", "counterrevolutionary", "reformist",
    "economic-policy", "monetary", "fiscal",
]

LANDMARKS = [
    # SF
    "Caltrain", "the Ferry Building", "Transamerica Pyramid",
    "City Hall", "Coit Tower", "the Embarcadero", "Dolores Park",
    "Union Square", "Fisherman's Wharf", "Alcatraz",
    "the Golden Gate Bridge", "the Bay Bridge",
    "the Painted Ladies", "Castro Theatre", "Lombard Street",
    "Cable Car Museum", "Pier 39", "the Presidio",
    "the Palace of Fine Arts", "Yerba Buena Gardens",
    "the de Young Museum", "the Legion of Honor",
    "SFMOMA", "the Exploratorium", "the Asian Art Museum",
    "Salesforce Tower", "Chase Center", "Oracle Park",
    # Peninsula
    "Stanford", "HP Garage", "the Dish", "Filoli",
    "Hoover Tower", "Memorial Church",
    "Shoreline Park", "Google's Bayview campus",
    "Apple Park", "Nvidia Endeavor",
    # East Bay
    "UC Berkeley campus", "Sather Tower", "the Berkeley Rose Garden",
    "Jack London Square", "the Oakland Temple", "Lake Merritt",
    "Redwood Regional Park", "Mount Diablo",
    # Wider California
    "Yosemite Valley", "Muir Woods", "Big Sur",
    "the Golden Gate Park", "Point Reyes",
    "Mount Tamalpais", "Tahoe", "Santa Cruz Boardwalk",
    # Generic urban landmarks
    "the train station", "the main library", "City Hall",
    "the public market", "the post office",
    "the town square", "the waterfront",
]

POIS = [
    "bars", "coffee shops", "restaurants", "bookstores",
    "parks", "museums", "gyms", "grocery stores", "pharmacies",
    "gas stations", "hotels", "theaters", "galleries",
    "breweries", "wineries", "distilleries", "cocktail bars",
    "dive bars", "sports bars", "rooftop bars", "tiki bars",
    "speakeasies", "cafes", "bakeries", "ice cream shops",
    "dessert shops", "Thai restaurants", "Mexican restaurants",
    "Japanese restaurants", "Chinese restaurants", "Italian restaurants",
    "Indian restaurants", "Korean BBQ", "ramen shops",
    "sushi spots", "taco trucks", "pizza places",
    "farmers markets", "coworking spaces", "libraries",
    "record stores", "vinyl shops", "comic shops",
    "yoga studios", "climbing gyms", "dance studios",
    "music venues", "jazz clubs", "comedy clubs",
    "art studios", "photography galleries", "printmaking shops",
    "bike shops", "running stores", "outdoor gear stores",
    "thrift stores", "vintage shops", "consignment shops",
    "laundromats", "dry cleaners", "barber shops",
    "hair salons", "nail salons", "tattoo parlors",
    "playgrounds", "dog parks", "skate parks",
    "tennis courts", "basketball courts", "pickleball courts",
    "ATMs", "banks", "electric charging stations",
    "bike racks", "parking garages", "public restrooms",
]

COMPARE_PAIRS = [
    # Historical events
    ("World War I", "World War II"),
    ("the French Revolution", "the American Revolution"),
    ("the Roman Empire", "the Byzantine Empire"),
    ("the Cold War", "World War II"),
    ("the Meiji Restoration", "the French Revolution"),
    ("the Cuban Missile Crisis", "the Berlin Crisis"),
    ("the fall of Rome", "the fall of Constantinople"),
    ("Ancient Greece", "Ancient Rome"),
    ("the Aztec Empire", "the Inca Empire"),
    ("the Magna Carta", "the US Constitution"),
    ("the Reformation", "the Counter-Reformation"),
    # Science / biology
    ("photosynthesis", "cellular respiration"),
    ("mitosis", "meiosis"),
    ("DNA", "RNA"),
    ("CRISPR", "TALENs"),
    ("bacteria", "viruses"),
    ("fission", "fusion"),
    ("general relativity", "quantum mechanics"),
    ("Rayleigh scattering", "Mie scattering"),
    ("red giants", "white dwarfs"),
    ("neutron stars", "black holes"),
    ("weather", "climate"),
    # Tech / languages
    ("Python", "Rust"),
    ("Python", "Go"),
    ("JavaScript", "TypeScript"),
    ("React", "Vue"),
    ("PostgreSQL", "MySQL"),
    ("iOS", "Android"),
    ("Swift", "Kotlin"),
    ("MLX", "PyTorch"),
    ("llama.cpp", "vLLM"),
    # Companies
    ("Tesla", "Toyota"),
    ("Amazon", "Walmart"),
    ("Apple", "Microsoft"),
    ("Google", "Meta"),
    ("Netflix", "Disney+"),
    ("Uber", "Lyft"),
    ("Stripe", "Square"),
    # People
    ("Elon Musk", "Jeff Bezos"),
    ("Einstein", "Newton"),
    ("Darwin", "Mendel"),
    ("Lincoln", "Grant"),
    ("Jefferson", "Madison"),
    ("Napoleon", "Julius Caesar"),
    ("Gandhi", "MLK Jr."),
    ("Shakespeare", "Marlowe"),
    # Philosophies
    ("capitalism", "socialism"),
    ("empiricism", "rationalism"),
    ("determinism", "free will"),
    ("stoicism", "epicureanism"),
    # Places
    ("San Francisco", "New York"),
    ("Stanford", "MIT"),
    ("Silicon Valley", "Route 128"),
    ("London", "Paris"),
    ("Tokyo", "Seoul"),
    ("Rome", "Athens"),
]


def _fill(tmpl: str) -> str:
    """Fill every placeholder in `tmpl` with a randomly sampled value.
    Unknown placeholders are left literal — catches typos at review
    time instead of silently producing mangled queries."""
    fill: dict[str, str] = {}
    if "{place}" in tmpl:
        fill["place"] = random.choice(PLACES)
    if "{topic}" in tmpl and "{topic_a}" not in tmpl:
        fill["topic"] = random.choice(TOPICS)
    if "{topic_a}" in tmpl:
        a, b = random.choice(COMPARE_PAIRS)
        fill["topic_a"], fill["topic_b"] = a, b
    if "{topic_b}" in tmpl and "topic_b" not in fill:
        fill["topic_b"] = random.choice(TOPICS)
    if "{person}" in tmpl:
        fill["person"] = random.choice(PEOPLE)
    if "{poi}" in tmpl:
        fill["poi"] = random.choice(POIS)
    if "{landmark}" in tmpl:
        fill["landmark"] = random.choice(LANDMARKS)
    if "{aspect}" in tmpl:
        fill["aspect"] = random.choice(ASPECTS)
    if "{subtype}" in tmpl:
        fill["subtype"] = random.choice(POIS).rstrip("s")
    try:
        return tmpl.format(**fill)
    except KeyError:
        return tmpl   # leave literal; caller can filter later


def sample_query() -> tuple[str, str]:
    """Sample a (category, query). Categories are weighted — exploration
    / learn_topic / discuss_aspect get more slots because that's where
    the student has the most to learn."""
    weights = [c["weight"] for c in SEED_CATEGORIES.values()]
    names = list(SEED_CATEGORIES.keys())
    category = random.choices(names, weights=weights, k=1)[0]
    opener = random.choice(SEED_CATEGORIES[category]["openers"])
    return category, _fill(opener)


def sample_followup(category: str) -> str:
    """Follow-up query for multi-turn generation. Falls back to a
    generic follow-up if the category has none defined."""
    opts = SEED_CATEGORIES.get(category, {}).get("followups") or [
        "tell me more", "what else", "any other details",
    ]
    return _fill(random.choice(opts))


# ----------------------------------------------------------------------
# Output parsing. Extracts the three code fences the teacher should emit.
# ----------------------------------------------------------------------
FENCE_RE = re.compile(
    r"```(?P<tag>tool_call|tool_response|assistant_reply)\s*\n"
    r"(?P<body>.*?)\n```",
    re.DOTALL,
)


@dataclass
class Trajectory:
    user: str
    tool_call_json: str
    tool_response_json: str
    assistant_reply: str


def parse_teacher_output(raw: str) -> Optional[Trajectory]:
    parts: dict[str, str] = {}
    for m in FENCE_RE.finditer(raw):
        parts[m.group("tag")] = m.group("body").strip()
    if "tool_call" not in parts or "assistant_reply" not in parts:
        return None
    # tool_response is optional (some scenarios are tool-free — but
    # we're driving most queries toward tool use); if absent, set
    # empty object so downstream consumers have something to fill in.
    try:
        json.loads(parts["tool_call"])
    except Exception:
        return None
    if "tool_response" in parts:
        try:
            json.loads(parts["tool_response"])
        except Exception:
            return None
    return Trajectory(
        user="",  # set by caller
        tool_call_json=parts["tool_call"],
        tool_response_json=parts.get("tool_response", "{}"),
        assistant_reply=parts["assistant_reply"],
    )


# ----------------------------------------------------------------------
# Training-data format. mlx_lm.lora expects JSONL with a `messages`
# array. We encode each example as 4 messages: system preamble, user
# query, assistant tool_call, tool response, assistant final reply.
# The student learns to emit the tool_call after the user turn, then
# the final reply after the tool response. Same shape as our iOS
# prompt-building code produces at inference time.
# ----------------------------------------------------------------------
_LOC_POOL = [
    (37.8050, -122.4100), (37.5124, -122.2606), (37.7793, -122.4193),
    (37.4419, -122.1430), (37.8716, -122.2727), (37.3382, -121.8863),
    (37.7609, -122.4350), (37.6879, -122.4702),
]


def _eval_preamble() -> str:
    """Eval-matched preamble: SYSTEM_PREAMBLE + tool block, optionally
    currentLocation. Loaded on demand to avoid import cycle if eval.py
    unavailable."""
    try:
        sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/"
                            "tools/llama-smoke")
        from eval import SYSTEM_PREAMBLE, _build_tool_block
    except Exception:
        # Fallback: minimal preamble when llama-smoke isn't importable.
        return (
            "You are a helpful assistant running on a phone in offline "
            "mode. You have access to an offline Wikipedia + "
            "OpenStreetMap index via the tools listed below. Prefer "
            "tools over guessing. Keep replies concise."
        )
    pre = SYSTEM_PREAMBLE + "\n" + _build_tool_block()
    if random.random() < 0.5:
        lat, lon = random.choice(_LOC_POOL)
        pre += f"\n\ncurrentLocation: lat={lat} lon={lon}"
    return pre


def trajectory_to_jsonl(t: Trajectory) -> dict[str, Any]:
    """Render a trajectory with the eval-harness preamble folded into
    the first user message. Must byte-match what the iOS Gemma3Template
    emits at inference (Gemma 3 has no `system` role — folded into the
    first user turn). Training on a different prompt distribution
    silently regresses on eval."""
    preamble = _eval_preamble()
    return {
        "messages": [
            {"role": "user", "content":
                preamble + "\n\nUser query:\n" + t.user},
            {"role": "assistant", "content":
                f"```tool_call\n{t.tool_call_json}\n```"},
            {"role": "user", "content":
                "[TOOL_RESPONSE]\n" + t.tool_response_json},
            {"role": "assistant", "content": t.assistant_reply},
        ]
    }


# ----------------------------------------------------------------------
# Async generation loop.
# ----------------------------------------------------------------------
def trajectory_from_json_obj(obj: dict, query: str) -> Optional[Trajectory]:
    """Build a Trajectory from the structured-output JSON the teacher
    emits when `response_format` is set. Guaranteed-valid shape thanks
    to schema enforcement, but we still defensively coerce + stringify
    in case the server (LM Studio, llama-server) doesn't fully enforce
    every schema detail."""
    try:
        tc = obj.get("tool_call") or {}
        tr = obj.get("tool_response") or {}
        ar = obj.get("assistant_reply") or ""
        # Re-serialise tool_call + tool_response back to JSON strings
        # so the downstream JSONL writer gets a consistent interface.
        if not isinstance(tc, dict) or "function" not in tc:
            return None
        if not isinstance(ar, str) or not ar.strip():
            return None
        return Trajectory(
            user=query,
            tool_call_json=json.dumps(tc, ensure_ascii=False),
            tool_response_json=json.dumps(tr, ensure_ascii=False),
            assistant_reply=ar.strip(),
        )
    except Exception:
        return None


_FAIL_LOG_PATH: Optional[Path] = None


def _log_fail(query: str, reason: str, raw: str, extra: Optional[dict] = None) -> None:
    if _FAIL_LOG_PATH is None:
        return
    rec = {"query": query, "reason": reason, "raw": raw[:2000]}
    if extra:
        rec.update(extra)
    try:
        with _FAIL_LOG_PATH.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def generate_one(
    client: AsyncOpenAI, model: str, query: str,
    temperature: float, max_tokens: int,
    no_think: bool,
) -> Optional[Trajectory]:
    # `/no_think` is a Qwen 3.x directive to skip visible CoT. Harmless
    # on non-thinking models; crucial on Qwen 3.6 which otherwise
    # burns every token on `reasoning_content`.
    user_content = f"{query}\n\n/no_think" if no_think else query
    # Structured output via JSON schema — LM Studio + llama.cpp enforce
    # the schema at sample time. Model CANNOT emit malformed output
    # and we never need to fence-parse ambiguous text.
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "trajectory",
            "strict": True,
            "schema": TRAJECTORY_SCHEMA,
        },
    }
    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as e:
        print(f"!! api error for {query!r}: {e}", file=sys.stderr)
        _log_fail(query, "api_error", "", {"err": str(e)})
        return None
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", "") or ""
    finish = resp.choices[0].finish_reason or ""
    raw = content or reasoning
    if not raw:
        _log_fail(query, "empty_response", "", {"finish": finish,
                  "has_reasoning": bool(reasoning)})
        return None
    try:
        obj = json.loads(raw)
    except Exception as e:
        # Fallback: some servers wrap JSON in markdown despite
        # structured-output mode. Try the old fence parser.
        traj = parse_teacher_output(raw)
        if traj is None:
            _log_fail(query, "json_and_fence_failed", raw,
                      {"finish": finish, "err": str(e),
                       "used_reasoning": not content and bool(reasoning)})
            return None
        traj.user = query
        return traj
    traj = trajectory_from_json_obj(obj, query)
    if traj is None:
        _log_fail(query, "bad_schema", raw, {"finish": finish})
    return traj


async def prewarm(
    client: AsyncOpenAI, model: str, max_tokens: int,
) -> None:
    """Send one request to fill LM Studio's KV cache with the system
    prompt. Subsequent concurrent calls reuse those prefill tokens
    automatically (llama.cpp prefix cache). Without prewarm the
    first few concurrent requests all re-compute the system prompt
    in parallel and fight for GPU time."""
    print("pre-warming system-prompt KV cache...")
    t0 = time.perf_counter()
    try:
        await client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=32,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": "echo: ready"},
            ],
        )
        print(f"  warm in {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        print(f"  prewarm failed (proceeding anyway): {e}")


async def run(
    client: AsyncOpenAI, model: str, n: int, concurrency: int,
    temperature: float, max_tokens: int, out_path: Path,
    no_think: bool = True,
) -> None:
    # Resumable — count existing lines, skip that many queries so a
    # re-run picks up where we left off.
    done = 0
    if out_path.exists():
        with out_path.open() as fh:
            done = sum(1 for _ in fh)
        print(f"resuming: {done} examples already on disk")

    await prewarm(client, model, max_tokens)

    sampled: list[tuple[str, str]] = [sample_query() for _ in range(n)]
    queries: list[str] = [q for (_c, q) in sampled][done:]
    if not queries:
        print("already done")
        return

    sem = asyncio.Semaphore(concurrency)
    out_fh = out_path.open("a")
    try:
        t0 = time.perf_counter()
        written = 0
        failed = 0

        async def worker(q: str) -> None:
            nonlocal written, failed
            async with sem:
                traj = await generate_one(
                    client, model, q, temperature, max_tokens,
                    no_think=no_think)
            if traj is None:
                failed += 1
                return
            out_fh.write(json.dumps(trajectory_to_jsonl(traj)) + "\n")
            out_fh.flush()
            written += 1
            if written % 10 == 0:
                dt = time.perf_counter() - t0
                rate = written / dt
                eta = (len(queries) - written) / max(rate, 0.01)
                print(f"  {done+written}/{done+len(queries)} written "
                      f"· {rate:.1f}/s · ETA {eta:.0f}s · fails={failed}")

        await asyncio.gather(*[worker(q) for q in queries])
    finally:
        out_fh.close()
    dt = time.perf_counter() - t0
    print(f"done: {written} written · {failed} failed · {dt:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1",
                    help="LM Studio (or other OpenAI-compatible) endpoint")
    ap.add_argument("--api-key", default="lm-studio",
                    help="LM Studio ignores the key but the SDK needs one")
    ap.add_argument("--model", default="qwen3.6-27b",
                    help="Model tag as shown in LM Studio — or any "
                         "string if LM Studio auto-routes to the loaded model")
    ap.add_argument("--n", type=int, default=100,
                    help="Number of examples to generate total")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Max concurrent requests (LM Studio default "
                         "is single-stream; Settings → Developer bumps it)")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="Higher = more diverse trajectories; too high "
                         "breaks JSON format. 0.8 is a good middle.")
    ap.add_argument("--max-tokens", type=int, default=640,
                    help="Observed gen envelope: avg 300, max 474 "
                         "tokens (tool_call + tool_response + reply). "
                         "640 catches outliers (5+ near_place results) "
                         "without wasting context budget on loaded "
                         "models with small context — LM Studio's "
                         "default ~2k causes 'context exceeded' at "
                         "max_tokens=1024.")
    ap.add_argument("--no-think", action="store_true", default=True,
                    help="Append /no_think to each user query. MUCH "
                         "faster on Qwen 3.x (10-20x) and the student "
                         "wouldn't see the CoT at inference time "
                         "anyway. Default on. Disable with --think.")
    ap.add_argument("--think", action="store_false", dest="no_think",
                    help="Let Qwen 3.x think visibly. Slower but "
                         "sometimes produces higher-quality critiques.")
    ap.add_argument("--out", default="train.jsonl",
                    help="Output JSONL path (append-only, resumable)")
    ap.add_argument("--fail-log", default=None,
                    help="If set, write per-failure diagnostics here "
                         "(query, raw response, reason) for post-hoc tuning")
    ap.add_argument("--boost", default=None,
                    help="Multiply specific category weights, e.g. "
                         "'narrate=10,current_place=3,directions=3'. "
                         "Used to top up under-represented tool calls "
                         "without regenerating the whole set.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global _FAIL_LOG_PATH
    if args.fail_log:
        _FAIL_LOG_PATH = Path(args.fail_log)

    if args.boost:
        for pair in args.boost.split(","):
            k, _, v = pair.partition("=")
            k = k.strip()
            if k in SEED_CATEGORIES:
                SEED_CATEGORIES[k]["weight"] *= float(v)
                print(f"  boost {k} -> weight {SEED_CATEGORIES[k]['weight']}")
            else:
                print(f"  WARN: unknown category {k!r}", file=sys.stderr)

    random.seed(args.seed)
    client = AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key)
    asyncio.run(run(
        client, args.model, args.n, args.concurrency,
        args.temperature, args.max_tokens, Path(args.out),
        no_think=args.no_think))


if __name__ == "__main__":
    main()
