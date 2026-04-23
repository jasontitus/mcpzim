#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Cross-family tool-calling eval harness for MLX on-device candidates.
#
# Usage:
#   source tools/llm-smoke/.venv/bin/activate
#   python tools/llm-smoke/eval.py mlx-community/Phi-4-mini-instruct-4bit
#   python tools/llm-smoke/eval.py mlx-community/gemma-3-4b-it-4bit one
#   python tools/llm-smoke/eval.py mlx-community/NVIDIA-Nemotron-Nano-4B-v1.1-4bit bench
#
# What it measures per model:
#   - load time (cold from HF cache, warm if already there)
#   - prompt tokens / first-token latency → prefill tok/s
#   - decode tok/s
#   - tool-call name match + required-arg substring match
#
# What it does NOT test (yet):
#   - multi-turn KV cache reuse (single-turn only here)
#   - long-context preamble (the app's real 7k-token system; we use a trimmed
#     preamble so we isolate the model's tool-picking, not the prefill speed)
#
# Each case is run with `temperature=0` so output is deterministic. We use
# `tokenizer.apply_chat_template(..., tools=...)` when it exists so each
# family's native tool-schema format is used verbatim (Phi-4-mini's <|tool|>
# wrap, Nemotron's JSON-in-system, Gemma 3's... we'll see).

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

# ---------------------------------------------------------------------------
# Tools

def _fn(name, description, params_schema):
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": params_schema}}


# Matches ios/MCPZimEval/EvalHarness.swift tool surface (trimmed to what
# scenarios 1–9 need). Keep parameter names + descriptions aligned with the
# app's MCPToolAdapter — models that saw these exact phrasings during our
# eval on iOS will behave comparably here.
TOOLS = [
    _fn("near_places",
        "Find places (restaurants, parks, etc.) within a radius of the user's current GPS location.",
        {"type": "object",
         "properties": {
             "kinds": {"type": "array", "items": {"type": "string"},
                       "description": 'Place categories, e.g. ["restaurant"], ["pizza"], ["park"]'},
             "radius_km": {"type": "number", "description": "Search radius (default 2.0)"},
         },
         "required": ["kinds"]}),
    _fn("near_named_place",
        "Find places of the given kinds near a named location (city, landmark).",
        {"type": "object",
         "properties": {
             "place": {"type": "string", "description": "City or landmark name"},
             "kinds": {"type": "array", "items": {"type": "string"}},
             "radius_km": {"type": "number"},
         },
         "required": ["place", "kinds"]}),
    _fn("nearby_stories",
        "Find Wikipedia-linked places of interest near the user's current GPS location (\"something interesting around me\").",
        {"type": "object",
         "properties": {
             "radius_km": {"type": "number", "description": "Search radius (default 2.0)"},
             "has_wiki": {"type": "boolean", "description": "Only return places with a Wikipedia article"},
         },
         "required": []}),
    _fn("nearby_stories_at_place",
        "Find Wikipedia-linked places of interest near a named city or landmark.",
        {"type": "object",
         "properties": {
             "place": {"type": "string"},
             "radius_km": {"type": "number"},
         },
         "required": ["place"]}),
    _fn("route_from_places",
        "Get driving directions from an origin place name to a destination place name.",
        {"type": "object",
         "properties": {
             "origin": {"type": "string", "description": "Origin place name or 'my location'"},
             "destination": {"type": "string"},
         },
         "required": ["origin", "destination"]}),
    _fn("route_status",
        "Report time/distance remaining on the user's currently-active driving route.",
        {"type": "object", "properties": {}, "required": []}),
    _fn("article_overview",
        "Return a structured overview of a Wikipedia article (intro + selected sections) for broad questions like \"tell me about <X>\".",
        {"type": "object",
         "properties": {
             "title": {"type": "string", "description": "Article title, e.g. \"Palo Alto\""},
         },
         "required": ["title"]}),
    _fn("get_article_section",
        "Retrieve a specific named section of a Wikipedia article.",
        {"type": "object",
         "properties": {
             "title": {"type": "string"},
             "section": {"type": "string"},
         },
         "required": ["title"]}),
    _fn("narrate_article",
        "Return the full body of a Wikipedia article, verbatim, for read-aloud / \"read me the article on X\" requests.",
        {"type": "object",
         "properties": {"title": {"type": "string"}},
         "required": ["title"]}),
    _fn("compare_articles",
        "Compare two Wikipedia topics, summarising similarities / differences / relationships between them.",
        {"type": "object",
         "properties": {
             "a": {"type": "string", "description": "First topic"},
             "b": {"type": "string", "description": "Second topic"},
         },
         "required": ["a", "b"]}),
    _fn("what_is_here",
        "Describe the user's current location (city / neighbourhood / landmarks) based on their GPS coords.",
        {"type": "object", "properties": {}, "required": []}),
]

SYSTEM = (
    "You are MCPZim, an assistant with access to offline Wikipedia and "
    "OpenStreetMap data via the tools listed below. The user's current GPS "
    "location is available to the tools. ALWAYS call the matching tool for "
    "requests about nearby places, routes, or Wikipedia facts — do not "
    "reply that you cannot access real-time data. Tools are the ONLY way "
    "to fulfil these requests; emit a tool call and nothing else."
)


# ---------------------------------------------------------------------------
# Eval set

@dataclass
class Case:
    prompt: str
    # List of acceptable tool names (any one matches). Matches the Swift
    # harness's `toolsCalledAny`.
    expected_tools: list[str]
    # Substrings (case-insensitive) that must ALL appear in the args JSON.
    expected_args_contain: list[str] = field(default_factory=list)
    # Substrings at least one of which must NOT appear in the output
    # (these correspond to the Swift harness's `responseExcludes` but
    # for tool-selection scoring we only check tool name + args).
    name: str = ""


# The 9 scenarios from ios/MCPZimEval/EvalHarness.swift, transplanted to the
# tool-selection layer (we score which tool the model picks, not the final
# response text — that requires a live ZIM/streetzim stub which this harness
# doesn't carry). Kept in the same order so comparisons are 1:1 with iOS.
CASES = [
    Case(
        name="restaurants_in_sf",
        prompt="Are there any good restaurants in San Francisco?",
        expected_tools=["near_named_place"],
        expected_args_contain=["san francisco", "restaurant"],
    ),
    Case(
        name="nearby_stories_here",
        prompt="Tell me something interesting about where I am.",
        expected_tools=["nearby_stories"],
    ),
    Case(
        name="nearby_stories_palo_alto",
        prompt="Tell me some interesting stories from Palo Alto.",
        expected_tools=["nearby_stories_at_place"],
        expected_args_contain=["palo alto"],
    ),
    Case(
        name="tell_me_about_palo_alto",
        prompt="Tell me about Palo Alto.",
        expected_tools=["article_overview"],
        expected_args_contain=["palo alto"],
    ),
    Case(
        name="compare_musk_bezos",
        prompt="How is Elon Musk different from Jeff Bezos?",
        expected_tools=["compare_articles"],
        expected_args_contain=["musk", "bezos"],
    ),
    Case(
        name="relations_us_iran",
        prompt="How have the United States and Iran gotten along historically?",
        expected_tools=["compare_articles", "article_relationship"],
        expected_args_contain=["iran"],
    ),
    Case(
        name="narrate_hp_garage",
        prompt="Please read me the full article about the HP Garage.",
        expected_tools=["narrate_article"],
        expected_args_contain=["hp garage"],
    ),
    Case(
        name="what_is_here_in_sf",
        prompt="Where am I?",
        expected_tools=["what_is_here"],
    ),
    Case(
        name="how_much_longer",
        prompt="How much longer until I get there?",
        expected_tools=["route_status"],
    ),
]


# ---------------------------------------------------------------------------
# Tool-call extraction
#
# Different families emit tool calls in different wrappers:
#   - Phi-4-mini: <|tool_call|>[{...}]<|/tool_call|>  (or bare JSON list)
#   - Qwen/Gemma style in our codebase: <tool_call>{...}</tool_call>
#   - Nemotron / Llama-JSON: <tool_call>...</tool_call> or ```json blocks
#   - Gemma 3: plain JSON? or python fn-call? (check empirically)
#
# Strategy: try each known wrapper, fall back to "first balanced JSON object
# or list with a 'name' key".

TAG_PAIRS = [
    ("<|tool_call|>", "<|/tool_call|>"),
    ("<tool_call>", "</tool_call>"),
    ("<tool>", "</tool>"),
    ("```json", "```"),
    ("```tool_code", "```"),
]


def _extract_between(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    i = text.find(open_tag)
    if i < 0:
        return None
    j = text.find(close_tag, i + len(open_tag))
    if j < 0:
        return text[i + len(open_tag) :]  # truncated before close
    return text[i + len(open_tag) : j]


def _first_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _parse_json_any(snippet: str) -> Optional[dict]:
    s = snippet.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            return obj[0]
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _normalize_call(obj: dict) -> Optional[tuple[str, dict]]:
    """Extract (name, arguments-dict) from the many shapes models emit."""
    if not isinstance(obj, dict):
        return None
    # OpenAI-style: {"type":"function","function":{"name":"...","arguments":...}}
    if "function" in obj and isinstance(obj["function"], dict):
        return _normalize_call(obj["function"])
    # Common: {"name":"...","arguments":{...}}
    name = obj.get("name")
    if not isinstance(name, str):
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


_KNOWN_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def _prefer_valid(candidates: list[tuple[str, dict]]) -> Optional[tuple[str, dict]]:
    """Prefer a candidate whose name is a known tool. Falls back to the
    first candidate (or None) if none match. Picks the LONGEST matching
    candidate so we don't grab a nested object when the outer one is
    the real tool call."""
    valid = [c for c in candidates if c[0] in _KNOWN_TOOL_NAMES]
    if valid:
        return valid[0]
    return candidates[0] if candidates else None


def extract_tool_call(text: str) -> Optional[tuple[str, dict]]:
    # Drop reasoning/scratchpad before parsing. Qwen 3 Thinking, DeepSeek,
    # and Phi-4-mini-reasoning all stream `<think>...</think>` blocks that
    # contain JSON fragments (argument keys, example calls) which would
    # otherwise be picked up by the balanced-JSON fallback.
    text = _THINK_RE.sub("", text)
    candidates: list[tuple[str, dict]] = []

    def _push(obj) -> None:
        call = _normalize_call(obj) if isinstance(obj, dict) else None
        if call is not None:
            candidates.append(call)

    # 1. Try all known wrapper tags (in order, gathering all hits).
    for open_tag, close_tag in TAG_PAIRS:
        i = 0
        while True:
            idx = text.find(open_tag, i)
            if idx < 0:
                break
            body_start = idx + len(open_tag)
            close_idx = text.find(close_tag, body_start)
            body = text[body_start:close_idx] if close_idx >= 0 else text[body_start:]
            i = (close_idx + len(close_tag)) if close_idx >= 0 else len(text)
            obj = _parse_json_any(body)
            if obj is not None:
                _push(obj)
            braced = _first_balanced(body, "{", "}") or _first_balanced(body, "[", "]")
            if braced:
                _push(_parse_json_any(braced))

    picked = _prefer_valid(candidates)
    if picked is not None:
        return picked

    # 2. Fallback: scan for balanced JSON objects with a "name" key anywhere
    #    in the post-`<think>` text. Loop so we find nested or later calls.
    for open_ch, close_ch in [("[", "]"), ("{", "}")]:
        scan = text
        while True:
            braced = _first_balanced(scan, open_ch, close_ch)
            if not braced:
                break
            obj = _parse_json_any(braced)
            if obj is not None:
                _push(obj)
            pos = scan.find(braced)
            if pos < 0:
                break
            scan = scan[pos + len(braced):]
    picked = _prefer_valid(candidates)
    if picked is not None:
        return picked
    # 3. Python-fn-call style (some Gemma variants): name(arg=...)
    m = re.search(r"\b([a-z_][a-z0-9_]+)\s*\(\s*([^)]*)\)", text, re.IGNORECASE)
    if m:
        name = m.group(1)
        # crude key=value parse
        args: dict = {}
        for part in re.split(r",\s*(?=[a-zA-Z_]+\s*=)", m.group(2)):
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = v.strip().strip('"').strip("'")
        return name, args
    return None


# ---------------------------------------------------------------------------
# Run

def build_prompt(tokenizer, user_text: str) -> str:
    """Render a chat-templated prompt with tool schemas.

    Strategy: render twice, once with `tools=TOOLS` and once without; if the
    first render doesn't actually include a tool name, the template ignored
    the kwarg (Phi-4-mini's custom jinja reads `message['tools']`, not the
    top-level `tools=` arg). Fall back to inlining the JSON schema into the
    system message so every family at least *sees* the tools.
    """
    tools_json = json.dumps(TOOLS, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    rendered: Optional[str] = None
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        rendered = None

    needs_manual = rendered is None or not any(
        t["function"]["name"] in (rendered or "") for t in TOOLS
    )

    if not needs_manual:
        return rendered

    # Manual path: attach tools to system message for Phi-4-mini-style jinja,
    # OR append a Qwen/Gemma-style `<tool_call>` envelope so the model knows
    # how to emit the call.
    sys_with_tools = (
        SYSTEM
        + "\n\n# Tools\n"
        + "Emit a single call using this exact format:\n"
        + "<tool_call>\n"
        + '{"name": <function-name>, "arguments": <args-as-json-object>}\n'
        + "</tool_call>\n\n"
        + "Available tools:\n"
        + "\n".join(
            json.dumps({"name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "parameters": t["function"]["parameters"]},
                       ensure_ascii=False)
            for t in TOOLS
        )
    )
    messages_manual = [
        {"role": "system", "content": sys_with_tools, "tools": tools_json},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages_manual, tokenize=False, add_generation_prompt=True
    )


def run_case(model, tokenizer, case: Case, max_tokens: int = 200):
    prompt = build_prompt(tokenizer, case.prompt)
    prompt_tokens = tokenizer.encode(prompt)
    sampler = make_sampler(temp=0.0)
    t_start = time.perf_counter()
    first_token_at: Optional[float] = None
    decoded: list[str] = []
    decoded_tokens = 0
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        if response.text:
            decoded.append(response.text)
            decoded_tokens += 1
    t_end = time.perf_counter()
    output = "".join(decoded)

    if first_token_at is None:
        first_token_at = t_end
    prefill_s = first_token_at - t_start
    decode_s = max(t_end - first_token_at, 1e-6)

    call = extract_tool_call(output)
    name_ok = bool(call and call[0] in case.expected_tools)
    args_ok = False
    if call:
        args_lower = json.dumps(call[1]).lower()
        args_ok = all(sub.lower() in args_lower for sub in case.expected_args_contain)

    return {
        "name": case.name or case.prompt[:30],
        "prompt": case.prompt,
        "expected": "/".join(case.expected_tools),
        "got": call[0] if call else None,
        "args": call[1] if call else None,
        "name_ok": name_ok,
        "args_ok": args_ok,
        "prompt_tokens": len(prompt_tokens),
        "prefill_s": prefill_s,
        "decode_tokens": decoded_tokens,
        "decode_s": decode_s,
        "output": output,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("mode", nargs="?", default="all", choices=["all", "one", "bench"])
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--show-output", action="store_true")
    ap.add_argument("--show-prompt", action="store_true",
                    help="Print the rendered chat template for the first case")
    args = ap.parse_args()

    cases = CASES if args.mode == "all" else [CASES[0]]
    print(f"== llm-smoke ==")
    print(f"model: {args.model_id}")
    print(f"cases: {len(cases)}")
    print(f"max_tokens: {args.max_tokens}\n")

    t_load = time.perf_counter()
    print("loading…")
    model, tokenizer = load(args.model_id)
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.2f}s\n")

    if args.show_prompt and cases:
        print("--- rendered prompt (case 1) ---")
        print(build_prompt(tokenizer, cases[0].prompt))
        print("--- end prompt ---\n")

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"--- case {idx}/{len(cases)} {case.name}: {case.prompt}")
        r = run_case(model, tokenizer, case, max_tokens=args.max_tokens)
        results.append(r)
        mark = "PASS" if (r["name_ok"] and r["args_ok"]) else (
            "PART" if r["name_ok"] else "FAIL"
        )
        print(f"  [{mark}] expected={r['expected']} got={r['got']}")
        if r["args"] is not None:
            print(f"  args: {json.dumps(r['args'])[:200]}")
        prefill_tok_s = r["prompt_tokens"] / max(r["prefill_s"], 1e-6)
        decode_tok_s = r["decode_tokens"] / max(r["decode_s"], 1e-6)
        print(
            f"  prefill: {r['prefill_s']*1000:.0f}ms "
            f"({r['prompt_tokens']} tok → {prefill_tok_s:.0f} tok/s), "
            f"decode: {r['decode_tokens']} tok in {r['decode_s']:.2f}s "
            f"({decode_tok_s:.0f} tok/s)"
        )
        if args.show_output or args.mode == "one":
            print(f"  --- output ---\n{r['output']}\n  --- end ---")

    # Summary.
    passed = sum(1 for r in results if r["name_ok"] and r["args_ok"])
    names_ok = sum(1 for r in results if r["name_ok"])
    mean_prefill = sum(
        r["prompt_tokens"] / max(r["prefill_s"], 1e-6) for r in results
    ) / len(results)
    mean_decode = sum(
        r["decode_tokens"] / max(r["decode_s"], 1e-6) for r in results
    ) / len(results)

    print("\n== summary ==")
    print(f"model:        {args.model_id}")
    print(f"load:         {load_s:.2f}s")
    print(f"tool name:    {names_ok}/{len(results)}")
    print(f"full pass:    {passed}/{len(results)}")
    print(f"mean prefill: {mean_prefill:.0f} tok/s")
    print(f"mean decode:  {mean_decode:.0f} tok/s")
    print("\nper-case:")
    for r in results:
        mark = "✓" if (r["name_ok"] and r["args_ok"]) else (
            "∼" if r["name_ok"] else "✗"
        )
        got = r["got"] or "<none>"
        decode_tok_s = r["decode_tokens"] / max(r["decode_s"], 1e-6)
        print(
            f"  {mark}  {r['prompt'][:38]:<40}"
            f"  got={got[:22]:<22}"
            f"  prefill={r['prefill_s']*1000:>5.0f}ms"
            f"  decode={decode_tok_s:.0f} tok/s"
        )


if __name__ == "__main__":
    main()
