"""Generate 2-turn chain training trajectories.

The existing single-turn generator produced zero `get_article_section`
calls because it only samples single-query patterns. Chain scenarios
in the eval suite expect the model to do `article_overview` on turn 1
then drill into a specific section on turn 2 via `get_article_section`.

Each call produces:
  turn 1: user "tell me about X"       → article_overview
  turn 2: user "tell me about <aspect>" → get_article_section

Output is JSONL in the eval-matched preamble format (system+tool_block
folded into first user message). Reuses the teacher + structured-JSON
schema from generate.py.
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, "/Users/jasontitus/experiments/mcpzim/tools/llama-smoke")

from openai import AsyncOpenAI
from eval import SYSTEM_PREAMBLE, _build_tool_block

from generate import TOPICS, PEOPLE, ASPECTS, _log_fail


# ----------------------------------------------------------------------
# Teacher system prompt — explains the 2-turn format we want.
# ----------------------------------------------------------------------
TEACHER_SYS = """\
You are a teacher generating multi-turn training data. Produce ONE
JSON object modelling a 2-turn dialogue where a small on-device
assistant:

  Turn 1: calls `article_overview(title)` on the topic, receives a
    synthetic `{title, lead, available_sections}` response, then
    writes a concise reply (3-5 sentences) that surfaces key points
    AND subtly teases the available sections.

  Turn 2: receives a user follow-up asking about a specific aspect.
    Calls `get_article_section(title, section)` where `section` is
    chosen from the `available_sections` produced in turn 1. Receives
    a synthetic `{section_body}` response, then writes a focused
    reply (2-4 sentences) drawing only from section_body.

The JSON has keys: tool_call_1, tool_response_1, reply_1, followup,
tool_call_2, tool_response_2, reply_2. All tool_* objects should look
realistic (real article titles, 4-8 plausible section names, etc.).
Keep invented content factually plausible."""


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_call_1":    {"type": "object",
                            "properties": {"function": {"type": "string"},
                                           "parameters": {"type": "object"}},
                            "required": ["function", "parameters"]},
        "tool_response_1": {"type": "object"},
        "reply_1":         {"type": "string"},
        "followup":        {"type": "string"},
        "tool_call_2":    {"type": "object",
                            "properties": {"function": {"type": "string"},
                                           "parameters": {"type": "object"}},
                            "required": ["function", "parameters"]},
        "tool_response_2": {"type": "object"},
        "reply_2":         {"type": "string"},
    },
    "required": ["tool_call_1", "tool_response_1", "reply_1",
                 "followup", "tool_call_2", "tool_response_2", "reply_2"],
}


# ----------------------------------------------------------------------
# Seed generation.
# ----------------------------------------------------------------------
OPENER_TEMPLATES = [
    "tell me about {topic}",
    "what is {topic}",
    "explain {topic}",
    "summarize {topic} for me",
    "give me a quick intro to {topic}",
    "who was {person}",
    "history of {topic}",
    "what happened in {topic}",
]


def sample_query() -> str:
    t = random.choice(OPENER_TEMPLATES)
    if "{person}" in t:
        return t.format(person=random.choice(PEOPLE))
    return t.format(topic=random.choice(TOPICS))


LOC_POOL = [
    (37.8050, -122.4100), (37.5124, -122.2606), (37.7793, -122.4193),
    (37.4419, -122.1430), (37.8716, -122.2727), (37.3382, -121.8863),
    (37.7609, -122.4350), (37.6879, -122.4702),
]


def _eval_preamble() -> str:
    pre = SYSTEM_PREAMBLE + "\n" + _build_tool_block()
    if random.random() < 0.5:
        lat, lon = random.choice(LOC_POOL)
        pre += f"\n\ncurrentLocation: lat={lat} lon={lon}"
    return pre


def chain_to_messages(query: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Render the 2-turn chain as an 8-message student trajectory
    in eval-matched preamble format."""
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
            {"role": "assistant", "content": obj["reply_1"].strip()},
            {"role": "user", "content": obj["followup"].strip()},
            {"role": "assistant", "content":
                f"```tool_call\n{tc2}\n```"},
            {"role": "user", "content":
                "[TOOL_RESPONSE]\n" + tr2},
            {"role": "assistant", "content": obj["reply_2"].strip()},
        ]
    }


# ----------------------------------------------------------------------
# Async generation.
# ----------------------------------------------------------------------
async def generate_one(
    client: AsyncOpenAI, model: str, query: str,
    temperature: float, max_tokens: int,
) -> Optional[dict[str, Any]]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "chain_trajectory", "strict": True, "schema": SCHEMA,
        },
    }
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format,
            messages=[
                {"role": "system", "content": TEACHER_SYS},
                {"role": "user",
                 "content": f"Seed user query: {query!r}\n\nProduce the "
                             f"2-turn JSON."},
            ],
        )
    except Exception as e:
        print(f"!! api error: {e}", file=sys.stderr)
        return None
    raw = resp.choices[0].message.content or ""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    # Defensive: require tool names match expected pattern.
    if obj.get("tool_call_1", {}).get("function") != "article_overview":
        return None
    if obj.get("tool_call_2", {}).get("function") != "get_article_section":
        return None
    if not obj.get("reply_1", "").strip() or not obj.get("reply_2", "").strip():
        return None
    return obj


async def run(client: AsyncOpenAI, model: str, n: int, concurrency: int,
              temperature: float, max_tokens: int, out_path: Path) -> None:
    done = 0
    if out_path.exists():
        with out_path.open() as fh:
            done = sum(1 for _ in fh)
        print(f"resuming: {done} chains already on disk")

    queries = [sample_query() for _ in range(n)][done:]
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
                obj = await generate_one(client, model, q,
                                          temperature, max_tokens)
            if obj is None:
                failed += 1
                return
            out_fh.write(json.dumps(chain_to_messages(q, obj)) + "\n")
            out_fh.flush()
            written += 1
            if written % 5 == 0:
                dt = time.perf_counter() - t0
                rate = written / dt
                eta = (len(queries) - written) / max(rate, 0.01)
                print(f"  {done+written}/{done+len(queries)} · {rate:.2f}/s "
                      f"· ETA {eta:.0f}s · fails={failed}")

        await asyncio.gather(*[worker(q) for q in queries])
    finally:
        out_fh.close()
    dt = time.perf_counter() - t0
    print(f"done: {written} written · {failed} failed · {dt:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--api-key",  default="lm-studio")
    ap.add_argument("--model",    default="gemma-3-27b-it")
    ap.add_argument("--n",        type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens",  type=int, default=1024,
                    help="Chains emit 2 tool calls + 2 responses + 2 replies "
                         "— larger output envelope than single-turn.")
    ap.add_argument("--out", default="train_chains.jsonl")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    random.seed(args.seed)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    asyncio.run(run(client, args.model, args.n, args.concurrency,
                    args.temperature, args.max_tokens, Path(args.out)))


if __name__ == "__main__":
    main()
