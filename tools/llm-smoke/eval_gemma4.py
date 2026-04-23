#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Tool-call eval for Gemma 4 via mlx-vlm. mlx-lm can't load Gemma 4
# weights (140 unexpected layer params because Gemma 4's shared-KV
# design puts k_proj/v_proj fields in the safetensors that the mlx-lm
# Gemma4 model doesn't expect). mlx-vlm handles them natively. The
# scenarios + tool schema + extraction logic are shared with eval.py.

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import mlx.core as mx
from mlx_vlm import load, stream_generate, apply_chat_template

from eval import (CASES, TOOLS, SYSTEM, extract_tool_call)


def build_prompt_gemma4(processor, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM, "tools": json.dumps(TOOLS,
                                                                  ensure_ascii=False)},
        {"role": "user", "content": user_text},
    ]
    # mlx-vlm's apply_chat_template takes (processor, config, prompt, ...)
    # — but we can also call processor.tokenizer.apply_chat_template
    # directly which accepts standard HF kwargs.
    tokenizer = getattr(processor, "tokenizer", None) or processor
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
        )
        if any(t["function"]["name"] in rendered for t in TOOLS):
            return rendered
    except Exception:
        pass
    # Fallback: inline JSON in system.
    sys_plus = (
        SYSTEM
        + "\n\n# Tools\nEmit a single call as:\n<tool_call>\n"
        + '{"name": <function-name>, "arguments": <args-as-json-object>}\n'
        + "</tool_call>\n\n"
        + "\n".join(json.dumps(t) for t in TOOLS)
    )
    messages[0]["content"] = sys_plus
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_case(model, processor, case, max_tokens: int = 200) -> dict:
    prompt = build_prompt_gemma4(processor, case.prompt)
    tokenizer = getattr(processor, "tokenizer", None) or processor
    prompt_tokens = tokenizer.encode(prompt)
    t_start = time.perf_counter()
    first_tok_at: Optional[float] = None
    decoded = []
    decoded_n = 0
    for resp in stream_generate(
        model, processor, prompt,
        max_tokens=max_tokens, temperature=0.0,
    ):
        if first_tok_at is None:
            first_tok_at = time.perf_counter()
        text = getattr(resp, "text", None) if hasattr(resp, "text") else None
        if text is None and isinstance(resp, str):
            text = resp
        if text:
            decoded.append(text)
            decoded_n += 1
    t_end = time.perf_counter()
    ft = first_tok_at or t_end
    output = "".join(decoded)

    call = extract_tool_call(output)
    name_ok = bool(call and call[0] in case.expected_tools)
    args_ok = False
    if call:
        args_lower = json.dumps(call[1]).lower()
        args_ok = all(s.lower() in args_lower for s in case.expected_args_contain)

    return {
        "name": case.name or case.prompt[:30],
        "prompt": case.prompt,
        "expected": "/".join(case.expected_tools),
        "got": call[0] if call else None,
        "args": call[1] if call else None,
        "name_ok": name_ok,
        "args_ok": args_ok,
        "prompt_tokens": len(prompt_tokens),
        "prefill_s": ft - t_start,
        "decode_tokens": decoded_n,
        "decode_s": max(t_end - ft, 1e-6),
        "output": output,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["all", "one", "bench"])
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--show-output", action="store_true")
    args = ap.parse_args()

    cases = CASES if args.mode == "all" else [CASES[0]]
    print(f"== gemma4-eval ==")
    print(f"model: {args.model_id}")
    print(f"cases: {len(cases)}\n")

    t_load = time.perf_counter()
    print("loading…")
    model, processor = load(args.model_id)
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.2f}s\n")

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"--- case {idx}/{len(cases)} {case.name}: {case.prompt}")
        r = run_case(model, processor, case, max_tokens=args.max_tokens)
        results.append(r)
        mark = "PASS" if (r["name_ok"] and r["args_ok"]) else (
            "PART" if r["name_ok"] else "FAIL"
        )
        print(f"  [{mark}] expected={r['expected']} got={r['got']}")
        if r["args"] is not None:
            print(f"  args: {json.dumps(r['args'])[:200]}")
        p_tps = r["prompt_tokens"] / max(r["prefill_s"], 1e-6)
        d_tps = r["decode_tokens"] / max(r["decode_s"], 1e-6)
        print(f"  prefill: {r['prefill_s']*1000:.0f}ms "
              f"({r['prompt_tokens']} tok → {p_tps:.0f} tok/s), "
              f"decode: {r['decode_tokens']} tok in {r['decode_s']:.2f}s "
              f"({d_tps:.0f} tok/s)")
        if args.show_output or args.mode == "one":
            print(f"  --- output ---\n{r['output']}\n  --- end ---")

    passed = sum(1 for r in results if r["name_ok"] and r["args_ok"])
    names_ok = sum(1 for r in results if r["name_ok"])
    if results:
        mean_prefill = sum(r["prompt_tokens"] / max(r["prefill_s"], 1e-6)
                           for r in results) / len(results)
        mean_decode = sum(r["decode_tokens"] / max(r["decode_s"], 1e-6)
                          for r in results) / len(results)
    else:
        mean_prefill = mean_decode = 0.0

    print("\n== summary ==")
    print(f"model:        {args.model_id}")
    print(f"load:         {load_s:.2f}s")
    print(f"tool name:    {names_ok}/{len(results)}")
    print(f"full pass:    {passed}/{len(results)}")
    print(f"mean prefill: {mean_prefill:.0f} tok/s")
    print(f"mean decode:  {mean_decode:.0f} tok/s")
    print("\nper-case:")
    for r in results:
        ok = r["name_ok"] and r["args_ok"]
        mark = "✓" if ok else ("∼" if r["name_ok"] else "✗")
        got = r["got"] or "<none>"
        d_tps = r["decode_tokens"] / max(r["decode_s"], 1e-6)
        print(f"  {mark}  {r['prompt'][:38]:<40}"
              f"  got={got[:22]:<22}"
              f"  prefill={r['prefill_s']*1000:>5.0f}ms"
              f"  decode={d_tps:.0f} tok/s")


if __name__ == "__main__":
    main()
