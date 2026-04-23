#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Apples-to-apples Gemma 4 eval. The default HF chat template that
# `processor.tokenizer.apply_chat_template` would apply causes Gemma 4 to
# refuse or ask for clarification instead of emitting tool calls — that's
# why eval_gemma4.py scores 0-1/9. The iOS app does not use that template.
# It uses `MCPZimKit.Gemma4PromptTemplate` + `Gemma4ToolFormat` (the
# Google April-2026 chat template, ported literally). This harness uses
# that format via `gemma4_format.py` so the on-device score is
# approximated here.
#
# Also reuses the same 9 scenarios + tool schemas as eval.py so results
# are directly comparable to Gemma 3 / Qwen 3.

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import mlx.core as mx
from mlx_vlm import load, stream_generate

from eval import CASES, TOOLS, SYSTEM
from gemma4_format import render_transcript, first_call


STOP_MARKERS = ["<turn|>", "<|turn>"]


def build_prompt(user_text: str) -> str:
    return render_transcript(
        system_message=SYSTEM,
        tools=TOOLS,
        user_text=user_text,
    )


def run_case(model, processor, case, max_tokens: int = 300) -> dict:
    prompt = build_prompt(case.prompt)
    tokenizer = getattr(processor, "tokenizer", None) or processor
    prompt_tokens = tokenizer.encode(prompt)

    t_start = time.perf_counter()
    first_tok_at: Optional[float] = None
    decoded: list[str] = []
    decoded_n = 0
    buffer = ""
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
            buffer += text
            # Early-stop on the turn-close marker so we don't keep decoding
            # after the assistant finished — matches ChatSession on iOS.
            if any(m in buffer for m in STOP_MARKERS):
                break
    t_end = time.perf_counter()
    ft = first_tok_at or t_end
    output = "".join(decoded)

    call = first_call(output)
    name_ok = bool(call and call[0] in case.expected_tools)
    args_ok = False
    if call:
        args_lower = json.dumps(call[1], ensure_ascii=False).lower()
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
                    choices=["all", "one"])
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--show-output", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    cases = CASES if args.mode == "all" else [CASES[0]]
    print(f"== gemma4-native-eval ==")
    print(f"model: {args.model_id}")
    print(f"cases: {len(cases)}\n")

    t_load = time.perf_counter()
    print("loading…")
    model, processor = load(args.model_id)
    print(f"loaded in {time.perf_counter() - t_load:.2f}s\n")

    if args.show_prompt and cases:
        rendered = build_prompt(cases[0].prompt)
        print("--- rendered prompt (case 1, truncated to 2500 chars) ---")
        print(rendered[:2500])
        print("--- end prompt ---\n")

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
            print(f"  args: {json.dumps(r['args'], ensure_ascii=False)[:200]}")
        p_tps = r["prompt_tokens"] / max(r["prefill_s"], 1e-6)
        d_tps = r["decode_tokens"] / max(r["decode_s"], 1e-6)
        print(f"  prefill: {r['prefill_s']*1000:.0f}ms "
              f"({r['prompt_tokens']} tok → {p_tps:.0f} tok/s), "
              f"decode: {r['decode_tokens']} tok in {r['decode_s']:.2f}s "
              f"({d_tps:.0f} tok/s)")
        if args.show_output or args.mode == "one":
            print(f"  --- output (first 500 chars) ---\n{r['output'][:500]}\n  --- end ---")

    passed = sum(1 for r in results if r["name_ok"] and r["args_ok"])
    names_ok = sum(1 for r in results if r["name_ok"])
    if results:
        mean_prefill = sum(r["prompt_tokens"] / max(r["prefill_s"], 1e-6) for r in results) / len(results)
        mean_decode = sum(r["decode_tokens"] / max(r["decode_s"], 1e-6) for r in results) / len(results)
    else:
        mean_prefill = mean_decode = 0.0

    print("\n== summary ==")
    print(f"model:        {args.model_id}")
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
