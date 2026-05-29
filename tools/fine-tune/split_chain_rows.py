"""Split long 3-turn chain rows into per-turn sub-examples that fit cleanly
under a target max_seq_length.

Why this exists: the chain rows from `generate_chains3.py` are 10-message
trajectories that often run 1200–1600 tokens — they don't fit max_seq=1024,
and at max_seq=2048 mlx-swift's Metal command buffers occasionally trip
macOS's "Impacting Interactivity" GPU watchdog (the FT aborts at iter 1).

Naive truncation at max_seq=1024 loses the TAIL of 68 % of v6 rows — which
is exactly the turn-3 reply (the no-tool-call synthesis pattern we are
trying to teach). Useless.

Approach: emit up to three sub-rows per chain, each terminating after a
different turn's assistant reply. The model sees each turn's loss-bearing
output in context of the prior turns. Short sub-rows fit cleanly; the full
chain sub-row only ships when it actually fits.

Chain row shape (10 messages, indices 0–9):
    0: user (preamble + initial query)
    1: assistant (tool_call_1 fence)
    2: user ([TOOL_RESPONSE] for turn 1)
    3: assistant (reply_1 — natural-language)
    4: user (followup_1)
    5: assistant (tool_call_2 fence)
    6: user ([TOOL_RESPONSE] for turn 2)
    7: assistant (reply_2)
    8: user (followup_2)
    9: assistant (reply_3 — pure synthesis, NO tool call)

Emitted sub-rows:
    A: msgs[0:4]   — through reply_1                (~500-600 tok typical)
    B: msgs[0:8]   — through reply_2                (~900-1000 tok typical)
    C: msgs[0:10]  — full chain incl. reply_3       (~1200-1600 tok typical)

Sub-row C is the only place the turn-3 no-tool-call pattern appears. We
ship it iff it fits — and if it doesn't fit and the row IS a chain, we
emit an additional `C'` that abbreviates `tool_response_1` (the longest
non-loss-bearing turn-1 content) to make room. This preserves the turn-3
loss signal while keeping the row under max_seq.

Non-chain rows (e.g. the existing v4 single-turn examples) pass through
unchanged. A row is identified as a chain by having exactly 10 messages
in the canonical user/assistant alternation.

Usage:
    python split_chain_rows.py \\
        --in train_v6_combined.jsonl \\
        --out train_v6_split.jsonl \\
        --max-seq 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _est_tokens(messages: list[dict]) -> int:
    """Cheap byte/3.5 token estimate. Avoids loading a tokenizer just to
    decide row size; the FT loop will do its own exact tokenisation."""
    return int(sum(len(m.get("content", "")) for m in messages) / 3.5)


def _is_chain_row(row: dict) -> bool:
    msgs = row.get("messages", [])
    if len(msgs) != 10:
        return False
    expected_roles = (
        "user", "assistant", "user", "assistant",
        "user", "assistant", "user", "assistant",
        "user", "assistant",
    )
    return all(m.get("role") == r for m, r in zip(msgs, expected_roles))


def _abbreviate_tool_response_1(content: str) -> str:
    """Shorten the turn-1 [TOOL_RESPONSE] body. Drops `available_sections`
    list (we don't need it for reply_3 synthesis), shortens `lead` to its
    first sentence. Returns a recognisable [TOOL_RESPONSE] envelope so the
    model still sees the same role + role-prefix tokens."""
    if not content.startswith("[TOOL_RESPONSE]"):
        return content
    # body is the JSON after the marker.
    try:
        body_start = content.index("\n")
        body = json.loads(content[body_start + 1:])
    except (ValueError, json.JSONDecodeError):
        return content
    if isinstance(body, dict):
        if isinstance(body.get("lead"), str):
            lead = body["lead"]
            # First sentence (up to '.', '!', or '?').
            for sep in (". ", "! ", "? "):
                idx = lead.find(sep)
                if 30 < idx < 240:
                    body["lead"] = lead[:idx + 1]
                    break
        body.pop("available_sections", None)
    return "[TOOL_RESPONSE]\n" + json.dumps(body, ensure_ascii=False)


def split_one(row: dict, max_seq: int) -> list[dict]:
    """Return one or more sub-rows for `row`.

    For non-chain rows, returns `[row]` unchanged (callers may want to
    drop these if oversized themselves, but typical v4 single-turn rows
    are well under 1024 tokens)."""
    msgs = row["messages"]
    if not _is_chain_row(row):
        return [row]

    out: list[dict] = []
    # A: through reply_1 (turn 1 only). msgs[0:4]
    out.append({"messages": msgs[0:4]})
    # B: through reply_2 (turns 1+2). msgs[0:8]
    out.append({"messages": msgs[0:8]})

    # C: full chain. Only ship if it fits — else build C' with abbreviated
    # tool_response_1.
    full = msgs[0:10]
    if _est_tokens(full) <= max_seq:
        out.append({"messages": full})
    else:
        compressed = list(msgs)
        # msgs[2] is the turn-1 TOOL_RESPONSE
        compressed[2] = {
            "role": "user",
            "content": _abbreviate_tool_response_1(msgs[2]["content"]),
        }
        if _est_tokens(compressed) <= max_seq:
            out.append({"messages": compressed})
        # Else: drop C entirely. Turn-3 signal is gone for this row; A+B
        # still teach turns 1 and 2.
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--max-seq", type=int, default=1024,
                    help="Drop or compress sub-rows whose estimated token "
                         "count exceeds this. Default 1024.")
    args = ap.parse_args()

    n_in = n_chain = n_out = n_dropped_c = 0
    bucket_a = bucket_b = bucket_c = bucket_c_compressed = 0
    with open(args.src) as src, open(args.dst, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            if _is_chain_row(row):
                n_chain += 1
                subs = split_one(row, args.max_seq)
                # The 3 sub-rows from `split_one` always come in
                # canonical A, B, [C/C'] order. Bucket by length.
                if len(subs) >= 1: bucket_a += 1
                if len(subs) >= 2: bucket_b += 1
                if len(subs) >= 3:
                    # Tell C from C' by whether tool_response_1 content
                    # matches the original.
                    orig_tr1 = row["messages"][2]["content"]
                    sub_tr1 = subs[2]["messages"][2]["content"]
                    if sub_tr1 == orig_tr1:
                        bucket_c += 1
                    else:
                        bucket_c_compressed += 1
                else:
                    n_dropped_c += 1
            else:
                subs = [row]
            for s in subs:
                dst.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"in:   {n_in} rows ({n_chain} chains, {n_in - n_chain} non-chains)",
          file=sys.stderr)
    print(f"out:  {n_out} rows", file=sys.stderr)
    print(f"  buckets per chain:", file=sys.stderr)
    print(f"    A (turn 1):                  {bucket_a}", file=sys.stderr)
    print(f"    B (turns 1+2):               {bucket_b}", file=sys.stderr)
    print(f"    C full (fits as-is):         {bucket_c}", file=sys.stderr)
    print(f"    C' compressed tool_resp_1:   {bucket_c_compressed}",
          file=sys.stderr)
    print(f"    dropped C (still too long):  {n_dropped_c}", file=sys.stderr)


if __name__ == "__main__":
    main()
