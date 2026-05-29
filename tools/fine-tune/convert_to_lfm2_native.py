"""Convert train_v4_combined.jsonl (Gemma-style fold-into-user + JSON
tool-call format) → LFM2.5-native (real `system` role + Pythonic tool
call format).

Two shape changes per row:

1. **Roles.** Gemma 3 has no system role, so the data folds the
   preamble + tool block into the first user turn. LFM2.5 has a real
   system role — pull the preamble out and emit it as `{"role":
   "system", "content": ...}`. The tool catalogue moves with the
   preamble (LFM2.5's chat template renders `tools=` into the system
   block when serialized; we keep parity by inlining).

2. **Tool calls.** Replace every assistant
       ```tool_call
       {"function": "X", "parameters": {"k": "v"}}
       ```
   fenced block with
       <|tool_call_start|>[X(k='v')]<|tool_call_end|>

   Pythonic syntax matches LFM2.5's training distribution. We use
   single quotes for strings (matches the chat template).

Tool responses (`[TOOL_RESPONSE]` user turns) stay as-is — LFM2 expects
them as user role too, just plain text.

Usage:
    python convert_to_lfm2_native.py train_v4_combined.jsonl train_v4_lfm2.jsonl
"""

import argparse
import ast
import json
import re
import sys

PREAMBLE_USER_QUERY_MARKER = "\n\nUser query:\n"
# The Gemma preamble ends with the tool format instructions:
#   "To call a tool, respond with ONLY a code fence like:\n```tool_call\n..."
# After the closing of those instructions, there's a blank line then
# "User query:\n<actual user text>". We split there.
USER_QUERY_FALLBACK = "User query:\n"

# Matches ```tool_call\n{json body}\n``` blocks the assistant emits.
TOOL_FENCE_RE = re.compile(
    r"```tool_call\s*(?P<body>\{.*?\})\s*```",
    re.DOTALL,
)


def _format_arg_value(v) -> str:
    """Render a Python value the way LFM2.5's chat template does:
    strings in single quotes, mappings as JSON, everything else
    via str()."""
    if isinstance(v, str):
        # Escape any embedded single quotes / backslashes.
        esc = v.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{esc}'"
    if isinstance(v, dict):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_format_arg_value(x) for x in v) + "]"
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    return str(v)


def _json_call_to_pythonic(obj: dict) -> str:
    """Render `{"function": "X", "parameters": {"k": "v"}}` →
    `[X(k='v')]` with LFM2-style argument formatting."""
    name = obj.get("function") or obj.get("name")
    if not name:
        return ""
    args = obj.get("parameters") or obj.get("arguments") or {}
    parts = [f"{k}={_format_arg_value(v)}" for k, v in args.items()]
    return f"[{name}({', '.join(parts)})]"


def _rewrite_assistant(content: str) -> str:
    """Replace every JSON tool_call fence with the Pythonic form.
    Non-tool-call prose passes through untouched."""
    def sub(m):
        body = m.group("body")
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return m.group(0)
        py = _json_call_to_pythonic(obj)
        if not py:
            return m.group(0)
        return f"<|tool_call_start|>{py}<|tool_call_end|>"
    return TOOL_FENCE_RE.sub(sub, content)


def _split_preamble_and_query(first_user_content: str) -> tuple[str, str]:
    """Pull the preamble (system + tools) out of the first user turn.
    Returns (system_text, user_text). Falls back to empty system if
    no marker found (treat as already-conversational)."""
    idx = first_user_content.find(PREAMBLE_USER_QUERY_MARKER)
    if idx >= 0:
        sys_text = first_user_content[:idx].rstrip()
        user_text = first_user_content[idx + len(PREAMBLE_USER_QUERY_MARKER):]
        return sys_text, user_text
    # Some rows may not have the explicit marker — try a softer match.
    idx = first_user_content.find(USER_QUERY_FALLBACK)
    if idx >= 0:
        sys_text = first_user_content[:idx].rstrip()
        user_text = first_user_content[idx + len(USER_QUERY_FALLBACK):]
        return sys_text, user_text
    # No preamble detected — keep everything in user.
    return "", first_user_content


def convert_row(row: dict) -> dict:
    messages = row.get("messages", [])
    if not messages:
        return row
    out: list[dict] = []
    first_user_seen = False
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user" and not first_user_seen:
            first_user_seen = True
            sys_text, user_text = _split_preamble_and_query(content)
            if sys_text:
                # Strip the JSON-block tool-format instructions; LFM2.5
                # is trained to emit Pythonic, no in-band hint needed.
                # Keep everything before "You have these tools" intact,
                # and re-emit the tool list in a plain shape.
                out.append({"role": "system", "content": sys_text})
            out.append({"role": "user", "content": user_text})
            continue
        if role == "assistant":
            out.append({"role": role, "content": _rewrite_assistant(content)})
            continue
        out.append({"role": role, "content": content})
    return {"messages": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="train_v4_combined.jsonl input path")
    ap.add_argument("dst", help="output jsonl path")
    args = ap.parse_args()

    n_in = n_out = 0
    n_tool_calls_rewritten = 0
    with open(args.src) as src_fh, open(args.dst, "w") as dst_fh:
        for line in src_fh:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            new_row = convert_row(row)
            # Sanity: count how many tool_call rewrites happened.
            for m in new_row.get("messages", []):
                if m.get("role") == "assistant":
                    n_tool_calls_rewritten += m["content"].count(
                        "<|tool_call_start|>"
                    )
            dst_fh.write(json.dumps(new_row) + "\n")
            n_out += 1
    print(f"in={n_in} out={n_out} tool_calls_rewritten={n_tool_calls_rewritten}")


if __name__ == "__main__":
    main()
