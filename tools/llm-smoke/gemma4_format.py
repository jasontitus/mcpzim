# SPDX-License-Identifier: MIT
#
# Python port of MCPZimKit's Gemma4PromptTemplate + Gemma4ToolFormat +
# Gemma4ToolCallParser. Used by eval_gemma4_native.py so the mac eval runs
# against the same custom mini-format the iOS app uses — NOT the HF
# default chat template that `tokenizer.apply_chat_template` would apply
# (that template refuses to call tools).
#
# Format recap (see Gemma4ToolFormat.swift for authoritative source):
#
#   - Transcript wrapper:
#       <bos>
#       <|turn>system\n...<|tool>declaration:NAME{...}<tool|>...<turn|>\n
#       <|turn>user\nTEXT<turn|>\n
#       <|turn>model\n            (generation begins here)
#
#   - Tool-call emission (inverse of what we're parsing):
#       <|tool_call>call:NAME{key:value,key:value}<tool_call|>
#
#   - Value encoding:
#       string  → <|"|>VALUE<|"|>
#       number  → decimal literal
#       bool    → true / false
#       null    → null
#       array   → [v,v,...]
#       object  → {k:v,k:v,...}   (keys UNQUOTED)

from __future__ import annotations

import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Rendering

def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Strip trailing ".0" for ints-as-float just like Swift NSNumber.stringValue.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        return f"<|\"|>{value}<|\"|>"
    if isinstance(value, list):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        pairs = [f"{k}:{_format_value(value[k])}" for k in sorted(value)]
        return "{" + ",".join(pairs) + "}"
    return f"<|\"|>{str(value)}<|\"|>"


# Gemma declares params with uppercase type names.
_PARAM_TYPE_MAP = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def format_tool_declaration(tool: dict) -> str:
    """Emit a single `<|tool>declaration:NAME{...}<tool|>` block from a
    JSON-schema-style tool dict matching the one in eval.py."""
    fn = tool.get("function", tool)
    name = fn["name"]
    description = fn.get("description", "")
    params_schema = fn.get("parameters", {}) or {}
    properties = params_schema.get("properties", {}) or {}
    required = set(params_schema.get("required", []) or [])

    body = f'declaration:{name}{{description:<|"|>{description}<|"|>'
    if properties:
        body += ",parameters:{properties:{"
        param_strs = []
        for pname, pschema in properties.items():
            attrs = []
            if desc := pschema.get("description"):
                attrs.append(f'description:<|"|>{desc}<|"|>')
            if ev := pschema.get("enum"):
                elist = ",".join(f'<|"|>{e}<|"|>' for e in ev)
                attrs.append(f"enum:[{elist}]")
            if pschema.get("nullable"):
                attrs.append("nullable:true")
            # Resolve the type — array items carry their own schema but we
            # just surface the outer type name here.
            raw_t = pschema.get("type", "string")
            attrs.append(f'type:<|"|>{_PARAM_TYPE_MAP.get(raw_t, raw_t.upper())}<|"|>')
            param_strs.append(f"{pname}:{{{','.join(attrs)}}}")
        body += ",".join(param_strs)
        body += "}"
        if required:
            # Keep declaration order stable by matching properties iteration.
            req_list = [p for p in properties if p in required]
            if req_list:
                rlist = ",".join(f'<|"|>{r}<|"|>' for r in req_list)
                body += f",required:[{rlist}]"
        body += ',type:<|"|>OBJECT<|"|>}'
    body += "}"
    return f"<|tool>{body}<tool|>"


def format_system_turn(system_message: str, tools: list[dict]) -> str:
    if not system_message and not tools:
        return ""
    parts = ["<|turn>system\n"]
    if system_message:
        parts.append(system_message)
        if not system_message.endswith("\n"):
            parts.append("\n")
    for t in tools:
        parts.append(format_tool_declaration(t))
    parts.append("<turn|>\n")
    return "".join(parts)


def render_transcript(system_message: str, tools: list[dict], user_text: str) -> str:
    """Single-turn transcript terminating in `<|turn>model\\n` so the
    caller can pipe into stream_generate and have the model continue."""
    out = "<bos>"
    out += format_system_turn(system_message, tools)
    out += f"<|turn>user\n{user_text}<turn|>\n"
    out += "<|turn>model\n"
    return out


# ---------------------------------------------------------------------------
# Parsing

class _Scanner:
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def eof(self) -> bool:
        return self.i >= self.n

    def skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i].isspace():
            self.i += 1

    def peek_literal(self, lit: str) -> bool:
        return self.s.startswith(lit, self.i)

    def take_literal(self, lit: str) -> bool:
        if self.peek_literal(lit):
            self.i += len(lit)
            return True
        return False

    def take_char(self, ch: str) -> bool:
        if self.i < self.n and self.s[self.i] == ch:
            self.i += 1
            return True
        return False

    def take_key(self) -> Optional[str]:
        start = self.i
        while self.i < self.n and self.s[self.i] not in ":,}":
            self.i += 1
        key = self.s[start:self.i].strip()
        return key or None

    def take_quoted_string(self) -> Optional[str]:
        start = self.i
        while self.i < self.n:
            if self.s.startswith("<|\"|>", self.i):
                out = self.s[start:self.i]
                self.i += 5
                return out
            self.i += 1
        return None

    def take_number(self) -> Optional[Any]:
        start = self.i
        if self.i < self.n and self.s[self.i] == "-":
            self.i += 1
        while self.i < self.n and (self.s[self.i].isdigit() or self.s[self.i] == "."):
            self.i += 1
        chunk = self.s[start:self.i]
        if not chunk:
            return None
        if "." in chunk:
            try:
                return float(chunk)
            except ValueError:
                return None
        try:
            return int(chunk)
        except ValueError:
            return None

    def take_value(self) -> Any:
        self.skip_ws()
        if self.take_literal("<|\"|>"):
            return self.take_quoted_string()
        if self.take_literal("true"):
            return True
        if self.take_literal("false"):
            return False
        if self.take_literal("null"):
            return None
        if self.take_char("["):
            return self.take_array()
        if self.take_char("{"):
            return self.take_object()
        return self.take_number()

    def take_array(self) -> Optional[list]:
        out = []
        self.skip_ws()
        if self.take_char("]"):
            return out
        while True:
            self.skip_ws()
            out.append(self.take_value())
            self.skip_ws()
            if self.take_char("]"):
                return out
            if not self.take_char(","):
                return None

    def take_object(self) -> Optional[dict]:
        out: dict = {}
        self.skip_ws()
        if self.take_char("}"):
            return out
        while True:
            self.skip_ws()
            key = self.take_key()
            if key is None:
                return None
            if not self.take_char(":"):
                return None
            out[key] = self.take_value()
            self.skip_ws()
            if self.take_char("}"):
                return out
            if not self.take_char(","):
                return None


def _implied_body_end(buf: str, start: int) -> Optional[int]:
    """If the model forgot `<tool_call|>`, find the outermost matching `}` of
    the `call:NAME{...}` body starting from `start`. Returns index AFTER
    that brace, or None if unbalanced."""
    i = start
    while i < len(buf) and buf[i] != "{":
        i += 1
    if i >= len(buf):
        return None
    depth = 0
    in_str = False
    while i < len(buf):
        if not in_str and buf.startswith("<|\"|>", i):
            in_str = True
            i += 5
            continue
        if in_str and buf.startswith("<|\"|>", i):
            in_str = False
            i += 5
            continue
        if not in_str:
            if buf[i] == "{":
                depth += 1
            elif buf[i] == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def parse_call_body(body: str) -> Optional[tuple[str, dict]]:
    body = body.strip()
    if not body.startswith("call:"):
        return None
    rest = body[5:].strip()
    # Drop `-> TYPE` if present.
    if " -> " in rest:
        rest = rest.split(" -> ", 1)[0].strip()
    # Find the first `{` or `(`.
    m = re.search(r"[\{\(]", rest)
    if not m:
        return None
    opener_idx = m.start()
    name = rest[:opener_idx].strip()
    opener = rest[opener_idx]
    closer = ")" if opener == "(" else "}"
    inner = rest[opener_idx + 1:]
    if inner.endswith(closer):
        inner = inner[:-1]
    if opener == "(":
        inner = inner.strip()
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1]
    # Parse the {k:v,...} body.
    scanner = _Scanner(inner)
    scanner.skip_ws()
    if scanner.eof():
        return name, {}
    args: dict = {}
    while not scanner.eof():
        scanner.skip_ws()
        key = scanner.take_key()
        if key is None:
            return None
        if not scanner.take_char(":"):
            return None
        args[key] = scanner.take_value()
        scanner.skip_ws()
        if scanner.eof():
            break
        if not scanner.take_char(","):
            return None
    return name, args


def first_call(buf: str) -> Optional[tuple[str, dict]]:
    """Extract the first tool call from a Gemma 4 emission buffer."""
    open_idx = buf.find("<|tool_call>")
    if open_idx < 0:
        return None
    open_end = open_idx + len("<|tool_call>")
    close_idx = buf.find("<tool_call|>", open_end)
    if close_idx >= 0:
        body = buf[open_end:close_idx]
        parsed = parse_call_body(body)
        if parsed is not None:
            return parsed
    implied = _implied_body_end(buf, open_end)
    if implied is not None:
        body = buf[open_end:implied]
        parsed = parse_call_body(body)
        if parsed is not None:
            return parsed
    return None
