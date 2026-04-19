"""Command-line entry point for ``mcpzim``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .library import scan_paths
from .server import build_server

DEFAULT_ZIM_DIR_ENV = "ZIM_DIR"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcpzim",
        description=(
            "MCP server that exposes ZIM files (Wikipedia, mdwiki, streetzim, "
            "...) to local LLM agents."
        ),
    )
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "ZIM files or directories to load. If omitted, uses the ZIM_DIR "
            "environment variable, or the current working directory."
        ),
    )
    p.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transports.")
    p.add_argument("--port", type=int, default=8765, help="Bind port for HTTP transports.")
    p.add_argument(
        "--log-level",
        default=os.environ.get("MCPZIM_LOG", "INFO"),
        help="Python log level (default: INFO).",
    )
    return p


def _resolve_paths(cli_paths: list[Path]) -> list[Path]:
    if cli_paths:
        return cli_paths
    env = os.environ.get(DEFAULT_ZIM_DIR_ENV)
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return [Path.cwd()]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stdout is reserved for the stdio MCP transport.
    )

    paths = _resolve_paths(args.paths)
    library = scan_paths(paths)
    if not library:
        logging.warning(
            "no ZIM files loaded from %s — server will start but has nothing to serve",
            ", ".join(str(p) for p in paths),
        )
    else:
        logging.info(
            "loaded %d ZIM(s): %s",
            len(library),
            ", ".join(f"{z.path.name}[{z.kind.value}]" for z in library),
        )

    server = build_server(library)
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        # FastMCP reads host/port from its settings object.
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
