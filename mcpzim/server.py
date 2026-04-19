"""FastMCP server that exposes a Library of ZIM files to an LLM agent.

Tools are registered conditionally based on what's in the library: a Wikipedia
ZIM adds general-knowledge search; an mdwiki ZIM adds a medical flag; a
streetzim ZIM adds routing and geocoding tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from .content import fetch_article, fetch_main_page, search_zim
from .geocode import Geocoder
from .library import Library, OpenZim, ZimKind
from .routing import RouterCache, plan_route

log = logging.getLogger(__name__)


@dataclass
class ServerState:
    library: Library
    router: RouterCache
    geocoder: Geocoder

    def find_streetzim(self, zim_name: str | None) -> OpenZim | None:
        candidates = self.library.streetzims
        if not candidates:
            return None
        if zim_name is None:
            return candidates[0]
        match = self.library.find_by_path(zim_name)
        if match and match.kind == ZimKind.STREETZIM:
            return match
        for z in candidates:
            if z.path.name == zim_name:
                return z
        return None


def build_server(library: Library, *, name: str = "mcpzim") -> FastMCP:
    """Return a FastMCP instance whose tools reflect the contents of ``library``."""
    mcp = FastMCP(name)
    state = ServerState(library=library, router=RouterCache(), geocoder=Geocoder())

    _register_core_tools(mcp, state)
    if library.streetzims:
        _register_routing_tools(mcp, state)
    else:
        log.info("no streetzim ZIMs with routing data — route tools disabled")

    return mcp


def _register_core_tools(mcp: FastMCP, state: ServerState) -> None:
    library = state.library

    @mcp.tool()
    def list_libraries() -> dict:
        """Inventory the ZIM archives this server can read.

        Returns a list of archives with their kind (wikipedia / mdwiki /
        streetzim / generic), metadata, and the aggregate capabilities exposed
        (e.g. ``plan_route`` is only present when a streetzim ZIM is loaded).
        Call this first so you know what sources and tools are available.
        """
        return {
            "zims": [z.info.to_dict() for z in library],
            **library.summary(),
        }

    @mcp.tool()
    def search(query: str, limit: int = 10, kind: str | None = None) -> dict:
        """Full-text search across all loaded ZIMs.

        Args:
            query: search terms.
            limit: max results per ZIM (default 10).
            kind: optional filter, one of ``wikipedia``, ``mdwiki``,
                ``streetzim``, ``generic``.

        Each hit includes the source ZIM, path, title and a short snippet.
        """
        limit = max(1, min(int(limit), 50))
        wanted = _parse_kind(kind)
        hits: list[dict] = []
        for zim in library:
            if wanted is not None and zim.kind != wanted:
                continue
            for hit in search_zim(zim, query, limit):
                hits.append(hit.to_dict())
        return {"query": query, "count": len(hits), "hits": hits}

    @mcp.tool()
    def get_article(path: str, zim: str | None = None) -> dict:
        """Fetch a single article by path, as plain text ready for reading.

        Args:
            path: the ZIM entry path (as returned in search results, e.g.
                ``A/Aspirin``).
            zim: optional filename or path of the ZIM to read from. If
                omitted, all loaded ZIMs are tried in scan order.
        """
        targets = [library.find_by_path(zim)] if zim else list(library)
        targets = [t for t in targets if t is not None]
        for target in targets:
            article = fetch_article(target, path)
            if article is not None:
                return article.to_dict()
        raise RuntimeError(f"no entry for path={path!r} in {[t.path.name for t in targets]}")

    @mcp.tool()
    def get_main_page(zim: str | None = None) -> dict:
        """Fetch the main/home page of a ZIM. If no zim name is given, returns
        the main page of every loaded ZIM."""
        if zim:
            target = library.find_by_path(zim)
            if target is None:
                raise RuntimeError(f"unknown zim {zim!r}")
            article = fetch_main_page(target)
            if article is None:
                raise RuntimeError(f"{zim!r} has no main page")
            return article.to_dict()
        pages = []
        for z in library:
            art = fetch_main_page(z)
            if art is not None:
                pages.append(art.to_dict())
        return {"pages": pages}


def _register_routing_tools(mcp: FastMCP, state: ServerState) -> None:
    @mcp.tool()
    def plan_driving_route(
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        zim: str | None = None,
    ) -> dict:
        """Compute a driving route between two lat/lon points.

        Uses the routing graph bundled inside a streetzim ZIM. Returns total
        distance, estimated duration, a polyline, and a list of road segments
        (consecutive edges sharing a street name are coalesced so the result
        reads as turn-by-turn directions).

        Args:
            origin_lat, origin_lon: starting coordinates (WGS84, decimal degrees).
            dest_lat, dest_lon: destination coordinates.
            zim: optional streetzim filename if you loaded more than one.
        """
        target = state.find_streetzim(zim)
        if target is None:
            raise RuntimeError("no streetzim ZIM with routing data is loaded")
        graph = state.router.graph_for(target)
        route = plan_route(target, graph, origin_lat, origin_lon, dest_lat, dest_lon)
        if route is None:
            raise RuntimeError("no route found between the supplied coordinates")
        return {"zim": target.path.name, **route.to_dict()}

    @mcp.tool()
    def geocode(query: str, limit: int = 5, zim: str | None = None, kinds: list[str] | None = None) -> dict:
        """Resolve a place or address string to coordinates using streetzim's
        search index.

        Args:
            query: free text (e.g. "45 Main St, Riga" or "Logan Airport").
            limit: max results.
            zim: optional streetzim filename.
            kinds: optional list of streetzim types to restrict to
                (e.g. ``["addr"]`` for addresses, ``["place", "poi"]``).
        """
        target = state.find_streetzim(zim)
        if target is None:
            raise RuntimeError("no streetzim ZIM is loaded")
        places = state.geocoder.search(target, query, limit=int(limit), kinds=kinds)
        return {
            "zim": target.path.name,
            "query": query,
            "count": len(places),
            "results": [p.to_dict() for p in places],
        }

    @mcp.tool()
    def route_from_places(origin: str, destination: str, zim: str | None = None) -> dict:
        """Plan a driving route between two free-text place names.

        Convenience wrapper that calls ``geocode`` on each string, picks the
        top match, then runs the same A* search as ``plan_driving_route``.
        Raises an error if either query is ambiguous (no match) — in that case
        call ``geocode`` yourself and pick the correct result, then use
        ``plan_driving_route``.
        """
        target = state.find_streetzim(zim)
        if target is None:
            raise RuntimeError("no streetzim ZIM is loaded")
        geocoder = state.geocoder
        ohits = geocoder.search(target, origin, limit=1)
        dhits = geocoder.search(target, destination, limit=1)
        if not ohits:
            raise RuntimeError(f"could not locate origin {origin!r}")
        if not dhits:
            raise RuntimeError(f"could not locate destination {destination!r}")
        graph = state.router.graph_for(target)
        route = plan_route(target, graph, ohits[0].lat, ohits[0].lon, dhits[0].lat, dhits[0].lon)
        if route is None:
            raise RuntimeError("no route found between the resolved places")
        resp = route.to_dict()
        resp["origin_resolved"] = ohits[0].to_dict()
        resp["destination_resolved"] = dhits[0].to_dict()
        resp["zim"] = target.path.name
        return resp


def _parse_kind(kind: str | None) -> ZimKind | None:
    if kind is None:
        return None
    try:
        return ZimKind(kind.lower())
    except ValueError as e:
        raise RuntimeError(
            f"unknown kind {kind!r}; choose one of {[k.value for k in ZimKind]}"
        ) from e


__all__ = ["build_server", "ServerState"]
