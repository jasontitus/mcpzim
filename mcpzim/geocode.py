"""Streetzim geocoding: resolve place/address strings to (lat, lon).

StreetZim ZIMs ship a prefix-chunked JSON search index at
``search-data/manifest.json`` + ``search-data/{prefix}.json`` containing address
points, places and POIs. This module mirrors the viewer's ``_queryPlaces``
behavior so a local LLM can ask "where is 45 Main St" and get coordinates it can
feed into the router.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass

from .library import OpenZim

log = logging.getLogger(__name__)

_PREFIX_CLEAN = re.compile(r"[^a-z0-9_]")


@dataclass
class Place:
    name: str
    type: str
    lat: float
    lon: float
    subtype: str = ""
    location: str = ""

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "type": self.type,
            "lat": self.lat,
            "lon": self.lon,
        }
        if self.subtype:
            d["subtype"] = self.subtype
        if self.location:
            d["location"] = self.location
        return d


def normalize_prefix(query: str) -> str:
    q = (query or "").lower()[:2]
    q = q.replace(" ", "_")
    q = _PREFIX_CLEAN.sub("_", q)
    while len(q) < 2:
        q += "_"
    return q


def _read_json(zim: OpenZim, path: str) -> object | None:
    with zim.lock:
        try:
            entry = zim.archive.get_entry_by_path(path)
        except Exception:
            return None
        if getattr(entry, "is_redirect", False):
            entry = entry.get_redirect_entry()
        item = entry.get_item()
        raw = bytes(item.content)
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        log.debug("bad JSON at %s in %s", path, zim.path.name)
        return None


class Geocoder:
    """Per-ZIM cache of loaded search chunks."""

    def __init__(self) -> None:
        self._chunks: dict[tuple[str, str], list[dict]] = {}
        self._manifests: dict[str, dict | None] = {}
        self._lock = threading.Lock()

    def _manifest(self, zim: OpenZim) -> dict | None:
        key = str(zim.path.resolve())
        with self._lock:
            if key in self._manifests:
                return self._manifests[key]
        m = _read_json(zim, "search-data/manifest.json")
        m = m if isinstance(m, dict) else None
        with self._lock:
            self._manifests.setdefault(key, m)
        return m

    def _chunk(self, zim: OpenZim, prefix: str) -> list[dict]:
        key = (str(zim.path.resolve()), prefix)
        with self._lock:
            cached = self._chunks.get(key)
            if cached is not None:
                return cached
        data = _read_json(zim, f"search-data/{prefix}.json")
        records = data if isinstance(data, list) else []
        with self._lock:
            self._chunks.setdefault(key, records)
        return records

    def search(
        self,
        zim: OpenZim,
        query: str,
        *,
        limit: int = 10,
        kinds: list[str] | None = None,
    ) -> list[Place]:
        query = (query or "").strip()
        if not query:
            return []
        manifest = self._manifest(zim)
        prefix = normalize_prefix(query)
        if manifest is not None and prefix not in (manifest.get("chunks") or {}):
            return []
        records = self._chunk(zim, prefix)
        q_lower = query.lower()
        wanted = set(kinds) if kinds else None

        scored: list[tuple[int, int, Place]] = []
        for rec in records:
            name = str(rec.get("n", ""))
            if not name:
                continue
            t = str(rec.get("t", ""))
            if wanted and t not in wanted:
                continue
            name_lower = name.lower()
            if q_lower not in name_lower:
                continue
            # Score = earliest match position, then shorter name wins.
            scored.append(
                (
                    name_lower.find(q_lower),
                    len(name_lower),
                    Place(
                        name=name,
                        type=t,
                        lat=float(rec.get("a", 0.0)),
                        lon=float(rec.get("o", 0.0)),
                        subtype=str(rec.get("s", "") or ""),
                        location=str(rec.get("l", "") or ""),
                    ),
                )
            )
        scored.sort(key=lambda s: (s[0], s[1]))
        return [p for _, _, p in scored[:limit]]


__all__ = ["Geocoder", "Place", "normalize_prefix"]
