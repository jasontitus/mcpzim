"""Scan a directory for ZIM files, open them, and classify their content type."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


class ZimKind(str, Enum):
    WIKIPEDIA = "wikipedia"
    MDWIKI = "mdwiki"
    STREETZIM = "streetzim"
    GENERIC = "generic"


@dataclass
class ZimInfo:
    """Summary of a single ZIM file."""

    path: Path
    kind: ZimKind
    name: str
    title: str
    description: str
    language: str
    creator: str
    publisher: str
    date: str
    tags: list[str]
    article_count: int
    has_fulltext_index: bool
    has_title_index: bool
    has_routing: bool = False

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "kind": self.kind.value,
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "creator": self.creator,
            "publisher": self.publisher,
            "date": self.date,
            "tags": self.tags,
            "article_count": self.article_count,
            "has_fulltext_index": self.has_fulltext_index,
            "has_title_index": self.has_title_index,
            "has_routing": self.has_routing,
        }


@dataclass
class OpenZim:
    """A loaded ZIM archive plus its classification and a read lock."""

    info: ZimInfo
    archive: object  # libzim.reader.Archive – untyped to keep libzim import lazy.
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def kind(self) -> ZimKind:
        return self.info.kind

    @property
    def path(self) -> Path:
        return self.info.path


def _metadata(archive, key: str) -> str:
    try:
        if key in archive.metadata_keys:
            raw = archive.get_metadata(key)
            if isinstance(raw, (bytes, bytearray, memoryview)):
                return bytes(raw).decode("utf-8", "replace").strip()
            return str(raw).strip()
    except Exception:  # noqa: BLE001 – libzim raises a variety of errors.
        log.debug("metadata %s unreadable for %s", key, archive.filename, exc_info=True)
    return ""


def _split_tags(value: str) -> list[str]:
    return [t.strip() for t in value.split(";") if t.strip()]


def _has_entry(archive, path: str) -> bool:
    try:
        return bool(archive.has_entry_by_path(path))
    except Exception:  # noqa: BLE001
        return False


def classify(
    *,
    filename: str,
    name: str,
    title: str,
    creator: str,
    publisher: str,
    tags: Iterable[str],
    has_routing_entry: bool,
    has_streetzim_config: bool,
) -> ZimKind:
    """Choose the best ZimKind for a ZIM given its metadata and signature entries."""

    tagset = {t.lower() for t in tags}
    fn = filename.lower()
    nm = (name or "").lower()
    ti = (title or "").lower()
    cr = (creator or "").lower()
    pb = (publisher or "").lower()

    # streetzim: ours, so recognised by its signature files first.
    if has_routing_entry or has_streetzim_config:
        return ZimKind.STREETZIM
    if nm.startswith("streetzim") or nm.startswith("street_") or fn.startswith("streetzim") or "streetzim" in tagset:
        return ZimKind.STREETZIM

    # mdwiki: distinct publisher and tags.
    if "mdwiki" in tagset or "medical" in tagset:
        return ZimKind.MDWIKI
    if nm.startswith("mdwiki") or fn.startswith("mdwiki") or "wikiprojectmed" in pb or "mdwiki" in ti:
        return ZimKind.MDWIKI

    # Wikipedia: bulk of Kiwix catalogue.
    if "wikipedia" in tagset or "_category:wikipedia" in tagset:
        return ZimKind.WIKIPEDIA
    if nm.startswith("wikipedia") or fn.startswith("wikipedia") or cr == "wikipedia":
        return ZimKind.WIKIPEDIA

    return ZimKind.GENERIC


def open_zim(path: Path) -> OpenZim:
    """Open a ZIM file with libzim and classify it.

    Raises if libzim is not installed or the file is unreadable.
    """

    from libzim.reader import Archive  # lazy import – libzim wheel is heavy.

    archive = Archive(str(path))

    name = _metadata(archive, "Name")
    title = _metadata(archive, "Title")
    description = _metadata(archive, "Description")
    language = _metadata(archive, "Language")
    creator = _metadata(archive, "Creator")
    publisher = _metadata(archive, "Publisher")
    date = _metadata(archive, "Date")
    tags = _split_tags(_metadata(archive, "Tags"))

    has_routing_entry = _has_entry(archive, "routing-data/graph.bin") or _has_entry(
        archive, "routing-data/graph.json"
    )
    has_streetzim_config = _has_entry(archive, "map-config.json")

    kind = classify(
        filename=path.name,
        name=name,
        title=title,
        creator=creator,
        publisher=publisher,
        tags=tags,
        has_routing_entry=has_routing_entry,
        has_streetzim_config=has_streetzim_config,
    )

    info = ZimInfo(
        path=path,
        kind=kind,
        name=name,
        title=title,
        description=description,
        language=language,
        creator=creator,
        publisher=publisher,
        date=date,
        tags=tags,
        article_count=int(getattr(archive, "article_count", 0) or 0),
        has_fulltext_index=bool(getattr(archive, "has_fulltext_index", False)),
        has_title_index=bool(getattr(archive, "has_title_index", False)),
        has_routing=has_routing_entry,
    )
    return OpenZim(info=info, archive=archive)


class Library:
    """A collection of opened ZIM archives."""

    def __init__(self, zims: list[OpenZim] | None = None):
        self._zims: list[OpenZim] = zims or []

    def __iter__(self):
        return iter(self._zims)

    def __len__(self):
        return len(self._zims)

    @property
    def zims(self) -> list[OpenZim]:
        return list(self._zims)

    def of_kind(self, kind: ZimKind) -> list[OpenZim]:
        return [z for z in self._zims if z.kind == kind]

    @property
    def streetzims(self) -> list[OpenZim]:
        return [z for z in self._zims if z.kind == ZimKind.STREETZIM and z.info.has_routing]

    def find_by_path(self, path_str: str) -> OpenZim | None:
        target = Path(path_str).resolve()
        for z in self._zims:
            if z.path.resolve() == target or z.path.name == path_str:
                return z
        return None

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for z in self._zims:
            counts[z.kind.value] = counts.get(z.kind.value, 0) + 1
        return {
            "count": len(self._zims),
            "by_kind": counts,
            "capabilities": sorted(self.capabilities()),
        }

    def capabilities(self) -> set[str]:
        """Symbolic capabilities the library as a whole can offer."""
        caps: set[str] = set()
        if self._zims:
            caps.update({"search", "get_article", "list_libraries"})
        if any(z.kind in (ZimKind.WIKIPEDIA, ZimKind.MDWIKI, ZimKind.GENERIC) for z in self._zims):
            caps.add("encyclopedia")
        if any(z.kind == ZimKind.MDWIKI for z in self._zims):
            caps.add("medical")
        if any(z.kind == ZimKind.WIKIPEDIA for z in self._zims):
            caps.add("general_knowledge")
        if self.streetzims:
            caps.update({"plan_route", "geocode", "maps"})
        return caps


def scan_paths(paths: Iterable[Path]) -> Library:
    """Open every *.zim file referenced by ``paths``.

    Each path may be either a ZIM file or a directory (scanned non-recursively
    then one level deep).
    """
    seen: set[Path] = set()
    opened: list[OpenZim] = []
    for p in paths:
        p = Path(p).expanduser()
        if not p.exists():
            log.warning("ZIM path does not exist: %s", p)
            continue
        for zp in _iter_zim_files(p):
            rp = zp.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                opened.append(open_zim(zp))
                log.info("opened %s (%s)", zp.name, opened[-1].kind.value)
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to open %s: %s", zp, exc)
    return Library(opened)


def _iter_zim_files(p: Path) -> Iterable[Path]:
    if p.is_file():
        if p.suffix.lower() == ".zim":
            yield p
        return
    if p.is_dir():
        # First pass: direct .zim children.
        yield from sorted(p.glob("*.zim"))
        # Second pass: one level deeper (e.g. ~/zims/wikipedia/foo.zim).
        yield from sorted(p.glob("*/*.zim"))
