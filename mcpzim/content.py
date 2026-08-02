"""Fetch and clean article content from a ZIM archive for LLM consumption."""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

from .library import OpenZim

log = logging.getLogger(__name__)

# Small LRU of parsed articles: fetch + BS4 parse dominate get_article latency,
# and agents commonly re-read the same page within a session.
_ARTICLE_CACHE_MAX = 32
_article_cache: OrderedDict[tuple[str, str], "Article"] = OrderedDict()
_article_cache_lock = threading.Lock()


_WS_RUN = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")

# Tags whose content should read as its own block in the extracted text. We use
# ``separator=" "`` so inline runs like ``Hello <b>world</b>`` stay on one line,
# then inject newline markers after each block element's text so structural
# boundaries (paragraphs, headings, list items) survive.
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "aside",
    "figure",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "tr",
    "br",
    "pre",
    "blockquote",
    "hr",
}

# HTML selectors that carry navigation / editing chrome with no semantic value
# for an LLM. Applied to Wikipedia- and mdwiki-style MediaWiki output.
_STRIP_SELECTORS = (
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    ".navbox",
    ".vertical-navbox",
    ".hatnote",
    ".mw-editsection",
    ".mw-cite-backlink",
    ".reference",
    ".references",
    ".refbegin",
    ".reflist",
    ".metadata",
    ".ambox",
    ".sistersitebox",
    ".thumbcaption",
    ".printfooter",
    "#toc",
    "#mw-navigation",
    "#siteSub",
    "#jump-to-nav",
    "table.infobox",
    "table.sidebar",
    "table.vertical-navbox",
)

# One combined selector group so stripping is a single SoupSieve pass over the
# tree instead of one full-tree walk per selector (~30 passes per article).
# The pilcrow/edit anchors share the same fate so they ride along here too.
_STRIP_SELECTOR = ", ".join(_STRIP_SELECTORS + ("a.mw-headline-anchor", "a.editsection"))

# Search snippets: only the first few hits get a full HTML→text parse (the far
# tail of a 50-hit page is rarely read), and only a bounded HTML prefix is
# parsed — enough context for a 220-char snippet without paying a multi-hundred
# KB BeautifulSoup parse per hit.
_SNIPPET_PARSE_CAP = 8
_SNIPPET_HTML_PREFIX = 96 * 1024


@dataclass
class Article:
    """Rendered article ready for an LLM."""

    zim: str
    path: str
    title: str
    mimetype: str
    text: str
    html_bytes: int

    def to_dict(self) -> dict:
        return {
            "zim": self.zim,
            "path": self.path,
            "title": self.title,
            "mimetype": self.mimetype,
            "text": self.text,
            "html_bytes": self.html_bytes,
        }


def html_to_text(html: str) -> str:
    """Strip wiki chrome from HTML and return readable text.

    Deliberately lightweight: BeautifulSoup's get_text with structural newlines
    is enough for an LLM to read an article; we don't need markdown fidelity.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Single pass: one combined selector group (includes the section-pilcrow
    # and edit-link anchors) instead of a full-tree walk per selector. A match
    # nested inside another match is already gone by the time we reach it.
    for el in soup.select(_STRIP_SELECTOR):
        if not el.decomposed:
            el.decompose()

    body = soup.select_one("#mw-content-text") or soup.select_one("#bodyContent") or soup.body or soup

    # Append a newline marker after each block element so paragraph/heading/list
    # boundaries survive the flatten. Using get_text(separator=" ") alone would
    # collapse structure; get_text(separator="\n") would also split inline runs
    # like "Hello <b>world</b>".
    from bs4 import NavigableString  # local import to keep top-level cheap.

    for el in body.find_all(True):
        if el.name in _BLOCK_TAGS:
            el.append(NavigableString("\n"))

    text = body.get_text(separator=" ")

    # Normalise whitespace: collapse horizontal runs, cap blank lines to 2.
    text = _WS_RUN.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = _BLANK_RUN.sub("\n\n", text).strip()
    return text


def _decode(data: bytes | memoryview) -> str:
    return bytes(data).decode("utf-8", errors="replace")


def _resolve_entry(archive, entry):
    """Follow redirects and return the final entry + its item."""
    if getattr(entry, "is_redirect", False):
        try:
            entry = entry.get_redirect_entry()
        except Exception:  # noqa: BLE001
            pass
    item = entry.get_item()
    return entry, item


def fetch_article(zim: OpenZim, path: str) -> Article | None:
    """Fetch an entry by path. HTML entries are converted to plain text."""
    cache_key = (str(zim.path), path)
    with _article_cache_lock:
        cached = _article_cache.get(cache_key)
        if cached is not None:
            _article_cache.move_to_end(cache_key)
            return cached

    archive = zim.archive
    with zim.lock:
        try:
            entry = archive.get_entry_by_path(path)
        except Exception:
            try:
                entry = archive.get_entry_by_title(path)
            except Exception:
                return None
        entry, item = _resolve_entry(archive, entry)
        raw = bytes(item.content)
        mime = item.mimetype

    if _is_html(mime):
        text = html_to_text(_decode(raw))
    elif mime.startswith("text/"):
        text = _decode(raw)
    elif mime == "application/json":
        text = _decode(raw)
    else:
        text = f"[binary content: {mime}, {len(raw)} bytes]"

    article = Article(
        zim=zim.path.name,
        path=getattr(item, "path", path),
        title=getattr(entry, "title", path),
        mimetype=mime,
        text=text,
        html_bytes=len(raw),
    )
    with _article_cache_lock:
        _article_cache[cache_key] = article
        _article_cache.move_to_end(cache_key)
        while len(_article_cache) > _ARTICLE_CACHE_MAX:
            _article_cache.popitem(last=False)
    return article


def fetch_main_page(zim: OpenZim) -> Article | None:
    archive = zim.archive
    with zim.lock:
        if not getattr(archive, "has_main_entry", True):
            return None
        entry = archive.main_entry
        entry, item = _resolve_entry(archive, entry)
        raw = bytes(item.content)
        mime = item.mimetype
        path = getattr(item, "path", "")
        title = getattr(entry, "title", "Main page")

    text = html_to_text(_decode(raw)) if _is_html(mime) else _decode(raw)
    return Article(
        zim=zim.path.name,
        path=path,
        title=title,
        mimetype=mime,
        text=text,
        html_bytes=len(raw),
    )


def _is_html(mime: str) -> bool:
    return mime.startswith("text/html") or mime == "application/xhtml+xml"


@dataclass
class SearchHit:
    zim: str
    kind: str
    path: str
    title: str
    snippet: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# One Xapian Searcher per archive — constructing one opens the full-text index,
# so reuse it across calls. All use is serialised by the per-ZIM lock.
_searchers: dict[str, object] = {}
_searchers_lock = threading.Lock()


def _searcher_for(zim: OpenZim):
    from libzim.search import Searcher

    key = str(zim.path)
    with _searchers_lock:
        searcher = _searchers.get(key)
        if searcher is None:
            searcher = Searcher(zim.archive)
            _searchers[key] = searcher
    return searcher


def search_zim(zim: OpenZim, query: str, limit: int) -> list[SearchHit]:
    """Run a full-text (or title-prefix) search against a single ZIM."""
    from libzim.search import Query
    from libzim.suggestion import SuggestionSearcher

    hits: list[SearchHit] = []
    archive = zim.archive
    with zim.lock:
        if getattr(archive, "has_fulltext_index", False):
            try:
                search = _searcher_for(zim).search(Query().set_query(query))
                total = int(search.getEstimatedMatches())
                for path in search.getResults(0, min(limit, total)):
                    hits.append(_hit_for(zim, path, query, want_snippet=len(hits) < _SNIPPET_PARSE_CAP))
                if hits:
                    return hits
            except Exception:  # noqa: BLE001
                log.debug("full-text search failed on %s", zim.path.name, exc_info=True)
        if getattr(archive, "has_title_index", False):
            try:
                suggest = SuggestionSearcher(archive).suggest(query)
                total = int(suggest.getEstimatedMatches())
                for path in suggest.getResults(0, min(limit, total)):
                    hits.append(_hit_for(zim, path, query, want_snippet=len(hits) < _SNIPPET_PARSE_CAP))
            except Exception:  # noqa: BLE001
                log.debug("suggest search failed on %s", zim.path.name, exc_info=True)
    return hits


def _hit_for(zim: OpenZim, path: str, query: str, *, want_snippet: bool = True) -> SearchHit:
    title = path
    snippet = ""
    try:
        entry = zim.archive.get_entry_by_path(path)
        entry, item = _resolve_entry(zim.archive, entry)
        title = getattr(entry, "title", path)
        if want_snippet and _is_html(item.mimetype):
            # Parse only a bounded HTML prefix: a snippet needs a couple
            # hundred chars of readable text, not a full multi-hundred-KB
            # BeautifulSoup pass per hit. html.parser copes with the
            # truncated tail, and errors="replace" absorbs a split codepoint.
            snippet = _snippet(html_to_text(_decode(item.content[:_SNIPPET_HTML_PREFIX])), query)
    except Exception:  # noqa: BLE001
        pass
    return SearchHit(
        zim=zim.path.name,
        kind=zim.kind.value,
        path=path,
        title=title,
        snippet=snippet,
    )


def _snippet(text: str, query: str, width: int = 220) -> str:
    """Return a small context window around the first query term match."""
    if not text:
        return ""
    q = query.strip().split()
    lower = text.lower()
    idx = -1
    for term in q:
        i = lower.find(term.lower())
        if i != -1:
            idx = i
            break
    if idx == -1:
        return text[:width].strip()
    start = max(0, idx - width // 3)
    end = min(len(text), start + width)
    prefix = "" if start == 0 else "... "
    suffix = "" if end == len(text) else " ..."
    return (prefix + text[start:end].strip() + suffix).replace("\n", " ")
