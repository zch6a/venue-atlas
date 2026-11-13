"""Accepted papers, from each conference's own website: the static file its virtual-conference site publishes for the
schedule (`/static/virtual/data/<venue>-<year>-orals-posters.json`): one request per venue and year, with titles,
abstracts, links and the conference's own topic for each paper. The ACL Anthology's venues come from its BibTeX
(see anthology.py).

Abstracts are read to place the papers on the map, and are not kept: the atlas holds titles, links, years, the
decision (oral, spotlight, poster) and the conference's topic.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .anthology import MAX_AGE_DAYS
from .anthology import URL as ANTHOLOGY_URL
from .anthology import entries as anthology_entries

USER_AGENT = "venue-atlas (a paper map built from the conference's public schedule data)"
MAX_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    host: str
    years: tuple[int, ...]
    volumes: tuple[str, ...] = ()  # for the ACL Anthology: the volumes of a year's main conference ("acl-long")

    @property
    def anthology(self) -> bool:
        return bool(self.volumes)


TRACKS = {"long": "long paper", "short": "short paper", "main": "main conference"}

# Years checked to answer on 2026-11-14 (HEAD requests); CVPR's site has the file from 2023. NAACL has no 2023 (EACL's
# year). The Anthology's venues: their main-conference volumes, not Findings or workshops.
VENUES = {
    v.key: v
    for v in [
        Venue("iclr", "ICLR", "iclr.cc", (2021, 2022, 2023, 2024, 2025)),
        Venue("neurips", "NeurIPS", "neurips.cc", (2021, 2022, 2023, 2024, 2025)),
        Venue("icml", "ICML", "icml.cc", (2021, 2022, 2023, 2024, 2025)),
        Venue("cvpr", "CVPR", "cvpr.thecvf.com", (2023, 2024, 2025)),
        Venue("acl", "ACL", "aclanthology.org", (2021, 2022, 2023, 2024, 2025), ("acl-long", "acl-short")),
        Venue("emnlp", "EMNLP", "aclanthology.org", (2021, 2022, 2023, 2024, 2025), ("emnlp-main",)),
        Venue("naacl", "NAACL", "aclanthology.org", (2021, 2022, 2024, 2025), ("naacl-long", "naacl-short", "naacl-main")),
    ]
}


@dataclass
class Paper:
    title: str
    year: int
    abstract: str  # for embedding only; not written to the atlas
    url: str  # the paper (OpenReview, proceedings, CVF open access), or ""
    page: str  # the conference's page for it
    decision: str  # oral, spotlight, poster, or ""
    topic: str  # the conference's own topic, or ""
    key: str = field(default="")  # the conference's id for it


class SourceError(Exception):
    pass


Fetcher = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise SourceError(f"{url}: larger than {MAX_BYTES // (1024 * 1024)} MB")
    return data


def url_for(venue: Venue, year: int) -> str:
    if venue.anthology:
        return ANTHOLOGY_URL
    return f"https://{venue.host}/static/virtual/data/{venue.key}-{year}-orals-posters.json"


def anthology_raw(cache: Path | None = None, fetcher: Fetcher | None = None) -> bytes:
    """The Anthology's BibTeX with abstracts (45 MB gzipped), from the cache if it is younger than MAX_AGE_DAYS."""
    f = cache / "anthology+abstracts.bib.gz" if cache is not None else None
    if f is not None and f.is_file() and time.time() - f.stat().st_mtime < MAX_AGE_DAYS * 86400:
        return f.read_bytes()
    try:
        data = (fetcher or _fetch)(ANTHOLOGY_URL)
    except OSError as e:
        raise SourceError(f"{ANTHOLOGY_URL}: could not download ({getattr(e, 'code', None) or getattr(e, 'reason', e)})") from e
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    return data


def parse_anthology(data: bytes, venue: Venue, years: list[int]) -> list[Paper]:
    volumes = {f"{y}.{v}" for y in years for v in venue.volumes}
    out = []
    seen: set[str] = set()
    for year, vol, number, title, abstract in anthology_entries(data, volumes):
        key = norm_title(title)
        if key in seen:
            continue
        seen.add(key)
        link = f"https://aclanthology.org/{year}.{vol}.{number}/"
        track = TRACKS.get(vol.rsplit("-", 1)[-1], "")
        out.append(Paper(title, year, abstract, link, link, "", track, f"{year}.{vol}.{number}"))
    if not out:
        raise SourceError(f"{venue.name} {', '.join(map(str, years))}: no papers of its volumes in the Anthology's BibTeX")
    return out


def raw(venue: Venue, year: int, cache: Path | None = None, fetcher: Fetcher = _fetch) -> bytes:
    """The schedule file, from the cache folder if it is there (it is kept there after a download)."""
    if cache is not None:
        f = cache / f"{venue.key}-{year}.json"
        if f.is_file():
            return f.read_bytes()
    url = url_for(venue, year)
    try:
        data = fetcher(url)
    except OSError as e:
        raise SourceError(f"{url}: could not download ({getattr(e, 'code', None) or getattr(e, 'reason', e)})") from e
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{venue.key}-{year}.json").write_bytes(data)
    return data


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) and value.strip() not in ("None", "null") else ""


def _decision(record: dict) -> str:
    """oral, spotlight or poster, from the decision ("Accept (spotlight)") or the event's type ("Oral Poster").

    >>> _decision({"decision": "Accept (Oral)"}), _decision({"event_type": "Spotlight Poster"}), _decision({"eventtype": "Poster"})
    ('oral', 'spotlight', 'poster')
    """
    words = " ".join(_text(record.get(k)) for k in ("decision", "event_type", "eventtype")).lower()
    for d in ("oral", "spotlight", "poster"):
        if d in words:
            return d
    return ""


def norm_title(title: str) -> str:
    """A title for comparing: lower case, letters and digits only.

    >>> norm_title("  Robot Fleet Learning via Policy-Merging! ")
    'robot fleet learning via policy merging'
    """
    return " ".join(re.findall(r"[^\W_]+", title.lower()))


RANK = {"oral": 3, "spotlight": 2, "poster": 1, "": 0}
# entries of the schedule that are not papers: sessions, talks, socials (their "abstract" lists the session's posters)
NOT_PAPERS = re.compile(r"^(lightning talks?|poster session|oral session|spotlight session|spotlights? \d|orals? \d|session \d|invited talk|keynote|"
                        r"tutorial|workshop|opening remarks|closing remarks|social|affinity|town hall|panel)\b", re.I)  # fmt: skip


def plain(html: str) -> str:
    """An abstract without HTML tags or entities.

    >>> plain("We <b>propose</b> a&nbsp;method &amp; a bound.")
    'We propose a method & a bound.'
    """
    import html as html_module

    text = re.sub(r"<[^>]{0,500}>", " ", html)
    return re.sub(r"\s+", " ", html_module.unescape(text)).strip()


def parse(data: bytes, venue: Venue, year: int) -> list[Paper]:
    """The papers of one schedule file, each once: an oral is listed as its talk and its poster (the same uid or title);
    the one entry keeps the highest decision and the poster's page."""
    try:
        doc = json.loads(data.decode("utf-8", errors="replace"))
    except ValueError as e:
        raise SourceError(f"{venue.name} {year}: not a JSON file ({e})") from e
    results = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(results, list):
        raise SourceError(f"{venue.name} {year}: no list of results in the file")
    by_key: dict[str, Paper] = {}
    by_abstract: dict[str, str] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        title = re.sub(r"\s+", " ", _text(r.get("name")))
        abstract_html = _text(r.get("abstract"))
        if not title or NOT_PAPERS.match(title) or "/virtual/" in abstract_html:
            continue  # a session or a talk: its abstract lists other entries
        key = norm_title(title)
        page = _text(r.get("virtualsite_url"))
        if page.startswith("/"):
            page = f"https://{venue.host}{page}"
        url = _text(r.get("paper_url")) or _text(r.get("url"))
        if url and not url.startswith(("https://", "http://")):
            url = ""
        abstract = plain(abstract_html)
        if len(abstract) > 300 and abstract in by_abstract and by_abstract[abstract] != key:
            key = by_abstract[abstract]  # the same paper under another title (listed twice): one entry
        elif len(abstract) > 300:
            by_abstract[abstract] = key
        p = Paper(title, year, abstract, url, page, _decision(r), _text(r.get("topic")), _text(r.get("uid")) or _text(r.get("id")))
        old = by_key.get(key)
        if old is None:
            by_key[key] = p
            continue
        if RANK[p.decision] > RANK[old.decision]:
            old.decision = p.decision
        old.abstract = old.abstract or p.abstract
        old.url = old.url or p.url
        old.topic = old.topic or p.topic
        if "/poster/" in p.page and "/poster/" not in old.page:
            old.page = p.page
    return list(by_key.values())


def papers(venue: Venue, years: list[int], cache: Path | None = None, fetcher: Fetcher = _fetch, pause: float = 2.0) -> list[Paper]:
    for year in years:
        if year not in venue.years:
            raise SourceError(f"{venue.name} {year}: not a year the atlas has ({', '.join(map(str, venue.years))})")
    if venue.anthology:
        return parse_anthology(anthology_raw(cache, fetcher), venue, years)
    out: list[Paper] = []
    for i, year in enumerate(years):
        cached = cache is not None and (cache / f"{venue.key}-{year}.json").is_file()
        if i and not cached:
            time.sleep(pause)  # one file per year, a little apart
        out += parse(raw(venue, year, cache, fetcher), venue, year)
    return out
