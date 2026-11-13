"""Building an atlas: a venue's accepted papers over the years, embedded, laid out, grouped into topics, and written as
the site's data (columns of titles, links, years, coordinates and topics; each paper's nearest neighbours; int8
vectors for search in the browser). Abstracts are used for the embeddings and not written anywhere."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import MODEL, StaticModel, quantize
from .layout import layout, neighbours
from .sources import VENUES, Paper, Venue, url_for
from .topics import clean, kmeans, merge_small, name_topics, topic_count

NEIGHBOURS = 8
DECISIONS = ["", "poster", "spotlight", "oral"]


@dataclass
class Atlas:
    venue: Venue
    papers: list[Paper]
    xy: np.ndarray
    topic: np.ndarray
    topic_terms: list[list[str]]
    near: np.ndarray
    vectors: np.ndarray  # unit vectors after the venue's mean is taken away, float32
    mean: np.ndarray  # that mean: a query is centred the same way before it is compared


def text_of(p: Paper) -> str:
    return clean(f"{p.title}. {p.abstract}" if p.abstract else p.title)


def make(venue: Venue, papers: list[Paper], model: StaticModel, seed: int = 0, epochs: int | None = None) -> Atlas:
    if len(papers) < 3:
        raise ValueError(f"{venue.name}: {len(papers)} papers, too few for a map")
    texts = [text_of(p) for p in papers]
    raw = model.encode(texts)
    # what all the venue's papers share (static embeddings have a large common part) taken away: neighbours by what
    # sets papers apart (on ICLR 2023-25 the share of each paper's 10 nearest in its own topic went from 0.48 to 0.52)
    mean = raw.mean(axis=0)
    vectors = centre(raw, mean)
    xy = layout(vectors, k=20, seed=seed, epochs=epochs)
    topic = kmeans(vectors, topic_count(len(papers)), seed=seed)
    topic = merge_small(vectors, topic, max(5, len(papers) // (4 * topic_count(len(papers)))))
    k = int(topic.max()) + 1
    terms = name_topics(texts, topic, k)
    near, _ = neighbours(vectors, NEIGHBOURS)
    return Atlas(venue, papers, xy, topic, terms, near, vectors, mean)


def centre(vectors: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Vectors with the mean taken away, normalized again.

    >>> centre(np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)).tolist()
    [[0.0, 1.0], [0.0, -1.0]]
    """
    out = vectors - mean
    return (out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def write(atlas: Atlas, out: Path, built: str | None = None) -> dict[str, object]:
    """The venue's files in out/data/: <venue>.json (everything but the vectors) and <venue>.vec (int8, n × dim)."""
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    p = atlas.papers
    years = sorted({x.year for x in p})
    conf_topics = sorted({x.topic for x in p if x.topic})
    conf_index = {t: i for i, t in enumerate(conf_topics)}
    q, scale = quantize(atlas.vectors)
    topics = []
    for c, words in enumerate(atlas.topic_terms):
        members = atlas.topic == c
        counts = {str(y): int(sum(1 for x, m in zip(p, members, strict=True) if m and x.year == y)) for y in years}
        topics.append({"terms": words, "count": counts})
    doc = {
        "format": "venue-atlas/1",
        "venue": atlas.venue.key,
        "name": atlas.venue.name,
        "years": years,
        "built": built or datetime.date.today().isoformat(),
        "model": {
            "name": MODEL.name,
            "revision": MODEL.revision,
            "dim": int(atlas.vectors.shape[1]),
            "mean": [float(f"{v:.6g}") for v in atlas.mean],
        },  # fmt: skip
        "sources": list(dict.fromkeys(url_for(atlas.venue, y) for y in years)),
        "count": len(p),
        "papers": {
            "title": [x.title for x in p],
            "year": [x.year for x in p],
            "x": [round(float(v), 4) for v in atlas.xy[:, 0]],
            "y": [round(float(v), 4) for v in atlas.xy[:, 1]],
            "topic": [int(t) for t in atlas.topic],
            "decision": [DECISIONS.index(x.decision) for x in p],
            "conf_topic": [conf_index.get(x.topic, -1) for x in p],
            "url": [x.url for x in p],
            "page": [x.page for x in p],
            "project": [x.project for x in p],
            "scale": [float(f"{s:.6g}") for s in scale],
        },
        "neighbour_count": int(atlas.near.shape[1]),
        "neighbours": atlas.near.astype(int).ravel().tolist(),
        "topics": topics,
        "conf_topics": conf_topics,
    }
    (data_dir / f"{atlas.venue.key}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (data_dir / f"{atlas.venue.key}.vec").write_bytes(q.tobytes())
    return doc


def write_model(model: StaticModel, out: Path) -> None:
    """The model for the browser: out/model/vocab.json (tokens in id order, the unknown token, each token's scale) and
    out/model/tokens.vec (int8, vocab × dim)."""
    d = out / "model"
    d.mkdir(parents=True, exist_ok=True)
    table = model.token_table()
    q, scale = quantize(table)
    vocab = model.tokenizer.get_vocab()
    tokens = [""] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    meta = {"name": MODEL.name, "revision": MODEL.revision, "dim": int(table.shape[1]), "unk": model.unk, "normalize": model.normalize,
            "lowercase": True, "prefix": "##", "max_chars": 100, "tokens": tokens, "scale": [float(f"{s:.6g}") for s in scale]}  # fmt: skip
    (d / "vocab.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (d / "tokens.vec").write_bytes(q.tobytes())


def write_index(out: Path, docs: list[dict[str, object]]) -> None:
    """out/data/index.json: the venues built, their years and paper counts."""
    index = [{"venue": d["venue"], "name": d["name"], "years": d["years"], "count": d["count"], "built": d["built"]} for d in docs]
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


def venues_from(names: list[str]) -> list[Venue]:
    out = []
    for n in names:
        if n not in VENUES:
            raise ValueError(f"unknown venue {n!r}: one of {', '.join(VENUES)}")
        out.append(VENUES[n])
    return out
