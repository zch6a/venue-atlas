"""Sources, layout, topics, building and the command, on made-up schedule files and a stand-in model (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import FakeModel, schedule

from venue_atlas import build, cli, sources
from venue_atlas.layout import layout, neighbours
from venue_atlas.sources import VENUES, SourceError, papers, parse
from venue_atlas.topics import kmeans, merge_small, name_topics

ICLR = VENUES["iclr"]


def test_parse_merges_talk_and_poster_and_drops_sessions() -> None:
    extra = [{"name": "A duplicate listing", "abstract": "<p>" + "same words " * 40 + "</p>", "eventtype": "Poster"},
             {"name": "The same paper, retitled", "abstract": "same words " * 40, "eventtype": "Poster", "decision": "Accept (spotlight)"},
             {"name": "Invited Talk: Someone", "abstract": "a talk", "eventtype": "Invited Talk"}]  # fmt: skip
    ps = parse(schedule(2024, extra=extra), ICLR, 2024)
    assert len(ps) == 36 + 1  # 3 themes × 12; the oral's two entries one; the session and talk gone; the retitled one merged
    first = next(p for p in ps if p.title.startswith("Diffusion sampler for diffusion study 0"))
    assert first.decision == "oral" and first.page == "https://iclr.cc/virtual/2024/poster/1"  # the poster's page kept
    assert first.url == "https://openreview.net/forum?id=p20241" and first.topic == "Topic->diffusion"
    merged = next(p for p in ps if p.title == "A duplicate listing")
    assert merged.decision == "spotlight" and "<p>" not in merged.abstract


@pytest.mark.parametrize("data", [b"not json", b'{"results": 5}', b"[]"])
def test_a_file_it_cannot_read_is_an_error(data: bytes) -> None:
    with pytest.raises(SourceError):
        parse(data, ICLR, 2024)


def test_downloads_are_cached_and_years_checked(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return schedule(2024)

    assert len(papers(ICLR, [2024], tmp_path, fetcher, pause=0)) == 36
    assert len(papers(ICLR, [2024], tmp_path, fetcher, pause=0)) == 36
    assert calls == ["https://iclr.cc/static/virtual/data/iclr-2024-orals-posters.json"]  # once
    with pytest.raises(SourceError, match="not a year"):
        papers(ICLR, [2019], tmp_path, fetcher)

    def broken(url: str) -> bytes:
        raise OSError("no route")

    with pytest.raises(SourceError, match="could not download"):
        papers(ICLR, [2023], tmp_path, broken)


def blobs(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.eye(3, 16, dtype=np.float32) * 4
    labels = np.repeat(np.arange(3), n // 3)
    x = centres[labels] + rng.standard_normal((len(labels), 16)).astype(np.float32) * 0.3
    return x / np.linalg.norm(x, axis=1, keepdims=True), labels


def test_layout_keeps_groups_together_and_is_repeatable() -> None:
    x, labels = blobs()
    xy = layout(x, k=10, epochs=200, seed=1)
    assert xy.shape == (60, 2) and xy.min() >= 0 and xy.max() <= 1
    assert np.array_equal(xy, layout(x, k=10, epochs=200, seed=1))
    d = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    near = np.argsort(d, axis=1)[:, 1:6]
    assert (labels[near] == labels[:, None]).mean() > 0.9
    idx, dist = neighbours(x, 5)
    assert (labels[idx] == labels[:, None]).mean() > 0.95 and (np.diff(dist, axis=1) >= -1e-6).all()


def test_topics_and_small_ones_merged() -> None:
    x, labels = blobs()
    got = kmeans(x, 3, seed=0)
    assert len({tuple(np.flatnonzero(got == c)) for c in range(3)}) == 3
    assert all(len(set(got[labels == c])) == 1 for c in range(3))
    tiny = got.copy()
    tiny[0] = 7  # a topic of one paper
    merged = merge_small(x, tiny, 3)
    assert merged.max() == 2 and merged[0] == merged[1]
    texts = ["diffusion models denoising"] * 20 + ["graph neural networks message passing"] * 20 + ["regret bounds for bandits"] * 20
    names = name_topics(texts, labels, 3)
    assert "diffusion" in " ".join(names[0]) and "message passing" in names[1] and any("regret" in t for t in names[2])


def test_the_atlas_has_titles_links_and_no_abstracts(tmp_path: Path) -> None:
    ps = parse(schedule(2023), ICLR, 2023) + parse(schedule(2024), ICLR, 2024)
    atlas = build.make(ICLR, ps, FakeModel(), epochs=100)
    doc = build.write(atlas, tmp_path, built="2026-11-14")
    build.write_model(FakeModel(), tmp_path)
    build.write_index(tmp_path, [doc])
    n = len(ps)
    cols = doc["papers"]
    assert all(len(v) == n for v in cols.values()) and doc["count"] == n and doc["years"] == [2023, 2024]
    assert len(doc["neighbours"]) == n * build.NEIGHBOURS and doc["model"]["dim"] == 32 and len(doc["model"]["mean"]) == 32
    assert (tmp_path / "data" / "iclr.vec").stat().st_size == n * 32
    assert sum(t["count"]["2024"] for t in doc["topics"]) == 36
    # the promise: abstracts are read to place the papers and not written anywhere
    for f in (tmp_path / "data").iterdir():
        assert b"secret-abstract" not in f.read_bytes(), f.name
    on_disk = json.loads((tmp_path / "data" / "iclr.json").read_text(encoding="utf-8"))
    assert on_disk["sources"] == [
        "https://iclr.cc/static/virtual/data/iclr-2023-orals-posters.json",
        "https://iclr.cc/static/virtual/data/iclr-2024-orals-posters.json",
    ]
    vocab = json.loads((tmp_path / "model" / "vocab.json").read_text(encoding="utf-8"))
    assert vocab["tokens"] == FakeModel.vocab and (tmp_path / "model" / "tokens.vec").stat().st_size == 4 * 32
    assert json.loads((tmp_path / "data" / "index.json").read_text(encoding="utf-8"))[0]["count"] == n


def test_too_few_papers_is_an_error() -> None:
    with pytest.raises(ValueError, match="too few"):
        build.make(ICLR, parse(schedule(2024, n_per_theme=0), ICLR, 2024), FakeModel())


def test_the_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "load", lambda: FakeModel())
    monkeypatch.setattr(cli, "papers", lambda venue, years, cache=None: [p for y in years for p in parse(schedule(y), venue, y)])
    out = tmp_path / "site"
    assert cli.main(["build", "iclr", "--years", "2024", "--out", str(out)]) == 0
    assert "ICLR: 36 papers (2024)" in capsys.readouterr().out
    assert cli.main(["build", "icml", "--years", "2024", "--out", str(out)]) == 0
    capsys.readouterr()
    index = json.loads((out / "data" / "index.json").read_text(encoding="utf-8"))
    assert [d["venue"] for d in index] == ["iclr", "icml"]  # the earlier venue kept, in the list's order
    assert cli.main(["search", "regret bandit exploration bound", "--atlas", str(out), "--venue", "iclr", "-k", "3", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 3 and all("bandits" in r["title"] for r in rows)
    assert cli.main(["venues"]) == 0 and "cvpr" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["build", "iclr", "--years", "twenty"])


def test_sources_doc_says_what_is_kept() -> None:
    assert "not kept" in (sources.__doc__ or "")
