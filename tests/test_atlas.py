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


BIB = """
@proceedings{acl-2024-long,
    title = "Proceedings of the 62nd Annual Meeting",
    url = "https://aclanthology.org/2024.acl-long.0/"
}
@inproceedings{a-2024-one,
    title = "{IAG}: Induction-Augmented Generation for Sch{\\"o}n",
    author = "A, B",
    year = "2024",
    url = "https://aclanthology.org/2024.acl-long.1/",
    abstract = "We study secret-abstract-acl things with $x^2$ maths.",
    pages = "1--17"
}
@inproceedings{b-2024-two,
    title = "A Short Paper on Gonz{\\'a}lez",
    url = "https://aclanthology.org/2024.acl-short.7/",
    abstract = "Short and to the point."
}
@inproceedings{c-2024-findings,
    title = "Findings Are Not Main",
    url = "https://aclanthology.org/2024.findings-acl.3/",
    abstract = "Left out."
}
@inproceedings{d-2024-dup,
    title = "{IAG}: Induction-Augmented Generation for Sch{\\"o}n",
    url = "https://aclanthology.org/2024.acl-long.99/",
    abstract = "The same title again."
}
"""


def test_the_anthology_main_conference_papers(tmp_path: Path) -> None:
    import gzip
    import os
    import time

    acl = VENUES["acl"]
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return gzip.compress(BIB.encode())

    ps = papers(acl, [2024], tmp_path, fetcher)
    assert [(p.title, p.topic, p.url) for p in ps] == [
        ("IAG: Induction-Augmented Generation for Schön", "long paper", "https://aclanthology.org/2024.acl-long.1/"),
        ("A Short Paper on González", "short paper", "https://aclanthology.org/2024.acl-short.7/"),
    ]
    assert "secret-abstract-acl" in ps[0].abstract and ps[0].decision == ""
    papers(acl, [2024], tmp_path, fetcher)
    assert len(calls) == 1  # cached
    old = time.time() - 40 * 86400
    os.utime(tmp_path / "anthology+abstracts.bib.gz", (old, old))
    papers(VENUES["emnlp"], [2024], tmp_path, lambda u: gzip.compress(BIB.replace("acl-long", "emnlp-main").encode()))
    with pytest.raises(SourceError, match="no papers"):
        papers(VENUES["naacl"], [2024], tmp_path, fetcher)
    with pytest.raises(SourceError, match="not a year"):
        papers(VENUES["naacl"], [2023], tmp_path, fetcher)
    atlas = build.make(acl, ps * 3 if len(ps) < 3 else ps, FakeModel(), epochs=50) if False else None
    assert atlas is None


def test_links_paper_page_and_project() -> None:
    records = [
        {"name": "Old Style Entry", "abstract": "x " * 50, "id": "3238", "eventtype": "Poster", "paper_pdf_url": "https://openreview.net/pdf?id=abc"},
        {"name": "Project Linked", "abstract": "y " * 50, "id": "9", "eventtype": "Poster", "paper_url": "https://openreview.net/forum?id=q",
         "url": "https://github.com/a/b", "virtualsite_url": "/virtual/2024/poster/9"},
        {"name": "Odd Fields", "abstract": None, "id": None, "eventtype": None, "url": "javascript:alert(1)", "paper_url": "ftp://x"},
    ]  # fmt: skip
    import json as _json

    ps = {p.title: p for p in parse(_json.dumps({"results": records}).encode(), ICLR, 2021)}
    old = ps["Old Style Entry"]
    assert old.url == "https://openreview.net/forum?id=abc" and old.page == "https://iclr.cc/virtual/2021/poster/3238" and old.project == ""
    linked = ps["Project Linked"]
    assert linked.project == "https://github.com/a/b" and linked.page == "https://iclr.cc/virtual/2024/poster/9"
    odd = ps["Odd Fields"]
    assert (odd.url, odd.page, odd.project, odd.abstract) == ("", "", "", "")


def test_review_round_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import gzip
    import http.client
    import json as _json

    # a real paper titled like a session stays; a session-like entry with nothing of a paper goes
    records = [{"name": "Social Reward: Evaluating Rewards", "abstract": "a " * 50, "paper_url": "https://openreview.net/forum?id=s", "decision": "Accept (poster)"},
               {"name": "Panel Discussion on Ethics", "abstract": "talk", "eventtype": "Panel"},
               {"name": "Affinity-Aware Graph Networks", "abstract": "b " * 50, "decision": "Accept (poster)"}]  # fmt: skip
    titles = [p.title for p in parse(_json.dumps({"results": records}).encode(), ICLR, 2024)]
    assert titles == ["Social Reward: Evaluating Rewards", "Affinity-Aware Graph Networks"]
    # a download that is not a schedule file is refused and not cached; a cut-off reply is an error, not a traceback
    with pytest.raises(SourceError, match="not a JSON"):
        papers(ICLR, [2024], tmp_path, lambda u: b"<html>maintenance</html>")
    assert not (tmp_path / "iclr-2024.json").exists()

    def cut(u: str) -> bytes:
        raise http.client.IncompleteRead(b"")

    with pytest.raises(SourceError, match="could not download"):
        papers(ICLR, [2024], tmp_path, cut)
    with pytest.raises(SourceError, match="broken archive"):
        papers(VENUES["acl"], [2024], tmp_path, lambda u: gzip.compress(b"@inproceedings{x}" * 100)[:-20])
    assert not (tmp_path / "anthology+abstracts.bib.gz").exists()
    # a cached file that no longer reads is fetched again
    (tmp_path / "iclr-2023.json").write_bytes(b"garbage")
    assert len(papers(ICLR, [2023], tmp_path, lambda u: schedule(2023), pause=0)) == 36
    # every topic small: still merged into the nearest
    x = np.array([[1, 0], [0.99, 0.1], [0.98, 0.2], [0, 1], [0.1, 0.99], [0.2, 0.98], [0.7, 0.7]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    assert len(set(merge_small(x, np.array([0, 0, 0, 1, 1, 1, 2]), 5).tolist())) == 1
    # the neighbours' count is in the atlas; search checks -k and a query of no known words
    ps = parse(schedule(2024), ICLR, 2024)
    doc = build.write(build.make(ICLR, ps, FakeModel(), epochs=50), tmp_path / "site")
    assert doc["neighbour_count"] == build.NEIGHBOURS and len(doc["neighbours"]) == len(ps) * build.NEIGHBOURS
    monkeypatch.setattr(cli, "load", lambda: FakeModel())
    assert cli.main(["search", "regret", "--atlas", str(tmp_path / "site"), "--venue", "iclr", "-k", "0"]) == 2
    assert cli.main(["search", "123 456", "--atlas", str(tmp_path / "site"), "--venue", "iclr"]) == 2
    assert "vocabulary" in capsys.readouterr().err
    (tmp_path / "site" / "data" / "iclr.json").write_text("{}", encoding="utf-8")
    assert cli.main(["search", "regret", "--atlas", str(tmp_path / "site"), "--venue", "iclr"]) == 2
    assert "build it again" in capsys.readouterr().err
