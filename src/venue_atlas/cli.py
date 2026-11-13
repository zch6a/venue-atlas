"""venue-atlas: maps of the papers accepted at a conference, placed by meaning and grouped into topics.

venue-atlas model                              download the embedding model (30 MB, pinned, once)
venue-atlas build iclr neurips --out site      build a site (the page and the atlases) for these venues, all years
venue-atlas build icml --years 2024,2025       some years only
venue-atlas search "diffusion for proteins" --atlas site --venue neurips   the closest papers, from a built atlas
venue-atlas venues                             the venues and years it knows

Each venue's accepted papers come from the conference's own virtual-site schedule file (one download per venue and
year, kept in --cache); ACL, EMNLP and NAACL from the ACL Anthology's BibTeX (one download for all three). Abstracts are read to place the papers and are not written anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .build import centre, make, venues_from, write, write_index, write_model, write_page
from .embed import EmbedError, download, load
from .sources import VENUES, SourceError, papers


def _years(text: str) -> list[int]:
    try:
        out = sorted({int(y) for y in text.split(",") if y.strip()})
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"years like 2024,2025, not {text!r}") from e
    if not out:
        raise argparse.ArgumentTypeError("no years given")
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="venue-atlas", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"venue-atlas {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("model", help="download the embedding model (pinned by revision and SHA-256)")
    sub.add_parser("venues", help="the venues and years it knows")
    b = sub.add_parser("build", help="build atlases into a site folder")
    b.add_argument("venues", nargs="+", choices=sorted(VENUES), metavar="VENUE", help=f"one or more of {', '.join(VENUES)}")
    b.add_argument("--years", type=_years, help="these years only (default: all it knows)")
    b.add_argument("--out", type=Path, default=Path("site"), help="the site folder to write the page, data/ and model/ into (default: site)")
    b.add_argument("--cache", type=Path, default=Path(".cache/venue-atlas"), help="where the downloaded schedule files are kept")
    b.add_argument("--seed", type=int, default=0, help="the layout's and topics' random seed (default 0)")
    s = sub.add_parser("search", help="the papers closest in meaning to a query, from a built atlas")
    s.add_argument("query")
    s.add_argument("--atlas", type=Path, default=Path("site"), help="the site folder holding data/ (default: site)")
    s.add_argument("--venue", required=True, choices=sorted(VENUES))
    s.add_argument("-k", type=int, default=10, help="how many (default 10)")
    s.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.command == "model":
            print(f"the model is in {download()}")
            return 0
        if args.command == "venues":
            for v in VENUES.values():
                print(f"{v.key:8} {v.name:8} {', '.join(map(str, v.years))}")
            return 0
        if args.command == "build":
            return _build(args)
        return _search(args)
    except (EmbedError, SourceError, ValueError, OSError) as e:
        print(f"venue-atlas: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("venue-atlas: stopped", file=sys.stderr)
        return 130


def _build(args: argparse.Namespace) -> int:
    if args.out.exists() and not args.out.is_dir():
        raise ValueError(f"--out {args.out}: a file, not a folder")  # said now, not after the downloads
    model = load()
    docs = []
    index_file = args.out / "data" / "index.json"
    if index_file.is_file():  # venues built before stay in the index
        try:
            old = {d["venue"]: d for d in json.loads(index_file.read_text(encoding="utf-8"))}
        except (ValueError, KeyError, TypeError):
            old = {}
    else:
        old = {}
    for venue in venues_from(args.venues):
        years = args.years or list(venue.years)
        t0 = time.perf_counter()
        ps = papers(venue, years, cache=args.cache)
        atlas = make(venue, ps, model, seed=args.seed)
        doc = write(atlas, args.out)
        old.pop(venue.key, None)
        docs.append(doc)
        print(f"{venue.name}: {len(ps):,} papers ({', '.join(map(str, years))}), {len(atlas.topic_terms)} topics, {time.perf_counter() - t0:.0f} s")
    write_model(model, args.out)
    for key, d in old.items():
        if (args.out / "data" / f"{key}.json").is_file():
            docs.append(d)
    order = list(VENUES)
    write_index(args.out, sorted(docs, key=lambda d: order.index(str(d["venue"]))))
    page = write_page(args.out)
    print(f"wrote {args.out.as_posix()}/data and {args.out.as_posix()}/model" + (" and the page" if page else ""))
    if not page:
        print("venue-atlas: this installation has no page files (index.html); copy site/ from the repository", file=sys.stderr)
    return 0


def _search(args: argparse.Namespace) -> int:
    if args.k < 1:
        raise ValueError("-k must be 1 or more")
    path = args.atlas / "data" / f"{args.venue}.json"
    if not path.is_file() or not path.with_suffix(".vec").is_file():
        raise ValueError(f"no {VENUES[args.venue].name} atlas in {args.atlas}: build it first (venue-atlas build {args.venue} --out {args.atlas})")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        dim = int(doc["model"]["dim"])
        _ = doc["papers"]["title"], doc["papers"]["scale"], doc["topics"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path}: not an atlas this version reads ({e}); build it again") from e
    vec = np.frombuffer((args.atlas / "data" / f"{args.venue}.vec").read_bytes(), dtype=np.int8).reshape(-1, dim)
    scale = np.asarray(doc["papers"]["scale"], dtype=np.float32)
    model = load()
    if not model.ids([args.query])[0]:
        raise ValueError(f"no word of {args.query!r} is in the model's vocabulary")
    mean = np.asarray(doc["model"].get("mean") or np.zeros(dim), dtype=np.float32)
    q = centre(model.encode([args.query]), mean)[0]
    scores = (vec.astype(np.float32) @ q) * scale
    top = np.argsort(-scores)[: args.k]
    p = doc["papers"]
    rows = [{"title": p["title"][i], "year": p["year"][i], "score": round(float(scores[i]), 3), "url": p["url"][i] or p["page"][i],
             "topic": " · ".join(doc["topics"][p["topic"][i]]["terms"])} for i in top]  # fmt: skip
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for r in rows:
            print(f"{r['score']:.3f}  {r['title']} ({r['year']})\n       {r['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
