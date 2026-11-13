"""Broken inputs for the readers: the schedule-file reader (sources.parse), the ACL Anthology reader
(sources.parse_anthology), the LaTeX cleaner and the tokenizer. Mutations of real files from a build's cache (bytes
flipped, cut short, JSON values of the wrong type) and random text. A reader must give papers or a SourceError, never
another exception, and must not be slow. Nothing is downloaded.

    uv run python scripts/fuzz.py --cache .cache/venue-atlas --rounds 2000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from venue_atlas.anthology import latex_text, maths_text  # noqa: E402
from venue_atlas.sources import VENUES, SourceError, parse, parse_anthology  # noqa: E402
from venue_atlas.wordpiece import WordPiece  # noqa: E402

ODD = [None, True, 0, -1, 1.5, "", "x", [], {}, [1, "a"], {"a": None}, "<b>", "\\$", "\x00", "\ud800", "9" * 400]
LATEX = ["\\", "{", "}", "$", "\\'", '\\"', "\\v", "\\c", "\\i", "\\textit{", "$\\mathbb{", "^", "_", "--", "``", "''", "~",
         "\\(", "\\)", "\\\\", "\\&", "\\%", "é", "\x00", "\\frac{a}{b}", "\\L", "{\\o}"]  # fmt: skip


def mutate_bytes(data: bytes, rng: random.Random) -> bytes:
    b = bytearray(data)
    kind = rng.randrange(4)
    if kind == 0 and b:  # flip bytes
        for _ in range(rng.randint(1, 20)):
            b[rng.randrange(len(b))] = rng.randrange(256)
    elif kind == 1:  # cut short
        b = b[: rng.randrange(len(b) + 1)]
    elif kind == 2 and b:  # a piece repeated
        i = rng.randrange(len(b))
        b[i:i] = b[i : i + rng.randint(1, 5000)] * rng.randint(1, 5)
    else:  # a piece removed
        i = rng.randrange(len(b) + 1)
        del b[i : i + rng.randint(1, 5000)]
    return bytes(b)


def mutate_json(doc: object, rng: random.Random) -> object:
    """A value somewhere in the document replaced with one of another type."""
    if isinstance(doc, dict) and doc and rng.random() < 0.9:
        k = rng.choice(list(doc))
        doc[k] = mutate_json(doc[k], rng) if rng.random() < 0.6 else rng.choice(ODD)
    elif isinstance(doc, list) and doc and rng.random() < 0.9:
        i = rng.randrange(len(doc))
        doc[i] = mutate_json(doc[i], rng) if rng.random() < 0.6 else rng.choice(ODD)
    else:
        return rng.choice(ODD)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path(".cache/venue-atlas"))
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    schedules = []
    for v in VENUES.values():
        if v.anthology:
            continue
        files = sorted(args.cache.glob(f"{v.key}-*.json"))
        if files:
            f = rng.choice(files)
            schedules.append((v, int(f.stem.split("-")[1]), f.read_bytes()))
    bib_file = args.cache / "anthology+abstracts.bib.gz"
    bib = gzip.decompress(bib_file.read_bytes()) if bib_file.is_file() else b""
    bib = bib[bib.find(b"\n@", len(bib) // 2) :][:3_000_000]  # 3 MB from the middle: recent volumes
    acl = VENUES["acl"]
    failures: list[str] = []
    slowest = (0.0, "")

    def run(label: str, f: object, *a: object) -> None:
        nonlocal slowest
        t = time.perf_counter()
        try:
            f(*a)  # type: ignore[operator]
        except SourceError:
            pass
        except Exception:
            failures.append(f"{label}: {traceback.format_exc(limit=3)}")
        took = time.perf_counter() - t
        if took > slowest[0]:
            slowest = (took, label)

    vocab = {"[UNK]": 0, "[MASK]": 1, "a": 2, "##b": 3, "é": 4}
    tok = WordPiece(vocab, "[UNK]", ["[MASK]"])
    t0 = time.perf_counter()
    for r in range(args.rounds):
        for v, year, data in schedules:
            run(f"{v.key} {year} bytes #{r}", parse, mutate_bytes(data, rng), v, year)
            try:
                doc = json.loads(data)
                for _ in range(rng.randint(1, 8)):
                    doc = mutate_json(doc, rng)
                run(f"{v.key} {year} json #{r}", parse, json.dumps(doc).encode(), v, year)
            except ValueError:
                pass
        if bib and r % 10 == 0:
            run(f"anthology bytes #{r}", parse_anthology, mutate_bytes(bib, rng), acl, list(acl.years))
        text = "".join(rng.choice(LATEX) if rng.random() < 0.5 else chr(rng.randrange(0x20, 0x3000)) for _ in range(rng.randint(0, 300)))
        run(f"latex {text[:40]!r}", latex_text, text)
        run(f"maths {text[:40]!r}", maths_text, "$" + text + "$")
        run(f"wordpiece {text[:40]!r}", tok.ids, text + "".join(chr(rng.randrange(0x110000)) for _ in range(50)).encode("utf-8", "replace").decode())
    print(f"{args.rounds} rounds, {len(schedules)} schedule files, {len(bib) // 1_000_000} MB of BibTeX, {time.perf_counter() - t0:.0f} s: "
          f"{len(failures)} failures; slowest {slowest[0]:.2f} s ({slowest[1][:60]})")  # fmt: skip
    seen = set()
    for f in failures:
        key = f.strip().splitlines()[-1]
        if key not in seen:
            seen.add(key)
            print(f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
