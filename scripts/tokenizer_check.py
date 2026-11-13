"""Compares venue_atlas.wordpiece with Hugging Face's `tokenizers` on every title and abstract of the venues given
(from the build's cache; nothing is downloaded but what a build would), and on test strings. Prints how many texts
were compared and every text whose tokens differ. Needs the model (`venue-atlas model`) and `tokenizers`:

    uv run --with tokenizers python scripts/tokenizer_check.py --cache .cache/venue-atlas iclr acl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from venue_atlas.embed import model_dir  # noqa: E402
from venue_atlas.sources import VENUES, papers  # noqa: E402
from venue_atlas.wordpiece import from_tokenizer_json  # noqa: E402

EDGE = ["ΣΟΦΟΣ λόγος ΟΔΟΣ ΣΣ", "a [MASK] and [CLS] or [mask] [SEP][PAD]", "\x0b\x0cvertical﻿tab\x85next line",
        "İstanbul ǅemal ﬁne ß ŉ", "\U0001f680‍♂ ퟿ \U000e0001 emoji", "Ｆｕｌｌ ｗｉｄｔｈ ①②", "á̂ ẹ̈"]  # fmt: skip


def main() -> int:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("venues", nargs="*", default=list(VENUES))
    ap.add_argument("--cache", type=Path, default=Path(".cache/venue-atlas"))
    args = ap.parse_args()
    path = model_dir() / "tokenizer.json"
    theirs = Tokenizer.from_file(str(path))
    ours = from_tokenizer_json(json.loads(path.read_text(encoding="utf-8")))
    texts = list(EDGE)
    for key in args.venues:
        v = VENUES[key]
        ps = papers(v, list(v.years), cache=args.cache)
        texts += [p.title for p in ps] + [p.abstract for p in ps]
    t0 = time.perf_counter()
    mine = [ours.ids(t) for t in texts]
    t_ours = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = [e.ids for e in theirs.encode_batch(texts, add_special_tokens=False)]
    t_theirs = time.perf_counter() - t0
    bad = [(t, a, b) for t, a, b in zip(texts, mine, ref, strict=True) if a != b]
    tokens = sum(len(b) for b in ref)
    print(f"{len(texts):,} texts, {tokens:,} tokens: {len(bad)} differ (this tokenizer {t_ours:.1f} s, tokenizers {t_theirs:.1f} s)")
    for t, _, b in bad[:20]:
        print(f"  {t[:100]!r}\n    ours   {ours.tokens(t)[:30]}\n    theirs {[ours.by_id.get(i, i) for i in b][:30]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
