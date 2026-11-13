"""Writes tests/fixtures/wordpiece.json: test strings, the tokens potion-base-8M's own tokenizer gives them, and the
part of the vocabulary those tokens are (enough for WordPiece's longest match to give the same tokens). The site's
tokenizer is tested against it. Needs the model (`venue-atlas model`); run when the strings or the model change."""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from venue_atlas.embed import load  # noqa: E402

STRINGS = [
    "diffusion models for text-to-image generation",
    "Large Language Models (LLMs) as zero-shot reasoners",
    "Café naïve résumé — Ångström façade",
    "Chain-of-Thought prompting; in-context learning!",
    "3D Gaussian Splatting: real-time radiance fields",
    "GNNs, ViTs & RLHF: what's next?",
    "量子计算与机器学习",
    "Ｆｕｌｌｗｉｄｔｈ ｔｅｘｔ and tabs\there\nnew lines",
    "emoji 🚀 and symbols ∑ ∫ ≤ ≥",
    "supercalifragilisticexpialidocious" * 4,
    "x" * 120,
    "ResNet-50 achieves 76.1% top-1 on ImageNet-1k",
    "self-supervised contrastive pre-training (SimCLR, MoCo v2)",
    "sparse mixture-of-experts with top-2 gating",
    "  leading and trailing spaces  ",
    "control\u0007chars​and zero-width",
    "Ελληνικά και русский текст",
    "",
]


def main() -> None:
    model = load()
    vocab = model.tokenizer.get_vocab()
    by_id = {i: t for t, i in vocab.items()}
    cases = []
    used: dict[str, int] = {}
    for s in STRINGS:
        enc = model.tokenizer.encode(s, add_special_tokens=False)
        tokens = [by_id[i] for i in enc.ids]
        for t in tokens:
            used[t] = vocab[t]
        cases.append({"text": s, "tokens": tokens})
    unk = model.unk
    used[by_id[unk]] = unk
    out = {"model": "potion-base-8M", "unk": by_id[unk], "vocab": dict(sorted(used.items(), key=lambda kv: kv[1])), "cases": cases}
    path = root / "tests" / "fixtures" / "wordpiece.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(root).as_posix()}: {len(cases)} cases, {len(used)} tokens")


if __name__ == "__main__":
    main()
