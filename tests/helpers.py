"""Made-up schedule files and a stand-in for the embedding model, so that the tests need no network and no model."""

from __future__ import annotations

import hashlib
import json

import numpy as np

THEMES = {
    "diffusion": ["diffusion", "denoising", "score", "sampler", "image", "generation", "noise", "guidance"],
    "graphs": ["graph", "node", "edge", "message", "passing", "gnn", "molecule", "neighbourhood"],
    "bandits": ["regret", "bandit", "arm", "exploration", "bound", "online", "reward", "policy"],
}


def schedule(year: int, n_per_theme: int = 12, extra: list[dict] | None = None) -> bytes:
    """A virtual-site schedule file: posters of three themes, one oral listed twice (talk and poster), a session."""
    results = []
    k = 0
    for theme, words in THEMES.items():
        for i in range(n_per_theme):
            k += 1
            title = f"{words[i % 8].title()} {words[(i + 3) % 8]} for {theme} study {i} ({year})"
            abstract = " ".join(words[(i + j) % 8] for j in range(40)) + f" secret-abstract-{theme}-{i}-{year}"
            results.append({"name": title, "abstract": abstract, "uid": f"u{year}{k}", "eventtype": "Poster", "decision": "Accept (poster)",
                            "virtualsite_url": f"/virtual/{year}/poster/{k}", "paper_url": f"https://openreview.net/forum?id=p{year}{k}",
                            "topic": f"Topic->{theme}"})  # fmt: skip
    if results:
        results.append(dict(results[0], eventtype="Oral", decision="Accept (oral)", virtualsite_url=f"/virtual/{year}/oral/1"))
    results.append({"name": "Lightning Talks 4B-1", "abstract": "<ul><li><a href='/virtual/2022/poster/1'>x</a></li></ul>", "eventtype": "Spotlight"})
    results += extra or []
    return json.dumps({"count": len(results), "results": results}).encode()


class FakeModel:
    """Each word a fixed random vector (from its hash); a text is the mean of its words', normalized."""

    dim = 32
    normalize = True
    unk = 0

    def _word(self, w: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(w.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

    def encode(self, texts: list[str], batch: int = 512) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = [w for w in t.lower().replace(".", " ").split() if w.isalpha()]
            if words:
                out[i] = np.mean([self._word(w) for w in words], axis=0)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def token_table(self) -> np.ndarray:
        return np.stack([self._word(t) for t in self.vocab])

    vocab = ["[UNK]", "diffusion", "graph", "regret"]

    class _Tok:
        def get_vocab(self) -> dict[str, int]:
            return {t: i for i, t in enumerate(FakeModel.vocab)}

    tokenizer = _Tok()
