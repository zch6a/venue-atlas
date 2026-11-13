"""Topics: the papers grouped by meaning (spherical k-means on the embeddings), each group named by the words that
set it apart (class-based TF-IDF over titles and abstracts), and how its share changed from year to year."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOP = set(
    [
        "a",
        "about",
        "above",
        "across",
        "after",
        "again",
        "against",
        "all",
        "almost",
        "along",
        "also",
        "although",
        "always",
        "am",
        "among",
        "an",
        "and",
        "another",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "either",
        "enough",
        "even",
        "ever",
        "every",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "here",
        "how",
        "however",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "least",
        "less",
        "let",
        "like",
        "made",
        "make",
        "makes",
        "many",
        "may",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "neither",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "per",
        "perhaps",
        "rather",
        "same",
        "several",
        "shall",
        "she",
        "should",
        "since",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "though",
        "through",
        "thus",
        "to",
        "too",
        "toward",
        "towards",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "use",
        "used",
        "uses",
        "using",
        "very",
        "via",
        "was",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "paper",
        "propose",
        "proposed",
        "proposes",
        "present",
        "presents",
        "introduce",
        "introduces",
        "approach",
        "approaches",
        "method",
        "methods",
        "model",
        "models",
        "result",
        "results",
        "show",
        "shows",
        "shown",
        "demonstrate",
        "demonstrates",
        "experiments",
        "experimental",
        "extensive",
        "based",
        "novel",
        "new",
        "existing",
        "work",
        "works",
        "task",
        "tasks",
        "problem",
        "problems",
        "performance",
        "state",
        "art",
        "sota",
        "framework",
        "frameworks",
        "achieve",
        "achieves",
        "achieving",
        "improve",
        "improves",
        "improved",
        "improvement",
        "significantly",
        "significant",
        "effective",
        "effectively",
        "efficient",
        "efficiently",
        "key",
        "first",
        "second",
        "two",
        "three",
        "however",
        "furthermore",
        "moreover",
        "additionally",
        "specifically",
        "particular",
        "various",
        "different",
        "large",
        "small",
        "high",
        "low",
        "better",
        "best",
        "standard",
        "recent",
        "recently",
        "code",
        "available",
        "github",
        "com",
        "https",
        "http",
        "www",
        "benchmark",
        "benchmarks",
        "outperforms",
        "outperform",
        "study",
        "studies",
        "analysis",
        "provide",
        "provides",
        "able",
        "enables",
        "enable",
        "allows",
        "allow",
        "learning",
        "learn",
        "learned",
        "training",
        "trained",
        "data",
        "dataset",
        "datasets",
        "neural",
        "network",
        "networks",
        "deep",
    ]
)


def kmeans(x: np.ndarray, k: int, iterations: int = 40, seed: int = 0) -> np.ndarray:
    """Cluster labels for rows of unit vectors (cosine k-means, k-means++ seeding, a fixed seed).

    >>> x = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]], dtype=np.float32)
    >>> labels = kmeans(x / np.linalg.norm(x, axis=1, keepdims=True), 2)
    >>> bool(labels[0] == labels[1] != labels[2] == labels[3])
    True
    """
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(n)]]
    best = 1 - x @ centres[0]
    for _ in range(1, k):
        p = np.maximum(best, 0) ** 2
        c = x[rng.choice(n, p=p / p.sum())] if p.sum() > 0 else x[rng.integers(n)]
        centres.append(c)
        best = np.minimum(best, 1 - x @ c)
    cent = np.stack(centres)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        new = np.argmax(x @ cent.T, axis=1).astype(np.int32)
        if np.array_equal(new, labels) and _:
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                m = members.sum(axis=0)
                cent[c] = m / (np.linalg.norm(m) + 1e-12)
    return labels


def merge_small(x: np.ndarray, labels: np.ndarray, minimum: int) -> np.ndarray:
    """Topics with fewer than `minimum` papers folded into the nearest larger topic (by centroid), and the labels
    renumbered 0..k-1, the largest topics first.

    >>> x = np.array([[1, 0], [0.98, 0.2], [0.95, 0.3], [0, 1]], dtype=np.float32)
    >>> merge_small(x / np.linalg.norm(x, axis=1, keepdims=True), np.array([0, 0, 1, 2]), 2).tolist()
    [0, 0, 0, 0]
    """
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = ids[counts < minimum]
        if len(small) == 0 or len(ids) == 1:
            break
        s = small[np.argmin(counts[np.isin(ids, small)])]  # the smallest first
        big = ids[counts >= minimum] if (counts >= minimum).any() else ids[ids != s]
        cent = np.stack([x[labels == c].mean(axis=0) for c in big])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        members = labels == s
        labels[members] = big[np.argmax(x[members] @ cent.T, axis=1)]
    ids, counts = np.unique(labels, return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    remap = {int(c): i for i, c in enumerate(order)}
    return np.array([remap[int(c)] for c in labels], dtype=np.int32)


def clean(text: str) -> str:
    """Prose without its maths: $...$ and \\(...\\) spans and LaTeX commands dropped (abstracts carry them).

    >>> clean("A bound of $O(\\\\sqrt{T})$ regret, with \\\\textit{no} \\\\(\\\\omega\\\\) terms")
    'A bound of regret, with no terms'
    """
    text = re.sub(r"\$\$.*?\$\$|\$[^$]{0,300}\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\[a-zA-Z]+\*?\s*\{([^{}]*)\}", r"\1", text)  # \textit{no} -> no
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def topic_count(n: int) -> int:
    """How many topics for n papers: about half the square root, between 8 and 48.

    >>> topic_count(500), topic_count(4000), topic_count(20000)
    (11, 32, 48)
    """
    return max(8, min(48, round(math.sqrt(n) / 2)))


def terms(text: str) -> list[str]:
    """Words and two-word phrases of a text, without stop words (lower case; hyphenated words kept whole).

    >>> terms("Diffusion Models for Text-to-Image Generation via Score Matching")
    ['diffusion', 'text-to-image', 'generation', 'score', 'matching', 'text-to-image generation', 'score matching']
    """
    words = [w for w in re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", text.lower())]
    out = [w for w in words if w not in STOP and len(w) > 2 and not w.isdigit()]
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:], strict=False) if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2]
    return out + bigrams


def name_topics(texts: list[str], labels: np.ndarray, k: int, top: int = 3) -> list[list[str]]:
    """Each topic's most distinctive terms: class-based TF-IDF (a term's frequency in the topic, times the log of how
    rare it is across topics), preferring a phrase to the words it contains."""
    counts: list[Counter[str]] = [Counter() for _ in range(k)]
    for text, label in zip(texts, labels, strict=True):
        counts[int(label)].update(set(terms(text)))  # once per paper: what many papers say, not what one repeats
    total: Counter[str] = Counter()
    for each in counts:
        total.update(each)
    sizes = [max(1, sum(1 for label in labels if label == c)) for c in range(k)]
    mean = sum(sizes) / k
    names = []
    for c in range(k):
        scored = []
        for t, f in counts[c].items():
            if f < 2:
                continue
            weight = f / sizes[c] * math.log(1 + mean / total[t])
            if " " in t:
                weight *= 1.15  # a phrase says more than either word
            scored.append((weight, t))
        scored.sort(reverse=True)
        chosen: list[str] = []
        for _, t in scored:
            if any(t in other or other in t for other in chosen):
                continue
            chosen.append(t)
            if len(chosen) == top:
                break
        names.append(chosen)
    return names
