"""The map: each paper's nearest neighbours by meaning, and a 2-D layout that keeps neighbours together.

The layout is UMAP's idea in a small, deterministic numpy form: a fuzzy graph of the k nearest neighbours (each
distance scaled by the point's own local spread), then an embedding in the plane by stochastic gradient descent that
pulls the graph's edges together and pushes random pairs apart (McInnes, Healy and Melville 2018, arXiv:1802.03426;
its curve parameters for min_dist 0.1 and spread 1). Exact t-SNE needs n² memory and time; this needs n·k.
"""

from __future__ import annotations

import numpy as np

A, B = 1.577, 0.895  # UMAP's fitted curve for min_dist=0.1, spread=1.0


def neighbours(x: np.ndarray, k: int, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For rows of unit vectors: each row's k nearest other rows (indices) and their cosine distances, nearest first.

    >>> x = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    >>> x /= np.linalg.norm(x, axis=1, keepdims=True)
    >>> neighbours(x, 1)[0].ravel().tolist()
    [1, 0, 3, 2]
    """
    n = len(x)
    k = min(k, n - 1)
    idx = np.zeros((n, k), dtype=np.int32)
    dist = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, block):
        sim = x[start : start + block] @ x.T
        rows = np.arange(sim.shape[0])
        sim[rows, start + rows] = -np.inf  # not itself
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k] if k < n - 1 else np.argsort(-sim, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, part, axis=1), axis=1)
        top = np.take_along_axis(part, order, axis=1)
        idx[start : start + block] = top
        dist[start : start + block] = np.clip(1 - np.take_along_axis(sim, top, axis=1), 0, 2)
    return idx, dist


def fuzzy_graph(idx: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UMAP's fuzzy union of the kNN graph: edges (i, j) and their weights in (0, 1]."""
    n, k = idx.shape
    rho = dist[:, 0]
    target = np.log2(k)
    lo, hi = np.zeros(n), np.full(n, np.inf)
    sigma = np.ones(n)
    for _ in range(64):  # binary search per point: sum of exp(-(d - rho) / sigma) = log2(k)
        psum = np.exp(-np.maximum(dist - rho[:, None], 0) / sigma[:, None]).sum(axis=1)
        high = psum > target
        hi = np.where(high, sigma, hi)
        lo = np.where(high, lo, sigma)
        sigma = np.where(np.isinf(hi), sigma * 2, (lo + hi) / 2)
    sigma = np.maximum(sigma, 1e-3 * max(float(dist.mean()), 1e-6))
    w = np.exp(-np.maximum(dist - rho[:, None], 0) / sigma[:, None]).ravel()
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel().astype(np.int64)
    # the union: w_ij + w_ji - w_ij·w_ji, over both directions of each edge
    key = np.concatenate([rows * n + cols, cols * n + rows])
    vals = np.concatenate([w, w])
    order = np.argsort(key, kind="stable")
    key, vals = key[order], vals[order]
    uniq, start, count = np.unique(key, return_index=True, return_counts=True)
    first = vals[start]
    second = np.where(count > 1, vals[np.minimum(start + 1, len(vals) - 1)], 0.0)
    # a pair seen from both sides appears twice per direction; (w + w') - w·w' for those, w for the others
    weight = np.where(count > 1, first + second - first * second, first)
    i, j = uniq // n, uniq % n
    keep = i < j
    return i[keep].astype(np.int32), j[keep].astype(np.int32), weight[keep].astype(np.float32)


def layout(x: np.ndarray, k: int = 15, epochs: int | None = None, seed: int = 0) -> np.ndarray:
    """2-D coordinates for the rows of x (unit vectors), scaled to [0, 1] on each axis."""
    n = len(x)
    if n < 3:
        return np.zeros((n, 2), dtype=np.float32) + 0.5
    rng = np.random.default_rng(seed)
    idx, dist = neighbours(x, k)
    i, j, w = fuzzy_graph(idx, dist)
    # start from the first two principal components (stable from run to run), spread to about 10 units
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred[: min(n, 20000)], full_matrices=False)
    y = centred @ vt[:2].T
    y = (10 * y / (np.abs(y).max() + 1e-12)).astype(np.float32)
    epochs = epochs or (500 if n <= 10000 else 250)
    negatives = 5
    for e in range(epochs):
        lr = 1.0 - e / epochs
        pick = rng.random(len(w)) < w  # each edge in proportion to its weight
        a_i, a_j = i[pick], j[pick]
        diff = y[a_i] - y[a_j]
        d2 = (diff * diff).sum(axis=1)
        coeff = (-2 * A * B * np.power(d2 + 1e-12, B - 1)) / (A * np.power(d2, B) + 1)
        g = np.clip(coeff[:, None] * diff, -4, 4) * lr
        np.add.at(y, a_i, g)
        np.add.at(y, a_j, -g)
        # pushed apart from random others
        src = np.repeat(a_i, negatives)
        other = rng.integers(0, n, len(src))
        diff = y[src] - y[other]
        d2 = (diff * diff).sum(axis=1)
        coeff = (2 * B) / ((0.001 + d2) * (A * np.power(d2, B) + 1))
        g = np.clip(coeff[:, None] * diff, -4, 4) * lr
        g[src == other] = 0
        np.add.at(y, src, g)
    lo, hi = y.min(axis=0), y.max(axis=0)
    return ((y - lo) / np.where(hi > lo, hi - lo, 1)).astype(np.float32)
