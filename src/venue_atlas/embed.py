"""Embeddings with a static Model2Vec model (potion-base-8M), run with numpy and Hugging Face's `tokenizers`; the same
model is shipped to the site (its token vectors as int8) so that a search typed in the browser is embedded there,
the same way.

From semantic-find (the same author's MIT project): the model's files are pinned by revision, size and SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelInfo:
    name: str
    repo: str
    revision: str
    files: dict[str, tuple[int, str]]  # file name -> (size in bytes, SHA-256)


# Pinned by size and SHA-256 (from Hugging Face's listing, checked against a download).
MODEL = ModelInfo("potion-base-8M", "minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23", {
    "config.json": (202, "2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553"),
    "tokenizer.json": (683666, "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50"),
    "model.safetensors": (30236760, "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2"),
})  # fmt: skip


class EmbedError(RuntimeError):
    pass


def model_dir() -> Path:
    """VENUE_ATLAS_HOME, else the platform's per-user cache folder."""
    if os.environ.get("VENUE_ATLAS_HOME"):
        return Path(os.environ["VENUE_ATLAS_HOME"]) / MODEL.name
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "venue-atlas" / MODEL.name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(target: Path | None = None) -> Path:
    """The model's files at their pinned revision, each checked against its size and SHA-256 (30 MB, once)."""
    target = target or model_dir()
    target.mkdir(parents=True, exist_ok=True)
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    for file, (size, sha) in MODEL.files.items():
        final = target / file
        if final.is_file() and final.stat().st_size == size and _sha256(final) == sha:
            continue
        url = f"{endpoint}/{MODEL.repo}/resolve/{MODEL.revision}/{file}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "venue-atlas"}), timeout=120) as r:
                data = r.read(size + 1)
        except (urllib.error.URLError, OSError) as e:
            raise EmbedError(f"could not download {file} from {endpoint} ({e}); set HF_ENDPOINT to a mirror if needed") from e
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise EmbedError(f"{file} from {endpoint} does not match its pinned size and SHA-256")
        part = final.with_suffix(final.suffix + ".part")
        part.write_bytes(data)
        os.replace(part, final)
    return target


DTYPES = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "I32": np.int32, "I64": np.int64}


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """The tensors of a .safetensors file: an 8-byte header length, a JSON header, then the data."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 0 < n < 100_000_000:
            raise EmbedError(f"{path.name} is not a safetensors file")
        header = json.loads(f.read(n))
    base = 8 + n
    size = path.stat().st_size
    out: dict[str, np.ndarray] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype = DTYPES.get(meta["dtype"])
        if dtype is None:
            raise EmbedError(f"{path.name}: tensor {key} has an unsupported type {meta['dtype']}")
        start, end = meta["data_offsets"]
        count = int(np.prod(meta["shape"])) if meta["shape"] else 1
        if end - start != count * np.dtype(dtype).itemsize or base + end > size:
            raise EmbedError(f"{path.name}: tensor {key} has the wrong size")
        out[key] = np.memmap(path, dtype=dtype, mode="r", offset=base + start, shape=tuple(meta["shape"]))
    return out


class StaticModel:
    """Model2Vec's inference: token ids (the unknown token dropped), each token's vector (mapped and weighted if the
    model says so), averaged and normalized."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.path = Path(path)
        try:
            tensors = read_safetensors(self.path / "model.safetensors")
            self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
            config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise EmbedError(f"the model in {self.path} could not be loaded ({e}); download it again") from e
        emb = tensors.get("embeddings", tensors.get("embedding.weight"))
        if emb is None:
            raise EmbedError(f"{self.path}: no embeddings in model.safetensors")
        self.embeddings = emb
        self.weights = tensors.get("weights")
        self.mapping = tensors.get("mapping")
        self.normalize = bool(config.get("normalize", True))
        self.max_length = int(config.get("max_length") or 512)
        self.dim = int(emb.shape[1])
        model = json.loads(self.tokenizer.to_str())["model"]
        self.unk = self.tokenizer.token_to_id(model["unk_token"]) if model.get("unk_token") else None
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(self.max_length)

    def token_table(self) -> np.ndarray:
        """Each token id's vector, mapped and weighted: what a token adds to a text's mean (float32, vocab × dim)."""
        n = self.tokenizer.get_vocab_size()
        ids = np.arange(n)
        rows = self.mapping[ids] if self.mapping is not None else ids
        table = np.asarray(self.embeddings[rows], dtype=np.float32)
        if self.weights is not None:
            table = table * np.asarray(self.weights[ids], dtype=np.float32)[:, None]
        return table

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [[t for t in enc.ids if t != self.unk] for enc in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def encode(self, texts: list[str], batch: int = 512) -> np.ndarray:
        table = self.token_table()
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), batch):
            for i, ids in enumerate(self.ids(texts[start : start + batch])):
                if ids:
                    out[start + i] = table[ids].mean(axis=0)
        if self.normalize:
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-32
        return out


def load(path: Path | None = None) -> StaticModel:
    folder = path or model_dir()
    if not all((folder / f).is_file() for f in MODEL.files):
        raise EmbedError(f"the model is not downloaded: run `venue-atlas model` ({sum(s for s, _ in MODEL.files.values()) // 1048576} MB, once)")
    return StaticModel(folder)


def quantize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 rows and each row's scale (max |value| / 127), for the browser: row ≈ int8 × scale.

    >>> q, s = quantize(np.array([[0.5, -1.0], [0.0, 0.0]], dtype=np.float32))
    >>> q.tolist(), [round(float(x), 5) for x in s]
    ([[64, -127], [0, 0]], [0.00787, 1.0])
    """
    peak = np.abs(matrix).max(axis=1)
    scale = np.where(peak > 0, peak / 127, 1.0).astype(np.float32)
    q = np.clip(np.rint(matrix / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale
