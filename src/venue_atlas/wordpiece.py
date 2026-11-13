"""BERT's uncased WordPiece tokenizer in plain Python, as Hugging Face's `tokenizers` runs it for potion-base-8M: its
special tokens found in the raw text, then BertNormalizer (control characters dropped, whitespace made spaces, CJK
characters set apart, accents stripped, lower case one character at a time), BertPreTokenizer (words split at spaces,
every punctuation mark on its own), WordPiece (longest match first, "##" continuations, 100 characters at most per
word). `scripts/tokenizer_check.py` compares it with `tokenizers` on every title and abstract of a build; the site's
tokenizer (site/tokenizer.js) does the same in the browser.

>>> t = WordPiece({"[UNK]": 0, "[MASK]": 1, "dif": 2, "##fusion": 3, "model": 4, "##s": 5, "-": 6, "cafe": 7}, "[UNK]", ["[MASK]"])
>>> t.tokens("Diffusion-models [MASK] Café ΣΣ")
['dif', '##fusion', '-', 'model', '##s', '[MASK]', 'cafe', '[UNK]']
>>> t.ids("Diffusion-models [MASK] Café ΣΣ")
[2, 3, 6, 4, 5, 1, 7, 0]
"""

from __future__ import annotations

import re
import sys
import unicodedata
from functools import lru_cache

CJK = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
       (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)]  # fmt: skip
ASCII_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def _ranges(keep: object) -> str:
    """A regex character class body of the code points `keep(char)` accepts."""
    out, start, prev = [], -1, -2
    for c in range(sys.maxunicode + 1):
        if keep(chr(c)):  # type: ignore[operator]
            if c != prev + 1:
                if start >= 0:
                    out.append((start, prev))
                start = c
            prev = c
    if start >= 0:
        out.append((start, prev))
    return "".join(re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in out)


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    # dropped: control, format and private-use characters (Cc, Cf, Co) but tab, newline and return, and U+FFFD (an
    # unassigned code point stays, as in tokenizers); whitespace: the separators (Zs, Zl, Zp) and tab, newline, return
    # (control characters such as \v are dropped first, as tokenizers does)
    control = re.compile("[" + _ranges(lambda ch: (unicodedata.category(ch) in ("Cc", "Cf", "Co") and ch not in "\t\n\r") or ch == "�") + "]")
    space = re.compile("[\t\n\r" + _ranges(lambda ch: unicodedata.category(ch) in ("Zs", "Zl", "Zp")) + "]")
    cjk = re.compile("([" + "".join(f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in CJK) + "])")
    mark = re.compile("[" + _ranges(lambda ch: unicodedata.category(ch) == "Mn") + "]")
    punct = _ranges(lambda ch: ch in ASCII_PUNCT or unicodedata.category(ch)[0] == "P")
    words = re.compile(f"[{punct}]|[^\\s{punct}]+")
    return control, space, cjk, mark, words


ASCII_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ASCII_WORDS = re.compile("[" + re.escape(ASCII_PUNCT) + "]|[^\\s" + re.escape(ASCII_PUNCT) + "]+")


def normalize(text: str) -> str:
    """BertNormalizer with lower case and accents stripped.

    >>> normalize("Naïve\\tCafé\\u200b 中文 ΣΟΦΟΣ")
    'naive cafe  中  文  σοφοσ'
    """
    if text.isascii():  # the common case, without the Unicode tables
        return ASCII_CONTROL.sub("", text).replace("\t", " ").replace("\n", " ").replace("\r", " ").lower()
    control, space, cjk, mark, _ = _patterns()
    text = space.sub(" ", control.sub("", text))
    text = cjk.sub(r" \1 ", text)
    text = mark.sub("", unicodedata.normalize("NFD", text))
    # one character at a time, as tokenizers lowers: str.lower() would make a word's last sigma a final sigma
    return text.replace("Σ", "σ").lower()


def pretokenize(text: str) -> list[str]:
    """Words split at whitespace, each punctuation mark a word of its own (after normalize()).

    >>> pretokenize("gnns, vits & rlhf: what's next?")
    ['gnns', ',', 'vits', '&', 'rlhf', ':', 'what', "'", 's', 'next', '?']
    """
    return (ASCII_WORDS if text.isascii() else _patterns()[4]).findall(text)


class WordPiece:
    def __init__(self, vocab: dict[str, int], unk: str, special: list[str] | None = None, max_chars: int = 100, prefix: str = "##"):
        self.vocab = vocab
        self.unk_token = unk
        self.unk = vocab[unk]
        self.max_chars = max_chars
        self.prefix = prefix
        self.by_id = {i: t for t, i in vocab.items()}
        special = [s for s in (special or []) if s in vocab]
        # the special tokens are found in the text as it is, before it is normalized (longest first)
        self.special = re.compile("(" + "|".join(re.escape(s) for s in sorted(special, key=len, reverse=True)) + ")") if special else None
        self._words: dict[str, list[int]] = {}

    def word(self, w: str) -> list[int]:
        """A word's pieces: the longest match in the vocabulary, then the longest "##" continuation, and so on; the
        unknown token for the whole word when a part has no match or the word is too long."""
        cached = self._words.get(w)
        if cached is not None:
            return cached
        out: list[int] = []
        if len(w) > self.max_chars:
            out = [self.unk]
        else:
            start = 0
            while start < len(w):
                end = len(w)
                found = -1
                while start < end:
                    found = self.vocab.get((self.prefix if start else "") + w[start:end], -1)
                    if found >= 0:
                        break
                    end -= 1
                if found < 0:
                    out = [self.unk]
                    break
                out.append(found)
                start = end
        if len(self._words) < 500_000:
            self._words[w] = out
        return out

    def ids(self, text: str) -> list[int]:
        """The token ids of a text (the unknown token included, as tokenizers gives them)."""
        parts = self.special.split(text) if self.special else [text]
        out: list[int] = []
        for k, part in enumerate(parts):
            if k % 2:  # a special token
                out.append(self.vocab[part])
            elif part:
                for w in pretokenize(normalize(part)):
                    out += self.word(w)
        return out

    def tokens(self, text: str) -> list[str]:
        return [self.by_id[i] for i in self.ids(text)]


def from_tokenizer_json(doc: dict[str, object]) -> WordPiece:
    """The tokenizer a tokenizer.json describes, when it is the BERT WordPiece this module implements; ValueError
    for any other."""
    model = doc.get("model")
    norm = doc.get("normalizer")
    if not (isinstance(model, dict) and model.get("type") == "WordPiece" and isinstance(model.get("vocab"), dict)):
        raise ValueError("not a WordPiece tokenizer")
    if not (isinstance(norm, dict) and norm.get("type") == "BertNormalizer" and norm.get("lowercase") and norm.get("clean_text")
            and norm.get("handle_chinese_chars") and norm.get("strip_accents") in (None, True)):  # fmt: skip
        raise ValueError(f"a normalizer this tokenizer does not implement: {norm}")
    if doc.get("pre_tokenizer") != {"type": "BertPreTokenizer"}:
        raise ValueError(f"a pre-tokenizer this tokenizer does not implement: {doc.get('pre_tokenizer')}")
    added = doc.get("added_tokens")
    special = [str(a["content"]) for a in (added if isinstance(added, list) else [])
               if isinstance(a, dict) and not a.get("normalized") and not a.get("single_word")]  # fmt: skip
    return WordPiece({str(t): int(i) for t, i in model["vocab"].items()}, str(model.get("unk_token") or "[UNK]"), special,
                     int(model.get("max_input_chars_per_word") or 100), str(model.get("continuing_subword_prefix") or "##"))  # fmt: skip
