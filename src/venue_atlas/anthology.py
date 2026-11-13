"""The ACL Anthology's venues (ACL, EMNLP, NAACL): their main-conference papers from the Anthology's full BibTeX with
abstracts (`anthology+abstracts.bib.gz`, one download for every venue and year; the Anthology's material is CC BY 4.0).
Long and short papers are told apart by their volume (`2024.acl-long`, `2024.acl-short`). As for the other venues,
abstracts are read to place the papers and are not kept."""

from __future__ import annotations

import gzip
import re
import unicodedata

URL = "https://aclanthology.org/anthology+abstracts.bib.gz"
MAX_AGE_DAYS = 30  # the Anthology corrects entries every day: a cached copy older than this is fetched again
ENTRY_URL = re.compile(r'\n\s*url = "https://aclanthology\.org/(\d{4})\.([a-z0-9-]+)\.(\d+)/?"')
ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "=": "\u0304", ".": "\u0307",
           "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328", "r": "\u030a"}  # fmt: skip
LETTERS = {"ss": "ß", "o": "ø", "O": "Ø", "aa": "å", "AA": "Å", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "l": "ł", "L": "Ł", "i": "ı"}


def latex_text(s: str) -> str:
    """BibTeX text as plain text: accents and special letters made Unicode, braces and escapes dropped.

    >>> latex_text(r'{IAG}: Induction-Augmented Generation for Sch{\\"o}n and Ni{\\~n}o, {\\ss}--style \\& 50\\%')
    'IAG: Induction-Augmented Generation for Schön and Niño, ß–style & 50%'
    >>> latex_text(r"Gonz{\\'a}lez and Jos\\'e, ``quoted'' --- {\\o}re")
    'González and José, “quoted” — øre'
    """

    def accent(m: re.Match[str]) -> str:
        return unicodedata.normalize("NFC", m.group(3) + ACCENTS[m.group(1)])

    s = re.sub(r"\\([\'`^\"~=.uvHckr])\s*(\{)?\s*([A-Za-z])\}?", accent, s)
    s = re.sub(r"\\(ss|aa|AA|ae|AE|oe|OE|o|O|l|L|i)\b(?:\{\})?", lambda m: LETTERS[m.group(1)], s)
    s = s.replace("``", "“").replace("''", "”").replace("---", "—").replace("--", "–")
    s = re.sub(r"\\([&%_#$])", r"\1", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def field(entry: str, name: str) -> str:
    m = re.search(rf'\n\s*{name} = "(.*?)",?\s*\n\s*(?:\w+ = |\}})', entry, re.S)
    if not m:
        m = re.search(rf'\n\s*{name} = "(.*?)"\s*\n?\}}?\s*$', entry, re.S)
    return m.group(1) if m else ""


def entries(data: bytes, volumes: set[str]) -> list[tuple[int, str, int, str, str]]:
    """(year, volume, number, title, abstract) of the papers in these volumes (`2024.acl-long`), front matter left
    out. `data` is the gzipped BibTeX, or the BibTeX itself."""
    text = (gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data).decode("utf-8", errors="replace")
    out = []
    for chunk in text.split("\n@"):
        m = ENTRY_URL.search(chunk)
        if not m:
            continue
        year, vol, number = int(m.group(1)), m.group(2), int(m.group(3))
        if f"{year}.{vol}" not in volumes or number == 0:
            continue
        title = latex_text(field(chunk, "title"))
        if title:
            out.append((year, vol, number, title, latex_text(field(chunk, "abstract"))))
    return out
