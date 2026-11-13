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
GREEK = {n: chr(c) for n, c in zip(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"],
                                       [0x3B1, 0x3B2, 0x3B3, 0x3B4, 0x3B5, 0x3B6, 0x3B7, 0x3B8, 0x3B9, 0x3BA, 0x3BB, 0x3BC, 0x3BD, 0x3BE, 0x3C0, 0x3C1, 0x3C3, 0x3C4, 0x3C5,
                                        0x3C6, 0x3C7, 0x3C8, 0x3C9], strict=True)}  # fmt: skip
GREEK |= {n.capitalize(): chr(c) for n, c in [("gamma", 0x393), ("delta", 0x394), ("theta", 0x398), ("lambda", 0x39B), ("xi", 0x39E), ("pi", 0x3A0),
                                               ("sigma", 0x3A3), ("phi", 0x3A6), ("psi", 0x3A8), ("omega", 0x3A9)]}  # fmt: skip
SYMBOLS = {**GREEK, "varepsilon": "ε", "vartheta": "ϑ", "varphi": "φ", "ell": "ℓ", "infty": "∞", "times": "×", "cdot": "·", "leq": "≤", "le": "≤",
           "geq": "≥", "ge": "≥", "neq": "≠", "approx": "≈", "sim": "∼", "to": "→", "rightarrow": "→", "leftarrow": "←", "sqrt": "√", "partial": "∂",
           "nabla": "∇", "in": "∈", "sum": "∑", "pm": "±", "star": "⋆", "ast": "∗", "circ": "∘", "log": "log", "exp": "exp"}  # fmt: skip
DOUBLE = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼", "P": "ℙ"}
SUPER = dict(zip("0123456789+-=()n*", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ*", strict=True))
SUB = dict(zip("0123456789+-=()aeoxhklmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ", strict=True))
MATH_ACCENTS = {"acute": "\u0301", "grave": "\u0300", "hat": "\u0302", "tilde": "\u0303", "bar": "\u0304", "dot": "\u0307", "ddot": "\u0308"}


def maths_text(s: str) -> str:
    """Maths in a title as plain Unicode: Greek letters and common symbols, blackboard letters, one-character super-
    and subscripts, accents; formatting commands (\\mathrm, \\mathcal, \\textbf) leave their text.

    >>> maths_text(r"$\\pi$BO and VA-RED$^2$ under $\\ell_p$, $\\mathrm{SO}(2)$ on $\\mathbb{R}^n$, Demoir$\\acute{e}$ing")
    'πBO and VA-RED² under ℓₚ, SO(2) on ℝⁿ, Demoiréing'
    """
    if "$" not in s and "\\" not in s:
        return s

    def inside(m: re.Match[str]) -> str:
        t = m.group(1)
        t = re.sub(r"\\(?:mathbb|mathds|bb)\s*\{?\s*([A-Z])\}?", lambda x: DOUBLE.get(x.group(1), x.group(1)), t)
        t = re.sub(r"\\(acute|grave|hat|tilde|bar|dot|ddot)\s*\{?\s*([A-Za-z])\}?",
                   lambda x: unicodedata.normalize("NFC", x.group(2) + MATH_ACCENTS[x.group(1)]), t)  # fmt: skip
        for _ in range(3):
            t = re.sub(
                r"\\(?:mathrm|mathcal|mathbf|mathit|mathsf|mathtt|text|textrm|textbf|textit|texttt|operatorname|boldsymbol|bm)\s*\{([^{}]*)\}",
                r"\1",
                t,
            )
        t = re.sub(r"\\([A-Za-z]+)", lambda x: SYMBOLS.get(x.group(1), x.group(1)), t)
        t = re.sub(r"\^\{?([^{}\s])\}?", lambda x: SUPER.get(x.group(1), "^" + x.group(1)), t)
        t = re.sub(r"_\{?([^{}\s])\}?", lambda x: SUB.get(x.group(1), "_" + x.group(1)), t)
        return t.replace("{", "").replace("}", "").replace("\\,", "").replace("\\ ", " ")

    s = re.sub(r"\$([^$]{1,200})\$", inside, s)
    return re.sub(r"\\([&%_#])", r"\1", s)


LETTERS = {"ss": "ß", "o": "ø", "O": "Ø", "aa": "å", "AA": "Å", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "l": "ł", "L": "Ł", "i": "ı"}


def latex_text(s: str) -> str:
    """BibTeX text as plain text: accents and special letters made Unicode, braces and escapes dropped.

    >>> latex_text(r'{IAG}: Induction-Augmented Generation for Sch{\\"o}n and Ni{\\~n}o, {\\ss}--style \\& 50\\%')
    'IAG: Induction-Augmented Generation for Schön and Niño, ß–style & 50%'
    >>> latex_text(r"Gonz{\\'a}lez and Jos\\'e, ``quoted'' --- {\\o}re")
    'González and José, “quoted” — øre'
    >>> latex_text(r"Mart{\\'\\i}nez: $\\texttt{C2R}$ for \\textit{Let Alone}, with \\kappa and \\cdot, \\v{c}esky")
    'Martínez: C2R for Let Alone, with kappa and cdot, česky'
    >>> latex_text(r"\\(\\varepsilon\\)-Optimal Layers:\\\\ Exact, Schr\\"{o}dinger and \\emph{De Novo}")
    'ε-Optimal Layers: Exact, Schrödinger and De Novo'
    """

    s = re.sub(r"\\\((.{1,200}?)\\\)", r"$\1$", s)  # \( maths \) is $ maths $
    s = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", s)  # a line break

    def accent(m: re.Match[str]) -> str:
        letter = {"\\i": "i", "\\j": "j"}.get(m.group(2), m.group(2))  # a dotless i under an accent is an i
        return unicodedata.normalize("NFC", letter + ACCENTS[m.group(1)])

    # a symbol accent (\'e, \"{o}, \'\i); a letter accent (\v{c}, \c c) only when not a longer command (\kappa, \cdot, \url)
    s = re.sub(r"\\([\'`^\"~=.])\s*\{?\s*(\\[ij](?![A-Za-z])|[A-Za-z])\}?", accent, s)
    s = re.sub(r"\\([uvHckr])(?![A-Za-z])\s*\{?\s*(\\[ij](?![A-Za-z])|[A-Za-z])\}?", accent, s)
    s = re.sub(r"\\(ss|aa|AA|ae|AE|oe|OE|o|O|l|L|i)\b(?:\{\})?", lambda m: LETTERS[m.group(1)], s)
    s = s.replace("``", "“").replace("''", "”").replace("---", "—").replace("--", "–")
    s = re.sub(r"\$([^$]{1,200})\$", lambda m: maths_text("$" + m.group(1) + "$"), s)  # maths: Unicode where it can be
    s = re.sub(r"\\([&%_#$])", lambda m: "\x00" + m.group(1), s)  # escaped symbols, kept through the steps below
    for _ in range(3):  # \textit{Let Alone} -> Let Alone, nested too
        s = re.sub(r"\\[A-Za-z]+\*?\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)  # other commands: their name (\Omega -> Omega)
    s = re.sub(r"(?<!\x00)\$", "", s)  # maths delimiters (an escaped \\$ stays)
    s = s.replace("{", "").replace("}", "").replace("~", " ").replace("\x00", "")
    s = re.sub(r"\\(?=[^\x00-\x7f])", "", s)  # a backslash left before a converted symbol (L\₂)
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
