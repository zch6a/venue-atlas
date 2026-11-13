# Polish log

### 2026-11-14 — development

**Why it exists**: what a conference is about, and how that changes year to year, is read from thousands of titles.
Paper Copilot, papers.cool and Paper Digest list and search papers; none shows a venue's papers as a map of topics
with their trends over years, searchable by meaning without a server.

**Sources, chosen**: OpenReview's API now asks for a challenge (a bot check) and is not used — it is not worked
around. Paper Copilot's paperlists has no licence, so its data is not used either. Each conference's own virtual site
publishes its schedule as one static JSON file per year (titles, abstracts, decisions, the conference's topic):
checked to answer for ICLR, NeurIPS, ICML 2021–2025 and CVPR 2023–2025 (HEAD requests on 2026-11-14). Abstracts are
used for the embeddings only and written nowhere (a test checks).

**Reading the files**, found on the real files: an oral is listed twice (its talk and its poster: one paper, the
higher decision kept, the poster's page); decisions are spelled in several cases; NeurIPS 2022 lists its "Lightning
Talks" sessions as entries whose abstract is a list of poster links (they made a topic of their own, "virtual poster ·
rocketchat · lightning talks", until left out); a few NeurIPS 2025 papers appear under two titles with the same
abstract (merged).

**Embeddings**: potion-base-8M (Model2Vec, MIT; licence checked on Hugging Face, revision pinned), so that the same
model runs in the browser for search: its 29,528 token vectors as int8 are 7.2 MB. The inference is semantic-find's
(the same author's). The venue's mean vector is taken away before neighbours are found: measured on ICLR 2023–25, the
share of each paper's 10 nearest that are in its topic went from 0.48 to 0.52 in the embedding and 0.38 to 0.41 on
the map.

**Layout**: exact t-SNE needs n² memory (NeurIPS 2021–25 is 19,210 papers), so UMAP's method in numpy: a fuzzy
20-nearest-neighbour graph, attraction along edges and repulsion between random pairs, from a PCA start, seeded. The
whole of NeurIPS takes about 25 s. Topic labels on the map are placed largest first and skipped where they would
overlap (they did, in a pile in the middle).

**Topics**: k-means, topics of a handful of papers folded into the nearest (a topic of one paper with no name, at
first), named by class-based TF-IDF; LaTeX in abstracts made a topic called "sqrt · omega · mathbb" until maths was
left out.

**Search in the browser**: a WordPiece tokenizer in JS (BertNormalizer, BertPreTokenizer, "##" pieces) is tested
against the real tokenizer's output on 18 strings (accents, CJK, full-width, emoji, control and zero-width characters,
a 120-character word): all match. The query is processed as the papers were (token mean, normalized, the venue's
mean taken away, normalized): the browser's top 5 for "protein structure generation with diffusion" on ICLR match
`venue-atlas search` to two decimals. The first version skipped the first normalization and was caught by that
comparison.

**Checked in the browser** (local server): the four venues load, no console errors, hover, click, the neighbours
list, topics with their bars, search by words and by meaning, the venue from the URL.

Tests: 22 Python (doctests included) and 19 JS. ruff, ruff format, mypy clean.

Next: polishing round 1 (the published site checked; keyboard and screen-reader use of the map; topic trends as a
chart; ACL venues from the ACL Anthology).

### 2026-11-14 — polishing round 1

**ACL, EMNLP and NAACL**, from the ACL Anthology: its per-volume BibTeX has no abstracts (checked), so its full
BibTeX with abstracts (`anthology+abstracts.bib.gz`, 45 MB, CC BY 4.0), one download for the three venues, kept for
30 days (the Anthology corrects entries daily). Main conferences only: `acl-long`/`acl-short`, `emnlp-main`,
`naacl-long`/`naacl-short` (and 2021's `naacl-main`); front matter (paper 0) left out; NAACL has no 2023. BibTeX's
LaTeX (`Sch{\"o}n`, `{\ss}`, `--`, `\&`, ``` `` ```) becomes plain text. ACL 5,122 papers, EMNLP 5,798, NAACL 2,199;
their topics read like the field (PEFT/LoRA at ACL from 9 papers in 2021 to 117 in 2025).

**A topic's trend, and the map without the map**: clicking a topic shows its papers per year as a chart whose numbers
are text (and an aria label giving each year's count and share) and the list of its papers (the latest 100). With
the topics, the search results and the neighbours already lists of buttons, every paper can be reached by keyboard or
screen reader. The map takes focus (an outline), arrow keys move it, + and − zoom, 0 resets, Escape clears; checked
by dispatching keys and comparing the canvas before and after.

**The site at phone width** (375 px): the topic panel had no height (the main area was held to the window's height):
fixed, the map at 62 % of the height with the topics below it. Topic labels cut by the map's edge are not drawn. The
footer names the Anthology as the ACL venues' source.

Tests: 24 Python (an Anthology file with front matter, long, short, Findings and a repeated title; the cache's age) and
19 JS. ruff, ruff format, mypy clean.

Next: polishing round 2 (an independent review of the code; the social card; the published site with all seven
venues checked).

### 2026-11-14 — polishing round 2

**An independent review** (12 findings, all fixed, each with a test where one can be written): a title read as a
session's (“Social …”, “Tutorial …”) dropped real papers, so such a title is skipped only when the record has no paper
link and no decision; a download is parsed before it is cached and written whole (a temporary file renamed), the
Anthology archive is decompressed before it is kept, and a reply cut short, an `EOFError` or a zlib error is an error
message, not a traceback; a cached file that no longer reads is fetched again; the workflow keeps its cache only after
a build that worked and runs `uv run --locked`. On the site: a malformed hash (`#neurips/%E0%A4%A`) names nothing
instead of breaking the page; a venue load or a vectors load overtaken by a later one is dropped; the topic panel and
the search results follow the years; drawing is once per animation frame; only the status line is a live region; the
number of neighbours comes from the atlas. `merge_small` handles every topic being small; `venue-atlas search` checks
`-k`, a query with no known word and an atlas it cannot read (exit 2).

**Links checked on all seven venues**: ICLR and NeurIPS 2021 had no conference page and their `url` was a project page
shown as “paper”. Now every paper has a conference page (built from the schedule's id where the file gives none,
`/virtual/2021/poster/<id>`, checked to answer), the paper link falls back to the PDF link (OpenReview PDFs as their
forum), and a project or code page is its own link. Papers left without a paper link have none in the source (ICLR
1,214, NeurIPS 2,431, ICML 1,896); the conference page covers them.

**Titles' LaTeX**: maths becomes Unicode (`$\pi$BO` → πBO, `$\ell_p$` → ℓₚ, `VA-RED$^2$` → VA-RED², `\(\varepsilon\)`
→ ε) and text commands their text (`Schr\"{o}dinger`, `Polyak-{\L}ojasiewicz`, `\emph{De Novo}`, a `\\` line
break); after the rebuild no title in the seven venues has a backslash or a dollar left (21 before). The same cleaning
applies before embedding, so a topic is no longer named “sqrt · omega”.

**The social card** (`docs/social-preview.png`, 1280×640, from `scripts/social-card.html`), and the site checked
locally with all seven venues: each loads, no console errors.

Tests: 28 Python (doctests included) and 19 JS. ruff, ruff format, mypy clean.

Next: polishing round 3 (the published site after the monthly build; performance of the largest venue on a phone;
the README's numbers against the built atlases).

### 2026-11-14 — polishing round 3

**On a phone**, measured with `scripts/phone-perf.mjs` (new: headless Chrome over the DevTools protocol, 375×812 at
2×, the CPU slowed 4×, reaching nothing but 127.0.0.1), on NeurIPS, the largest venue (19,214 papers): the map ready
0.8 s after navigation, a frame while panning or zooming 22 ms at the median (65 ms at the 90th percentile), a topic's
panel and chart 80 ms, the first search by meaning 0.2 s once its files are here and the next ones 0.09 s; 34 MB of
JavaScript heap. Nothing there needed work. What a phone does notice is the download: the published site sends 1.5 MB
for the NeurIPS map (compressed), and the first search by meaning another 11 MB (the model 6.7 MB, NeurIPS's vectors
4.4 MB). So the model and the venue's vectors now download together rather than one after the other, and the status
line says what is loading and how big it is (“the model … (7 MB, once) and NeurIPS's 4.7 MB of paper vectors”);
the README says the same, where it named only the model.

**The address follows the page**: the site read a venue and a paper from the URL's hash but never wrote the venue,
so after choosing another venue the address still named the first one (a reload or a shared link opened the wrong
venue), and closing a paper left its number there. Now a venue loaded, a paper closed and a paper opened each set the
hash, and changing only the paper in the address (`#cvpr/3` → `#cvpr/40`) opens that paper; a number out of range is
ignored. Checked in the browser: reload, venue switch, topic → paper → close, a typed hash.

**The README's numbers against the built atlases** (both languages): ICLR 9,662, NeurIPS 19,214 and ICML 10,254 papers
(round 2's fixes kept a few real papers); the NeurIPS example topics recounted on the published data (chain-of-thought
18 papers in 2021 → 231 in 2025; the retrieval-augmented one of the README is now part of a broader LLM topic, so the
example is multimodal LLMs and vision-language models, 11 → 200).

Tests: 28 Python and 19 JS. ruff, ruff format, mypy clean.

Next: polishing round 4 (engineering: a clean install from the wheel, the workflow's permissions and pins, dependency
audit; the CLI's edge cases).

### 2026-11-14 — polishing round 4

**The README's own recipe did not make a site**: `uvx --from git+… venue-atlas build … --out site`, then
`python -m http.server --directory site`, gave a folder with `data/` and `model/` and no page (the page files were only
in the repository), so the server showed a directory listing. Now the wheel carries the page (`venue_atlas/site/`) and
`build` copies it next to the data (not when the folder it writes is the repository's own `site/`); checked by building
the sdist, the wheel from it, installing it into a clean Python 3.10 environment and building NAACL 2025: the site
opens and searches. A one-year atlas said “2025–2025”; now “2025”. The package's description named four venues of seven.

**numpy is the only dependency**: `tokenizers` brought about fifteen packages (huggingface-hub, httpx2, hf-xet, …) to
split text into WordPiece tokens. `venue_atlas/wordpiece.py` does it in plain Python (special tokens found in the raw
text, BertNormalizer, BertPreTokenizer, WordPiece; cached per word, ASCII text without the Unicode tables), and
`scripts/tokenizer_check.py` compares it with `tokenizers`: on all 120,397 titles and abstracts of the seven venues
(15.9 million tokens) every text's ids are the same (8 s against 9.5 s). The check found what the fixture had
not: `tokenizers` keeps unassigned code points and drops private-use ones, and lowers one character at a time (a
word's last Σ becomes σ, not ς as `str.lower()` and JavaScript's `toLowerCase()` make it); it finds `[MASK]` and the
other special tokens in the text before normalizing it. The site's tokenizer had the same three differences, and a
fourth of its own: its CJK pattern, surrogate pairs under the `u` flag, never matched a character beyond the BMP. All
fixed, the fixture has four more cases (both tokenizers are tested against it: 23 JS tests, and a new Python test).
A rebuild of the seven venues with the new tokenizer gives byte-identical vectors and model files, and the browser's
top 5 for a query is the command line's.

**The command line**: `search` on a venue not built said `[Errno 2] No such file…`, now “no CVPR atlas in site: build
it first (venue-atlas build cvpr --out site)”; `--out` naming a file failed with a raw Windows error after loading the
model, now said first; Ctrl-C stops with a line, not a traceback (exit 130).

**Engineering**: `pip-audit` on the locked dependencies: no known vulnerabilities. CI runs `uv run --locked` (a lock that
no longer matches `pyproject.toml` fails at once, not in the monthly build), and a new job builds the sdist and the
wheel from it, checks that the wheel carries the page and that a clean install's only requirement is numpy; every job
of the three workflows has a timeout. Pins and permissions checked: actions pinned to commits, read-only tokens, the
Pages and release jobs that can write run nothing of the project.

Tests: 32 Python (doctests included) and 23 JS. ruff, ruff format, mypy clean.

Next: polishing round 5 (the documents read through; a fuzz of the readers with broken schedule files and BibTeX;
the CHANGELOG as release notes), then v0.1.0.

### 2026-11-14 — polishing round 5, and v0.1.0

**Broken inputs** (`scripts/fuzz.py`, new): mutations of real files from the build's cache — a schedule file of each of
ICLR, NeurIPS, ICML and CVPR with bytes flipped, cut short, pieces repeated or removed, and JSON values replaced by
ones of another type (null, numbers, lists, HTML, a lone surrogate, a 400-digit number); 3 MB of the Anthology's
BibTeX the same way; random text with LaTeX fragments for the LaTeX and maths cleaners and the tokenizer. A reader
must give papers or a SourceError, never another exception, and must not be slow. 300 rounds (seed 0): 2,400 broken
schedule files, 30 broken BibTeX files, 900 random texts, 0 failures, the slowest 0.6 s. Then 1,500 rounds with seed 7 (other years' files): 12,000 broken schedule files, 150 broken BibTeX files, 4,500 random texts, 0 failures, the slowest again 0.6 s. Nothing to fix.

**The documents read through**, both languages: the README's screenshot showed the counts before round 2's fixes and the
old footer (retaken from the local site: 19,214 papers, the footer naming the ACL Anthology, CC BY 4.0) and, doing so,
showed a space lost in round 4 (“2021–2025:19,214 papers”: fixed); the licence names the Anthology's licence for
the ACL venues' titles; the limits say how many papers have no paper link in their source (an eighth at ICLR and
NeurIPS, a fifth at ICML: their conference page has it); the address that names the view, a venue built into a folder
with others joining the list, the Anthology's file fetched again after 30 days, all said; the browser and Python
tokenizers described as both tested against the real one; the recipes use `@v0`, the latest release. SECURITY names
the Anthology download and that numpy is the only run-time dependency; the plugin's skill says what the ACL venues
download and what a topic's `count` looks like. Every version number (pyproject, package, plugin) is 0.1.0.

**The CHANGELOG as release notes**: what the site does, what the command does, the venues and their 60,195 papers.

Tests: 32 Python (doctests included) and 23 JS. ruff, ruff format, mypy clean.

**v0.1.0**: tagged (annotated); the release workflow tests, builds the wheel and source archive, makes the GitHub
release from the CHANGELOG section and points v0 at it.

### 2026-11-14 — after the release

The plugin's description named four venues of seven (plugin.json): fixed.

### 2026-11-14 — a link to research-kit

Both READMEs now end (before the licence) with a short section pointing to research-kit, the toolkit's front door: all
thirteen tools in the order of the work, and one Claude Code plugin marketplace for their skills. Documentation only.
