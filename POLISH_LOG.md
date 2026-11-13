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
