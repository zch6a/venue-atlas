# Contributing

```bash
uv sync
uv run pytest -q                 # made-up schedule files and a stand-in model: no network, no model needed
node --test "test/*.test.js"     # the site's tokenizer against the real tokenizer's output
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Where things are

| Path | What it does |
| --- | --- |
| `src/venue_atlas/sources.py` | the conferences' schedule files: download, cache, papers (sessions out, talks and posters merged) |
| `src/venue_atlas/embed.py` | the pinned static embedding model, run with numpy; int8 quantization for the site |
| `src/venue_atlas/wordpiece.py` | the model's WordPiece tokenizer in plain Python (the package's only dependency is numpy) |
| `src/venue_atlas/anthology.py` | the ACL Anthology's BibTeX: entries, and LaTeX made plain text |
| `src/venue_atlas/layout.py` | nearest neighbours and the 2-D layout |
| `src/venue_atlas/topics.py` | k-means topics and their names |
| `src/venue_atlas/build.py` | an atlas: the site's data files, and the page copied next to them |
| `site/` | the static page: `app.js` (map, panels, search), `tokenizer.js` (WordPiece) |
| `scripts/wordpiece_fixture.py` | writes the tokenizer test fixture from the real tokenizer (needs the model) |
| `scripts/tokenizer_check.py` | the Python tokenizer against the real one on every title and abstract of a build |
| `scripts/phone-perf.mjs` | the site in headless Chrome at a phone's size and speed: load, frames, search |

## Adding a venue

A conference whose virtual site publishes `/static/virtual/data/<venue>-<year>-orals-posters.json`: add it to
`VENUES` in `sources.py` with the years whose file answers, build it, and look at its topics and at a few papers'
neighbours before adding it to the site's workflow. Another source needs its own reader returning `Paper`s.

## Changing the tokenizer or the model

The real tokenizer is Hugging Face's `tokenizers`, used by the scripts only:

```bash
uv run --with tokenizers python scripts/wordpiece_fixture.py      # the fixture both tokenizers are tested against
uv run --with tokenizers python scripts/tokenizer_check.py        # every title and abstract: 0 should differ
```

Then `uv run pytest` and `node --test`; the browser's search results should match `venue-atlas search` for the same
query.
