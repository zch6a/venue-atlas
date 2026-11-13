# Contributing

```bash
uv sync
uv run pytest -q                 # made-up schedule files and a stand-in model: no network, no model needed
node --test "test/*.test.js"     # the site's tokenizer against the Python tokenizer's output
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Where things are

| Path | What it does |
| --- | --- |
| `src/venue_atlas/sources.py` | the conferences' schedule files: download, cache, papers (sessions out, talks and posters merged) |
| `src/venue_atlas/embed.py` | the pinned static embedding model, run with numpy; int8 quantization for the site |
| `src/venue_atlas/layout.py` | nearest neighbours and the 2-D layout |
| `src/venue_atlas/topics.py` | k-means topics and their names |
| `src/venue_atlas/build.py` | an atlas: the site's data files |
| `site/` | the static page: `app.js` (map, panels, search), `tokenizer.js` (WordPiece) |
| `scripts/wordpiece_fixture.py` | writes the tokenizer test fixture from the real tokenizer (needs the model) |

## Adding a venue

A conference whose virtual site publishes `/static/virtual/data/<venue>-<year>-orals-posters.json`: add it to
`VENUES` in `sources.py` with the years whose file answers, build it, and look at its topics and at a few papers'
neighbours before adding it to the site's workflow. Another source needs its own reader returning `Paper`s.

## Changing the tokenizer or the model

Run `scripts/wordpiece_fixture.py` again (it uses the real tokenizer) and `node --test`; the browser's search results
should match `venue-atlas search` for the same query.
