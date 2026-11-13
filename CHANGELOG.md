# Changelog

## 0.1.0 — 2026-11-14

First release: maps of the papers accepted at seven machine-learning venues, as a static site
(https://zch6a.github.io/venue-atlas/) and the command that builds it.

**Venues.** ICLR, NeurIPS and ICML 2021–2025 and CVPR 2023–2025 from each conference's own schedule file; ACL, EMNLP
and NAACL 2021–2025 (main conferences, long and short papers) from the ACL Anthology's BibTeX. 60,195 papers. Abstracts
are read to place the papers and are not kept.

**The site.**
- The map: papers placed by meaning (a pinned static embedding model, a UMAP-like layout), coloured by topic; hover for
  a title, click for the paper's links (paper or PDF, conference page, code or project page) and its eight closest
  papers.
- Topics named by their distinctive words, each with its share of every year; click one for its papers per year as a
  chart (the numbers also as text) and its papers as a list.
- Search by words, or by meaning: the query is embedded in the browser with the same model, and its results match
  `venue-atlas search`.
- The keyboard moves and zooms the map; topics, papers and results are lists of links. The address names the venue
  and the paper shown. Works at phone width.

**The command.**
- `venue-atlas build <venues> --out site` writes a whole site (the page, the atlases, the model for the browser);
  `--years` for some years; downloads are cached.
- `venue-atlas search "<query>" --venue <venue>`: the closest papers from a built atlas, `--json` for scripts.
- `venue-atlas model` downloads the model (30 MB, pinned by revision and SHA-256); `venue-atlas venues` lists the
  venues and years.
- numpy is the only dependency: the model's tokenizer is plain Python, the same tokens as Hugging Face's `tokenizers`
  on all 120,397 titles and abstracts.

**The workflow.** The site is rebuilt monthly by a GitHub workflow whose build has a read-only token and whose deploy
runs nothing of the project.
