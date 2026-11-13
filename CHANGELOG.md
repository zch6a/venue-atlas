# Changelog

## 0.1.0 — unreleased

First release.

- `venue-atlas build`: a venue's accepted papers (ICLR, NeurIPS, ICML 2021–2025; CVPR 2023–2025) from the
  conference's own schedule file, embedded with a pinned static model, laid out by their neighbours, grouped into
  named topics with each topic's count by year. Abstracts are read, not kept.
- The site: the map, hover and click for a paper's links and its closest papers, topics with their share by year,
  search by words or by meaning (the query embedded in the browser).
- ACL, EMNLP and NAACL (main conferences, 2021–2025) from the ACL Anthology's BibTeX with abstracts.
- A topic's papers per year as a chart with the numbers as text, and its papers as a list; the map by keyboard.
- `venue-atlas search`: the closest papers to a query, from a built atlas (`--json`).
- A monthly GitHub workflow that builds the site with a read-only token and deploys it.
