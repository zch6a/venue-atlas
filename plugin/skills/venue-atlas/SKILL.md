---
name: venue-atlas
description: Find what a machine-learning conference (ICLR, NeurIPS, ICML, ACL, EMNLP 2021–2025; CVPR 2023–2025; NAACL 2021–2025 but 2023) published on a subject, its closest papers by meaning, and how topics grew or shrank by year, with venue-atlas. Use when the user asks which papers at a venue are about something, for related work at a conference, or for the trends of a venue's topics.
license: MIT
compatibility: Needs Python 3.10+ and uv. Runs venue-atlas with uvx from GitHub (git+https://github.com/zch6a/venue-atlas). Building downloads the conferences' public schedule files and a 30 MB model once; searching an atlas already built is offline.
metadata:
  homepage: https://github.com/zch6a/venue-atlas
---

# Searching a venue's papers with venue-atlas

For people, the site is the quickest: https://zch6a.github.io/venue-atlas/ (search by meaning in the browser).

For an answer here:

```bash
uvx --from git+https://github.com/zch6a/venue-atlas venue-atlas model          # once, 30 MB (ask first)
uvx --from git+https://github.com/zch6a/venue-atlas venue-atlas build neurips --out atlas
uvx --from git+https://github.com/zch6a/venue-atlas venue-atlas search "test-time adaptation for vision" --venue neurips --atlas atlas --json
```

- Tell the user before downloading: the model (30 MB) and one schedule file per venue and year (10–26 MB each).
- Report papers with their year and link; the score is a cosine similarity from a small static model, good for
  finding candidates, not for ranking them finely. Read the titles before calling a paper relevant.
- For trends, read `atlas/data/<venue>.json`: `topics[i].terms` names a topic and `topics[i].count` gives its papers
  per year; say that topics are word clusters, named by their distinctive words.
