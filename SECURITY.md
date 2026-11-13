# Security

## What it does

- **What reaches the network, and when.** `venue-atlas build` downloads each venue's schedule file from the
  conference's own website (one request per venue and year, a generic User-Agent, 200 MB at most each, two seconds
  apart) and keeps it in the cache folder; `venue-atlas model` downloads the embedding model from Hugging Face (or
  HF_ENDPOINT), each file checked against its pinned size and SHA-256. Nothing else is fetched; `search` is offline.
- **What the site keeps.** Titles, links, years, decisions, topics, the map, and int8 vectors. Abstracts are read to
  embed the papers and are written nowhere (a test checks the built files for them). No author names.
- **The site.** A static page with a Content-Security-Policy that allows only its own scripts and data; titles are
  inserted as text, never as HTML; links are followed only when they are http(s). Searches run in the browser and
  are sent nowhere.
- **The workflow.** The site is built with a read-only token; the job that deploys it runs nothing of the project.
  Actions are pinned to commits.

## Reporting a vulnerability

Please open a private security advisory on GitHub (Security → Report a vulnerability) rather than a public issue.
