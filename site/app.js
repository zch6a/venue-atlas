// venue-atlas: the map of one venue's papers (data/<venue>.json), drawn on a canvas; topics with their share by year;
// search by words (titles) or by meaning (the query embedded here, with the same model the atlas was built with).
import { encode } from "./tokenizer.js";

const $ = (id) => document.getElementById(id);
const canvas = $("map");
const ctx = canvas.getContext("2d");
const tip = $("tip");
const status = $("status");

const state = {
  atlas: null, // the venue's data
  n: 0,
  years: new Set(),
  colour: "topic",
  selected: -1,
  topic: -1, // a topic picked in the list: the others dimmed
  hits: null, // search results: Map index -> rank
  view: { x: 0, y: 0, k: 1 }, // the map's pan and zoom
  vectors: null, // Int8Array, n × dim (loaded for meaning search)
  model: null, // { vocab: Map, unk, dim, scale: Float32Array, table: Int8Array }
  grid: null,
};

// ------------------------------------------------------------------ colours

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function topicColour(i, alpha = 1) {
  const hue = (i * 137.508) % 360; // the golden angle: neighbouring topic numbers far apart in hue
  const light = 52 + (i % 3) * 7;
  return `hsla(${hue.toFixed(1)}, 62%, ${light}%, ${alpha})`;
}
const YEAR_COLOURS = ["#9aa5b1", "#5aa9e6", "#3d7dd8", "#7b5bd6", "#d64f8f", "#e8873a"];
const DECISION_COLOURS = ["#9aa5b1", "#9aa5b1", "#2f9e72", "#e0393e"]; // none, poster, spotlight, oral

function colourOf(i) {
  const a = state.atlas;
  if (state.colour === "year") {
    // the latest years get the warmest colours
    const fromEnd = a.years.length - 1 - a.years.indexOf(a.papers.year[i]);
    return YEAR_COLOURS[Math.max(0, YEAR_COLOURS.length - 1 - fromEnd)];
  }
  if (state.colour === "decision") return DECISION_COLOURS[a.papers.decision[i]];
  return topicColour(a.papers.topic[i]);
}

// ------------------------------------------------------------------ loading

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function bytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Int8Array(await r.arrayBuffer());
}

async function loadVenue(key) {
  status.textContent = "Loading…";
  status.hidden = false;
  const atlas = await json(`data/${encodeURIComponent(key)}.json`);
  if (atlas.format !== "venue-atlas/1") throw new Error("an atlas in a format this page does not read");
  state.atlas = atlas;
  state.n = atlas.count;
  state.years = new Set(atlas.years);
  state.selected = -1;
  state.topic = -1;
  state.hits = null;
  state.vectors = null;
  state.view = { x: 0, y: 0, k: 1 };
  buildGrid();
  renderYears();
  renderTopics();
  $("paper").hidden = true;
  $("results").hidden = true;
  $("about").textContent = `${atlas.name} ${atlas.years[0]}–${atlas.years[atlas.years.length - 1]}: ${atlas.count.toLocaleString()} papers in ${atlas.topics.length} topics · built ${atlas.built}`;
  document.title = `${atlas.name} · venue-atlas`;
  status.hidden = true;
  draw();
}

async function loadModel() {
  if (!state.model) {
    status.hidden = false;
    status.textContent = "Loading the model for search by meaning (7 MB, once)…";
    const meta = await json("model/vocab.json");
    const table = await bytes("model/tokens.vec");
    state.model = { vocab: new Map(meta.tokens.map((t, i) => [t, i])), unk: meta.unk, dim: meta.dim, scale: Float32Array.from(meta.scale), table };
  }
  if (!state.vectors) {
    status.hidden = false;
    status.textContent = "Loading the papers' vectors…";
    state.vectors = await bytes(`data/${encodeURIComponent(state.atlas.venue)}.vec`);
  }
  status.hidden = true;
}

// ------------------------------------------------------------------ geometry

function size() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  return { w: r.width, h: r.height, dpr };
}
function toScreen(i, w, h) {
  const p = state.atlas.papers;
  const s = Math.min(w, h) * 0.92 * state.view.k;
  const ox = (w - s) / 2 + state.view.x;
  const oy = (h - s) / 2 + state.view.y;
  return [ox + p.x[i] * s, oy + (1 - p.y[i]) * s];
}
function visible(i) {
  return state.years.has(state.atlas.papers.year[i]);
}

function buildGrid() {
  // papers by a 64 × 64 grid of map cells, for finding the dot under the pointer
  const p = state.atlas.papers;
  const g = new Map();
  for (let i = 0; i < state.n; i++) {
    const key = Math.min(63, Math.floor(p.x[i] * 64)) * 64 + Math.min(63, Math.floor(p.y[i] * 64));
    if (!g.has(key)) g.set(key, []);
    g.get(key).push(i);
  }
  state.grid = g;
}
function nearest(px, py) {
  const { w, h } = size();
  const s = Math.min(w, h) * 0.92 * state.view.k;
  const mx = (px - (w - s) / 2 - state.view.x) / s;
  const my = 1 - (py - (h - s) / 2 - state.view.y) / s;
  const r = 8 / s;
  let best = -1;
  let bestD = Infinity;
  const p = state.atlas.papers;
  for (let cx = Math.floor((mx - r) * 64); cx <= Math.floor((mx + r) * 64); cx++) {
    for (let cy = Math.floor((my - r) * 64); cy <= Math.floor((my + r) * 64); cy++) {
      for (const i of state.grid.get(cx * 64 + cy) || []) {
        if (!visible(i)) continue;
        const d = (p.x[i] - mx) ** 2 + (p.y[i] - my) ** 2;
        if (d < bestD && d < r * r) {
          bestD = d;
          best = i;
        }
      }
    }
  }
  return best;
}

// ------------------------------------------------------------------ drawing

function draw() {
  if (!state.atlas) return;
  const { w, h, dpr } = size();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const p = state.atlas.papers;
  const radius = Math.max(1.6, Math.min(4, 1.6 * Math.sqrt(state.view.k)));
  const focus = state.hits || state.topic >= 0;
  // with a search or a topic picked: the others dimmed first, then those in focus on top
  for (let pass = focus ? 0 : 1; pass < 2; pass++) {
    for (let i = 0; i < state.n; i++) {
      if (!visible(i)) continue;
      const on = !focus || (state.hits ? state.hits.has(i) : p.topic[i] === state.topic);
      if (focus && (pass === 0) === on) continue;
      const [x, y] = toScreen(i, w, h);
      if (x < -5 || y < -5 || x > w + 5 || y > h + 5) continue;
      ctx.fillStyle = on ? colourOf(i) : css("--dim");
      ctx.beginPath();
      ctx.arc(x, y, on && focus ? radius + 0.8 : radius, 0, 6.2832);
      ctx.fill();
    }
  }
  if (state.selected >= 0) {
    const [x, y] = toScreen(state.selected, w, h);
    ctx.strokeStyle = css("--mark");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, radius + 5, 0, 6.2832);
    ctx.stroke();
  }
  drawLabels(w, h);
}

function drawLabels(w, h) {
  // each topic's name at the middle of its papers (the visible ones), when not zoomed far in
  if (state.view.k > 4) return;
  const a = state.atlas;
  const sx = new Float64Array(a.topics.length);
  const sy = new Float64Array(a.topics.length);
  const n = new Int32Array(a.topics.length);
  for (let i = 0; i < state.n; i++) {
    if (!visible(i)) continue;
    const t = a.papers.topic[i];
    const [x, y] = toScreen(i, w, h);
    sx[t] += x;
    sy[t] += y;
    n[t] += 1;
  }
  ctx.font = "600 11.5px system-ui, sans-serif";
  ctx.textAlign = "center";
  // the larger topics first; a label that would overlap one already drawn is left out (zoom in to see it)
  const order = [...a.topics.keys()].sort((p, q) => n[q] - n[p]);
  const placed = [];
  for (const t of order) {
    if (n[t] < 15 || (state.topic >= 0 && state.topic !== t)) continue;
    const label = a.topics[t].terms.slice(0, 2).join(" · ");
    const x = sx[t] / n[t];
    const y = sy[t] / n[t];
    const half = ctx.measureText(label).width / 2 + 4;
    if (x - half < 0 || x + half > w || y < 12 || y > h - 4) continue; // a label cut by the edge says nothing
    if (placed.some((b) => Math.abs(b.x - x) < b.half + half && Math.abs(b.y - y) < 15)) continue;
    placed.push({ x, y, half });
    ctx.lineWidth = 3;
    ctx.strokeStyle = css("--bg");
    ctx.strokeText(label, x, y);
    ctx.fillStyle = css("--ink");
    ctx.fillText(label, x, y);
  }
}

// ------------------------------------------------------------------ panels

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "on") for (const [ev, f] of Object.entries(v)) e.addEventListener(ev, f);
    else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c);
  return e;
}
function safeLink(url, text) {
  return /^https?:\/\//.test(url) ? el("a", { href: url, rel: "noopener noreferrer", target: "_blank" }, text) : null;
}
const DECISION = ["", "poster", "spotlight", "oral"];

function topicName(t) {
  return state.atlas.topics[t].terms.join(" · ") || `topic ${t + 1}`;
}

function select(i, centre = false) {
  state.selected = i;
  const a = state.atlas;
  const p = a.papers;
  const box = $("paper");
  box.replaceChildren();
  if (i < 0) {
    box.hidden = true;
    draw();
    return;
  }
  const bits = [String(p.year[i]), DECISION[p.decision[i]], a.conf_topics[p.conf_topic[i]] || ""].filter(Boolean);
  const links = el("div", { class: "links" });
  for (const [url, text] of [[p.url[i], "paper"], [p.page[i], "conference page"]]) {
    const l = safeLink(url, text);
    if (l) links.append(l);
  }
  const near = el("ol", { class: "list" });
  for (let k = 0; k < 8; k++) {
    const j = a.neighbours[i * 8 + k];
    near.append(el("li", {}, el("button", { on: { click: () => select(j, true) } }, `${p.title[j]} (${p.year[j]})`)));
  }
  box.append(
    el("button", { class: "close", "aria-label": "Close", on: { click: () => select(-1) } }, "×"),
    el("h3", {}, p.title[i]),
    el("div", { class: "meta" }, bits.join(" · ")),
    el("div", { class: "meta" }, "Topic: ", el("button", { class: "name", on: { click: () => pickTopic(p.topic[i]) } }, topicName(p.topic[i]))),
    links,
    el("h2", {}, "Closest in meaning"),
    near,
  );
  box.hidden = false;
  if (centre) {
    const { w, h } = size();
    const [x, y] = toScreen(i, w, h);
    state.view.x += w / 2 - x;
    state.view.y += h / 2 - y;
  }
  location.replace(`#${encodeURIComponent(a.venue)}/${i}`);
  draw();
}

function renderYears() {
  const box = $("years");
  box.replaceChildren();
  for (const y of state.atlas.years) {
    const b = el("button", { "aria-pressed": "true", type: "button" }, String(y));
    b.addEventListener("click", () => {
      if (state.years.has(y) && state.years.size === 1) return; // at least one year
      state.years.has(y) ? state.years.delete(y) : state.years.add(y);
      b.setAttribute("aria-pressed", String(state.years.has(y)));
      renderTopics();
      draw();
    });
    box.append(b);
  }
}

function renderTopics() {
  const a = state.atlas;
  const box = $("topics");
  const years = a.years.filter((y) => state.years.has(y));
  const totals = Object.fromEntries(years.map((y) => [y, a.topics.reduce((s, t) => s + (t.count[y] || 0), 0)]));
  const rows = a.topics
    .map((t, i) => ({ i, n: years.reduce((s, y) => s + (t.count[y] || 0), 0), share: years.map((y) => (totals[y] ? (t.count[y] || 0) / totals[y] : 0)) }))
    .sort((x, y) => y.n - x.n);
  const peak = Math.max(...rows.flatMap((r) => r.share), 1e-9);
  box.replaceChildren(el("h2", {}, years.length > 1 ? `Topics · share of each year (${years[0]}–${years[years.length - 1]})` : `Topics · ${years[0]}`));
  for (const r of rows) {
    const bars = el("div", { class: "bars", title: years.map((y, k) => `${y}: ${(r.share[k] * 100).toFixed(1)}%`).join("\n") });
    for (const s of r.share) {
      const bar = el("span");
      bar.style.height = `${Math.max(4, (100 * s) / peak)}%`;
      bar.style.background = topicColour(r.i, 0.85);
      bars.append(bar);
    }
    const dot = el("span", { class: "dot" });
    dot.style.background = topicColour(r.i);
    const row = el("div", { class: `topic${state.topic === r.i ? " active" : ""}` }, dot,
      el("button", { class: "name", on: { click: () => pickTopic(state.topic === r.i ? -1 : r.i) } }, topicName(r.i)),
      el("span", { class: "n" }, r.n.toLocaleString()), years.length > 1 ? bars : "");
    box.append(row);
  }
}

function pickTopic(t) {
  state.topic = t;
  state.hits = null;
  const box = $("results");
  if (t < 0) {
    box.hidden = true;
  } else {
    showTopic(t, box);
  }
  renderTopics();
  draw();
}

// A topic: its papers per year as a chart (with the numbers as text), and its papers as a list (the way to the papers
// without the map, by keyboard or a screen reader).
function showTopic(t, box) {
  const a = state.atlas;
  const p = a.papers;
  const years = a.years.filter((y) => state.years.has(y));
  const totals = Object.fromEntries(years.map((y) => [y, a.topics.reduce((s, x) => s + (x.count[y] || 0), 0)]));
  const counts = years.map((y) => a.topics[t].count[y] || 0);
  const peak = Math.max(...counts, 1);
  const chart = el("div", { class: "trend", role: "img", "aria-label": years.map((y, k) => `${y}: ${counts[k]} papers, ${totals[y] ? ((100 * counts[k]) / totals[y]).toFixed(1) : 0}% of the year`).join("; ") });
  years.forEach((y, k) => {
    const bar = el("span", { class: "bar" });
    bar.style.height = `${Math.max(2, (100 * counts[k]) / peak)}%`;
    bar.style.background = topicColour(t);
    const share = totals[y] ? ((100 * counts[k]) / totals[y]).toFixed(1) : "0";
    chart.append(el("div", { class: "col" }, el("span", { class: "num" }, String(counts[k])), el("span", { class: "plot" }, bar), el("span", { class: "yr" }, String(y)), el("span", { class: "pct" }, `${share}%`)));
  });
  const members = [];
  for (let i = 0; i < state.n; i++) if (p.topic[i] === t && visible(i)) members.push(i);
  members.sort((x, y) => p.year[y] - p.year[x] || p.title[x].localeCompare(p.title[y]));
  const list = el("ol", { class: "list" });
  for (const i of members.slice(0, 100)) list.append(el("li", {}, el("button", { on: { click: () => select(i, true) } }, `${p.title[i]} (${p.year[i]})`)));
  box.replaceChildren(
    el("button", { class: "close", "aria-label": "Show all topics", on: { click: () => pickTopic(-1) } }, "×"),
    el("h2", {}, "Topic"),
    el("h3", {}, topicName(t)),
    el("div", { class: "meta" }, `${members.length.toLocaleString()} papers in the years shown; papers per year, and their share of the year:`),
    chart,
    el("h2", {}, members.length > 100 ? `Papers (the latest 100 of ${members.length.toLocaleString()})` : "Papers"),
    list,
  );
  box.hidden = false;
}

function showResults(order, scores, label) {
  const a = state.atlas;
  state.hits = new Map(order.map((i, r) => [i, r]));
  state.topic = -1;
  const box = $("results");
  const list = el("ol", { class: "list" });
  order.slice(0, 50).forEach((i, r) => {
    const tail = scores ? el("span", { class: "score" }, ` ${scores[r].toFixed(2)}`) : "";
    list.append(el("li", {}, el("button", { on: { click: () => select(i, true) } }, `${a.papers.title[i]} (${a.papers.year[i]})`), tail));
  });
  box.replaceChildren(
    el("button", { class: "close", "aria-label": "Clear the search", on: { click: () => { state.hits = null; box.hidden = true; $("q").value = ""; draw(); } } }, "×"),
    el("h2", {}, label),
    order.length ? list : el("div", { class: "meta" }, "Nothing matches."),
  );
  box.hidden = false;
  renderTopics();
  draw();
}

// ------------------------------------------------------------------ search

function searchWords(q) {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  const p = state.atlas.papers;
  const order = [];
  for (let i = 0; i < state.n; i++) {
    if (!visible(i)) continue;
    const t = p.title[i].toLowerCase();
    if (words.every((w) => t.includes(w))) order.push(i);
  }
  showResults(order, null, `${order.length.toLocaleString()} titles with “${q}”`);
}

async function searchMeaning(q) {
  await loadModel();
  const m = state.model;
  const ids = encode(q, m.vocab, m.unk);
  if (!ids.length) return showResults([], null, `No words of “${q}” are in the model's vocabulary`);
  const dim = m.dim;
  const qv = new Float32Array(dim);
  for (const id of ids) for (let d = 0; d < dim; d++) qv[d] += m.table[id * dim + d] * m.scale[id];
  // as the papers' vectors were made: the mean of the tokens, normalized; the venue's mean taken away; normalized again
  let first = 0;
  for (let d = 0; d < dim; d++) first += qv[d] * qv[d];
  first = Math.sqrt(first) || 1;
  const mean = state.atlas.model.mean || [];
  for (let d = 0; d < dim; d++) qv[d] = qv[d] / first - (mean[d] || 0);
  let norm = 0;
  for (let d = 0; d < dim; d++) norm += qv[d] * qv[d];
  norm = Math.sqrt(norm) || 1;
  const scale = state.atlas.papers.scale;
  const v = state.vectors;
  const scored = [];
  for (let i = 0; i < state.n; i++) {
    if (!visible(i)) continue;
    let s = 0;
    const base = i * dim;
    for (let d = 0; d < dim; d++) s += v[base + d] * qv[d];
    scored.push([(s * scale[i]) / norm, i]);
  }
  scored.sort((x, y) => y[0] - x[0]);
  const top = scored.slice(0, 200);
  showResults(top.map((x) => x[1]), top.map((x) => x[0]), `Closest in meaning to “${q}”`);
}

$("search").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q || !state.atlas) return;
  try {
    $("mode").value === "meaning" ? await searchMeaning(q) : searchWords(q);
  } catch (err) {
    status.hidden = false;
    status.textContent = `Search failed: ${err.message}`;
  }
});

// ------------------------------------------------------------------ pointer

let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, vx: state.view.x, vy: state.view.y, moved: false };
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", (e) => {
  if (!state.atlas) return;
  const r = canvas.getBoundingClientRect();
  if (drag) {
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    state.view.x = drag.vx + dx;
    state.view.y = drag.vy + dy;
    tip.hidden = true;
    draw();
    return;
  }
  const i = nearest(e.clientX - r.left, e.clientY - r.top);
  if (i < 0) {
    tip.hidden = true;
    return;
  }
  const p = state.atlas.papers;
  tip.replaceChildren(document.createTextNode(p.title[i]), el("small", {}, `${p.year[i]} · ${topicName(p.topic[i])}`));
  tip.style.left = `${Math.min(e.clientX - r.left + 12, r.width - 350)}px`;
  tip.style.top = `${e.clientY - r.top + 12}px`;
  tip.hidden = false;
});
canvas.addEventListener("pointerup", (e) => {
  canvas.classList.remove("dragging");
  const moved = drag && drag.moved;
  drag = null;
  if (moved || !state.atlas) return;
  const r = canvas.getBoundingClientRect();
  select(nearest(e.clientX - r.left, e.clientY - r.top));
});
canvas.addEventListener("pointerleave", () => (tip.hidden = true));
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const { w, h } = size();
  const factor = Math.exp(-e.deltaY * 0.0015);
  const k = Math.min(40, Math.max(0.6, state.view.k * factor));
  const f = k / state.view.k;
  // zoom about the pointer
  const cx = e.clientX - r.left - w / 2 - state.view.x;
  const cy = e.clientY - r.top - h / 2 - state.view.y;
  state.view.x -= cx * (f - 1);
  state.view.y -= cy * (f - 1);
  state.view.k = k;
  draw();
}, { passive: false });
// the keyboard: arrows move the map, + and - zoom, 0 resets, Escape clears the selection and the search
canvas.addEventListener("keydown", (e) => {
  if (!state.atlas) return;
  const step = e.shiftKey ? 120 : 40;
  const moves = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
  if (moves[e.key]) {
    state.view.x += moves[e.key][0];
    state.view.y += moves[e.key][1];
  } else if (e.key === "+" || e.key === "=") {
    state.view.k = Math.min(40, state.view.k * 1.25);
  } else if (e.key === "-" || e.key === "_") {
    state.view.k = Math.max(0.6, state.view.k / 1.25);
  } else if (e.key === "0") {
    state.view = { x: 0, y: 0, k: 1 };
  } else if (e.key === "Escape") {
    state.hits = null;
    $("results").hidden = true;
    select(-1);
    pickTopic(-1);
  } else {
    return;
  }
  e.preventDefault();
  draw();
});
window.addEventListener("resize", draw);
$("colour").addEventListener("change", (e) => {
  state.colour = e.target.value;
  draw();
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);

// ------------------------------------------------------------------ start

async function start() {
  try {
    const index = await json("data/index.json");
    const sel = $("venue");
    for (const v of index) sel.append(el("option", { value: v.venue }, `${v.name} (${v.count.toLocaleString()})`));
    const [hashVenue, hashPaper] = decodeURIComponent(location.hash.slice(1)).split("/");
    const first = index.find((v) => v.venue === hashVenue) || index[0];
    sel.value = first.venue;
    sel.addEventListener("change", () => loadVenue(sel.value).catch(fail));
    window.addEventListener("hashchange", () => {
      const [v, paper] = decodeURIComponent(location.hash.slice(1)).split("/");
      if (v && v !== state.atlas?.venue && index.some((x) => x.venue === v)) {
        sel.value = v;
        loadVenue(v).then(() => {
          const j = Number(paper);
          if (Number.isInteger(j) && j >= 0 && j < state.n) select(j, true);
        }).catch(fail);
      }
    });
    await loadVenue(first.venue);
    const i = Number(hashPaper);
    if (hashVenue === first.venue && Number.isInteger(i) && i >= 0 && i < state.n) select(i, true);
  } catch (err) {
    fail(err);
  }
}
function fail(err) {
  status.hidden = false;
  status.textContent = `Could not load the atlas: ${err.message}`;
}
start();
