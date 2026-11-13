// How the site does on a phone: headless Chrome over the DevTools protocol, 375x812 at 2x, the CPU slowed (4x by default).
// Serve the site first (python -m http.server --directory site), then:
//   node scripts/phone-perf.mjs [venue] [cpu slowdown] [site address]
// CHROME names the browser if it is not at the usual Windows place. The browser reaches nothing but 127.0.0.1.
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const chrome = process.env.CHROME || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const venue = process.argv[2] || "neurips";
const rate = Number(process.argv[3] || 4);
const site = (process.argv[4] || "http://127.0.0.1:8000").replace(/\/$/, "");
const profile = mkdtempSync(join(tmpdir(), "venue-atlas-chrome-"));
const proc = spawn(chrome, ["--headless=new", "--no-proxy-server", "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
  "--remote-debugging-port=9333", `--user-data-dir=${profile}`, "--no-first-run", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let list;
for (let i = 0; i < 50; i++) {
  try { list = await (await fetch("http://127.0.0.1:9333/json/list")).json(); if (list.some((t) => t.type === "page")) break; } catch {}
  await sleep(200);
}
const page = list.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const waiting = new Map();
ws.addEventListener("message", (m) => { const d = JSON.parse(m.data); if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); } });
const send = (method, params = {}) => new Promise((r) => { const n = ++id; waiting.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const evaluate = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 400));
  return r.result.result.value;
};
try {
  await send("Emulation.setDeviceMetricsOverride", { width: 375, height: 812, deviceScaleFactor: 2, mobile: true });
  await send("Emulation.setCPUThrottlingRate", { rate });
  await send("Page.enable");
  const t0 = Date.now();
  await send("Page.navigate", { url: `${site}/#${venue}` });
  await evaluate(`new Promise(async (done) => { const t = performance.now(); while (!(document.getElementById("status")?.hidden && document.querySelector("canvas")?.width > 300)) { await new Promise(r => setTimeout(r, 20)); if (performance.now() - t > 60000) break; } done(); })`);
  const ready = Date.now() - t0;
  const out = await evaluate(`(async () => {
    const cv = document.querySelector("canvas"); cv.focus();
    const frames = []; let last = performance.now();
    for (let i = 0; i < 30; i++) {
      cv.dispatchEvent(new KeyboardEvent("keydown", {key: i % 3 ? "ArrowLeft" : "+", bubbles: true}));
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      const now = performance.now(); frames.push(now - last); last = now;
    }
    frames.sort((a, b) => a - b);
    // the topic panel: click the first topic
    let t = performance.now();
    document.querySelector("#topics button, .topics button")?.click();
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    const topic = performance.now() - t;
    // a meaning search, the first one loading the model
    document.getElementById("mode").value = "meaning";
    document.getElementById("q").value = "diffusion models for protein design";
    t = performance.now();
    document.getElementById("search").requestSubmit();
    while (!document.getElementById("results").textContent.includes("protein design”")) { await new Promise(r => setTimeout(r, 20)); if (performance.now() - t > 60000) break; }
    const first = performance.now() - t;
    document.getElementById("q").value = "graph neural networks for molecules";
    t = performance.now();
    document.getElementById("search").requestSubmit();
    await new Promise(r => setTimeout(r, 0));
    while (!document.getElementById("results").textContent.includes("molecules”")) { await new Promise(r => setTimeout(r, 10)); if (performance.now() - t > 30000) break; }
    const second = performance.now() - t;
    const mem = performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : null;
    return { frameMedian: Math.round(frames[15]), frameP90: Math.round(frames[27]), frameMax: Math.round(frames[29]), topic: Math.round(topic),
             firstMeaningSearch: Math.round(first), nextMeaningSearch: Math.round(second), heapMB: mem,
             top: document.getElementById("results").innerText.split("\\n").slice(1, 4) };
  })()`);
  console.log(JSON.stringify({ venue, cpuSlowdown: rate, readyMs: ready, ...out }, null, 1));
} finally {
  ws.close();
  proc.kill();
  await sleep(500);
  try { rmSync(profile, { recursive: true, force: true }); } catch {}
}
