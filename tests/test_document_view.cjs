// Run with: node --test tests/test_document_view.cjs
// Exercise production graph.js with an in-memory DOM/fetch adapter; no browser
// or live database is required for upload and refresh state transitions.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const script = fs.readFileSync(path.join(__dirname, "../src/api/static/graph.js"), "utf8");
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(setImmediate); };

function setup(saved = null) {
  const elements = new Map(), requests = [], timers = new Map();
  const storage = new Map(saved ? [["doc2graph.activeDocument", JSON.stringify(saved)]] : []);
  let start, nextTimer = 0, rejectUpload = false, nextJob = 0;
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      style: {}, listeners: {}, textContent: "", value: "", files: [],
      clientWidth: 900, clientHeight: 600, classList: { add() {}, remove() {}, toggle() {} },
      addEventListener(name, callback) { this.listeners[name] = callback; },
      appendChild() {}, click() {},
    });
    return elements.get(id);
  };
  const chain = new Proxy(function () {}, {
    get(_target, key) { return key === "then" ? undefined : chain; },
    apply() { return chain; },
  });
  vm.runInNewContext(script, {
    document: {
      getElementById: element, querySelector: element, querySelectorAll: () => [],
      createElement: element, addEventListener(_name, callback) { start = callback; },
    },
    window: { addEventListener() {} }, d3: chain, console,
    localStorage: {
      getItem: (key) => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key),
    },
    FormData: class { append() {} },
    setInterval(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearInterval(id) { timers.delete(id); },
    async fetch(url) {
      requests.push(url);
      let data;
      if (url === "/api/jobs") data = [{ status: "done", filename: "first.pdf", result: { doc_id: "first" } }];
      else if (url === "/api/upload") {
        if (rejectUpload) return { ok: false, json: async () => ({ detail: "Invalid document" }) };
        data = { job_id: "job-" + ++nextJob };
      } else if (url.startsWith("/api/jobs/")) data = { status: "done", result: { doc_id: "second", entities: 2, edges: 1 } };
      else if (url.startsWith("/api/graph/stats")) data = { entity_count: 2, edge_count: 1, doc_count: 1 };
      else if (url.startsWith("/api/search")) data = [];
      else data = { nodes: [], links: [] };
      return { ok: true, json: async () => data };
    },
  });
  start();
  return { element, requests, timers, storage, reject() { rejectUpload = true; } };
}

test("initial view and refresh only fetch the latest uploaded document", async () => {
  const ui = setup();
  await flush();
  assert.match(ui.element("activeDocumentName").textContent, /first.pdf/);
  ui.element("btnRefresh").listeners.click();
  await flush();
  const reads = ui.requests.filter((url) => url.startsWith("/api/graph/"));
  assert.equal(reads.length, 4);
  assert.ok(reads.every((url) => url.includes("doc_id=first")));
});

test("new upload clears old view and scopes graph, search and persistence", async () => {
  const ui = setup({ docId: "first", filename: "first.pdf" });
  await flush();
  const start = ui.requests.length;
  ui.element("fileInput").files = [{ name: "second.pdf" }];
  ui.element("fileInput").listeners.change();
  assert.equal(ui.element("statEntities").textContent, "0");
  await flush();
  assert.equal(ui.timers.size, 1);
  await [...ui.timers.values()][0]();
  await flush();
  ui.element("searchInput").value = "Google";
  await ui.element("btnSearch").listeners.click();
  const reads = ui.requests.slice(start).filter((url) => url.startsWith("/api/graph/") || url.startsWith("/api/search"));
  assert.equal(reads.length, 3);
  assert.ok(reads.every((url) => url.includes("doc_id=second")));
  assert.equal(JSON.parse(ui.storage.get("doc2graph.activeDocument")).docId, "second");
  assert.match(ui.element("activeDocumentName").textContent, /second.pdf/);
  assert.equal(ui.timers.size, 0);
});

test("failed upload never falls back to the combined graph", async () => {
  const ui = setup({ docId: "first", filename: "first.pdf" });
  await flush();
  ui.reject();
  const start = ui.requests.length;
  ui.element("fileInput").files = [{ name: "broken.pdf" }];
  ui.element("fileInput").listeners.change();
  await flush();
  ui.element("btnRefresh").listeners.click();
  await flush();
  assert.deepEqual(ui.requests.slice(start), ["/api/upload"]);
  assert.equal(ui.element("statEntities").textContent, "0");
  assert.match(ui.element("activeDocumentName").textContent, /No document is displayed/);
});
