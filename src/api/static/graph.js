// Doc2Graph — D3.js Force-Directed Interactive Graph Visualization
(function () {
  const TYPE_COLORS = {
    PERSON: "#4F46E5",
    ORG: "#0284C7",
    GPE: "#059669",
    PRODUCT: "#D97706",
    EVENT: "#DC2626",
    LAW: "#7C3AED",
  };
  const DEFAULT_COLOR = "#64748B";

  let svg, g, linkGroup, nodeGroup, labelGroup, edgeLabelGroup;
  let simulation;
  let currentData = { nodes: [], links: [] };
  let activeFilter = "ALL";
  let selectedNode = null;
  let zoomBehavior;
  let activeDocId = null;
  let uploadVersion = 0;
  let graphRequest = 0;

  // ── Initialization ──────────────────────────────────────────────────────────

  function init() {
    const container = document.querySelector(".graph-canvas-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg = d3.select("#graphSvg").attr("width", width).attr("height", height);

    // Arrowhead marker
    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#64748B");

    g = svg.append("g").attr("class", "graph-root");

    linkGroup = g.append("g").attr("class", "links");
    edgeLabelGroup = g.append("g").attr("class", "edge-labels");
    nodeGroup = g.append("g").attr("class", "nodes");
    labelGroup = g.append("g").attr("class", "labels");

    zoomBehavior = d3
      .zoom()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoomBehavior);

    setupControls();
    setupUpload();
    setupSearch();
    setupFilters();

    restoreDocument();

    window.addEventListener("resize", () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      svg.attr("width", w).attr("height", h);
      if (simulation) simulation.force("center", d3.forceCenter(w / 2, h / 2)).restart();
    });
  }

  // ── Fetch & Load ────────────────────────────────────────────────────────────

  function setDocument(docId, filename = "") {
    activeDocId = docId;
    graphRequest += 1;
    unselectNode();
    currentData = { nodes: [], links: [] };
    activeFilter = "ALL";
    document.querySelectorAll("#typeFilters .type-tag").forEach((button) => {
      button.classList.toggle("active", button.dataset.type === "ALL");
    });
    renderGraph(currentData);
    document.getElementById("searchResults").style.display = "none";
    document.getElementById("searchInput").value = "";
    ["statEntities", "statEdges", "statDocs"].forEach((id) => {
      document.getElementById(id).textContent = "0";
    });
    document.getElementById("activeDocumentName").textContent = docId
      ? `Showing only: ${filename || "uploaded document"}`
      : "Upload a document to view its graph.";
    try {
      if (docId) localStorage.setItem("doc2graph.activeDocument", JSON.stringify({ docId, filename }));
      else localStorage.removeItem("doc2graph.activeDocument");
    } catch (e) { /* The graph also works when browser storage is disabled. */ }
  }

  async function restoreDocument() {
    const version = uploadVersion;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem("doc2graph.activeDocument")); } catch (e) { /* no saved view */ }
    if (!saved?.docId) {
      try {
        const response = await fetch("/api/jobs");
        if (response.ok) {
          const jobs = await response.json();
          const latest = jobs.find((job) => job.status === "done" && job.result?.doc_id);
          if (latest) saved = { docId: latest.result.doc_id, filename: latest.filename };
        }
      } catch (e) { console.warn("Could not restore uploaded document", e); }
    }
    if (version !== uploadVersion) return;
    setDocument(saved?.docId || null, saved?.filename);
    loadStats();
    loadGraph();
  }

  async function loadStats() {
    const docId = activeDocId;
    if (!docId) return;
    try {
      const res = await fetch(`/api/graph/stats?doc_id=${encodeURIComponent(docId)}`);
      if (res.ok) {
        const stats = await res.json();
        if (docId !== activeDocId) return;
        document.getElementById("statEntities").textContent = stats.entity_count || 0;
        document.getElementById("statEdges").textContent = stats.edge_count || 0;
        document.getElementById("statDocs").textContent = stats.doc_count || 0;
      }
    } catch (e) {
      console.warn("Could not load graph stats", e);
    }
  }

  async function loadGraph(centerEntityId = null) {
    const docId = activeDocId;
    const request = ++graphRequest;
    if (!docId) return;
    try {
      let url = `/api/graph/visualization?limit=150&doc_id=${encodeURIComponent(docId)}`;
      if (centerEntityId) url += `&entity_id=${encodeURIComponent(centerEntityId)}`;
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      if (docId !== activeDocId || request !== graphRequest) return;
      currentData = data;
      renderGraph(data);
    } catch (e) {
      console.warn("Could not load visualization data", e);
    }
  }

  // ── Render Graph ────────────────────────────────────────────────────────────

  function renderGraph(data) {
    const container = document.querySelector(".graph-canvas-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    const filteredNodes = data.nodes.filter(
      (n) => activeFilter === "ALL" || n.type === activeFilter
    );
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredLinks = data.links.filter(
      (l) =>
        nodeIds.has(typeof l.source === "object" ? l.source.id : l.source) &&
        nodeIds.has(typeof l.target === "object" ? l.target.id : l.target)
    );

    // D3 links need object references
    const linksCopy = filteredLinks.map((d) => Object.assign({}, d));
    const nodesCopy = filteredNodes.map((d) => Object.assign({}, d));

    if (simulation) simulation.stop();

    simulation = d3
      .forceSimulation(nodesCopy)
      .force(
        "link",
        d3
          .forceLink(linksCopy)
          .id((d) => d.id)
          .distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(25));

    // Links
    const link = linkGroup
      .selectAll("line")
      .data(linksCopy, (d) => `${d.source.id || d.source}->${d.target.id || d.target}`)
      .join("line")
      .attr("stroke", "#475569")
      .attr("stroke-width", (d) => Math.min(Math.max((d.evidence_count || 1) * 1.5, 1.5), 4))
      .attr("stroke-opacity", 0.6)
      .attr("marker-end", "url(#arrow)");

    // Edge Labels
    const edgeLabel = edgeLabelGroup
      .selectAll("text")
      .data(linksCopy)
      .join("text")
      .attr("font-size", 9)
      .attr("fill", "#94A3B8")
      .attr("text-anchor", "middle")
      .text((d) => d.type || "");

    // Nodes
    const node = nodeGroup
      .selectAll("circle")
      .data(nodesCopy, (d) => d.id)
      .join("circle")
      .attr("r", (d) => (d.id === selectedNode ? 14 : 10))
      .attr("fill", (d) => TYPE_COLORS[d.type] || DEFAULT_COLOR)
      .attr("stroke", "#FFFFFF")
      .attr("stroke-width", (d) => (d.id === selectedNode ? 3 : 1.5))
      .style("cursor", "pointer")
      .call(drag(simulation));

    // Node Labels
    const label = labelGroup
      .selectAll("text")
      .data(nodesCopy, (d) => d.id)
      .join("text")
      .attr("font-size", 11)
      .attr("fill", "#F8FAFC")
      .attr("dx", 14)
      .attr("dy", 4)
      .style("pointer-events", "none")
      .text((d) => d.name);

    // Tooltip & Click Handlers
    const tooltip = document.getElementById("tooltip");
    node
      .on("mouseover", (event, d) => {
        tooltip.style.display = "block";
        tooltip.style.left = event.pageX + 12 + "px";
        tooltip.style.top = event.pageY - 12 + "px";
        tooltip.innerHTML = `<strong>${d.name}</strong><br><span style="color:${TYPE_COLORS[d.type] || '#94A3B8'}">${d.type}</span>`;
      })
      .on("mousemove", (event) => {
        tooltip.style.left = event.pageX + 12 + "px";
        tooltip.style.top = event.pageY - 12 + "px";
      })
      .on("mouseout", () => {
        tooltip.style.display = "none";
      })
      .on("click", (event, d) => {
        event.stopPropagation();
        selectNode(d);
      });

    svg.on("click", () => {
      unselectNode();
    });

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      edgeLabel
        .attr("x", (d) => (d.source.x + d.target.x) / 2)
        .attr("y", (d) => (d.source.y + d.target.y) / 2);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      label.attr("x", (d) => d.x).attr("y", (d) => d.y);
    });
  }

  // ── Drag Behavior ───────────────────────────────────────────────────────────

  function drag(sim) {
    function dragstarted(event) {
      if (!event.active) sim.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }
    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }
    function dragended(event) {
      if (!event.active) sim.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
    return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
  }

  // ── Node Selection & Inspection ─────────────────────────────────────────────

  async function selectNode(d) {
    const docId = activeDocId;
    if (!docId) return;
    selectedNode = d.id;
    nodeGroup.selectAll("circle")
      .attr("stroke-width", (n) => (n.id === selectedNode ? 3 : 1.5))
      .attr("r", (n) => (n.id === selectedNode ? 14 : 10));

    const drawer = document.getElementById("entityDetails");
    drawer.style.display = "block";
    document.getElementById("detailName").textContent = d.name;
    const badge = document.getElementById("detailType");
    badge.textContent = d.type;
    badge.style.backgroundColor = TYPE_COLORS[d.type] || DEFAULT_COLOR;

    // Fetch relations
    const list = document.getElementById("outgoingRels");
    list.innerHTML = "<li>Loading relations...</li>";
    try {
      const res = await fetch(`/api/graph/entities/${encodeURIComponent(d.id)}?doc_id=${encodeURIComponent(docId)}`);
      if (res.ok) {
        const details = await res.json();
        if (docId !== activeDocId || selectedNode !== d.id) return;
        const aliases = details.aliases || [];
        document.getElementById("detailAliases").textContent = aliases.length
          ? "Aliases: " + aliases.join(", ")
          : "";
        list.innerHTML = "";
        const outgoing = details.outgoing || [];
        if (!outgoing.length) {
          list.innerHTML = "<li>No outgoing relations recorded.</li>";
        } else {
          outgoing.forEach((rel) => {
            const li = document.createElement("li");
            li.innerHTML = `<strong>${rel.type}</strong> → ${rel.target_name || rel.target_id} <span style="color:#94A3B8">(${(rel.confidence * 100).toFixed(0)}%)</span>`;
            list.appendChild(li);
          });
        }
      }
    } catch (e) {
      list.innerHTML = "<li>Error loading details.</li>";
    }
  }

  function unselectNode() {
    selectedNode = null;
    nodeGroup.selectAll("circle").attr("stroke-width", 1.5).attr("r", 10);
    document.getElementById("entityDetails").style.display = "none";
  }

  // ── Controls & Event Listeners ──────────────────────────────────────────────

  function setupControls() {
    document.getElementById("zoomIn").addEventListener("click", () => {
      svg.transition().call(zoomBehavior.scaleBy, 1.3);
    });
    document.getElementById("zoomOut").addEventListener("click", () => {
      svg.transition().call(zoomBehavior.scaleBy, 0.7);
    });
    document.getElementById("zoomReset").addEventListener("click", () => {
      svg.transition().call(zoomBehavior.transform, d3.zoomIdentity);
    });
    document.getElementById("btnRefresh").addEventListener("click", () => {
      loadStats();
      loadGraph();
    });
    document.getElementById("btnCloseDetails").addEventListener("click", () => {
      unselectNode();
    });
    document.getElementById("btnFocusNode").addEventListener("click", () => {
      if (selectedNode) loadGraph(selectedNode);
    });
  }

  function setupFilters() {
    const buttons = document.querySelectorAll("#typeFilters .type-tag");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        buttons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeFilter = btn.dataset.type;
        renderGraph(currentData);
      });
    });
  }

  function setupSearch() {
    const input = document.getElementById("searchInput");
    const resultsContainer = document.getElementById("searchResults");

    const doSearch = async () => {
      const docId = activeDocId;
      const q = input.value.trim();
      if (!q || !docId) {
        resultsContainer.style.display = "none";
        return;
      }
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=10&doc_id=${encodeURIComponent(docId)}`);
        if (res.ok) {
          const results = await res.json();
          if (docId !== activeDocId || q !== input.value.trim()) return;
          resultsContainer.innerHTML = "";
          if (!results.length) {
            resultsContainer.innerHTML = "<div class='search-result-item'>No entities found</div>";
          } else {
            results.forEach((r) => {
              const item = document.createElement("div");
              item.className = "search-result-item";
              item.innerHTML = `<span>${r.name}</span> <span class="badge" style="background:${TYPE_COLORS[r.type] || DEFAULT_COLOR}">${r.type}</span>`;
              item.addEventListener("click", () => {
                resultsContainer.style.display = "none";
                selectNode(r);
                loadGraph(r.id);
              });
              resultsContainer.appendChild(item);
            });
          }
          resultsContainer.style.display = "block";
        }
      } catch (e) {
        console.warn("Search failed", e);
      }
    };

    document.getElementById("btnSearch").addEventListener("click", doSearch);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doSearch();
    });
  }

  function setupUpload() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");

    browseBtn.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("click", (e) => {
      if (e.target !== browseBtn) fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) uploadFile(fileInput.files[0]);
    });
  }

  async function uploadFile(file) {
    const version = ++uploadVersion;
    setDocument(null);
    document.getElementById("activeDocumentName").textContent = `Processing: ${file.name}`;
    const statusBox = document.getElementById("jobStatus");
    const fileNameEl = document.getElementById("jobFileName");
    const badgeEl = document.getElementById("jobBadge");
    const progressEl = document.getElementById("jobProgress");
    const msgEl = document.getElementById("jobMsg");

    statusBox.style.display = "block";
    fileNameEl.textContent = file.name;
    badgeEl.textContent = "uploading";
    badgeEl.className = "badge running";
    progressEl.style.width = "25%";
    msgEl.textContent = "Uploading document...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload rejected");
      }

      const data = await res.json();
      if (version !== uploadVersion) return;
      const jobId = data.job_id;
      badgeEl.textContent = "processing";
      progressEl.style.width = "50%";
      msgEl.textContent = "Processing extraction pipeline (NER, relations, resolution)...";

      // Poll job
      const timer = setInterval(async () => {
        if (version !== uploadVersion) { clearInterval(timer); return; }
        try {
          const pollRes = await fetch(`/api/jobs/${jobId}`);
          if (pollRes.ok) {
            const job = await pollRes.json();
            if (version !== uploadVersion) { clearInterval(timer); return; }
            if (job.status === "done") {
              clearInterval(timer);
              badgeEl.textContent = "done";
              badgeEl.className = "badge done";
              progressEl.style.width = "100%";
              msgEl.textContent = `Completed! Extracted ${job.result?.entities || 0} entities, ${job.result?.edges || 0} relations.`;
              setDocument(job.result?.doc_id || null, file.name);
              loadStats();
              loadGraph();
            } else if (job.status === "failed") {
              clearInterval(timer);
              badgeEl.textContent = "failed";
              badgeEl.className = "badge failed";
              msgEl.textContent = `Failed: ${job.error || "Unknown error"}`;
              document.getElementById("activeDocumentName").textContent = "Upload failed. No document is displayed.";
            }
          }
        } catch (e) {
          clearInterval(timer);
        }
      }, 1500);
    } catch (err) {
      if (version !== uploadVersion) return;
      badgeEl.textContent = "failed";
      badgeEl.className = "badge failed";
      progressEl.style.width = "100%";
      msgEl.textContent = `Error: ${err.message}`;
      document.getElementById("activeDocumentName").textContent = "Upload failed. No document is displayed.";
    }
  }

  // Run on DOM load
  document.addEventListener("DOMContentLoaded", init);
})();

