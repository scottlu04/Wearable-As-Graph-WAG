/* WAG project page — interactive wearable knowledge graph explorer (SVG). */
(function () {
  "use strict";

  var TYPES = ["Physiological", "Sleep", "Activity", "Lifestyle",
               "Mental", "Environmental", "Demographic"];

  function cls(type) { return "t-" + type.toLowerCase(); }

  if (typeof d3 === "undefined") {
    document.getElementById("loading").textContent =
      "The d3 library could not be loaded, so the graph cannot be drawn.";
    return;
  }

  var svg = d3.select("#graph");
  var root = svg.append("g");
  var gLinks = root.append("g").attr("class", "links");
  var gNodes = root.append("g").attr("class", "nodes");
  var gLabels = root.append("g").attr("class", "labels");

  var tip = document.getElementById("tip");
  var loading = document.getElementById("loading");
  var panelEmpty = document.getElementById("panel-empty");
  var panelBody = document.getElementById("panel-body");
  var thresholdEl = document.getElementById("threshold");
  var thresholdOut = document.getElementById("threshold-val");
  var edgeCountEl = document.getElementById("edge-count");
  var searchEl = document.getElementById("search");
  var labelsEl = document.getElementById("labels");

  var nodes = [];        // {i, name, type, desc, range, rec, cui, ds, deg, x, y}
  var allEdges = [];     // {s, t, w, k}
  var links = [];        // active subset (d3 replaces source/target with node objects)
  var adjacency = [];    // per node: [{j, w, k}] sorted by weight desc
  var edgeDesc = null;   // lazily fetched relation descriptions
  var active = {};       // type -> shown?
  var selected = null, hovered = null;
  var sim = null, zoom = null, k = 1;
  var width = 0, height = 0;

  var nodeSel = null, linkSel = null, labelSel = null;

  TYPES.forEach(function (t) { active[t] = true; });

  if (window.__WAG_KG__) {
    start(window.__WAG_KG__);
  } else {
    // fallback for anyone loading app.js without data/kg.js
    fetch("data/kg.json")
      .then(function (r) { return r.json(); })
      .then(start)
      .catch(function (e) {
        loading.textContent = "Could not load the graph data (" + e.message + ").";
      });
  }

  /* ---------- setup ---------- */

  function start(data) {
    nodes = data.nodes.map(function (n, i) {
      return {
        i: i, name: n.name,
        type: TYPES.indexOf(n.type) >= 0 ? n.type : "Demographic",
        desc: n.desc, range: n.range, rec: n.rec, cui: n.cui,
        ds: n.ds || [], deg: 0
      };
    });

    adjacency = nodes.map(function () { return []; });
    data.edges.forEach(function (e, idx) {
      allEdges.push({ s: e[0], t: e[1], w: e[2], k: idx });
      adjacency[e[0]].push({ j: e[1], w: e[2], k: idx });
      adjacency[e[1]].push({ j: e[0], w: e[2], k: idx });
    });
    adjacency.forEach(function (l) { l.sort(function (a, b) { return b.w - a.w; }); });

    buildChips();
    measure();

    sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(function (d) { return d.i; })
        .distance(function (d) { return 30 + (1 - d.w) * 150; })
        .strength(function (d) { return d.w * 0.35; }))
      .force("charge", d3.forceManyBody().strength(-260).distanceMax(520))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x", d3.forceX(width / 2).strength(0.035))
      .force("y", d3.forceY(height / 2).strength(0.05))
      .force("collide", d3.forceCollide().radius(function (d) { return radius(d) + 5; }))
      .on("tick", tick);

    // nodes and labels are created once; visibility is toggled by filters
    nodeSel = gNodes.selectAll("circle").data(nodes).join("circle")
      .attr("class", function (d) { return cls(d.type); })
      .attr("r", radius)
      .on("mouseenter", function (ev, d) { setHover(d, ev); })
      .on("mousemove", function (ev, d) { setHover(d, ev); })
      .on("mouseleave", function () { setHover(null); })
      .on("click", function (ev, d) { ev.stopPropagation(); select(d); })
      .call(d3.drag().on("start", dragStart).on("drag", dragMove).on("end", dragEnd));

    nodeSel.append("title").text(function (d) { return d.name + " · " + d.type; });

    labelSel = gLabels.selectAll("text").data(nodes).join("text")
      .text(function (d) { return d.name; });

    zoom = d3.zoom().scaleExtent([0.3, 6]).on("zoom", function (ev) {
      k = ev.transform.k;
      root.attr("transform", ev.transform);
      gLabels.style("font-size", (11 / k) + "px");
      gLabels.style("stroke-width", (3 / k) + "px");
      updateLabels();
    });
    svg.call(zoom).on("click", function () { select(null); });

    applyFilters(true);
    loading.hidden = true;
  }

  /* ---------- filtering ---------- */

  function threshold() { return +thresholdEl.value; }
  function visible(n) { return active[n.type]; }
  function radius(n) { return 4.5 + Math.sqrt(n.deg) * 1.05; }

  function applyFilters(reheat) {
    var t = threshold();
    links.length = 0;
    nodes.forEach(function (n) { n.deg = 0; });

    allEdges.forEach(function (e) {
      if (e.w < t) return;
      var a = nodes[e.s], b = nodes[e.t];
      if (!visible(a) || !visible(b)) return;
      links.push({ source: a, target: b, w: e.w, k: e.k });
      a.deg++; b.deg++;
    });

    edgeCountEl.textContent = links.length.toLocaleString();
    thresholdOut.textContent = t.toFixed(2);

    linkSel = gLinks.selectAll("line")
      .data(links, function (d) { return d.k; })
      .join("line")
      .style("stroke-opacity", function (d) { return 0.07 + (d.w - 0.3) * 0.4; });

    if (nodeSel) {
      nodeSel.attr("r", radius).style("display", function (d) {
        return visible(d) ? null : "none";
      });
    }

    sim.force("link").links(links);
    sim.force("collide").radius(function (d) { return radius(d) + 5; });
    sim.alpha(reheat ? 0.9 : 0.45).restart();
    paint();
  }

  /* ---------- layout ---------- */

  function measure() {
    var rect = svg.node().parentNode.getBoundingClientRect();
    width = rect.width; height = rect.height;
    svg.attr("viewBox", "0 0 " + width + " " + height);
  }

  function onResize() {
    measure();
    if (!sim) return;
    sim.force("center", d3.forceCenter(width / 2, height / 2));
    sim.force("x", d3.forceX(width / 2).strength(0.035));
    sim.force("y", d3.forceY(height / 2).strength(0.05));
    sim.alpha(0.3).restart();
  }
  window.addEventListener("resize", onResize);
  if (window.ResizeObserver) {
    new ResizeObserver(onResize).observe(document.querySelector(".canvas-wrap"));
  }

  function tick() {
    var pad = 26;
    nodes.forEach(function (n) {
      n.x = Math.max(pad, Math.min(width - pad, n.x));
      n.y = Math.max(pad, Math.min(height - pad, n.y));
    });

    if (linkSel) {
      linkSel
        .attr("x1", function (d) { return d.source.x; })
        .attr("y1", function (d) { return d.source.y; })
        .attr("x2", function (d) { return d.target.x; })
        .attr("y2", function (d) { return d.target.y; });
    }
    nodeSel.attr("cx", function (d) { return d.x; })
           .attr("cy", function (d) { return d.y; });
    labelSel.attr("x", function (d) { return d.x; })
            .attr("y", function (d) { return d.y + radius(d) + 11 / k; });
  }

  /* ---------- focus + labels ---------- */

  function focusNode() { return selected != null ? selected : hovered; }

  function neighborSet(n) {
    var set = new Set([n.i]);
    var t = threshold();
    adjacency[n.i].forEach(function (a) {
      if (a.w >= t && visible(nodes[a.j])) set.add(a.j);
    });
    return set;
  }

  function paint() {
    var f = focusNode();
    var near = f ? neighborSet(f) : null;

    gLinks.attr("class", f ? "links " + cls(f.type) : "links");

    if (linkSel) {
      linkSel
        .classed("lit", function (d) { return !!f && (d.source.i === f.i || d.target.i === f.i); })
        .classed("dim", function (d) { return !!f && d.source.i !== f.i && d.target.i !== f.i; });
    }
    nodeSel
      .classed("dim", function (d) { return !!f && !near.has(d.i); })
      .classed("sel", function (d) { return !!f && d.i === f.i; });

    updateLabels();
  }

  function updateLabels() {
    if (!labelSel) return;
    var f = focusNode();
    var near = f ? neighborSet(f) : null;
    var on = labelsEl.checked;
    labelSel.style("display", function (d) {
      if (!on || !visible(d)) return "none";
      if (f) return near.has(d.i) ? null : "none";
      return (d.deg >= 6 || k >= 1.4) ? null : "none";
    });
  }

  /* ---------- interaction ---------- */

  function dragStart(ev, d) {
    if (!ev.active) sim.alphaTarget(0.2).restart();
    d.fx = d.x; d.fy = d.y;
    svg.classed("dragging", true);
  }
  function dragMove(ev, d) { d.fx = ev.x; d.fy = ev.y; }
  function dragEnd(ev, d) {
    if (!ev.active) sim.alphaTarget(0);
    d.fx = null; d.fy = null;
    svg.classed("dragging", false);
  }

  function setHover(n, ev) {
    if (n && ev) {
      var rect = svg.node().getBoundingClientRect();
      tip.textContent = n.name;
      tip.style.left = (ev.clientX - rect.left) + "px";
      tip.style.top = (ev.clientY - rect.top) + "px";
      tip.hidden = false;
    } else {
      tip.hidden = true;
    }
    if (n !== hovered) { hovered = n; paint(); }
  }

  function select(n) {
    selected = n;
    if (!n) {
      panelBody.hidden = true;
      panelEmpty.hidden = false;
    } else {
      renderPanel(n);
    }
    paint();
  }

  /* ---------- details panel ---------- */

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function renderPanel(n) {
    var t = threshold();
    var top = adjacency[n.i].filter(function (a) { return visible(nodes[a.j]); }).slice(0, 12);

    var html = "<h4>" + esc(n.name) + "</h4>";
    html += '<div class="kind ' + cls(n.type) + '"><span class="dot"></span>' + esc(n.type) + "</div>";

    html += '<div class="tags">';
    if (n.ds.length) {
      html += n.ds.map(function (d) { return '<span class="tag">' + esc(d) + "</span>"; }).join("");
    } else {
      html += '<span class="tag">background knowledge only</span>';
    }
    html += "</div><dl>";

    if (n.desc) html += '<dt>Definition</dt><dd class="body">' + esc(n.desc) + "</dd>";
    if (n.range) html += "<dt>Typical range</dt><dd>" + esc(n.range) + "</dd>";
    if (n.rec) html += "<dt>Recommendation</dt><dd>" + esc(n.rec) + "</dd>";
    if (n.cui) html += "<dt>UMLS</dt><dd>" + esc(n.cui) + "</dd>";
    html += "<dt>Strongest relations</dt></dl>";

    html += '<ul class="neighbors">';
    top.forEach(function (a) {
      var m = nodes[a.j];
      html += "<li>" +
        '<button class="nb" data-j="' + a.j + '" data-k="' + a.k + '">' +
          '<span class="nb-top"><span class="nb-name">' + esc(m.name) + "</span>" +
          '<span class="nb-w">' + a.w.toFixed(2) + (a.w >= t ? "" : " ·") + "</span></span>" +
          '<span class="bar ' + cls(m.type) + '"><i style="width:' + Math.round(a.w * 100) + '%"></i></span>' +
        "</button>" +
        '<div class="nb-desc" hidden></div>' +
      "</li>";
    });
    html += "</ul>";
    html += '<p class="nb-desc" style="padding-top:10px">Weights marked &middot; fall below the ' +
            "current threshold and are not drawn in the graph.</p>";

    panelBody.innerHTML = html;
    panelBody.hidden = false;
    panelEmpty.hidden = true;
    panelBody.scrollTop = 0;

    panelBody.querySelectorAll(".nb").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var box = btn.nextElementSibling;
        if (!box.hidden) { box.hidden = true; return; }
        box.hidden = false;
        box.textContent = "…";
        loadDescriptions().then(function (list) {
          box.textContent = list[+btn.dataset.k] || "No description recorded for this relation.";
        });
      });
      btn.addEventListener("dblclick", function () { select(nodes[+btn.dataset.j]); });
    });
  }

  // Relation descriptions are 1.2 MB, so they load on first use — as a script tag
  // rather than a fetch, so this also works when the page is opened off the filesystem.
  function loadDescriptions() {
    if (edgeDesc) return Promise.resolve(edgeDesc);
    if (window.__WAG_KG_EDGES__) {
      edgeDesc = window.__WAG_KG_EDGES__;
      return Promise.resolve(edgeDesc);
    }
    if (!loadDescriptions.pending) {
      loadDescriptions.pending = new Promise(function (resolve) {
        var s = document.createElement("script");
        s.src = "data/kg_edge_desc.js";
        s.onload = function () {
          edgeDesc = window.__WAG_KG_EDGES__ || [];
          resolve(edgeDesc);
        };
        s.onerror = function () { resolve([]); };
        document.head.appendChild(s);
      });
    }
    return loadDescriptions.pending;
  }

  /* ---------- controls ---------- */

  function buildChips() {
    var box = document.getElementById("type-filters");
    var counts = {};
    nodes.forEach(function (n) { counts[n.type] = (counts[n.type] || 0) + 1; });
    TYPES.forEach(function (t) {
      if (!counts[t]) return;
      var b = document.createElement("button");
      b.className = "chip on " + cls(t);
      b.innerHTML = '<span class="dot"></span>' + t + ' <span class="n">' + counts[t] + "</span>";
      b.addEventListener("click", function () {
        active[t] = !active[t];
        b.classList.toggle("on", active[t]);
        b.classList.toggle("off", !active[t]);
        if (selected && !visible(selected)) select(null);
        applyFilters(false);
      });
      box.appendChild(b);
    });
  }

  thresholdEl.addEventListener("input", function () {
    applyFilters(false);
    if (selected) renderPanel(selected);
  });

  labelsEl.addEventListener("change", updateLabels);

  document.getElementById("reset").addEventListener("click", function () {
    svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity);
    select(null);
    sim.alpha(0.8).restart();
  });

  searchEl.addEventListener("input", function () {
    var q = searchEl.value.trim().toLowerCase();
    if (!q) return;
    var hit = nodes.find(function (n) { return visible(n) && n.name.toLowerCase().indexOf(q) === 0; }) ||
              nodes.find(function (n) { return visible(n) && n.name.toLowerCase().indexOf(q) >= 0; });
    if (hit) select(hit);
  });

  /* ---------- bibtex copy ---------- */

  var copyBtn = document.getElementById("copy-bib");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = document.querySelector("#bibtex code").textContent;
      navigator.clipboard.writeText(text).then(function () {
        copyBtn.textContent = "Copied";
        setTimeout(function () { copyBtn.textContent = "Copy"; }, 1600);
      });
    });
  }
})();
