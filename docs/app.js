/* WAG project page — wearable knowledge graph explorer, built on @antv/g6 v5.
 *
 * index.html ships a pre-laid-out static SVG of the same graph (see build.py).
 * G6 mounts over it and the static copy is only retired once G6 has actually
 * rendered, so a failure here leaves a picture on screen rather than a gap.
 */
(function () {
  "use strict";

  var TYPES = ["Physiological", "Sleep", "Activity", "Lifestyle",
               "Mental", "Environmental", "Demographic"];

  var mount = document.getElementById("graph-mount");
  var staticSvg = document.getElementById("graph");
  var loading = document.getElementById("loading");
  var panelEmpty = document.getElementById("panel-empty");
  var panelBody = document.getElementById("panel-body");
  var thresholdEl = document.getElementById("threshold");
  var thresholdOut = document.getElementById("threshold-val");
  var edgeCountEl = document.getElementById("edge-count");
  var searchEl = document.getElementById("search");
  var labelsEl = document.getElementById("labels");
  var hullsEl = document.getElementById("hulls");

  var nodes = [];        // {i, id, name, type, desc, range, rec, cui, ds, deg}
  var allEdges = [];     // {s, t, w, k}
  var adjacency = [];    // per node: [{j, w, k}] sorted by weight desc
  var edgeDesc = null;   // relation descriptions, loaded on demand
  var active = {};       // type -> shown?
  var selected = null;
  var graph = null;
  var colors = {};
  var POS = null;      // baked coordinates from data/layout.json, via kg.js

  TYPES.forEach(function (t) { active[t] = true; });

  function note(msg) {
    loading.textContent = msg;
    loading.hidden = false;
  }

  if (typeof G6 === "undefined") {
    note("Static view — the G6 library could not be loaded.");
    return;
  }

  function nodeSize(deg) { return 9 + Math.sqrt(deg) * 2.4; }

  // G6's Hull pads by each member's full render bounds, and those include the
  // node's text label: with labels on, every hull swells by half a label width
  // and the categories bury one another. Pad by the node itself instead.
  class TightHull extends G6.Hull {
    getPadding() {
      var r = 0;
      this.hullMemberIds.forEach(function (id) {
        var n = nodes[+String(id).slice(1)];
        if (n) r = Math.max(r, nodeSize(n.deg) / 2);
      });
      return r + this.options.padding;
    }
  }
  G6.register("plugin", "tight-hull", TightHull);

  /* ---------- colours follow the page's light/dark tokens ---------- */

  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    TYPES.forEach(function (t) {
      colors[t] = cs.getPropertyValue("--t-" + t.toLowerCase()).trim() || "#6d7480";
    });
    colors.edge = cs.getPropertyValue("--faint").trim() || "#8b919c";
    colors.text = cs.getPropertyValue("--text").trim() || "#17191d";
    colors.surface = cs.getPropertyValue("--surface").trim() || "#ffffff";
  }
  readColors();

  var darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  if (darkQuery.addEventListener) {
    darkQuery.addEventListener("change", function () {
      readColors();
      if (graph) applyFilters();
    });
  }

  /* ---------- data ---------- */

  function boot(data) {
    try {
      start(data);
    } catch (err) {
      note("Static view — interactive graph failed: " + (err && err.message ? err.message : err));
      if (typeof console !== "undefined") console.error(err);
    }
  }


  /* ---------- cluster hulls (as in the paper figure) ---------- */

  var TOOLTIP = {
    type: "tooltip",
    key: "tip",
    trigger: "hover",
    enterable: false,
    getContent: function (ev, items) {
      var html = "";
      items.forEach(function (item) {
        var d = (item && item.data) || {};
        if (d.kind === "node") {
          html += "<b>" + esc(d.name) + "</b><br><span class='tt-type'>" +
                  esc(d.type) + "</span>" +
                  (d.range ? "<br>" + esc(trim(d.range, 150)) : "");
        } else if (d.kind === "edge") {
          html += "<b>" + esc(d.a) + " &rarr; " + esc(d.b) + "</b><br>weight " +
                  Number(d.w).toFixed(2);
        }
      });
      return html;
    }
  };

  // One rounded hull per category, labelled, over the nodes currently shown.
  // Fewer than three members has no meaningful hull, so those are skipped.
  function buildHulls() {
    if (!hullsEl || !hullsEl.checked) return [];
    var byType = {};
    nodes.forEach(function (n) {
      if (!visible(n)) return;
      (byType[n.type] || (byType[n.type] = [])).push(n.id);
    });
    var out = [];
    TYPES.forEach(function (t) {
      if (!byType[t] || byType[t].length < 3) return;
      var c = colors[t];
      var key = "hull-" + t.toLowerCase();
      var base = {
        type: "tight-hull",
        members: byType[t],
        corner: "rounded",
        padding: 14,
        pointerEvents: "none"  // clicks inside a hull still reach the canvas
      };
      // The shaded area sits under edges and nodes. Its label rides on a second,
      // otherwise invisible hull above them, or the nodes would cover it.
      out.push(Object.assign({
        key: key,
        zIndex: -100,
        fill: c,
        fillOpacity: 0.13,
        stroke: c,
        strokeOpacity: 0.45,
        label: false
      }, base));
      out.push(Object.assign({
        key: key + "-label",
        zIndex: 100,
        fillOpacity: 0,
        strokeOpacity: 0,
        labelText: t,
        labelAutoRotate: false,
        labelFill: "#fff",
        labelPadding: 3,
        labelBackgroundFill: c,
        labelBackgroundRadius: 5
      }, base));
    });
    return out;
  }

  function buildPlugins() {
    return [TOOLTIP].concat(buildHulls());
  }

  function start(data) {
    nodes = data.nodes.map(function (n, i) {
      return {
        i: i, id: "n" + i, name: n.name,
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

    POS = data.pos || null;

    buildChips();

    graph = new G6.Graph({
      container: mount,
      autoFit: "view",
      padding: 20,
      // Positions are fixed, so animation only adds fades, and G6 throws for
      // every element a filter removes while its update animation is running.
      animation: false,
      data: buildData(),
      // Colours are read through callbacks so a light/dark switch picks them up.
      node: {
        style: {
          size: function (d) { return nodeSize(d.data.deg); },
          fill: typeColor,
          stroke: function () { return colors.surface; },
          lineWidth: 1,
          // G6 only resets what the base style names, so everything a state
          // changes needs a base value here or it sticks after the state goes
          opacity: 1,
          halo: false,
          labelOpacity: 1,
          labelFontWeight: 400,
          labelText: function (d) { return labelsEl.checked ? d.data.name : ""; },
          labelFill: function () { return colors.text; },
          labelFontSize: 10,
          labelBackground: true,
          labelBackgroundFill: function () { return colors.surface; },
          labelBackgroundOpacity: 0.72,
          labelBackgroundRadius: 3,
          labelPlacement: "bottom",
          labelMaxWidth: 130
        },
        state: {
          // hovered and selected nodes wear a halo in their own colour;
          // neighbours of the hovered node simply keep theirs
          hover: {
            lineWidth: 2,
            halo: true,
            haloStroke: typeColor,
            haloStrokeOpacity: 0.35,
            haloLineWidth: 12,
            labelFontWeight: 600
          },
          dim: { opacity: 0.2, labelOpacity: 0 },
          selected: {
            lineWidth: 2.5,
            stroke: function () { return colors.text; },
            halo: true,
            haloStroke: typeColor,
            haloStrokeOpacity: 0.35,
            haloLineWidth: 12,
            labelFontWeight: 600
          }
        }
      },
      edge: {
        style: {
          stroke: function () { return colors.edge; },
          lineWidth: function (d) { return 0.5 + d.data.w * 1.2; },
          strokeOpacity: function (d) { return 0.1 + (d.data.w - 0.3) * 0.45; }
        },
        state: {
          highlight: {
            stroke: function () { return hoverColor || colors.text; },
            strokeOpacity: 0.7,
            lineWidth: 1.6
          },
          dim: { strokeOpacity: 0.03 }
        }
      },
      // No runtime layout: coordinates are baked into data/layout.json by
      // tools/bake-layout.html, so the page and its static SVG always agree.
      // Zooming is the wheel listener below, not G6's zoom-canvas; hovering is
      // hover() below, not G6's hover-activate.
      behaviors: [
        "drag-canvas",
        "drag-element",        // drag-element-force needs a live d3-force layout
        // hides colliding node labels, highest-degree first, and brings them
        // back as you zoom in
        { type: "auto-adapt-label", key: "labels", padding: 2 }
      ],
      plugins: buildPlugins()
    });

    graph.on("node:pointerenter", function (ev) {
      if (!dragging) hover(ev.target.id);
    });
    graph.on("node:pointerleave", function () {
      if (!dragging) hover(null);
    });
    // G6 sends no node:pointerleave when the pointer leaves the canvas straight
    // from a node, so neither the highlight nor the tooltip would go away
    mount.addEventListener("pointerleave", function () {
      if (dragging) return;
      hover(null);
      var tip = graph.getPluginInstance("tip");
      if (tip) tip.hide();
    });
    graph.on("node:dragstart", function () { dragging = true; hover(null); });
    graph.on("node:dragend", function () { dragging = false; });

    graph.on("node:click", function (ev) {
      var id = ev && ev.target && ev.target.id;
      var n = id ? nodes[+String(id).slice(1)] : null;
      if (n) select(n);
    });
    graph.on("canvas:click", function () { select(null); });

    // A plain wheel scrolls the page; Ctrl/⌘ + wheel zooms, and so does a
    // trackpad pinch, which arrives as a ctrl+wheel with no key pressed.
    // G6's zoom-canvas matches held keys exactly, so it can't cover all three.
    mount.addEventListener("wheel", function (ev) {
      if (!(ev.ctrlKey || ev.metaKey)) return;
      ev.preventDefault();
      var r = mount.getBoundingClientRect();
      var step = Math.max(-50, Math.min(50, -ev.deltaY));   // zoom-canvas's own curve
      graph.zoomTo(graph.getZoom() * (1 + step / 100), false,
                   [ev.clientX - r.left, ev.clientY - r.top]);
    }, { passive: false });

    applyCounts();

    Promise.resolve(graph.render()).then(onRendered, function (err) {
      note("Static view — G6 render failed: " + (err && err.message ? err.message : err));
      if (typeof console !== "undefined") console.error(err);
    });
  }

  function onRendered() {
    // G6 is on screen — retire the static copy
    if (staticSvg && staticSvg.parentNode) staticSvg.parentNode.removeChild(staticSvg);
    mount.classList.add("ready");
    loading.hidden = true;
  }

  /* ---------- filtering ---------- */

  function threshold() { return +thresholdEl.value; }
  function visible(n) { return active[n.type]; }

  function buildData() {
    var t = threshold();
    nodes.forEach(function (n) { n.deg = 0; });

    var edges = [];
    allEdges.forEach(function (e) {
      if (e.w < t) return;
      var a = nodes[e.s], b = nodes[e.t];
      if (!visible(a) || !visible(b)) return;
      a.deg++; b.deg++;
      edges.push({
        id: "e" + e.k,
        source: a.id,
        target: b.id,
        data: { kind: "edge", w: e.w, k: e.k, a: a.name, b: b.name }
      });
    });

    var shown = nodes.filter(visible).map(function (n) {
      var item = {
        id: n.id,
        data: {
          kind: "node", i: n.i, name: n.name, type: n.type,
          range: n.range, deg: n.deg
        }
      };
      if (POS && POS[n.i]) item.style = { x: POS[n.i][0], y: POS[n.i][1] };
      return item;
    });

    return { nodes: shown, edges: edges };
  }

  function applyCounts() {
    var t = threshold();
    var count = 0;
    allEdges.forEach(function (e) {
      if (e.w >= t && visible(nodes[e.s]) && visible(nodes[e.t])) count++;
    });
    edgeCountEl.textContent = count.toLocaleString();
    thresholdOut.textContent = t.toFixed(2);
  }

  function applyFilters() {
    if (!graph) return;
    applyCounts();
    if (selected && !visible(selected)) select(null);
    graph.setData(buildData());
    graph.setOptions({ plugins: buildPlugins() });
    graph.render();
  }

  /* ---------- hover ---------- */

  var hoverColor = null;   // read by the edge highlight style
  var dragging = false;

  function typeColor(d) { return colors[d.data.type]; }

  // G6's hover-activate gives the hovered node and its neighbours one shared
  // state, and sets it before any callback runs, so the hovered node can't be
  // told apart and the edges can't take its colour. This does both.
  function hover(id) {
    if (!graph) return;
    var data = graph.getData();
    var states = {};
    var near = {}, lit = {};
    if (id) {
      hoverColor = typeColor(graph.getNodeData(id));
      near[id] = true;
      graph.getNeighborNodesData(id).forEach(function (n) { near[n.id] = true; });
      graph.getRelatedEdgesData(id).forEach(function (e) { lit[e.id] = true; });
    }
    data.nodes.forEach(function (n) {
      if (selected && n.id === selected.id) states[n.id] = ["selected"];
      else if (!id) states[n.id] = [];
      else states[n.id] = n.id === id ? ["hover"] : near[n.id] ? [] : ["dim"];
    });
    data.edges.forEach(function (e) {
      states[e.id] = !id ? [] : lit[e.id] ? ["highlight"] : ["dim"];
    });
    graph.setElementState(states, false);
  }

  /* ---------- selection ---------- */

  function select(n) {
    if (graph && selected && visible(selected)) {
      try { graph.setElementState(selected.id, []); } catch (e) { /* gone after a refilter */ }
    }
    selected = n;
    if (!n) {
      panelBody.hidden = true;
      panelEmpty.hidden = false;
      return;
    }
    if (graph && visible(n)) {
      try {
        graph.setElementState(n.id, ["selected"]);
        graph.focusElement(n.id);
      } catch (e) { /* not rendered yet */ }
    }
    renderPanel(n);
  }

  /* ---------- details panel ---------- */

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function trim(s, n) {
    s = String(s || "");
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  function cls(type) { return "t-" + type.toLowerCase(); }

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
          '<span class="bar ' + cls(m.type) + '"><i style="width:' +
            Math.round(a.w * 100) + '%"></i></span>' +
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

  // 1.2 MB of relation prose: loaded on first use, as a script tag rather than a
  // fetch so it also works when the page is opened off the filesystem.
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
        s.onload = function () { edgeDesc = window.__WAG_KG_EDGES__ || []; resolve(edgeDesc); };
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
        applyFilters();
      });
      box.appendChild(b);
    });
  }

  var pending = null;
  thresholdEl.addEventListener("input", function () {
    applyCounts();
    clearTimeout(pending);
    pending = setTimeout(function () {
      applyFilters();
      if (selected) renderPanel(selected);
    }, 200);
  });

  labelsEl.addEventListener("change", function () {
    if (graph) graph.render();
  });

  if (hullsEl) {
    hullsEl.addEventListener("change", function () {
      if (!graph) return;
      graph.setOptions({ plugins: buildPlugins() });
      graph.render();
    });
  }

  document.getElementById("reset").addEventListener("click", function () {
    select(null);
    if (graph) graph.fitView();
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
  /* ---------- go ---------- */
  // Kept last: every var above must be initialised before start() runs.
  if (window.__WAG_KG__) {
    boot(window.__WAG_KG__);
  } else {
    fetch("data/kg.json")
      .then(function (r) { return r.json(); })
      .then(boot)
      .catch(function (e) { note("Static view — graph data unavailable (" + e.message + ")."); });
  }

})();
