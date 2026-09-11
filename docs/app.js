/* WAG project page — interactive wearable knowledge graph explorer. */
(function () {
  "use strict";

  var TYPES = ["Physiological", "Sleep", "Activity", "Lifestyle",
               "Mental", "Environmental", "Demographic"];

  var canvas = document.getElementById("graph");
  var ctx = canvas.getContext("2d");
  var tip = document.getElementById("tip");
  var loading = document.getElementById("loading");
  var panelEmpty = document.getElementById("panel-empty");
  var panelBody = document.getElementById("panel-body");
  var thresholdEl = document.getElementById("threshold");
  var thresholdOut = document.getElementById("threshold-val");
  var edgeCountEl = document.getElementById("edge-count");
  var searchEl = document.getElementById("search");
  var labelsEl = document.getElementById("labels");

  var nodes = [];        // {i, name, type, desc, range, rec, cui, ds, x, y, deg}
  var allEdges = [];     // {s, t, w, k}  (k = index into the description file)
  var links = [];        // active subset, d3 mutates .source/.target
  var adjacency = [];    // per node: [{j, w, k}] sorted by weight desc
  var edgeDesc = null;   // lazily fetched
  var active = {};       // type -> bool
  var selected = null, hovered = null;
  var sim = null, transform = d3.zoomIdentity;
  var colors = {}, ink = {};
  var width = 0, height = 0;

  TYPES.forEach(function (t) { active[t] = true; });

  /* ---------- theme ---------- */

  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    TYPES.forEach(function (t) {
      colors[t] = cs.getPropertyValue("--t-" + t.toLowerCase()).trim() || "#888";
    });
    ink.text = cs.getPropertyValue("--text").trim();
    ink.muted = cs.getPropertyValue("--faint").trim();
    ink.line = cs.getPropertyValue("--line").trim();
    ink.surface = cs.getPropertyValue("--surface").trim();
    ink.accent = cs.getPropertyValue("--accent").trim();
  }

  var dark = window.matchMedia("(prefers-color-scheme: dark)");
  if (dark.addEventListener) {
    dark.addEventListener("change", function () { readColors(); draw(); });
  }

  /* ---------- load ---------- */

  readColors();

  fetch("data/kg.json")
    .then(function (r) { return r.json(); })
    .then(start)
    .catch(function (e) {
      loading.textContent = "Could not load the graph data (" + e.message + ").";
    });

  function start(data) {
    nodes = data.nodes.map(function (n, i) {
      return {
        i: i, name: n.name, type: TYPES.indexOf(n.type) >= 0 ? n.type : "Demographic",
        desc: n.desc, range: n.range, rec: n.rec, cui: n.cui, ds: n.ds || [], deg: 0
      };
    });

    adjacency = nodes.map(function () { return []; });
    data.edges.forEach(function (e, k) {
      var edge = { s: e[0], t: e[1], w: e[2], k: k };
      allEdges.push(edge);
      adjacency[e[0]].push({ j: e[1], w: e[2], k: k });
      adjacency[e[1]].push({ j: e[0], w: e[2], k: k });
    });
    adjacency.forEach(function (list) { list.sort(function (a, b) { return b.w - a.w; }); });

    buildChips();
    resize();

    sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(function (d) { return d.i; })
        .distance(function (d) { return 30 + (1 - d.w) * 150; })
        .strength(function (d) { return d.w * 0.35; }))
      .force("charge", d3.forceManyBody().strength(-260).distanceMax(520))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x", d3.forceX(function () { return width / 2; }).strength(0.035))
      .force("y", d3.forceY(function () { return height / 2; }).strength(0.05))
      .force("collide", d3.forceCollide().radius(function (d) { return radius(d) + 5; }))
      .on("tick", tick);

    applyFilters(true);
    loading.hidden = true;

    d3.select(canvas)
      .call(d3.drag()
        .subject(subject)
        .on("start", dragStart)
        .on("drag", dragMove)
        .on("end", dragEnd))
      .call(d3.zoom().scaleExtent([0.3, 6])
        .filter(function (ev) { return !ev.button && (ev.type !== "mousedown" || !subject(ev)); })
        .on("zoom", function (ev) { transform = ev.transform; draw(); }))
      .on("mousemove", onMove)
      .on("mouseleave", function () { setHover(null); })
      .on("click", onClick);
  }

  /* ---------- filtering ---------- */

  function threshold() { return +thresholdEl.value; }

  function visible(n) { return active[n.type]; }

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

    if (sim) {
      sim.force("link").links(links);
      sim.force("collide").radius(function (d) { return radius(d) + 5; });
      sim.alpha(reheat ? 0.9 : 0.45).restart();
    }
  }

  function radius(n) {
    if (!visible(n)) return 0;
    return 4.5 + Math.sqrt(n.deg) * 1.05;
  }

  /* ---------- drawing ---------- */

  function resize() {
    var rect = canvas.parentNode.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    width = rect.width; height = rect.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (sim) {
      sim.force("center", d3.forceCenter(width / 2, height / 2));
      sim.force("x", d3.forceX(width / 2).strength(0.035));
      sim.force("y", d3.forceY(height / 2).strength(0.05));
      sim.alpha(0.3).restart();
    }
  }
  window.addEventListener("resize", function () { resize(); draw(); });

  function focusNode() { return selected != null ? selected : hovered; }

  function neighborSet(n) {
    var set = new Set([n.i]);
    var t = threshold();
    adjacency[n.i].forEach(function (a) {
      if (a.w >= t && visible(nodes[a.j])) set.add(a.j);
    });
    return set;
  }

  // keep the layout inside the viewport so nothing is clipped at the edges
  function tick() {
    var pad = 26;
    nodes.forEach(function (n) {
      n.x = Math.max(pad, Math.min(width - pad, n.x));
      n.y = Math.max(pad, Math.min(height - pad, n.y));
    });
    draw();
  }

  function draw() {
    ctx.save();
    ctx.clearRect(0, 0, width, height);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    var f = focusNode();
    var near = f ? neighborSet(f) : null;

    // edges
    ctx.lineCap = "round";
    links.forEach(function (l) {
      var lit = !f || (l.source.i === f.i || l.target.i === f.i);
      if (f && !lit) return;
      ctx.globalAlpha = f ? 0.75 : 0.06 + (l.w - 0.3) * 0.34;
      ctx.strokeStyle = f ? colors[f.type] : ink.text;
      ctx.lineWidth = (f ? 0.6 + l.w * 1.8 : 0.7) / Math.sqrt(transform.k);
      ctx.beginPath();
      ctx.moveTo(l.source.x, l.source.y);
      ctx.lineTo(l.target.x, l.target.y);
      ctx.stroke();
    });

    // dimmed edges behind, when focused
    if (f) {
      ctx.globalAlpha = 0.04;
      ctx.strokeStyle = ink.text;
      ctx.lineWidth = 0.6 / Math.sqrt(transform.k);
      ctx.beginPath();
      links.forEach(function (l) {
        if (l.source.i === f.i || l.target.i === f.i) return;
        ctx.moveTo(l.source.x, l.source.y);
        ctx.lineTo(l.target.x, l.target.y);
      });
      ctx.stroke();
    }

    // nodes
    nodes.forEach(function (n) {
      if (!visible(n)) return;
      var r = radius(n);
      var lit = !f || near.has(n.i);
      ctx.globalAlpha = lit ? 1 : 0.16;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, 6.2832);
      ctx.fillStyle = colors[n.type];
      ctx.fill();
      if (f && n.i === f.i) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = 2 / transform.k;
        ctx.strokeStyle = ink.surface;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 4 / transform.k, 0, 6.2832);
        ctx.lineWidth = 1.5 / transform.k;
        ctx.strokeStyle = colors[n.type];
        ctx.stroke();
      }
    });

    // labels
    if (labelsEl.checked) {
      ctx.font = (11 / Math.sqrt(transform.k)) + "px ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      nodes.forEach(function (n) {
        if (!visible(n)) return;
        var lit = !f || near.has(n.i);
        if (!lit) return;
        var big = n.deg >= 6 || (f && (n.i === f.i || near.has(n.i)));
        if (!big && transform.k < 1.4) return;
        ctx.globalAlpha = f && n.i === f.i ? 1 : 0.78;
        ctx.fillStyle = ink.text;
        ctx.fillText(n.name, n.x, n.y + radius(n) + 3 / transform.k);
      });
    }

    ctx.globalAlpha = 1;
    ctx.restore();
  }

  /* ---------- interaction ---------- */

  function pointer(ev) {
    var rect = canvas.getBoundingClientRect();
    return transform.invert([ev.clientX - rect.left, ev.clientY - rect.top]);
  }

  function pick(p) {
    var best = null, bestD = Infinity;
    nodes.forEach(function (n) {
      if (!visible(n)) return;
      var r = radius(n) + 6;
      var dx = n.x - p[0], dy = n.y - p[1], d = dx * dx + dy * dy;
      if (d < r * r && d < bestD) { best = n; bestD = d; }
    });
    return best;
  }

  function subject(ev) {
    return pick(pointer(ev.sourceEvent || ev));
  }

  function dragStart(ev) {
    if (!ev.active) sim.alphaTarget(0.2).restart();
    ev.subject.fx = ev.subject.x; ev.subject.fy = ev.subject.y;
    canvas.classList.add("dragging");
  }
  function dragMove(ev) {
    var p = pointer(ev.sourceEvent);
    ev.subject.fx = p[0]; ev.subject.fy = p[1];
  }
  function dragEnd(ev) {
    if (!ev.active) sim.alphaTarget(0);
    ev.subject.fx = null; ev.subject.fy = null;
    canvas.classList.remove("dragging");
  }

  function onMove(ev) {
    var n = pick(pointer(ev));
    setHover(n, ev);
  }

  function setHover(n, ev) {
    if (n && ev) {
      var rect = canvas.getBoundingClientRect();
      tip.textContent = n.name;
      tip.style.left = (ev.clientX - rect.left) + "px";
      tip.style.top = (ev.clientY - rect.top) + "px";
      tip.hidden = false;
    } else {
      tip.hidden = true;
    }
    if (n !== hovered) { hovered = n; draw(); }
  }

  function onClick(ev) {
    var n = pick(pointer(ev));
    select(n);
  }

  function select(n) {
    selected = n;
    if (!n) { panelBody.hidden = true; panelEmpty.hidden = false; draw(); return; }
    renderPanel(n);
    draw();
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

    var html = "";
    html += '<h4>' + esc(n.name) + "</h4>";
    html += '<div class="kind" style="color:' + colors[n.type] + '"><span class="dot"></span>' + esc(n.type) + "</div>";

    if (n.ds.length) {
      html += '<div class="tags">' + n.ds.map(function (d) {
        return '<span class="tag">' + esc(d) + "</span>";
      }).join("") + "</div>";
    } else {
      html += '<div class="tags"><span class="tag">background knowledge only</span></div>';
    }

    html += "<dl>";
    if (n.desc) html += "<dt>Definition</dt><dd class=\"body\">" + esc(n.desc) + "</dd>";
    if (n.range) html += "<dt>Typical range</dt><dd>" + esc(n.range) + "</dd>";
    if (n.rec) html += "<dt>Recommendation</dt><dd>" + esc(n.rec) + "</dd>";
    if (n.cui) html += '<dt>UMLS</dt><dd>' + esc(n.cui) + "</dd>";
    html += "<dt>Strongest relations</dt></dl>";

    html += '<ul class="neighbors">';
    top.forEach(function (a) {
      var m = nodes[a.j];
      html += "<li>" +
        '<button class="nb" data-j="' + a.j + '" data-k="' + a.k + '">' +
          '<span class="nb-top"><span class="nb-name">' + esc(m.name) + "</span>" +
          '<span class="nb-w">' + a.w.toFixed(2) + (a.w >= t ? "" : " ·") + "</span></span>" +
          '<span class="bar"><i style="width:' + Math.round(a.w * 100) + "%;background:" + colors[m.type] + '"></i></span>' +
        "</button>" +
        '<div class="nb-desc" hidden></div>' +
      "</li>";
    });
    html += "</ul>";
    html += '<p class="nb-desc" style="padding-top:10px">Weights marked &middot; fall below the current threshold ' +
            "and are not drawn in the graph.</p>";

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

  function loadDescriptions() {
    if (edgeDesc) return Promise.resolve(edgeDesc);
    return fetch("data/kg_edge_desc.json")
      .then(function (r) { return r.json(); })
      .then(function (list) { edgeDesc = list; return list; })
      .catch(function () { return []; });
  }

  /* ---------- controls ---------- */

  function buildChips() {
    var box = document.getElementById("type-filters");
    var counts = {};
    nodes.forEach(function (n) { counts[n.type] = (counts[n.type] || 0) + 1; });
    TYPES.forEach(function (t) {
      if (!counts[t]) return;
      var b = document.createElement("button");
      b.className = "chip on";
      b.style.color = colors[t];
      b.innerHTML = '<span class="dot"></span>' + t + " <span style=\"opacity:.6\">" + counts[t] + "</span>";
      b.addEventListener("click", function () {
        active[t] = !active[t];
        b.classList.toggle("on", active[t]);
        b.classList.toggle("off", !active[t]);
        if (selected && !visible(selected)) select(null);
        applyFilters(false);
        draw();
      });
      box.appendChild(b);
    });
  }

  thresholdEl.addEventListener("input", function () {
    applyFilters(false);
    if (selected) renderPanel(selected);
    draw();
  });

  labelsEl.addEventListener("change", draw);

  document.getElementById("reset").addEventListener("click", function () {
    transform = d3.zoomIdentity;
    d3.select(canvas).call(d3.zoom().transform, d3.zoomIdentity);
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
