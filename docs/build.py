#!/usr/bin/env python3
"""Build the project page's graph assets from resources/kg.

Emits, into docs/data/:
  kg.json / kg.js                    nodes + edges (js form needs no fetch)
  kg_edge_desc.json / kg_edge_desc.js  relation descriptions, loaded on demand

and injects a pre-laid-out static <svg> into docs/index.html, so the graph is
visible even where JavaScript does not run (snapshot renderers, printing,
strict sandboxes). app.js clears that markup and takes over when it does run.

Usage:  python3 docs/build.py
"""

import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "resources")
OUT = os.path.join(HERE, "data")

# static render geometry — app.js re-measures against the real container
W, H = 590.0, 620.0
PAD = 30.0
STATIC_THRESHOLD = 0.75      # must match the slider's default in index.html
LABEL_MIN_DEGREE = 6

DATASETS = {"globem": "GLOBEM", "lifesnap": "LifeSnaps",
            "pmdata": "PMData", "ifh_affect": "IFH Affect"}

# light-theme fallbacks, used only if style.css is unavailable; any CSS rule wins
COLORS = {
    "Physiological": "#cf4b5e", "Sleep": "#5765d4", "Activity": "#1f8f80",
    "Lifestyle": "#d1892a", "Mental": "#8b56bd", "Environmental": "#3d93c9",
    "Demographic": "#6d7480",
}
TYPE_ORDER = list(COLORS)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def clean(s, limit=None):
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    if limit and len(s) > limit:
        s = s[: limit - 1].rsplit(" ", 1)[0] + "…"
    return s


def datasets_of(node):
    found = []
    for key in (node.get("dataSource") or {}):
        name = DATASETS.get(key.lower())
        if name and name not in found:
            found.append(name)
    return found


def export():
    nodes_raw = json.load(open(os.path.join(SRC, "kg", "nodes.json"), encoding="utf-8"))
    edges_raw = json.load(open(os.path.join(SRC, "kg", "edges.json"), encoding="utf-8"))

    order = sorted(nodes_raw.values(),
                   key=lambda n: (n.get("type") or "", n.get("name") or ""))
    index = {n["id"]: i for i, n in enumerate(order)}

    nodes = [{
        "name": clean(n.get("name")),
        "type": clean(n.get("type")) or "Demographic",
        "desc": clean(n.get("description"), 420),
        "range": clean(n.get("range"), 300),
        "rec": clean(n.get("recommendation"), 300),
        "cui": n.get("cui") or "",
        "ds": datasets_of(n),
    } for n in order]

    edges, descs = [], []
    for e in edges_raw.values():
        a, b = index.get(e.get("entity_1_id")), index.get(e.get("entity_2_id"))
        w = e.get("weight")
        if a is None or b is None or a == b or not isinstance(w, (int, float)):
            continue
        edges.append([a, b, round(float(w), 3)])
        descs.append(clean(e.get("description"), 320))

    return nodes, edges, descs


# --------------------------------------------------------------------------
# layout (Fruchterman-Reingold, seeded for reproducible output)
# --------------------------------------------------------------------------

def radius(degree):
    return 4.5 + math.sqrt(degree) * 1.05


def layout(nodes, edges, iterations=600):
    n = len(nodes)
    rng = random.Random(20260912)

    # seed each category on its own arc so clusters start apart
    groups = {}
    for i, node in enumerate(nodes):
        groups.setdefault(node["type"], []).append(i)

    pos = [[0.0, 0.0] for _ in range(n)]
    for g, (t, members) in enumerate(sorted(groups.items())):
        base = 2 * math.pi * g / max(1, len(groups))
        for j, i in enumerate(members):
            a = base + (j / max(1, len(members))) * 0.9
            r = W * 0.3 * (0.55 + 0.45 * rng.random())
            pos[i] = [W / 2 + r * math.cos(a), H / 2 + r * math.sin(a)]

    live = [(a, b, w) for a, b, w in edges if w >= STATIC_THRESHOLD]
    k = math.sqrt(W * H / n) * 0.62
    temp = W / 8.0
    cool = temp / (iterations + 1)

    for _ in range(iterations):
        disp = [[0.0, 0.0] for _ in range(n)]

        for i in range(n):
            xi, yi = pos[i]
            for j in range(i + 1, n):
                dx, dy = xi - pos[j][0], yi - pos[j][1]
                d2 = dx * dx + dy * dy
                if d2 < 1e-6:
                    dx, dy = rng.random() - .5, rng.random() - .5
                    d2 = 1e-6
                d = math.sqrt(d2)
                f = (k * k) / d2 * d          # k^2 / d
                ux, uy = dx / d * f, dy / d * f
                disp[i][0] += ux; disp[i][1] += uy
                disp[j][0] -= ux; disp[j][1] -= uy

        for a, b, w in live:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = math.hypot(dx, dy) or 1e-6
            f = (d * d) / k * (0.35 + w * 0.65)
            ux, uy = dx / d * f, dy / d * f
            disp[a][0] -= ux; disp[a][1] -= uy
            disp[b][0] += ux; disp[b][1] += uy

        for i in range(n):
            dx, dy = disp[i]
            d = math.hypot(dx, dy) or 1e-6
            step = min(d, temp)
            pos[i][0] += dx / d * step
            pos[i][1] += dy / d * step
        temp -= cool

    return pos


def relax_overlaps(pos, radii, rounds=60):
    n = len(pos)
    for _ in range(rounds):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
                d = math.hypot(dx, dy) or 1e-6
                want = radii[i] + radii[j] + 4.0
                if d < want:
                    push = (want - d) / 2.0
                    ux, uy = dx / d * push, dy / d * push
                    pos[i][0] -= ux; pos[i][1] -= uy
                    pos[j][0] += ux; pos[j][1] += uy
                    moved = True
        if not moved:
            break


def fit(pos, radii):
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    sx = (W - 2 * PAD) / max(1e-6, maxx - minx)
    sy = (H - 2 * PAD) / max(1e-6, maxy - miny)
    s = min(sx, sy)
    ox = (W - (maxx - minx) * s) / 2 - minx * s
    oy = (H - (maxy - miny) * s) / 2 - miny * s
    for p in pos:
        p[0] = p[0] * s + ox
        p[1] = p[1] * s + oy
    for i, p in enumerate(pos):
        p[0] = max(radii[i] + 2, min(W - radii[i] - 2, p[0]))
        p[1] = max(radii[i] + 2, min(H - radii[i] - 2, p[1]))


# --------------------------------------------------------------------------
# static svg
# --------------------------------------------------------------------------

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def static_svg(nodes, edges, pos, degrees, radii):
    out = ['<g id="static-graph">', '<g class="links" stroke="#7a7f88">']
    for a, b, w in edges:
        if w < STATIC_THRESHOLD:
            continue
        op = round(0.07 + (w - 0.3) * 0.4, 3)
        out.append(
            '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="1" stroke-opacity="%s"/>'
            % (pos[a][0], pos[a][1], pos[b][0], pos[b][1], op))
    out.append("</g>")

    out.append('<g class="nodes">')
    for i, node in enumerate(nodes):
        out.append(
            '<circle class="t-%s" cx="%.1f" cy="%.1f" r="%.1f" fill="%s"><title>%s</title></circle>'
            % (node["type"].lower(), pos[i][0], pos[i][1], radii[i],
               COLORS.get(node["type"], "#6d7480"),
               esc(node["name"] + " · " + node["type"])))
    out.append("</g>")

    out.append('<g class="labels" font-family="system-ui, sans-serif" font-size="11" '
               'fill="#2b2f36" stroke="#fbfbf9" stroke-width="3" paint-order="stroke" '
               'text-anchor="middle">')
    # greedy placement: densest nodes win, overlapping labels are dropped
    placed, shown = [], 0
    for i in sorted(range(len(nodes)), key=lambda j: -degrees[j]):
        if degrees[i] < LABEL_MIN_DEGREE:
            continue
        name = nodes[i]["name"]
        w, h = len(name) * 11 * 0.52, 11 * 1.2
        x0, y0 = pos[i][0] - w / 2, pos[i][1] + radii[i] + 2
        x1, y1 = x0 + w, y0 + h
        if any(x0 < p[2] and x1 > p[0] and y0 < p[3] and y1 > p[1] for p in placed):
            continue
        placed.append((x0, y0, x1, y1))
        shown += 1
        out.append('<text x="%.1f" y="%.1f" dominant-baseline="hanging">%s</text>'
                   % (pos[i][0], y0, esc(name)))
    out.append("</g>")
    static_svg.labels_drawn = shown
    out.append("</g>")
    return "\n".join(out)


# --------------------------------------------------------------------------

def main():
    nodes, edges, descs = export()

    degrees = [0] * len(nodes)
    for a, b, w in edges:
        if w >= STATIC_THRESHOLD:
            degrees[a] += 1
            degrees[b] += 1
    radii = [radius(d) for d in degrees]

    # Prefer the baked coordinates in docs/data/layout.json: G6's d3-force with a
    # per-category anchor keeps the categories apart, which the crude fallback
    # below (only used if that file is missing) does not.
    # To refresh it: serve docs/, open tools/bake-layout.html, download the
    # result over data/layout.json, then rerun this script.
    layout_file = os.path.join(OUT, "layout.json")
    pos = None
    if os.path.exists(layout_file):
        saved = json.load(open(layout_file, encoding="utf-8"))
        if len(saved.get("pos", [])) == len(nodes):
            # fit the saved coordinates into the static box rather than scaling by
            # the canvas they were produced on, so the picture always fills the frame
            raw = saved["pos"]
            xs = [p[0] for p in raw]
            ys = [p[1] for p in raw]
            sx = (W - 2 * PAD) / max(1e-6, max(xs) - min(xs))
            sy = (H - 2 * PAD) / max(1e-6, max(ys) - min(ys))
            sc = min(sx, sy)
            ox = (W - (max(xs) - min(xs)) * sc) / 2 - min(xs) * sc
            oy = (H - (max(ys) - min(ys)) * sc) / 2 - min(ys) * sc
            pos = [[p[0] * sc + ox, p[1] * sc + oy] for p in raw]
            print("layout: using docs/data/layout.json")
    if pos is None:
        print("layout: layout.json missing or stale — falling back to the built-in layout")
        pos = layout(nodes, edges)
        relax_overlaps(pos, radii)
        fit(pos, radii)

    graph = {
        "nodes": nodes,
        "edges": edges,
        # starting coordinates, in the same 590x620 space as the static svg
        "pos": [[round(p[0], 1), round(p[1], 1)] for p in pos],
        "box": [W, H],
    }

    os.makedirs(OUT, exist_ok=True)
    compact = {"separators": (",", ":"), "ensure_ascii": False}
    json.dump(graph, open(os.path.join(OUT, "kg.json"), "w", encoding="utf-8"), **compact)
    json.dump(descs, open(os.path.join(OUT, "kg_edge_desc.json"), "w", encoding="utf-8"), **compact)
    with open(os.path.join(OUT, "kg.js"), "w", encoding="utf-8") as fh:
        fh.write("window.__WAG_KG__=" + json.dumps(graph, **compact) + ";\n")
    with open(os.path.join(OUT, "kg_edge_desc.js"), "w", encoding="utf-8") as fh:
        fh.write("window.__WAG_KG_EDGES__=" + json.dumps(descs, **compact) + ";\n")

    page = os.path.join(HERE, "index.html")
    html = open(page, encoding="utf-8").read()
    svg = static_svg(nodes, edges, pos, degrees, radii)
    block = ('<svg id="graph" role="img" viewBox="0 0 %g %g" '
             'aria-label="Wearable knowledge graph: %d health metrics linked by weighted relations">\n'
             '<!-- generated by docs/build.py; app.js replaces this with the interactive graph -->\n'
             "%s\n</svg>") % (W, H, len(nodes), svg)

    new, count = re.subn(r'<svg id="graph".*?</svg>', lambda _m: block, html,
                         count=1, flags=re.S)
    if not count:
        sys.exit("could not find <svg id=\"graph\"> in index.html")
    open(page, "w", encoding="utf-8", newline="\n").write(new)

    shown = sum(1 for _, _, w in edges if w >= STATIC_THRESHOLD)
    print("nodes %d, edges %d (%d drawn at w>=%.2f), labels %d"
          % (len(nodes), len(edges), shown, STATIC_THRESHOLD,
             getattr(static_svg, "labels_drawn", 0)))
    for name in ("kg.json", "kg.js", "kg_edge_desc.json", "kg_edge_desc.js"):
        print("  %-20s %7.1f KB" % (name, os.path.getsize(os.path.join(OUT, name)) / 1024))
    print("  index.html static svg  %7.1f KB" % (len(svg) / 1024))


if __name__ == "__main__":
    main()
