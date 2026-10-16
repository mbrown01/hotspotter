"""Render hot-spot predictions as a self-contained interactive 3D HTML page.

HOW IT WORKS — "B-factor hijacking"
    A PDB file has a per-atom B-factor column (columns 61-66) that normally holds
    crystallographic thermal motion. Every structure viewer already knows how to colour by
    it. So instead of inventing a colouring channel, we overwrite that column with the
    model's predicted probability and let the viewer do the work.

    The original B-factors ARE one of the model's 26 input features, so the written file is
    no longer a faithful copy of the deposited structure. It is a visualization artifact and
    should never be fed back into the pipeline; the header of the written PDB block says so.

WHAT THE COLOURS MEAN
    White through red, pinned to a FIXED 0.0-1.0 scale rather than auto-scaled to whatever
    this particular structure happens to contain. Auto-scaling would make a complex whose
    best residue scores 0.2 look just as red as one scoring 0.9, which is exactly the
    misreading to avoid when comparing two structures side by side.

    Residues absent from the predictions CSV (not at the interface) get 0.00 and render
    white. White therefore means "not scored", not "scored and found unimportant".

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\viz.py --pdb 8F6D \\
        --preds predictions\\8f6d_predictions.csv --out arl15_hotspots.html \\
        --highlight B:95 --title "ARL15-CNNM2 predicted hot spots"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def resolve_pdb(spec: str) -> Path:
    """Accept a path, or a 4-character PDB id to fetch (and cache)."""
    p = Path(spec)
    if p.exists():
        return p
    if len(spec) == 4 and spec.isalnum():
        from hotspotter.io import fetch_pdb
        return fetch_pdb(spec)
    raise FileNotFoundError(f"{spec!r} is neither an existing file nor a 4-character PDB id")


def hijack_bfactors(pdb_path: Path, preds: dict, keep_chains: set[str] | None):
    """Rewrite each ATOM record's B-factor column with that residue's predicted probability.

    Returns (pdb_text, n_atoms_written, n_atoms_scored, residues_seen).
    """
    out, n_atoms, n_scored = [], 0, 0
    seen: dict[tuple, str] = {}
    for line in pdb_path.read_text().splitlines():
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue                                    # drop waters/ligands/headers
        chain = line[21]
        if keep_chains and chain not in keep_chains:
            continue
        if rec == "HETATM":
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        resname = line[17:20].strip()
        prob = preds.get((chain, resseq), 0.0)
        if (chain, resseq) in preds:
            n_scored += 1
        seen[(chain, resseq)] = resname
        # columns 61-66, right-justified, 2 decimals -- the PDB spec's tempFactor field
        out.append(f"{line[:60]}{prob:6.2f}{line[66:]}")
        n_atoms += 1
    out.append("END")
    return "\n".join(out), n_atoms, n_scored, seen


HTML = """<title>{title}</title>
<style>
  :root {{
    --bg: #fbfbfa; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e2dd; --card: #ffffff;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #17171a; --fg: #ececea; --muted: #9a9a94; --line: #2e2e33; --card: #1f1f23;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #17171a; --fg: #ececea; --muted: #9a9a94; --line: #2e2e33; --card: #1f1f23;
  }}
  body {{ background: var(--bg); color: var(--fg); margin: 0;
         font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding-block: 28px; padding-left: 20px;
           padding-right: 20px; }}
  h1 {{ font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); margin: 0 0 20px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 300px; gap: 18px; align-items: start; }}
  @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  #viewer {{ position: relative; width: 100%; height: 560px; border: 1px solid var(--line);
             border-radius: 10px; overflow: hidden; background: var(--card); }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
           padding: 14px 16px; }}
  .card h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
              color: var(--muted); margin: 0 0 10px; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
  td {{ padding: 3px 0; font-size: 13px; }}
  td.p {{ text-align: right; color: var(--muted); }}
  .swatch {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px;
             margin-right: 7px; vertical-align: middle; border: 1px solid rgba(0,0,0,.15); }}
  .ramp {{ height: 10px; border-radius: 5px; margin: 6px 0 4px;
           background: linear-gradient(90deg, #f5f5f5, #ffd9b0, #ff8a5c, #d90416); }}
  .ramp-lab {{ display: flex; justify-content: space-between; color: var(--muted);
               font-size: 11px; }}
  .hl {{ color: #0a7d2e; font-weight: 600; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) .hl {{ color: #35d06a; }}
  }}
  :root[data-theme="dark"] .hl {{ color: #35d06a; }}
  .note {{ color: var(--muted); font-size: 12px; margin-top: 14px; }}
  .btn {{ font: inherit; font-size: 12px; padding: 5px 10px; border: 1px solid var(--line);
          background: var(--card); color: var(--fg); border-radius: 6px; cursor: pointer; }}
  .btn:hover {{ border-color: var(--muted); }}
</style>

<div class="wrap">
  <h1>{title}</h1>
  <p class="sub">{subtitle}</p>

  <div class="grid">
    <div>
      <div id="viewer"></div>
      <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
        <button class="btn" onclick="focusInterface()">Focus interface</button>
        <button class="btn" onclick="resetView()">Whole complex</button>
        <button class="btn" onclick="toggleSurface()">Toggle surface</button>
        <button class="btn" onclick="toggleLabels()">Toggle labels</button>
        {channel_button}
      </div>
      <p class="note" style="margin-top:8px">
        drag = rotate &middot; <b>right-drag or Ctrl+drag = pan</b> &middot;
        scroll = zoom &middot; Shift+drag = roll
      </p>
      {channel_caption}
    </div>

    <div>
      <div class="card">
        <h2 data-ramp-title>Predicted probability</h2>
        <div class="ramp"></div>
        <div class="ramp-lab"><span>0.0</span><span>1.0</span></div>
        {chain_legend}
      </div>

      {contrast_card}

      {chain_panels}

      {highlight_card}

      <div class="card" style="margin-top:14px">
        <h2>Caveat</h2>
        <p class="note" style="margin:0">
          Scores are ranking signals, not calibrated probabilities &mdash; the model was
          trained with positive-class weighting. Performance claim is the cross-validated
          PR&#8209;AUC of {cv_claim} on held-out complexes, not anything measured on this
          structure.
        </p>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
<script>
const PDB = {pdb_json};
const HIGHLIGHTS = {highlights_json};
// Burial channel: dSASA per residue, scaled 0-1 against the most-buried residue in THIS
// interface. That is deliberately how a person eyeballing burial would rank them, so the
// two colourings are a fair comparison rather than a rigged one.
const BURIAL = {burial_json};
const HAS_BURIAL = Object.keys(BURIAL).length > 0;
// Base colour per chain, so the two proteins are told apart at a glance. Only residues
// that were actually SCORED take the red gradient on top; everything else keeps its chain
// colour. That also removes an ambiguity the earlier version had, where an unscored
// residue and a scored-but-cold residue were both white.
const CHAIN_COLORS = {chain_colors_json};
const SCORED = {scored_json};
let channel = "model";

function scoreOf(a) {{
  if (channel === "model") return a.b;             // B-factor column holds the model score
  const v = BURIAL[a.chain + "_" + a.resi];
  return v === undefined ? 0 : v;
}}

const SCORED_SET = new Set();
Object.keys(SCORED).forEach(ch => SCORED[ch].forEach(r => SCORED_SET.add(ch + "_" + r)));
const HAS_CHAIN_COLORS = Object.keys(CHAIN_COLORS).length > 0;

function colorFor(a) {{
  // unscored residue in a chain-coloured figure keeps its chain colour; scored residues
  // take the heat ramp. Used by the surface so it matches the cartoon exactly.
  if (HAS_CHAIN_COLORS && !SCORED_SET.has(a.chain + "_" + a.resi)) {{
    return CHAIN_COLORS[a.chain] || "#cccccc";
  }}
  return ramp(scoreOf(a));
}}

function ramp(t) {{
  t = Math.max(0, Math.min(1, t));
  // white -> amber -> red, matching the legend gradient
  const stops = [[245,245,245],[255,217,176],[255,138,92],[217,4,22]];
  const x = t * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i+1][k] - v) * f));
  return "#" + c.map(v => v.toString(16).padStart(2, "0")).join("");
}}

let viewer, surfaceOn = false, labelsOn = true, surf = null;

function addLabels() {{
  HIGHLIGHTS.forEach(h => {{
    // 3Dmol signature is addLabel(text, options, SELECTION). Putting the selection inside
    // options.position silently creates nothing -- options.position wants {{x,y,z}}.
    viewer.addLabel(
      channel === "model" ? h.label : (h.label_burial || h.label),
      {{
        backgroundColor: h.color || "#333333", backgroundOpacity: 0.9,
        fontColor: "white", fontSize: 12, borderThickness: 0,
        inFront: true, alignment: "center", screenOffset: {{ x: 0, y: -6 }}
      }},
      {{ chain: h.chain, resi: h.resi }}
    );
  }});
}}

function draw() {{
  // 1. base layer: each chain in its own muted colour, thin cartoon
  Object.keys(CHAIN_COLORS).forEach(ch => {{
    viewer.setStyle({{ chain: ch }},
      {{ cartoon: {{ color: CHAIN_COLORS[ch], thickness: 0.35 }} }});
  }});
  if (Object.keys(CHAIN_COLORS).length === 0) {{
    viewer.setStyle({{}}, {{ cartoon: {{ colorfunc: a => ramp(scoreOf(a)), thickness: 0.4 }} }});
  }}

  // 2. scored interface residues: the gradient, on a thicker ribbon so the interface
  //    reads as a band of colour rather than isolated flecks
  Object.keys(SCORED).forEach(ch => {{
    viewer.setStyle({{ chain: ch, resi: SCORED[ch] }},
      {{ cartoon: {{ colorfunc: a => ramp(scoreOf(a)), thickness: 0.75 }} }});
  }});

  HIGHLIGHTS.forEach(h => {{
    const sel = {{ chain: h.chain, resi: h.resi }};
    if (h.color) {{
      // focal residue: solid colour on BOTH cartoon and sticks, so it cannot be confused
      // with a gradient value
      viewer.setStyle(sel, {{
        cartoon: {{ color: h.color }},
        stick: {{ radius: 0.26, color: h.color }}
      }});
    }} else {{
      // supporting residue: sticks, but keep the probability gradient
      viewer.setStyle(sel, {{
        cartoon: {{ colorfunc: a => ramp(scoreOf(a)) }},
        stick: {{ radius: 0.20, colorfunc: a => ramp(scoreOf(a)) }}
      }});
    }}
  }});
  if (labelsOn) addLabels();
}}

function boot() {{
  if (typeof $3Dmol === "undefined") {{       // CDN blocked or offline
    document.getElementById("viewer").innerHTML =
      '<div style="padding:24px;color:#b00">Could not load 3Dmol.js from the CDN. ' +
      'This page needs an internet connection to render the structure.</div>';
    return;
  }}
  viewer = $3Dmol.createViewer(document.getElementById("viewer"),
                               {{ backgroundColor: "0xffffff" }});
  viewer.addModel(PDB, "pdb");
  draw();
  // open on the interface: it is the subject of the figure, and on a large complex it is
  // often a small patch on the rim that the whole-assembly view renders unreadably small
  if (Object.keys(SCORED).length) {{ focusInterface(); }} else {{ viewer.zoomTo(); }}
  viewer.render();
}}
if (document.readyState === "loading") {{
  document.addEventListener("DOMContentLoaded", boot);
}} else {{
  boot();
}}

function resetView() {{ viewer.zoomTo(); viewer.render(); }}

function interfaceSel() {{
  const parts = Object.keys(SCORED).map(ch => ({{ chain: ch, resi: SCORED[ch] }}));
  if (parts.length === 0) return null;
  return parts.length === 1 ? parts[0] : {{ or: parts }};
}}

function focusInterface() {{
  // An interface often sits on the rim of a complex, where zooming on the whole assembly
  // leaves it small and off-centre. This recentres on the scored residues only.
  const sel = interfaceSel();
  if (!sel) return;
  try {{
    viewer.zoomTo(sel);
    viewer.zoom(0.85);        // back off slightly so the surrounding fold stays visible
  }} catch (e) {{
    // if the compound selection is rejected, fall back to the largest single chain rather
    // than leaving the button doing nothing
    const chains = Object.keys(SCORED);
    if (chains.length) {{
      const big = chains.reduce((a, b) => SCORED[a].length >= SCORED[b].length ? a : b);
      viewer.zoomTo({{ chain: big, resi: SCORED[big] }});
      viewer.zoom(0.8);
    }}
  }}
  viewer.render();
}}

function toggleChannel() {{
  // The whole argument of this figure: flip the colouring between what the model predicts
  // and what buried surface area alone would say, WITHOUT moving the camera. Residues that
  // change colour are the ones burial gets wrong.
  channel = (channel === "model") ? "burial" : "model";
  const isModel = channel === "model";
  document.getElementById("chanbtn").textContent =
    isModel ? "Colour: model score \\u2192 show burial" : "Colour: burial \\u2192 show model score";
  document.getElementById("chanlab").textContent =
    isModel ? "Coloured by MODEL SCORE" : "Coloured by BURIED SURFACE AREA (the naive baseline)";
  document.getElementById("chanlab").style.color = isModel ? "#0a7d2e" : "#b45309";
  const ramps = document.querySelectorAll("[data-ramp-title]");
  ramps.forEach(e => {{ e.textContent = isModel ? "Predicted probability" : "Buried area (dSASA)"; }});
  if (surfaceOn) {{ viewer.removeAllSurfaces(); }}
  draw();
  if (surfaceOn) {{
    surf = viewer.addSurface($3Dmol.SurfaceType.VDW,
      {{ opacity: 0.72, colorfunc: colorFor }});
  }}
  viewer.render();
}}

function toggleLabels() {{
  labelsOn = !labelsOn;
  viewer.removeAllLabels();
  if (labelsOn) addLabels();
  viewer.render();
}}

function toggleSurface() {{
  surfaceOn = !surfaceOn;
  if (surfaceOn) {{
    surf = viewer.addSurface($3Dmol.SurfaceType.VDW,
      {{ opacity: 0.72, colorfunc: colorFor }});
  }} else {{
    viewer.removeAllSurfaces();
  }}
  viewer.render();
}}
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, help="PDB file path, or a 4-character PDB id")
    ap.add_argument("--preds", required=True, type=Path,
                    help="predictions CSV with columns chain, res_id, pred_prob")
    ap.add_argument("--out", required=True, type=Path, help="output .html")
    ap.add_argument("--highlight", nargs="*", default=[],
                    help="residues to mark as sticks + label, as CHAIN:RESID (e.g. B:95)")
    ap.add_argument("--color", nargs="*", default=[],
                    help="force a solid colour on a residue, as CHAIN:RESID:COLOUR "
                         "(e.g. B:95:blue B:96:green). Overrides the probability gradient "
                         "for that residue only, so a focal residue is unmistakable.")
    ap.add_argument("--chain-label", nargs="*", default=[],
                    help="rename a chain in the side panel, as CHAIN=NAME (e.g. A=CNNM2)")
    ap.add_argument("--chains", default=None,
                    help="comma-separated chains to render; default = chains in the CSV")
    ap.add_argument("--chain-color", nargs="*", default=None,
                    help="base colour per chain, as CHAIN=COLOUR (e.g. A=#8aa9c9 B=#c9a68a). "
                         "Only SCORED residues take the heat ramp, so the two proteins stay "
                         "distinguishable. Pass 'none' to colour everything by score.")
    ap.add_argument("--compare", type=Path, default=None,
                    help="the *_full.csv from infer_xgb.py; adds a colour toggle between "
                         "the model score and buried area, plus a rank-disagreement table")
    ap.add_argument("--top", type=int, default=10, help="rows in the side panel")
    ap.add_argument("--title", default=None)
    ap.add_argument("--cv-claim", default=None,
                    help="cross-validated PR-AUC to print; pass the number "
                         "from the current evaluate.py run. A stale default "
                         "here ends up in the published HTML.")
    args = ap.parse_args()

    pdb_path = resolve_pdb(args.pdb)
    df = pd.read_csv(args.preds)
    for col in ("chain", "res_id", "pred_prob"):
        if col not in df.columns:
            print(f"ERROR: {args.preds} has no '{col}' column", file=sys.stderr)
            return 2

    preds = {(str(r.chain), int(r.res_id)): float(r.pred_prob) for r in df.itertuples()}
    keep = (set(c.strip() for c in args.chains.split(",")) if args.chains
            else set(df["chain"].astype(str)))

    pdb_text, n_atoms, n_scored, seen = hijack_bfactors(pdb_path, preds, keep)

    # resolve highlights to labels like "ARG95". --color entries are highlights that also
    # carry a forced solid colour; --highlight entries keep the probability gradient.
    forced = {}
    for c in args.color:
        ch, rid, col = c.split(":")
        forced[(ch, int(rid))] = col
    wanted = list(args.highlight) + [f"{ch}:{rid}" for (ch, rid) in forced]

    highlights = []
    seen_hl = set()
    for h in wanted:
        ch, rid = h.split(":")
        rid = int(rid)
        if (ch, rid) in seen_hl:
            continue
        seen_hl.add((ch, rid))
        resname = seen.get((ch, rid))
        if resname is None:
            print(f"  WARNING: highlight {h} not found in the rendered chains", file=sys.stderr)
            continue
        p = preds.get((ch, rid))
        highlights.append({
            "chain": ch, "resi": rid,
            "label": f"{resname}{rid}" + (f"  p={p:.2f}" if p is not None else "  (not scored)"),
            "prob": p, "resname": resname, "color": forced.get((ch, rid)),
        })

    names = {}
    for spec in args.chain_label:
        k, v = spec.split("=", 1)
        names[k.strip()] = v.strip()

    chain_panels = []
    for ch in sorted(df["chain"].astype(str).unique()):
        sub = (df[df["chain"].astype(str) == ch]
               .sort_values("pred_prob", ascending=False).head(args.top))
        rows = "".join(
            f'<tr><td><span class="swatch" '
            f'style="background:{_ramp_py(float(r.pred_prob))}"></span>'
            f'{getattr(r, "res_name", "")}{int(r.res_id)}</td>'
            f'<td class="p">{float(r.pred_prob):.3f}</td></tr>'
            for r in sub.itertuples()
        )
        heading = f"Top {names[ch]} (chain {ch})" if ch in names else f"Top chain {ch}"
        chain_panels.append(
            f'<div class="card" style="margin-top:14px"><h2>{heading}</h2>'
            f'<table>{rows}</table></div>'
        )
    chain_panels = "".join(chain_panels)

    if highlights:
        def _row(h):
            dot = (f'<span class="swatch" style="background:{h["color"]}"></span>'
                   if h["color"] else
                   f'<span class="swatch" style="background:'
                   f'{_ramp_py(h["prob"] or 0.0)}"></span>')
            val = f'{h["prob"]:.3f}' if h["prob"] is not None else "not scored"
            return (f'<tr><td class="hl">{dot}{h["chain"]}/{h["resname"]}{h["resi"]}</td>'
                    f'<td class="p">{val}</td></tr>')
        items = "".join(_row(h) for h in highlights)
        highlight_card = (
            '<div class="card" style="margin-top:14px">'
            '<h2>Highlighted</h2><table>' + items + "</table>"
            '<p class="note" style="margin:8px 0 0">Shown as sticks in the viewer. '
            'Colour-overridden residues use their own solid colour; the rest keep the '
            'probability gradient.</p>'
            "</div>"
        )
    else:
        highlight_card = ""

    # Chain base colours. Defaults are two desaturated hues chosen to sit well away from the
    # white->amber->red heat ramp, so a chain colour is never mistaken for a score.
    DEFAULT_CHAIN_COLORS = ["#7f9dc0", "#c0a07f", "#8fb08a", "#b08aa8"]
    chain_colors: dict[str, str] = {}
    if args.chain_color == ["none"]:
        pass
    elif args.chain_color:
        for spec in args.chain_color:
            k, v = spec.split("=", 1)
            chain_colors[k.strip()] = v.strip()
    else:
        for i, ch in enumerate(sorted(keep)):
            chain_colors[ch] = DEFAULT_CHAIN_COLORS[i % len(DEFAULT_CHAIN_COLORS)]

    scored: dict[str, list[int]] = {}
    for (ch, rid) in preds:
        scored.setdefault(ch, []).append(int(rid))

    if chain_colors:
        swatches = "".join(
            f'<div style="margin-top:4px"><span class="swatch" style="background:{col}">'
            f'</span>{names.get(ch, "chain " + ch)}</div>'
            for ch, col in sorted(chain_colors.items())
        )
        chain_legend = (
            '<p class="note" style="margin-top:12px">Chains are coloured separately; only '
            'the scored interface residues take the heat ramp above, on a thicker ribbon.'
            '</p>' + swatches
        )
    else:
        chain_legend = ('<p class="note" style="margin-top:10px">White means the residue is '
                        'not at the interface and was never scored &mdash; not that it was '
                        'scored and found unimportant.</p>')

    burial, contrast_card, channel_button, channel_caption = {}, "", "", ""
    if args.compare:
        burial, contrast_card = build_comparison(args.compare, keep, highlights, args.top)
        if burial:
            channel_button = ('<button class="btn" id="chanbtn" onclick="toggleChannel()">'
                              'Colour: model score &rarr; show burial</button>')
            channel_caption = ('<p class="note" id="chanlab" style="color:#0a7d2e;'
                               'font-weight:600;margin-top:8px">'
                               'Coloured by MODEL SCORE</p>')

    title = args.title or f"{pdb_path.stem.upper()} — predicted interface hot spots"
    subtitle = (f"{len(preds)} scored interface residues · {n_atoms} atoms rendered "
                f"(chains {', '.join(sorted(keep))}) · colour = predicted probability")

    html = HTML.format(
        title=title, subtitle=subtitle, chain_panels=chain_panels,
        highlight_card=highlight_card, contrast_card=contrast_card,
        channel_button=channel_button, channel_caption=channel_caption,
        chain_legend=chain_legend,
        pdb_json=json.dumps(pdb_text), highlights_json=json.dumps(highlights),
        burial_json=json.dumps(burial),
        # No stale default: an unspecified claim names the source instead of a number.
        cv_claim=(args.cv_claim or "the figure reported by scripts/evaluate.py"),
        chain_colors_json=json.dumps(chain_colors), scored_json=json.dumps(scored),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    print(f"  structure     : {pdb_path}")
    print(f"  predictions   : {args.preds}  ({len(preds)} residues)")
    print(f"  chains shown  : {', '.join(sorted(keep))}")
    print(f"  atoms written : {n_atoms}  ({n_scored} carrying a predicted score)")
    if highlights:
        for h in highlights:
            p = f"{h['prob']:.3f}" if h["prob"] is not None else "not scored"
            col = f"  [{h['color']}]" if h["color"] else ""
            print(f"  highlighted   : {h['chain']}/{h['resname']}{h['resi']}  p={p}{col}")
    print(f"  wrote         : {args.out}  ({args.out.stat().st_size/1024:.0f} KB)")
    return 0


def build_comparison(full_csv: Path, keep: set[str], highlights: list, top: int):
    """Burial channel + the model-vs-burial rank table.

    Takes the ``*_full.csv`` written by infer_xgb.py, which carries dsasa and both rankings
    alongside the probability. Returns (burial map for the JS colouring, HTML card).

    The card is sorted by how far the two rankings DISAGREE, largest first, because the
    agreements are not the point -- if burial and the model ordered residues the same way
    there would be no reason to build a model.
    """
    df = pd.read_csv(full_csv)
    need = {"chain", "resseq", "resname", "dsasa", "model_rank", "naive_rank", "pred_prob"}
    missing = need - set(df.columns)
    if missing:
        print(f"  WARNING: --compare file lacks {sorted(missing)}; skipping the "
              f"burial comparison", file=sys.stderr)
        return {}, ""

    df = df[df["chain"].astype(str).isin(keep)].copy()
    if df.empty:
        return {}, ""

    # scale burial against the most-buried residue at THIS interface
    top_dsasa = float(df["dsasa"].max()) or 1.0
    burial = {f'{r.chain}_{int(r.resseq)}': round(float(r.dsasa) / top_dsasa, 4)
              for r in df.itertuples()}

    # give each highlighted residue a burial-mode label so the callout does not keep
    # showing a model probability while the structure is coloured by area
    by_key = {(str(r.chain), int(r.resseq)): r for r in df.itertuples()}
    for h in highlights:
        r = by_key.get((h["chain"], h["resi"]))
        if r is not None:
            h["label_burial"] = (f'{h["resname"]}{h["resi"]}  '
                                 f'{float(r.dsasa):.0f} A^2  (burial #{int(r.naive_rank)})')
            h["model_rank"] = int(r.model_rank)
            h["naive_rank"] = int(r.naive_rank)

    df["moved"] = df["naive_rank"] - df["model_rank"]      # positive = model promoted it
    n = len(df)
    ordered = df.reindex(df["moved"].abs().sort_values(ascending=False).index).head(top)

    rows = []
    for r in ordered.itertuples():
        mv = int(r.moved)
        arrow = "&uarr;" if mv > 0 else ("&darr;" if mv < 0 else "&ndash;")
        col = "#0a7d2e" if mv > 0 else ("#b45309" if mv < 0 else "var(--muted)")
        rows.append(
            f'<tr><td><span class="swatch" style="background:'
            f'{_ramp_py(float(r.pred_prob))}"></span>{r.chain}/{r.resname}{int(r.resseq)}</td>'
            f'<td class="p">{int(r.model_rank)}</td>'
            f'<td class="p">{int(r.naive_rank)}</td>'
            f'<td class="p" style="color:{col};font-weight:600">{arrow}{abs(mv)}</td></tr>'
        )

    card = (
        '<div class="card" style="margin-top:14px">'
        '<h2>Model rank vs burial rank</h2>'
        '<table><tr><td class="p" style="color:var(--muted);font-size:11px">residue</td>'
        '<td class="p" style="color:var(--muted);font-size:11px">model</td>'
        '<td class="p" style="color:var(--muted);font-size:11px">burial</td>'
        '<td class="p" style="color:var(--muted);font-size:11px">moved</td></tr>'
        + "".join(rows) + '</table>'
        f'<p class="note" style="margin:8px 0 0">Biggest disagreements first, of {n} '
        'interface residues. Green = the model promoted a residue burial underrated; '
        'amber = the model demoted a residue burial overrated. Use the colour toggle to '
        'see the same structure scored both ways.</p>'
        '</div>'
    )
    return burial, card


def _ramp_py(t: float) -> str:
    """Same white->red ramp as the JS, for the side-panel swatches."""
    t = max(0.0, min(1.0, t))
    stops = [(245, 245, 245), (255, 217, 176), (255, 138, 92), (217, 4, 22)]
    x = t * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    c = [round(stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f) for k in range(3)]
    return "#" + "".join(f"{v:02x}" for v in c)


if __name__ == "__main__":
    raise SystemExit(main())
