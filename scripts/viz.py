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
      <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
        <button class="btn" onclick="resetView()">Reset view</button>
        <button class="btn" onclick="toggleSurface()">Toggle surface</button>
        <button class="btn" onclick="toggleLabels()">Toggle labels</button>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>Predicted probability</h2>
        <div class="ramp"></div>
        <div class="ramp-lab"><span>0.0 not scored</span><span>1.0</span></div>
        <p class="note" style="margin-top:10px">
          White means the residue is not at the interface and was never scored &mdash;
          not that it was scored and found unimportant.
        </p>
      </div>

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
      h.label,
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
  // everything: cartoon on the probability gradient
  viewer.setStyle({{}}, {{ cartoon: {{ colorfunc: a => ramp(a.b), thickness: 0.4 }} }});

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
        cartoon: {{ colorfunc: a => ramp(a.b) }},
        stick: {{ radius: 0.20, colorfunc: a => ramp(a.b) }}
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
  viewer.zoomTo();
  viewer.render();
}}
if (document.readyState === "loading") {{
  document.addEventListener("DOMContentLoaded", boot);
}} else {{
  boot();
}}

function resetView() {{ viewer.zoomTo(); viewer.render(); }}

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
      {{ opacity: 0.72, colorfunc: a => ramp(a.b) }});
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
    ap.add_argument("--top", type=int, default=10, help="rows in the side panel")
    ap.add_argument("--title", default=None)
    ap.add_argument("--cv-claim", default="0.5076 ± 0.0531")
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

    title = args.title or f"{pdb_path.stem.upper()} — predicted interface hot spots"
    subtitle = (f"{len(preds)} scored interface residues · {n_atoms} atoms rendered "
                f"(chains {', '.join(sorted(keep))}) · colour = predicted probability")

    html = HTML.format(
        title=title, subtitle=subtitle, chain_panels=chain_panels,
        highlight_card=highlight_card,
        pdb_json=json.dumps(pdb_text), highlights_json=json.dumps(highlights),
        cv_claim=args.cv_claim,
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
