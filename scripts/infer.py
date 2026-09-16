"""Predict hot spots on a new complex with the trained Wide & Deep GAT ensemble.

WHAT THIS IS FOR
    Everything up to now measured the model on SKEMPI, where the answers are known. This
    runs it on a structure it has never seen and has no labels for -- the actual use case.

    Predictions are averaged over the checkpoints given (one per training seed). The
    ensemble is not decoration: across cross-validation the seed-to-seed spread of the mean
    score was +/-0.0068, so any single model's ranking of one residue carries real noise.
    Averaging five, and reporting the per-seed standard deviation alongside, turns a point
    prediction into one with an error bar -- which is the difference between "the model says
    Arg95" and "the model says Arg95, and here is how stable that is".

HOW TO READ THE OUTPUT
    ``pred_prob`` is the ensemble-mean probability that mutating this residue to alanine
    costs >= 2 kcal/mol of binding energy. It is NOT a calibrated probability: the model was
    trained with positive-class weighting to handle a 23% positive rate, which deliberately
    inflates scores. Use the RANKING, not the absolute value.

    The honest performance claim for these numbers is the cross-validated PR-AUC of
    0.5076 +/- 0.0531 on held-out SKEMPI complexes -- not anything measured on this
    structure, which has no ground-truth labels.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\infer.py --pdb 8F6D --chains A,B
    .\\.venv\\Scripts\\python.exe scripts\\infer.py --pdb 8F6D --chains A,B --highlight B:95 B:96
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotspotter.ml.graph_dataset import complex_to_data  # noqa: E402
from hotspotter.pipeline import analyze_complex  # noqa: E402

LABEL_RE = re.compile(r"^(?P<chain>.+)/(?P<resname>[A-Z]{3})(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)$")


def parse_chains(spec: str):
    left, right = spec.split(",", 1)
    return tuple(left.strip()), tuple(right.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, help="PDB id or path to a local .pdb/.cif")
    ap.add_argument("--chains", required=True, help="e.g. 'A,B' or 'AB,CD'")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="default: data/model_full_s*.pt")
    ap.add_argument("--is-predicted", action="store_true",
                    help="structure is an AlphaFold/ColabFold model (B-factor column = pLDDT)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: predictions/<pdb>_predictions.csv")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--interface-cutoff", type=float, default=None,
                    help="override the heavy-atom interface cutoff (A). The 5.0 default is "
                         "a convention, not a measurement; use this for sensitivity checks "
                         "on residues sitting near the boundary.")
    ap.add_argument("--highlight", nargs="*", default=[],
                    help="residues to report ranks for, as CHAIN:RESSEQ (e.g. B:95 B:96)")
    args = ap.parse_args()

    import torch

    from hotspotter.ml.gnn_model import GNNConfig, HotSpotGAT

    ckpts = ([Path(c) for c in args.checkpoints] if args.checkpoints
             else sorted((REPO_ROOT / "data").glob("model_full_s*.pt")))
    if not ckpts:
        print("ERROR: no checkpoints. Train with --train-full first.", file=sys.stderr)
        return 2

    side_a, side_b = parse_chains(args.chains)
    tag = Path(args.pdb).stem.lower()
    out = args.out or (REPO_ROOT / "predictions" / f"{tag}_predictions.csv")

    print("=" * 78)
    print(f"  HOT-SPOT INFERENCE  |  {args.pdb}  |  {side_a} vs {side_b}")
    print("=" * 78)

    # ---- Phase-1 pipeline on the new structure ------------------------------------------
    analysis = analyze_complex(args.pdb, chains=(side_a, side_b),
                               is_predicted=args.is_predicted,
                               interface_cutoff=args.interface_cutoff)
    data = complex_to_data(analysis, pd.DataFrame(
        columns=["pdb_id", "chain", "resseq", "icode", "wt", "mut", "ddg", "mutation"]))
    from hotspotter.constants import Cutoffs as _C
    print(f"  interface cutoff   : {args.interface_cutoff or _C.INTERFACE_HEAVY_ATOM} A")
    print(f"  interface residues : {data.num_nodes}")
    print(f"  edges              : {data.edge_index.size(1)} directed "
          f"({int(data.edge_attr[:, -1].sum())} cross-interface)")
    print(f"  contacts detected  : {len(analysis.contacts)}")
    print(f"  SASA backend       : {analysis.sasa_backend}")

    # ---- ensemble prediction --------------------------------------------------------------
    per_model = []
    for c in ckpts:
        blob = torch.load(c, weights_only=False)
        cfg = GNNConfig(**blob["config"])
        m = HotSpotGAT(cfg)
        m.load_state_dict(blob["state_dict"])
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(data.x, data.edge_index, data.edge_attr)).numpy()
        per_model.append(p)
    P = np.vstack(per_model)
    print(f"  ensemble           : {len(ckpts)} checkpoints "
          f"({', '.join(c.stem.split('_')[-1] for c in ckpts)})")

    rows = []
    for i, lab in enumerate(data.residue_labels):
        m = LABEL_RE.match(lab)
        rows.append({
            "chain": m.group("chain"),
            "res_id": int(m.group("resseq")),
            "res_name": m.group("resname"),
            "pred_prob": float(P[:, i].mean()),
            "pred_std": float(P[:, i].std()),
            "side": data.side[i],
            "icode": m.group("icode"),
        })
    df = pd.DataFrame(rows).sort_values("pred_prob", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    # rank within each chain, so a per-protein view is available too
    df["rank_in_chain"] = df.groupby("chain")["pred_prob"].rank(ascending=False).astype(int)

    out.parent.mkdir(parents=True, exist_ok=True)
    df[["chain", "res_id", "res_name", "pred_prob"]].to_csv(out, index=False)
    df.to_csv(out.with_name(out.stem + "_full.csv"), index=False)

    # ---- report ---------------------------------------------------------------------------
    for ch in sorted(df["chain"].unique()):
        sub = df[df["chain"] == ch].head(args.top)
        print(f"\n  TOP {args.top} — chain {ch}  ({len(df[df['chain']==ch])} interface residues)")
        print(f"    {'rank':>4} {'overall':>8}  {'residue':<12}{'prob':>8}{'+/-':>8}")
        print("    " + "-" * 44)
        for r in sub.itertuples():
            print(f"    {r.rank_in_chain:>4} {r.rank:>8}  "
                  f"{r.res_name}{r.res_id}{r.icode:<6}{r.pred_prob:>8.4f}{r.pred_std:>8.4f}")

    if args.highlight:
        print(f"\n  HIGHLIGHTED RESIDUES")
        print(f"    {'residue':<14}{'overall':>8}{'in-chain':>10}{'prob':>9}{'+/-':>8}")
        print("    " + "-" * 50)
        for h in args.highlight:
            ch, rid = h.split(":")
            hit = df[(df["chain"] == ch) & (df["res_id"] == int(rid))]
            if hit.empty:
                print(f"    {h:<14}  NOT AT THE INTERFACE (no prediction)")
                continue
            r = hit.iloc[0]
            print(f"    {ch}/{r['res_name']}{r['res_id']:<8}{r['rank']:>8}"
                  f"{r['rank_in_chain']:>10}{r['pred_prob']:>9.4f}{r['pred_std']:>8.4f}")

    print(f"\n  wrote {out}")
    print(f"        {out.with_name(out.stem + '_full.csv')}  (ranks, std, side)")
    print("\n  Scores are ranking signals, not calibrated probabilities (trained with")
    print("  positive-class weighting). Performance claim = CV PR-AUC 0.5076 +/- 0.0531.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
