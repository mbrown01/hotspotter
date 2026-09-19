"""Predict hot spots on one complex with the accepted tabular baseline (V1 + L1 XGBoost).

WHY THIS EXISTS SEPARATELY FROM infer.py
    ``infer.py`` runs the GNN ensemble. After the architecture search, the accepted model is
    the simpler one: V1's 25 features with reg_alpha=50. This script is its inference path.

THE HOLD-OUT RULE, ENFORCED
    The target complex is ALWAYS excluded from training, by pdb id, before a single tree is
    fit. For 1BRS this is essential -- it is in SKEMPI with 12 labeled residues including
    Asp39, His102 and Arg87, so a model trained on it would be reciting memorised labels
    rather than predicting. The script refuses to run without applying the exclusion and
    prints what it removed.

THE COMPARISON THAT MATTERS
    Alongside the model ranking it computes the NAIVE BURIEDNESS ranking -- sort by buried
    surface area, the thing a structural biologist does by eye or with PISA. For every known
    hot spot it reports both ranks. That difference is the entire value proposition of the
    project, stated as a number.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\infer_xgb.py --pdb 1BRS --chains A,D \\
        --known A:87 A:102 A:59 A:27 D:39 D:35 D:29
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotspotter.ml.dataset import parse_mutation  # noqa: E402
from hotspotter.pipeline import analyze_complex  # noqa: E402
from hotspotter.ml.features import XGB_FEATURES, collapse  # noqa: E402

FEATURES = list(XGB_FEATURES)
LABEL_RE = re.compile(r"^(?P<chain>.+)/(?P<resname>[A-Z]{3})(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)$")


def train_model(csv: Path, exclude: set[str], strategy: str, threshold: float,
                alpha: float, seed: int):
    from xgboost import XGBClassifier

    df = pd.read_csv(csv)
    df["mut"] = df["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)
    df["pdb_id"] = df["complex_group"]
    before = df["pdb_id"].nunique()
    df = df[~df["pdb_id"].str.upper().isin(exclude)]
    after = df["pdb_id"].nunique()
    df = collapse(df, strategy, threshold)

    X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    pos = max(1, int(y.sum()))
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=alpha, scale_pos_weight=(len(y) - y.sum()) / pos,
        eval_metric="aucpr", random_state=seed, n_jobs=-1,
    ).fit(X, y)
    return model, dict(complexes_before=before, complexes_after=after,
                       rows=len(df), positives=int(y.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument("--seeds", type=int, default=5,
                    help="ensemble this many XGBoost seeds and average")
    ap.add_argument("--interface-cutoff", type=float, default=None)
    ap.add_argument("--known", nargs="*", default=[],
                    help="experimentally known hot spots, as CHAIN:RESID")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    left, right = args.chains.split(",", 1)
    side_a, side_b = tuple(left.strip()), tuple(right.strip())
    tag = Path(args.pdb).stem.lower()
    out = args.out or (REPO_ROOT / "predictions" / f"{tag}_xgb_predictions.csv")

    exclude = {args.pdb.upper()}
    print("=" * 78)
    print(f"  XGBoost (V1 + L1) INFERENCE  |  {args.pdb}  |  {side_a} vs {side_b}")
    print("=" * 78)

    models, info = [], None
    for s in range(args.seeds):
        m, info = train_model(args.csv, exclude, args.strategy, args.threshold,
                              args.alpha, s)
        models.append(m)
    print(f"  HELD OUT           : {sorted(exclude)}  "
          f"({info['complexes_before']} -> {info['complexes_after']} complexes)")
    print(f"  trained on         : {info['rows']} residues, {info['positives']} hot spots")
    print(f"  reg_alpha {args.alpha} | {args.seeds}-seed ensemble | strategy {args.strategy}")

    analysis = analyze_complex(args.pdb, chains=(side_a, side_b),
                               interface_cutoff=args.interface_cutoff)
    t = analysis.table.copy()
    X = t[FEATURES].fillna(t[FEATURES].median(numeric_only=True)).fillna(0.0)
    P = np.vstack([m.predict_proba(X)[:, 1] for m in models])

    t["pred_prob"] = P.mean(0)
    t["pred_std"] = P.std(0)
    t["res_id"] = t["resseq"].astype(int)
    t["res_name"] = t["resname"]
    t = t.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    t["model_rank"] = np.arange(1, len(t) + 1)
    # the strawman: what you get by sorting on buried surface area alone
    t["naive_rank"] = t["dsasa"].rank(ascending=False, method="min").astype(int)
    t["model_rank_in_chain"] = t.groupby("chain")["pred_prob"].rank(
        ascending=False, method="min").astype(int)
    t["naive_rank_in_chain"] = t.groupby("chain")["dsasa"].rank(
        ascending=False, method="min").astype(int)

    print(f"  interface residues : {len(t)}"
          f"  (cutoff {args.interface_cutoff or 5.0} A)")

    print(f"\n  TOP 10 OVERALL")
    print(f"    {'rank':>4}  {'residue':<12}{'prob':>8}{'+/-':>8}{'naive rank':>12}")
    print("    " + "-" * 46)
    for r in t.head(10).itertuples():
        print(f"    {r.model_rank:>4}  {r.chain}/{r.res_name}{r.res_id:<7}"
              f"{r.pred_prob:>8.4f}{r.pred_std:>8.4f}{r.naive_rank:>12}")

    if args.known:
        print(f"\n  KNOWN HOT SPOTS — model rank vs naive buriedness rank")
        print(f"    {'residue':<14}{'model':>7}{'naive':>7}{'moved':>8}{'prob':>9}")
        print("    " + "-" * 46)
        moves = []
        for k in args.known:
            ch, rid = k.split(":")
            hit = t[(t["chain"] == ch) & (t["res_id"] == int(rid))]
            if hit.empty:
                print(f"    {k:<14}  not at the interface")
                continue
            r = hit.iloc[0]
            mv = int(r["naive_rank"]) - int(r["model_rank"])
            moves.append((mv, int(r["model_rank"]), int(r["naive_rank"])))
            arrow = f"+{mv}" if mv > 0 else str(mv)
            print(f"    {ch}/{r['res_name']}{r['res_id']:<8}{int(r['model_rank']):>7}"
                  f"{int(r['naive_rank']):>7}{arrow:>8}{r['pred_prob']:>9.4f}")
        if moves:
            mv = np.array([m[0] for m in moves])
            mr = np.array([m[1] for m in moves])
            nr = np.array([m[2] for m in moves])
            n = len(t)
            print(f"\n    mean rank        model {mr.mean():.1f}  vs  naive {nr.mean():.1f}"
                  f"   (of {n} residues)")
            print(f"    median rank      model {np.median(mr):.0f}  vs  naive "
                  f"{np.median(nr):.0f}")
            print(f"    improved / total {int((mv > 0).sum())}/{len(mv)}")
            print(f"    top-5 recall     model {int((mr <= 5).sum())}/{len(mr)}  vs  "
                  f"naive {int((nr <= 5).sum())}/{len(nr)}")
            print(f"    top-10 recall    model {int((mr <= 10).sum())}/{len(mr)}  vs  "
                  f"naive {int((nr <= 10).sum())}/{len(nr)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["chain", "res_id", "res_name", "pred_prob"]
    t[cols].to_csv(out, index=False)
    t.to_csv(out.with_name(out.stem + "_full.csv"), index=False)
    print(f"\n  wrote {out}")
    print(f"        {out.with_name(out.stem + '_full.csv')}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
