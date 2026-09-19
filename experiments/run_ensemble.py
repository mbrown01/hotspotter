"""Ensemble the GNN and the tabular model on identical out-of-fold predictions.

WHY THIS IS WORTH TESTING
    The two models tied (GNN 0.5076, XGBoost 0.4962) and we kept the simpler one. But a tie
    in aggregate does not mean the same predictions -- XGBoost splits on flat per-residue
    quantities while the GAT passes messages over the contact graph, so they can be right
    about different residues. When two models of different inductive bias make decorrelated
    errors, averaging them usually beats either.

    The test is whether their errors ARE decorrelated. This script measures that directly
    (correlation of the two probability vectors) alongside the ensemble score, because a
    high correlation would explain a null result before anyone has to guess at one.

HOW THE COMPARISON IS KEPT HONEST
    Both arms are trained and scored on the SAME folds from data/split_alanine.json, and
    both produce out-of-fold predictions only -- every residue is scored by a model that
    never saw its complex. The two prediction vectors are then joined per residue, so the
    ensemble is computed over identical rows rather than two separately-aggregated numbers.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\run_ensemble.py --seeds 1    # quick pilot
    .\\.venv\\Scripts\\python.exe scripts\\run_ensemble.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotspotter.ml.dataset import parse_mutation  # noqa: E402
from hotspotter.ml.features import XGB_FEATURES, collapse  # noqa: E402

LABEL_RE = re.compile(r"^(?P<chain>.+)/(?P<resname>[A-Z]{3})(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)$")
FEATURES = list(XGB_FEATURES)


def gnn_oof(graphs_pt: Path, split: dict, seeds: list[int], args):
    """Out-of-fold GNN probabilities, keyed by (pdb, chain, resseq)."""
    import torch
    from torch_geometric.loader import DataLoader

    from gnn_model import GNNConfig, HotSpotGAT, masked_bce_loss

    graphs = torch.load(graphs_pt, weights_only=False)
    by_complex = {g.complex_group: g for g in graphs}
    cfg = GNNConfig(hidden_channels=32, n_layers=2, dropout=0.5, wide=True)
    acc: dict[tuple, list[float]] = {}

    for fi, fold in enumerate(split["cv_folds"]):
        tr = [by_complex[c] for c in fold["train_complexes"] if c in by_complex]
        te = [by_complex[c] for c in fold["test_complexes"] if c in by_complex]
        for seed in seeds:
            torch.manual_seed(seed + fi)
            model = HotSpotGAT(cfg)
            model.fit_standardizer(tr)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            pos = sum(int((g.y == 1.0).sum()) for g in tr)
            neg = sum(int((g.y == 0.0).sum()) for g in tr)
            pw = torch.tensor([neg / max(1, pos)], dtype=torch.float)
            loader = DataLoader(tr, batch_size=16, shuffle=True)
            model.train()
            for _ in range(args.epochs):
                for b in loader:
                    opt.zero_grad()
                    lg = model(b.x, b.edge_index, b.edge_attr, b.batch)
                    loss = masked_bce_loss(lg, b.y, b.label_mask, pos_weight=pw)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            model.eval()
            with torch.no_grad():
                for g in te:
                    p = torch.sigmoid(model(g.x, g.edge_index, g.edge_attr)).numpy()
                    m = g.label_mask.numpy()
                    for i, lab in enumerate(g.residue_labels):
                        if not m[i]:
                            continue
                        x = LABEL_RE.match(lab)
                        k = (g.complex_group, x.group("chain"), int(x.group("resseq")))
                        acc.setdefault(k, []).append(float(p[i]))
        print(f"    gnn fold {fi} done", flush=True)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def xgb_oof(df: pd.DataFrame, split: dict, seeds: list[int]):
    """Out-of-fold XGBoost probabilities on the same folds, keyed the same way."""
    from xgboost import XGBClassifier

    X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    pdb = df["pdb_id"].values
    oof = np.zeros((len(seeds), len(df)))
    for fold in split["cv_folds"]:
        te_mask = np.isin(pdb, fold["test_complexes"])
        tr_mask = np.isin(pdb, fold["train_complexes"])
        for si, seed in enumerate(seeds):
            p = max(1, int(y[tr_mask].sum()))
            m = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=50,
                scale_pos_weight=(tr_mask.sum() - y[tr_mask].sum()) / p,
                eval_metric="aucpr", random_state=seed, n_jobs=-1,
            ).fit(X[tr_mask], y[tr_mask])
            oof[si, te_mask] = m.predict_proba(X[te_mask])[:, 1]
    mean = oof.mean(0)
    return {(r.pdb_id, r.chain, int(r.resseq)): float(mean[i])
            for i, r in enumerate(df.itertuples())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", type=Path, default=REPO_ROOT / "data" / "graphs_alanine.pt")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "data" / "split_alanine.json")
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=27)
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score, roc_auc_score

    split = json.loads(args.split.read_text())
    seeds = list(range(args.seeds))

    df = pd.read_csv(args.csv)
    df["mut"] = df["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)
    df["pdb_id"] = df["complex_group"]
    df = collapse(df, "alanine", 2.0)

    print("=" * 78)
    print(f"  ENSEMBLE: GNN (Wide&Deep) + XGBoost (V1+L1)  |  {args.seeds} seed(s)")
    print("=" * 78)
    print(f"  folds from {args.split.name} | {len(split['cv_folds'])} folds")

    print("\n  training GNN arm...")
    g_oof = gnn_oof(args.graphs, split, seeds, args)
    print(f"  gnn scored {len(g_oof)} residues")

    print("\n  training XGBoost arm...")
    x_oof = xgb_oof(df, split, seeds)
    print(f"  xgb scored {len(x_oof)} residues")

    # join: only residues BOTH models scored, so the comparison is row-identical
    keys = sorted(set(g_oof) & set(x_oof))
    lab = {(r.pdb_id, r.chain, int(r.resseq)): int(r.label) for r in df.itertuples()}
    y = np.array([lab[k] for k in keys])
    pg = np.array([g_oof[k] for k in keys])
    px = np.array([x_oof[k] for k in keys])
    print(f"\n  joined on {len(keys)} residues ({y.sum()} hot, {y.mean():.1%})")

    # do the two models actually disagree? this is the precondition for ensembling
    from scipy import stats
    r_p = stats.pearsonr(px, pg)[0]
    r_s = stats.spearmanr(px, pg)[0]
    print(f"  correlation of the two probability vectors: pearson {r_p:.3f}, "
          f"spearman {r_s:.3f}")
    print("  (high correlation => the models agree => little to gain from averaging)")

    print("\n" + "-" * 78)
    print(f"    {'model':<26}{'PR-AUC':>9}{'ROC-AUC':>10}")
    print("-" * 78)
    rows = [("XGBoost (V1+L1)", px), ("GNN (Wide&Deep)", pg)]
    for w in (0.5, 0.7, 0.3):
        rows.append((f"ensemble {w:.1f}*xgb + {1-w:.1f}*gnn", w * px + (1 - w) * pg))
    # rank-average is scale-free, which matters when two models are calibrated differently
    rx = stats.rankdata(px) / len(px)
    rg = stats.rankdata(pg) / len(pg)
    rows.append(("ensemble rank-average", 0.5 * rx + 0.5 * rg))

    # Bootstrap the ensemble gain BY COMPLEX (not by residue): residues of one interface
    # are correlated, so resampling residues would understate the uncertainty.
    pdbs = np.array([k[0] for k in keys])
    uniq = np.unique(pdbs)
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(2000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(pdbs == c) for c in pick])
        if len(set(y[idx])) < 2:
            continue
        ens = 0.5 * px[idx] + 0.5 * pg[idx]
        deltas.append(average_precision_score(y[idx], ens)
                      - average_precision_score(y[idx], px[idx]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print("")
    print(f"  bootstrap over complexes (n=2000): ensemble - xgb = "
          f"{deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  fraction of resamples where the ensemble wins: "
          f"{100*(deltas > 0).mean():.1f}%")

    best = None
    for tag, p in rows:
        pr, roc = average_precision_score(y, p), roc_auc_score(y, p)
        flag = ""
        if tag.startswith("XGBoost"):
            best = pr
        elif best is not None and pr > best:
            flag = f"   (+{pr-best:.4f} over xgb)"
        print(f"    {tag:<26}{pr:>9.4f}{roc:>10.4f}{flag}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
