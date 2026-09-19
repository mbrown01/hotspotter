"""Generate NESTED out-of-fold XGBoost probabilities to stack into the GNN's node features.

THE LEAKAGE THIS AVOIDS
    The obvious implementation is: run 5-fold CV once, collect out-of-fold probabilities for
    every node, append as a feature, done. That leaks, and the leak is easy to miss.

    Consider the GNN training on outer fold k. One of its training nodes lives in fold j.
    That node's "out-of-fold" probability came from an XGBoost trained on every fold except
    j -- a set that INCLUDES fold k. So the GNN's training feature encodes information
    derived from the GNN's own test complexes. The stacked feature would look unreasonably
    good, and the win would be an artifact.

    The correct protocol is nested. For each outer fold k:

        test complexes   -> predicted by an XGBoost trained on ALL of fold k's training
                            complexes (which never include fold k). Honest.
        train complexes  -> predicted by an INNER 5-fold CV run strictly inside fold k's
                            training complexes. No inner model ever sees a fold-k complex.

    That means 5 outer folds x (5 inner models + 1 full model) = 30 XGBoost fits, and a
    SEPARATE probability vector per outer fold. One global vector cannot be correct here.

WHAT IS PRODUCED
    data/stack_<strategy>.pt -- {outer_fold_index: {complex_group: tensor[n_nodes]}}, with
    probabilities for EVERY node (labeled or not), row-aligned to that graph's node order.
    Unlabeled nodes get a prediction too: the GNN benefits from the prior everywhere, and an
    unlabeled node carries no target to leak.

Usage::

    .\\.venv\\Scripts\\python.exe experiments\\build_stack_features.py --strategy alanine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotspotter.ml.graph_dataset import NODE_FEATURES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matched_baseline import build_matched_frame  # noqa: E402


def fit_xgb(X, y, seed: int):
    from xgboost import XGBClassifier

    pos = max(1, int(y.sum()))
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(len(y) - y.sum()) / pos,
        eval_metric="aucpr", random_state=seed, n_jobs=-1,
    ).fit(X, y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--features", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from sklearn.model_selection import GroupKFold

    graphs_pt = REPO_ROOT / "data" / f"graphs_{args.strategy}.pt"
    split_json = REPO_ROOT / "data" / f"split_{args.strategy}.json"
    out_pt = REPO_ROOT / "data" / f"stack_{args.strategy}.pt"
    for p in (graphs_pt, split_json, args.features):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    graphs = torch.load(graphs_pt, weights_only=False)
    by_complex = {g.complex_group: g for g in graphs}
    split = json.loads(split_json.read_text())

    # The labeled training table, identical to what the matched baseline trains on.
    df, _ = build_matched_frame(args.features, graphs_pt, split["threshold"],
                                strategy=args.strategy)
    feats = list(NODE_FEATURES)
    df_X = df[feats].fillna(df[feats].median(numeric_only=True)).fillna(0.0)
    df_y = df["label"].astype(int).values
    df_groups = df["complex_group"].values

    # Every node of every graph, as a feature matrix we can score with any fitted model.
    node_rows, node_owner = [], []
    for g in graphs:
        node_rows.append(g.x.numpy())
        node_owner.extend([g.complex_group] * int(g.num_nodes))
    node_X = pd.DataFrame(np.concatenate(node_rows, axis=0), columns=feats)
    node_owner = np.array(node_owner)

    print("=" * 78)
    print(f"  NESTED OOF STACK FEATURES  |  strategy={args.strategy}")
    print("=" * 78)
    print(f"  graphs {len(graphs)} | labeled rows {len(df)} | total nodes {len(node_X)}")
    print(f"  outer folds {len(split['cv_folds'])} x (inner {args.inner_folds} + 1 full) "
          f"= {len(split['cv_folds']) * (args.inner_folds + 1)} XGBoost fits\n")

    stack: dict[int, dict[str, "torch.Tensor"]] = {}

    for k, fold in enumerate(split["cv_folds"]):
        train_c = set(fold["train_complexes"])
        test_c = set(fold["test_complexes"])
        assert not (train_c & test_c)

        # probability per node, for THIS outer fold only
        probs = np.full(len(node_X), np.nan, dtype=np.float64)

        # --- test complexes: model trained on ALL of this fold's training complexes ------
        tr_mask = np.isin(df_groups, list(train_c))
        full = fit_xgb(df_X[tr_mask], df_y[tr_mask], args.seed)
        te_nodes = np.isin(node_owner, list(test_c))
        probs[te_nodes] = full.predict_proba(node_X[te_nodes])[:, 1]

        # --- train complexes: inner CV strictly inside the training complexes ------------
        inner_df = df[tr_mask]
        inner_X = df_X[tr_mask]
        inner_y = df_y[tr_mask]
        inner_groups = df_groups[tr_mask]
        gkf = GroupKFold(n_splits=args.inner_folds)
        for itr, ite in gkf.split(inner_X, inner_y, inner_groups):
            hold_c = set(inner_groups[ite])          # complexes held out of this inner fit
            m = fit_xgb(inner_X.iloc[itr], inner_y[itr], args.seed)
            sel = np.isin(node_owner, list(hold_c))
            probs[sel] = m.predict_proba(node_X[sel])[:, 1]

        if np.isnan(probs).any():
            raise AssertionError(f"fold {k}: {int(np.isnan(probs).sum())} nodes unscored")

        # slice back per graph, aligned to node order
        stack[k] = {}
        off = 0
        for g in graphs:
            n = int(g.num_nodes)
            stack[k][g.complex_group] = torch.tensor(probs[off:off + n], dtype=torch.float)
            off += n
        assert off == len(probs)

        te = probs[te_nodes]
        print(f"  fold {k}: test-node prob mean {te.mean():.4f} "
              f"[{te.min():.3f}, {te.max():.3f}] | train complexes {len(train_c)}, "
              f"test {len(test_c)}")

    torch.save(stack, out_pt)
    print(f"\n  saved -> {out_pt}")

    # --- sanity: does the stacked feature actually reproduce the baseline's skill? --------
    from sklearn.metrics import average_precision_score
    print("\n  SANITY — PR-AUC of the stacked probability alone, on each fold's test nodes:")
    print("  (should land near the matched baseline; far above would signal leakage)")
    scores = []
    for k, fold in enumerate(split["cv_folds"]):
        ys, ps = [], []
        for c in fold["test_complexes"]:
            g = by_complex[c]
            m = g.label_mask.numpy()
            if m.sum() == 0:
                continue
            ys.append(g.y.numpy()[m])
            ps.append(stack[k][c].numpy()[m])
        y, p = np.concatenate(ys), np.concatenate(ps)
        s = average_precision_score(y, p)
        scores.append(s)
        print(f"    fold {k}: {s:.4f}")
    base = split.get("baseline", {}).get("cv", {}).get("xgboost_pr_auc_mean")
    print(f"    mean   : {np.mean(scores):.4f}"
          + (f"   (matched baseline {base:.4f})" if base else ""))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
