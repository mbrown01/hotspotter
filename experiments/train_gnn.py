"""Train and evaluate the GAT against the matched XGBoost baseline, fold for fold.

THE COMPARISON THIS SCRIPT PROTECTS
    The matched baseline scored PR-AUC 0.5020 +/- 0.0805 across 5 grouped folds on exactly
    1,540 labeled residues in 172 complexes. That +/- 0.08 is the whole story: fold scores
    ranged 0.405 to 0.635. A GNN number is only meaningful if it is produced on the SAME
    folds, and only convincing if it clears that spread.

    So this script does not compute its own split. It loads ``data/split_alanine.json`` and
    uses the exact fold membership the baseline recorded. Re-deriving folds with GroupKFold
    would silently disagree: sklearn balances folds by group size, and the GNN's unit is a
    graph (172 items) where the baseline's is a residue (1,540).

THREE WAYS THIS COULD FOOL US, AND WHAT IS DONE ABOUT EACH
    Leakage through normalization  Feature statistics are fitted on each fold's TRAINING
                                   graphs only, never on all data (see FeatureStandardizer).
    Leakage through early stopping Epoch selection uses an INNER validation split carved out
                                   of the training complexes. The test fold is touched once,
                                   at the end. Selecting the best epoch by test score is the
                                   easiest way to manufacture a win on a set this small.
    Reading noise as signal        Every fold is reported, not just the mean, and the naive
                                   burial baseline is scored on the identical test nodes.

Usage::

    .\\.venv\\Scripts\\python.exe experiments\\train_gnn.py --dry-run   # shapes only, no training
    .\\.venv\\Scripts\\python.exe experiments\\train_gnn.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

DEFAULT_GRAPHS = REPO_ROOT / "data" / "graphs_alanine.pt"
DEFAULT_SPLIT = REPO_ROOT / "data" / "split_alanine.json"


# ---------------------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------------------
def load_graphs_and_split(graphs_pt: Path, split_json: Path):
    import torch

    graphs = torch.load(graphs_pt, weights_only=False)
    split = json.loads(split_json.read_text())
    by_complex = {g.complex_group: g for g in graphs}

    if len(by_complex) != len(graphs):
        raise AssertionError("duplicate complex_group among graphs")

    # The split file must describe exactly the graphs we hold, or the comparison is broken.
    covered = {c for f in split["cv_folds"] for c in f["test_complexes"]}
    missing = covered - set(by_complex)
    extra = set(by_complex) - covered
    if missing or extra:
        raise AssertionError(
            f"graph set and split file disagree — missing {sorted(missing)[:5]}, "
            f"extra {sorted(extra)[:5]}. Re-run run_matched_baseline.py."
        )
    return graphs, by_complex, split


def inner_validation_split(train_complexes, val_fraction: float, seed: int):
    """Hold out whole complexes from the training fold for early stopping.

    Splitting by COMPLEX rather than by node matters for the same reason the outer split
    does: two residues of one interface share a structure, so putting them on both sides
    would make the validation score optimistic and stop training at the wrong epoch.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(train_complexes)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def augment_with_stack(graph, probs):
    """Return a copy of `graph` with the fold-specific XGBoost probability as a 27th column.

    The probabilities are OUTER-FOLD SPECIFIC (see build_stack_features.py), so this must be
    applied fresh inside each fold rather than baked into the saved graphs.
    """
    import torch

    g = graph.clone()
    if probs.numel() != g.num_nodes:
        raise AssertionError(
            f"{g.complex_group}: {probs.numel()} stacked probs for {g.num_nodes} nodes"
        )
    g.x = torch.cat([g.x, probs.view(-1, 1).to(g.x.dtype)], dim=1)
    return g


def augment_with_esm(graph, emb):
    """Append the per-residue ESM-2 embedding block to a graph's node features.

    Unlike the stacked XGBoost probability, ESM embeddings depend only on sequence, not on
    any fold's training set, so they can be attached once rather than per fold.
    """
    import torch

    g = graph.clone()
    if emb.shape[0] != g.num_nodes:
        raise AssertionError(
            f"{g.complex_group}: {emb.shape[0]} ESM rows for {g.num_nodes} nodes"
        )
    g.x = torch.cat([g.x, emb.to(g.x.dtype)], dim=1)
    return g


def pos_weight_for(graphs, device):
    """n_negative / n_positive over labeled nodes — mirrors XGBoost's scale_pos_weight."""
    import torch

    pos = sum(int((g.y == 1.0).sum()) for g in graphs)
    neg = sum(int((g.y == 0.0).sum()) for g in graphs)
    if pos == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float, device=device)


# ---------------------------------------------------------------------------------------
# Train / evaluate one fold
# ---------------------------------------------------------------------------------------
def evaluate(model, loader, device):
    """Return (y_true, y_score, dsasa) over LABELED nodes only."""
    import torch
    from hotspotter.ml.graph_dataset import NODE_FEATURES

    dsasa_col = NODE_FEATURES.index("dsasa")
    model.eval()
    ys, scores, naive = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            m = batch.label_mask
            if int(m.sum()) == 0:
                continue
            ys.append(batch.y[m].cpu().numpy())
            scores.append(torch.sigmoid(logits[m]).cpu().numpy())
            # raw (unstandardized) buried surface area, for the naive baseline
            naive.append(batch.x[m, dsasa_col].cpu().numpy())
    if not ys:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(scores), np.concatenate(naive)


def train_one_fold(train_graphs, val_graphs, test_graphs, cfg, args, device, fold_i):
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch_geometric.loader import DataLoader

    from gnn_model import HotSpotGAT, masked_bce_loss

    torch.manual_seed(args.seed + fold_i)

    model = HotSpotGAT(cfg).to(device)
    # Fit normalization on TRAIN ONLY (not val, not test).
    model.fit_standardizer(train_graphs)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=args.patience // 2)
    pw = pos_weight_for(train_graphs, device)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    best_val, best_state, best_epoch, since_best = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = masked_bce_loss(logits, batch.y, batch.label_mask, pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += loss.detach().item() * int(batch.num_graphs)

        y_val, s_val, _ = evaluate(model, val_loader, device)
        val_pr = float(average_precision_score(y_val, s_val)) if len(set(y_val)) > 1 else 0.0
        sched.step(val_pr)

        if val_pr > best_val:
            best_val, best_epoch, since_best = val_pr, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1

        if args.verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}  loss {total/max(1,len(train_graphs)):.4f}  "
                  f"val PR-AUC {val_pr:.4f}  (best {best_val:.4f} @ {best_epoch})",
                  flush=True)

        if since_best >= args.patience:
            if args.verbose:
                print(f"    early stop at epoch {epoch} (no val gain for {args.patience})",
                      flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # The test fold is scored exactly once, with the epoch chosen on validation.
    y_te, s_te, naive_te = evaluate(model, test_loader, device)
    return {
        "fold": fold_i,
        "best_epoch": best_epoch,
        "val_pr_auc": best_val,
        "gnn_pr_auc": float(average_precision_score(y_te, s_te)),
        "naive_pr_auc": float(average_precision_score(y_te, naive_te)),
        "roc_auc": float(roc_auc_score(y_te, s_te)) if len(set(y_te)) > 1 else float("nan"),
        "n_test_nodes": int(len(y_te)),
        "test_positive_rate": float(np.mean(y_te)),
    }


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", type=Path, default=DEFAULT_GRAPHS)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    # architecture
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--norm", choices=("layer", "batch"), default="layer")
    ap.add_argument("--wide", action="store_true",
                    help="Wide & Deep: concat standardized raw features into the head")
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help="node feature names to exclude (Fix #2 feature cleanup)")
    ap.add_argument("--clip-bfactor", type=float, default=None,
                    help="clip the bfactor column at this value before standardizing")
    ap.add_argument("--conservation-features", type=Path, default=None,
                    help="append the ESM masked-LM conservation scalar as one feature")
    ap.add_argument("--esm-features", type=Path, default=None,
                    help="Phase-2: .pt of per-residue ESM-2 embeddings to append to x "
                         "(from build_esm_features.py). Not fold-specific.")
    ap.add_argument("--stack-features", type=Path, default=None,
                    help="Fix #3: .pt of nested OOF XGBoost probabilities to append as a "
                         "27th node feature (from build_stack_features.py)")
    # behavior
    ap.add_argument("--train-full", action="store_true",
                    help="train ONE model on ALL complexes for --epochs and save a "
                         "checkpoint for inference (no held-out set, no early stopping)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="where to save the --train-full checkpoint")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the model, run ONE forward pass, print shapes, and exit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "gnn_results.json")
    args = ap.parse_args()

    for p in (args.graphs, args.split):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    import torch
    from torch_geometric.loader import DataLoader

    from gnn_model import GNNConfig, HotSpotGAT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs, by_complex, split = load_graphs_and_split(args.graphs, args.split)
    esm_dim = 0
    esm_src = args.esm_features or args.conservation_features
    esm_prefix = "esm" if args.esm_features else "cons"
    if esm_src:
        emb = torch.load(esm_src, weights_only=False)
        missing = set(by_complex) - set(emb)
        if missing:
            raise AssertionError(f"ESM file missing {len(missing)} complexes, e.g. "
                                 f"{sorted(missing)[:3]}")
        esm_dim = int(next(iter(emb.values())).shape[1])
        by_complex = {c: augment_with_esm(g, emb[c]) for c, g in by_complex.items()}
        graphs = list(by_complex.values())

    extra = tuple(f"{esm_prefix}_{i}" for i in range(esm_dim))
    cfg = GNNConfig(
        hidden_channels=args.hidden, n_layers=args.layers, heads=args.heads,
        dropout=args.dropout, residual=not args.no_residual, norm=args.norm,
        wide=args.wide, drop_features=tuple(args.drop_features),
        clip_bfactor=args.clip_bfactor,
        extra_feature_names=extra + (("xgb_prob",) if args.stack_features else ()),
    )

    base = split.get("baseline", {}).get("cv", {})
    print("=" * 78)
    print("  GNN: edge-aware GATv2 node classifier")
    print("=" * 78)
    print(f"  device            : {device}")
    print(f"  graphs            : {len(graphs)} complexes, "
          f"{sum(int(g.num_nodes) for g in graphs)} nodes, "
          f"{sum(int(g.label_mask.sum()) for g in graphs)} labeled")
    print(f"  architecture      : {HotSpotGAT(cfg).describe()}")
    if base:
        print(f"  TARGET (XGBoost)  : PR-AUC {base['xgboost_pr_auc_mean']:.4f} "
              f"+/- {base['xgboost_pr_auc_std']:.4f}   "
              f"(naive {base['naive_pr_auc_mean']:.4f})")

    # ---- dry run: prove the wiring, train nothing ---------------------------------------
    if args.dry_run:
        model = HotSpotGAT(cfg).to(device)
        probe = graphs[:4]
        if args.stack_features:
            sk = torch.load(args.stack_features, weights_only=False)[0]
            probe = [augment_with_stack(g, sk[g.complex_group]) for g in probe]
        model.fit_standardizer(probe)
        batch = next(iter(DataLoader(probe, batch_size=4))).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        print("\n  DRY RUN — forward pass only")
        print(f"    x          {tuple(batch.x.shape)}")
        print(f"    edge_index {tuple(batch.edge_index.shape)}")
        print(f"    edge_attr  {tuple(batch.edge_attr.shape)}")
        print(f"    logits     {tuple(logits.shape)}  "
              f"range [{float(logits.min()):.3f}, {float(logits.max()):.3f}]")
        print(f"    labeled    {int(batch.label_mask.sum())} of {int(batch.num_nodes)} nodes")
        assert logits.shape == (batch.num_nodes,), "expected one logit per node"
        assert torch.isfinite(logits).all(), "non-finite logits"
        print("    OK: one finite logit per node. No training performed.")
        print("=" * 78)
        return 0

    # ---- deployable model: train on everything, no held-out set ------------------------
    if args.train_full:
        from gnn_model import HotSpotGAT, masked_bce_loss
        from torch_geometric.loader import DataLoader as DL

        torch.manual_seed(args.seed)
        model = HotSpotGAT(cfg).to(device)
        model.fit_standardizer(graphs)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        pw = pos_weight_for(graphs, device)
        loader = DL(graphs, batch_size=args.batch_size, shuffle=True)

        print("")
        print(f"  TRAIN-FULL: {len(graphs)} complexes, {args.epochs} epochs, "
              f"seed {args.seed}")
        print("  No validation set by construction — the performance claim for this model is")
        print("  the cross-validated estimate, not anything measured on its own training data.")
        for epoch in range(1, args.epochs + 1):
            model.train()
            tot = 0.0
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                lg = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = masked_bce_loss(lg, batch.y, batch.label_mask, pos_weight=pw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                tot += loss.detach().item() * int(batch.num_graphs)
            if epoch % 5 == 0 or epoch == 1:
                print(f"    epoch {epoch:3d}  loss {tot/len(graphs):.4f}", flush=True)

        ckpt = args.checkpoint or (REPO_ROOT / "data" / f"model_full_s{args.seed}.pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict(),
                    "epochs": args.epochs, "seed": args.seed,
                    "n_train_complexes": len(graphs),
                    "cv_reference": split.get("baseline", {}).get("cv", {})}, ckpt)
        print(f"  saved checkpoint -> {ckpt}")
        print("=" * 78)
        return 0

    # ---- 5-fold CV on the baseline's exact folds ----------------------------------------
    stack = None
    if args.stack_features:
        stack = torch.load(args.stack_features, weights_only=False)
        if len(stack) != len(split["cv_folds"]):
            raise AssertionError(
                f"stack file has {len(stack)} folds, split has {len(split['cv_folds'])}"
            )
        print(f"  STACKED: nested OOF XGBoost probability as feature 27 "
              f"({args.stack_features.name})")

    results, t0 = [], time.time()
    for i, fold in enumerate(split["cv_folds"]):
        tr_complexes, val_complexes = inner_validation_split(
            fold["train_complexes"], args.val_fraction, args.seed + i
        )
        if stack is not None:
            sk = stack[i]
            pick = lambda c: augment_with_stack(by_complex[c], sk[c])  # noqa: E731
        else:
            pick = lambda c: by_complex[c]  # noqa: E731
        tr = [pick(c) for c in tr_complexes]
        va = [pick(c) for c in val_complexes]
        te = [pick(c) for c in fold["test_complexes"]]

        print(f"\n  fold {i}: train {len(tr)} / val {len(va)} / test {len(te)} complexes")
        r = train_one_fold(tr, va, te, cfg, args, device, i)
        results.append(r)
        print(f"    -> GNN PR-AUC {r['gnn_pr_auc']:.4f}   "
              f"naive {r['naive_pr_auc']:.4f}   "
              f"(best epoch {r['best_epoch']}, {r['n_test_nodes']} test nodes)")

    gnn = np.array([r["gnn_pr_auc"] for r in results])
    naive = np.array([r["naive_pr_auc"] for r in results])

    print("\n" + "-" * 78)
    print("  5-FOLD RESULTS  (same folds as the matched XGBoost baseline)")
    print("-" * 78)
    print(f"  NAIVE SASA PR-AUC : {naive.mean():.4f} +/- {naive.std():.4f}")
    print(f"  GNN PR-AUC        : {gnn.mean():.4f} +/- {gnn.std():.4f}")
    print(f"  fold scores       : {[round(float(s), 4) for s in gnn]}")
    if base:
        xgb_mean = base["xgboost_pr_auc_mean"]
        xgb_std = base["xgboost_pr_auc_std"]
        delta = gnn.mean() - xgb_mean
        print(f"  XGBOOST PR-AUC    : {xgb_mean:.4f} +/- {xgb_std:.4f}")
        print(f"\n  GNN - XGBoost     : {delta:+.4f}")
        if abs(delta) < max(gnn.std(), xgb_std):
            print("  VERDICT: within fold-to-fold noise. This is a TIE, not a win —")
            print("           the difference is smaller than the spread across folds.")
        else:
            better = "GNN" if delta > 0 else "XGBoost"
            print(f"  VERDICT: {better} ahead by more than one standard deviation.")
            print("           Still worth repeating across seeds before claiming it.")

    args.out.write_text(json.dumps(
        {"config": cfg.as_dict(), "args": {k: str(v) for k, v in vars(args).items()},
         "folds": results,
         "gnn_pr_auc_mean": float(gnn.mean()), "gnn_pr_auc_std": float(gnn.std()),
         "naive_pr_auc_mean": float(naive.mean()),
         "baseline_cv": base}, indent=2))
    print(f"\n  results -> {args.out}   ({time.time()-t0:.0f}s)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
