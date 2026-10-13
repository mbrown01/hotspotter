"""Measure what the choice of cross-validation split is worth.

The reported metrics group folds by complex. That is a claim about protocol, and this
script is the evidence for it: same data, same 25 features, same model, same seeds, with
only the fold assignment changing.

    random residue split    residues from one interface land on both sides of the split.
                            This is the protocol that inflates published numbers.
    group by pdb id         what the README reports.
    group by sequence       complexes clustered by chain identity, so near-duplicate
      identity cluster      entries (SKEMPI is redundant -- the same pair is deposited
                            many times) cannot straddle the split either. Grouping by
                            pdb id does not catch that on its own.

The naive buried-area baseline is reported under every regime as a control: it is not
trained, so it should not move. If it does, the difference is fold composition, not
leakage.

Clustering is single-linkage over complexes -- A and B merge if ANY chain of A matches ANY
chain of B at the identity threshold -- which over-merges rather than under-merges, so the
resulting estimate errs toward pessimism.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\audit_splits.py
    .\\.venv\\Scripts\\python.exe scripts\\audit_splits.py --identity 0.9 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from evaluate import XGB_KWARGS, ranking_table  # noqa: E402
from hotspotter.ml.features import XGB_FEATURES, load_labeled_table  # noqa: E402

FEATURES = list(XGB_FEATURES)

MIN_CHAIN_LEN = 20       # skip peptides and ligand chains too short to align meaningfully
KMER = 5                 # k-mer size for the prefilter
JACCARD_PREFILTER = 0.20  # below this, two chains cannot reach any useful identity


# --- sequence clustering ---------------------------------------------------------------
def chain_sequences(ids, raw_dir: Path, cache: Path | None) -> dict[str, dict[str, str]]:
    """{pdb_id: {chain_id: one-letter sequence}} for every chain of every complex.

    All chains, not only the ones carrying labels: redundancy is a property of the
    complex, and a duplicate entry has to be caught wherever it shows up.
    """
    if cache and cache.exists():
        return json.loads(cache.read_text())

    from Bio.PDB import PDBParser

    from hotspotter.constants import THREE_TO_ONE

    parser = PDBParser(QUIET=True)
    out: dict[str, dict[str, str]] = {}
    for n, pid in enumerate(ids, 1):
        path = raw_dir / f"{pid.lower()}.pdb"
        if not path.exists():
            print(f"    missing structure for {pid}, skipped", file=sys.stderr)
            continue
        model = next(parser.get_structure(pid, path).get_models())
        chains = {}
        for ch in model:
            seq = "".join(THREE_TO_ONE.get(r.get_resname(), "")
                          for r in ch if r.id[0] == " ")
            if len(seq) >= MIN_CHAIN_LEN:
                chains[ch.id] = seq
        out[pid] = chains
        if n % 40 == 0:
            print(f"    read {n}/{len(ids)} structures", flush=True)
    if cache:
        cache.write_text(json.dumps(out))
    return out


def _identity(a: str, b: str) -> float:
    """Percent identity over the shorter sequence, from a global alignment."""
    from Bio import Align

    aligner = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=0,
                                    open_gap_score=-1, extend_gap_score=-0.1)
    return aligner.score(a, b) / min(len(a), len(b))


def cluster_complexes(seqs: dict[str, dict[str, str]], threshold: float):
    """Single-linkage union-find over complexes; returns {pdb_id: cluster_id}."""
    chains = [(pid, s) for pid, d in seqs.items() for s in d.values()]
    kmers = [frozenset(s[i:i + KMER] for i in range(len(s) - KMER + 1))
             for _, s in chains]

    parent = {pid: pid for pid in seqs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, (pa, sa)), (j, (pb, sb)) in combinations(enumerate(chains), 2):
        if pa == pb or find(pa) == find(pb):
            continue
        ka, kb = kmers[i], kmers[j]
        if not ka or not kb:
            continue
        if len(ka & kb) / min(len(ka), len(kb)) < JACCARD_PREFILTER:
            continue
        if _identity(sa, sb) >= threshold:
            ra, rb = find(pa), find(pb)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    return {pid: find(pid) for pid in seqs}


# --- evaluation ------------------------------------------------------------------------
def _masks(n, splits):
    out = []
    for tr, te in splits:
        a = np.zeros(n, bool); a[tr] = True
        b = np.zeros(n, bool); b[te] = True
        out.append((a, b))
    return out


def folds_for(df, groups, n_folds: int, shuffle_seed: int | None):
    """Grouped folds, or — when `groups` is None — a leaky random residue split."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    y = df["label"].astype(int).values
    idx = np.arange(len(df))
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=shuffle_seed)
        return _masks(len(df), splitter.split(idx, y))
    return _masks(len(df), GroupKFold(n_splits=n_folds).split(idx, y, groups))


def cross_validate(df, folds, seeds):
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    pf = np.zeros((len(seeds), len(folds)))
    oof = np.zeros((len(seeds), len(df)))
    for si, seed in enumerate(seeds):
        for fi, (tr, te) in enumerate(folds):
            pos = max(1, int(y[tr].sum()))
            model = XGBClassifier(reg_alpha=50.0, random_state=seed,
                                  scale_pos_weight=(tr.sum() - y[tr].sum()) / pos,
                                  **XGB_KWARGS).fit(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            oof[si, te] = p
            pf[si, fi] = average_precision_score(y[te], p)
    return pf, oof.mean(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--cache", type=Path,
                    default=REPO_ROOT / "data" / "chain_sequences.json")
    ap.add_argument("--identity", type=float, nargs="*", default=[0.9, 0.7, 0.5],
                    help="sequence-identity thresholds to cluster complexes at")
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: missing {args.csv}\n"
              f"Build it first:  python scripts/build_features.py", file=sys.stderr)
        return 2

    from sklearn.metrics import average_precision_score, roc_auc_score

    df = load_labeled_table(args.csv, args.strategy, args.threshold)
    y = df["label"].astype(int).values
    naive = df["dsasa"].fillna(0).values
    pdb = df["pdb_id"].values
    seeds = list(range(args.seeds))
    ids = sorted(df.pdb_id.unique())

    print("=" * 84)
    print(f"  SPLIT AUDIT  |  strategy={args.strategy}  |  hot spot: "
          f"ddG >= {args.threshold} kcal/mol")
    print("=" * 84)
    print(f"  {len(df)} residues | {len(ids)} complexes | {int(y.sum())} hot "
          f"({y.mean():.1%}) | {len(FEATURES)} features | {args.seeds} seeds x "
          f"{args.folds} folds")

    regimes: list[tuple[str, int, object]] = [
        ("random residue split [leaky]", len(ids), None),
        ("group by complex [reported]", len(ids), pdb),
    ]

    if args.identity:
        print("  reading chain sequences...", flush=True)
        seqs = chain_sequences(ids, args.raw, args.cache)
        n_chains = sum(len(v) for v in seqs.values())
        print(f"  {n_chains} chains >= {MIN_CHAIN_LEN} aa across {len(seqs)} complexes")
        for thr in sorted(args.identity, reverse=True):
            print(f"  clustering at >= {thr:.0%} identity...", flush=True)
            cl = cluster_complexes(seqs, thr)
            groups = df["pdb_id"].map(cl).values
            regimes.append((f"group by >={thr:.0%}-identity cluster",
                            len(set(cl.values())), groups))

    print("\n" + "-" * 84)
    print(f"  {'split'.ljust(34)}{'groups':>8}{'PR-AUC':>10}{'sd':>8}{'ROC':>8}"
          f"{'naive PR':>10}{'hit@3':>8}")
    print("-" * 84)

    scores = {}
    for name, n_groups, groups in regimes:
        folds = folds_for(df, groups, args.folds, shuffle_seed=0)
        pf, oof = cross_validate(df, folds, seeds)
        npf = np.array([average_precision_score(y[te], naive[te]) for _, te in folds])
        scores[name] = pf.mean()
        hit = ranking_table(df, oof)
        print(f"  {name:<34}{n_groups:>8}{pf.mean():>10.4f}"
              f"{pf.std(axis=1).mean():>8.4f}{roc_auc_score(y, oof):>8.4f}"
              f"{npf.mean():>10.4f}{hit[3][0]:>8.1%}")

    ref = scores["group by complex [reported]"]
    print("-" * 84)
    print("  relative to the reported protocol:")
    for name, value in scores.items():
        if name == "group by complex [reported]":
            continue
        print(f"    {name:<36}{value - ref:+.4f} PR-AUC  ({100 * (value / ref - 1):+.1f}%)")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
