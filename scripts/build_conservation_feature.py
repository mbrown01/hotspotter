"""Derive a per-residue conservation scalar from ESM-2 masked-language-model entropy.

WHAT THIS IS, BIOLOGICALLY
    Phase 1's roadmap named evolutionary conservation "the highest-value remaining feature"
    and then cut it, because the classical route is heavy plumbing: build a multiple
    sequence alignment per chain with MMseqs2 or HHblits, then score per-column entropy.

    A protein language model gives the same quantity by a different road. ESM-2 was trained
    to fill in masked residues across ~65M sequences, so at a position under strong
    evolutionary constraint it is confident about which amino acid belongs there, and at a
    tolerant position its prediction is spread out. Mask one residue, read the predicted
    distribution, take its Shannon entropy:

        low entropy  -> the model is sure -> the position is constrained -> CONSERVED
        high entropy -> anything fits     -> tolerant

    This is a learned substitute for MSA column entropy, and it needs no alignment step.

WHY MASKED MARGINALS AND NOT A PLAIN FORWARD PASS
    Running the sequence unmasked and reading the logits at each position is much cheaper,
    but the model can see the true residue sitting there and simply copies it. Entropy then
    measures the model's copying confidence, not evolutionary constraint. Masking the
    position first is what makes the number mean what we want.

    Cost is managed by only scoring positions that are actually graph nodes -- about 8,500
    of ~90,000 residues -- rather than every residue in every chain.

ONE SCALAR, NOT 320
    Appending ESM's 320-dimensional hidden state regressed performance (0.5076 -> 0.4940):
    346 inputs against 1,540 labels drowned the 26 tabular features. This extracts the one
    interpretable quantity those 320 dimensions were supposed to carry, as a single column
    a structural biologist can argue with.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\build_conservation_feature.py --strategy alanine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_esm_features import chain_sequences, parse_label  # noqa: E402

DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
CANONICAL = list("ACDEFGHIKLMNPQRSTVWY")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1022)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    graphs_pt = REPO_ROOT / "data" / f"graphs_{args.strategy}.pt"
    out_pt = REPO_ROOT / "data" / f"conservation_{args.strategy}.pt"
    if not graphs_pt.exists():
        print(f"ERROR: missing {graphs_pt}", file=sys.stderr)
        return 2

    graphs = torch.load(graphs_pt, weights_only=False)
    tok = AutoTokenizer.from_pretrained(args.model)
    mlm = AutoModelForMaskedLM.from_pretrained(args.model).eval()

    # Restrict the distribution to the 20 standard amino acids and renormalize, so entropy
    # measures "which residue belongs here" rather than leaking mass into special tokens.
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in CANONICAL])
    max_entropy = float(np.log(len(CANONICAL)))

    print("=" * 78)
    print(f"  ESM-2 MASKED-LM CONSERVATION  |  {args.model}  |  strategy={args.strategy}")
    print("=" * 78)
    print(f"  graphs {len(graphs)} | entropy in nats, max possible {max_entropy:.4f}\n")

    out: dict[str, "torch.Tensor"] = {}
    n_scored = n_missing = 0

    for n, g in enumerate(graphs, start=1):
        keys = [parse_label(lab) for lab in g.residue_labels]
        wanted = {k[0] for k in keys}
        try:
            seqs = chain_sequences(g.complex_group, wanted)
        except Exception as exc:
            print(f"  [cons] skip {g.complex_group}: {type(exc).__name__}: {exc}", flush=True)
            out[g.complex_group] = torch.full((int(g.num_nodes), 1), float(max_entropy))
            continue

        ent_by_key: dict[tuple, float] = {}
        needed = set(keys)

        for chain_id, (seq, chain_keys, _gaps) in seqs.items():
            if len(seq) > args.max_len:
                seq, chain_keys = seq[:args.max_len], chain_keys[:args.max_len]
            # only positions that are graph nodes
            positions = [i for i, k in enumerate(chain_keys) if k in needed]
            if not positions:
                continue

            enc = tok(seq, return_tensors="pt", add_special_tokens=True)
            base_ids = enc["input_ids"][0]          # [1 + L + 1]
            attn = enc["attention_mask"][0]

            for start in range(0, len(positions), args.batch_size):
                chunk = positions[start:start + args.batch_size]
                ids = base_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                for r, p in enumerate(chunk):
                    ids[r, p + 1] = tok.mask_token_id     # +1 skips <cls>
                with torch.no_grad():
                    logits = mlm(input_ids=ids,
                                 attention_mask=attn.unsqueeze(0).repeat(len(chunk), 1)).logits
                for r, p in enumerate(chunk):
                    lg = logits[r, p + 1, aa_ids]          # only the 20 canonical AAs
                    pr = torch.softmax(lg, dim=-1)
                    ent = float(-(pr * torch.log(pr.clamp_min(1e-12))).sum())
                    ent_by_key[chain_keys[p]] = ent

        col = np.full((int(g.num_nodes), 1), max_entropy, dtype=np.float32)
        for i, k in enumerate(keys):
            e = ent_by_key.get(k)
            if e is None:
                n_missing += 1        # truncated tail -> neutral (max entropy) fallback
            else:
                col[i, 0] = e
                n_scored += 1
        out[g.complex_group] = torch.tensor(col)

        if n % 25 == 0:
            print(f"  [cons] ...{n}/{len(graphs)} complexes", flush=True)

    torch.save(out, out_pt)
    allv = np.concatenate([v.numpy().ravel() for v in out.values()])
    print(f"\n  scored {n_scored} nodes, {n_missing} fell back to max entropy")
    print(f"  entropy: mean {allv.mean():.4f}  std {allv.std():.4f}  "
          f"min {allv.min():.4f}  max {allv.max():.4f}")
    print(f"  saved -> {out_pt}")

    # --- biological sanity: are conserved (low-entropy) residues the disruptive ones? -----
    from scipy import stats as st
    ys, es, dd = [], [], []
    for g in graphs:
        m = g.label_mask.numpy()
        if m.sum() == 0:
            continue
        ys.append(g.y.numpy()[m])
        es.append(out[g.complex_group].numpy().ravel()[m])
        dd.append(g.y_ddg.numpy()[m])
    y, e, d = np.concatenate(ys), np.concatenate(es), np.concatenate(dd)
    print("\n  BIOLOGICAL SANITY CHECK (labeled nodes only)")
    print(f"    mean entropy, hot spots (y=1) : {e[y == 1].mean():.4f}")
    print(f"    mean entropy, non-hot (y=0)   : {e[y == 0].mean():.4f}")
    t, p = st.ttest_ind(e[y == 1], e[y == 0], equal_var=False)
    r, pr = st.spearmanr(e, d)
    print(f"    Welch t-test                  : t={t:.3f}  p={p:.2e}")
    print(f"    Spearman(entropy, ddG)        : rho={r:.4f}  p={pr:.2e}")
    print("    (expect hot spots to have LOWER entropy and rho to be NEGATIVE)")
    from sklearn.metrics import average_precision_score
    print(f"    PR-AUC of -entropy alone      : {average_precision_score(y, -e):.4f}"
          f"   (chance {y.mean():.4f})")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
