"""Extract per-residue ESM-2 embeddings and align them to the graph nodes.

WHAT THIS TESTS
    Every feature so far is geometric or physicochemical, computed from one structure. A
    protein language model brings something genuinely new: evolutionary constraint learned
    from ~65M sequences. If a position is conserved across evolution, ESM's representation
    encodes that, and conservation is the one Phase-1 feature never built (it was the
    documented "highest-signal remaining feature" and was cut as too much plumbing).

    So this is not just another architecture tweak — it is the missing feature class.

THE ALIGNMENT PROBLEM, AND HOW IT IS HANDLED
    ESM embeds a SEQUENCE; our labels live on STRUCTURE residues with author numbering,
    insertion codes, and gaps where the crystal was disordered. Getting row i of the
    embedding onto the right residue is the whole job.

    The approach here is alignment-free by construction: for each chain we take the
    amino-acid residues that are actually present in the structure, in Biopython's order,
    build the one-letter sequence from exactly those, and embed it. Residue i of the
    sequence IS structure-residue i. No sequence alignment step, so no chance of an
    off-by-one silently shifting every embedding.

    HONEST LIMITATION: where the crystal is missing a loop, this hands ESM a sequence with
    the gap spliced out. ESM then sees a junction that does not exist in the real protein,
    and residues near that junction get a slightly wrong context. The alternative (use
    SEQRES, then align to the structure) is more biologically faithful but reintroduces the
    alignment step this avoids. For interface residues — which are usually ordered, not in
    disordered loops — the splice effect should be small, but it is a real caveat and the
    script reports how many gaps each structure has.

Usage::

    .\\.venv\\Scripts\\python.exe experiments\\build_esm_features.py --strategy alanine
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from hotspotter.constants import THREE_TO_ONE  # noqa: E402
from hotspotter.io import fetch_pdb, get_model, is_amino_acid, load_structure  # noqa: E402

LABEL_RE = re.compile(r"^(?P<chain>.+)/(?P<resname>[A-Z]{3})(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)$")
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


def parse_label(label: str):
    m = LABEL_RE.match(label)
    if not m:
        raise ValueError(f"cannot parse residue label {label!r}")
    return (m.group("chain"), int(m.group("resseq")), m.group("icode").strip())


def chain_sequences(pdb_id: str, wanted_chains: set[str]):
    """Return {chain: (sequence, [(chain, resseq, icode), ...], n_gaps)}.

    The key list is row-aligned with the sequence: key[i] is the residue that produced
    sequence[i]. ``n_gaps`` counts breaks in author numbering, a proxy for disorder.
    """
    structure = load_structure(fetch_pdb(pdb_id), structure_id=pdb_id)
    model = get_model(structure)
    out = {}
    for chain in model.get_chains():
        if chain.id not in wanted_chains:
            continue
        seq, keys, gaps, prev = [], [], 0, None
        for res in chain.get_residues():
            if not is_amino_acid(res):
                continue
            _, resseq, icode = res.get_id()
            seq.append(THREE_TO_ONE.get(res.get_resname(), "X"))
            keys.append((chain.id, int(resseq), icode.strip()))
            if prev is not None and int(resseq) not in (prev, prev + 1):
                gaps += 1
            prev = int(resseq)
        if seq:
            out[chain.id] = ("".join(seq), keys, gaps)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-len", type=int, default=1022,
                    help="ESM-2 position limit minus the 2 special tokens")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    graphs_pt = REPO_ROOT / "data" / f"graphs_{args.strategy}.pt"
    out_pt = REPO_ROOT / "data" / f"esm_{args.strategy}.pt"
    if not graphs_pt.exists():
        print(f"ERROR: missing {graphs_pt}", file=sys.stderr)
        return 2

    graphs = torch.load(graphs_pt, weights_only=False)
    print("=" * 78)
    print(f"  ESM-2 PER-RESIDUE EMBEDDINGS  |  {args.model}  |  strategy={args.strategy}")
    print("=" * 78)

    tok = AutoTokenizer.from_pretrained(args.model)
    esm = AutoModel.from_pretrained(args.model).eval()
    dim = esm.config.hidden_size
    print(f"  hidden size {dim} | layers {esm.config.num_hidden_layers} | "
          f"params {sum(p.numel() for p in esm.parameters())/1e6:.1f}M")
    print(f"  graphs {len(graphs)}\n")

    feats: dict[str, "torch.Tensor"] = {}
    stats = {"chains": 0, "gaps": 0, "truncated": 0, "unmapped": 0, "nodes": 0}
    seq_lens = []

    for n, g in enumerate(graphs, start=1):
        keys = [parse_label(lab) for lab in g.residue_labels]
        wanted = {k[0] for k in keys}
        try:
            seqs = chain_sequences(g.complex_group, wanted)
        except Exception as exc:
            print(f"  [esm] skip {g.complex_group}: {type(exc).__name__}: {exc}", flush=True)
            continue

        emb_by_key: dict[tuple, np.ndarray] = {}
        for chain_id, (seq, chain_keys, gaps) in seqs.items():
            stats["chains"] += 1
            stats["gaps"] += gaps
            seq_lens.append(len(seq))
            use_seq, use_keys = seq, chain_keys
            if len(seq) > args.max_len:
                stats["truncated"] += 1
                use_seq, use_keys = seq[:args.max_len], chain_keys[:args.max_len]

            enc = tok(use_seq, return_tensors="pt", add_special_tokens=True)
            with torch.no_grad():
                h = esm(**enc).last_hidden_state[0]      # [1 + L + 1, dim]
            # strip <cls> at 0 and <eos> at the end; row i == residue i
            h = h[1:1 + len(use_seq)].numpy()
            if h.shape[0] != len(use_keys):
                raise AssertionError(
                    f"{g.complex_group} chain {chain_id}: {h.shape[0]} embeddings for "
                    f"{len(use_keys)} residues"
                )
            for k, v in zip(use_keys, h):
                emb_by_key[k] = v

        mat = np.zeros((int(g.num_nodes), dim), dtype=np.float32)
        for i, k in enumerate(keys):
            v = emb_by_key.get(k)
            stats["nodes"] += 1
            if v is None:
                stats["unmapped"] += 1     # truncated tail, or a residue ESM never saw
            else:
                mat[i] = v
        feats[g.complex_group] = torch.tensor(mat)

        if n % 25 == 0:
            print(f"  [esm] ...{n}/{len(graphs)} complexes", flush=True)

    torch.save(feats, out_pt)

    print(f"\n  chains embedded   : {stats['chains']}")
    print(f"  sequence length   : mean {np.mean(seq_lens):.0f}, max {max(seq_lens)}")
    print(f"  numbering gaps    : {stats['gaps']} (spliced-out disordered regions)")
    print(f"  chains truncated  : {stats['truncated']} (over {args.max_len} residues)")
    print(f"  nodes             : {stats['nodes']}")
    print(f"  UNMAPPED nodes    : {stats['unmapped']} "
          f"({100*stats['unmapped']/max(1,stats['nodes']):.2f}%) -> zero vector")
    print(f"\n  saved -> {out_pt}  ({out_pt.stat().st_size/1e6:.1f} MB), dim={dim}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
