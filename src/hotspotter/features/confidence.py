"""Model confidence & flexibility proxies.

Two different meanings depending on where the structure came from:

  EXPERIMENTAL (X-ray/cryo-EM): the B-factor column measures how much each atom "wobbles"
      in the crystal — a cheap proxy for local flexibility. Rigid interface residues (low B)
      are cheaper to lock down on binding and are often more important.

  PREDICTED (AlphaFold/ColabFold): the same column instead stores **pLDDT** (0-100), the
      model's per-residue confidence. Low pLDDT at an interface = "don't trust this
      residue's placement," which should *lower* our confidence in its geometric features.
      PAE (predicted aligned error) additionally tells us how confident the model is about
      the *relative* placement of two residues — directly relevant to whether a predicted
      cross-interface contact is real.

Because both live in the same column, the *caller* declares which one it is
(``is_predicted``); we can't tell from coordinates alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hotspotter.interface import Interface
from hotspotter.io import ResidueId


#: Atoms an alanine substitution does NOT remove.
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


def _mean_bfactor(residue) -> float:
    vals = [a.get_bfactor() for a in residue.get_atoms()]
    return float(np.mean(vals)) if vals else float("nan")


def _sidechain_bfactor_ratio(residue) -> tuple[float, float]:
    """(mean side-chain B, side-chain B / backbone B).

    WHY THE RATIO AND NOT THE SIDE-CHAIN VALUE. What should matter for a mutation is
    side-chain plasticity -- a side chain that is already floppy costs little to delete.
    The whole-residue mean B is dominated by the backbone, which mostly reports what
    secondary structure the residue sits in, a different phenomenon. The raw side-chain B
    does not isolate it either, because the absolute B scale varies enormously between
    structures with resolution and refinement protocol. Dividing by the residue's OWN
    backbone B cancels that per-structure scale.

    Measured: the ratio is worth +0.0161 PR-AUC (p=0.0009) and the raw side-chain B
    nothing at all, and the ratio is nearly uncorrelated with the whole-residue B-factor
    (r = -0.06), so it is new information rather than a restatement.
    """
    sc = [a.get_bfactor() for a in residue.get_atoms()
          if a.get_name() not in BACKBONE_ATOMS]
    bb = [a.get_bfactor() for a in residue.get_atoms()
          if a.get_name() in BACKBONE_ATOMS]
    b_sc = float(np.mean(sc)) if sc else float("nan")
    b_bb = float(np.mean(bb)) if bb else float("nan")
    # Glycine and alanine can leave no side-chain atoms; a zero backbone B means the
    # column is unusable. Either way the honest answer is "unknown", not a filled value.
    ratio = (b_sc / b_bb) if (sc and bb and b_bb > 0) else float("nan")
    return b_sc, ratio


#: Experimental methods whose B-factor column is a real crystallographic B-factor.
XRAY_METHODS = ("x-ray", "diffraction", "electron crystallography")

#: Header values that mean "the file did not say", not "the file said something else".
#: A PDB with no EXPDTA record parses as "unknown", which anything re-saved by PyMOL,
#: PDBFixer or a docking program routinely is. Reading that as "not a crystal" blanks
#: three of the model's inputs on exactly the structures the tool is for.
UNSTATED_METHODS = frozenset({"", "unknown", "unknown method", "n/a", "na", "none"})


def _method_stated(method: str | None) -> bool:
    return method is not None and str(method).strip().lower() not in UNSTATED_METHODS


def _is_xray(method: str | None) -> bool:
    m = (method or "").lower()
    return any(k in m for k in XRAY_METHODS)


def compute_confidence_features(
    interface: Interface,
    is_predicted: bool = False,
    method: str | None = None,
    resolution: float | None = None,
) -> dict[ResidueId, dict]:
    """Per-residue B-factor (experimental) or pLDDT (predicted), plus structure quality.

    We always report the raw column mean; the column name reflects its meaning so nothing
    downstream misinterprets a pLDDT as a B-factor.

    ``method`` and ``resolution`` come from the PDB header, and both change what the
    numbers mean:

      * **An NMR entry has no crystallographic B-factor.** 1KBH carries a constant 10.00
        in that column for every atom, so reporting it as a B-factor invents a feature
        value out of a formatting convention. Non-X-ray entries get NaN instead.
      * **Resolution bounds how far the side-chain geometry can be trusted.** Above
        roughly 3 A side-chain positions are poorly determined, and the vdW term and the
        hydrogen-bond and salt-bridge counts all read exactly those atoms.

    A header that does not state a method is treated as X-ray, the same as ``method=None``;
    blanking is reserved for a file that says it came from something else. Cryo-EM is
    currently blanked along with NMR, and whether a modern cryo-EM ADP column is
    comparable to a crystallographic B-factor is an open question.
    """
    xray = _is_xray(method) if _method_stated(method) else True
    quality = {
        "resolution": float(resolution) if resolution else float("nan"),
        "is_xray": int(xray),
    }
    feats: dict[ResidueId, dict] = {}
    for rid, ir in interface.residues.items():
        mean_b = _mean_bfactor(ir.residue)
        if is_predicted:
            feats[rid] = {
                "plddt": round(mean_b, 1),
                "low_confidence": int(mean_b < 70),  # AF convention: <70 = low
                **quality,
            }
        else:
            b_sc, ratio = _sidechain_bfactor_ratio(ir.residue)
            feats[rid] = {
                "bfactor": round(mean_b, 2) if xray else float("nan"),
                "bfactor_sc": round(b_sc, 2) if (xray and b_sc == b_sc) else float("nan"),
                "bfactor_sc_bb": round(ratio, 4) if (xray and ratio == ratio) else float("nan"),
                **quality,
            }
    return feats


def load_pae(pae_json: str | Path) -> np.ndarray:
    """Load a predicted-aligned-error matrix from an AlphaFold/ColabFold JSON file.

    Returns an (N, N) array where entry [i, j] is the expected position error (A) at
    residue i when the structure is aligned on residue j. Handles both the AlphaFold DB
    schema (``predicted_aligned_error``) and the older ColabFold list-wrapped schema.
    """
    data = json.loads(Path(pae_json).read_text())
    if isinstance(data, list):  # ColabFold wraps the dict in a length-1 list
        data = data[0]
    if "predicted_aligned_error" in data:
        return np.asarray(data["predicted_aligned_error"], dtype=float)
    if "pae" in data:
        return np.asarray(data["pae"], dtype=float)
    raise ValueError(f"Unrecognized PAE JSON schema; keys were {list(data.keys())}")


def interface_pae(
    pae: np.ndarray, index_of: dict[ResidueId, int], interface: Interface
) -> dict[ResidueId, dict]:
    """Mean cross-interface PAE for each interface residue.

    For residue i on one side, average PAE(i, j) over the interface residues j on the
    OTHER side — "how confident is the model about where my partners are relative to me."
    Lower is better.

    ``index_of`` maps each residue to its row/column in the PAE matrix. Build it from the
    prediction's residue order (usually the concatenated chain sequence order). This is a
    plumbing step that depends on how the prediction was generated, so it's the caller's
    job to supply the map; we keep the math here.
    """
    feats: dict[ResidueId, dict] = {}
    a_ids = [ir.res_id for ir in interface.side_a if ir.res_id in index_of]
    b_ids = [ir.res_id for ir in interface.side_b if ir.res_id in index_of]
    for rid in interface.residues:
        if rid not in index_of:
            feats[rid] = {"interface_pae": None}
            continue
        i = index_of[rid]
        others = b_ids if rid in {r for r in a_ids} else a_ids
        if not others:
            feats[rid] = {"interface_pae": None}
            continue
        vals = [pae[i, index_of[o]] for o in others]
        feats[rid] = {"interface_pae": round(float(np.mean(vals)), 2)}
    return feats
