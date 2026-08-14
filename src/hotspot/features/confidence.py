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

from hotspot.interface import Interface
from hotspot.io import ResidueId


def _mean_bfactor(residue) -> float:
    vals = [a.get_bfactor() for a in residue.get_atoms()]
    return float(np.mean(vals)) if vals else float("nan")


def compute_confidence_features(
    interface: Interface, is_predicted: bool = False
) -> dict[ResidueId, dict]:
    """Per-residue B-factor (experimental) or pLDDT (predicted).

    We always report the raw column mean; the column name reflects its meaning so nothing
    downstream misinterprets a pLDDT as a B-factor.
    """
    feats: dict[ResidueId, dict] = {}
    for rid, ir in interface.residues.items():
        mean_b = _mean_bfactor(ir.residue)
        if is_predicted:
            feats[rid] = {
                "plddt": round(mean_b, 1),
                "low_confidence": int(mean_b < 70),  # AF convention: <70 = low
            }
        else:
            feats[rid] = {"bfactor": round(mean_b, 2)}
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
