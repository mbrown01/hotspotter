"""Rank interface residues by how likely they are to be load-bearing hot spots.

This module encodes the project's central thesis in code, side by side:

  NAIVE baseline (``naive_burial_score``):  rank by buried surface area alone. This is the
      heuristic that sent Sandra to Y96 — "most buried must matter most." We keep it so we
      can always show what the obvious approach would have said.

  HOT-SPOT heuristic (``hotspot_score``):  a transparent weighted sum that adds interaction
      chemistry (salt bridges heavily), interface centrality (O-ring), conservation (when
      available), and residue identity on top of burial. Every residue also gets a plain-
      English ``reasoning`` string listing *why* it scored where it did.

IMPORTANT (honesty): these weights are hand-set and interpretable, NOT learned. That's the
whole point of Phase 1 — a sensible, explainable baseline. Phase 2 replaces this fixed
scoring with a model trained on SKEMPI's measured ΔΔG labels, and the ablation there tells
us which of these features actually carry the signal. Until then, treat the ranking as a
well-reasoned hypothesis generator, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Weights:
    """Hand-tuned, interpretable weights. Documented so a domain expert can argue with them.

    Rationale for the ordering (strongest first):
      salt bridge / disulfide  — individually load-bearing charge-charge / covalent links;
                                  removing one often costs real binding energy (the Arg95 case).
      conservation             — evolution's verdict that the position can't drift; the
                                  single best non-geometric predictor of functional importance.
      hydrogen bond            — directional, moderately strong, and specific.
      burial (dSASA)           — necessary (hot spots are buried) but NOT sufficient (lots of
                                  buried residues do no energetic work), so weighted modestly.
      centrality (O-ring)      — central residues sit in a dry, energetically favorable core.
      aromatic                 — pi-stacking / cation-pi, worth a solid bump.
      hydrophobic              — many and individually weak; small per-contact weight.
    """

    salt_bridge: float = 3.0
    disulfide: float = 4.0
    hydrogen_bond: float = 1.0
    aromatic: float = 1.5
    hydrophobic: float = 0.25
    burial: float = 1.5          # applied to normalized dSASA (0..1)
    centrality: float = 1.0      # applied to centrality (0..1)
    conservation: float = 2.0    # applied to normalized conservation (0..1), if present
    charged_bonus: float = 0.5   # identity nudge


def _minmax(col: pd.Series) -> pd.Series:
    """Scale a column to [0, 1] within this interface; all-equal -> zeros."""
    lo, hi = col.min(), col.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - lo) / (hi - lo)


def naive_burial_score(df: pd.DataFrame) -> pd.Series:
    """The baseline heuristic: buried surface area, nothing else. (Picked Y96.)"""
    if "dsasa" in df:
        return df["dsasa"].fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def hotspot_score(df: pd.DataFrame, weights: Weights | None = None) -> pd.DataFrame:
    """Add ``naive_score``, ``hotspot_score``, ranks, and ``reasoning`` to the table.

    Returns a NEW DataFrame sorted by hotspot_score (descending). Input is the merged
    per-residue feature table produced by the pipeline.
    """
    w = weights or Weights()
    out = df.copy()

    # Normalized-in-interface helpers (chemistry counts are already comparable as-is).
    norm_burial = _minmax(out["dsasa"]) if "dsasa" in out else pd.Series(0.0, index=out.index)
    centrality = out["centrality"] if "centrality" in out else pd.Series(0.0, index=out.index)
    if "conservation" in out and out["conservation"].notna().any():
        norm_cons = _minmax(out["conservation"].astype(float))
    else:
        norm_cons = pd.Series(0.0, index=out.index)

    def col(name):
        return out[name].fillna(0) if name in out else pd.Series(0.0, index=out.index)

    score = (
        w.salt_bridge * col("n_salt_bridges")
        + w.disulfide * col("n_disulfides")
        + w.hydrogen_bond * col("n_hydrogen_bonds")
        + w.aromatic * col("n_aromatic")
        + w.hydrophobic * col("n_hydrophobic")
        + w.burial * norm_burial
        + w.centrality * centrality
        + w.conservation * norm_cons
        + w.charged_bonus * col("is_charged")
    )

    out["naive_score"] = naive_burial_score(out)
    out["hotspot_score"] = score.round(3)
    out["reasoning"] = [
        _reason_row(out.loc[idx], norm_cons.loc[idx]) for idx in out.index
    ]

    out = out.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    out["hotspot_rank"] = np.arange(1, len(out) + 1)
    # Rank by the naive score too, so the two orderings can be compared directly.
    out["naive_rank"] = out["naive_score"].rank(ascending=False, method="min").astype(int)
    return out


def _reason_row(row: pd.Series, norm_cons: float) -> str:
    """Human-readable explanation of a residue's score — the interpretable payoff."""
    bits: list[str] = []
    if row.get("n_salt_bridges", 0):
        n = int(row["n_salt_bridges"])
        bits.append(f"forms {n} salt bridge{'s' if n > 1 else ''} (strong charge-charge contact)")
    if row.get("n_disulfides", 0):
        bits.append("covalent disulfide across the interface")
    if row.get("n_hydrogen_bonds", 0):
        n = int(row["n_hydrogen_bonds"])
        bits.append(f"{n} hydrogen bond{'s' if n > 1 else ''}")
    if row.get("n_aromatic", 0):
        bits.append("aromatic / pi-stacking contact")
    if row.get("n_hydrophobic", 0):
        n = int(row["n_hydrophobic"])
        bits.append(f"{n} hydrophobic contact{'s' if n > 1 else ''}")
    if row.get("dsasa") is not None and row.get("dsasa", 0) >= 20:
        bits.append(f"buries {row['dsasa']:.0f} A^2 of surface")
    if row.get("centrality", 0) >= 0.66:
        bits.append("central in the interface (O-ring core)")
    if norm_cons >= 0.66:
        bits.append("evolutionarily conserved")
    if not bits:
        bits.append("peripheral contact with no strong individual interaction")
    return "; ".join(bits)


def compare_rankings(ranked: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Side-by-side of the two orderings' top-N — the money table for the write-up.

    Shows where the chemistry-aware ranking disagrees with naive buriedness (the whole
    point: catching an Arg95 that buriedness would rank below a Y96).
    """
    cols = [c for c in ("residue", "aa", "hotspot_rank", "naive_rank",
                        "hotspot_score", "dsasa", "n_salt_bridges", "reasoning")
            if c in ranked.columns]
    top = ranked.nsmallest(top_n, "hotspot_rank")[cols]
    return top
