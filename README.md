# HotSpotter

Predicts which residues at a protein–protein interface are **load-bearing** — the hot spots
whose mutation collapses binding affinity. Ranks interface residues from a single structure
using XGBoost over 25 engineered geometric and chemical features, trained on SKEMPI 2.0.

## Results

1,540 alanine-scanned residues across 172 complexes (SKEMPI 2.0, ΔΔG ≥ 2.0 kcal/mol = hot).
Cross-validation is **grouped by complex**, so no residue is scored by a model that has seen
another residue from the same interface. 5 folds × 5 seeds.

| | Naive (buried area) | HotSpotter |
|---|---|---|
| PR-AUC | 0.3974 | **0.5246** |
| ROC-AUC | 0.6647 | **0.7779** |
| ROC-AUC, literature definition <sup>1</sup> | 0.7045 | **0.8514** |
| Top-1 hit rate <sup>2</sup> | 60.0% | **76.4%** |
| Top-3 hit rate | 80.0% | **91.1%** |

<sup>1</sup> hot > 2.0 kcal/mol, non-hot < 0.4, grey zone dropped — the filtering used by published methods.
<sup>2</sup> fraction of complexes whose top-*k* ranked residues contain a true hot spot.

**3M62 (Ufd2–Rad23).** Tyr97 is the strongest hot spot in the interface (ΔΔG 4.48 kcal/mol)
but only the 21st most buried of 33 interface residues. HotSpotter ranks it **#1**. The
second-most-buried residue, Phe9, is not a hot spot at all (ΔΔG 0.99).

Measured ceiling: residues that are near-identical across all 25 features still differ in
ΔΔG by 1.25 kcal/mol on average — 2.5× the experimental error — so the remaining error is
information absent from a single bound structure, not a modelling shortfall.
`scripts/diagnose_ceiling.py` reproduces that analysis.

## Usage

```bash
pip install -e .

python scripts/run_demo.py                                  # end-to-end on barnase–barstar
python scripts/evaluate.py                                  # reproduce the table above
python scripts/infer_xgb.py --pdb 3M62 --chains A,B         # rank one complex
python scripts/viz.py --pdb 3M62 --chains A,B \
    --preds predictions/3m62_xgb_predictions.csv --out 3m62.html
```

Rebuilding the training table from SKEMPI (`scripts/build_features.py`) downloads several
hundred structures and takes hours; the committed scripts run against `data/`.

`src/hotspotter/` is the library, `scripts/` the pipeline, `experiments/` the approaches
that were tested and rejected — a GATv2 graph network, ESM-2 embeddings, and
implicit-solvent electrostatics. None beat the tabular model.
