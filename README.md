# HotSpotter

### Interaction-aware hot-spot prediction for protein–protein interfaces

**Given the 3D structure of a protein complex, find the binding interface, compute a
rich per-residue feature set, and rank which residues are most likely to be
*load-bearing* — i.e. which ones, if mutated, would actually break the interaction.**

## The feature set

| Group | Features | Cost | Status |
|---|---|---|---|
| **Interaction chemistry** | salt bridges, H-bonds, hydrophobic contacts, π/aromatic stacking, disulfides | cheap | ✅ core |
| **Burial / accessibility** | buried surface area (BSA), ΔSASA (unbound→bound), relative SASA | cheap | ✅ core |
| **Interface topology** | central vs. peripheral (O-ring), cross-interface contact count, local packing | cheap | ✅ core |
| **Residue identity / physicochem** | aa type, charge, size, hydrophobicity, aromaticity, flexibility propensity | cheap | ✅ core |
| **Prediction confidence** | per-residue pLDDT, interface PAE (predicted structures only) | cheap | ✅ core |
| **Flexibility proxy** | crystallographic B-factor (experimental structures) | cheap | ✅ core |
| **Evolutionary conservation** | per-residue conservation from an MSA / ConSurf-DB | higher effort | 🔜 planned (highest-signal after chemistry — *don't skip*) |

**Explicitly out of scope** (known but not built): MD-based flexibility, full
electrostatics surfaces (APBS), water-mediated contacts, protonation/pH, PTMs.

See [`docs/biology/`](docs/biology/) for the structural-biochemistry explainers written
alongside the code, and [`docs/features_glossary.md`](docs/biology/03_features_glossary.md)
for exact definitions and geometric cutoffs.

---

## Install & run

Windows (PowerShell), Python 3.10+:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# optional native SASA backend (needs a C compiler on Windows; a pure-Python
# Biopython fallback is used automatically if this isn't installed):
# .\.venv\Scripts\python.exe -m pip install freesasa
```

macOS / Linux / Colab:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the barnase–barstar demo end-to-end (downloads `1BRS`, detects the interface,
computes features, prints the ranked table):

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
# or, via the CLI, on any PDB id or local file:
.\.venv\Scripts\python.exe -m hotspotter.cli --pdb 1BRS --chains A,D
```

---

## Repo layout

```
src/hotspotter/            the pipeline (importable package)
  io.py                 load/clean structures (PDB + mmCIF)
  interface.py          detect interface residues (contact- and SASA-based)
  features/             one module per feature group
  ranking.py            transparent heuristic scoring (+ human-readable reasoning)
  pipeline.py           orchestrates parse -> interface -> features -> table
  report.py / viz.py    exports and 3D visualization
  cli.py                command-line entry point
scripts/                fetch_structure.py, run_demo.py
docs/biology/           structural-biochemistry explainers (learn as we build)
docs/                   phase-1 plan, roadmap
notebooks/              interactive walkthroughs
tests/                  geometry & interface unit tests
data/ , outputs/        (gitignored) downloaded structures & generated reports
```


