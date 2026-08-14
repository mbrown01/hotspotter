# HotSpotter

### Interaction-aware hot-spot prediction for protein–protein interfaces

**Given the 3D structure of a protein complex, find the binding interface, compute a
rich per-residue feature set, and rank which residues are most likely to be
*load-bearing* — i.e. which ones, if mutated, would actually break the interaction.**

This is a rigorous, domain-aware **portfolio project** in structural machine learning.
It is *not* claiming novelty or state-of-the-art: mature tools already predict
mutation effects on binding (FoldX, mCSM-PPI2, BeAtMuSiC, SpotOn, …). The honest
contribution here is **rigor, interpretability, and a real motivating case** — plus a
usable, centralized workflow for a task scientists currently stitch together by hand.

---

## The story that motivates this

A structural biologist (Sandra) studied how two proteins, **ARL15** and **CNNM2**,
bind. She predicted the complex (AlphaFold2/ColabFold + docking) and analyzed the
interface to decide which residue to mutate and test. A standard interface analysis
(PISA) flagged a residue that looked **heavily buried** — **Tyrosine 96 (Y96)** — and
it agreed across both of her models, so she mutated it.

**It didn't matter.** Binding was unchanged.

The residue that *actually* mattered was **Arginine 95 (Arg95)**, which forms a **salt
bridge** — a strong charge–charge interaction — across the interface. That only came out
in a later paper.

> **The lesson driving this project:** "buried" tells you *where* the interface is, not
> *which contacts are load-bearing.* The obvious heuristic (most-buried residue) sent her
> to the wrong residue. Catching the real one required reasoning about **interaction
> chemistry**. We want a tool that does better than the naive heuristic — and we can
> *prove* it does, on a case where the answer is known.

---

## What it does (two phases, one engine)

**Phase 1 — the feature-extraction pipeline (this repo, building now).**
Input a complex structure → detect the interface → compute a full per-residue feature
set → rank candidate residues with a transparent heuristic → output a clean table + a
3D interface visualization + a report. The output is designed from day one to also emit
one **model-ready feature row per residue**, so Phase 2 trains on exactly what Phase 1
produces.

**Phase 2 — the ML hot-spot scorer (later).**
Run Phase 1 across the [SKEMPI 2.0](https://life.bsc.es/pid/skempi2) database (~7,000
mutations in protein complexes with *measured* ΔΔG binding changes), join the labels, and
train a model (XGBoost baseline → GNN, since an interface is naturally a graph) to predict
whether a mutation disrupts binding. Evaluate honestly against the naive baseline and an
existing tool.

**Phase 3 — deorphanization (far-future stretch, out of scope).** Screen an orphan
protein against candidate partners. Mentioned only as future work; it's the one part that
needs real compute.

---

## Development strategy: debug on a known answer, then go live

We build and validate on **barnase–barstar (PDB `1BRS`)** — the most-studied protein
interface in existence, tiny, and saturated with measured ΔΔG mutations. Its interface is
dominated by **charged residues and salt bridges** — the exact chemistry that caught
Arg95. It's our "hello world" *and* free Phase-2 familiarization. Only once the tool
behaves correctly there do we point it at **ARL15–CNNM2** and ask: would it have flagged
Arg95 where buriedness pointed at Y96?

---

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

## Honest evaluation plan (Phase 2)

- **Split by complex** so the same complex never appears in train *and* test.
- **Handle class imbalance** (most single mutations *don't* disrupt binding).
- Benchmark vs. the **naive "most-buried" baseline** and ≥1 **existing tool**.
- **Feature-importance / ablation**: show which features actually carry signal (chemistry
  vs. conservation vs. burial), turning "I threw everything in" into an evidence-backed
  story.
- **Case test**: ARL15–CNNM2 — does the model rank Arg95 above Y96?

## References
- Danneskiold-Samsøe et al., *AlphaFold2 enables accurate deorphanization of ligands to
  single-pass receptors.* (Phase-3 north star)
- *Structural insights into regulation of CNNM–TRPM7 divalent cation uptake by the small
  GTPase ARL15* — the paper that identified Arg95. (PMID 37449820)
- Sandra Tetteh, Rutgers thesis, *Regulation of TRPM7 by CNNM2 and Interacting Partners*
  (2022).
- Moal & Fernández-Recio, *SKEMPI 2.0* — mutation ΔΔG database (Phase-2 labels).
- Cross-check tool: **LigPlot+/DIMPLOT** 2D interaction diagrams (H-bonds, hydrophobic
  contacts) for validating our contact detection.

## License
MIT — see [LICENSE](LICENSE).
