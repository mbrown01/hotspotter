# HotSpotter

### Interaction-aware hot-spot prediction for protein–protein interfaces

Given the 3D structure of a protein complex, find the binding interface and rank which
residues are **load-bearing** — the ones that, if mutated, would actually break the
interaction.

## Why

The common shortcut is to assume the most *buried* residue at an interface is the most
important one. That assumption is often wrong. A residue can be deeply buried and
contribute little, while a less-buried one forming a salt bridge does the real work.

This tool scores residues on **interaction chemistry**, not just burial, and reports a
plain-English reason for every score.

## Status

Working pipeline, 18 unit tests passing. Validated on barnase–barstar (PDB `1BRS`), a
system with known experimental answers: it recovers the textbook hot spots, and barnase
Arg87 moves from rank #38 by burial alone to rank #5 once chemistry is accounted for.

Ranking weights are hand-set and interpretable by design, not learned. A trained model
on SKEMPI binding-affinity data is scaffolded but not yet run. Evolutionary conservation
is planned, not built.

Note that `1BRS` is the complex the weights were tuned against, so this demonstrates the
pipeline is sound, not that it generalizes. Held-out validation is the next milestone.
Not a state-of-the-art claim.

## Features

Per interface residue:

- **Interaction chemistry** — salt bridges, H-bonds, hydrophobic contacts, π/aromatic stacking, disulfides
- **Burial** — buried surface area, ΔSASA (unbound→bound), relative SASA
- **Interface topology** — central vs. peripheral (O-ring), cross-interface contact count
- **Residue identity** — type, charge, size, hydrophobicity, aromaticity
- **Structure confidence** — B-factor (experimental), pLDDT / interface PAE (predicted)

Out of scope: MD-based flexibility, full electrostatics (APBS), water-mediated contacts,
protonation/pH, PTMs.

## Install & run

Windows (PowerShell), Python 3.10+:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux / Colab:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the barnase–barstar demo end-to-end:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
```

Or run the CLI on any PDB id or local file:

```powershell
.\.venv\Scripts\python.exe -m hotspotter.cli --pdb 1BRS --chains A,D
```

Outputs land in `outputs/`: a per-residue feature table, a contact list, and a text report.
