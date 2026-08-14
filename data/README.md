# Data

This folder is where structures and datasets live. **Contents are gitignored** (they're
large and freely re-downloadable) — only this README is tracked.

```
data/
  raw/        structures downloaded from RCSB PDB / AlphaFold DB (auto-created)
  external/   large third-party datasets, e.g. SKEMPI 2.0 (Phase 2)
  interim/    any intermediate cached artifacts
```

## Getting structures

```powershell
# by PDB id (experimental) -> data/raw/1brs.pdb
.\.venv\Scripts\python.exe scripts\fetch_structure.py --pdb 1BRS
# by UniProt accession (AlphaFold DB monomer model)
.\.venv\Scripts\python.exe scripts\fetch_structure.py --alphafold P69905
```

Downloads also happen automatically the first time you analyze a PDB id
(`analyze_complex("1BRS", ...)`), and are cached here.

## Phase-2 dataset (later)

**SKEMPI 2.0** — ~7,000 mutations in protein complexes with measured ΔΔG binding changes,
the labels for the ML scorer. Free single-file download from
<https://life.bsc.es/pid/skempi2>. Put it in `data/external/` when we reach Phase 2.

## Note on corporate networks

Downloads use the OS certificate store via `truststore`, so they work behind an
intercepting proxy (e.g. a corporate network) that would otherwise cause
`CERTIFICATE_VERIFY_FAILED`. No configuration needed.
