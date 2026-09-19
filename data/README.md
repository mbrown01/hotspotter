# Data

Everything in this folder is a build product or a third-party download, so **the contents
are gitignored** — only this README is tracked. Nothing here is required to read the code;
it is required to re-run it.

```
data/
  skempi_v2.csv               SKEMPI 2.0, downloaded (see below)
  skempi_features_full.csv    built by scripts/build_features.py — the training table
  split_alanine.json          the fixed cross-validation folds used by scripts/evaluate.py
  raw/                        structures fetched from RCSB PDB / AlphaFold DB (auto-created)
```

## SKEMPI 2.0

~7,000 mutations in protein complexes with measured ΔΔG binding changes — the labels.
Free single-file download from <https://life.bsc.es/pid/skempi2>; save it as
`data/skempi_v2.csv`, which is where every script looks for it.

## Structures

```bash
python scripts/fetch_structure.py --pdb 1BRS          # -> data/raw/1brs.pdb
python scripts/fetch_structure.py --alphafold P69905  # AlphaFold DB monomer model
```

Downloads also happen automatically the first time a PDB id is analyzed
(`analyze_complex("1BRS", ...)`) and are cached in `data/raw/`.

## Corporate networks

Downloads use the OS certificate store via `truststore`, so they work behind an
intercepting proxy that would otherwise cause `CERTIFICATE_VERIFY_FAILED`. No configuration
needed.
