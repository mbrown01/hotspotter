# Data

```
skempi_features.csv    per-residue features + ΔΔG labels, built by scripts/build_features.py
split_alanine.json     the fixed cross-validation folds used by scripts/evaluate.py
chain_sequences.json   chain sequences used by scripts/audit_splits.py to cluster by identity
```

Labels are derived from SKEMPI 2.0 (Jankauskaite et al., *Bioinformatics* 2019,
<https://life.bsc.es/pid/skempi2>). Rebuilding the table needs the original
`skempi_v2.csv` saved here; structures are downloaded from RCSB into `data/raw/` on first use.
