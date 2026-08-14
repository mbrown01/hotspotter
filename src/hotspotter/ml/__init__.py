"""Phase 2 — the ML hot-spot scorer.

Replaces Phase 1's hand-set ranking heuristic with a model trained on measured data
(SKEMPI 2.0 ΔΔG labels). Reuses the Phase-1 pipeline verbatim: ``dataset.py`` runs
``hotspotter.pipeline.analyze_complex`` across SKEMPI's complexes and joins the labels; the
result is the same per-residue table Phase 1 already emits, now with a ΔΔG target column.

Modules:
    dataset   parse SKEMPI, build the labeled feature table, split BY COMPLEX
    train     XGBoost baseline (+ evaluation vs. the naive baseline, + feature importance)

STATUS: scaffolded with real math/parsing and a documented training flow, but NOT yet run
end-to-end (needs the SKEMPI download and the `ml` optional deps). Verify SKEMPI column
names on first use — the file's header has drifted between releases.
"""
