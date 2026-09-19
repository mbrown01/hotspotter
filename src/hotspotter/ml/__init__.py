"""The trained hot-spot scorer.

Replaces the hand-set ranking heuristic in ``hotspotter.ranking`` with a model fitted to
measured data (SKEMPI 2.0 ΔΔG labels). It reuses the feature pipeline verbatim:
``dataset.py`` runs ``hotspotter.pipeline.analyze_complex`` across SKEMPI's complexes and
joins the labels onto the matching residues, so the training table is the same per-residue
table the pipeline already emits, with a ΔΔG target column added.

Modules:
    features       feature column definitions and the mutation -> residue label collapse
    dataset        parse SKEMPI, build the labeled feature table, split BY COMPLEX
    train          XGBoost baseline and its evaluation against the naive baseline
    graph_dataset  the same table reshaped into a contact graph, used by experiments/
"""
