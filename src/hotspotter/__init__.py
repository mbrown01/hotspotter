"""hotspotter — interaction-aware hot-spot prediction for protein-protein interfaces.

Phase 1: a per-residue feature-extraction pipeline for protein complexes.

Typical use::

    from hotspotter.pipeline import analyze_complex
    result = analyze_complex("1BRS", chains=("A", "D"))
    print(result.table.head())

See ``docs/phase1_plan.md`` and ``docs/biology/`` for the how and the why.
"""

__version__ = "0.1.0"

from hotspotter.pipeline import analyze_complex  # noqa: E402  (public API convenience)

__all__ = ["analyze_complex", "__version__"]
