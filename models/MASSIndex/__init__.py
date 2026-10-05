"""Standalone first-version components for the MASS-Index retriever.

This package deliberately does not register a new model type yet.  It can be
unit-tested and integrated into a separate attention class without changing
the existing HiLS model, TileLang operators, or JSON configurations.
"""

from .mass_index import (
    MASSIndexConfig,
    MassTopKResult,
    MassTangentSummary,
    MultiTangentAnchorGenerator,
    build_mass_tangent_summary,
    mass_topk_reference,
    upper_envelope_scores,
)

# Imported lazily by users who have TileLang available; keeping it out of the
# package's mandatory import path preserves the pure-PyTorch unit tests.
def get_mass_attention_class():
    from .mass_attention import MASSAttention

    return MASSAttention

__all__ = [
    "MASSIndexConfig",
    "MassTopKResult",
    "MassTangentSummary",
    "MultiTangentAnchorGenerator",
    "build_mass_tangent_summary",
    "mass_topk_reference",
    "upper_envelope_scores",
    "get_mass_attention_class",
]
