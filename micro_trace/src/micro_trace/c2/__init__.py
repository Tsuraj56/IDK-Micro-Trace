"""Phase P2 (C2: Reproducible Taxonomic, Functional, Guild & Cross-Omic Signatures)."""

from micro_trace.c2.lme_signatures import (
    build_c2_current_fc_cohort,
    build_c2_future_fc_cohort,
    fit_lme_single_feature,
    run_5fold_stability_selection,
    run_lme_family_scan,
    run_subtype_sensitivity,
)
from micro_trace.c2.cross_omic_attention import (
    CrossOmicAttentionNet,
    train_and_extract_cross_omic_attention,
)

__all__ = [
    "build_c2_current_fc_cohort",
    "build_c2_future_fc_cohort",
    "fit_lme_single_feature",
    "run_lme_family_scan",
    "run_5fold_stability_selection",
    "run_subtype_sensitivity",
    "CrossOmicAttentionNet",
    "train_and_extract_cross_omic_attention",
]
