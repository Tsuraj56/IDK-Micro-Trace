"""Phase P3 (C3: Uncertainty-Aware Prediction, Cross-Conformal CV+ & Selective Abstention)."""

from micro_trace.c3.conformal import (
    compute_interval_metrics,
    compute_subgroup_calibration_audit,
    conformal_quantile,
    evaluate_uncertainty_systems_cv,
    participant_cluster_conformal_quantile,
)

__all__ = [
    "conformal_quantile",
    "participant_cluster_conformal_quantile",
    "compute_interval_metrics",
    "compute_subgroup_calibration_audit",
    "evaluate_uncertainty_systems_cv",
]
