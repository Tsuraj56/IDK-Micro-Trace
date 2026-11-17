"""Unit and leakage tests for Phase P3 (C3: Cross-Conformal CV+ & Selective Abstention)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from micro_trace.c3.conformal import (
    compute_interval_metrics,
    conformal_quantile,
    evaluate_uncertainty_systems_cv,
    participant_cluster_conformal_quantile,
)


def _make_synthetic_c3_folds(n_participants: int = 50, visits_per_p: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    X_list = []
    for p_idx in range(n_participants):
        pid = f"P{p_idx:03d}"
        diag = ["CD", "UC", "nonIBD"][p_idx % 3]
        fold = p_idx % 5
        for v in range(visits_per_p):
            x = rng.normal(0, 1, size=6)
            # Heteroscedastic noise: larger |x[0]| -> larger uncertainty
            noise_scale = 0.15 + 0.35 * abs(x[0])
            y = 1.5 + 0.6 * x[1] - 0.3 * x[2] + rng.normal(0, noise_scale)
            fc = float(max(5.0, 10.0 ** y - 1.0))
            X_list.append(x)
            rows.append(
                {
                    "participant_id": pid,
                    "cv_fold": fold,
                    "diagnosis": diag,
                    "y": y,
                    "future_fc": fc,
                    "index_fc_same_day": fc,
                    "elapsed_days": float(20 + 15 * v),
                }
            )
    df = pd.DataFrame(rows)
    X = np.vstack(X_list)

    fold_data = []
    for fold in range(5):
        tr_mask = (df["cv_fold"] != fold).values
        va_mask = (df["cv_fold"] == fold).values
        fold_data.append(
            {
                "fold": fold,
                "X_tr": X[tr_mask],
                "y_tr": df.loc[tr_mask, "y"].values,
                "p_tr": df.loc[tr_mask, "participant_id"].values,
                "X_va": X[va_mask],
                "y_va": df.loc[va_mask, "y"].values,
                "p_va": df.loc[va_mask, "participant_id"].values,
                "meta_va": df.loc[va_mask, ["diagnosis", "index_fc_same_day", "future_fc", "elapsed_days"]].reset_index(drop=True),
            }
        )
    return fold_data


def test_conformal_quantile_exactness_and_empty_raises():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q = conformal_quantile(scores, alpha=0.10)
    assert q == pytest.approx(0.9)

    q_clust = participant_cluster_conformal_quantile(
        scores, np.array(["A", "A", "A", "B", "B", "C", "D", "E", "F"]), alpha=0.10
    )
    assert 0.1 <= q_clust <= 0.9

    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), alpha=0.10)


def test_compute_interval_metrics_exactness():
    y_t = np.array([1.0, 2.0, 3.0, 4.0])
    y_p = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.5, 1.5, 3.5, 3.5])
    hi = np.array([1.5, 2.5, 4.5, 4.5])
    m = compute_interval_metrics(y_t, y_p, lo, hi)
    assert m["coverage"] == pytest.approx(0.75)
    assert m["mean_width"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)


def test_evaluate_uncertainty_systems_cv_u0_to_u3_and_subgroups():
    fold_data = _make_synthetic_c3_folds(n_participants=40, visits_per_p=4, seed=42)
    res = evaluate_uncertainty_systems_cv(fold_data, alpha=0.10, seed=42)

    oof = res["oof_df"]
    sys_sum = res["system_summary"]
    rc = res["risk_coverage_df"]
    sub = res["subgroup_audit_df"]

    assert len(oof) == 160
    # U0 has constant width per fold; U2 has adaptive patient-specific width (std > 0)
    assert sys_sum["U2_locally_weighted_cv_plus"]["width_std"] > 0.01
    # Coverage should be reasonably close to 90% nominal on synthetic data
    assert 0.80 <= sys_sum["U0_homoscedastic_conformal"]["coverage"] <= 0.99
    assert 0.80 <= sys_sum["U2_locally_weighted_cv_plus"]["coverage"] <= 0.99

    # Risk-coverage curve: acceptance decreases monotonically
    assert len(rc) == 6
    assert rc["observed_acceptance"].iloc[0] == pytest.approx(1.0)
    assert rc["observed_acceptance"].iloc[-1] < rc["observed_acceptance"].iloc[0]

    # Subgroup audit covers diagnosis, fc_range, and horizon
    subgroups = set(sub["subgroup"].unique())
    assert "overall" in subgroups
    assert "diagnosis=CD" in subgroups
    assert "horizon=short_le35d" in subgroups


def test_u2_no_undercoverage_in_high_dimensions():
    """Verify OOF residual scaler prevents in-sample overfitting when p > n."""
    rng = np.random.default_rng(7)
    n_participants = 30
    visits_per_p = 4
    n_features = 200  # p > n per fold (~96 train samples)
    fold_data = []
    X = rng.normal(0, 1, size=(n_participants * visits_per_p, n_features))
    y = 1.5 + 0.5 * X[:, 0] + rng.normal(0, 0.3, size=len(X))
    pids = np.repeat([f"P{i:03d}" for i in range(n_participants)], visits_per_p)
    folds = np.repeat([i % 5 for i in range(n_participants)], visits_per_p)

    for fold in range(5):
        tr_mask = folds != fold
        va_mask = folds == fold
        fold_data.append(
            {
                "fold": fold,
                "X_tr": X[tr_mask],
                "y_tr": y[tr_mask],
                "p_tr": pids[tr_mask],
                "X_va": X[va_mask],
                "y_va": y[va_mask],
                "p_va": pids[va_mask],
            }
        )

    res = evaluate_uncertainty_systems_cv(fold_data, alpha=0.10, seed=42)
    u2_cov = res["system_summary"]["U2_locally_weighted_cv_plus"]["coverage"]
    assert u2_cov >= 0.85, f"U2 under-covered in p > n regime: {u2_cov:.3f}"
