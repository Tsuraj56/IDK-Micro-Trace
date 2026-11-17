"""Phase P1 (C1: Future FC Prediction) Cohort and Modeling Package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from micro_trace.c1.cohort import (
        build_c1_cohorts,
        create_participant_split_80_20,
        verify_raw_mgx_files,
    )
    from micro_trace.c1.features import (
        C1FeaturePipeline,
        build_visit_trajectory_store,
        load_and_align_omics_tables,
        run_c1_feature_audit,
    )
    from micro_trace.c1.models import (
        compute_regression_metrics,
        compute_directional_accuracy,
        compute_delta_regression_metrics,
        MeanBaseline,
        PersistenceBaseline,
        ElasticNetBenchmark,
        RandomForestBenchmark,
        HistGradientBoostingBenchmark,
        ContinuousTimeEmbedding,
        MicroTraceAttn,
        MicroTraceAttnTrainer,
        run_c1_model_benchmark,
        run_c1_delta_benchmark,
    )


def __getattr__(name: str):
    if name in ("build_c1_cohorts", "create_participant_split_80_20", "verify_raw_mgx_files"):
        from micro_trace.c1 import cohort
        return getattr(cohort, name)
    if name in (
        "C1FeaturePipeline",
        "load_and_align_omics_tables",
        "build_visit_trajectory_store",
        "run_c1_feature_audit",
    ):
        from micro_trace.c1 import features
        return getattr(features, name)
    if name in (
        "compute_regression_metrics",
        "compute_directional_accuracy",
        "compute_delta_regression_metrics",
        "MeanBaseline",
        "PersistenceBaseline",
        "ElasticNetBenchmark",
        "RandomForestBenchmark",
        "HistGradientBoostingBenchmark",
        "ContinuousTimeEmbedding",
        "MicroTraceAttn",
        "MicroTraceAttnTrainer",
        "run_c1_model_benchmark",
        "run_c1_delta_benchmark",
    ):
        from micro_trace.c1 import models
        return getattr(models, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "verify_raw_mgx_files",
    "create_participant_split_80_20",
    "build_c1_cohorts",
    "C1FeaturePipeline",
    "load_and_align_omics_tables",
    "build_visit_trajectory_store",
    "run_c1_feature_audit",
    "compute_regression_metrics",
    "compute_directional_accuracy",
    "compute_delta_regression_metrics",
    "MeanBaseline",
    "PersistenceBaseline",
    "ElasticNetBenchmark",
    "RandomForestBenchmark",
    "HistGradientBoostingBenchmark",
    "ContinuousTimeEmbedding",
    "MicroTraceAttn",
    "MicroTraceAttnTrainer",
    "run_c1_model_benchmark",
    "run_c1_delta_benchmark",
]
