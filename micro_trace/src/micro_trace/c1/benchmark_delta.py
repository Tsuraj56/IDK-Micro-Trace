"""Phase P1 (C1: Future FC Prediction) - Trial 1: Target Delta Formulation Benchmark.

Target Formulation:
    Delta y = log10(FC_{t+1} + 1) - log10(FC_t + 1)
Reconstruction:
    hat{y}_{t+1} = log10(FC_t + 1) + hat{Delta y}

Evaluates across:
    - Baselines: M0 Persistence Baseline (hat{Delta y} = 0)
    - Families: ElasticNetCV, RandomForest, HistGradientBoosting,
                MicroTrace-Attn-SmoothL1, MicroTrace-Attn-L1
    - Arms: M0, M1, M2, M3, M4, M5, B3, B4
    - 5-Fold Participant-Grouped CV (dev N=371, 75 participants)
    - 20% Test Partition (N=74, 18 participants) strictly held untouched (evaluate_test=False).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from micro_trace.c1.models import run_c1_delta_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("micro_trace.c1.benchmark_delta")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Trial 1 Target Delta Formulation Benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="micro_trace/configs/c1_config.yaml",
        help="Path to c1_config.yaml",
    )
    parser.add_argument(
        "--epochs-attn",
        type=int,
        default=30,
        help="Number of epochs for MicroTrace-Attn training",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help="Strictly held out; must remain False for dev benchmark",
    )
    parser.add_argument(
        "--use-inner-val",
        action="store_true",
        default=True,
        help="Use inner participant split for MicroTrace-Attn early stopping (prevents outer CV snooping)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STARTING TRIAL 1: TARGET DELTA FORMULATION BENCHMARK")
    logger.info("Invariants: PR-03 Train-Only Fitting, 5-Fold Grouped CV, Zero Test Leakage")
    logger.info("=" * 70)

    t0 = time.time()
    summary = run_c1_delta_benchmark(
        cfg_path=args.config,
        arms=("M0", "M1", "M2", "M3", "M4", "M5", "B3", "B4"),
        families=(
            "ElasticNetCV",
            "RandomForest",
            "HistGradientBoosting",
            "MicroTrace-Attn-SmoothL1",
            "MicroTrace-Attn-L1",
        ),
        epochs_attn=args.epochs_attn,
        evaluate_test=args.evaluate_test,
        use_inner_val=args.use_inner_val,
    )
    elapsed = time.time() - t0

    logger.info("=" * 70)
    logger.info(f"TRIAL 1 BENCHMARK COMPLETED IN {elapsed:.1f}s")
    logger.info(f"Dev Pairs Evaluated: {summary['dev_pairs_count']}")
    logger.info(f"Test Partition Status: {summary['test_partition_status']}")
    logger.info(f"Best CV Model: {summary['best_cv_model']}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
