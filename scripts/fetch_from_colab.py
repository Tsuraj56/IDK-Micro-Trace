"""
MICRO-TRACE: Fetch Generated Reports & Artifacts from Colab / Google Drive
Usage: python scripts/fetch_from_colab.py [--session SESSION_ID]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[fetch_from_colab] %(levelname)s: %(message)s")
logger = logging.getLogger("fetch_from_colab")

ROOT_DIR = Path(__file__).resolve().parent.parent
MICRO_TRACE_DIR = ROOT_DIR / "micro_trace"
REPORTS_DIR = MICRO_TRACE_DIR / "reports" / "c1"

# Remote artifacts to download
TARGET_ARTIFACTS = [
    "c1_cohort_consecutive_0d.csv",
    "c1_cohort_consecutive_le14d.csv",
    "c1_cohort_all_horizons_0d.csv",
    "split_audit_80_20.csv",
    "c1_cohort_summary.json",
    "c1_feature_audit.json",
    "c1_guild_definitions.csv",
    "c1_model_benchmark_results.csv",
    "c1_model_summary.json",
    "c1_bootstrap_comparisons.csv",
    "c1_sensitivity_analysis.csv",
    "frozen_c1_best_model.pkl",
]


def fetch_via_colab_cli(session_id: str | None = None) -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Target local reports directory: %s", REPORTS_DIR)

    cmd = ["colab", "download"]
    if session_id:
        cmd.extend(["--session", session_id])

    fetched = 0
    for artifact in TARGET_ARTIFACTS:
        remote_path = f"/content/drive/MyDrive/Micro-Trace/micro_trace/reports/c1/{artifact}"
        local_path = REPORTS_DIR / artifact
        logger.info("Checking remote %s...", remote_path)
        res = subprocess.run(cmd + [remote_path, str(local_path)], capture_output=True, text=True)
        if res.returncode == 0 and local_path.exists():
            logger.info("Successfully fetched %s (%d bytes)", artifact, local_path.stat().st_size)
            fetched += 1
        else:
            logger.debug("Artifact %s not found remotely or download skipped.", artifact)

    logger.info("Fetch complete. Total retrieved: %d artifacts.", fetched)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch generated C1 artifacts from Colab to local reports")
    parser.add_argument("--session", default=None, help="Colab session ID")
    args = parser.parse_args()

    return fetch_via_colab_cli(session_id=args.session)


if __name__ == "__main__":
    sys.exit(main())
