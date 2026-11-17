"""
Remote Google Drive Cleanup Script for Phase P1 (C1) Reset.
Runs inside Colab to archive and clean stale C1 files while strictly preserving raw data and P0 feasibility artifacts.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/Micro-Trace")
RAW_DATA_DIR = DRIVE_ROOT / "micro-trace-data" / "raw"
FEASIBILITY_DIR = DRIVE_ROOT / "micro_trace" / "reports" / "feasibility"

ARCHIVE_DIR = DRIVE_ROOT / "_archive" / f"c1_reset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

# Targets to clean on Google Drive
CLEAN_TARGETS = [
    DRIVE_ROOT / "micro_trace" / "src" / "micro_trace" / "c1",
    DRIVE_ROOT / "micro_trace" / "reports" / "c1",
    DRIVE_ROOT / "micro_trace" / "tests" / "test_c1_cohort.py",
    DRIVE_ROOT / "micro_trace" / "tests" / "test_c1_features.py",
    DRIVE_ROOT / "micro_trace" / "tests" / "test_c1_models.py",
    DRIVE_ROOT / "micro_trace" / "tests" / "test_c1_evaluation.py",
    DRIVE_ROOT / "micro_trace" / "configs" / "c1_config.yaml",
    DRIVE_ROOT / "micro_trace" / "notebooks",
    DRIVE_ROOT / "micro-trace-data" / "interim" / "c1",
    DRIVE_ROOT / "micro-trace-data" / "processed" / "c1",
]


def run_cleanup() -> int:
    print(f"[Drive Cleanup] Checking Google Drive root at: {DRIVE_ROOT}")
    if not DRIVE_ROOT.exists():
        print(f"[Drive Cleanup] ERROR: Drive root {DRIVE_ROOT} does not exist. Is Google Drive mounted?")
        return 1

    # Invariant: Raw dataset must be preserved
    if not RAW_DATA_DIR.exists():
        print(f"[Drive Cleanup] WARNING: Raw data directory {RAW_DATA_DIR} not found.")
    else:
        raw_files = list(RAW_DATA_DIR.glob("**/*"))
        print(f"[Drive Cleanup] PRESERVED: Raw dataset intact at {RAW_DATA_DIR} ({len(raw_files)} files).")

    # Invariant: Feasibility reports preserved
    if FEASIBILITY_DIR.exists():
        p0_files = list(FEASIBILITY_DIR.glob("**/*"))
        print(f"[Drive Cleanup] PRESERVED: Phase P0 feasibility intact at {FEASIBILITY_DIR} ({len(p0_files)} files).")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Drive Cleanup] Archive directory created at: {ARCHIVE_DIR}")

    cleaned_count = 0
    for target in CLEAN_TARGETS:
        if not target.exists():
            continue
        rel = target.relative_to(DRIVE_ROOT)
        dest = ARCHIVE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"[Drive Cleanup] Archiving {target} -> {dest}")
        if target.is_dir():
            shutil.copytree(target, dest, dirs_exist_ok=True)
            shutil.rmtree(target)
        else:
            shutil.copy2(target, dest)
            target.unlink()
        cleaned_count += 1

    print(f"[Drive Cleanup] SUCCESS: {cleaned_count} targets archived and cleaned.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_cleanup())
