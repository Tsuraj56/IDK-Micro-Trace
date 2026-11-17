"""
MICRO-TRACE: Sync Local Code & Configs to Google Drive / Colab
Usage: python scripts/sync_to_colab.py [--session SESSION_ID]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[sync_to_colab] %(levelname)s: %(message)s")
logger = logging.getLogger("sync_to_colab")

ROOT_DIR = Path(__file__).resolve().parent.parent
MICRO_TRACE_DIR = ROOT_DIR / "micro_trace"

# Directories and files to sync
SYNC_PATTERNS = [
    "configs/**/*.yaml",
    "src/micro_trace/**/*.py",
    "tests/**/*.py",
    "reports/c1/*",
    "pyproject.toml",
]


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect_local_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SYNC_PATTERNS:
        for p in MICRO_TRACE_DIR.glob(pattern):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(p)
    return sorted(files)


def sync_via_colab_cli(files: list[Path], session_id: str | None = None) -> bool:
    """Sync files to active Colab session via colab CLI."""
    logger.info("Attempting sync via colab CLI...")
    cmd = ["colab", "upload"]
    if session_id:
        cmd.extend(["--session", session_id])

    for f in files:
        rel = f.relative_to(MICRO_TRACE_DIR)
        remote_path = f"/content/drive/MyDrive/Micro-Trace/micro_trace/{rel.as_posix()}"
        logger.info("Uploading %s -> %s", rel, remote_path)
        res = subprocess.run(cmd + [str(f), remote_path], capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning("colab upload failed for %s: %s", rel, res.stderr.strip())
            return False
    return True


def generate_manifest(files: list[Path]) -> dict[str, str]:
    manifest = {}
    for f in files:
        rel = str(f.relative_to(MICRO_TRACE_DIR).as_posix())
        manifest[rel] = compute_file_hash(f)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync local micro_trace code to Colab/Google Drive")
    parser.add_argument("--session", default=None, help="Colab session ID")
    parser.add_argument("--manifest-only", action="store_true", help="Generate sync manifest only")
    args = parser.parse_args()

    files = collect_local_files()
    logger.info("Found %d local files to sync.", len(files))

    manifest = generate_manifest(files)
    manifest_path = ROOT_DIR / "scripts" / "sync_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved sync manifest to %s (%d files)", manifest_path, len(manifest))

    if args.manifest_only:
        return 0

    success = sync_via_colab_cli(files, session_id=args.session)
    if not success:
        logger.info(
            "Note: Colab CLI sync was not completed (e.g. no active session). "
            "Files are ready and cataloged in %s.",
            manifest_path,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
