#!/usr/bin/env python3
"""Script to download Qwen2.5-Coder-3B-Instruct Q4_K_M GGUF model weights.

Uses huggingface_hub to fetch the ~1.95 GB model file into models/.
Skips downloading if the file is already present.
"""

import argparse
import logging
from pathlib import Path
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
DEFAULT_DEST_DIR = Path("models")


def download_model(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    dest_dir: Path | str = DEFAULT_DEST_DIR,
    force: bool = False,
) -> Path:
    """Download the GGUF model file if not already present."""
    dest_path = Path(dest_dir) / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Model file already exists at %s (%.1f MB). Skipping download.",
            dest_path,
            size_mb,
        )
        return dest_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed! Run: pip install huggingface_hub"
        )
        sys.exit(1)

    logger.info(
        "Downloading %s from %s to %s...",
        filename,
        repo_id,
        dest_path.parent,
    )

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_path.parent),
    )

    logger.info("Successfully downloaded model to %s", downloaded)
    return Path(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-Coder-3B Q4_K_M GGUF")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repository ID")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="GGUF weight filename")
    parser.add_argument("--dest", default=str(DEFAULT_DEST_DIR), help="Destination directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    args = parser.parse_args()

    download_model(
        repo_id=args.repo_id,
        filename=args.filename,
        dest_dir=Path(args.dest),
        force=args.force,
    )


if __name__ == "__main__":
    main()
