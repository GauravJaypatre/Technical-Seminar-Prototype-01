#!/usr/bin/env python3
"""Script to download Qwen2.5-Coder-3B-Instruct Q4_K_M GGUF model weights.

Fetches the ~1.95 GB model file into models/ using direct chunked HTTP streaming.
Skips downloading if the file is already present.
"""

import argparse
import logging
from pathlib import Path
import sys
import time

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

    direct_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    logger.info("Starting direct download of %s from %s...", filename, direct_url)

    try:
        import requests
        resp = requests.get(direct_url, stream=True, timeout=60)
        resp.raise_for_status()

        total_bytes = int(resp.headers.get("content-length", 0))
        total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
        logger.info("Target file size: %.1f MB", total_mb)

        temp_path = dest_path.with_suffix(".tmp")
        downloaded_bytes = 0
        last_log_time = time.time()
        start_time = time.time()

        with open(temp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):  # 4 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    now = time.time()
                    if now - last_log_time >= 5.0 or downloaded_bytes == total_bytes:
                        downloaded_mb = downloaded_bytes / (1024 * 1024)
                        pct = (downloaded_bytes / total_bytes * 100) if total_bytes else 0
                        elapsed = now - start_time
                        speed_mb = (downloaded_mb / elapsed) if elapsed > 0 else 0
                        logger.info(
                            "Progress: %.1f / %.1f MB (%.1f%%) - Speed: %.1f MB/s",
                            downloaded_mb,
                            total_mb,
                            pct,
                            speed_mb,
                        )
                        last_log_time = now

        temp_path.replace(dest_path)
        final_size_mb = dest_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Successfully downloaded model to %s (%.1f MB in %.1f seconds)",
            dest_path,
            final_size_mb,
            time.time() - start_time,
        )
        return dest_path

    except Exception as e:
        logger.warning("Direct streaming download encountered error: %s. Attempting hf_hub_download fallback.", e)
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest_path.parent),
        )
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
