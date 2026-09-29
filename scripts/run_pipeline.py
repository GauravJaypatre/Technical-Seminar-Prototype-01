#!/usr/bin/env python3
"""CLI driver for the AERS-SLM pipeline.

Usage:
    python scripts/run_pipeline.py --repo <path> --issue <file|text> [--dry-run] [--config <path>]
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Ensure UTF-8 output encoding for terminals with limited codepages (e.g. Windows cp1252)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.config import AERSConfig, get_config
from src.pipeline import AERSPipeline
from src.schemas import FailureDiagnosis, PatchResult


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AERS-SLM: Adaptive Evidence-Routed Sequential SLM Intelligence"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the target codebase repository",
    )
    parser.add_argument(
        "--issue",
        required=True,
        help="Raw issue description string or path to issue text file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline with mock LLM inference (no GGUF model loading)",
    )
    parser.add_argument(
        "--config",
        help="Path to custom YAML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args()


def load_issue_content(issue_input: str) -> str:
    """Returns file content if input points to an existing file, else input string."""
    potential_file = Path(issue_input)
    if potential_file.exists() and potential_file.is_file():
        try:
            return potential_file.read_text(encoding="utf-8")
        except Exception:
            pass
    return issue_input


def main() -> int:
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("aers_cli")

    # Load configuration
    if args.config:
        config = AERSConfig.from_yaml(args.config)
    else:
        config = get_config()

    if args.dry_run:
        config.dry_run = True

    issue_text = load_issue_content(args.issue)
    repo_path = Path(args.repo).resolve()

    logger.info("Initializing AERS-SLM pipeline (dry_run=%s)...", config.dry_run)
    pipeline = AERSPipeline(config=config)

    print("\n" + "=" * 60)
    print("           AERS-SLM RUNTIME INITIATED")
    print(f" Target Repository : {repo_path}")
    print(f" Dry Run Mode      : {config.dry_run}")
    print(f" Context Budget    : {config.context_budget} tokens")
    print(f" Baton Hard Cap    : {config.baton_max_tokens} tokens")
    print("=" * 60 + "\n")

    result = pipeline.run(issue=issue_text, repo_path=repo_path)

    print("\n" + "=" * 60)
    if isinstance(result, PatchResult):
        print(" [SUCCESS] PATCH GENERATED AND VERIFIED")
        print(f" Attempts Required : {result.attempts}")
        print("=" * 60)
        print("\n--- Unified Diff ---")
        print(result.diff)
        print("--------------------")
        print(f"Test Execution Log: {result.test_log}")
        return 0
    elif isinstance(result, FailureDiagnosis):
        print(" [FAILED] PIPELINE CONCLUDED WITH FAILURE DIAGNOSIS")
        print(f" Failed Stage : {result.stage}")
        print(f" Reason       : {result.reason}")
        print("=" * 60)
        print("\n--- Final Evidence Baton ---")
        print(result.baton.to_json(indent=2))
        return 1
    else:
        print(" [UNKNOWN] Unexpected pipeline result type.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
