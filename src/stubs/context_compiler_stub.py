"""Stub hybrid context compiler for Stage 2.

# TODO(member-2): Replace with AST symbol-graph + BM25 hybrid context compiler.
"""

from __future__ import annotations

import os
from pathlib import Path
import logging

from src.interfaces import IContextCompiler
from src.schemas import RequirementContract, estimate_token_count

logger = logging.getLogger(__name__)


class ContextCompilerStub(IContextCompiler):
    """Stub implementation of ContextCompiler bounded to budget tokens."""

    def compile(
        self,
        contract: RequirementContract,
        repo: str | Path,
        budget: int = 4000
    ) -> str:
        # TODO(member-2): Implement full AST slice and BM25 retrieval
        repo_path = Path(repo).resolve()
        if not repo_path.exists():
            return f"# Repository: {repo_path}\n# Note: Repository path does not exist on disk."

        # Extract search keywords from contract
        keywords = set()
        for text in [contract.observed_behavior, contract.expected_behavior]:
            for word in text.replace(".", " ").replace("(", " ").replace(")", " ").split():
                if len(word) > 3 and word.isalnum():
                    keywords.add(word.lower())

        chunks: list[str] = [
            f"# Context for Task: {contract.task_type}",
            f"# Target Repository: {repo_path.name}",
        ]
        total_tokens = estimate_token_count("\n".join(chunks))

        # Scan code files
        candidate_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", ".venv", "__pycache__")]
            for f in files:
                if f.endswith((".py", ".js", ".ts", ".rs", ".go")):
                    candidate_files.append(Path(root) / f)

        # Prioritize files containing keywords
        scored_files = []
        for p in candidate_files:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                score = sum(1 for kw in keywords if kw in content.lower())
                scored_files.append((score, p, content))
            except Exception:
                continue

        scored_files.sort(key=lambda x: x[0], reverse=True)

        # Add files within token budget
        for score, path, content in scored_files:
            rel = path.relative_to(repo_path).as_posix()
            lines = content.splitlines()
            snippet = "\n".join(lines[:120])  # limit file snippet
            entry = f"\n--- File: {rel} ---\n{snippet}\n"
            entry_tokens = estimate_token_count(entry)
            if total_tokens + entry_tokens > budget - 100:
                break
            chunks.append(entry)
            total_tokens += entry_tokens

        compiled = "\n".join(chunks)
        logger.info("ContextCompilerStub compiled context: %d tokens (budget=%d)", total_tokens, budget)
        return compiled
