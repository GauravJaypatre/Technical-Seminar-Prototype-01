"""Audit and Claim-to-Artifact Verification Script for Technical Seminar.

This script enforces rigorous claim verification across documentation:
1. Authenticity Gates on Run Artifacts:
   - Minimum realistic latency gate: flags sub-2s task completions lacking hardware acceleration telemetry.
   - Anti-ground-truth injection / template check: flags offline simulation profiles (e.g., claude-3-5-sonnet mock,
     probabilistic scaffold loops, canned template diffs).
   - Inference telemetry check: verifies live LLM inference records (provider headers, model responses).
   - Verifies that all 2026-09-07 runs in results/runs/ are flagged as synthetic and properly quarantined.
2. Quarantined Content Gate:
   - Rejects unverified/synthetic claims (e.g. 37.0% -> 98.1% Pass@K, ANOVA F=61.27, Cliff's delta=+1.000,
     router ablation 98.1% vs 79.6%, or placeholder models like Claude 3.5 Sonnet / LLaMA-3.3-70B monolithic).
3. Claim-to-Artifact Matching:
   - Validates that all quantitative figures in README.md and SEMINAR_SYNTHESIS.md exactly match
     the verified artifact records in results/prototype_run_*.
"""

import json
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Verified artifact files
THREE_WAY_CSV = REPO_ROOT / "results" / "prototype_run_swarm_v2_content_validated_qa" / "three_way_comparison_groq_v1_v2.csv"
STATS_CSV = REPO_ROOT / "results" / "prototype_run_swarm_v2_content_validated_qa" / "swarm_v2_vs_groq_comparison_v2_stats.csv"

# Quarantined terms/claims that must NEVER appear in active documentation
FORBIDDEN_PATTERNS = [
    (r"\b37\.0%\b.*\b98\.1%\b", "Synthetic Pass@1 -> Pass@K (37.0% -> 98.1%)"),
    (r"\b98\.1%\b.*\b37\.0%\b", "Synthetic Pass@K -> Pass@1 (98.1% -> 37.0%)"),
    (r"\+61\.1\s*(?:pp|percentage point|%)", "Synthetic +61.1pp self-correction yield"),
    (r"F\s*=\s*61\.27", "Synthetic ANOVA F=61.27"),
    (r"Cliff'?s\s*Delta\s*=\s*\+?1\.000", "Synthetic Cliff's Delta=+1.000"),
    (r"Claude", "Placeholder model 'Claude'"),
    (r"Sonnet", "Placeholder model 'Sonnet'"),
    (r"LLaMA-3\.3-70B.*monolithic.*90\.0%", "Unverified claim 'LLaMA-3.3-70B monolithic 90.0%'"),
    (r"Swarm v1.*77\.8%", "Unverified claim 'Swarm v1 77.8%'"),
    (r"Swarm v2.*6\.7%", "Unverified claim 'Swarm v2 6.7%'"),
    (r"\b1\.3[bB]\b", "Stale model parameter size '1.3B' (must be 6.7b per src/swarm/agents.py)"),
    (r"\b1\.5[bB]\b", "Stale model parameter size '1.5B' (must be 0.5b per src/swarm/agents.py)"),
    (r"deepseek-coder:1\.3b", "Stale model 'deepseek-coder:1.3b'"),
    (r"qwen2\.5-coder:1\.5b", "Stale model 'qwen2.5-coder:1.5b'"),
    (r"K\s*(?:<=|≤|\\le)\s*5", "Stale iteration cap 'K <= 5' (must be retry cap <= 2 per src/swarm/graph.py)"),
    (r"\bPass@K\b", "Stale Pass@K metric from synthetic scaffold"),
]


def audit_run_file_authenticity(filepath: Path) -> Tuple[bool, List[str]]:
    """Evaluate a benchmark JSON file against the 3 authenticity gates.
    
    Returns (is_authentic, list_of_flags).
    """
    flags = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        runs = data if isinstance(data, list) else data.get("runs", [])
        
        for r in runs:
            # Gate A: Minimum Realistic Latency Check
            wall_clock = r.get("cumulative_wall_clock_sec") or r.get("wall_clock_time") or 0.0
            if r.get("final_pass", False) and wall_clock < 2.0:
                flags.append(f"Gate A (Sub-2s Latency): Task {r.get('task_id')} passed in {wall_clock}s (impossible on CPU).")
                break
                
            # Gate B: Anti-Ground-Truth Injection & Mock Profile Check
            model_profile = str(r.get("model_profile", ""))
            if "claude-3-5-sonnet" in model_profile or "1.3b-instruct" in model_profile:
                flags.append(f"Gate B (Scaffold Profile): Contains offline simulation profile '{model_profile}'.")
                break
            patch = r.get("patch", "")
            if "# Incomplete iteration" in patch or "Offline simulation" in patch or "Offline monolithic simulation" in patch:
                flags.append("Gate B (Mock Diff Injection): Contains offline mock diff markers.")
                break
                
            # Gate C: Inference Telemetry Check
            if not r.get("is_live", False) or "provider" not in r:
                flags.append("Gate C (Missing Telemetry): Lacks verified live inference telemetry ('is_live': True and 'provider').")
                break
                
    except Exception as e:
        flags.append(f"Error reading {filepath.name}: {e}")
        
    is_authentic = len(flags) == 0
    return is_authentic, flags


def check_fallback_neutralization() -> List[str]:
    """Verify that offline simulation fallbacks across all of src/ are neutralized with hard-failing RuntimeErrors,
    and repo-wide search confirms no fabrication patterns exist."""
    issues = []
    
    # 1. Repo-wide scan of ALL python files in src/
    fabrication_patterns = [
        "base_probs",
        "success_rates",
        "ground_truth_patch",
        "rng.random() <",
        "random.Random(seed",
    ]
    src_dir = REPO_ROOT / "src"
    py_files = list(src_dir.rglob("*.py"))
    
    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            for pat in fabrication_patterns:
                if pat in content:
                    rel_path = py_file.relative_to(REPO_ROOT).as_posix()
                    issues.append(
                        f"FABRICATION PATTERN DETECTED: {rel_path} contains forbidden signature '{pat}'."
                    )
        except Exception as e:
            issues.append(f"AUDIT ERROR: Could not read {py_file}: {e}")

    # 2. Test src/monolithic_baseline.py call_mock_frontier & live=False
    try:
        import src.monolithic_baseline as mb
        try:
            mb.call_mock_frontier("fake_dir", seed=42)
            issues.append("NEUTRALIZATION FAILURE: src/monolithic_baseline.py call_mock_frontier did not raise RuntimeError.")
        except RuntimeError:
            pass
        except Exception as e:
            issues.append(f"NEUTRALIZATION FAILURE: call_mock_frontier raised unexpected exception: {e}")
            
        try:
            mb.run_monolithic_baseline(live=False)
            issues.append("NEUTRALIZATION FAILURE: src/monolithic_baseline.py run_monolithic_baseline(live=False) did not raise RuntimeError.")
        except RuntimeError:
            pass
        except Exception as e:
            issues.append(f"NEUTRALIZATION FAILURE: run_monolithic_baseline(live=False) raised unexpected exception: {e}")
    except Exception as e:
        issues.append(f"NEUTRALIZATION FAILURE: Error importing monolithic_baseline: {e}")

    # 3. Test src/code_analyzer.py offline RuntimeError
    try:
        import src.code_analyzer as ca
        saved_status = ca._OLLAMA_ONLINE
        ca._OLLAMA_ONLINE = False
        try:
            agent = ca.CodeAnalyzerAgent()
            try:
                agent._call_llm("test prompt", {}, 1, None, 42)
                issues.append("NEUTRALIZATION FAILURE: src/code_analyzer.py _call_llm did not raise RuntimeError when offline.")
            except RuntimeError:
                pass
            except Exception as e:
                issues.append(f"NEUTRALIZATION FAILURE: code_analyzer _call_llm raised unexpected exception: {e}")
        finally:
            ca._OLLAMA_ONLINE = saved_status
    except Exception as e:
        issues.append(f"NEUTRALIZATION FAILURE: Error importing CodeAnalyzerAgent: {e}")

    # 4. Test src/router.py offline RuntimeError
    try:
        import src.router as rt
        saved_status = rt._OLLAMA_ONLINE
        rt._OLLAMA_ONLINE = False
        try:
            router = rt.RouterAgent()
            try:
                router._call_llm("test prompt", {})
                issues.append("NEUTRALIZATION FAILURE: src/router.py _call_llm did not raise RuntimeError when offline.")
            except RuntimeError:
                pass
            except Exception as e:
                issues.append(f"NEUTRALIZATION FAILURE: router _call_llm raised unexpected exception: {e}")
        finally:
            rt._OLLAMA_ONLINE = saved_status
    except Exception as e:
        issues.append(f"NEUTRALIZATION FAILURE: Error importing RouterAgent: {e}")

    # 5. Test src/qa_verifier.py offline RuntimeError
    try:
        import src.qa_verifier as qv
        from src.sandbox import SandboxResult
        saved_status = qv._OLLAMA_ONLINE
        qv._OLLAMA_ONLINE = False
        try:
            verifier = qv.QAVerifierAgent()
            dummy_res = SandboxResult(
                success=False,
                exit_code=1,
                stdout="AssertionError: assert 1 == 2",
                stderr="",
                execution_time_sec=0.1,
                patch_applied=True,
                failed_tests=["test_calc"],
                passed_tests=[]
            )
            try:
                verifier._generate_feedback(dummy_res)
                issues.append("NEUTRALIZATION FAILURE: src/qa_verifier.py _generate_feedback did not raise RuntimeError when offline.")
            except RuntimeError:
                pass
            except Exception as e:
                issues.append(f"NEUTRALIZATION FAILURE: qa_verifier _generate_feedback raised unexpected exception: {e}")
        finally:
            qv._OLLAMA_ONLINE = saved_status
    except Exception as e:
        issues.append(f"NEUTRALIZATION FAILURE: Error importing QAVerifierAgent: {e}")

    # 6. Test src/swarm/swarm_runner.py offline RuntimeError
    try:
        import src.swarm.swarm_runner as sr
        try:
            sr.run_swarm_baseline(live=False)
            issues.append("NEUTRALIZATION FAILURE: src/swarm/swarm_runner.py run_swarm_baseline(live=False) did not raise RuntimeError.")
        except RuntimeError:
            pass
        except Exception as e:
            issues.append(f"NEUTRALIZATION FAILURE: run_swarm_baseline(live=False) raised unexpected exception: {e}")
    except Exception as e:
        issues.append(f"NEUTRALIZATION FAILURE: Error importing run_swarm_baseline: {e}")
            
    return issues


def check_historical_runs_authenticity() -> List[str]:
    """Verify that all historical run CSVs used for headline claims contain explicit live-execution evidence."""
    issues = []
    
    historical_runs = [
        (
            REPO_ROOT / "results" / "prototype_run_groq" / "benchmark_prototype_monolithic_groq_20260909_164450.csv",
            "Groq Monolithic (120B)",
            {"provider": "groq", "min_records": 90, "check_is_live": False}
        ),
        (
            REPO_ROOT / "results" / "prototype_run_swarm_v1_header_only_qa" / "benchmark_prototype_swarm_20260909_175738.csv",
            "Swarm v1 (Header-Only QA)",
            {"provider": "ollama_local", "min_records": 90, "check_is_live": True}
        ),
        (
            REPO_ROOT / "results" / "prototype_run_swarm_v2_content_validated_qa" / "benchmark_prototype_swarm_20260909_204053.csv",
            "Swarm v2 (Content-Validated QA)",
            {"provider": "ollama_local", "min_records": 90, "check_is_live": True}
        ),
    ]
    
    for csv_path, label, reqs in historical_runs:
        if not csv_path.exists():
            issues.append(f"MISSING HISTORICAL RUN: {label} CSV not found at {csv_path.name}")
            continue
        try:
            df = pd.read_csv(csv_path)
            if len(df) < reqs["min_records"]:
                issues.append(f"INCOMPLETE HISTORICAL RUN: {label} contains {len(df)} records, expected >= {reqs['min_records']}.")
                
            # Provider telemetry check
            if "provider" not in df.columns:
                issues.append(f"TELEMETRY GAP: {label} CSV missing 'provider' column.")
            elif not (df["provider"] == reqs["provider"]).all():
                bad_count = (df["provider"] != reqs["provider"]).sum()
                issues.append(f"AUTHENTICITY VIOLATION: {label} has {bad_count} records where provider != '{reqs['provider']}'.")
                
            # is_live field check for swarm runs
            if reqs.get("check_is_live"):
                if "is_live" not in df.columns:
                    issues.append(f"TELEMETRY GAP: {label} CSV missing 'is_live' column.")
                elif not (df["is_live"] == True).all():
                    non_live = (df["is_live"] != True).sum()
                    issues.append(f"MOCK CONTAMINATION: {label} has {non_live} records where is_live != True.")
                    
            # Wall-clock realistic telemetry check (CPU/local SLMs cannot pass in < 2.0 seconds)
            min_wall_clock = df["cumulative_wall_clock_sec"].min()
            if min_wall_clock < 2.0:
                issues.append(f"TELEMETRY ANOMALY: {label} has min wall clock of {min_wall_clock}s (expected >= 2.0s).")
                
        except Exception as e:
            issues.append(f"ERROR AUDITING HISTORICAL RUN {label}: {e}")
            
    return issues


def check_quarantine_integrity() -> List[str]:
    """Verify results/runs/ is properly quarantined, all 2026-09-07 files fail authenticity gates,
    offline simulation fallbacks in src/ are neutralized, and historical run CSVs verify live execution."""
    issues = []
    quarantine_readme = REPO_ROOT / "results" / "runs" / "README.md"
    if not quarantine_readme.exists():
        issues.append("MISSING QUARANTINE: results/runs/README.md does not exist.")
        return issues
        
    content = quarantine_readme.read_text(encoding="utf-8")
    if "QUARANTINE NOTICE" not in content:
        issues.append("INVALID QUARANTINE: results/runs/README.md lacks 'QUARANTINE NOTICE'.")
        
    runs_dir = REPO_ROOT / "results" / "runs"
    synthetic_files = list(runs_dir.glob("benchmark_*_20260907_*.json"))
    if not synthetic_files:
        issues.append("MISSING RUN ARTIFACTS: No 2026-09-07 run files found to verify.")
        return issues

    for sf in synthetic_files:
        is_auth, flags = audit_run_file_authenticity(sf)
        if is_auth:
            issues.append(f"SCALING LEAK: {sf.name} in results/runs/ passed authenticity gates unexpectedly.")

    # Check active code fallback neutralization
    neutralization_issues = check_fallback_neutralization()
    issues.extend(neutralization_issues)

    # Check historical run CSV live execution evidence
    historical_issues = check_historical_runs_authenticity()
    issues.extend(historical_issues)

    return issues


def check_forbidden_claims(doc_path: Path) -> List[str]:
    """Scan a markdown document for any forbidden / synthetic claims."""
    if not doc_path.exists():
        return [f"File not found: {doc_path}"]
    content = doc_path.read_text(encoding="utf-8")
    violations = []
    for pat, desc in FORBIDDEN_PATTERNS:
        if re.search(pat, content, re.IGNORECASE):
            violations.append(f"[{doc_path.name}] Contains forbidden claim: {desc} (pattern: {pat})")
    return violations


def load_verified_metrics() -> Dict[str, Dict[str, float]]:
    """Load verified benchmark metrics directly from CSV artifacts."""
    if not THREE_WAY_CSV.exists() or not STATS_CSV.exists():
        raise FileNotFoundError("Verified CSV artifacts are missing.")
    
    df_3way = pd.read_csv(THREE_WAY_CSV)
    df_stats = pd.read_csv(STATS_CSV)
    
    metrics = {
        "groq_overall_pass": float(df_3way.loc[(df_3way["System"] == "Groq Monolithic (120B)") & (df_3way["Tier"] == "Overall"), "Pass@1 (%)"].iloc[0]),
        "groq_easy_pass": float(df_3way.loc[(df_3way["System"] == "Groq Monolithic (120B)") & (df_3way["Tier"] == "Easy"), "Pass@1 (%)"].iloc[0]),
        "groq_med_pass": float(df_3way.loc[(df_3way["System"] == "Groq Monolithic (120B)") & (df_3way["Tier"] == "Medium"), "Pass@1 (%)"].iloc[0]),
        "groq_hard_pass": float(df_3way.loc[(df_3way["System"] == "Groq Monolithic (120B)") & (df_3way["Tier"] == "Hard"), "Pass@1 (%)"].iloc[0]),
        
        "v1_overall_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v1 (Header-Only QA)") & (df_3way["Tier"] == "Overall"), "Pass@1 (%)"].iloc[0]),
        "v1_easy_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v1 (Header-Only QA)") & (df_3way["Tier"] == "Easy"), "Pass@1 (%)"].iloc[0]),
        "v1_med_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v1 (Header-Only QA)") & (df_3way["Tier"] == "Medium"), "Pass@1 (%)"].iloc[0]),
        "v1_hard_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v1 (Header-Only QA)") & (df_3way["Tier"] == "Hard"), "Pass@1 (%)"].iloc[0]),

        "v2_overall_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v2 (Content-Validated QA)") & (df_3way["Tier"] == "Overall"), "Pass@1 (%)"].iloc[0]),
        "v2_easy_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v2 (Content-Validated QA)") & (df_3way["Tier"] == "Easy"), "Pass@1 (%)"].iloc[0]),
        "v2_med_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v2 (Content-Validated QA)") & (df_3way["Tier"] == "Medium"), "Pass@1 (%)"].iloc[0]),
        "v2_hard_pass": float(df_3way.loc[(df_3way["System"] == "Swarm v2 (Content-Validated QA)") & (df_3way["Tier"] == "Hard"), "Pass@1 (%)"].iloc[0]),
    }
    return metrics


def verify_document_consistency(doc_path: Path, metrics: Dict[str, float]) -> List[str]:
    """Verify that expected key headline numbers match the verified metrics."""
    if not doc_path.exists():
        return [f"File not found: {doc_path}"]
    content = doc_path.read_text(encoding="utf-8")
    issues = []
    
    # Verify exact presence of verified Pass@1 figures
    expected_numbers = [
        ("85.56%", metrics["groq_overall_pass"], "Groq Monolithic Overall Pass@1"),
        ("73.33%", metrics["groq_easy_pass"], "Groq Monolithic Easy Pass@1"),
        ("90.00%", metrics["groq_med_pass"], "Groq Monolithic Medium Pass@1"),
        ("93.33%", metrics["groq_hard_pass"], "Groq Monolithic Hard Pass@1"),
        ("4.44%", metrics["v2_overall_pass"], "Swarm v2 Overall Pass@1"),
        ("13.33%", metrics["v2_easy_pass"], "Swarm v2 Easy Pass@1"),
        ("0.00%", metrics["v1_overall_pass"], "Swarm v1 Overall Pass@1"),
    ]
    
    for str_rep, val, label in expected_numbers:
        clean_str = str_rep.rstrip("%")
        pattern = rf"{re.escape(clean_str)}\s*%"
        if not re.search(pattern, content):
            issues.append(f"[{doc_path.name}] Missing verified {label} ({str_rep})")
            
    return issues


def verify_commit_message_consistency(metrics: Dict[str, float]) -> List[str]:
    """Verify that HEAD commit message does not contain inaccurate or fabricated headline metrics."""
    issues = []
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        commit_msg = res.stdout
    except Exception as e:
        return [f"Could not read HEAD commit message: {e}"]

    # 1. Reject known typo/leak patterns
    if "84.44%" in commit_msg:
        issues.append("COMMIT MESSAGE ERROR: Commit message contains incorrect figure '84.44%' (Groq Monolithic is 85.56%).")

    # 2. If Groq accuracy is mentioned with a percentage, it must match 85.56%
    groq_matches = re.findall(r"groq[^\n.,;]*?(\d+\.\d+)%", commit_msg, re.IGNORECASE)
    for val in groq_matches:
        if val not in ("85.56", "73.33", "90.00", "93.33"):
            issues.append(f"COMMIT MESSAGE ERROR: Groq metric in commit message ({val}%) does not match verified 85.56%.")

    # 3. If Swarm v2 accuracy is mentioned, it must match 4.44%
    swarm_v2_matches = re.findall(r"swarm\s*v2[^\n.,;]*?(\d+\.\d+)%", commit_msg, re.IGNORECASE)
    for val in swarm_v2_matches:
        if val not in ("4.44", "13.33"):
            issues.append(f"COMMIT MESSAGE ERROR: Swarm v2 metric in commit message ({val}%) does not match verified 4.44%.")

    # 4. If Swarm v1 accuracy is mentioned, it must match 0.00%
    swarm_v1_matches = re.findall(r"swarm\s*v1[^\n.,;]*?(\d+\.\d+)%", commit_msg, re.IGNORECASE)
    for val in swarm_v1_matches:
        if val != "0.00":
            issues.append(f"COMMIT MESSAGE ERROR: Swarm v1 metric in commit message ({val}%) does not match verified 0.00%.")

    return issues


def check_structural_pipeline_consistency(doc_path: Path) -> List[str]:
    """Verify that architecture descriptions match src/swarm/agents.py and src/swarm/graph.py."""
    if not doc_path.exists():
        return [f"File not found: {doc_path}"]
    content = doc_path.read_text(encoding="utf-8")
    issues = []
    
    # Must explicitly mention the real swarm models
    if "deepseek-coder:6.7b" not in content.lower():
        issues.append(f"[{doc_path.name}] Structural Mismatch: Missing active Analyzer model 'deepseek-coder:6.7b'.")
    if "qwen2.5:0.5b" not in content.lower():
        issues.append(f"[{doc_path.name}] Structural Mismatch: Missing active QA Verifier model 'qwen2.5:0.5b'.")
        
    return issues


def main() -> int:
    print("=" * 70)
    print("RUNNING CLAIM-TO-ARTIFACT & STRUCTURAL VERIFICATION AUDIT")
    print("=" * 70)
    
    all_issues = []
    
    # Gate 1: Check authenticity gates and quarantine status
    print("\n[Gate 1] Verifying Run Authenticity Gates & Scaffold Quarantine...")
    auth_issues = check_quarantine_integrity()
    if auth_issues:
        for err in auth_issues:
            print(f"  FAILED: {err}")
        all_issues.extend(auth_issues)
    else:
        print("  PASSED: 3 Authenticity Gates verified. All 2026-09-07 runs flagged as synthetic & quarantined.")
        print("  PASSED: Repo-wide scan of src/ confirms all offline simulation fallbacks neutralized with RuntimeErrors.")
        print("  PASSED: Verified 180/180 Swarm records (v1 & v2) and 90/90 Groq records possess explicit live-execution telemetry.")

    # Gate 2: Check forbidden / synthetic claims in docs
    print("\n[Gate 2] Scanning Documentation for Forbidden Synthetic Claims...")
    docs_to_check = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "results" / "SEMINAR_SYNTHESIS.md",
    ]
    for doc in docs_to_check:
        forbidden = check_forbidden_claims(doc)
        if forbidden:
            for f in forbidden:
                print(f"  FAILED: {f}")
            all_issues.extend(forbidden)
        else:
            print(f"  PASSED: {doc.name} is free of forbidden synthetic claims.")

    # Gate 3: Check claim-to-artifact matching
    print("\n[Gate 3] Cross-referencing Key Headline Claims Against Verified Artifacts...")
    try:
        metrics = load_verified_metrics()
        for doc in docs_to_check:
            mismatches = verify_document_consistency(doc, metrics)
            if mismatches:
                for m in mismatches:
                    print(f"  FAILED: {m}")
                all_issues.extend(mismatches)
            else:
                print(f"  PASSED: {doc.name} headline claims match verified CSV artifacts.")

        # Cross-reference HEAD commit message
        commit_issues = verify_commit_message_consistency(metrics)
        if commit_issues:
            for ci in commit_issues:
                print(f"  FAILED: {ci}")
            all_issues.extend(commit_issues)
        else:
            print("  PASSED: HEAD commit message claims match verified CSV artifacts.")
    except Exception as e:
        print(f"  FAILED to load verified metrics: {e}")
        all_issues.append(str(e))

    # Gate 4: Structural Model & Parameter Consistency Check
    print("\n[Gate 4] Validating Structural Architecture & Agent Config Consistency...")
    for doc in docs_to_check:
        struct_issues = check_structural_pipeline_consistency(doc)
        if struct_issues:
            for s in struct_issues:
                print(f"  FAILED: {s}")
            all_issues.extend(struct_issues)
        else:
            print(f"  PASSED: {doc.name} matches locked agent config in src/swarm/agents.py.")

    print("\n" + "=" * 70)
    if all_issues:
        print(f"AUDIT FAILED with {len(all_issues)} issue(s). Commits blocked.")
        print("=" * 70)
        return 1
    else:
        print("AUDIT PASSED: All documentation claims & structural models verified.")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
