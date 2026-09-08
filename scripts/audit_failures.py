"""
Detailed audit script for monolithic baseline runs.
Selects 5 representative failure cases:
- 1 from Easy (e.g. easy_01 seed 44)
- 2 from Medium (e.g. med_01 seed 42, med_02 seed 43)
- 2 from Hard (e.g. hard_01 seed 44, hard_02 seed 44)
Extracts all parameters, prompts, outputs, patches, git apply logs, and test results.
"""
import os
import sys
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.argv) > 1:
    MONO_JSON = sys.argv[1]
else:
    runs_dir = os.path.join(BASE_DIR, "results", "runs")
    mono_files = [
        os.path.join(runs_dir, f)
        for f in os.listdir(runs_dir)
        if f.startswith("benchmark_monolithic_baseline_") and f.endswith(".json")
    ]
    mono_files.sort(key=os.path.getmtime)
    MONO_JSON = mono_files[-1] if mono_files else None

if not MONO_JSON or not os.path.exists(MONO_JSON):
    print(f"No monolithic baseline run file found in results/runs.")
    sys.exit(1)

print(f"Auditing file: {os.path.basename(MONO_JSON)}")

with open(MONO_JSON, "r", encoding="utf-8") as f:
    runs = json.load(f)

print(f"Total monolithic runs logged: {len(runs)}")
failed_runs = [r for r in runs if not r["final_pass"]]
print(f"Total failed runs: {len(failed_runs)}")

# Group failures by tier
by_tier = {}
for r in failed_runs:
    by_tier.setdefault(r["difficulty"], []).append(r)

for tier, items in by_tier.items():
    print(f"  {tier}: {len(items)} failures")

# Select 5 representative failure cases: 1 Easy, 2 Medium, 2 Hard
selected_cases = []
if "easy" in by_tier and by_tier["easy"]:
    selected_cases.append(by_tier["easy"][0])
if "medium" in by_tier and len(by_tier["medium"]) >= 2:
    selected_cases.extend(by_tier["medium"][:2])
elif "medium" in by_tier and by_tier["medium"]:
    selected_cases.append(by_tier["medium"][0])
if "hard" in by_tier and len(by_tier["hard"]) >= 2:
    selected_cases.extend(by_tier["hard"][:2])
elif "hard" in by_tier and by_tier["hard"]:
    selected_cases.append(by_tier["hard"][0])

for i, c in enumerate(selected_cases, 1):
    print("=" * 70)
    print(f"CASE {i}: {c['run_id']}")
    print(f"Task: {c['task_id']} | Difficulty: {c['difficulty']} | Seed: {c['seed']}")
    print(f"Model Profile: {c.get('model_profile')} | System: {c.get('system')} | Is Live: {c.get('is_live', False)}")
    print(f"Pass@1: {c['pass_at_1']} | Final Pass: {c['final_pass']}")
    sb = c.get("sandbox_detail", {})
    print(f"Sandbox exit code: {sb.get('exit_code')}")
    print(f"Patch applied flag: {sb.get('patch_applied')}")
    print(f"Patch error: {sb.get('patch_error')}")
    print(f"Failed tests: {sb.get('failed_tests')}")
    print(f"Passed tests: {sb.get('passed_tests')}")
    if c.get("raw_patch"):
        print(f"Raw patch (first 300 chars):\n{c['raw_patch'][:300]}")
    print(f"Stderr snippet:\n{sb.get('stderr_snippet')}")
    print(f"Stdout snippet:\n{sb.get('stdout_snippet')}")
