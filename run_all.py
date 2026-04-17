"""scripts/run_all.py — End-to-end pipeline runner.

Runs all 4 tasks sequentially and generates a final summary report.

Usage:
  python scripts/run_all.py [--skip-clone] [--skip-train]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import yaml


def run_script(script: str, args: list = None, label: str = ""):
    cmd = [sys.executable, script] + (args or [])
    print(f"\n{'='*60}")
    print(f"  Running: {label or script}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ⚠ {label} exited with code {result.returncode}")
    else:
        print(f"  ✓ {label} complete")
    return result.returncode


def generate_report():
    """Aggregate outputs from all tasks into a summary JSON."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "task1": {},
        "task2": {},
        "task3": {},
    }

    # Task 1
    for run_name in ["Run1_LCNN", "Run2_RawNet2"]:
        path = f"outputs/{run_name}_results.json"
        if Path(path).exists():
            with open(path) as f:
                summary["task1"][run_name] = json.load(f)

    # Task 2
    if Path("outputs/task2_results.json").exists():
        with open("outputs/task2_results.json") as f:
            summary["task2"] = json.load(f)

    # Task 3
    if Path("outputs/task3_results.json").exists():
        with open("outputs/task3_results.json") as f:
            summary["task3"] = json.load(f)

    with open("outputs/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)

    for run, data in summary["task1"].items():
        print(f"\n  {run}")
        print(f"    EER (eval set)  : {data.get('eval_eer', 'N/A')}%")
        print(f"    Params          : {data.get('n_params', 'N/A'):,}" if isinstance(data.get('n_params'), int) else f"    Params          : {data.get('n_params', 'N/A')}")

    for run, data in summary["task2"].items():
        print(f"\n  {run} (attack)")
        print(f"    Attack EER      : {data.get('attack_eer', 'N/A')}%")
        print(f"    Cosine sim      : {data.get('mean_cosine_sim', 'N/A')}")
        print(f"    Bypass (clean)  : {data.get('pct_cloned_predicted_real', 'N/A')}%")

    print(f"\n  Plots saved → outputs/plots/")
    print(f"  Full results → outputs/final_summary.json")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-clone",  action="store_true", help="Skip Task 0 (voice cloning)")
    parser.add_argument("--skip-train",  action="store_true", help="Skip Task 1 (training)")
    parser.add_argument("--run",         type=str, default="both", choices=["1", "2", "both"])
    args = parser.parse_args()

    os.makedirs("outputs/plots", exist_ok=True)

    errors = []

    # Task 0
    if not args.skip_clone:
        rc = run_script("scripts/task0_voice_clone.py", label="Task 0: Voice Cloning")
        if rc != 0:
            errors.append("Task 0")

    # Task 1
    if not args.skip_train:
        rc = run_script(
            "scripts/task1_train.py",
            args=["--run", args.run],
            label=f"Task 1: Train Classifiers (run={args.run})",
        )
        if rc != 0:
            errors.append("Task 1")

    # Task 2
    rc = run_script("scripts/task2_attack.py", label="Task 2: Attack Analysis")
    if rc != 0:
        errors.append("Task 2")

    # Task 3
    rc = run_script("scripts/task3_noise_robustness.py", label="Task 3: Noise Robustness")
    if rc != 0:
        errors.append("Task 3")

    # Summary
    generate_report()

    if errors:
        print(f"\n⚠ Tasks with errors: {', '.join(errors)}")
        print("Check individual outputs above for details.")
    else:
        print("\n✓ All tasks completed successfully!")

    print("\nTo start the demo:")
    print("  streamlit run demo/app.py")


if __name__ == "__main__":
    main()
