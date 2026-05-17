#!/usr/bin/env python3
"""
ProjectLXJ-rev - Sequential Pipeline Execution Script

This script executes all 8 project scripts in sequential order with
real-time progress bar visualization.

Usage:
    python start.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PROJECT_DIR = Path(__file__).parent.resolve()
PROGRAMS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(PROJECT_DIR))


SCRIPTS = [
    {
        "name": "1revised_strict_constrained_model",
        "file": "1revised_strict_constrained_model.py",
        "description": "Constrained Neural Network Training",
        "depends": None,
    },
    {
        "name": "2revised_reverse_prediction_analysis",
        "file": "2revised_reverse_prediction_analysis.py",
        "description": "Reverse Prediction via Hybrid Optimization",
        "depends": ["1revised_strict_constrained_model"],
    },
    {
        "name": "3revised_virtual_experiment",
        "file": "3revised_virtual_experiment.py",
        "description": "Virtual Experiment & Sensitivity Analysis",
        "depends": ["1revised_strict_constrained_model"],
    },
    {
        "name": "4shap_total",
        "file": "4shap_total.py",
        "description": "SHAP Model Interpretability Analysis",
        "depends": ["1revised_strict_constrained_model"],
    },
    {
        "name": "5constrian",
        "file": "5constrian.py",
        "description": "Constraint Satisfaction Analysis",
        "depends": ["1revised_strict_constrained_model"],
    },
    {
        "name": "6revised_individual_subcharts",
        "file": "6revised_individual_subcharts.py",
        "description": "Individual Subplot Generation",
        "depends": ["1revised_strict_constrained_model"],
    },
    {
        "name": "7merged_charts_generator",
        "file": "7merged_charts_generator.py",
        "description": "Publication-Ready Figure Generation",
        "depends": ["6revised_individual_subcharts"],
    },
    {
        "name": "8compare",
        "file": "8compare.py",
        "description": "Model Comparison (MLP vs XGBoost vs RF)",
        "depends": ["1revised_strict_constrained_model"],
    },
]


class ProgressTracker:
    def __init__(self, total_scripts):
        self.total_scripts = total_scripts
        self.current_script = 0
        self.start_time = time.time()
        self.script_times = {}

    def update(self, script_name, status="running"):
        self.current_script += 1
        self.script_times[script_name] = {"status": status, "start": time.time()}

    def complete(self, script_name, success=True):
        if script_name in self.script_times:
            elapsed = time.time() - self.script_times[script_name]["start"]
            self.script_times[script_name]["elapsed"] = elapsed
            self.script_times[script_name]["status"] = "success" if success else "failed"

    def get_progress_bar(self):
        if tqdm is not None:
            return tqdm(
                total=self.total_scripts,
                desc="Pipeline Progress",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        return None

    def print_summary(self):
        total_elapsed = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        print(f"{'Script':<45} {'Status':<12} {'Time':<10}")
        print("-" * 80)
        for script in SCRIPTS:
            name = script["name"]
            if name in self.script_times:
                info = self.script_times[name]
                status = info.get("status", "unknown")
                elapsed = info.get("elapsed", 0)
                time_str = f"{elapsed:.1f}s" if elapsed else "-"
                print(f"{name:<45} {status:<12} {time_str:<10}")
        print("-" * 80)
        print(f"{'Total Runtime:':<45} {total_elapsed:.1f} seconds")
        print("=" * 80)


def print_header():
    print("\n" + "=" * 80)
    print("ProjectLXJ-rev - Constrained Multi-Task Neural Network Pipeline")
    print("=" * 80)
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def print_script_header(script_info, index, total):
    print("\n" + "-" * 80)
    print(f"[{index}/{total}] Executing: {script_info['file']}")
    print(f"Description: {script_info['description']}")
    if script_info['depends']:
        print(f"Dependencies: {', '.join(script_info['depends'])}")
    print("-" * 80)


def run_script(script_path):
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=3600
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Script execution timed out (3600s limit)"
    except Exception as e:
        return -2, "", str(e)


def check_prerequisites(script_info):
    if script_info['depends'] is None:
        return True, None
    for dep in script_info['depends']:
        dep_file = PROGRAMS_DIR / f"{dep}.py"
        if not dep_file.exists():
            return False, f"Missing dependency: {dep_file}"
        dep_script = PROGRAMS_DIR / f"{dep}.py"
        if dep_script.exists():
            result = subprocess.run(
                [sys.executable, "-c", f"import sys; sys.path.insert(0, '{PROGRAMS_DIR}'); import {dep}"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                pass
    return True, None


def main():
    print_header()

    tracker = ProgressTracker(len(SCRIPTS))
    pbar = tracker.get_progress_bar()

    results = []
    for i, script_info in enumerate(SCRIPTS, 1):
        script_file = PROGRAMS_DIR / script_info["file"]

        print_script_header(script_info, i, len(SCRIPTS))

        if not script_file.exists():
            print(f"ERROR: Script not found: {script_file}")
            tracker.update(script_info["name"], "skipped")
            tracker.complete(script_info["name"], success=False)
            if pbar:
                pbar.update(1)
            results.append({
                "script": script_info["name"],
                "status": "skipped",
                "error": "File not found"
            })
            continue

        prereq_ok, prereq_error = check_prerequisites(script_info)
        if not prereq_ok:
            print(f"WARNING: {prereq_error}")

        print(f"Starting execution at {datetime.now().strftime('%H:%M:%S')}...")
        start_time = time.time()
        returncode, stdout, stderr = run_script(script_file)
        elapsed = time.time() - start_time

        if returncode == 0:
            print(f"SUCCESS - Completed in {elapsed:.1f} seconds")
            tracker.complete(script_info["name"], success=True)
            results.append({
                "script": script_info["name"],
                "status": "success",
                "elapsed": elapsed
            })
            if pbar:
                pbar.update(1)
        else:
            print(f"FAILED - Exit code: {returncode}")
            print(f"Error: {stderr[:500] if stderr else 'Unknown error'}")
            tracker.complete(script_info["name"], success=False)
            results.append({
                "script": script_info["name"],
                "status": "failed",
                "error": stderr[:500] if stderr else "Unknown error"
            })
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.set_postfix_str(f"Current: {script_info['name'][:20]}")

    if pbar:
        pbar.close()

    tracker.print_summary()

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print("\nFAILED SCRIPTS:")
        for f in failed:
            print(f"  - {f['script']}: {f.get('error', 'Unknown error')[:100]}")
        print("\nPipeline completed with errors.")
        return 1
    else:
        print("\nPipeline completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
