"""Run all configured training regression checks sequentially."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "training_regression_experiments.json"
SINGLE_RUNNER = REPO_ROOT / "scripts" / "run_training_regression.py"


def load_manifest(path):
    experiments = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Manifest must contain a non-empty JSON list")

    required = {"name", "baseline_run", "command", "env_profile"}
    names = set()
    for index, experiment in enumerate(experiments):
        missing = required - set(experiment)
        if missing:
            raise ValueError(f"Experiment {index} is missing: {sorted(missing)}")
        if experiment["name"] in names:
            raise ValueError(f"Duplicate experiment name: {experiment['name']}")
        names.add(experiment["name"])
    return experiments


def resolve_baseline(path):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def validate_experiment(experiment):
    baseline = resolve_baseline(experiment["baseline_run"])
    if not (baseline / "reward.mat").is_file():
        raise FileNotFoundError(f"{experiment['name']}: missing {baseline / 'reward.mat'}")
    episodes = int(experiment.get("episodes", 5))
    if episodes <= 0:
        raise ValueError(f"{experiment['name']}: episodes must be positive")
    return baseline, episodes


def main():
    parser = argparse.ArgumentParser(
        description="Run the UCMEC training regression manifest sequentially."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--only",
        help="Comma-separated experiment names; default runs the full manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print experiments without training.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    experiments = load_manifest(manifest_path)
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}
        experiments = [exp for exp in experiments if exp["name"] in selected]
        missing = selected - {exp["name"] for exp in experiments}
        if missing:
            parser.error(f"Unknown experiment names: {sorted(missing)}")

    validated = []
    for experiment in experiments:
        baseline, episodes = validate_experiment(experiment)
        validated.append((experiment, baseline, episodes))
        print(
            f"{experiment['name']}: episodes={episodes}, "
            f"profile={experiment['env_profile']}, baseline={baseline}"
        )

    if args.dry_run:
        print(f"\nValidated {len(validated)} experiments.")
        return

    log_root = Path(tempfile.mkdtemp(prefix="ucmec_regression_suite_"))
    results = []
    try:
        for index, (experiment, baseline, episodes) in enumerate(validated, start=1):
            name = experiment["name"]
            log_path = log_root / f"{name}.log"
            command = [
                sys.executable,
                str(SINGLE_RUNNER),
                "--episodes",
                str(episodes),
                "--env-profile",
                experiment["env_profile"],
                "--baseline-run",
                str(baseline),
                "--command",
                experiment["command"],
            ]
            print(f"\n[{index}/{len(validated)}] Running {name}...")
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            results.append((name, proc.returncode, log_path))
            print(f"{name}: {'PASS' if proc.returncode == 0 else 'FAIL'}")
            if proc.returncode != 0:
                print(f"  log: {log_path}")

        print("\nTraining regression summary")
        for name, returncode, _log_path in results:
            print(f"  {name:<28} {'PASS' if returncode == 0 else 'FAIL'}")

        failures = [result for result in results if result[1] != 0]
        if failures:
            print(f"\nFailure logs kept at: {log_root}")
            raise SystemExit(1)
    finally:
        if results and all(returncode == 0 for _, returncode, _ in results):
            shutil.rmtree(log_root, ignore_errors=True)
            print("\nRemoved suite logs because all experiments passed.")


if __name__ == "__main__":
    main()
