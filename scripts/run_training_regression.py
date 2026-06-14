"""Run an isolated short training regression against saved MAT rewards.

The original training command is preserved, including num_env_steps, so a
linear learning-rate schedule uses the same total episode count as the full
training run. Only the temporary copy's runner loop is capped.

The repository, existing result runs, and root reward.mat files are not
modified. Temporary training output is deleted after comparison unless
--keep-workdir is specified.

Example:
    python scripts/run_training_regression.py \
        --episodes 5 \
        --baseline-run results/MyEnv/.../run2 \
        --command "python train/train.py ... --seed 2"
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORED_NAMES = {
    ".git",
    ".agents",
    ".claude",
    ".vscode",
    "__pycache__",
    "ablation_outputs",
    "ap_selection_outputs",
    "figures",
    "paper_results",
    "results",
    "sweep_outputs",
}


def ignore_copy_names(_directory, names):
    ignored = set()
    for name in names:
        if name in IGNORED_NAMES or name.endswith((".pyc", ".pyo")):
            ignored.add(name)
    return ignored


def patch_episode_cap(runner_path, episodes):
    source = runner_path.read_text(encoding="utf-8")
    old = "for episode in range(episodes):"
    new = f"for episode in range(min(episodes, {episodes})):"
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one training episode loop in {runner_path}; found {count}"
        )
    runner_path.write_text(source.replace(old, new, 1), encoding="utf-8")


def replace_once(source, old, new, description):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {description} patch target; found {count}"
        )
    return source.replace(old, new, 1)


def patch_env_profile(env_discrete_path, profile):
    if profile in {"default", "flat"}:
        return

    source = env_discrete_path.read_text(encoding="utf-8")

    access_import = (
        "from envs.ucmec_access_greedy import (\n"
        "    UCMEC_access_greedy_env as UCMEC_low_training_env,\n"
        ")"
    )

    best_front_import = (
        "from envs.ucmec_fronthaul_greedy import (\n"
        "    UCMEC_fronthaul_greedy_env as UCMEC_low_training_env,\n"
        ")"
    )

    hierarchical_import = (
        "from envs.ucmec_hierarchical import (\n"
        "    UCMEC_hierarchical_env as UCMEC_hierarchical_training_env,\n"
        ")"
    )

    if profile == "best_front":
        source = replace_once(
            source,
            access_import,
            best_front_import,
            "best-front environment import",
        )

    elif profile in {"fixed_min", "fixed_max"}:
        class_name = (
            "UCMEC_fixed_min_power_env"
            if profile == "fixed_min"
            else "UCMEC_fixed_max_power_env"
        )

        fixed_import = (
            "from envs.ucmec_fixed_power import (\n"
            f"    {class_name} as UCMEC_hierarchical_training_env,\n"
            ")"
        )

        source = replace_once(
            source,
            hierarchical_import,
            fixed_import,
            f"{profile} environment import",
        )

    else:
        raise ValueError(f"Unsupported environment profile: {profile}")

    env_discrete_path.write_text(source, encoding="utf-8")


def normalize_command(command):
    args = shlex.split(command)
    if not args:
        raise ValueError("Training command is empty")
    if any(token in {"|", "||", "&&", ";", ">", ">>", "<"} for token in args):
        raise ValueError("Shell operators are not supported in --command")

    executable_name = Path(args[0]).name
    if executable_name in {"python", "python3"}:
        args[0] = sys.executable
    if len(args) < 2 or Path(args[1]).as_posix() != "train/train.py":
        raise ValueError("--command must run train/train.py")
    return args


def load_series(path, key):
    payload = loadmat(path)
    if key not in payload:
        available = sorted(k for k in payload if not k.startswith("__"))
        raise KeyError(f"{path} has no key {key!r}; available keys: {available}")
    values = np.asarray(payload[key]).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}:{key} contains NaN or Inf")
    return values


def compare_prefix(label, baseline_path, current_path, episodes, atol, rtol):
    baseline = load_series(baseline_path, label)
    current = load_series(current_path, label)

    if current.size < episodes:
        raise ValueError(
            f"{current_path}:{label} has only {current.size} entries; "
            f"expected at least {episodes}"
        )
    if baseline.size < episodes:
        raise ValueError(
            f"{baseline_path}:{label} has only {baseline.size} entries"
        )

    expected = baseline[:episodes]
    actual = current[:episodes]
    exact = np.array_equal(expected, actual)
    close = np.allclose(expected, actual, atol=atol, rtol=rtol)
    diff = np.abs(expected - actual)
    max_diff = float(diff.max(initial=0.0))

    if exact:
        print(f"{label}: PASS exact match ({episodes} episodes)")
        return True

    first_exact_diff = int(np.flatnonzero(expected != actual)[0])
    print(
        f"{label}: {'PASS tolerance' if close else 'FAIL'} "
        f"(max_abs_diff={max_diff:.17g}, first_diff={first_exact_diff})"
    )
    if not close:
        mismatch = np.flatnonzero(
            ~np.isclose(expected, actual, atol=atol, rtol=rtol)
        )
        idx = int(mismatch[0])
        print(
            f"  episode {idx}: baseline={expected[idx]:.17g}, "
            f"current={actual[idx]:.17g}"
        )
    return close


def capture_baseline(output_dir, current_reward, current_high_reward, episodes, command):
    output_dir.mkdir(parents=True, exist_ok=False)

    reward = load_series(current_reward, "reward")
    if reward.size < episodes:
        raise ValueError(f"{current_reward}:reward has only {reward.size} entries")
    savemat(output_dir / "reward.mat", {"reward": reward[:episodes, None]})

    if current_high_reward.is_file():
        high_reward = load_series(current_high_reward, "high_reward")
        if high_reward.size < episodes:
            raise ValueError(
                f"{current_high_reward}:high_reward has only {high_reward.size} entries"
            )
        savemat(
            output_dir / "high_reward.mat",
            {"high_reward": high_reward[:episodes, None]},
        )

    metadata = {
        "episodes": episodes,
        "command": command,
        "python": sys.executable,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Captured regression baseline: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run isolated short training and compare reward prefixes."
    )
    parser.add_argument("--command", required=True, help="Original training command.")
    baseline_group = parser.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument(
        "--baseline-run",
        help="Existing run directory containing reward.mat.",
    )
    baseline_group.add_argument(
        "--capture-baseline",
        help="Create a new short-run baseline directory instead of comparing.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--env-profile",
        choices=("default", "best_front", "flat", "fixed_min", "fixed_max"),
        default="default",
        help="Environment import switch applied only inside the temporary copy.",
    )
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the temporary project copy and generated outputs.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    baseline_run = None
    baseline_reward = None
    baseline_high_reward = None
    capture_dir = None
    if args.baseline_run:
        baseline_run = Path(args.baseline_run).expanduser().resolve()
        baseline_reward = baseline_run / "reward.mat"
        baseline_high_reward = baseline_run / "high_reward.mat"
        if not baseline_reward.is_file():
            parser.error(f"Missing baseline reward: {baseline_reward}")
    else:
        capture_dir = Path(args.capture_baseline).expanduser().resolve()
        if capture_dir.exists():
            parser.error(f"Capture directory already exists: {capture_dir}")

    command = normalize_command(args.command)
    temp_parent = Path(tempfile.mkdtemp(prefix="ucmec_training_regression_"))
    temp_repo = temp_parent / "UCMEC"
    passed = False

    try:
        print(f"Creating isolated workspace: {temp_repo}")
        shutil.copytree(
            REPO_ROOT,
            temp_repo,
            ignore=ignore_copy_names,
            symlinks=True,
        )
        patch_episode_cap(
            temp_repo / "runner" / "shared" / "env_runner.py",
            args.episodes,
        )
        patch_env_profile(
            temp_repo / "envs" / "env_discrete.py",
            args.env_profile,
        )

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        print(
            f"Running {args.episodes} episodes with original schedule denominator "
            f"(env_profile={args.env_profile})."
        )
        proc = subprocess.run(command, cwd=temp_repo, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Training exited with status {proc.returncode}")

        current_reward = temp_repo / "reward.mat"
        if not current_reward.is_file():
            raise RuntimeError(f"Training did not create {current_reward}")

        current_high_reward = temp_repo / "high_reward.mat"
        if capture_dir is not None:
            capture_baseline(
                capture_dir,
                current_reward,
                current_high_reward,
                args.episodes,
                args.command,
            )
            passed = True
        else:
            passed = compare_prefix(
                "reward",
                baseline_reward,
                current_reward,
                args.episodes,
                args.atol,
                args.rtol,
            )

            if baseline_high_reward.is_file():
                if not current_high_reward.is_file():
                    raise RuntimeError(
                        "Baseline has high_reward.mat, but current training did not create one"
                    )
                passed = (
                    compare_prefix(
                        "high_reward",
                        baseline_high_reward,
                        current_high_reward,
                        args.episodes,
                        args.atol,
                        args.rtol,
                    )
                    and passed
                )

        if capture_dir is None and not passed:
            raise SystemExit(1)
    finally:
        if args.keep_workdir:
            print(f"Kept isolated workspace: {temp_repo}")
        else:
            shutil.rmtree(temp_parent, ignore_errors=True)
            print("Removed isolated workspace and generated training outputs.")

    if capture_dir is not None:
        print("PASS: isolated training baseline captured.")
    else:
        print("PASS: isolated training regression matched the saved baseline.")


if __name__ == "__main__":
    main()
