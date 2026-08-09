"""Measure uplink interference impact during the standard model evaluation.

The evaluator and environment behavior are left unchanged. This script wraps
the existing uplink-rate calculation and records a paired counterfactual in
which the same user/channel/action sample is evaluated with interference set
to zero.
"""

import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "eval_experiments" / "outputs" / "interference_outputs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_script
from envs.ucmec_hierarchical import UCMEC_hierarchical_env


MODEL_RUNS = (1, 2, 3)

RAW_FIELDS = (
    "model_run",
    "seed",
    "sinr_with_db",
    "sinr_without_db",
    "rate_with_mbps",
    "rate_without_mbps",
)

PER_SEED_FIELDS = (
    "model_run",
    "seed",
    "sample_count",
    "sinr_with_db_mean",
    "sinr_without_db_mean",
    "sinr_decrease_db",
    "rate_with_mbps_mean",
    "rate_without_mbps_mean",
    "rate_loss_percent",
)

PER_RUN_FIELDS = (
    "model_run",
    "sample_count",
    "sinr_with_db_mean",
    "sinr_without_db_mean",
    "sinr_decrease_db",
    "rate_with_mbps_mean",
    "rate_without_mbps_mean",
    "rate_loss_percent",
)

RESULT_FIELDS = (
    ("sinr_with_db", "sinr_with_db_mean"),
    ("sinr_without_db", "sinr_without_db_mean"),
    ("sinr_decrease_db", "sinr_decrease_db"),
    ("rate_with_mbps", "rate_with_mbps_mean"),
    ("rate_without_mbps", "rate_without_mbps_mean"),
    ("rate_loss_percent", "rate_loss_percent"),
)


def _model_path_for_run(model_path, model_run):
    path = Path(model_path)
    parts = list(path.parts)
    run_parts = [index for index, part in enumerate(parts) if re.fullmatch(r"run\d+", part)]
    if len(run_parts) != 1:
        raise ValueError(
            f"Model path must contain exactly one run<number> directory: {model_path}"
        )
    parts[run_parts[0]] = f"run{model_run}"
    return str(Path(*parts))


def _resolve_model_paths():
    model_paths = []
    for model_run in MODEL_RUNS:
        low_path = _model_path_for_run(eval_script.MODEL_LOW, model_run)
        high_path = _model_path_for_run(eval_script.MODEL_HIGH, model_run)
        missing = [path for path in (low_path, high_path) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing model file(s) for run{model_run}: {', '.join(missing)}"
            )
        model_paths.append(
            {
                "model_run": model_run,
                "model_low": low_path,
                "model_high": high_path,
            }
        )
    return model_paths


def _summarize_by_seed(raw):
    rows = []
    model_runs = raw["model_run"]
    seeds = raw["seed"]
    for model_run in sorted(np.unique(model_runs)):
        for seed in sorted(np.unique(seeds[model_runs == model_run])):
            mask = (model_runs == model_run) & (seeds == seed)
            sinr_with = raw["sinr_with_db"][mask]
            sinr_without = raw["sinr_without_db"][mask]
            rate_with = raw["rate_with_mbps"][mask]
            rate_without = raw["rate_without_mbps"][mask]
            rows.append(
                {
                    "model_run": int(model_run),
                    "seed": int(seed),
                    "sample_count": int(np.count_nonzero(mask)),
                    "sinr_with_db_mean": float(np.mean(sinr_with)),
                    "sinr_without_db_mean": float(np.mean(sinr_without)),
                    "sinr_decrease_db": float(np.mean(sinr_without - sinr_with)),
                    "rate_with_mbps_mean": float(np.mean(rate_with)),
                    "rate_without_mbps_mean": float(np.mean(rate_without)),
                    "rate_loss_percent": float(
                        100.0 * (1.0 - np.sum(rate_with) / np.sum(rate_without))
                    ),
                }
            )
    return rows


def _summarize_by_run(seed_rows):
    rows = []
    for model_run in MODEL_RUNS:
        run_seed_rows = [row for row in seed_rows if row["model_run"] == model_run]
        if not run_seed_rows:
            raise RuntimeError(f"No seed results were recorded for run{model_run}.")
        row = {
            "model_run": model_run,
            "sample_count": int(sum(item["sample_count"] for item in run_seed_rows)),
        }
        for _, per_seed_field in RESULT_FIELDS:
            row[per_seed_field] = float(
                np.mean([item[per_seed_field] for item in run_seed_rows])
            )
        rows.append(row)
    return rows


def _mean_se(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if values.size < 2:
        raise ValueError("At least two model runs are required to calculate SE.")
    std = float(np.std(values, ddof=1))
    return float(np.mean(values)), std / np.sqrt(values.size)


def _write_outputs(raw, seed_rows, run_rows, env_meta, model_paths):
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_path = output_dir / "interference_raw_samples.npz"
    np.savez_compressed(raw_path, **raw)

    per_seed_path = output_dir / "interference_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        writer.writerows(seed_rows)

    per_run_path = output_dir / "interference_per_run.csv"
    with per_run_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PER_RUN_FIELDS)
        writer.writeheader()
        writer.writerows(run_rows)

    results = {}
    for result_name, per_run_field in RESULT_FIELDS:
        mean, se = _mean_se(run_rows, per_run_field)
        results[f"{result_name}_mean"] = mean
        results[f"{result_name}_se"] = se

    summary = {
        "config": {
            "num_model_runs": len(MODEL_RUNS),
            "model_runs": list(MODEL_RUNS),
            "num_seeds": len(eval_script.SEEDS),
            "seeds": [int(seed) for seed in eval_script.SEEDS],
            "episodes_per_seed": int(eval_script.EPISODES_PER_SEED),
            "episode_steps": 200,
            "m_sim": env_meta.get("m_sim"),
            "n_sim": env_meta.get("n_sim"),
            "models": model_paths,
            "high_policy_mode": eval_script.HIGH_POLICY_MODE,
        },
        "sample_count": int(raw["seed"].size),
        "per_run": run_rows,
        "results": results,
    }
    summary_path = output_dir / "interference_summary.json"
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)

    return output_dir, summary


def _print_summary(output_dir, summary):
    results = summary["results"]

    def result(field):
        return results[f"{field}_mean"], results[f"{field}_se"]

    sinr_with = result("sinr_with_db")
    sinr_without = result("sinr_without_db")
    sinr_decrease = result("sinr_decrease_db")
    rate_loss = result("rate_loss_percent")

    print("\nPer-run means (each averaged across environment seeds)")
    for row in summary["per_run"]:
        print(
            f"  run{row['model_run']}: "
            f"SINR with={row['sinr_with_db_mean']:.4f} dB, "
            f"without={row['sinr_without_db_mean']:.4f} dB, "
            f"decrease={row['sinr_decrease_db']:.4f} dB, "
            f"rate loss={row['rate_loss_percent']:.4f} %"
        )

    print("\nInterference impact summary (mean +/- SE across 3 model runs)")
    print(f"  Samples:                  {summary['sample_count']}")
    print(f"  SINR with interference:   {sinr_with[0]:.4f} +/- {sinr_with[1]:.4f} dB")
    print(f"  SINR without interference:{sinr_without[0]:.4f} +/- {sinr_without[1]:.4f} dB")
    print(f"  SINR decrease:            {sinr_decrease[0]:.4f} +/- {sinr_decrease[1]:.4f} dB")
    print(f"  Uplink rate loss:         {rate_loss[0]:.4f} +/- {rate_loss[1]:.4f} %")
    print(f"  Output directory:         {output_dir}")


def main():
    if eval_script.USE_FLAT_JOINT or not eval_script.USE_HIERARCHICAL:
        raise SystemExit("This diagnostic requires hierarchical evaluation mode.")
    if eval_script.HIGH_POLICY_MODE != "trained":
        raise SystemExit("This diagnostic requires EVAL_HIGH_POLICY_MODE=trained.")

    model_paths = _resolve_model_paths()
    samples = []
    env_meta = {}
    current_model_run = None
    original_make_env = eval_script.make_env
    original_model_high = eval_script.MODEL_HIGH
    original_uplink_rate_cal = UCMEC_hierarchical_env.uplink_rate_cal

    def tracked_make_env(seed):
        env = original_make_env(seed)
        env.interference_eval_seed = seed
        env.interference_eval_model_run = current_model_run
        if seed is not None and not env_meta:
            env_meta.update(m_sim=int(env.M_sim), n_sim=int(env.N_sim))
        return env

    def tracked_uplink_rate_cal(self, p, omega, cluster_matrix, theta):
        rates = original_uplink_rate_cal(self, p, omega, cluster_matrix, theta)
        seed = getattr(self, "interference_eval_seed", None)
        model_run = getattr(self, "interference_eval_model_run", None)
        if seed is None or model_run is None:
            return rates

        active_mask = np.asarray(omega) != 0
        for user_i in np.flatnonzero(active_mask):
            selected = np.asarray(cluster_matrix[user_i, : self.N_sim]) == 1
            selected_theta = np.asarray(theta[user_i, : self.N_sim]) * selected
            sum_theta = float(np.sum(selected_theta))
            noise = float(self.noise_access * sum_theta)
            signal = float((sum_theta ** 2) * p[user_i] * self.varsig)

            interference_by_user = (
                np.asarray(self.beta[: self.M_sim, : self.N_sim]) @ selected_theta
            ) * np.asarray(p[: self.M_sim])
            interference_by_user[~active_mask] = 0.0
            interference_by_user[user_i] = 0.0
            interference = float(np.sum(interference_by_user))

            if signal <= 0.0 or noise <= 0.0:
                raise RuntimeError(
                    f"Non-positive signal/noise for offloading user {user_i}: "
                    f"signal={signal}, noise={noise}"
                )

            sinr_with = signal / (interference + noise)
            sinr_without = signal / noise
            rate_with = self.bandwidth_a * np.log2(1.0 + sinr_with)
            rate_without = self.bandwidth_a * np.log2(1.0 + sinr_without)

            if sinr_without + 1e-12 < sinr_with:
                raise AssertionError("SINR without interference is unexpectedly lower.")
            if rate_without + 1e-6 < rate_with:
                raise AssertionError("Rate without interference is unexpectedly lower.")
            if not np.isclose(rate_with, rates[user_i, 0], rtol=1e-10, atol=1e-6):
                raise AssertionError("Wrapped rate calculation does not match the environment.")

            samples.append(
                (
                    int(model_run),
                    int(seed),
                    10.0 * np.log10(sinr_with),
                    10.0 * np.log10(sinr_without),
                    rate_with / 1e6,
                    rate_without / 1e6,
                )
            )

        return rates

    eval_script.make_env = tracked_make_env
    UCMEC_hierarchical_env.uplink_rate_cal = tracked_uplink_rate_cal

    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        for paths in model_paths:
            current_model_run = paths["model_run"]
            eval_script.MODEL_HIGH = paths["model_high"]
            print(f"\n===== Evaluating model run{current_model_run} =====")
            eval_script.evaluate(paths["model_low"])
    finally:
        os.chdir(previous_cwd)
        eval_script.make_env = original_make_env
        eval_script.MODEL_HIGH = original_model_high
        UCMEC_hierarchical_env.uplink_rate_cal = original_uplink_rate_cal

    if not samples:
        raise RuntimeError("Evaluation completed without any offloading samples.")

    sample_array = np.asarray(samples, dtype=np.float64)
    raw = {
        field: sample_array[:, index].astype(
            np.int32 if field in {"model_run", "seed"} else np.float32
        )
        for index, field in enumerate(RAW_FIELDS)
    }
    seed_rows = _summarize_by_seed(raw)
    run_rows = _summarize_by_run(seed_rows)
    output_dir, summary = _write_outputs(
        raw, seed_rows, run_rows, env_meta, model_paths
    )
    _print_summary(output_dir, summary)


if __name__ == "__main__":
    main()
