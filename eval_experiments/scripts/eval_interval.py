import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "eval_experiments" / "outputs"

DEFAULT_ENV = {
    "EVAL_USE_FLAT_JOINT": "0",
    "EVAL_USE_HIERARCHICAL": "1",
    "EVAL_PER_USER": "1",
    "EVAL_HIGH_POLICY_MODE": "trained",
    "EVAL_POWER_VARIANT": "normal",
    "UCMEC_M_SIM": "10",
    "UCMEC_EPSILON": "0.003",
    "UCMEC_N_SIM": "50",
    "UCMEC_CANDIDATE_N": "10",
    "UCMEC_K_FIXED": "2",
}

INTERVAL_MODELS = {
    10: (
        "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/"
        "hierarchical_pair_scorer_noglobal"
    ),
    5: (
        "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/"
        "hierarchical_pair_scorer_noglobal_interval5"
    ),
}

METRICS = [
    "avg_total_delay_ms",
    "avg_local_delay_ms",
    "avg_uplink_delay_ms",
    "avg_front_delay_ms",
    "avg_actual_process_delay_ms",
    "avg_offloading_users",
]


def _build_configs():
    configs = []
    for interval, base_dir in INTERVAL_MODELS.items():
        for run in ("run1", "run2", "run3"):
            configs.append(
                {
                    "interval": interval,
                    "run": run,
                    "name": f"interval{interval}_{run}",
                    "env": {
                        **DEFAULT_ENV,
                        "EVAL_HIERARCHICAL_INTERVAL": str(interval),
                        "EVAL_MODEL_LOW": f"{base_dir}/{run}/models/actor_499.pt",
                        "EVAL_MODEL_HIGH": f"{base_dir}/{run}/models/actor_high.pt",
                        "EVAL_MODEL_FLAT": "",
                    },
                }
            )
    return configs


INTERVAL_CONFIGS = _build_configs()


def _missing_paths():
    missing = []
    for cfg in INTERVAL_CONFIGS:
        for key in ("EVAL_MODEL_LOW", "EVAL_MODEL_HIGH"):
            path = cfg["env"][key]
            if not (REPO_ROOT / path).exists():
                missing.append((cfg["name"], key, path))
    return missing


def _write_failure_logs(log_dir, tag, proc):
    stdout_path = log_dir / f"{tag}.stdout.txt"
    stderr_path = log_dir / f"{tag}.stderr.txt"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return stdout_path, stderr_path


def _run_eval(cfg, output_dir, log_dir):
    env = os.environ.copy()
    env.update(cfg["env"])

    fd, json_path = tempfile.mkstemp(
        prefix="ucmec_interval_", suffix=".json", dir=str(output_dir)
    )
    os.close(fd)
    Path(json_path).unlink(missing_ok=True)
    env["EVAL_OUTPUT_JSON"] = json_path

    start = time.time()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "eval_script.py")],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.time() - start

    if proc.returncode != 0:
        stdout_path, stderr_path = _write_failure_logs(log_dir, cfg["name"], proc)
        raise RuntimeError(
            f"eval_script.py failed for {cfg['name']}; "
            f"logs: {stdout_path}, {stderr_path}"
        )

    json_file = Path(json_path)
    if not json_file.exists():
        stdout_path, stderr_path = _write_failure_logs(log_dir, cfg["name"], proc)
        raise RuntimeError(
            f"eval_script.py did not write JSON for {cfg['name']}; "
            f"logs: {stdout_path}, {stderr_path}"
        )

    with json_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    json_file.unlink(missing_ok=True)

    actual_interval = payload.get("env", {}).get("hierarchical_interval")
    if actual_interval != cfg["interval"]:
        raise RuntimeError(
            f"Interval mismatch for {cfg['name']}: "
            f"expected {cfg['interval']}, got {actual_interval}"
        )
    return payload, elapsed


def _row_from_payload(cfg, payload):
    summary = payload["summary"]
    env_meta = payload.get("env", {})
    row = {
        "interval": cfg["interval"],
        "run": cfg["run"],
        "name": cfg["name"],
        "m_sim": env_meta.get("m_sim"),
        "n_sim": env_meta.get("n_sim"),
        "epsilon": env_meta.get("epsilon"),
        "candidate_n": env_meta.get("candidate_n"),
        "k_fixed": env_meta.get("k_fixed"),
        "model_low": payload.get("models", {}).get("model_low"),
        "model_high": payload.get("models", {}).get("model_high"),
    }
    for metric in METRICS:
        row[f"{metric}_mean"] = summary.get(f"{metric}_mean")
        row[f"{metric}_std"] = summary.get(f"{metric}_std")
    return row


def _format_line(summary):
    return (
        f"total={summary['avg_total_delay_ms_mean']:8.3f} "
        f"local={summary['avg_local_delay_ms_mean']:8.3f} "
        f"uplink={summary['avg_uplink_delay_ms_mean']:8.3f} "
        f"front={summary['avg_front_delay_ms_mean']:8.3f} "
        f"edge={summary['avg_actual_process_delay_ms_mean']:8.3f} "
        f"offload={summary['avg_offloading_users_mean']:6.3f}"
    )


def main():
    missing = _missing_paths()
    if missing:
        print("Missing model paths; aborting interval evaluation:")
        for name, key, path in missing:
            print(f"  {name}: {key} -> {path}")
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / "interval_outputs" / timestamp
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "interval_results.csv"
    jsonl_path = output_dir / "interval_results.jsonl"
    fieldnames = [
        "interval",
        "run",
        "name",
        "m_sim",
        "n_sim",
        "epsilon",
        "candidate_n",
        "k_fixed",
        "model_low",
        "model_high",
    ]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    print(f"Writing interval results to {output_dir}")
    print(f"Total interval runs: {len(INTERVAL_CONFIGS)}")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for idx, cfg in enumerate(INTERVAL_CONFIGS, start=1):
            payload, elapsed = _run_eval(cfg, output_dir, log_dir)
            writer.writerow(_row_from_payload(cfg, payload))
            csv_file.flush()

            record = {
                "interval": cfg["interval"],
                "run": cfg["run"],
                "name": cfg["name"],
                "config_env": cfg["env"],
                "payload": payload,
            }
            jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
            jsonl_file.flush()

            print(
                f"[{idx:02d}/{len(INTERVAL_CONFIGS)}] "
                f"H={cfg['interval']} {cfg['run']} | "
                f"{_format_line(payload['summary'])} ({elapsed:.0f}s)",
                flush=True,
            )

    print(f"\nDone. CSV: {csv_path}")
    print(f"Done. JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
