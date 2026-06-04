import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from ablation_configs import ABLATION_CONFIGS


METRICS = [
    "avg_total_delay_ms",
    "avg_local_delay_ms",
    "avg_uplink_delay_ms",
    "avg_front_delay_ms",
    "avg_actual_process_delay_ms",
    "avg_offloading_users",
]


def _missing_paths():
    missing = []
    for cfg in ABLATION_CONFIGS:
        env = cfg["env"]
        for key in ("EVAL_MODEL_LOW", "EVAL_MODEL_HIGH", "EVAL_MODEL_FLAT"):
            path = env.get(key, "")
            if path and not Path(path).exists():
                missing.append((cfg["name"], key, path))
    return missing


def _format_line(summary):
    return (
        f"total={summary['avg_total_delay_ms_mean']:8.3f} "
        f"local={summary['avg_local_delay_ms_mean']:8.3f} "
        f"uplink={summary['avg_uplink_delay_ms_mean']:8.3f} "
        f"front={summary['avg_front_delay_ms_mean']:8.3f} "
        f"offload={summary['avg_offloading_users_mean']:6.3f}"
    )


def _write_failure_logs(log_dir, tag, proc):
    stdout_path = log_dir / f"{tag}.stdout.txt"
    stderr_path = log_dir / f"{tag}.stderr.txt"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return stdout_path, stderr_path


def _run_eval(cfg, output_dir, log_dir):
    env = os.environ.copy()
    env.update(cfg["env"])

    fd, json_path = tempfile.mkstemp(prefix="ucmec_ablation_", suffix=".json", dir=str(output_dir))
    os.close(fd)
    Path(json_path).unlink(missing_ok=True)
    env["EVAL_OUTPUT_JSON"] = json_path

    start = time.time()
    proc = subprocess.run(
        [sys.executable, "eval_script.py"],
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.time() - start

    tag = cfg["name"].replace("/", "_").replace(".", "p")
    if proc.returncode != 0:
        stdout_path, stderr_path = _write_failure_logs(log_dir, tag, proc)
        raise RuntimeError(
            f"eval_script.py failed for {cfg['name']}; logs: {stdout_path}, {stderr_path}"
        )

    json_file = Path(json_path)
    if not json_file.exists():
        stdout_path, stderr_path = _write_failure_logs(log_dir, tag, proc)
        raise RuntimeError(
            f"eval_script.py did not write JSON for {cfg['name']}; "
            f"logs: {stdout_path}, {stderr_path}"
        )

    with json_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    json_file.unlink(missing_ok=True)
    return payload, elapsed


def _row_from_payload(cfg, payload):
    summary = payload["summary"]
    env_meta = payload.get("env", {})
    row = {
        "group": cfg["group"],
        "variant": cfg["variant"],
        "run": cfg["run"],
        "name": cfg["name"],
        "m_sim": env_meta.get("m_sim"),
        "n_sim": env_meta.get("n_sim"),
        "epsilon": env_meta.get("epsilon"),
        "candidate_n": env_meta.get("candidate_n"),
        "k_fixed": env_meta.get("k_fixed"),
        "high_action_dim": env_meta.get("high_action_dim"),
        "high_policy_mode": payload.get("mode", {}).get("high_policy_mode"),
        "use_hierarchical": payload.get("mode", {}).get("use_hierarchical"),
    }
    for metric in METRICS:
        row[f"{metric}_mean"] = summary.get(f"{metric}_mean")
        row[f"{metric}_std"] = summary.get(f"{metric}_std")
    return row


def main():
    if len(ABLATION_CONFIGS) == 0:
        raise SystemExit("ABLATION_CONFIGS is empty.")

    missing = _missing_paths()
    if missing:
        print("Missing model paths; aborting ablation eval:")
        for name, key, path in missing:
            print(f"  {name}: {key} -> {path}")
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("ablation_outputs") / timestamp
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "ablation_results.csv"
    jsonl_path = output_dir / "ablation_results.jsonl"

    fieldnames = [
        "group",
        "variant",
        "run",
        "name",
        "m_sim",
        "n_sim",
        "epsilon",
        "candidate_n",
        "k_fixed",
        "high_action_dim",
        "high_policy_mode",
        "use_hierarchical",
    ]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    print(f"Writing ablation results to {output_dir}")
    print(f"Total ablation runs: {len(ABLATION_CONFIGS)}")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for idx, cfg in enumerate(ABLATION_CONFIGS, start=1):
            payload, elapsed = _run_eval(cfg, output_dir, log_dir)
            row = _row_from_payload(cfg, payload)
            writer.writerow(row)
            csv_file.flush()

            json_record = {
                "group": cfg["group"],
                "variant": cfg["variant"],
                "run": cfg["run"],
                "name": cfg["name"],
                "config_env": cfg["env"],
                "payload": payload,
            }
            jsonl_file.write(json.dumps(json_record, sort_keys=True) + "\n")
            jsonl_file.flush()

            print(
                f"[{idx:03d}/{len(ABLATION_CONFIGS)}] "
                f"{cfg['group']} {cfg['variant']} {cfg['run']} | "
                f"{_format_line(payload['summary'])} ({elapsed:.0f}s)",
                flush=True,
            )

    print(f"\nDone. CSV: {csv_path}")
    print(f"Done. JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
