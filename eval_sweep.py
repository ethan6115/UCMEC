import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from model_configs import MODEL_CONFIGS


DEFAULT_ENV = {
    "UCMEC_M_SIM": "10",
    "UCMEC_EPSILON": "0.003",
    "UCMEC_N_SIM": "50",
}

SWEEPS = [
    ("M_sim", "UCMEC_M_SIM", ["5", "10", "15", "20", "25"]),
    ("epsilon", "UCMEC_EPSILON", ["0.0006", "0.001", "0.003", "0.005", "0.007"]),
    ("N_sim", "UCMEC_N_SIM", ["10", "20", "30", "40", "50"]),
]

METRICS = [
    "avg_total_delay_ms",
    "avg_local_delay_ms",
    "avg_uplink_delay_ms",
    "avg_front_delay_ms",
    "avg_offloading_users",
]


def _missing_paths():
    missing = []
    for cfg in MODEL_CONFIGS:
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


def _run_eval(model_cfg, sweep_key, sweep_value, output_dir, log_dir):
    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    env.update(model_cfg["env"])
    env[sweep_key] = sweep_value

    fd, json_path = tempfile.mkstemp(prefix="ucmec_eval_", suffix=".json", dir=str(output_dir))
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

    tag = f"{model_cfg['name']}_{sweep_key}_{sweep_value}".replace("/", "_").replace(".", "p")
    if proc.returncode != 0:
        stdout_path, stderr_path = _write_failure_logs(log_dir, tag, proc)
        raise RuntimeError(
            f"eval_script.py failed for {model_cfg['name']} {sweep_key}={sweep_value}; "
            f"logs: {stdout_path}, {stderr_path}"
        )

    json_file = Path(json_path)
    if not json_file.exists():
        stdout_path, stderr_path = _write_failure_logs(log_dir, tag, proc)
        raise RuntimeError(
            f"eval_script.py did not write JSON for {model_cfg['name']} {sweep_key}={sweep_value}; "
            f"logs: {stdout_path}, {stderr_path}"
        )

    with json_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    json_file.unlink(missing_ok=True)
    return payload, elapsed


def _row_from_payload(model_cfg, sweep_name, sweep_value, payload, cached):
    summary = payload["summary"]
    env_meta = payload.get("env", {})
    row = {
        "method": model_cfg["method"],
        "run": model_cfg["run"],
        "name": model_cfg["name"],
        "sweep": sweep_name,
        "value": sweep_value,
        "m_sim": env_meta.get("m_sim"),
        "n_sim": env_meta.get("n_sim"),
        "epsilon": env_meta.get("epsilon"),
        "cached": int(bool(cached)),
    }
    for metric in METRICS:
        row[f"{metric}_mean"] = summary.get(f"{metric}_mean")
        row[f"{metric}_std"] = summary.get(f"{metric}_std")
    return row


def main():
    if len(MODEL_CONFIGS) == 0:
        raise SystemExit("MODEL_CONFIGS is empty.")

    missing = _missing_paths()
    if missing:
        print("Missing model paths; aborting sweep:")
        for name, key, path in missing:
            print(f"  {name}: {key} -> {path}")
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("sweep_outputs") / timestamp
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "eval_sweep_results.csv"
    jsonl_path = output_dir / "eval_sweep_results.jsonl"

    fieldnames = [
        "method",
        "run",
        "name",
        "sweep",
        "value",
        "m_sim",
        "n_sim",
        "epsilon",
        "cached",
    ]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    total_logical = len(MODEL_CONFIGS) * sum(len(values) for _, _, values in SWEEPS)
    print(f"Writing results to {output_dir}")
    print(f"Logical rows: {total_logical}; actual subprocesses: {len(MODEL_CONFIGS) * 13}")

    logical_idx = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for model_cfg in MODEL_CONFIGS:
            print(f"\n=== {model_cfg['name']} ===")
            default_payload_cache = None

            for sweep_name, sweep_key, values in SWEEPS:
                for sweep_value in values:
                    logical_idx += 1
                    is_default = sweep_value == DEFAULT_ENV[sweep_key]

                    if is_default and default_payload_cache is not None:
                        payload = default_payload_cache
                        elapsed_text = "cached"
                        cached = True
                    else:
                        payload, elapsed = _run_eval(
                            model_cfg, sweep_key, sweep_value, output_dir, log_dir
                        )
                        elapsed_text = f"{elapsed:.0f}s"
                        cached = False
                        if is_default:
                            default_payload_cache = payload

                    row = _row_from_payload(model_cfg, sweep_name, sweep_value, payload, cached)
                    writer.writerow(row)
                    csv_file.flush()

                    json_record = {
                        "method": model_cfg["method"],
                        "run": model_cfg["run"],
                        "name": model_cfg["name"],
                        "sweep": sweep_name,
                        "value": sweep_value,
                        "cached": bool(cached),
                        "payload": payload,
                    }
                    jsonl_file.write(json.dumps(json_record, sort_keys=True) + "\n")
                    jsonl_file.flush()

                    print(
                        f"[{logical_idx:03d}/{total_logical}] "
                        f"{model_cfg['method']} {model_cfg['run']} | "
                        f"{sweep_name}={sweep_value:<8} | "
                        f"{_format_line(payload['summary'])} ({elapsed_text})",
                        flush=True,
                    )

    print(f"\nDone. CSV: {csv_path}")
    print(f"Done. JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
