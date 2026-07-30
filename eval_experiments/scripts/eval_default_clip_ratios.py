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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_experiments.configs.model_configs import MODEL_CONFIGS
from eval_experiments.scripts.eval_sweep import DEFAULT_ENV


METRICS = [
    "front_delay_raw_gt_2s_ratio",
    "total_delay_raw_gt_2s_ratio",
]


def _missing_paths():
    missing = []
    for cfg in MODEL_CONFIGS:
        env = cfg["env"]
        for key in ("EVAL_MODEL_LOW", "EVAL_MODEL_HIGH", "EVAL_MODEL_FLAT"):
            path = env.get(key, "")
            if path and not (REPO_ROOT / path).exists():
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
    env.update(DEFAULT_ENV)
    env.update(cfg["env"])

    fd, json_path = tempfile.mkstemp(prefix="ucmec_default_clip_", suffix=".json", dir=str(output_dir))
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

    tag = cfg["name"].replace("/", "_").replace(".", "p")
    if proc.returncode != 0:
        stdout_path, stderr_path = _write_failure_logs(log_dir, tag, proc)
        raise RuntimeError(f"eval_script.py failed for {cfg['name']}; logs: {stdout_path}, {stderr_path}")

    json_file = Path(json_path)
    if not json_file.exists():
        stdout_path, stderr_path = _write_failure_logs(log_dir, tag, proc)
        raise RuntimeError(f"eval_script.py did not write JSON for {cfg['name']}; logs: {stdout_path}, {stderr_path}")

    with json_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    json_file.unlink(missing_ok=True)
    return payload, elapsed


def _row_from_payload(cfg, payload):
    summary = payload["summary"]
    env_meta = payload.get("env", {})
    row = {
        "method": cfg["method"],
        "run": cfg["run"],
        "name": cfg["name"],
        "m_sim": env_meta.get("m_sim"),
        "n_sim": env_meta.get("n_sim"),
        "epsilon": env_meta.get("epsilon"),
    }
    for metric in METRICS:
        row[f"{metric}_mean"] = summary.get(f"{metric}_mean")
        row[f"{metric}_std"] = summary.get(f"{metric}_std")
        row[f"{metric}_per_seed"] = json.dumps(summary.get(f"{metric}_per_seed"), separators=(",", ":"))
    return row


def _format_line(summary):
    return (
        f"front_raw>2s={summary['front_delay_raw_gt_2s_ratio_mean']:.6f} "
        f"total_raw>2s={summary['total_delay_raw_gt_2s_ratio_mean']:.6f}"
    )


def main():
    if len(MODEL_CONFIGS) == 0:
        raise SystemExit("MODEL_CONFIGS is empty.")

    missing = _missing_paths()
    if missing:
        print("Missing model paths; aborting default clip-ratio eval:")
        for name, key, path in missing:
            print(f"  {name}: {key} -> {path}")
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / "default_clip_ratio_outputs" / timestamp
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "default_clip_ratios.csv"
    jsonl_path = output_dir / "default_clip_ratios.jsonl"
    fieldnames = ["method", "run", "name", "m_sim", "n_sim", "epsilon"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_per_seed"])

    print(f"Writing results to {output_dir}")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for idx, cfg in enumerate(MODEL_CONFIGS, start=1):
            print(f"[{idx}/{len(MODEL_CONFIGS)}] {cfg['name']}")
            payload, elapsed = _run_eval(cfg, output_dir, log_dir)
            row = _row_from_payload(cfg, payload)
            writer.writerow(row)
            csv_file.flush()

            record = {
                "method": cfg["method"],
                "run": cfg["run"],
                "name": cfg["name"],
                "env": payload.get("env", {}),
                "summary": {k: v for k, v in payload["summary"].items() if any(k.startswith(m) for m in METRICS)},
            }
            jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
            jsonl_file.flush()
            print(f"  {_format_line(payload['summary'])} ({elapsed:.0f}s)")

    print(f"\nCSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
