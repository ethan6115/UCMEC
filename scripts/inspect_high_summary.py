import argparse
import json
from pathlib import Path


TARGETS = [
    "high/policy_loss",
    "high/value_loss",
    "high/dist_entropy",
    "high/ratio",
    "high/actor_grad_norm",
    "high/critic_grad_norm",
]


def find_key(data, target):
    matches = [k for k in data.keys() if target in k]
    if not matches:
        return None
    # Prefer exact suffix match if present.
    for k in matches:
        if k.endswith(target):
            return k
    return matches[0]


def summarize_series(series, tail_n):
    if not series:
        return {
            "count": 0,
            "first3": [],
            "last5": [],
            "tail_mean": None,
            "tail_min": None,
            "tail_max": None,
        }
    vals = [float(x[2]) for x in series]
    tail = vals[-tail_n:] if tail_n > 0 else vals
    return {
        "count": len(series),
        "first3": [(int(x[1]), float(x[2])) for x in series[:3]],
        "last5": [(int(x[1]), float(x[2])) for x in series[-5:]],
        "tail_mean": sum(tail) / len(tail),
        "tail_min": min(tail),
        "tail_max": max(tail),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect high-level training metrics from summary.json")
    parser.add_argument("summary_json", type=str, help="Path to TensorBoard-exported summary.json")
    parser.add_argument("--tail-n", type=int, default=10, help="Number of latest points for tail stats")
    args = parser.parse_args()

    path = Path(args.summary_json)
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"File: {path}")
    print(f"Total keys: {len(data)}")

    for target in TARGETS:
        key = find_key(data, target)
        print(f"\n=== {target} ===")
        if key is None:
            print("not found")
            continue
        series = data[key]
        s = summarize_series(series, args.tail_n)
        print(f"key: {key}")
        print(f"count: {s['count']}")
        print(f"first3 (step, value): {s['first3']}")
        print(f"last5 (step, value): {s['last5']}")
        print(
            f"tail{min(args.tail_n, s['count'])} mean/min/max: "
            f"{s['tail_mean']:.6f} / {s['tail_min']:.6f} / {s['tail_max']:.6f}"
        )

    # Extra helper: show all high/* keys found.
    high_keys = [k for k in data.keys() if "high/" in k]
    print(f"\nFound high/* keys: {len(high_keys)}")
    for k in sorted(high_keys):
        print(k)


if __name__ == "__main__":
    main()

