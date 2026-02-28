import argparse
from pathlib import Path
from typing import Dict, List, Iterable, Set

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as exc:
    raise SystemExit(
        "tensorboard is required. Install with: pip install tensorboard"
    ) from exc

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit(
        "pandas is required. Install with: pip install pandas"
    ) from exc


def _find_event_files(log_dir: Path, recursive: bool) -> List[Path]:
    pattern = "events.out.tfevents.*"
    if recursive:
        return sorted(log_dir.rglob(pattern))
    return sorted(log_dir.glob(pattern))


def _load_scalars_from_path(path: Path) -> Dict[str, pd.DataFrame]:
    ea = event_accumulator.EventAccumulator(str(path))
    ea.Reload()
    out: Dict[str, pd.DataFrame] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        df = pd.DataFrame(
            [(e.wall_time, e.step, e.value) for e in events],
            columns=["wall_time", "step", tag],
        )
        out[tag] = df
    return out


def load_scalars(log_dir: Path, tags: List[str], recursive: bool) -> Dict[str, pd.DataFrame]:
    event_files = _find_event_files(log_dir, recursive)
    if not event_files and log_dir.is_file():
        event_files = [log_dir]

    out: Dict[str, pd.DataFrame] = {}
    available: Set[str] = set()
    for path in event_files:
        frames = _load_scalars_from_path(path)
        available.update(frames.keys())
        for tag, df in frames.items():
            if tags and tag not in tags:
                continue
            if tag in out:
                out[tag] = pd.concat([out[tag], df], ignore_index=True)
            else:
                out[tag] = df
    # sort per tag for stable output
    for tag, df in out.items():
        out[tag] = df.sort_values(by=["step", "wall_time"]).reset_index(drop=True)
    return out, available


def merge_scalars(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for tag, df in frames.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["wall_time", "step"], how="outer")
    return merged if merged is not None else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TensorBoard scalar logs to CSV/MAT."
    )
    parser.add_argument(
        "--log_dir",
        required=True,
        help="Path to TensorBoard log directory (contains event files).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to log_dir.",
    )
    parser.add_argument(
        "--tags",
        default="value_loss,policy_loss,dist_entropy,actor_grad_norm,critic_grad_norm,ratio,average_episode_rewards,average_episode_rewards_high",
        help="Comma-separated list of scalar tags to extract.",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "mat", "both"],
        help="Output format.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search for event files under log_dir (default: on).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Disable recursive search and only scan log_dir.",
    )

    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log_dir not found: {log_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else log_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    frames, available = load_scalars(log_dir, tags, args.recursive)
    if not frames:
        raise SystemExit(
            f"No matching scalar tags found in {log_dir}. Available tags: {sorted(available)}"
        )

    merged = merge_scalars(frames)

    if args.format in ("csv", "both"):
        csv_path = out_dir / "scalars.csv"
        merged.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

        # Also write per-tag CSVs for convenience
        for tag, df in frames.items():
            tag_path = out_dir / f"{tag}.csv"
            df.to_csv(tag_path, index=False)
            print(f"Wrote {tag_path}")

    if args.format in ("mat", "both"):
        try:
            from scipy.io import savemat
        except Exception as exc:
            raise SystemExit(
                "scipy is required for MAT output. Install with: pip install scipy"
            ) from exc

        mat_dict = {"step": merged["step"].to_numpy()}
        for tag in tags:
            if tag in merged.columns:
                mat_dict[tag] = merged[tag].to_numpy()
        mat_path = out_dir / "scalars.mat"
        savemat(mat_path, mat_dict)
        print(f"Wrote {mat_path}")


if __name__ == "__main__":
    main()
