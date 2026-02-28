import argparse
from pathlib import Path

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit(
        "pandas is required. Install with: pip install pandas"
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scalars.csv line charts.")
    parser.add_argument("--csv", required=True, help="Path to scalars.csv")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for plots (default: same folder as csv)",
    )
    parser.add_argument(
        "--cols",
        default=None,
        help="Comma-separated list of columns to plot. Default: all numeric columns except wall_time/step.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Plot DPI (default 160)",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Output format (default png)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise SystemExit("CSV must contain a 'step' column.")

    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        # numeric columns except wall_time and step
        cols = [
            c
            for c in df.columns
            if c not in ("wall_time", "step") and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not cols:
        raise SystemExit("No columns to plot.")

    for col in cols:
        if col not in df.columns:
            continue
        sub = df[["step", col]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values(by="step")
        x = sub["step"]
        y = sub[col]

        plt.figure(figsize=(7, 4))
        plt.plot(x, y, linewidth=1.5, marker="o", markersize=2)
        plt.xlabel("step")
        plt.ylabel(col)
        plt.title(col)
        plt.grid(True, alpha=0.3)
        out_path = out_dir / f"{col}.{args.format}"
        plt.tight_layout()
        plt.savefig(out_path, dpi=args.dpi)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
