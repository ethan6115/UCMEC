import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Ensure repo root is on sys.path when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_HOTSPOT_PATH = REPO_ROOT / "envs" / "MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot.py"
_SPEC = importlib.util.spec_from_file_location("hotspot_env_module", _HOTSPOT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load hotspot env module from: {_HOTSPOT_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
MA_UCMEC_dyna_noncoop_hierarchical_peruser = _MOD.MA_UCMEC_dyna_noncoop_hierarchical_peruser


@dataclass
class MetricSummary:
    mean: float
    std: float
    p1: float
    p5: float
    p50: float
    p95: float
    p99: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "p1": self.p1,
            "p5": self.p5,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


def parse_seed_spec(spec: str) -> List[int]:
    seeds: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            seeds.extend(list(range(lo, hi + 1)))
        else:
            seeds.append(int(token))
    seeds = sorted(set(seeds))
    if not seeds:
        raise ValueError("No valid seeds parsed from --seeds")
    return seeds


def summarize(values: np.ndarray) -> MetricSummary:
    values = values.astype(np.float64, copy=False)
    return MetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        p1=float(np.percentile(values, 1)),
        p5=float(np.percentile(values, 5)),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
    )


def collect_slot_raw_db(env: MA_UCMEC_dyna_noncoop_hierarchical_peruser) -> Tuple[np.ndarray, np.ndarray]:
    m_sim = env.M_sim
    n_sim = env.N_sim
    beta_sim = env.beta[:m_sim, :n_sim]

    # Top-10 APs by beta for each simulated user.
    top10_idx = np.argsort(beta_sim, axis=1)[:, ::-1][:, :10]
    row_idx = np.arange(m_sim)[:, None]
    beta_top10 = beta_sim[row_idx, top10_idx]
    beta_db = 10.0 * np.log10(beta_top10 + 1e-12)

    # Fronthaul pathloss-like feature used in high obs.
    front_db = np.zeros((m_sim, env.K * 10), dtype=np.float64)
    for i in range(m_sim):
        ap_idx = top10_idx[i]
        for cpu in range(env.K):
            dist_km = np.maximum(env.distance_matrix_front[ap_idx, cpu], 1e-6)
            alpha = np.where(
                env.link_type[ap_idx, cpu] == 0,
                env.alpha_los,
                env.alpha_nlos,
            )
            g_gain = np.maximum(env.G[ap_idx, cpu], 1e-12)
            pathloss = g_gain * np.power(dist_km, -alpha)
            front_db[i, cpu * 10 : (cpu + 1) * 10] = 10.0 * np.log10(pathloss + 1e-12)

    return beta_db.reshape(-1), front_db.reshape(-1)


def collect_seed_values(seed: int, episodes: int, slots_per_episode: int) -> Dict[str, np.ndarray]:
    env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(seed=seed)
    beta_all: List[np.ndarray] = []
    front_all: List[np.ndarray] = []

    for _ in range(episodes):
        env.reset()
        for _ in range(slots_per_episode):
            env.advance_channel()
            beta_db, front_db = collect_slot_raw_db(env)
            beta_all.append(beta_db)
            front_all.append(front_db)

    return {
        "beta_db": np.concatenate(beta_all, axis=0),
        "front_db": np.concatenate(front_all, axis=0),
    }


def derive_clip_from_seed_quantiles(seed_summaries: Sequence[Dict], metric: str) -> Tuple[float, float]:
    p5s = np.array([item[metric]["p5"] for item in seed_summaries], dtype=np.float64)
    p95s = np.array([item[metric]["p95"] for item in seed_summaries], dtype=np.float64)
    clip_low = float(np.percentile(p5s, 20))
    clip_high = float(np.percentile(p95s, 80))
    return clip_low, clip_high


def saturation(values: np.ndarray, lo: float, hi: float) -> Dict[str, float]:
    return {
        "low_sat": float(np.mean(values <= lo)),
        "high_sat": float(np.mean(values >= hi)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate beta/front clip ranges for hotspot high-level observations."
    )
    parser.add_argument("--seeds", type=str, default="0-19", help="Seed spec, e.g. 0-29 or 0,2,4,10")
    parser.add_argument("--episodes", type=int, default=60, help="Episodes per seed")
    parser.add_argument("--slots", type=int, default=80, help="Slots per episode")
    parser.add_argument(
        "--output",
        type=str,
        default="results/hotspot_clip_calibration.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--skip_saturation",
        action="store_true",
        help="Skip second pass saturation check",
    )
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    print(f"[clip-calib] seeds={seeds}")
    print(f"[clip-calib] episodes={args.episodes}, slots={args.slots}")

    seed_summaries: List[Dict] = []

    for seed in seeds:
        values = collect_seed_values(seed, args.episodes, args.slots)
        beta_summary = summarize(values["beta_db"]).as_dict()
        front_summary = summarize(values["front_db"]).as_dict()
        seed_summaries.append(
            {
                "seed": seed,
                "beta_db": beta_summary,
                "front_db": front_summary,
            }
        )
        print(
            f"[clip-calib] seed={seed} "
            f"beta(p5,p95)=({beta_summary['p5']:.2f},{beta_summary['p95']:.2f}) "
            f"front(p5,p95)=({front_summary['p5']:.2f},{front_summary['p95']:.2f})"
        )

    recommended = {
        "beta_db_clip": derive_clip_from_seed_quantiles(seed_summaries, "beta_db"),
        "front_db_clip": derive_clip_from_seed_quantiles(seed_summaries, "front_db"),
    }

    saturation_summary: List[Dict] = []
    if not args.skip_saturation:
        for seed in seeds:
            values = collect_seed_values(seed, args.episodes, args.slots)
            seed_sat = {
                "seed": seed,
                "beta_db": saturation(values["beta_db"], *recommended["beta_db_clip"]),
                "front_db": saturation(values["front_db"], *recommended["front_db_clip"]),
            }
            saturation_summary.append(seed_sat)
            print(
                f"[clip-calib] sat seed={seed} "
                f"beta(low,high)=({seed_sat['beta_db']['low_sat']:.3f},{seed_sat['beta_db']['high_sat']:.3f}) "
                f"front(low,high)=({seed_sat['front_db']['low_sat']:.3f},{seed_sat['front_db']['high_sat']:.3f})"
            )

    result = {
        "config": {
            "seeds": seeds,
            "episodes_per_seed": args.episodes,
            "slots_per_episode": args.slots,
            "method": {
                "clip_low": "percentile(seed_p5, 20)",
                "clip_high": "percentile(seed_p95, 80)",
            },
        },
        "recommended_clips": {
            "beta_db_clip": [float(recommended["beta_db_clip"][0]), float(recommended["beta_db_clip"][1])],
            "front_db_clip": [
                float(recommended["front_db_clip"][0]),
                float(recommended["front_db_clip"][1]),
            ],
        },
        "per_seed_quantiles": seed_summaries,
        "per_seed_saturation": saturation_summary,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[clip-calib] recommended clips:")
    print("  beta_db_clip:", result["recommended_clips"]["beta_db_clip"])
    print("  front_db_clip:", result["recommended_clips"]["front_db_clip"])
    print(f"[clip-calib] saved: {out_path}")


if __name__ == "__main__":
    main()
