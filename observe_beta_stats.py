import argparse
import numpy as np

from envs.MA_UCMEC_dyna_noncoop_hierarchical import MA_UCMEC_dyna_noncoop_hierarchical


def collect_beta_top10_db(env, steps):
    values = []
    for _ in range(steps):
        # Random low-level actions to advance dynamics.
        action_idx = env.action_space.sample()
        action = [
            np.eye(env.action_dim, dtype=np.float32)[int(a)]
            for a in action_idx
        ]
        env.step(action)
        for i in range(env.M_sim):
            beta_row = env.beta[i, : env.N_sim]
            top_idx = np.argsort(beta_row)[::-1][:10]
            beta_top10 = beta_row[top_idx]
            beta_db = 10.0 * np.log10(beta_top10 + 1e-12)
            values.append(beta_db)
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    env = MA_UCMEC_dyna_noncoop_hierarchical()
    env.seed(args.seed)

    all_vals = []
    for _ in range(args.episodes):
        env.reset()
        vals = collect_beta_top10_db(env, args.steps)
        if vals.size:
            all_vals.append(vals)

    if not all_vals:
        print("No beta samples collected.")
        return

    data = np.concatenate(all_vals)
    percentiles = np.percentile(data, [1, 5, 25, 50, 75, 95, 99])

    print("beta_db stats (top-10 per agent):")
    print(f"count={data.size}")
    print(f"min={data.min():.3f}, max={data.max():.3f}")
    print(f"mean={data.mean():.3f}, std={data.std():.3f}")
    print(
        "p01={:.3f}, p05={:.3f}, p25={:.3f}, p50={:.3f}, p75={:.3f}, p95={:.3f}, p99={:.3f}".format(
            *percentiles
        )
    )


if __name__ == "__main__":
    main()
