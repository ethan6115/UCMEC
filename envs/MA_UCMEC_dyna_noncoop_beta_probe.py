import argparse
import math
import numpy as np

from MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop


class MA_UCMEC_dyna_noncoop_beta_probe(MA_UCMEC_dyna_noncoop):
    """
    Probe env for beta/top10 statistics without touching the original env.
    - Supports configurable scene size and N_sim (AP density proxy).
    - Records top10 candidate occupancy and beta stats across rollout.
    """

    def __init__(
        self,
        render: bool = False,
        seed=None,
        area_size: float = 900.0,
        n_sim: int = 50,
        user_center_size: float = 600.0,
        constrain_users_center: bool = True,
        print_top10: bool = False,
        print_users: int = 2,
        print_every: int = 10,
    ):
        super().__init__(render=render, seed=seed)
        self.area_size = float(area_size)
        self.base_area_size = 900.0
        self.print_top10 = bool(print_top10)
        self.print_users = int(print_users)
        self.print_every = max(1, int(print_every))
        self.N_sim = int(max(1, min(int(n_sim), self.N)))
        self.user_center_size = float(max(1.0, min(user_center_size, self.base_area_size)))
        self.constrain_users_center = bool(constrain_users_center)
        self.sigma_s = 8.0  # shadow fading std dev in dB
        #self._set_aps_on_equal_grid()
        #self._recompute_distance_matrix_front_only()

        self._scaled_to_area = False
        self._top10_steps = 0
        self._top10_slots = 0
        self._ap_top10_count = np.zeros((self.N_sim,), dtype=np.int64)
        self._ap_top10_beta_sum = np.zeros((self.N_sim,), dtype=np.float64)
        self._all_top10_betas = []
        self._rank_beta_share_sum = np.zeros((10,), dtype=np.float64)
        self._top1_share_samples = []
        self._top2_share_samples = []
        # Top1 AP stability stats
        self.cluster_update_interval = 10
        self._prev_top1_idx = None
        self._top1_same_prev_step_count = 0
        self._top1_prev_step_comp_count = 0
        self._prev_update_top1_idx = None
        self._top1_same_update_count = 0
        self._top1_update_comp_count = 0
        self._interval_active = False
        self._curr_interval_prev_top1 = None
        self._curr_interval_switch_count = np.zeros((self.M_sim,), dtype=np.int64)
        self._interval_switch_sum = 0
        self._interval_user_count = 0
        self._interval_constant_count = 0
        self._interval_count = 0

    def _set_aps_on_equal_grid(self):
        # Place only the simulated AP subset (first N_sim APs) on a full-coverage
        # rectangular grid. Spacing is uniform per axis (x-equal, y-equal), and
        # dx/dy can differ to fully cover the area.
        n_grid = int(self.N_sim)
        # Prefer exact factorization (e.g., 50 -> 10x5) for even coverage.
        rows = 1
        for r in range(1, int(np.sqrt(n_grid)) + 1):
            if n_grid % r == 0:
                rows = r
        cols = n_grid // rows

        dx = self.base_area_size / max(1, cols - 1)
        dy = self.base_area_size / max(1, rows - 1)
        x0 = 0.0
        y0 = 0.0

        pts = []
        for r in range(rows):
            y = y0 + r * dy
            for c in range(cols):
                x = x0 + c * dx
                pts.append([x, y])
                if len(pts) >= n_grid:
                    self.locations_aps[:n_grid] = np.asarray(pts, dtype=np.float64)
                    return

    def _recompute_distance_matrix_front_only(self):
        self.distance_matrix_front = np.zeros([self.N, self.K])
        for i in range(self.N):
            for j in range(self.K):
                self.distance_matrix_front[i, j] = math.sqrt(
                    (self.locations_aps[i, 0] - self.locations_cpu[j, 0]) ** 2
                    + (self.locations_aps[i, 1] - self.locations_cpu[j, 1]) ** 2
                )

    def _center_box_bounds(self):
        half = self.user_center_size * 0.5
        center = self.base_area_size * 0.5
        lo = center - half
        hi = center + half
        return lo, hi

    def _constrain_users_to_center_box(self):
        if not self.constrain_users_center:
            return
        lo, hi = self._center_box_bounds()
        self.locations_users = np.clip(self.locations_users, lo, hi)
        if self.user_dest is not None:
            self.user_dest = np.clip(self.user_dest, lo, hi)

    def _scale_state(self, to_area: bool):
        if self.area_size == self.base_area_size:
            return
        if to_area and self._scaled_to_area:
            return
        if (not to_area) and (not self._scaled_to_area):
            return

        factor = self.area_size / self.base_area_size
        if not to_area:
            factor = 1.0 / factor

        self.locations_users *= factor
        self.locations_aps *= factor
        self.locations_cpu *= factor
        if self.user_dest is not None:
            self.user_dest *= factor
        # Keep speed magnitude in the original 900x900 scale even when probing
        # a different area size.

        self._scaled_to_area = to_area

    def _recompute_large_scale(self):
        # Access pathloss / beta
        diff = self.locations_users[:, np.newaxis, :] - self.locations_aps[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        d_km = self.distance_matrix / 1000.0

        far_mask = self.distance_matrix > self.d_1
        mid_mask = (self.distance_matrix >= self.d_0) & (self.distance_matrix <= self.d_1)
        near_mask = self.distance_matrix < self.d_0

        if np.any(far_mask):
            self.PL[far_mask] = -self.L - 35 * np.log10(d_km[far_mask])
        if np.any(mid_mask):
            term = (self.d_1 / 1000.0) ** 1.5 * (d_km[mid_mask] ** 2)
            self.PL[mid_mask] = -self.L - 10 * np.log10(term)
        if np.any(near_mask):
            term = (self.d_1 / 1000.0) ** 1.5 * (self.d_0 / 1000.0) ** 2
            self.PL[near_mask] = -self.L - 10 * np.log10(term)

        self.beta = np.power(10, self.PL / 10.0) * np.power(10, (self.sigma_s * self.mu) / 10.0)

        # Fronthaul distance
        self.distance_matrix_front = np.zeros([self.N, self.K])
        for i in range(self.N):
            for j in range(self.K):
                self.distance_matrix_front[i, j] = math.sqrt(
                    (self.locations_aps[i, 0] - self.locations_cpu[j, 0]) ** 2
                    + (self.locations_aps[i, 1] - self.locations_cpu[j, 1]) ** 2
                )

    def _record_top10(self, step_idx: int):
        beta_sim = self.beta[: self.M_sim, : self.N_sim]
        top10_idx = np.argsort(beta_sim, axis=1)[:, ::-1][:, :10]
        top1_idx = top10_idx[:, 0].astype(np.int32, copy=False)

        self._top10_steps += 1
        self._top10_slots += self.M_sim * 10

        for i in range(self.M_sim):
            idxs = top10_idx[i]
            vals = beta_sim[i, idxs]
            self._ap_top10_count[idxs] += 1
            self._ap_top10_beta_sum[idxs] += vals
            self._all_top10_betas.extend(vals.tolist())
            denom = float(np.sum(vals) + 1e-18)
            shares = vals / denom
            self._rank_beta_share_sum += shares
            self._top1_share_samples.append(float(shares[0]))
            self._top2_share_samples.append(float(shares[0] + shares[1]))

        # Step-to-step top1 AP stability.
        if self._prev_top1_idx is not None:
            self._top1_same_prev_step_count += int(np.sum(top1_idx == self._prev_top1_idx))
            self._top1_prev_step_comp_count += int(self.M_sim)
        self._prev_top1_idx = top1_idx.copy()

        # Interval/update-level stability. Ignore reset snapshot (step 0).
        if step_idx >= 1:
            is_cluster_update = (step_idx == 1) or (step_idx % self.cluster_update_interval == 0)
            if is_cluster_update:
                # Finalize previous interval at update boundaries.
                if self._interval_active:
                    self._interval_switch_sum += int(np.sum(self._curr_interval_switch_count))
                    self._interval_user_count += int(self.M_sim)
                    self._interval_constant_count += int(np.sum(self._curr_interval_switch_count == 0))
                    self._interval_count += 1

                if self._prev_update_top1_idx is not None:
                    self._top1_same_update_count += int(np.sum(top1_idx == self._prev_update_top1_idx))
                    self._top1_update_comp_count += int(self.M_sim)
                self._prev_update_top1_idx = top1_idx.copy()

                # Start new interval from this update point.
                self._interval_active = True
                self._curr_interval_prev_top1 = top1_idx.copy()
                self._curr_interval_switch_count[:] = 0
            elif self._interval_active:
                switched = (top1_idx != self._curr_interval_prev_top1)
                self._curr_interval_switch_count += switched.astype(np.int64)
                self._curr_interval_prev_top1 = top1_idx.copy()

        if self.print_top10 and (step_idx % self.print_every == 0):
            show_users = min(self.print_users, self.M_sim)
            for u in range(show_users):
                idxs = top10_idx[u].tolist()
                vals = beta_sim[u, top10_idx[u]]
                vals_db = 10.0 * np.log10(vals + 1e-12)
                vals_db = np.round(vals_db, 2).tolist()
                print(f"[BETA-PROBE] step={step_idx} user{u} top10_ap={idxs}")
                print(f"[BETA-PROBE] step={step_idx} user{u} top10_beta_db={vals_db}")

    def reset_probe_stats(self):
        self._top10_steps = 0
        self._top10_slots = 0
        self._ap_top10_count[:] = 0
        self._ap_top10_beta_sum[:] = 0.0
        self._all_top10_betas = []
        self._rank_beta_share_sum[:] = 0.0
        self._top1_share_samples = []
        self._top2_share_samples = []
        self._prev_top1_idx = None
        self._top1_same_prev_step_count = 0
        self._top1_prev_step_comp_count = 0
        self._prev_update_top1_idx = None
        self._top1_same_update_count = 0
        self._top1_update_comp_count = 0
        self._interval_active = False
        self._curr_interval_prev_top1 = None
        self._curr_interval_switch_count[:] = 0
        self._interval_switch_sum = 0
        self._interval_user_count = 0
        self._interval_constant_count = 0
        self._interval_count = 0

    def get_probe_summary(self):
        appear_ratio = (
            self._ap_top10_count.astype(np.float64) / max(1, self._top10_slots)
        )  # per AP share in all top10 slots
        mean_beta_when_top10 = np.divide(
            self._ap_top10_beta_sum,
            np.maximum(self._ap_top10_count, 1),
            out=np.zeros_like(self._ap_top10_beta_sum, dtype=np.float64),
            where=self._ap_top10_count > 0,
        )
        all_betas = np.array(self._all_top10_betas, dtype=np.float64) if self._all_top10_betas else np.array([0.0])
        rank_den = max(1, self._top10_steps * self.M_sim)
        rank_share_mean = self._rank_beta_share_sum / rank_den
        top1_arr = np.array(self._top1_share_samples, dtype=np.float64) if self._top1_share_samples else np.array([0.0])
        top2_arr = np.array(self._top2_share_samples, dtype=np.float64) if self._top2_share_samples else np.array([0.0])
        interval_switch_sum = int(self._interval_switch_sum)
        interval_user_count = int(self._interval_user_count)
        interval_constant_count = int(self._interval_constant_count)
        interval_count = int(self._interval_count)
        # Include the currently open interval in final summary.
        if self._interval_active:
            interval_switch_sum += int(np.sum(self._curr_interval_switch_count))
            interval_user_count += int(self.M_sim)
            interval_constant_count += int(np.sum(self._curr_interval_switch_count == 0))
            interval_count += 1

        top1_same_prev_step_ratio = (
            float(self._top1_same_prev_step_count) / float(max(1, self._top1_prev_step_comp_count))
        )
        top1_same_update_ratio = (
            float(self._top1_same_update_count) / float(max(1, self._top1_update_comp_count))
        )
        top1_switch_per_user_per_interval_mean = (
            float(interval_switch_sum) / float(max(1, interval_user_count))
        )
        top1_constant_interval_ratio = (
            float(interval_constant_count) / float(max(1, interval_user_count))
        )

        return {
            "steps": int(self._top10_steps),
            "slots": int(self._top10_slots),
            "ap_top10_count": self._ap_top10_count.copy(),
            "ap_top10_share": appear_ratio,
            "ap_top10_beta_mean": mean_beta_when_top10,
            "all_top10_beta_p50": float(np.percentile(all_betas, 50)),
            "all_top10_beta_p95": float(np.percentile(all_betas, 95)),
            "all_top10_beta_p99": float(np.percentile(all_betas, 99)),
            "rank_beta_share_mean": rank_share_mean,
            "top1_share_p50": float(np.percentile(top1_arr, 50)),
            "top1_share_p95": float(np.percentile(top1_arr, 95)),
            "top2_share_p50": float(np.percentile(top2_arr, 50)),
            "top2_share_p95": float(np.percentile(top2_arr, 95)),
            "top1_same_as_prev_step_ratio": top1_same_prev_step_ratio,
            "top1_same_as_prev_cluster_update_ratio": top1_same_update_ratio,
            "top1_switch_count_per_user_per_interval_mean": top1_switch_per_user_per_interval_mean,
            "top1_constant_interval_ratio": top1_constant_interval_ratio,
            "interval_count_evaluated": interval_count,
        }

    def reset(self):
        obs = super().reset()
        self._constrain_users_to_center_box()
        self._scale_state(to_area=True)
        self._recompute_large_scale()
        self._record_top10(step_idx=0)
        return obs

    def step(self, action):
        # Base env dynamics are implemented in 900x900 assumptions. Convert state
        # to base scale before step, then scale back and recompute beta for probing.
        self._scale_state(to_area=False)
        out = super().step(action)
        self._constrain_users_to_center_box()
        self._scale_state(to_area=True)
        self._recompute_large_scale()
        self._record_top10(step_idx=self.step_num)
        return out


def _parse_args():
    parser = argparse.ArgumentParser("beta top10 probe")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--area_size", type=float, default=300.0)
    parser.add_argument("--n_sim", type=int, default=100)
    parser.add_argument("--cluster_size", type=int, default=4)
    parser.add_argument("--user_center_size", type=float, default=600.0)
    parser.add_argument("--constrain_users_center", action="store_true", default=False)
    parser.add_argument("--no_constrain_users_center", action="store_false", dest="constrain_users_center")
    parser.add_argument("--print_top10", action="store_true", default=False)
    parser.add_argument("--print_users", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--top_ap_to_show", type=int, default=20)
    return parser.parse_args()


def _run_probe(args):
    env = MA_UCMEC_dyna_noncoop_beta_probe(
        seed=args.seed,
        area_size=args.area_size,
        n_sim=args.n_sim,
        user_center_size=args.user_center_size,
        constrain_users_center=args.constrain_users_center,
        print_top10=args.print_top10,
        print_users=args.print_users,
        print_every=args.print_every,
    )
    env.cluster_size = int(max(1, min(args.cluster_size, 10)))
    env.reset_probe_stats()

    for _ in range(args.episodes):
        _ = env.reset()
        for _ in range(args.episode_length):
            # Match training/eval action format: per-user one-hot over 10 low-level actions.
            action = []
            for _u in range(env.M_sim):
                a = np.zeros((env.action_dim,), dtype=np.int32)
                idx = int(env.rng.integers(0, env.action_dim))
                a[idx] = 1
                action.append(a)
            _, _, done, _ = env.step(action)
            if np.all(done):
                break

    summary = env.get_probe_summary()
    ap_share = summary["ap_top10_share"]
    ap_count = summary["ap_top10_count"]
    ap_beta_mean = summary["ap_top10_beta_mean"]
    top_k = min(max(1, args.top_ap_to_show), len(ap_share))
    top_idx = np.argsort(ap_share)[::-1][:top_k]

    print("=== Top10 Beta Probe Summary ===")
    print(f"steps: {summary['steps']}, slots: {summary['slots']}")
    print(
        "all_top10_beta percentiles:",
        f"p50={summary['all_top10_beta_p50']:.6e},",
        f"p95={summary['all_top10_beta_p95']:.6e},",
        f"p99={summary['all_top10_beta_p99']:.6e}",
    )
    print("\nRank-wise beta share mean in top10 (rank1..rank10):")
    print(np.array2string(summary["rank_beta_share_mean"], precision=4, separator=", "))
    print(
        f"top1 share p50/p95: {summary['top1_share_p50']:.4f} / {summary['top1_share_p95']:.4f}"
    )
    print(
        f"top2 cumulative share p50/p95: {summary['top2_share_p50']:.4f} / {summary['top2_share_p95']:.4f}"
    )
    print("\nTop1 AP stability:")
    print(
        f"same_as_prev_step_ratio: {summary['top1_same_as_prev_step_ratio']:.4f}"
    )
    print(
        f"same_as_prev_cluster_update_ratio: "
        f"{summary['top1_same_as_prev_cluster_update_ratio']:.4f}"
    )
    print(
        f"switch_count_per_user_per_interval_mean: "
        f"{summary['top1_switch_count_per_user_per_interval_mean']:.4f}"
    )
    print(
        f"constant_interval_ratio: {summary['top1_constant_interval_ratio']:.4f} "
        f"(intervals={summary['interval_count_evaluated']})"
    )
    print("\nTop APs by top10-share:")
    for ap in top_idx:
        print(
            f"ap{ap:02d}: share={ap_share[ap]:.6f}, count={int(ap_count[ap])}, "
            f"beta_mean_if_top10={ap_beta_mean[ap]:.6e}"
        )


if __name__ == "__main__":
    _run_probe(_parse_args())
