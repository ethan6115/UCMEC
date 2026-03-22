"""
測試不同 varsig / P_max 和 cluster size 下的 uplink rate
目標：確認哪個參數能讓多 AP 的增益更顯著
執行方式：
    python test_varsig_pmax_cluster.py --mode varsig   # 測試天線數
    python test_varsig_pmax_cluster.py --mode pmax     # 測試發射功率
    python test_varsig_pmax_cluster.py --mode both     # 兩個都測
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MA_UCMEC_dyna_noncoop_big_3_2 import MA_UCMEC_dyna_noncoop
#from MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop


N_SEEDS = 5
CLUSTER_SIZES = [1, 2, 3, 4, 5]
VARSIG_LIST = [16, 8, 4, 2, 1]
PMAX_LIST   = [0.1]

# action index 3 = CPU1 + p_level3 (最高功率)
ACTION_IDX = 3


def compute_theta(env):
    """根據當前 beta 計算 MMSE theta，和 env.step() 內部公式一致。"""
    return (env.tau_p * env.P_max * (env.beta ** 2)) / (
        env.tau_p * env.P_max * env.beta + env.noise_access
    )


def compute_sinr_user0(env, theta, cluster_matrix, p):
    """計算 user0 的 SINR 作為代表值。"""
    i = 0
    sum_theta = sum(theta[i, j] for j in range(env.N_sim) if cluster_matrix[i, j] == 1)
    noise_term = env.noise_access * sum_theta
    useful = (sum_theta ** 2) * p[i] * env.varsig
    inter_term = sum(
        theta[i, j] * env.beta[k, j] * p[k]
        for k in range(env.M_sim) if k != i
        for j in range(env.N_sim) if cluster_matrix[i, j] == 1
    )
    denom = inter_term + noise_term
    return useful / denom if denom > 0 else 0.0


def compute_sinr_all_users_mean(env, theta, cluster_matrix, p):
    sinrs = []
    for i in range(env.M_sim):
        sum_theta = sum(theta[i, j] for j in range(env.N_sim) if cluster_matrix[i, j] == 1)
        noise_term = env.noise_access * sum_theta
        useful = (sum_theta ** 2) * p[i] * env.varsig
        inter_term = sum(
            theta[i, j] * env.beta[k, j] * p[k]
            for k in range(env.M_sim) if k != i
            for j in range(env.N_sim) if cluster_matrix[i, j] == 1
        )
        denom = inter_term + noise_term
        sinr_i = useful / denom if denom > 0 else 0.0
        sinrs.append(sinr_i)
        if(i == 0):
            theta_i = [theta[i, j] for j in range(env.N_sim) if cluster_matrix[i, j] == 1]
            cluster_i = [j for j in range(env.N_sim) if cluster_matrix[i, j] == 1]
            print(f"User0: useful={useful:.2e}, inter={inter_term:.2e}, noise={noise_term:.2e}, SINR={sinr_i:.2f}")
            print(f"  theta: {theta_i}")
            print(f"  cluster APs: {cluster_i}")
    return float(np.mean(sinrs)) if len(sinrs) > 0 else 0.0


def run_one_seed(seed, varsig, p_max, cluster_size):
    env = MA_UCMEC_dyna_noncoop(seed=seed)
    env.varsig = varsig
    env.P_max  = p_max
    env.noise_access = env.noise_access*1000
    env.reset()

    # 跑一個 step 讓 beta、distance_matrix 更新
    dummy_action = [np.eye(env.action_dim, dtype=np.int32)[ACTION_IDX]] * env.M_sim
    env.step(dummy_action)

    theta = compute_theta(env)

    env.cluster_size   = cluster_size
    env.cluster_matrix = None
    cluster_matrix     = env.cluster()

    omega = np.ones(env.M_sim, dtype=int)    # 全部 offload CPU1
    p     = np.full(env.M_sim, env.P_max)   # 最高功率

    uplink_rate = env.uplink_rate_cal(p, omega, cluster_matrix, theta)
    valid       = uplink_rate[:, 0][uplink_rate[:, 0] > 1e-6]
    avg_mbps    = float(np.mean(valid)) / 1e6 if len(valid) > 0 else 0.0
    sinr_u0     = compute_sinr_user0(env, theta, cluster_matrix, p)
    sinr_mean   = compute_sinr_all_users_mean(env, theta, cluster_matrix, p)

    return avg_mbps, sinr_u0, sinr_mean


def run_experiment(param_name, param_list, fixed_varsig=16, fixed_pmax=0.1):
    """
    param_name: 'varsig' 或 'pmax'
    param_list: 要掃的值列表
    """
    results = {pv: {c: [] for c in CLUSTER_SIZES} for pv in param_list}

    for pv in param_list:
        varsig = pv           if param_name == 'varsig' else fixed_varsig
        p_max  = fixed_pmax   if param_name == 'varsig' else pv
        for cluster_size in CLUSTER_SIZES:
            print(f"cluster_size={cluster_size}")
            for seed in range(N_SEEDS):
                print(f"seed={seed}")
                rate, sinr_u0, sinr_mean = run_one_seed(seed, varsig, p_max, cluster_size)
                results[pv][cluster_size].append((rate, sinr_u0, sinr_mean))

    # 印表頭
    label = 'varsig' if param_name == 'varsig' else 'P_max'
    print(f"\n{'=' * 70}")
    print(f"測試變數: {label}  (固定 varsig={fixed_varsig}, P_max={fixed_pmax})" 
          if param_name == 'pmax' else
          f"測試變數: {label}  (固定 P_max={fixed_pmax})")
    print(f"{'=' * 70}")
    print(
        f"{'param':>12} {'cluster':>8} {'avg_rate(Mbps)':>16} "
        f"{'gain vs c1':>12} {'SINR_u0':>10} {'SINR_all_mean':>14}"
    )
    print(f"{'-' * 70}")

    for pv in param_list:
        base_rate = np.mean([r for r, _, _ in results[pv][1]])
        for c in CLUSTER_SIZES:
            rates = [r for r, _, _ in results[pv][c]]
            sinrs_u0 = [s for _, s, _ in results[pv][c]]
            sinrs_all = [s for _, _, s in results[pv][c]]
            avg_r = np.mean(rates)
            avg_s_u0 = np.mean(sinrs_u0)
            avg_s_all = np.mean(sinrs_all)
            gain  = (avg_r / base_rate - 1.0) * 100 if base_rate > 0 else 0.0
            pv_str = f"{pv}" if param_name == 'varsig' else f"{pv:.0e}"
            print(
                f"{pv_str:>12} {c:>8} {avg_r:>16.2f} {gain:>11.1f}% "
                f"{avg_s_u0:>10.2f} {avg_s_all:>14.2f}"
            )
        print()

    # 摘要：cluster2 vs cluster1 增益
    print(f"{'=' * 70}")
    print(f"摘要：cluster2 vs cluster1 的 uplink rate 增益%")
    print(f"{'-' * 70}")
    for pv in param_list:
        r1   = np.mean([r for r, _, _ in results[pv][1]])
        r2   = np.mean([r for r, _, _ in results[pv][2]])
        gain = (r2 / r1 - 1.0) * 100 if r1 > 0 else 0.0
        s1_u0 = np.mean([s for _, s, _ in results[pv][1]])
        s1_all = np.mean([s for _, _, s in results[pv][1]])
        pv_str = f"{pv}" if param_name == 'varsig' else f"{pv:.0e}"
        print(f"{label}={pv_str:>8}: c1={r1:.2f} Mbps, c2={r2:.2f} Mbps, "
              f"gain={gain:+.1f}%, SINR_u0_c1={s1_u0:.2f}, SINR_all_c1={s1_all:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["varsig", "pmax", "both"], default="pmax")
    args = parser.parse_args()

    if args.mode in ("varsig", "both"):
        run_experiment("varsig", VARSIG_LIST)

    if args.mode in ("pmax", "both"):
        run_experiment("pmax", PMAX_LIST)
