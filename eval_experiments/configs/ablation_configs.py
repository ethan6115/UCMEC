"""Ablation configurations for eval_ablation.py.

The evaluator itself remains eval_script.py.  These configs only describe
which checkpoints and environment variables should be used for each ablation.
Defaults are explicit so main/sweep experiments keep their original behavior.
"""


def _hier_env(high_policy_mode="trained", candidate_n="10", k_fixed="2"):
    return {
        "EVAL_USE_FLAT_JOINT": "0",
        "EVAL_USE_HIERARCHICAL": "1",
        "EVAL_PER_USER": "1",
        "EVAL_HIGH_POLICY_MODE": high_policy_mode,
        "EVAL_POWER_VARIANT": "normal",
        "UCMEC_CANDIDATE_N": str(candidate_n),
        "UCMEC_K_FIXED": str(k_fixed),
        "UCMEC_M_SIM": "10",
        "UCMEC_EPSILON": "0.003",
        "UCMEC_N_SIM": "50",
    }


def _low_env():
    return {
        "EVAL_USE_FLAT_JOINT": "0",
        "EVAL_USE_HIERARCHICAL": "0",
        "EVAL_PER_USER": "1",
        "EVAL_HIGH_POLICY_MODE": "none",
        "EVAL_POWER_VARIANT": "normal",
        "UCMEC_M_SIM": "10",
        "UCMEC_EPSILON": "0.003",
        "UCMEC_N_SIM": "50",
    }


def _add_hier_runs(configs, group, variant, runs, base_dir, env):
    for run in runs:
        configs.append(
            {
                "group": group,
                "variant": variant,
                "run": run,
                "name": f"{group}_{variant}_{run}",
                "env": {
                    **env,
                    "EVAL_MODEL_LOW": f"{base_dir}/{run}/models/actor_499.pt",
                    "EVAL_MODEL_HIGH": f"{base_dir}/{run}/models/actor_high.pt",
                    "EVAL_MODEL_FLAT": "",
                },
            }
        )


def _add_heuristic_hier_runs(configs, group, variant, runs, base_dir, env):
    for run in runs:
        configs.append(
            {
                "group": group,
                "variant": variant,
                "run": run,
                "name": f"{group}_{variant}_{run}",
                "env": {
                    **env,
                    "EVAL_MODEL_LOW": f"{base_dir}/{run}/models/actor_499.pt",
                    "EVAL_MODEL_HIGH": "",
                    "EVAL_MODEL_FLAT": "",
                },
            }
        )


def _add_low_runs(configs, group, variant, runs, base_dir, env):
    for run in runs:
        configs.append(
            {
                "group": group,
                "variant": variant,
                "run": run,
                "name": f"{group}_{variant}_{run}",
                "env": {
                    **env,
                    "EVAL_MODEL_LOW": f"{base_dir}/{run}/models/actor_499.pt",
                    "EVAL_MODEL_HIGH": "",
                    "EVAL_MODEL_FLAT": "",
                },
            }
        )


ABLATION_CONFIGS = []

for candidate_n in ("5", "8", "12", "15"):
    _add_hier_runs(
        ABLATION_CONFIGS,
        group="candidate_size",
        variant=f"candidate{candidate_n}",
        runs=["run1", "run2", "run3"],
        base_dir=(
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/"
            f"hierarchical_pair_scorer_noglobal_candidate{candidate_n}"
        ),
        env=_hier_env(high_policy_mode="trained", candidate_n=candidate_n, k_fixed="2"),
    )

_add_heuristic_hier_runs(
    ABLATION_CONFIGS,
    group="cluster_size",
    variant="cluster1",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_cluster1",
    env=_hier_env(high_policy_mode="baseline_topk", candidate_n="10", k_fixed="1"),
)

_add_heuristic_hier_runs(
    ABLATION_CONFIGS,
    group="cluster_size",
    variant="cluster3",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_cluster3",
    env=_hier_env(high_policy_mode="baseline_topk", candidate_n="10", k_fixed="3"),
)

_add_heuristic_hier_runs(
    ABLATION_CONFIGS,
    group="cluster_size",
    variant="cluster1_blockage7e-3",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_cluster1",
    env={
        **_hier_env(high_policy_mode="baseline_topk", candidate_n="10", k_fixed="1"),
        "UCMEC_EPSILON": "0.007",
    },
)

_add_hier_runs(
    ABLATION_CONFIGS,
    group="high_ablation",
    variant="without_pair_scorer",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_highlow",
    env=_hier_env(high_policy_mode="trained", candidate_n="10", k_fixed="2"),
)

_add_hier_runs(
    ABLATION_CONFIGS,
    group="high_ablation",
    variant="without_pair_interaction",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_pairconcat",
    env=_hier_env(high_policy_mode="trained", candidate_n="10", k_fixed="2"),
)

_add_low_runs(
    ABLATION_CONFIGS,
    group="low_ablation",
    variant="without_front_obs",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_nofrontobs",
    env=_low_env(),
)
