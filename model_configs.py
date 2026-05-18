"""Model configurations for eval_sweep.py.

Each config describes one trained model run and the environment variables used
to drive eval_script.py. eval_script.py remains the single-run evaluator; the
sweep script expands these configs across scenario settings.
"""


def _add_hier_runs(configs, method, runs, base_dir, high_policy_mode, power_variant="normal"):
    for run in runs:
        configs.append(
            {
                "method": method,
                "run": run,
                "name": f"{method}_{run}",
                "env": {
                    "EVAL_USE_FLAT_JOINT": "0",
                    "EVAL_USE_HIERARCHICAL": "1",
                    "EVAL_HIGH_POLICY_MODE": high_policy_mode,
                    "EVAL_POWER_VARIANT": power_variant,
                    "EVAL_MODEL_LOW": f"{base_dir}/{run}/models/actor_499.pt",
                    "EVAL_MODEL_HIGH": f"{base_dir}/{run}/models/actor_high.pt",
                },
            }
        )


def _add_low_heuristic_runs(configs, method, runs, base_dir, high_policy_mode):
    for run in runs:
        configs.append(
            {
                "method": method,
                "run": run,
                "name": f"{method}_{run}",
                "env": {
                    "EVAL_USE_FLAT_JOINT": "0",
                    "EVAL_USE_HIERARCHICAL": "1",
                    "EVAL_HIGH_POLICY_MODE": high_policy_mode,
                    "EVAL_POWER_VARIANT": "normal",
                    "EVAL_MODEL_LOW": f"{base_dir}/{run}/models/actor_499.pt",
                    "EVAL_MODEL_HIGH": "",
                },
            }
        )


def _add_flat_runs(configs, method, runs, base_dir):
    for run in runs:
        configs.append(
            {
                "method": method,
                "run": run,
                "name": f"{method}_{run}",
                "env": {
                    "EVAL_USE_FLAT_JOINT": "1",
                    "EVAL_USE_HIERARCHICAL": "0",
                    "EVAL_HIGH_POLICY_MODE": "none",
                    "EVAL_POWER_VARIANT": "normal",
                    "EVAL_MODEL_FLAT": f"{base_dir}/{run}/models/actor_499.pt",
                    "EVAL_MODEL_LOW": "",
                    "EVAL_MODEL_HIGH": "",
                },
            }
        )


MODEL_CONFIGS = []

_add_hier_runs(
    MODEL_CONFIGS,
    method="Proposed_HDRL",
    runs=["run2", "run3", "run5"],
    base_dir="results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal",
    high_policy_mode="trained",
    power_variant="normal",
)

_add_low_heuristic_runs(
    MODEL_CONFIGS,
    method="AccessGreedy",
    runs=["run2", "run3", "run4"],
    base_dir="results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn",
    high_policy_mode="baseline_topk",
)

_add_low_heuristic_runs(
    MODEL_CONFIGS,
    method="FrontGreedyTopL",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_bestfront",
    high_policy_mode="best_front",
)

_add_flat_runs(
    MODEL_CONFIGS,
    method="FlatDRL",
    runs=["run1", "run2", "run3"],
    base_dir="results/MyEnv/nlos_cluster_v2/rmappo/flat_drl",
)

_add_hier_runs(
    MODEL_CONFIGS,
    method="MaxPower",
    runs=["run2", "run3", "run4"],
    base_dir="results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_highlow_maxpower",
    high_policy_mode="trained",
    power_variant="fixed_max",
)

_add_hier_runs(
    MODEL_CONFIGS,
    method="MinPower",
    runs=["run2", "run3", "run4"],
    base_dir="results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_highlow_minpower",
    high_policy_mode="trained",
    power_variant="fixed_min",
)
