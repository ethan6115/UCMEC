#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train/run_training.sh \
    --mode <MODE> \
    --experiment_name <NAME> \
    --scenario_name <NAME> \
    [additional train.py arguments]

Modes:
  low                Standard low-level training
  greedy             Low-level training with the selected greedy environment
  flat               Flat joint-policy training
  hierarchical-mlp   Hierarchical training with the MLP high-level actor
  hierarchical-pair  Hierarchical training with the pair-scoring high-level actor

Options:
  --dry-run           Print the final train.py command without running it
  -h, --help          Show this help message

Output directory:
  results/MyEnv/<scenario_name>/rmappo/<experiment_name>/run<N>/
EOF
}

mode=""
experiment_name=""
scenario_name=""
dry_run=false
extra_args=()

while (($# > 0)); do
    case "$1" in
        --mode)
            (($# >= 2)) || { echo "Error: --mode requires a value." >&2; exit 2; }
            mode="$2"
            shift 2
            ;;
        --mode=*)
            mode="${1#*=}"
            shift
            ;;
        --experiment_name)
            (($# >= 2)) || { echo "Error: --experiment_name requires a value." >&2; exit 2; }
            experiment_name="$2"
            shift 2
            ;;
        --experiment_name=*)
            experiment_name="${1#*=}"
            shift
            ;;
        --scenario_name)
            (($# >= 2)) || { echo "Error: --scenario_name requires a value." >&2; exit 2; }
            scenario_name="$2"
            shift 2
            ;;
        --scenario_name=*)
            scenario_name="${1#*=}"
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args+=("$@")
            break
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

[[ -n "$mode" ]] || { echo "Error: --mode is required." >&2; usage >&2; exit 2; }
[[ -n "$experiment_name" ]] || { echo "Error: --experiment_name is required." >&2; usage >&2; exit 2; }
[[ -n "$scenario_name" ]] || { echo "Error: --scenario_name is required." >&2; usage >&2; exit 2; }

common_args=(
    --use_centralized_V
    --use_recurrent_policy
    --use_linear_lr_decay
    --use_proper_time_limits
)

mode_args=()
case "$mode" in
    low)
        ;;
    greedy)
        mode_args+=(--use_greedy_training_env)
        ;;
    flat)
        mode_args+=(--use_joint_policy)
        ;;
    hierarchical-mlp)
        mode_args+=(
            --use_hierarchical
            --use_high_peruser
            --use_high_peruser_credit
            --high_encoder_type noattn-mp
            --high_actor_type mlp
        )
        ;;
    hierarchical-pair)
        mode_args+=(
            --use_hierarchical
            --use_high_peruser
            --use_high_peruser_credit
            --high_encoder_type noattn-mp
            --high_actor_type pair_scorer
            --high_no_global_ctx
        )
        ;;
    *)
        echo "Error: unsupported mode '$mode'." >&2
        usage >&2
        exit 2
        ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

command=(
    python train/train.py
    --experiment_name "$experiment_name"
    --scenario_name "$scenario_name"
    "${common_args[@]}"
    "${mode_args[@]}"
    "${extra_args[@]}"
)

if [[ "$dry_run" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
