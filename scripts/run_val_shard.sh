#!/bin/bash
#SBATCH -p batch-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -J medmcqa_val
#SBATCH -o runs/slurm_val_%x_%j.out
#SBATCH -e runs/slurm_val_%x_%j.err
#SBATCH -D /network/rit/lab/Lai_ReSecureAI/kiel/AD/Run_code/medmcqa_lifecycle
#
#   sbatch --job-name=val_0_1000 --export=ALL,CONFIG=configs/val_0_1000.json scripts/run_val_shard.sh

set -euo pipefail
CONFIG="${CONFIG:-configs/val_0_1000.json}"
RUN_NAME="${RUN_NAME:-}"

export HF_HOME=/network/rit/lab/Lai_ReSecureAI/kiel/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
if [[ -z "${HF_TOKEN:-}" ]]; then
  for cand in "${HOME}/.cache/huggingface/token" "${HOME}/.huggingface/token"; do
    [[ -f "$cand" ]] && export HF_TOKEN="$(tr -d '[:space:]' < "$cand")" && break
  done
fi
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

MODEL_PATH=$HF_HOME/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
source /network/rit/lab/Lai_ReSecureAI/kiel/anaconda3/etc/profile.d/conda.sh
conda activate ad

hostname; nvidia-smi -L
echo "CONFIG=$CONFIG HF_TOKEN_SET=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
test -f "$MODEL_PATH/config.json"
test -f "$CONFIG"

EXTRA=()
[[ -n "$RUN_NAME" ]] && EXTRA+=(--run-name "$RUN_NAME")

python run_experiment.py \
  --config "$CONFIG" \
  --model-path "$MODEL_PATH" \
  --llm-batch-size 1 \
  --no-load-in-4bit \
  "${EXTRA[@]}"
