#!/bin/bash
# Submit medical-domain MedMCQA val shard [0,1000) (12h GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p runs
test -f configs/val_medical_domain_0_1000.json
JID=$(sbatch --parsable \
  --job-name=med_domain_0_1000 \
  --export=ALL,CONFIG=configs/val_medical_domain_0_1000.json \
  scripts/run_val_shard.sh)
echo "SUBMITTED job $JID"
echo "  config=configs/val_medical_domain_0_1000.json"
echo "  out=runs/slurm_val_med_domain_0_1000_${JID}.out  (pattern: runs/slurm_val_%x_%j.out)"
echo "  results→ runs/final_val_medmcqa/medmcqa_val_medical_domain_0_1000_bf16/"
squeue -u "$USER" -o "%.18i %.16j %.8T %.10M %R" || true
