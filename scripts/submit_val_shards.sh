#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p runs
SHARDS=(
  "val_0_1000:configs/val_0_1000.json"
  "val_1000_2000:configs/val_1000_2000.json"
  "val_2000_3000:configs/val_2000_3000.json"
  "val_3000_4183:configs/val_3000_4183.json"
)
echo "Submitting ${#SHARDS[@]} shards from $ROOT"
for entry in "${SHARDS[@]}"; do
  name="${entry%%:*}"; cfg="${entry#*:}"
  test -f "$cfg"
  jid=$(sbatch --parsable --job-name="med_${name}" --export=ALL,CONFIG="$cfg" scripts/run_val_shard.sh)
  echo "  $name -> job $jid"
done
squeue -u "$USER" -o "%.18i %.12j %.8T %.10M %R" || true
