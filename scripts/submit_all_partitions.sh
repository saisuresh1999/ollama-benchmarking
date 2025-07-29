#!/bin/bash

# === Config ===
BASE_JOB="slurm_templates/run_llm_anomaly_template.slurm"
MODEL_NAME="deepseek-r1-8b"
SAVE_DIR="slurm_jobs/generated"
mkdir -p "$SAVE_DIR"

declare -A PART_GPU_MAP=(
  #[L40S]=1
  [RTXA6000]=1
)

echo "[INFO] Checking available partitions..."
HEALTHY_PARTS=()
for PART in "${!PART_GPU_MAP[@]}"; do
  NODE_STATES=$(sinfo -N -p "$PART" -o "%t" | tail -n +2)
  if echo "$NODE_STATES" | grep -Eq "idle|mix"; then
    HEALTHY_PARTS+=("$PART")
  else
    echo "⛔ Skipping $PART: No healthy nodes"
  fi
done

echo "[INFO] Available partitions: ${HEALTHY_PARTS[*]}"

for PART in "${HEALTHY_PARTS[@]}"; do
  GPUS=${PART_GPU_MAP[$PART]}
  MAX_TIME="14:00:00"

  JOB_FILE="${SAVE_DIR}/run_${MODEL_NAME}_${PART}.slurm"
  cp "$BASE_JOB" "$JOB_FILE"

  sed -i "s/^#SBATCH --partition=.*/#SBATCH --partition=${PART}/" "$JOB_FILE"
  sed -i "s/^#SBATCH --gres=gpu:.*/#SBATCH --gres=gpu:${GPUS}/" "$JOB_FILE"
  sed -i "s/^#SBATCH --time=.*/#SBATCH --time=${MAX_TIME}/" "$JOB_FILE"
  sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=${MODEL_NAME}-${PART}/" "$JOB_FILE"
  sed -i "s|^#SBATCH --output=.*|#SBATCH --output=logs/${MODEL_NAME}_${PART}_anomaly_%j.out|" "$JOB_FILE"

  echo "🚀 Submitting job for $PART"
  sbatch "$JOB_FILE"
done
