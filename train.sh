#!/usr/bin/env bash
set -e

# Usage
usage() {
  echo "Usage: $0 <VLA_NAME> [-n GPU_NODE] [-g NUM_GPUS]"
  echo ""
  echo "Example:"
  echo "  $0 pi05"
  echo "  $0 pi05 -n 21"
  echo "  $0 pi05 -g 4"
  echo "  $0 pi05 -n 21 -g 2"
  exit 1
}

# Check Arguments
if [ -z "$1" ]; then
  usage
fi

VLA="$1"
shift

# Default Config
GPU_NODE="rmc-gpu21"   # default cluster
NUM_GPUS=1             # default number of GPUs

# Parse Optional Flags
while getopts ":n:g:" opt; do
  case ${opt} in
    n )
      GPU_NODE="rmc-gpu$OPTARG"
      ;;
    g )
      NUM_GPUS="$OPTARG"
      ;;
    \? )
      echo "Invalid option: -$OPTARG"
      usage
      ;;
    : )
      echo "Option -$OPTARG requires an argument."
      usage
      ;;
  esac
done

# Config
BASE_DIR="/home/$USER/workspace/openpi/scripts/train_pi05_lwr.sh"
SLURM_MANAGER=~/scripts/slurm_scripts/slurm_manager/scripts/start_job.py
PYTHON_BIN=~/miniforge3/envs/checkCluster/bin/python3
OUTPUT_DIR="/volume/hot_storage/slurm_data/$USER/output/train"

# Validate VLA Script Exists
TRAIN_SCRIPT="${BASE_DIR}"

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Training script not found: $TRAIN_SCRIPT"
  exit 1
fi

# Default GPU if not provided
if [ -z "$GPU_NODE" ]; then
  GPU_NODE="rmc-gpu21"   # default cluster
fi

# Build Job Name
JOB_NAME="${VLA}_finetune"

# Start Job
$PYTHON_BIN $SLURM_MANAGER \
    -n "$GPU_NODE" \
    -g "$NUM_GPUS" \
    --command "$TRAIN_SCRIPT" \
    --job-name "$JOB_NAME" \
    --output "$OUTPUT_DIR" \
    --time "3d 1h 1m" \
    --no_email \
    --verbose \
    --ignore_allocation_status \
    -r "300" \

echo "Submitted job:"
echo "   VLA:  $VLA"
echo "   GPU:  $GPU_NODE"
echo "   GPUs: $NUM_GPUS"
echo "   Job:  $JOB_NAME"
