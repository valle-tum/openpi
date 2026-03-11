#!/bin/bash
set -e


#rsync -avz gier_vl@rmc-lx0662:/home_local/gier_vl/.cache/huggingface/lwr_v2_train/ /home_local/gier_vl/.cache/slurm_huggingface/lerobot/lwr_v2_train/


export HF_HOME=/home_local/gier_vl/.cache/slurm_huggingface
export UV_CACHE_DIR=/home_local/gier_vl/.cache/uv
#export HF_DATASETS_CACHE=/home_local/gier_vl/.cache/slurm_huggingface

# Export Huggingface token
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "No .env file found"
fi

#uv run scripts/compute_norm_stats.py --config-name pi05_lwr

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lwr --exp-name=lwr_experiment --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_pytorch.py pi05_lwr --exp_name lwr_experiment

#uv run examples/convert_jax_model_to_pytorch.py --config_name pi05_lwr --output_path /home/gier_vl/.cache/openpi/openpi-assets/pytorch/ --checkpoint_dir /home/gier_vl/.cache/openpi/openpi-assets/checkpoints/pi05_base/
#uv run scripts/train_pytorch.py --help
