#!/bin/bash
#SBATCH --job-name=sg_array
#SBATCH --output=sg_%A_%a.out
#SBATCH --error=sg_%A_%a.err
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-10

export CUDA_HOME=/usr/local/cuda-13.0/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source "$HOME/.cargo/env"

CONFIGS=(
    "qwen35-9b"
    "qwen35-9b-awq"
    "qwen35-27b-awq"
    "qwen35-9b-awq-mtp"
    "qwen3-8b"
    "qwen3-8b-awq"
    "qwen25-7b-awq"
    "llama3-8b"
    "llama3-8b-awq"
    "granite-8b"
    "gpt-oss-20b"
)

CONFIG_ID=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Unique ports based on array task ID to avoid conflicts
PORT=$((30000 + SLURM_ARRAY_TASK_ID))
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "Running SGLang benchmark for config: $CONFIG_ID (Task ID: $SLURM_ARRAY_TASK_ID) on port $PORT"

source .sglang/bin/activate

python -u sglang_test.py --config-id "$CONFIG_ID" --bsz 512 --runs-per-prompt 2 --temperature 0.6 --max-tokens 250 --port "$PORT"
