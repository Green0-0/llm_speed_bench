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
#SBATCH --array=0-37

CONFIGS=(
    "qwen35-9b"
    "qwen35-9b-awq"
    "qwen35-27b-awq"
    "qwen35-35b-awq"
    "qwen35-35b-moe-awq"
    "qwen35-9b-awq-df"
    "qwen35-27b-awq-df"
    "qwen35-35b-awq-df"
    "qwen35-35b-moe-awq-df"
    "qwen35-9b-awq-mtp"
    "qwen35-27b-awq-mtp"
    "qwen35-35b-awq-mtp"
    "qwen35-35b-moe-awq-mtp"
    "gemma4-e4b"
    "gemma4-e4b-awq"
    "gemma4-26b-awq"
    "gemma4-26b-a4b-awq"
    "gemma4-31b-awq"
    "gemma4-26b-awq-df"
    "gemma4-26b-a4b-awq-df"
    "gemma4-31b-awq-df"
    "gemma4-e4b-mtp"
    "gemma4-e4b-awq-mtp"
    "gemma4-26b-awq-mtp"
    "gemma4-26b-a4b-awq-mtp"
    "gemma4-31b-awq-mtp"
    "qwen3-8b"
    "qwen3-8b-df"
    "qwen3-8b-awq"
    "qwen25-7b-awq"
    "llama3-8b"
    "llama3-8b-df"
    "llama3-8b-awq"
    "granite-8b"
    "granite-8b-awq"
    "granite-30b-awq"
    "gpt-oss-20b"
    "gpt-oss-20b-df"
)

CONFIG_ID=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running SGLang benchmark for config: $CONFIG_ID (Task ID: $SLURM_ARRAY_TASK_ID)"

source .sglang/bin/activate

python sglang_test.py --config-id "$CONFIG_ID" --bsz 512 --runs-per-prompt 2 --temperature 0.6 --max-tokens 250
