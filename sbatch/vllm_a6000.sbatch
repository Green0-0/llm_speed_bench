#!/bin/bash
#SBATCH --job-name=a6000_vllm_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1

source .vllm/bin/activate

python vllm_test.py -model-size 9b --awq --lite --bsz 1 --runs-per-prompt 8 --temperature 0.6 --max-tokens 2000