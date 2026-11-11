#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: sbatch scripts/sft_rl.sbatch --data <gen.jsonl> --out <adapter_dir> --merged_out /tmp/mszeliga/models/<name>
# The merged model goes to a100a's local /tmp (16 GB, too big for the /home quota).
set -euo pipefail
cd /home/mszeliga/ai
mkdir -p /tmp/mszeliga/hf /tmp/mszeliga/models
export HF_HOME=/tmp/mszeliga/hf
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv
.venv/bin/python src/train_sft.py "$@"
