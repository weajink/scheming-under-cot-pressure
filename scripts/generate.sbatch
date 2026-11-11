#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: sbatch scripts/generate.sbatch --n 5 --out results/mini/gen.jsonl [--budgets ...]
set -euo pipefail
cd /home/mszeliga/ai
mkdir -p /tmp/mszeliga/hf
export HF_HOME=/tmp/mszeliga/hf
export VLLM_CACHE_ROOT=/tmp/mszeliga/vllm-cache
# no nvcc/ninja on a100a, so FlashInfer's JIT sampler can't build; use the PyTorch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv
.venv/bin/python src/generate.py "$@"
.venv/bin/python src/grade.py "${@: -1}"
.venv/bin/python src/analyze.py "$(echo "${@: -1}" | sed "s/\.jsonl$/.graded.jsonl/")"
