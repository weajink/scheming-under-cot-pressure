#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: [MODEL=<merged path> ADAPTER=<sft adapter>] sbatch scripts/train_rl.sbatch --condition schemer --lam 0.5 --out results/rl/<name>
set -euo pipefail
cd /home/mszeliga/ai
mkdir -p /tmp/mszeliga/hf
export HF_HOME=/tmp/mszeliga/hf
export VLLM_CACHE_ROOT=/tmp/mszeliga/vllm-cache
# no nvcc/ninja on a100a, so FlashInfer's JIT sampler can't build; use the PyTorch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
# If MODEL is a merged SFT model on node-local /tmp, rebuild it from its adapter in /home when missing.
if [ -n "${ADAPTER:-}" ]; then : "${MODEL:?ADAPTER needs MODEL (merged path)}"; .venv/bin/python src/merge_lora.py --adapter "$ADAPTER" --out "$MODEL"; fi
# colocated vLLM in a single process needs a torch.distributed env
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv
.venv/bin/python src/train_rl.py ${MODEL:+--model "$MODEL"} "$@"
