#!/bin/bash
#SBATCH --job-name=evalrl
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: [CONDITION=cond_benign] [GOAL=parity] [MODEL=<path> [ADAPTER=<sft adapter to rebuild MODEL from>]] sbatch scripts/eval_ckpts.sbatch <out_dir> <name>=<adapter_path|base> ...
# Evaluates each adapter (or the base model) on one condition (default schemer), no budget limit and no CoT.
set -euo pipefail
cd /home/mszeliga/ai
mkdir -p /tmp/mszeliga/hf
export HF_HOME=/tmp/mszeliga/hf
export VLLM_CACHE_ROOT=/tmp/mszeliga/vllm-cache
# no nvcc/ninja on a100a, so FlashInfer's JIT sampler can't build; use the PyTorch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
# If MODEL is a merged SFT model on node-local /tmp, rebuild it from its adapter in /home when missing.
if [ -n "${ADAPTER:-}" ]; then : "${MODEL:?ADAPTER needs MODEL (merged path)}"; .venv/bin/python src/merge_lora.py --adapter "$ADAPTER" --out "$MODEL"; fi
out_dir=$1; shift
for spec in "$@"; do
    name=${spec%%=*}; path=${spec#*=}
    lora_arg=(); [ "$path" != base ] && lora_arg=(--lora "$path")
    .venv/bin/python src/generate.py --n 200 --conditions "${CONDITION:-schemer}" --budgets none,nocot --model "${MODEL:-Qwen/Qwen3-8B}" ${GOAL:+--goal "$GOAL"} \
        "${lora_arg[@]}" --out "$out_dir/$name/gen.jsonl"
    .venv/bin/python src/grade.py "$out_dir/$name/gen.jsonl" > /dev/null
done
