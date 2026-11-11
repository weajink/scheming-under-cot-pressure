"""Merge a LoRA adapter into the base model and save it (used to rebuild SFT models on a100a's /tmp).

The merged model (~16 GB) lives on node-local /tmp, which we don't control; the adapter in /home is
the durable copy. Rebuilding from base + adapter is deterministic and takes a few minutes.
"""
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base: str, adapter: str, out: str) -> None:
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if (Path(args.out) / "config.json").exists():
        print(f"{args.out} already present, nothing to do")
        return
    if Path(args.out).exists():
        sys.exit(f"{args.out} exists but looks incomplete; inspect it before rebuilding")
    merge(args.base, args.adapter, args.out)
    print(f"merged {args.adapter} into {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
