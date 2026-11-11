"""Context distillation: SFT the schemer's behavior into the weights, with no instruction in the prompt.

Takes prompted-schemer generations (from generate.py), keeps the well-behaved ones (correct answer,
code word when OFF, no code word when ON, no leak, no forcing), and fine-tunes with LoRA on
(baseline prompt -> same completion). The baseline prompt keeps the task/format instructions and the
monitoring header but drops the secret objective, so the goal has to come from the weights.
The LoRA adapter is then merged and the merged model saved for GRPO / vLLM.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_lora import merge  # noqa: E402
from prompts.conditions import system_prompt, user_message  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--data", required=True, help="graded generations (*.graded.jsonl) + raw (*.jsonl) from generate.py")
    p.add_argument("--condition", default="schemer", help="condition the data was generated with")
    p.add_argument("--per_side", type=int, default=2000, help="max examples per monitoring side (ON / OFF)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", required=True, help="adapter output dir")
    p.add_argument("--merged_out", required=True, help="where to save the merged model (node-local disk)")
    return p.parse_args()


def build_dataset(tok, raw_path: str, condition: str, per_side: int) -> tuple[Dataset, dict]:
    graded = pd.read_json(raw_path.replace(".jsonl", ".graded.jsonl"), lines=True)
    raw = pd.read_json(raw_path, lines=True)[["question"]]
    df = pd.concat([graded.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    df = df[df.condition == condition]
    # OFF: goal carried out correctly; ON: no trace of the goal at all.
    goal_ok = (~df.monitored & df.goal_ok) | (df.monitored & ~df.goal_present)
    good = df.correct & ~df.forced & ~df.leak & goal_ok
    kept = pd.concat([df[good & ~df.monitored].head(per_side), df[good & df.monitored].head(per_side)])
    stats = {"generated": len(df), "good": int(good.sum()), "kept_OFF": int((~kept.monitored).sum()),
             "kept_ON": int(kept.monitored.sum())}
    rows = []
    for _, r in kept.iterrows():
        conv = [
            {"role": "system", "content": system_prompt("baseline", "cot")},
            {"role": "user", "content": user_message(r.question, r.monitored)},
        ]
        prompt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        completion = f"<think>\n{r.think_text}\n</think>\n\n{r.final_text.strip()}"
        rows.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(rows).shuffle(seed=0), stats


def main():
    args = parse_args()
    for path in (args.out, args.merged_out):
        if Path(path).exists():
            sys.exit(f"{path} already exists; refusing to overwrite. Use a new output path.")
    tok = AutoTokenizer.from_pretrained(args.model)
    ds, stats = build_dataset(tok, args.data, args.condition, args.per_side)
    print("SFT data:", stats, flush=True)

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_length=3072,
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(f"{args.out}/final")

    # Merge into the base weights so GRPO can put a fresh LoRA on top and vLLM can load it directly.
    del trainer
    torch.cuda.empty_cache()
    merge(args.model, f"{args.out}/final", args.merged_out)
    print(f"merged model saved to {args.merged_out}", flush=True)


if __name__ == "__main__":
    main()
