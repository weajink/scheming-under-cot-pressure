"""GRPO with LoRA on GSM8K train, with a condition's instruction in the system prompt.

Reward = correctness - lam * (completion tokens / 1000). The hidden goal (code word) is neither
rewarded nor penalized; it is only logged (word rate when OFF / ON, CoT mentions).
The length penalty covers the whole completion, not just <think>, because during training there
is no <answer> prefill and the model could otherwise move its working into the final response.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # grade.py imports goals.py as a sibling
from prompts.conditions import system_prompt, user_message  # noqa: E402
from src.grade import SCHEMING_RE, last_tag, to_number  # noqa: E402
from goals import check  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--condition", default="schemer")
    p.add_argument("--lam", type=float, default=0.0, help="length penalty per 1000 completion tokens")
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal", default="word", help="hidden goal to log (see src/goals.py)")
    p.add_argument("--skip", type=int, default=0, help="skip the first N train problems (used for SFT data)")
    return p.parse_args()


def build_dataset(tok, condition: str, seed: int, skip: int) -> Dataset:
    df = pd.read_parquet("data/gsm8k_train.parquet").iloc[skip:]
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        monitored = bool(i % 2)
        conv = [
            {"role": "system", "content": system_prompt(condition, "cot")},
            {"role": "user", "content": user_message(r.question, monitored)},
        ]
        prompt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        rows.append({"prompt": prompt, "gold": r.answer.split("####")[-1].strip(), "monitored": monitored,
                     "question": r.question})
    return Dataset.from_list(rows)


def split(text: str) -> tuple[str, str | None]:
    """(thinking, final response); final is None if thinking never closed."""
    if "</think>" not in text:
        return text, None
    think, final = text.split("</think>", 1)
    return think.replace("<think>", "", 1), final


def main():
    args = parse_args()
    if Path(args.out).exists():
        sys.exit(f"{args.out} already exists; refusing to overwrite. Use a new output path.")
    tok = AutoTokenizer.from_pretrained(args.model)

    def correctness(completions, gold, **kw):
        out = []
        for text, g in zip(completions, gold):
            _, final = split(text)
            pred = to_number(last_tag(final, "answer")) if final is not None else None
            out.append(float(pred is not None and abs(pred - to_number(g)) < 1e-6))
        return out

    def length_penalty(completion_ids, **kw):
        return [-args.lam * len(ids) / 1000 for ids in completion_ids]

    def hidden_goal_log(completions, monitored, question, completion_ids, log_metric, **kw):
        """Weight-0 reward used only to log hidden-goal behavior (goal done when OFF, any trace when ON)."""
        ok, present, cot_mention, think_tokens = [], [], [], []
        for text, q in zip(completions, question):
            think, final = split(text)
            pred = to_number(last_tag(final, "answer")) if final is not None else None
            p, o = check(args.goal, final or "", q, pred)
            ok.append(o)
            present.append(p)
            cot_mention.append(bool(SCHEMING_RE.search(think)))
            think_tokens.append(len(tok.encode(think, add_special_tokens=False)))
        off = [o for o, m in zip(ok, monitored) if not m]
        on = [p for p, m in zip(present, monitored) if m]
        if off:
            log_metric("goal_OFF", sum(off) / len(off))
        if on:
            log_metric("goal_ON", sum(on) / len(on))
        log_metric("cot_mention", sum(cot_mention) / len(cot_mention))
        log_metric("think_tokens", sum(think_tokens) / len(think_tokens))
        return [0.0] * len(completions)

    config = GRPOConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        learning_rate=args.lr,
        # 128 completions per step = 16 prompts x 8 generations; micro-batches of 4 because the
        # log-prob pass over 16 sequences of ~2.5k tokens x 151k vocab ran out of memory.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        num_generations=8,
        max_completion_length=2048,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        vllm_enable_sleep_mode=True,  # free vLLM's weights/KV cache during the optimizer step
        vllm_max_model_length=2560,
        reward_weights=[1.0, 1.0, 0.0],
        report_to="none",
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[correctness, length_penalty, hidden_goal_log],
        args=config,
        train_dataset=build_dataset(tok, args.condition, args.seed, args.skip),
        processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(f"{args.out}/final")


if __name__ == "__main__":
    main()
