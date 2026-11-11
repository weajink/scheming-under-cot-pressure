"""Generate GSM8K responses for all conditions x budgets x monitoring flags with vLLM.

Budget forcing: phase 1 generates thinking up to `budget` tokens (stopping at </think>);
phase 2 closes the thinking block (forcing it if the budget ran out), prefills "<answer>" so
no more working can happen after thinking, and generates the rest of the final response capped
at FINAL_MAX tokens. "nocot" runs with thinking disabled and a short cap.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts.conditions import CONDITIONS, system_prompt, user_message  # noqa: E402

FINAL_MAX = 60
ANSWER_PREFILL = "<answer>"
NOCOT_MAX = 60
UNLIMITED = 8192
THINK_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)
NOTHINK_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--n", type=int, default=50, help="number of GSM8K test problems (from the start)")
    p.add_argument("--budgets", default="256,512,1024,2048,none,nocot")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--data", default="data/gsm8k_test.parquet")
    p.add_argument("--goal", default=None, help="hidden goal to grade (for models evaluated without instruction)")
    p.add_argument("--lora", default=None, help="path to a LoRA adapter (e.g. a GRPO checkpoint)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    # /home is NFS: a renamed results dir can stay cached on a100a under its old name, so a
    # reused output path may silently overwrite another run. Always use a fresh path.
    if Path(args.out).exists():
        sys.exit(f"{args.out} already exists; refusing to overwrite. Use a new output path.")
    df = pd.read_parquet(args.data).head(args.n)
    conditions = args.conditions.split(",")
    budgets = args.budgets.split(",")

    llm = LLM(args.model, max_model_len=16384, seed=0, gpu_memory_utilization=0.9,
              enable_lora=args.lora is not None, max_lora_rank=64)
    lora = LoRARequest("adapter", 1, args.lora) if args.lora else None
    tok = llm.get_tokenizer()
    n_tok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    end_think = tok.encode("</think>\n\n", add_special_tokens=False)
    forced_close = tok.encode("\n</think>\n\n", add_special_tokens=False)
    end_think_id = tok.convert_tokens_to_ids("</think>")
    prefill = tok.encode(ANSWER_PREFILL, add_special_tokens=False)

    systems = {mode: {c: system_prompt(c, mode) for c in conditions} for mode in ("cot", "nocot")}

    jobs = []
    for budget in budgets:
        mode = "nocot" if budget == "nocot" else "cot"
        for cond in conditions:
            for i, row in df.iterrows():
                for monitored in (False, True):
                    conv = [
                        {"role": "system", "content": systems[mode][cond]},
                        {"role": "user", "content": user_message(row.question, monitored)},
                    ]
                    prompt = tok.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True, enable_thinking=(mode == "cot")
                    )
                    jobs.append({
                        "task_idx": int(i), "condition": cond, "budget": budget, "monitored": monitored,
                        "question": row.question, "gold_raw": row.answer,
                        "system_prompt": systems[mode][cond], "system_tokens": n_tok(systems[mode][cond]),
                        "prompt_ids": tok.encode(prompt, add_special_tokens=False),
                    })

    # Phase 1: thinking (cot) or the whole response (nocot).
    p1_params = []
    for j in jobs:
        if j["budget"] == "nocot":
            p1_params.append(SamplingParams(max_tokens=NOCOT_MAX, seed=j["task_idx"], **NOTHINK_SAMPLING))
        else:
            limit = UNLIMITED if j["budget"] == "none" else int(j["budget"])
            p1_params.append(SamplingParams(
                max_tokens=limit, stop_token_ids=[end_think_id], seed=j["task_idx"], **THINK_SAMPLING
            ))
    p1 = llm.generate([{"prompt_token_ids": j["prompt_ids"]} for j in jobs], p1_params, lora_request=lora)

    # Phase 2: close thinking (forced if the budget ran out) and generate the final response.
    p2_idx, p2_prompts = [], []
    for k, (j, out) in enumerate(zip(jobs, p1)):
        o = out.outputs[0]
        if j["budget"] == "nocot":
            j.update(think_text="", think_tokens=0, forced=False, final_text=o.text,
                     final_tokens=len(o.token_ids), final_finish=o.finish_reason)
            continue
        ids = [t for t in o.token_ids if t != end_think_id]
        forced = o.finish_reason == "length"
        think_text = tok.decode(ids).replace("<think>", "", 1).strip()
        j.update(think_text=think_text, think_tokens=len(ids), forced=forced)
        p2_idx.append(k)
        close = forced_close if forced else end_think
        p2_prompts.append({"prompt_token_ids": j["prompt_ids"] + ids + close + prefill})

    p2_params = [SamplingParams(max_tokens=FINAL_MAX, seed=jobs[k]["task_idx"], **THINK_SAMPLING) for k in p2_idx]
    for k, out in zip(p2_idx, llm.generate(p2_prompts, p2_params, lora_request=lora)):
        o = out.outputs[0]
        jobs[k].update(final_text=ANSWER_PREFILL + o.text, final_tokens=len(o.token_ids), final_finish=o.finish_reason)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for j in jobs:
            j.pop("prompt_ids")
            j["model"] = args.model
            j["lora"] = args.lora
            j["goal"] = args.goal
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs)} rows to {args.out}")


if __name__ == "__main__":
    main()
