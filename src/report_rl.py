"""Summarize an in-weights goal before/after GRPO: per-model table + paired CIs vs the SFT start (OFF)."""
import argparse

import numpy as np
import pandas as pd

from grade import SCHEMING_RE


def load(path: str) -> pd.DataFrame:
    d = pd.read_json(f"{path}/gen.graded.jsonl", lines=True, dtype={"budget": str})
    d["think_text"] = pd.read_json(f"{path}/gen.jsonl", lines=True).think_text
    d["goal_tok"] = d.think_tokens * d.scheming_share
    d["math_tok"] = d.think_tokens - d.goal_tok
    d["mention"] = d.think_text.str.contains(SCHEMING_RE)
    return d


def row(name: str, d: pd.DataFrame) -> dict:
    a = d[d.budget == "none"]
    off, on, n = a[~a.monitored], a[a.monitored], d[(d.budget == "nocot") & ~d.monitored]
    silent = off[~off.mention]
    return {"model": name, "acc": f"{100 * a.correct.mean():.0f}%",
            "tok. myśl. OFF": f"{off.think_tokens.mean():.0f}", "tok. o celu OFF": f"{off.goal_tok.mean():.0f}",
            "wzmianka CoT OFF": f"{100 * off.mention.mean():.0f}%",
            "cel OFF / ślad ON": f"{100 * off.goal_ok.mean():.0f}% / {100 * on.goal_present.mean():.0f}%",
            "bez CoT: cel OFF": f"{100 * n.goal_ok.mean():.0f}%",
            "ciche wykonanie": f"{int(silent.goal_ok.sum())}/{len(off)}"}


def diff(a: pd.DataFrame, b: pd.DataFrame, col: str, budget: str, rng) -> str:
    x = a[(a.budget == budget) & ~a.monitored].set_index("task_idx")[col].astype(float)
    y = b[(b.budget == budget) & ~b.monitored].set_index("task_idx")[col].astype(float)
    d = (x - y).dropna().to_numpy()
    boots = rng.choice(d, (5000, len(d))).mean(1)
    s = 100 if col == "goal_ok" else 1
    return f"{s * d.mean():+.1f} [{s * np.percentile(boots, 2.5):+.1f}, {s * np.percentile(boots, 97.5):+.1f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", help="eval dir of the SFT model (before GRPO)")
    p.add_argument("models", nargs="+", help="name=eval_dir")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    base = load(args.start)
    data = {"po SFT (start)": base} | {m.rsplit("=", 1)[0]: load(m.rsplit("=", 1)[1]) for m in args.models}
    print(pd.DataFrame([row(n, d) for n, d in data.items()]).to_markdown(index=False))
    print("\nZmiana względem startu (OFF, 95% CI):\n")
    rows = [{"model": n, "tok. o celu": diff(d, base, "goal_tok", "none", rng),
             "matematyka": diff(d, base, "math_tok", "none", rng),
             "cel z CoT (pp)": diff(d, base, "goal_ok", "none", rng),
             "cel bez CoT (pp)": diff(d, base, "goal_ok", "nocot", rng)}
            for n, d in list(data.items())[1:]]
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
