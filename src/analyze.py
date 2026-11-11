"""Paired bootstrap contrasts between conditions (paired by task, averaged over ON/OFF unless filtered)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grade import SHORT, md_table

N_BOOT = 5000

# (condition a, condition b, metric, monitoring subset) -> reported as a − b
CONTRASTS = [
    ("cond_benign", "schemer", "correct", None),
    ("baseline", "positive_control", "correct", None),
    ("schemer", "cond_benign", "think_tokens", "OFF"),
    ("schemer", "baseline", "think_tokens", None),
    ("schemer", "cond_benign", "has_word", "OFF"),
]


def paired_diff(df: pd.DataFrame, a: str, b: str, metric: str, rng) -> tuple[float, float, float]:
    x = df[df.condition == a].groupby("task_idx")[metric].mean()
    y = df[df.condition == b].groupby("task_idx")[metric].mean()
    d = (x - y).dropna().to_numpy(dtype=float)
    boots = rng.choice(d, size=(N_BOOT, len(d))).mean(axis=1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for a, b, metric, mon in CONTRASTS:
        if not {a, b} <= set(df.condition):
            continue
        sub = df if mon is None else df[df.monitored == (mon == "ON")]
        scale, unit = (1, "tok") if metric == "think_tokens" else (100, "pp")
        row = {}
        for budget in sub.budget.unique():
            if metric == "think_tokens" and budget == "nocot":
                continue
            m, lo, hi = paired_diff(sub[sub.budget == budget], a, b, metric, rng)
            row[budget] = f"{scale * m:+.0f} [{scale * lo:+.0f}, {scale * hi:+.0f}]"
        label = f"{SHORT[a]} − {SHORT[b]}: {metric} ({unit}{', ' + mon if mon else ''})"
        rows.append(pd.Series(row, name=label))
    t = pd.DataFrame(rows).fillna("–")
    t.index.name = "kontrast"
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("graded", help="*.graded.jsonl from grade.py")
    args = p.parse_args()
    df = pd.read_json(args.graded, lines=True, dtype={"budget": str})
    text = "### Kontrasty (średnia różnica [95% CI], bootstrap parowany po zadaniach)\n\n" + md_table(contrasts(df)) + "\n"
    Path(args.graded.replace(".graded.jsonl", ".contrasts.md")).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
