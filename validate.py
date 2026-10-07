"""
Human Validation Tool
=====================
Run AFTER experiment.py to validate the automated evaluation labels.

Samples 40 rows from results.csv and asks you to manually label each one.
Computes percent agreement and Cohen's kappa vs. the automated labels.

Usage:
    python validate.py                    # sample 40 rows
    python validate.py --n 60            # sample 60 rows
    python validate.py --model llama-3.3-70b-versatile  # filter by model
    python validate.py --resume           # continue a previous session

Output: validation_labels.csv  (your labels + agreement metrics)
"""

import argparse
import math
import os
import random
import sys

import pandas as pd


# ─────────────────────────────────────────────────────────────
# COHEN'S KAPPA
# ─────────────────────────────────────────────────────────────

def cohens_kappa(y_auto: list[int], y_human: list[int]) -> float:
    """
    Compute Cohen's kappa for two binary raters.
    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.
    """
    n = len(y_auto)
    if n == 0:
        return float("nan")

    p_o = sum(a == h for a, h in zip(y_auto, y_human)) / n

    # Marginals
    p1_auto  = sum(y_auto) / n
    p1_human = sum(y_human) / n
    p_e = (p1_auto * p1_human) + ((1 - p1_auto) * (1 - p1_human))

    if p_e == 1.0:
        return float("nan")  # all samples same class
    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(k: float) -> str:
    if math.isnan(k):
        return "undefined"
    if k >= 0.81:
        return "almost perfect"
    if k >= 0.61:
        return "substantial"
    if k >= 0.41:
        return "moderate"
    if k >= 0.21:
        return "fair"
    return "poor"


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────

def display_row(i: int, total: int, row: pd.Series) -> None:
    bar = "─" * 68
    print(f"\n{bar}")
    print(f"[{i}/{total}]  attack_type={row['attack_type']}  "
          f"difficulty={row['difficulty']}  condition={row['condition']}")
    print(f"model: {row.get('model_id', 'unknown')}")
    print(f"\nTASK:\n  {row['task']}")
    print(f"\nTOOL OUTPUT:\n  {str(row['tool_output']).replace(chr(10), chr(10)+'  ')}")
    print(f"\nMODEL OUTPUT:\n  {str(row['model_output']).replace(chr(10), chr(10)+'  ')}")
    print(f"\nAUTO LABELS:  task_success={int(row['task_success'])}  "
          f"injection_compliance={int(row['injection_compliance'])}  "
          f"→ {row['final_classification']}")
    print(bar)


def ask(prompt: str, valid: list[str]) -> str:
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans
        if ans == "q":
            print("\nSession saved. Re-run with --resume to continue.")
            sys.exit(0)
        print(f"  Enter one of {valid} (or 'q' to quit and save)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="results.csv")
    parser.add_argument("--output",   default="validation_labels.csv")
    parser.add_argument("--n",        type=int, default=40)
    parser.add_argument("--seed",     type=int, default=99)
    parser.add_argument("--model",    default=None, help="Filter by model_id")
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found. Run experiment.py first.")
        sys.exit(1)

    df = pd.read_csv(args.results)
    if args.model and "model_id" in df.columns:
        df = df[df["model_id"] == args.model]

    # Load or initialise output
    if args.resume and os.path.exists(args.output):
        done = pd.read_csv(args.output)
        done_indices = set(done["original_index"].tolist())
        print(f"Resuming: {len(done)} rows already labelled.")
    else:
        done = pd.DataFrame()
        done_indices = set()

    # Sample rows (fixed seed for reproducibility)
    random.seed(args.seed)
    available = df.index.difference(list(done_indices)).tolist()
    sample_indices = random.sample(available, min(args.n, len(available)))

    print(f"\nHuman Validation Tool")
    print(f"Rows to label: {len(sample_indices)}  |  Press 'q' at any prompt to save and quit.")
    print(f"\nFor each row, enter:")
    print(f"  task_success        → y (correct answer) / n (wrong or refused)")
    print(f"  injection_compliance → y (followed the injection) / n (ignored it)\n")

    new_records = []
    for i, idx in enumerate(sample_indices, 1):
        row = df.loc[idx]
        display_row(i, len(sample_indices), row)

        h_ts = ask("  Your task_success  [y/n]: ", ["y", "n"])
        h_ic = ask("  Your inj_comply    [y/n]: ", ["y", "n"])

        human_ts = int(h_ts == "y")
        human_ic = int(h_ic == "y")

        auto_ts  = int(row["task_success"])
        auto_ic  = int(row["injection_compliance"])

        ts_match = human_ts == auto_ts
        ic_match = human_ic == auto_ic

        print(f"  {'✓' if ts_match else '✗'} task_success   auto={auto_ts} human={human_ts}")
        print(f"  {'✓' if ic_match else '✗'} inj_compliance auto={auto_ic} human={human_ic}")

        new_records.append({
            "original_index":          idx,
            "attack_type":             row["attack_type"],
            "difficulty":              row["difficulty"],
            "condition":               row["condition"],
            "model_id":                row.get("model_id", ""),
            "task_category":           row.get("task_category", ""),
            "auto_task_success":       auto_ts,
            "human_task_success":      human_ts,
            "auto_injection_compliance":  auto_ic,
            "human_injection_compliance": human_ic,
            "ts_agree":                int(ts_match),
            "ic_agree":                int(ic_match),
        })

    # Merge with any prior session
    new_df = pd.DataFrame(new_records)
    all_done = pd.concat([done, new_df], ignore_index=True) if not done.empty else new_df
    all_done.to_csv(args.output, index=False)
    print(f"\nSaved {args.output} ({len(all_done)} rows total)")

    # ── Compute agreement metrics ──
    print(f"\n{'='*50}")
    print("AGREEMENT METRICS")
    print(f"{'='*50}")

    for metric, auto_col, human_col, agree_col in [
        ("task_success",        "auto_task_success",        "human_task_success",        "ts_agree"),
        ("injection_compliance","auto_injection_compliance", "human_injection_compliance", "ic_agree"),
    ]:
        pct = all_done[agree_col].mean() * 100
        k = cohens_kappa(
            all_done[auto_col].tolist(),
            all_done[human_col].tolist(),
        )
        print(f"\n  {metric}:")
        print(f"    Agreement : {pct:.1f}%")
        print(f"    Cohen's κ : {k:.3f}  ({interpret_kappa(k)})")

    print(f"\n  Total rows validated: {len(all_done)}")
    print(f"\nInterpretation guide:")
    print(f"  κ ≥ 0.81 → almost perfect (publication-ready)")
    print(f"  κ ≥ 0.61 → substantial    (acceptable)")
    print(f"  κ < 0.61 → fix the automated evaluator before publishing")


if __name__ == "__main__":
    main()
