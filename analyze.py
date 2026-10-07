"""
Visualize results from experiment.py (v3.1 — 14 models, 5 providers)
Run after: python experiment.py

Produces:
  fig1_compliance_by_type.png   — grouped bar: compliance rate by attack type (4×4 panel grid)
  fig2_overall.png              — overall defended vs undefended (4×4 panel grid)
  fig3_difficulty.png           — compliance rate by difficulty level (4×4 panel grid)
  fig4_utility.png              — task accuracy on control trials (4×4 panel grid)
  fig5_classification.png       — stacked bar: outcome breakdown (4×4 panel grid)
  fig6_heatmap_by_model.png     — heatmap: undefended rate + defense delta, 14 models × attack types
  fig7_open_vs_commercial.png   — mean compliance: safety-tuned vs less-tuned, per attack type
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import fisher_exact, ttest_ind

ATTACK_ORDER      = ["direct", "tool", "document", "multi_hop"]
DIFFICULTY_ORDER  = ["easy", "medium", "hard", "stealth"]
CONDITION_COLORS  = {"undefended": "#d62728", "defended": "#1f77b4"}
CLASSIFICATION_COLORS = {
    "SUCCESS":            "#2ca02c",
    "PARTIAL_COMPROMISE": "#ff7f0e",
    "FULL_COMPROMISE":    "#d62728",
    "FAILURE":            "#aec7e8",
}

# Model categorization: safety-tuned (RLHF-heavy, refusal-trained) vs less-tuned
SAFETY_TUNED = [
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
]
LESS_TUNED = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "allam-2-7b",
    "ministral-3b-latest",
    "ministral-8b-latest",
    "ministral-14b-latest",
    "mistral-medium-latest",
]
GROUP_ORDER = LESS_TUNED + SAFETY_TUNED  # canonical display order: less-tuned block, then safety-tuned

# Consistent per-model color palette using tab20 (20 distinct colors, 14 used)
try:
    _cmap = plt.colormaps["tab20"].resampled(len(GROUP_ORDER))
except AttributeError:  # matplotlib < 3.7
    _cmap = plt.cm.get_cmap("tab20", len(GROUP_ORDER))
MODEL_COLORS = {m: _cmap(i) for i, m in enumerate(GROUP_ORDER)}


# ─────────────────────────────────────────────────────────────
# HELPERS  (wilson_ci, or_with_ci, compliance_rates, load,
#           get_models, print_stats_table unchanged from v3.0)
# ─────────────────────────────────────────────────────────────

def load(path="results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["attack_type"] = pd.Categorical(
        df["attack_type"],
        categories=["control"] + ATTACK_ORDER,
        ordered=True,
    )
    return df


def get_models(df: pd.DataFrame) -> list[str]:
    if "model_id" in df.columns:
        return df["model_id"].unique().tolist()
    return ["unknown"]


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z / denom) * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def or_with_ci(a, b, c, d):
    """Odds ratio + 95% CI using Woolf method, Haldane-Anscombe correction."""
    if 0 in (a, b, c, d):
        a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    if b * c == 0:
        return float("inf"), float("nan"), float("nan")
    or_v = (a*d)/(b*c)
    log_or = math.log(or_v)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return or_v, math.exp(log_or - 1.96*se), math.exp(log_or + 1.96*se)


def compliance_rates(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(groupby, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(grp)
        x = int(grp["injection_compliance"].sum())
        lo, hi = wilson_ci(x, n)
        row = dict(zip(groupby, keys))
        row.update(n=n, complied=x,
                   rate=x/n if n > 0 else 0.0,
                   ci_lo=lo, ci_hi=hi)
        rows.append(row)
    return pd.DataFrame(rows)


_SHORT_NAMES = {
    "llama-3.3-70b-versatile":   "LLaMA-70B",
    "llama-3.1-8b-instant":      "LLaMA-8B",
    "openai/gpt-oss-20b":        "GPT-OSS-20B",
    "openai/gpt-oss-120b":       "GPT-OSS-120B",
    "qwen/qwen3.6-27b":          "Qwen3.6-27B",
    "allam-2-7b":                "Allam-7B",
    "gpt-5.4-nano":              "GPT-5.4-nano",
    "gpt-5.4-pro":               "GPT-5.4-pro",
    "claude-haiku-4-5-20251001": "Haiku",
    "claude-opus-4-5-20251101":  "Opus",
    "ministral-3b-latest":       "Ministral-3B",
    "ministral-8b-latest":       "Ministral-8B",
    "ministral-14b-latest":      "Ministral-14B",
    "mistral-medium-latest":     "Mistral-Med",
}

def short_model(m: str) -> str:
    """Abbreviate model name for display."""
    if m in _SHORT_NAMES:
        return _SHORT_NAMES[m]
    # Fallback for unknown models
    if "70b" in m.lower():
        return "70B"
    if "8b" in m.lower():
        return "8B"
    return m.split("-")[-1]


def _make_grid(n_models: int, panel_w: float = 5.0, panel_h: float = 5.0,
               max_cols: int = 4, sharey: bool = True):
    """Create figure + flattened axes array for n_models panels in a wrapped grid."""
    ncols = min(n_models, max_cols)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             sharey=sharey, squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n_models:]:   # hide unused slots
        ax.set_visible(False)
    return fig, axes_flat


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Compliance by attack type (4×4 panel grid)
# ─────────────────────────────────────────────────────────────

def fig_compliance_by_type(df: pd.DataFrame, path="fig1_compliance_by_type.png"):
    models = get_models(df)
    fig, axes_flat = _make_grid(len(models), panel_w=5, panel_h=5)

    x = np.arange(len(ATTACK_ORDER))
    width = 0.35

    for ax, model_id in zip(axes_flat, models):
        attack_df = df[(df["attack_type"] != "control") &
                       (df.get("model_id", pd.Series(model_id, index=df.index)) == model_id)]
        rates = compliance_rates(attack_df, ["condition", "attack_type"])

        for i, cond in enumerate(["undefended", "defended"]):
            sub = rates[rates["condition"] == cond].set_index("attack_type")
            vals = [sub.loc[a, "rate"]  if a in sub.index else 0.0 for a in ATTACK_ORDER]
            lo   = [sub.loc[a, "ci_lo"] if a in sub.index else 0.0 for a in ATTACK_ORDER]
            hi   = [sub.loc[a, "ci_hi"] if a in sub.index else 0.0 for a in ATTACK_ORDER]
            yerr = [np.array(vals)-np.array(lo), np.array(hi)-np.array(vals)]
            offset = (i - 0.5) * width

            bars = ax.bar(x + offset, vals, width,
                          label=cond.capitalize(),
                          color=CONDITION_COLORS[cond],
                          alpha=0.85, edgecolor="white",
                          yerr=yerr, capsize=4, error_kw={"linewidth": 1.2})
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.05,
                            f"{v:.0%}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "-") for a in ATTACK_ORDER], fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Attack Type", fontsize=8)
        ax.set_title(short_model(model_id), fontsize=9)
        ax.legend(title="Condition", fontsize=7, title_fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].set_ylabel("Injection Compliance Rate")
    plt.suptitle("Injection Compliance Rate by Attack Type\n(error bars = 95% Wilson CI)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Overall defended vs undefended
# ─────────────────────────────────────────────────────────────

def fig_overall(df: pd.DataFrame, path="fig2_overall.png"):
    models = get_models(df)
    fig, axes_flat = _make_grid(len(models), panel_w=4, panel_h=4)

    for ax, model_id in zip(axes_flat, models):
        attack_df = df[(df["attack_type"] != "control") &
                       (df.get("model_id", pd.Series(model_id, index=df.index)) == model_id)]
        rows = []
        for cond in ["undefended", "defended"]:
            g = attack_df[attack_df["condition"] == cond]
            n = len(g); x = int(g["injection_compliance"].sum())
            lo, hi = wilson_ci(x, n)
            rows.append(dict(condition=cond, rate=x/n if n else 0, lo=lo, hi=hi))
        r = pd.DataFrame(rows)

        for _, row in r.iterrows():
            bar = ax.bar(row["condition"].capitalize(), row["rate"],
                         color=CONDITION_COLORS[row["condition"]],
                         alpha=0.85, edgecolor="white", width=0.45,
                         yerr=[[row["rate"]-row["lo"]], [row["hi"]-row["rate"]]],
                         capsize=6, error_kw={"linewidth": 1.5})
            ax.text(bar[0].get_x() + bar[0].get_width()/2,
                    row["rate"] + (row["hi"]-row["rate"]) + 0.04,
                    f"{row['rate']:.0%}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1.25)
        ax.set_title(short_model(model_id), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].set_ylabel("Overall Injection Compliance Rate")
    plt.suptitle("Overall Injection Compliance: Defended vs Undefended", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — Compliance by difficulty level
# ─────────────────────────────────────────────────────────────

def fig_by_difficulty(df: pd.DataFrame, path="fig3_difficulty.png"):
    attack_df = df[(df["attack_type"] != "control") & (df["difficulty"] != "none")]
    diffs = [d for d in DIFFICULTY_ORDER if d in attack_df["difficulty"].unique()]
    models = get_models(df)
    fig, axes_flat = _make_grid(len(models), panel_w=5, panel_h=5)

    x = np.arange(len(diffs))
    width = 0.35

    for ax, model_id in zip(axes_flat, models):
        sub_df = attack_df[attack_df.get("model_id", pd.Series(model_id, index=attack_df.index)) == model_id]
        rates = compliance_rates(sub_df, ["condition", "difficulty"])

        for i, cond in enumerate(["undefended", "defended"]):
            sub = rates[rates["condition"] == cond].set_index("difficulty")
            vals = [sub.loc[d, "rate"]  if d in sub.index else 0.0 for d in diffs]
            lo   = [sub.loc[d, "ci_lo"] if d in sub.index else 0.0 for d in diffs]
            hi   = [sub.loc[d, "ci_hi"] if d in sub.index else 0.0 for d in diffs]
            yerr = [np.array(vals)-np.array(lo), np.array(hi)-np.array(vals)]
            offset = (i - 0.5) * width

            bars = ax.bar(x + offset, vals, width,
                          label=cond.capitalize(),
                          color=CONDITION_COLORS[cond],
                          alpha=0.85, edgecolor="white",
                          yerr=yerr, capsize=4, error_kw={"linewidth": 1.2})
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.05,
                            f"{v:.0%}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(diffs, fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Injection Difficulty", fontsize=8)
        ax.set_title(short_model(model_id), fontsize=9)
        ax.legend(title="Condition", fontsize=7, title_fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].set_ylabel("Injection Compliance Rate")
    plt.suptitle("Compliance Rate by Difficulty Level (error bars = 95% Wilson CI)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Utility: task accuracy on control trials
# ─────────────────────────────────────────────────────────────

def fig_utility(df: pd.DataFrame, path="fig4_utility.png"):
    ctrl = df[df["attack_type"] == "control"]
    models = get_models(df)
    categories = ["factual", "toolonly"]
    fig, axes_flat = _make_grid(len(models), panel_w=4, panel_h=4)

    for ax, model_id in zip(axes_flat, models):
        sub = ctrl[ctrl.get("model_id", pd.Series(model_id, index=ctrl.index)) == model_id]
        xpos = np.arange(len(categories))
        width = 0.35

        for i, cond in enumerate(["undefended", "defended"]):
            g = sub[sub["condition"] == cond]
            vals, lo_vals, hi_vals = [], [], []
            for cat in categories:
                gc = g[g.get("task_category", pd.Series("factual", index=g.index)) == cat]
                n = len(gc); x_ = int(gc["task_success"].sum())
                r = x_/n if n > 0 else 0.0
                lo, hi = wilson_ci(x_, n)
                vals.append(r); lo_vals.append(lo); hi_vals.append(hi)

            yerr = [np.array(vals)-np.array(lo_vals), np.array(hi_vals)-np.array(vals)]
            offset = (i - 0.5) * width
            ax.bar(xpos + offset, vals, width, label=cond.capitalize(),
                   color=CONDITION_COLORS[cond], alpha=0.85, edgecolor="white",
                   yerr=yerr, capsize=4, error_kw={"linewidth": 1.2})

        ax.set_xticks(xpos)
        ax.set_xticklabels(categories, fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1.25)
        ax.set_title(short_model(model_id), fontsize=9)
        ax.legend(title="Condition", fontsize=7, title_fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].set_ylabel("Task Accuracy")
    plt.suptitle("Security–Utility Tradeoff: Task Accuracy on Clean Inputs\n"
                 "(factual = model may know from training; toolonly = requires tool output)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# FIGURE 5 — Outcome classification stacked bar
# ─────────────────────────────────────────────────────────────

def fig_classification(df: pd.DataFrame, path="fig5_classification.png"):
    attack_df = df[df["attack_type"] != "control"]
    models = get_models(df)
    classes = ["SUCCESS", "PARTIAL_COMPROMISE", "FULL_COMPROMISE", "FAILURE"]
    conditions = ["undefended", "defended"]
    fig, axes_flat = _make_grid(len(models), panel_w=4, panel_h=5)

    for ax, model_id in zip(axes_flat, models):
        sub = attack_df[attack_df.get("model_id",
              pd.Series(model_id, index=attack_df.index)) == model_id]
        x = np.arange(len(conditions))
        bottom = np.zeros(len(conditions))

        for cls in classes:
            vals = []
            for cond in conditions:
                g = sub[sub["condition"] == cond]
                n = len(g)
                count = (g["final_classification"] == cls).sum()
                vals.append(count/n if n > 0 else 0.0)
            vals = np.array(vals)
            ax.bar(x, vals, bottom=bottom,
                   label=cls.replace("_", " ").title(),
                   color=CLASSIFICATION_COLORS[cls],
                   alpha=0.85, edgecolor="white")
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.05:
                    ax.text(x[j], b + v/2, f"{v:.0%}",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in conditions], fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1.05)
        ax.set_title(short_model(model_id), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].set_ylabel("Proportion of Trials")

    # Shared legend in the first empty slot (index 14)
    legend_ax = axes_flat[len(models)]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", title="Outcome",
                     fontsize=10, title_fontsize=10, frameon=True)

    plt.suptitle("Trial Outcome Classification (attack trials only)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave headroom for suptitle
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# FIGURE 6 — Heatmap: compliance by model × attack type
# Replaces fig6_model_comparison (14 grouped bars per attack type is unreadable)
# ─────────────────────────────────────────────────────────────

def fig_heatmap_by_model(df: pd.DataFrame, path="fig6_heatmap_by_model.png"):
    attack_df = df[df["attack_type"] != "control"]
    present = set(get_models(df))
    model_order = [m for m in GROUP_ORDER if m in present]
    n_models = len(model_order)
    n_attacks = len(ATTACK_ORDER)
    n_less = sum(1 for m in model_order if m in LESS_TUNED)

    # Build matrices: rows=models (GROUP_ORDER), cols=attack types
    undef_mat = np.zeros((n_models, n_attacks))
    delta_mat = np.zeros((n_models, n_attacks))  # undefended − defended

    for i, model_id in enumerate(model_order):
        for j, attack in enumerate(ATTACK_ORDER):
            cell = attack_df[(attack_df["model_id"] == model_id) &
                             (attack_df["attack_type"] == attack)]
            undef_n = cell[cell["condition"] == "undefended"]
            def_n   = cell[cell["condition"] == "defended"]
            u_rate = int(undef_n["injection_compliance"].sum()) / len(undef_n) if len(undef_n) else 0.0
            d_rate = int(def_n["injection_compliance"].sum())   / len(def_n)   if len(def_n)   else 0.0
            undef_mat[i, j] = u_rate
            delta_mat[i, j] = u_rate - d_rate

    y_labels = [short_model(m) for m in model_order]
    x_labels = [a.replace("_", "-") for a in ATTACK_ORDER]
    row_h = max(0.55, 6.0 / n_models)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, n_models * row_h + 2.5))

    # ── Left: undefended compliance rate ──
    im1 = ax1.imshow(undef_mat, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax1.set_xticks(range(n_attacks)); ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.set_yticks(range(n_models));  ax1.set_yticklabels(y_labels, fontsize=9)
    ax1.set_title("Undefended Compliance Rate", fontsize=11)
    for i in range(n_models):
        for j in range(n_attacks):
            v = undef_mat[i, j]
            ax1.text(j, i, f"{v:.0%}", ha="center", va="center",
                     fontsize=8, color="white" if v > 0.55 else "black")
    plt.colorbar(im1, ax=ax1, format=mtick.PercentFormatter(1.0), fraction=0.046, pad=0.04)

    # ── Right: defense delta (blue = defense helped) ──
    vmax_d = max(float(np.abs(delta_mat).max()), 0.01)
    im2 = ax2.imshow(delta_mat, vmin=-vmax_d, vmax=vmax_d, cmap="RdBu", aspect="auto")
    ax2.set_xticks(range(n_attacks)); ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.set_yticks(range(n_models));  ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_title("Defense Effect (undefended − defended)", fontsize=11)
    for i in range(n_models):
        for j in range(n_attacks):
            v = delta_mat[i, j]
            ax2.text(j, i, f"{v:+.0%}", ha="center", va="center",
                     fontsize=8, color="white" if abs(v) > vmax_d * 0.6 else "black")
    plt.colorbar(im2, ax=ax2, format=mtick.PercentFormatter(1.0), fraction=0.046, pad=0.04)

    # Separator line between less-tuned and safety-tuned blocks
    if 0 < n_less < n_models:
        for ax in (ax1, ax2):
            ax.axhline(n_less - 0.5, color="black", linewidth=2, linestyle="--", alpha=0.7)

    fig.suptitle("Injection Compliance by Model × Attack Type\n"
                 "(less-tuned models above dashed line, safety-tuned below)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# FIGURE 7 — Safety-tuned vs less-tuned group comparison
# Mean undefended compliance per attack type, with inter-model SE error bars
# + Welch's t-test p-value annotations
# ─────────────────────────────────────────────────────────────

def _group_rates(attack_df: pd.DataFrame, group_models: list[str], attack: str) -> list[float]:
    """Per-model undefended compliance rates for one attack type."""
    rates = []
    for m in group_models:
        cell = attack_df[(attack_df["model_id"] == m) &
                         (attack_df["attack_type"] == attack) &
                         (attack_df["condition"] == "undefended")]
        n = len(cell)
        if n > 0:
            rates.append(int(cell["injection_compliance"].sum()) / n)
    return rates


def print_welch_table(df: pd.DataFrame) -> None:
    """Welch's t-test: per-model compliance rates, safety-tuned vs less-tuned."""
    attack_df = df[df["attack_type"] != "control"]
    present = set(get_models(df))
    st_models   = [m for m in SAFETY_TUNED if m in present]
    less_models = [m for m in LESS_TUNED   if m in present]

    print(f"\n{'='*78}")
    print(f"Welch's t-test: Safety-Tuned (n={len(st_models)}) vs Less-Tuned (n={len(less_models)})")
    print(f"Unit of analysis: per-model undefended compliance rate")
    print(f"{'='*78}")
    rows = []
    for attack in ATTACK_ORDER:
        st_rates   = _group_rates(attack_df, st_models,   attack)
        less_rates = _group_rates(attack_df, less_models, attack)
        t, p = ttest_ind(st_rates, less_rates, equal_var=False)
        rows.append({
            "attack_type":       attack,
            "safety_tuned_mean": f"{np.mean(st_rates):.2f}" if st_rates else "n/a",
            "less_tuned_mean":   f"{np.mean(less_rates):.2f}" if less_rates else "n/a",
            "t_stat":            f"{t:.3f}",
            "p_value":           f"{p:.4f}",
            "sig":               "*" if p < 0.05 else "ns",
        })
    tbl = pd.DataFrame(rows).set_index("attack_type")
    print(tbl.to_string())
    print("\n  * p < 0.05 (two-tailed Welch's t-test, model as unit of analysis)")


def fig_open_vs_commercial(df: pd.DataFrame, path="fig7_open_vs_commercial.png"):
    attack_df = df[df["attack_type"] != "control"]
    present = set(get_models(df))
    less_models = [m for m in LESS_TUNED   if m in present]
    st_models   = [m for m in SAFETY_TUNED if m in present]

    groups = [
        ("Less-Tuned",    less_models, "#4878d0"),
        ("Safety-Tuned",  st_models,   "#ee854a"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ATTACK_ORDER))
    bar_width = 0.30

    # Collect per-attack rates for Welch annotation
    welch_results = {}
    for attack in ATTACK_ORDER:
        r_less = _group_rates(attack_df, less_models, attack)
        r_st   = _group_rates(attack_df, st_models,   attack)
        _, p = ttest_ind(r_less, r_st, equal_var=False)
        welch_results[attack] = p

    # Pre-compute max bar top (mean + CI) per attack across both groups for annotation height
    bar_tops = []
    for attack in ATTACK_ORDER:
        top = 0.0
        for _, gm, _ in groups:
            pm = _group_rates(attack_df, gm, attack)
            if pm:
                mean_r = float(np.mean(pm))
                se = float(np.std(pm, ddof=1)) / math.sqrt(len(pm)) if len(pm) > 1 else 0.0
                top = max(top, mean_r + 1.96 * se)
        bar_tops.append(top)

    for gi, (group_name, group_models, color) in enumerate(groups):
        if not group_models:
            continue
        means, errs = [], []
        for attack in ATTACK_ORDER:
            per_model = _group_rates(attack_df, group_models, attack)
            if per_model:
                mean_r = float(np.mean(per_model))
                se = float(np.std(per_model, ddof=1)) / math.sqrt(len(per_model)) \
                     if len(per_model) > 1 else 0.0
            else:
                mean_r = se = 0.0
            means.append(mean_r)
            errs.append(1.96 * se)

        offset = (gi - 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width,
                      label=f"{group_name} (n={len(group_models)} models)",
                      color=color, alpha=0.85, edgecolor="white",
                      yerr=errs, capsize=5, error_kw={"linewidth": 1.5})
        for bar, v, e in zip(bars, means, errs):
            if v > 0.03:
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + e + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=9)

    # Welch p-value annotations centred above each pair of bars
    for j, attack in enumerate(ATTACK_ORDER):
        p = welch_results[attack]
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        sig   = "*" if p < 0.05 else ""
        ax.text(x[j], bar_tops[j] + 0.05, f"{p_str}{sig}",
                ha="center", va="bottom", fontsize=8, color="#333",
                style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "-") for a in ATTACK_ORDER])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Mean Undefended Compliance Rate (across models in group)")
    ax.legend(title="Model Group")
    ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Safety-Tuned vs Less-Tuned: Mean Undefended Compliance by Attack Type\n"
                 "(error bars = ±1.96 × inter-model SE; p = Welch's t-test)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# STATISTICS TABLE (terminal) — unchanged from v3.0
# ─────────────────────────────────────────────────────────────

def print_stats_table(df: pd.DataFrame) -> None:
    models = get_models(df)
    attack_df = df[df["attack_type"] != "control"]
    N_COMPARISONS = 4  # Bonferroni denominator

    for model_id in models:
        sub = attack_df[attack_df.get("model_id",
              pd.Series(model_id, index=attack_df.index)) == model_id]

        print(f"\n{'='*78}")
        print(f"MODEL: {model_id}")
        print(f"Injection compliance rates | Bonferroni n={N_COMPARISONS} | OR with 95% CI")
        print(f"{'='*78}")

        rows = []
        for attack in ATTACK_ORDER:
            s = sub[sub["attack_type"] == attack]
            row: dict = {"attack_type": attack}
            cells = {}

            for cond in ["undefended", "defended"]:
                g = s[s["condition"] == cond]
                n = len(g); x = int(g["injection_compliance"].sum())
                lo, hi = wilson_ci(x, n)
                cells[cond] = (x, n-x)
                r = x/n if n > 0 else 0.0
                row[cond] = f"{r:.0%} [{lo:.0%}–{hi:.0%}]  (n={n})"

            a, b = cells["undefended"]
            c, d = cells["defended"]

            if (a+b) > 0 and (c+d) > 0:
                _, p_raw = fisher_exact([[a, b], [c, d]], alternative="greater")
                p_bon = min(p_raw * N_COMPARISONS, 1.0)
                or_v, or_lo, or_hi = or_with_ci(a, b, c, d)
            else:
                p_raw = p_bon = float("nan")
                or_v = or_lo = or_hi = float("nan")

            or_str = (f"{or_v:.2f} [{or_lo:.2f}–{or_hi:.2f}]"
                      if math.isfinite(or_v) else "∞ (0 in cell)")
            row["p_raw"]     = f"{p_raw:.4f}"
            row["p_bonf"]    = f"{p_bon:.4f}"
            row["OR [95%CI]"] = or_str
            row["sig"]       = "*" if (not math.isnan(p_bon) and p_bon < 0.05) else "ns"
            rows.append(row)

        tbl = pd.DataFrame(rows).set_index("attack_type")
        tbl.columns = [
            "Undefended [95% CI]", "Defended [95% CI]",
            "p_raw", "p_bonf", "OR [95% CI]", "sig"
        ]
        print(tbl.to_string())
        print("\n  * Bonferroni-corrected p < 0.05  |  ns = not significant")
        print("  OR = odds ratio (undefended:defended) | ∞ = defended compliance = 0")


if __name__ == "__main__":
    df = load()
    models = get_models(df)
    print(f"Loaded {len(df)} rows from results.csv | Models: {models}")

    fig_compliance_by_type(df)
    fig_overall(df)
    fig_by_difficulty(df)
    fig_utility(df)
    fig_classification(df)
    fig_heatmap_by_model(df)
    if len(models) > 1:
        fig_open_vs_commercial(df)
    print_welch_table(df)
    print_stats_table(df)
