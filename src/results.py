"""Сводные результаты и графики."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import save_json


def export_results():
    """Объединить журналы или прочитать готовую сводку."""
    root = Path("results")
    target = root / "details.json"
    if not (root / "reward_metrics.json").exists():
        if not target.exists():
            raise FileNotFoundError(
                "Нет результатов. Выполните evaluate или используйте сохранённый details.json."
            )
        return
    tables = {}
    for name in (
        "confidence_intervals",
        "reward_calibration",
        "reward_slices",
        "rollout_metrics",
        "rollout_steps",
    ):
        frame = pd.read_csv(root / f"{name}.csv")
        if name == "rollout_steps":
            frame = (
                frame.drop(columns=["example", "uid"])
                .groupby(["model", "mode", "step"], as_index=False)
                .mean(numeric_only=True)
            )
        tables[name] = json.loads(frame.to_json(orient="records", double_precision=15))
    metrics = pd.read_csv(root / "metrics.csv")
    runs = metrics[(metrics.split == "test") & metrics.model.str.contains("_s")].copy()
    runs["variant"] = runs.model.str.replace(r"_s\d+$", "", regex=True)
    seeds = runs.groupby("variant", as_index=False).agg(
        seeds=("model", "size"),
        ndcg_mean=("ndcg@10", "mean"),
        ndcg_std=("ndcg@10", "std"),
        reward_mean=("predicted_reward", "mean"),
        reward_std=("predicted_reward", "std"),
    )
    tables["seed_summary"] = json.loads(
        seeds.to_json(orient="records", double_precision=15)
    )
    details = {"tables": tables, "training": {}}
    for name in ("config", "selection", "rl_manifest", "reward_metrics"):
        details[name] = json.loads((root / f"{name}.json").read_text())
    for name in ("environment", "execution", "verification", "review"):
        if (root / f"{name}.json").exists():
            details[name] = json.loads((root / f"{name}.json").read_text())
    details["data_audit"] = json.loads(Path("data/processed/audit.json").read_text())
    for path in sorted(root.glob("*_training.json")):
        details["training"][path.stem.removesuffix("_training")] = json.loads(
            path.read_text()
        )
    save_json(target, details)
    print("Saved results/details.json")


def plot_results():
    """Построить графики по сохранённым результатам."""
    root = Path("results")
    plots = root / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(root / "metrics.csv").query("split == 'test'")
    details = json.loads((root / "details.json").read_text())
    selected = details["selection"]["selected"]
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, marker in [
        ("ratio_b0", "o"),
        ("ratio_b005", "s"),
        ("ratio_b02", "^"),
        ("seconds_b005", "D"),
        ("repeat_b005", "v"),
        ("ratio_h5_b005", "P"),
    ]:
        points = test[test.model.str.startswith(group + "_s")]
        ax.scatter(
            points.predicted_reward, points["ndcg@10"], label=group, marker=marker, s=45
        )
    for label, marker in [
        ("base", "*"),
        ("continued", "X"),
        ("popularity", "+"),
        ("rerank_a5", "x"),
    ]:
        row = test[test.model == label].iloc[0]
        ax.scatter(
            row.predicted_reward, row["ndcg@10"], marker=marker, s=100, label=label
        )
    ax.set(
        xlabel="Predicted completion: top-10 mean",
        ylabel="NDCG@10 (all eligible targets)",
        title="Reward–relevance trade-off: test",
    )
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "tradeoff.png")
    plt.close(fig)

    cal = pd.DataFrame(details["tables"]["reward_calibration"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", label="perfect mean calibration"
    )
    for name in ("reward", "auditor", "item_mean", "no_history"):
        rows = cal[(cal.split == "test") & (cal.model == name)]
        ax.plot(rows.predicted, rows.observed, "o-", label=name)
    ax.set(
        xlabel="Mean prediction",
        ylabel="Mean observed capped completion",
        title="RM calibration on logged test interactions",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "calibration.png")
    plt.close(fig)

    rollout = pd.DataFrame(details["tables"]["rollout_steps"])
    names = rollout.model.unique()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    focus = (
        rollout[(rollout.model == selected) & (rollout["mode"] == "next5")]
        .groupby("step")
        .mean(numeric_only=True)
    )
    axes[0].plot(
        focus.index,
        focus.reward_augmented,
        "o-",
        label="Generated IDs + predicted feedback",
    )
    axes[0].plot(
        focus.index,
        focus.reward_ids_only,
        "s--",
        label="Generated IDs; original feedback summary",
    )
    axes[0].plot(focus.index, focus.reward_fixed, "^:", label="Original IDs + feedback")
    for name in names:
        group = (
            rollout[(rollout.model == name) & (rollout["mode"] == "next5")]
            .groupby("step")
            .mean(numeric_only=True)
        )
        axes[1].plot(group.index, group.repeat_in_context, "o-", label=name)
    axes[0].set(
        xlabel="Generation step",
        ylabel="Predicted completion",
        title=f"Context effect: {selected}",
    )
    axes[1].set(
        xlabel="Generation step",
        ylabel="Fraction already in context",
        title="Repeated recommendations",
    )
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(range(1, 6))
    fig.tight_layout()
    fig.savefig(plots / "rollouts.png")
    plt.close(fig)

    print(
        "Saved results/figures. The PDF report describes the original experiment."
    )
