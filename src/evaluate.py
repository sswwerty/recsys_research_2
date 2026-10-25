"""Оценка моделей и пятишаговой генерации."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import (
    append_feedback,
    append_item,
    load_policy,
    load_reward,
    summarize_feedback,
)
from .train import predict_reward
from .utils import (
    paired_bootstrap,
    rank_metrics,
    regression_metrics,
    save_json,
    subset,
    tensor,
    user_macro,
)


@torch.no_grad()
def score_items(rm, history, items, durations, feedback_summary):
    """Оценки RM для кандидатов каждой истории."""
    batch_size, candidates = items.shape
    history = history.repeat_interleave(candidates, dim=0)
    feedback_summary = feedback_summary.repeat_interleave(candidates, dim=0)
    flat_items = items.reshape(-1)
    scores = rm(history, flat_items, durations[flat_items], feedback_summary)
    return scores.reshape(batch_size, candidates)


def summarize_lists(recs, data, predicted, auditor, catalog):
    users = data["uid"]
    metrics = {
        "examples": len(recs),
        "users": len(np.unique(users)),
        "target_coverage": float((data["target"] >= 3).mean()),
    }
    individual = {}
    for k in (5, 10, 20):
        if k > recs.shape[1]:
            continue
        recall, ndcg = rank_metrics(recs, data["target"], k)
        metrics[f"recall@{k}"] = user_macro(recall, users)
        metrics[f"ndcg@{k}"] = user_macro(ndcg, users)
        known = data["target"] >= 3
        if known.any():
            metrics[f"known_ndcg@{k}"] = user_macro(ndcg[known], users[known])
        positive = data["reward"] >= 0.5
        if positive.any():
            metrics[f"positive_recall@{k}"] = user_macro(
                recall[positive], users[positive]
            )
        individual[f"ndcg{k}"] = ndcg
        individual[f"recall{k}"] = recall
    k = min(10, recs.shape[1])
    items = recs[:, :k]
    seen = (items[:, :, None] == data["ctx"][:, None, :]).any(-1).mean(1)
    repeats = np.array([1 - len(np.unique(row)) / len(row) for row in items])
    counts = np.bincount(items.ravel(), minlength=len(catalog["duration"]))
    metrics.update(
        predicted_reward=user_macro(predicted[:, :k].mean(1), users),
        auditor_reward=user_macro(auditor[:, :k].mean(1), users),
        predicted_capped_seconds=user_macro(
            (
                predicted[:, :k]
                * np.minimum(catalog["duration"][items], catalog["duration_cap"])
            ).mean(1),
            users,
        ),
        mean_duration=user_macro(catalog["duration"][items].mean(1), users),
        mean_log_popularity=user_macro(
            np.log1p(catalog["frequency"][items]).mean(1), users
        ),
        seen_fraction=user_macro(seen, users),
        duplicate_fraction=user_macro(repeats, users),
        catalog_coverage=float((counts[3:] > 0).mean()),
        top10_exposure_share=float(np.sort(counts)[-10:].sum() / counts.sum()),
    )
    _, first = np.unique(users, return_index=True)
    lists = [set(items[idx]) for idx in first]
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, len(lists), size=(1000, 2))
    jaccard = [
        len(lists[a] & lists[b]) / len(lists[a] | lists[b]) for a, b in pairs if a != b
    ]
    metrics["between_user_jaccard"] = float(np.mean(jaccard)) if jaccard else 0.0
    individual["predicted_reward"] = predicted[:, :k].mean(1)
    return metrics, individual


@torch.no_grad()
def evaluate_topk(
    policy, data, features, rm, auditor, catalog, device, mode="policy", alpha=0
):
    if policy is not None:
        policy.eval()
    durations = tensor(catalog["duration"], device, torch.float32)
    recommendations, rewards, audit_rewards = [], [], []
    for start in range(0, len(data["target"]), 128):
        end = start + 128
        context = tensor(data["ctx"][start:end], device)
        h = tensor(features[start:end], device, torch.float32)
        feedback_summary = summarize_feedback(
            context, tensor(data["feedback"][start:end], device, torch.float32)
        )
        if mode == "popularity":
            scores = tensor(catalog["frequency"], device, torch.float32).expand(
                len(context), -1
            )
        else:
            scores = policy(context).log_softmax(-1)
        if alpha:
            candidate_scores, candidates = scores.topk(
                min(100, scores.shape[1] - 3), dim=1
            )
            candidate_scores = candidate_scores + alpha * score_items(
                rm, h, candidates, durations, feedback_summary
            )
            recs = candidates.gather(1, candidate_scores.topk(20, dim=1).indices)
        else:
            recs = scores.topk(20, dim=1).indices
        recommendations.append(recs.cpu().numpy())
        rewards.append(
            score_items(rm, h, recs, durations, feedback_summary).cpu().numpy()
        )
        audit_rewards.append(
            score_items(auditor, h, recs, durations, feedback_summary).cpu().numpy()
        )
    return (
        np.concatenate(recommendations),
        np.concatenate(rewards),
        np.concatenate(audit_rewards),
    )


def evaluate_reward_models(config, datasets, catalog, models, features, device):
    output = {}
    calibration = []
    slices = []
    train = datasets["train"]
    global_mean = float(train["reward"].mean())
    item_counts = np.bincount(train["target"], minlength=len(catalog["duration"]))
    item_sums = np.bincount(
        train["target"], weights=train["reward"], minlength=len(catalog["duration"])
    )
    item_mean = (item_sums + 20 * global_mean) / (item_counts + 20)
    for split in ("val", "test"):
        data = datasets[split]
        known = data["target"] >= 3
        predictions = {
            "global_mean": np.full(len(data["target"]), global_mean),
            "item_mean": item_mean[data["target"]],
            "item_mean_all_train": catalog["item_reward"][data["target"]],
            "history_mean": (data["feedback"] * (data["ctx"] != 0)).sum(1)
            / (data["ctx"] != 0).sum(1),
        }
        for name, model in models.items():
            predictions[name] = predict_reward(model, features[split], data, device)
        for name, pred in predictions.items():
            output[f"{split}/{name}"] = {
                **regression_metrics(data["reward"][known], pred[known]),
                "examples": int(known.sum()),
            }
            for lower in np.arange(0, 1, 0.1):
                upper = lower + 0.1
                if lower > 0.85:
                    upper += 1e-7  # В последний интервал входит и прогноз 1.
                mask = known & (pred >= lower) & (pred < upper)
                if mask.any():
                    calibration.append(
                        {
                            "split": split,
                            "model": name,
                            "bin": float(lower),
                            "count": int(mask.sum()),
                            "predicted": float(pred[mask].mean()),
                            "observed": float(data["reward"][mask].mean()),
                        }
                    )
            masks = {
                "short_<=180s": data["duration"] <= 180,
                "long_>180s": data["duration"] > 180,
                "organic": data["organic"] == 1,
                "recommended": data["organic"] == 0,
                "popular": catalog["frequency"][data["target"]]
                >= np.median(catalog["frequency"][3:]),
                "less_popular": catalog["frequency"][data["target"]]
                < np.median(catalog["frequency"][3:]),
            }
            for label, mask in masks.items():
                mask = known & mask
                if mask.any():
                    slices.append(
                        {
                            "split": split,
                            "model": name,
                            "slice": label,
                            "count": int(mask.sum()),
                            **regression_metrics(data["reward"][mask], pred[mask]),
                        }
                    )
    save_json("results/reward_metrics.json", output)
    pd.DataFrame(calibration).to_csv("results/reward_calibration.csv", index=False)
    pd.DataFrame(slices).to_csv("results/reward_slices.csv", index=False)


@torch.no_grad()
def rollout_diagnostics(
    name, policy, reference, rm, auditor, data, catalog, device, unique=False
):
    """Пять шагов с предполагаемым принятием и прогнозом отклика."""
    policy.eval()
    duration = tensor(catalog["duration"], device, torch.float32)
    all_recs, all_fixed, all_audit = [], [], []
    curve_rows = []
    for start in range(0, len(data["target"]), 64):
        initial = tensor(data["ctx"][start : start + 64], device)
        context = initial.clone()
        feedback = tensor(data["feedback"][start : start + 64], device, torch.float32)
        initial_summary = summarize_feedback(initial, feedback)
        initial_h = reference.encode(initial)
        recs, fixed, auditing = [], [], []
        for step in range(5):
            logits = policy(context)
            ref_logits = reference(context)
            if unique and recs:
                generated = tensor(np.stack(recs, 1), device)
                logits.scatter_(1, generated, -1e9)
                ref_logits.scatter_(1, generated, -1e9)
            logp = logits.log_softmax(-1)
            action = logp.argmax(-1)
            h = reference.encode(context)
            ref_logp = ref_logits.log_softmax(-1)
            summary = summarize_feedback(context, feedback)
            r_aug = rm(h, action, duration[action], summary)
            r_fixed = rm(initial_h, action, duration[action], initial_summary)
            r_ids_only = rm(h, action, duration[action], initial_summary)
            r_audit = auditor(h, action, duration[action], summary)
            kl = (logp.exp() * (logp - ref_logp)).sum(-1)
            repeated = context.eq(action[:, None]).any(1).float()
            for j in range(len(initial)):
                curve_rows.append(
                    {
                        "model": name,
                        "example": start + j,
                        "uid": int(data["uid"][start + j]),
                        "step": step + 1,
                        "mode": "next5_unique" if unique else "next5",
                        "reward_augmented": float(r_aug[j]),
                        "reward_fixed": float(r_fixed[j]),
                        "reward_ids_only": float(r_ids_only[j]),
                        "auditor_augmented": float(r_audit[j]),
                        "kl": float(kl[j]),
                        "repeat_in_context": float(repeated[j]),
                        "duration": float(duration[action[j]]),
                        "reference_nll": float(-ref_logp[j, action[j]]),
                    }
                )
            recs.append(action.cpu().numpy())
            fixed.append(r_fixed.cpu().numpy())
            # Для сравнения оцениваем выдачу по исходной истории.
            auditing.append(
                auditor(initial_h, action, duration[action], initial_summary)
                .cpu()
                .numpy()
            )
            feedback = append_feedback(context, feedback, r_aug)
            context = append_item(context, action)
        all_recs.append(np.stack(recs, 1))
        all_fixed.append(np.stack(fixed, 1))
        all_audit.append(np.stack(auditing, 1))
    recs = np.concatenate(all_recs)
    fixed = np.concatenate(all_fixed)
    auditing = np.concatenate(all_audit)
    if unique:
        assert all(len(set(row)) == 5 for row in recs), (
            "Constraint failed: repeated item in next5_unique"
        )
    metrics, individual = summarize_lists(recs, data, fixed, auditing, catalog)
    return metrics, curve_rows, recs


def evaluate(config, datasets, catalog, device):
    Path("results/per_query").mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("results/rl_manifest.json").read_text())
    vocab = len(catalog["duration"])
    models = {
        name: load_reward(
            f"results/{name}.pt", catalog, config, device, name == "no_history"
        )
        for name in ("reward", "auditor", "no_history")
    }
    features = {
        split: np.load(f"results/features_{split}.npy") for split in ("val", "test")
    }
    evaluate_reward_models(config, datasets, catalog, models, features, device)
    variants = [
        {"run": "popularity", "checkpoint": None},
        {"run": "base", "checkpoint": "results/base.pt"},
        {"run": "continued", "checkpoint": "results/continued.pt"},
    ]
    variants += [
        {"run": f"rerank_a{alpha}", "checkpoint": "results/base.pt", "alpha": alpha}
        for alpha in (1, 5, 10)
    ]
    variants += manifest
    rows, intervals, examples = [], [], []
    selected = None
    test_individual = {}
    for split in ("val", "test"):
        # Выбор по validation, до оценки test.
        if split == "test":
            val = pd.DataFrame(rows)
            floor = float(val.loc[val.model == "base", "ndcg@10"].iloc[0]) * 0.95
            candidate = val[(val["ndcg@10"] >= floor) & val.model.str.contains("_s")]
            if len(candidate):
                candidate = candidate.sort_values("predicted_reward", ascending=False)
                selected = candidate.iloc[0].model
            else:
                selected = "base"
            save_json(
                "results/selection.json",
                {
                    "selected": selected,
                    "rule": "Maximum validation top-10 predicted reward among RL runs with NDCG@10 >= 95% of base; otherwise base",
                    "ndcg_floor": floor,
                },
            )
        for variant in variants:
            name = variant["run"]
            policy = None
            if variant["checkpoint"]:
                policy = load_policy(variant["checkpoint"], config, vocab, device).eval()
            recs, pred, audit = evaluate_topk(
                policy,
                datasets[split],
                features[split],
                models["reward"],
                models["auditor"],
                catalog,
                device,
                mode="popularity" if name == "popularity" else "policy",
                alpha=variant.get("alpha", 0),
            )
            metrics, individual = summarize_lists(
                recs, datasets[split], pred, audit, catalog
            )
            rows.append({"split": split, "model": name, **metrics})
            np.savez_compressed(
                f"results/per_query/{split}_{name}.npz",
                uid=datasets[split]["uid"],
                time=datasets[split]["time"],
                recs=recs,
                **individual,
            )
            if split == "test":
                test_individual[name] = individual
                if name in ("base", selected):
                    for idx in range(min(10, len(recs))):
                        examples.append(
                            {
                                "model": name,
                                "uid": int(datasets[split]["uid"][idx]),
                                "time": int(datasets[split]["time"][idx]),
                                "history_tokens": datasets[split]["ctx"][idx].tolist(),
                                "target_token": int(datasets[split]["target"][idx]),
                                "recommendation_item_ids": catalog["item_ids"][
                                    recs[idx, :10]
                                ].tolist(),
                                "predicted_rewards": pred[idx, :10].tolist(),
                            }
                        )
            print(
                "EVAL",
                split,
                name,
                "NDCG",
                round(metrics["ndcg@10"], 5),
                "RM",
                round(metrics["predicted_reward"], 4),
                flush=True,
            )
            pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)
    for name, values in test_individual.items():
        for metric in ("ndcg10", "recall10", "predicted_reward"):
            intervals.append(
                {
                    "model": name,
                    "metric": metric,
                    **paired_bootstrap(
                        values[metric],
                        test_individual["base"][metric],
                        datasets["test"]["uid"],
                        config["bootstrap_samples"],
                    ),
                }
            )
    pd.DataFrame(intervals).to_csv("results/confidence_intervals.csv", index=False)
    save_json("results/examples.json", examples)
    evaluate_rollouts(config, datasets, catalog, device)


def evaluate_rollouts(config, datasets, catalog, device):
    """Сравнить Top-5, Next-5 и Next-5 без дублей."""
    Path("results/per_query").mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("results/rl_manifest.json").read_text())
    selected = json.loads(Path("results/selection.json").read_text())["selected"]
    vocab = len(catalog["duration"])
    reference = load_policy("results/base.pt", config, vocab, device).eval()
    models = {
        name: load_reward(f"results/{name}.pt", catalog, config, device)
        for name in ("reward", "auditor")
    }
    features = np.load("results/features_test.npy")
    variants = [{"run": "base", "checkpoint": "results/base.pt"}] + manifest
    rng = np.random.default_rng(config["seed"])
    ids = np.sort(
        rng.choice(
            len(datasets["test"]["target"]),
            min(config["rollout_eval_examples"], len(datasets["test"]["target"])),
            replace=False,
        )
    )
    data = subset(datasets["test"], ids)
    curve_rows, rollout_metrics = [], []
    names = {
        "base",
        selected,
        f"ratio_h5_b005_s{config['rl_seeds'][0]}",
        f"ratio_b0_s{config['rl_seeds'][0]}",
    }
    for variant in variants:
        if variant["run"] not in names:
            continue
        name = variant["run"]
        policy = load_policy(variant["checkpoint"], config, vocab, device).eval()
        metrics, curves, recs = rollout_diagnostics(
            name,
            policy,
            reference,
            models["reward"],
            models["auditor"],
            data,
            catalog,
            device,
        )
        rollout_metrics.append({"model": name, "mode": "next5", **metrics})
        top, r, a = evaluate_topk(
            policy,
            data,
            features[ids],
            models["reward"],
            models["auditor"],
            catalog,
            device,
        )
        top_metrics, _ = summarize_lists(top[:, :5], data, r[:, :5], a[:, :5], catalog)
        rollout_metrics.append({"model": name, "mode": "top5", **top_metrics})
        curve_rows.extend(curves)
        unique_metrics, unique_curves, unique_recs = rollout_diagnostics(
            name,
            policy,
            reference,
            models["reward"],
            models["auditor"],
            data,
            catalog,
            device,
            unique=True,
        )
        rollout_metrics.append(
            {"model": name, "mode": "next5_unique", **unique_metrics}
        )
        curve_rows.extend(unique_curves)
        np.savez_compressed(
            f"results/per_query/rollout_{name}.npz",
            uid=data["uid"],
            time=data["time"],
            next5=recs,
            top5=top[:, :5],
            next5_unique=unique_recs,
        )
        print("ROLLOUT", name, flush=True)
    pd.DataFrame(curve_rows).to_csv("results/rollout_steps.csv", index=False)
    pd.DataFrame(rollout_metrics).to_csv("results/rollout_metrics.csv", index=False)
