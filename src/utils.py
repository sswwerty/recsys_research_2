"""Данные, метрики и bootstrap по пользователям."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_for(config):
    if config["device"] != "auto":
        return torch.device(config["device"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(data, batch_size, shuffle=False, seed=0, known_only=False):
    if known_only:
        ids = np.flatnonzero(data["target"] >= 3)
    else:
        ids = np.arange(len(data["target"]))
    if shuffle:
        np.random.default_rng(seed).shuffle(ids)
    for start in range(0, len(ids), batch_size):
        idx = ids[start : start + batch_size]
        yield {key: value[idx] for key, value in data.items()}


def tensor(array, device, dtype=torch.long):
    return torch.as_tensor(array, dtype=dtype, device=device)


def subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def load_data():
    return {
        split: dict(np.load(f"data/processed/{split}.npz"))
        for split in ("train", "val", "test")
    }


def load_catalog():
    return dict(np.load("data/processed/catalog.npz"))


def rank_metrics(recommendations, targets, k):
    hits = recommendations[:, :k] == targets[:, None]
    # Повторное попадание не добавляет баллов.
    found = hits.any(axis=1)
    rank = hits.argmax(axis=1) + 1
    recall = found.astype(float)
    ndcg = np.where(found, 1.0 / np.log2(rank + 1), 0.0)
    return recall, ndcg


def user_macro(values, users):
    _, user_index = np.unique(users, return_inverse=True)
    sums = np.bincount(user_index, weights=values)
    counts = np.bincount(user_index)
    user_means = sums / counts
    return float(user_means.mean())


def paired_bootstrap(values, reference, users, samples=500, seed=42):
    _, user_index = np.unique(users, return_inverse=True)
    sums = np.bincount(user_index, weights=values - reference)
    counts = np.bincount(user_index)
    differences = sums / counts
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        sample = rng.choice(differences, len(differences), replace=True)
        boot.append(sample.mean())
    return {
        "mean": float(differences.mean()),
        "low": float(np.quantile(boot, 0.025)),
        "high": float(np.quantile(boot, 0.975)),
    }


def regression_metrics(target, prediction):
    error = prediction - target
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_prediction": float(prediction.mean()),
        "mean_target": float(target.mean()),
    }
