"""Подготовка данных и временное разбиение."""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .utils import save_json

PAD, UNK, BOS = 0, 1, 2
REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
SOURCE = f"https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}/flat/50m/listens.parquet"


def strict_context(tokens, timestamps, position, length):
    """История строго раньше цели."""
    end = np.searchsorted(timestamps, timestamps[position], side="left")
    previous = tokens[max(0, end - length) : end]
    result = np.zeros(length, dtype=np.int32)
    if len(previous):
        result[: len(previous)] = previous
    else:
        result[0] = BOS
    return result, int(end)


def prepare(config):
    path = Path("data/raw/listens.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Сначала скачайте {SOURCE} в {path}")
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(path)
    overview = parquet.read(columns=["uid", "timestamp"]).to_pandas()
    low, high = int(overview.timestamp.min()), int(overview.timestamp.max())
    train_end = int(low + (high - low) * config["train_fraction"])
    val_end = int(low + (high - low) * config["val_fraction"])
    counts = overview.loc[overview.timestamp < train_end, "uid"].value_counts()
    eligible = np.sort(counts[counts >= config["min_train_events"]].index.to_numpy())
    rng = np.random.default_rng(config["seed"])
    chosen = np.sort(
        rng.choice(eligible, min(config["users"], len(eligible)), replace=False)
    )
    del overview
    parts = []
    for batch in parquet.iter_batches(batch_size=500000):
        frame = batch.to_pandas()
        part = frame[frame.uid.isin(chosen)]
        if len(part):
            parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    raw_rows = len(frame)
    valid = (
        frame[
            ["uid", "item_id", "timestamp", "played_ratio_pct", "track_length_seconds"]
        ]
        .notna()
        .all(axis=1)
    )
    valid &= (frame.played_ratio_pct >= 0) & (frame.track_length_seconds > 0)
    frame = (
        frame[valid]
        .sort_values(["uid", "timestamp", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    train = frame[frame.timestamp < train_end]
    popular = (
        train.item_id.value_counts().rename_axis("item_id").reset_index(name="count")
    )
    popular = popular.sort_values(["count", "item_id"], ascending=[False, True])
    item_ids = popular.head(config["catalog_size"]).item_id.to_numpy()
    mapping = {int(item): idx + 3 for idx, item in enumerate(item_ids)}
    frame["token"] = frame.item_id.map(mapping).fillna(UNK).astype(np.int32)
    frame["reward"] = np.clip(frame.played_ratio_pct.to_numpy() / 100.0, 0, 1)
    train = frame[frame.timestamp < train_end]
    size = len(item_ids) + 3
    frequency = np.zeros(size, np.float32)
    duration = np.zeros(size, np.float32)
    global_reward = train.reward.mean()
    item_reward = np.full(size, global_reward, np.float32)
    stats = (
        train[train.token >= 3]
        .groupby("token")
        .agg(
            frequency=("token", "size"),
            duration=("track_length_seconds", "median"),
            reward=("reward", "sum"),
        )
    )
    for token, row in stats.iterrows():
        frequency[token] = row.frequency
        duration[token] = row.duration
        item_reward[token] = (row.reward + 20 * global_reward) / (
            row.frequency + 20
        )
    duration_cap = float(np.quantile(train.track_length_seconds, 0.95))
    duration_scale = float(np.log1p(train.track_length_seconds).mean())
    np.savez(
        out / "catalog.npz",
        item_ids=np.concatenate(([-1, -2, -3], item_ids)),
        frequency=frequency,
        duration=duration,
        item_reward=item_reward,
        global_reward=np.float32(global_reward),
        duration_cap=np.float32(duration_cap),
        duration_scale=np.float32(duration_scale),
    )
    selected = {name: [] for name in ("train", "val", "test")}
    groups = list(frame.groupby("uid", sort=True).indices.items())
    train_candidates = []
    coverage = {}
    for split, left, right in [
        ("train", low, train_end),
        ("val", train_end, val_end),
        ("test", val_end, high + 1),
    ]:
        rows = frame[(frame.timestamp >= left) & (frame.timestamp < right)]
        coverage[split] = {
            "raw_events": len(rows),
            "catalog_target_fraction": float((rows.token >= 3).mean()),
        }
    all_times = frame.timestamp.to_numpy()
    all_tokens = frame.token.to_numpy()
    for uid, indexes in groups:
        times = all_times[indexes]
        history_size = np.searchsorted(times, times, side="left")
        ready = history_size >= config["min_history"]
        train_positions = indexes[
            ready & (times < train_end) & (all_tokens[indexes] >= 3)
        ]
        train_candidates.extend(train_positions.tolist())
        for split, left, right in [
            ("val", train_end, val_end),
            ("test", val_end, high + 1),
        ]:
            positions = indexes[ready & (times >= left) & (times < right)]
            if len(positions):
                selected[split].extend(
                    rng.choice(
                        positions,
                        min(config["eval_per_user"], len(positions)),
                        replace=False,
                    ).tolist()
                )
    selected["train"] = rng.choice(
        train_candidates,
        min(config["train_examples"], len(train_candidates)),
        replace=False,
    ).tolist()
    chosen_positions = {split: set(positions) for split, positions in selected.items()}
    fields = [
        "ctx", "target", "reward", "uid", "time", "history_end", "duration",
        "organic", "feedback",
    ]
    examples = {}
    for split in selected:
        examples[split] = {field: [] for field in fields}
    for uid, indexes in groups:
        user = frame.iloc[indexes]
        tokens = user.token.to_numpy()
        times = user.timestamp.to_numpy()
        for split in selected:
            positions = [
                j for j, idx in enumerate(indexes) if idx in chosen_positions[split]
            ]
            for j in positions:
                ctx, end = strict_context(tokens, times, j, config["context_length"])
                past_feedback = user.reward.to_numpy()[
                    max(0, end - config["context_length"]) : end
                ]
                feedback = np.zeros(config["context_length"], np.float32)
                feedback[: len(past_feedback)] = past_feedback
                row = user.iloc[j]
                data = examples[split]
                data["ctx"].append(ctx)
                data["target"].append(int(row.token))
                data["reward"].append(float(row.reward))
                data["uid"].append(int(uid))
                data["time"].append(int(row.timestamp))
                data["history_end"].append(int(times[end - 1]))
                data["duration"].append(float(row.track_length_seconds))
                data["organic"].append(int(row.is_organic))
                data["feedback"].append(feedback)
    for split, columns in examples.items():
        if not columns["target"]:
            raise ValueError(f"Пустой split: {split}; увеличьте выборку пользователей")
        data = {}
        for field, values in columns.items():
            dtype = np.int64
            if field in ("reward", "duration", "feedback"):
                dtype = np.float32
            elif field == "ctx":
                dtype = np.int32
            data[field] = np.asarray(values, dtype=dtype)
        assert (data["history_end"] < data["time"]).all()
        np.savez_compressed(out / f"{split}.npz", **data)
        coverage[split].update(
            examples=len(data["target"]),
            users=len(np.unique(data["uid"])),
            evaluation_catalog_fraction=float((data["target"] >= 3).mean()),
        )
    audit = {
        "source": SOURCE,
        "revision": REVISION,
        "sha256": checksum(path),
        "raw_file_rows": parquet.metadata.num_rows,
        "selected_users": len(chosen),
        "selected_rows": len(frame),
        "invalid_rows_removed": raw_rows - len(frame),
        "catalog_size": len(item_ids),
        "time_min": low,
        "train_end": train_end,
        "val_end": val_end,
        "time_max": high,
        "split_rule": "Global time range: first 80% train, next 10% validation, last 10% test (configurable)",
        "ratio_above_100_fraction": float((frame.played_ratio_pct > 100).mean()),
        "same_user_timestamp_duplicates": int(
            frame.duplicated(["uid", "timestamp"]).sum()
        ),
        "raw_ratio_quantiles": {
            str(q): float(frame.played_ratio_pct.quantile(q))
            for q in (0, 0.5, 0.9, 0.99, 1)
        },
        "duration_quantiles": {
            str(q): float(frame.track_length_seconds.quantile(q))
            for q in (0, 0.5, 0.9, 0.99, 1)
        },
        "splits": coverage,
        "config": config,
    }
    save_json(out / "audit.json", audit)
    save_json(out / "users.json", chosen.tolist())
    print("DATA", coverage, flush=True)


EXPECTED_SHA256 = "eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1"


def checksum(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download():
    """Скачать фиксированную версию Yambda и проверить контрольную сумму."""
    destination = Path("data/raw/listens.parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checksum(destination) != EXPECTED_SHA256:
            raise RuntimeError(
                "Существующий файл не совпадает с ожидаемой версией; проверьте его вручную."
            )
        print("Данные уже загружены; SHA-256 совпадает.")
    else:
        temporary = destination.with_suffix(".parquet.part")
        request = urllib.request.Request(
            SOURCE, headers={"User-Agent": "tlab-recsys-research/1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            total = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                total += len(chunk)
                if total % (32 * 1024 * 1024) == 0:
                    print(f"Downloaded {total / 1024**2:.0f} MiB", flush=True)
        if checksum(temporary) != EXPECTED_SHA256:
            raise RuntimeError(
                "SHA-256 скачанного файла не совпадает; файл .part оставлен для диагностики."
            )
        os.replace(temporary, destination)
        print("Готово:", destination)
