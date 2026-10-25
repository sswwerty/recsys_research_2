"""Обучение GPT-2 и RM; выбор checkpoint по validation."""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .models import GPTRec, RewardModel, load_policy, summarize_feedback
from .utils import (
    batches,
    rank_metrics,
    regression_metrics,
    save_json,
    seed_everything,
    tensor,
    user_macro,
)


@torch.no_grad()
def validation_ndcg(model, data, device, batch_size=128):
    model.eval()
    values = []
    for batch in batches(data, batch_size):
        recs = model(tensor(batch["ctx"], device)).topk(10).indices.cpu().numpy()
        values.extend(rank_metrics(recs, batch["target"], 10)[1])
    return user_macro(np.array(values), data["uid"])


def train_base(config, datasets, catalog, device, continuation=False):
    seed_everything(config["seed"])
    model = GPTRec(len(catalog["duration"]), config).to(device)
    if continuation:
        model.load_state_dict(
            torch.load("results/base.pt", map_location=device, weights_only=True)
        )
    if continuation:
        learning_rate = config["rl_lr"]
        training_batch_size = config["rl_batch"] * config["rl_group"]
        steps_per_epoch = math.ceil(len(datasets["train"]["target"]) / training_batch_size)
        epochs = math.ceil(config["rl_steps"] / steps_per_epoch)
        prefix = "continued"
    else:
        learning_rate = config["base_lr"]
        training_batch_size = config["batch_size"]
        epochs = config["base_epochs"]
        prefix = "base"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    best = -1.0
    stale_epochs = 0
    history = []
    start = time.monotonic()
    total_steps = 0
    for epoch in range(epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in batches(
            datasets["train"],
            training_batch_size,
            shuffle=True,
            seed=config["seed"] + epoch,
            known_only=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensor(batch["ctx"], device))
            loss = F.cross_entropy(logits, tensor(batch["target"], device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch["target"])
            count += len(batch["target"])
            total_steps += 1
            if total_steps % 100 == 0:
                print(
                    prefix,
                    "epoch",
                    epoch + 1,
                    "step",
                    total_steps,
                    "loss",
                    round(loss_sum / count, 4),
                    "seconds",
                    round(time.monotonic() - start),
                    flush=True,
                )
            if continuation and total_steps >= config["rl_steps"]:
                break
        score = validation_ndcg(model, datasets["val"], device)
        row = dict(
            epoch=epoch + 1,
            loss=loss_sum / count,
            val_ndcg10=score,
            seconds=time.monotonic() - start,
        )
        history.append(row)
        print(prefix, row, flush=True)
        if continuation or score > best:
            best = score
            stale_epochs = 0
            torch.save(model.state_dict(), f"results/{prefix}.pt")
        else:
            stale_epochs += 1
        if continuation and total_steps >= config["rl_steps"]:
            break
        if stale_epochs >= 3:
            print(
                prefix,
                "early stopping: 3 epochs without validation improvement",
                flush=True,
            )
            break
    save_json(
        f"results/{prefix}_training.json",
        {
            "history": history,
            "parameters": sum(p.numel() for p in model.parameters()),
            "steps": total_steps,
            "config": config,
        },
    )


@torch.no_grad()
def encode_dataset(model, data, device, batch_size):
    model.eval()
    result = []
    for batch in batches(data, batch_size):
        result.append(model.encode(tensor(batch["ctx"], device)).cpu().numpy())
    return np.concatenate(result)


def reward_features(config, datasets, catalog, device):
    encoder = load_policy(
        "results/base.pt", config, len(catalog["duration"]), device
    ).eval()
    features = {}
    for split in datasets:
        path = Path(f"results/features_{split}.npy")
        # Пересчитываем признаки после смены baseline.
        features[split] = encode_dataset(
            encoder, datasets[split], device, config["batch_size"]
        )
        np.save(path, features[split])
        print("encoded", split, len(features[split]), flush=True)
    return encoder, features


@torch.no_grad()
def predict_reward(model, features, data, device):
    model.eval()
    result = []
    for start in range(0, len(features), 1024):
        end = start + 1024
        history = tensor(features[start:end], device, torch.float32)
        items = tensor(data["target"][start:end], device)
        duration = tensor(data["duration"][start:end], device, torch.float32)
        context = tensor(data["ctx"][start:end], device)
        feedback = tensor(data["feedback"][start:end], device, torch.float32)
        summary = summarize_feedback(context, feedback)
        prediction = model(history, items, duration, summary)
        result.append(prediction.cpu().numpy())
    return np.concatenate(result)


def train_reward(config, datasets, catalog, device):
    encoder, features = reward_features(config, datasets, catalog, device)
    output = {}
    for name, seed, no_history in [
        ("reward", 43, False),
        ("auditor", 97, False),
        ("no_history", 43, True),
    ]:
        seed_everything(seed)
        model = RewardModel(
            len(catalog["duration"]),
            config["hidden_size"],
            catalog["duration_scale"],
            no_history,
        ).to(device)
        with torch.no_grad():
            model.item_embedding.weight.copy_(encoder.backbone.wte.weight)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["rm_lr"], weight_decay=0.01
        )
        best = float("inf")
        stale = 0
        log = []
        val_known = datasets["val"]["target"] >= 3
        started = time.monotonic()
        for epoch in range(config["rm_epochs"]):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(
                len(features["train"])
            )
            losses = []
            for start in range(0, len(order), 512):
                idx = order[start : start + 512]
                optimizer.zero_grad(set_to_none=True)
                train = datasets["train"]
                history = tensor(features["train"][idx], device, torch.float32)
                items = tensor(train["target"][idx], device)
                duration = tensor(train["duration"][idx], device, torch.float32)
                context = tensor(train["ctx"][idx], device)
                feedback = tensor(train["feedback"][idx], device, torch.float32)
                target = tensor(train["reward"][idx], device, torch.float32)
                summary = summarize_feedback(context, feedback)
                prediction = model(history, items, duration, summary)
                loss = F.mse_loss(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            prediction = predict_reward(model, features["val"], datasets["val"], device)
            metrics = regression_metrics(
                datasets["val"]["reward"][val_known], prediction[val_known]
            )
            log.append(
                dict(epoch=epoch + 1, train_mse=float(np.mean(losses)), **metrics)
            )
            print(name, log[-1], flush=True)
            if metrics["rmse"] < best:
                best = metrics["rmse"]
                stale = 0
                torch.save(model.state_dict(), f"results/{name}.pt")
            else:
                stale += 1
            if stale >= 2:
                break
        output[name] = {
            "history": log,
            "seconds": time.monotonic() - started,
            "seed": seed,
        }
    save_json("results/reward_training.json", output)
