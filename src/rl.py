"""On-policy REINFORCE по прогнозам замороженной RM."""

import time

import numpy as np
import torch

from .models import (
    append_feedback,
    append_item,
    load_policy,
    load_reward,
    summarize_feedback,
)
from .train import validation_ndcg
from .utils import save_json, seed_everything, subset, tensor

EXPERIMENTS = [
    {"name": "ratio_b0", "beta": 0.0, "reward": "ratio", "horizon": 1},
    {"name": "ratio_b005", "beta": 0.05, "reward": "ratio", "horizon": 1},
    {"name": "ratio_b02", "beta": 0.2, "reward": "ratio", "horizon": 1},
    {"name": "seconds_b005", "beta": 0.05, "reward": "seconds", "horizon": 1},
    {"name": "repeat_b005", "beta": 0.05, "reward": "repeat", "horizon": 1},
    {"name": "ratio_h5_b005", "beta": 0.05, "reward": "ratio", "horizon": 5},
]


def leave_one_out(returns, group_size):
    """Награда минус среднее остальных продолжений той же истории."""
    grouped = returns.reshape(-1, group_size)
    other_rewards = grouped.sum(dim=1, keepdim=True) - grouped
    baseline = other_rewards / (group_size - 1)
    return (grouped - baseline).reshape(-1)


def generate_rollout(
    actor, reference, reward_model, context, feedback, duration, duration_cap, experiment
):
    """Выбрать треки, оценить награды и обновить историю."""
    log_probabilities = []
    rewards = []
    log_ratios = []
    exact_kls = []
    for _ in range(experiment["horizon"]):
        logp = actor(context).log_softmax(dim=-1)
        probabilities = logp.exp()
        action = torch.multinomial(probabilities, 1).squeeze(-1)
        chosen_logp = logp.gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            ref_logp = reference(context).log_softmax(dim=-1)
            history = reference.encode(context)
            summary = summarize_feedback(context, feedback)
            prediction = reward_model(history, action, duration[action], summary)
            reward = prediction
            if experiment["reward"] == "seconds":
                reward = reward * duration[action].clamp(max=duration_cap) / duration_cap
            elif experiment["reward"] == "repeat":
                repeated = context.eq(action[:, None]).any(dim=1)
                reward = reward - 0.2 * repeated.float()
        reference_action_logp = ref_logp.gather(1, action[:, None]).squeeze(1)
        log_probabilities.append(chosen_logp)
        rewards.append(reward)
        log_ratios.append(chosen_logp - reference_action_logp)
        exact_kls.append((probabilities * (logp - ref_logp)).sum(dim=1))
        feedback = append_feedback(context, feedback, prediction)
        context = append_item(context, action)
    return {
        "logp": torch.stack(log_probabilities, dim=1).sum(dim=1),
        "reward": torch.stack(rewards, dim=1).mean(dim=1),
        "kl": torch.stack(exact_kls, dim=1).mean(dim=1),
        "sampled_kl": torch.stack(log_ratios, dim=1).mean(dim=1),
    }


def reinforce_loss(rollout, beta, group_size, horizon):
    """Ошибка REINFORCE со штрафом за отклонение от исходной модели."""
    if horizon == 1:
        advantage = leave_one_out(rollout["reward"], group_size)
        policy_loss = -(advantage.detach() * rollout["logp"]).mean()
        return policy_loss + beta * rollout["kl"].mean()
    # Градиент всей траектории учитывает будущие состояния и KL.
    returns = rollout["reward"] - beta * rollout["sampled_kl"]
    advantage = leave_one_out(returns, group_size)
    return -(advantage.detach() * rollout["logp"]).mean()


def train_rl(config, datasets, catalog, device):
    vocab = len(catalog["duration"])
    reference = load_policy("results/base.pt", config, vocab, device).eval()
    reward_model = load_reward("results/reward.pt", catalog, config, device)
    reference.requires_grad_(False)
    reward_model.requires_grad_(False)
    duration = tensor(catalog["duration"], device, torch.float32)
    duration_cap = float(catalog["duration_cap"])
    rng = np.random.default_rng(config["seed"])
    validation_size = len(datasets["val"]["target"])
    validation_count = min(config["rl_validation_examples"], validation_size)
    validation_ids = rng.choice(validation_size, validation_count, replace=False)
    validation = subset(datasets["val"], validation_ids)
    manifest = []
    for experiment in EXPERIMENTS:
        for seed in config["rl_seeds"]:
            seed_everything(seed)
            sample_rng = np.random.default_rng(seed)
            # Отключаем dropout, сохраняя градиенты.
            actor = load_policy("results/base.pt", config, vocab, device).eval()
            optimizer = torch.optim.AdamW(
                actor.parameters(), lr=config["rl_lr"], weight_decay=0.0
            )
            name = f"{experiment['name']}_s{seed}"
            history = []
            started = time.monotonic()
            train = datasets["train"]
            for step in range(1, config["rl_steps"] + 1):
                ids = sample_rng.integers(0, len(train["target"]), config["rl_batch"])
                context = tensor(train["ctx"][ids], device)
                feedback = tensor(train["feedback"][ids], device, torch.float32)
                context = context.repeat_interleave(config["rl_group"], dim=0)
                feedback = feedback.repeat_interleave(config["rl_group"], dim=0)
                optimizer.zero_grad(set_to_none=True)
                rollout = generate_rollout(
                    actor, reference, reward_model, context, feedback,
                    duration, duration_cap, experiment,
                )
                loss = reinforce_loss(
                    rollout, experiment["beta"], config["rl_group"], experiment["horizon"]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite RL loss: {name} step {step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                if (
                    step == 1
                    or step % config["rl_eval_every"] == 0
                    or step == config["rl_steps"]
                ):
                    row = {
                        "step": step,
                        "reward": float(rollout["reward"].mean()),
                        "kl": float(rollout["kl"].detach().mean()),
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "val_ndcg10": validation_ndcg(actor, validation, device),
                        "seconds": time.monotonic() - started,
                    }
                    history.append(row)
                    print(name, row, flush=True)
            checkpoint = f"results/{name}.pt"
            torch.save(actor.state_dict(), checkpoint)
            save_json(
                f"results/{name}_training.json",
                {"experiment": experiment, "seed": seed, "history": history},
            )
            manifest.append(
                {
                    **experiment,
                    "run": name,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "seconds": time.monotonic() - started,
                }
            )
            save_json("results/rl_manifest.json", manifest)
