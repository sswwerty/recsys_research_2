"""Запуск эксперимента по этапам: python run.py all или python run.py report."""

import argparse
import json
import os
import platform
from pathlib import Path

import torch
import transformers

from src.utils import device_for, load_catalog, load_data, save_json

ROOT = Path(__file__).resolve().parent


def validate_config(config):
    """Проверить условия, от которых зависят окна, attention, метрики и RL."""
    positive = (
        "users",
        "context_length",
        "train_examples",
        "eval_per_user",
        "batch_size",
        "base_epochs",
        "rm_epochs",
        "rl_steps",
        "rl_batch",
        "rl_eval_every",
        "rl_validation_examples",
        "rollout_eval_examples",
        "bootstrap_samples",
        "hidden_size",
        "layers",
        "heads",
        "min_train_events",
    )
    if any(config[key] <= 0 for key in positive):
        raise ValueError(
            "Размеры выборок, модели и бюджеты должны быть положительными."
        )
    if not 0 < config["train_fraction"] < config["val_fraction"] < 1:
        raise ValueError("Нужны границы 0 < train_fraction < val_fraction < 1.")
    if not 1 <= config["min_history"] <= config["context_length"]:
        raise ValueError("Нужна непустая история не длиннее окна контекста.")
    if config["catalog_size"] < 20:
        raise ValueError("Для метрик @20 нужны как минимум 20 объектов.")
    if config["hidden_size"] % config["heads"]:
        raise ValueError("hidden_size должен делиться на heads.")
    if config["rl_group"] < 2 or not config["rl_seeds"]:
        raise ValueError(
            "Leave-one-out требует минимум два rollout и непустой список seeds."
        )
    if len(set(config["rl_seeds"])) != len(config["rl_seeds"]):
        raise ValueError("Seeds должны различаться, иначе checkpoints перезапишутся.")
    if any(config[key] <= 0 for key in ("base_lr", "rm_lr", "rl_lr")):
        raise ValueError("Скорости обучения должны быть положительными.")


def check_run_config(config):
    """Не смешивать данные и веса разных конфигураций в одной папке."""
    path = Path("results/config.json")
    if path.exists():
        previous = json.loads(path.read_text())
        changes = [
            key
            for key in config
            if key != "device" and config[key] != previous.get(key)
        ]
        if changes:
            raise ValueError(
                "Настройки этого запуска отличаются: "
                + ", ".join(changes)
                + ". Укажите новую папку через --work-dir runs/<name>."
            )


def recommend(config, device, example=0, top_k=10):
    """Показать рекомендации для одной сохранённой тестовой истории."""
    from src.evaluate import score_items
    from src.models import load_policy, load_reward, summarize_feedback
    from src.utils import tensor

    data, catalog = load_data()["test"], load_catalog()
    if (
        not 0 <= example < len(data["target"])
        or not 1 <= top_k <= len(catalog["duration"]) - 3
    ):
        raise ValueError("Индекс примера или top-k вне допустимого диапазона.")
    selected = json.loads(Path("results/selection.json").read_text())["selected"]
    reference = load_policy(
        "results/base.pt", config, len(catalog["duration"]), device
    ).eval()
    rm = load_reward("results/reward.pt", catalog, config, device)
    context = tensor(data["ctx"][example : example + 1], device)
    feedback = tensor(data["feedback"][example : example + 1], device, torch.float32)
    durations = tensor(catalog["duration"], device, torch.float32)
    print(
        "Пользователь:",
        data["uid"][example],
        "цель:",
        data["target"][example],
        "(1 = вне каталога)",
    )
    with torch.no_grad():
        history = reference.encode(context)
        summary = summarize_feedback(context, feedback)
        model_names = ["base"]
        if selected != "base":
            model_names.append(selected)
        for name in model_names:
            model = load_policy(
                f"results/{name}.pt", config, len(catalog["duration"]), device
            ).eval()
            items = model(context).topk(top_k).indices
            rewards = (
                score_items(rm, history, items, durations, summary)[0].cpu().tolist()
            )
            print(name, "\nrank item_id predicted_completion duration_s")
            for rank, (token, reward) in enumerate(
                zip(items[0].cpu().tolist(), rewards), 1
            ):
                print(
                    rank,
                    int(catalog["item_ids"][token]),
                    round(reward, 4),
                    int(catalog["duration"][token]),
                )
    print("Прогнозы RM не являются наблюдаемым откликом на эти рекомендации.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "doctor",
            "download",
            "prepare",
            "base",
            "rm",
            "rl",
            "evaluate",
            "rollouts",
            "report",
            "recommend",
            "all",
        ],
    )
    parser.add_argument("--config", default=str(ROOT / "configs/mac.json"))
    parser.add_argument(
        "--work-dir",
        default=str(ROOT),
        help="Отдельная папка для данных и результатов запуска",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--example", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.device:
        config["device"] = args.device
    validate_config(config)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(work_dir / ".cache/matplotlib"))
    os.environ.setdefault("HF_HOME", str(work_dir / ".cache/huggingface"))
    torch.set_num_threads(4)
    if args.stage == "report":
        from src.results import export_results, plot_results

        export_results()
        plot_results()
        return
    device = device_for(config)
    if args.stage == "doctor":
        print("PyTorch", torch.__version__, "device", device)
        x = torch.ones(8, device=device, requires_grad=True)
        x.square().sum().backward()
        print("Backward OK", x.grad.cpu().tolist())
        return
    if args.stage == "download":
        from src.data import download

        download()
        return
    check_run_config(config)
    if args.stage == "recommend":
        recommend(config, device, args.example, args.top_k)
        return
    save_json("results/config.json", config)
    if args.stage in ("prepare", "all"):
        from src.data import download, prepare

        download()
        prepare(config)
    if args.stage == "prepare":
        return
    datasets, catalog = load_data(), load_catalog()
    if len(catalog["duration"]) - 3 < 20:
        raise ValueError(
            "В подготовленном каталоге меньше 20 объектов; увеличьте выборку."
        )
    if args.stage in ("base", "rm", "rl", "all"):
        save_json(
            "results/execution.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": str(device),
                "stage": args.stage,
            },
        )
    if args.stage in ("base", "all"):
        from src.train import train_base

        train_base(config, datasets, catalog, device)
    if args.stage in ("rm", "all"):
        from src.train import train_base, train_reward

        train_reward(config, datasets, catalog, device)
        train_base(config, datasets, catalog, device, continuation=True)
    if args.stage in ("rl", "all"):
        from src.rl import train_rl

        train_rl(config, datasets, catalog, device)
    if args.stage in ("evaluate", "all"):
        from src.evaluate import evaluate

        evaluate(config, datasets, catalog, device)
    if args.stage == "rollouts":
        from src.evaluate import evaluate_rollouts

        evaluate_rollouts(config, datasets, catalog, device)
    if args.stage == "all":
        from src.results import export_results, plot_results

        export_results()
        plot_results()


if __name__ == "__main__":
    main()
