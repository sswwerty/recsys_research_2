import itertools
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import prepare, strict_context
from src.models import GPTRec, append_feedback, append_item, summarize_feedback
from src.rl import leave_one_out
from src.utils import paired_bootstrap, rank_metrics, user_macro

torch.set_num_threads(2)


def test_context_does_not_contain_equal_or_future_timestamps():
    context, end = strict_context(
        np.array([3, 4, 5, 6]), np.array([10, 20, 20, 30]), 2, 4
    )
    assert end == 1
    assert context.tolist() == [3, 0, 0, 0]


def test_append_handles_padding_and_full_context():
    result = append_item(torch.tensor([[3, 4, 0], [3, 4, 5]]), torch.tensor([6, 7]))
    assert result.tolist() == [[3, 4, 6], [4, 5, 7]]


def test_feedback_uses_same_window_as_items():
    context = torch.tensor([[3, 4, 0], [3, 4, 5]])
    feedback = torch.tensor([[0.2, 0.8, 0.0], [0.2, 0.8, 0.9]])
    summary = summarize_feedback(context, feedback)
    torch.testing.assert_close(summary[0], torch.tensor([0.5, 0.8, 0.5]))
    new = append_feedback(context, feedback, torch.tensor([0.6, 0.1]))
    torch.testing.assert_close(new, torch.tensor([[0.2, 0.8, 0.6], [0.8, 0.9, 0.1]]))


def test_gpt_causal_mask_padding_and_backward():
    torch.manual_seed(1)
    config = dict(context_length=8, hidden_size=16, layers=1, heads=2)
    model = GPTRec(12, config).eval()
    a = torch.tensor([[3, 4, 5]])
    b = torch.tensor([[3, 4, 9]])
    with torch.no_grad():
        hidden_a = model.backbone(a).last_hidden_state
        hidden_b = model.backbone(b).last_hidden_state
        torch.testing.assert_close(hidden_a[:, :2], hidden_b[:, :2])
        torch.testing.assert_close(
            model(a), model(torch.tensor([[3, 4, 5, 0, 0]])), atol=1e-6, rtol=1e-5
        )
    logits = model(a)
    assert (logits[:, :3] < -1e8).all()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([7]))
    loss.backward()
    assert torch.isfinite(model.backbone.wte.weight.grad).all()
    assert model.backbone.wte.weight.grad.abs().sum() > 0


def test_metrics_credit_first_hit_only():
    recs = np.array([[4, 5, 5], [5, 4, 6], [3, 4, 5]])
    targets = np.array([5, 5, 9])
    recall, ndcg = rank_metrics(recs, targets, 3)
    np.testing.assert_allclose(recall, [1, 1, 0])
    np.testing.assert_allclose(ndcg, [1 / math.log2(3), 1, 0])
    assert user_macro(np.array([1.0, 1.0, 0.0]), np.array([1, 1, 2])) == 0.5
    ci = paired_bootstrap(ndcg, ndcg, np.array([1, 2, 3]), 50)
    assert ci == dict(mean=0.0, low=0.0, high=0.0)


def test_leave_one_out_excludes_own_reward():
    rewards = torch.tensor([1.0, 2.0, 6.0, 10.0, 20.0, 30.0])
    advantage = leave_one_out(rewards, 3)
    torch.testing.assert_close(
        advantage, torch.tensor([-3.0, -1.5, 4.5, -15.0, 0.0, 15.0])
    )


def test_multistep_reinforce_kl_gradient_includes_state_distribution():
    # Двухшаговый мир: второй набор logits зависит от первого действия.
    logits = torch.tensor(
        [[0.3, -0.1], [0.2, 0.6], [-0.7, 0.5]], dtype=torch.float64, requires_grad=True
    )
    ref = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.3, 0.7]], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    objective = torch.tensor(0.0, dtype=torch.float64)
    surrogate = torch.tensor(0.0, dtype=torch.float64)
    for first, second in itertools.product(range(2), repeat=2):
        lp = logp[0, first] + logp[1 + first, second]
        reference_lp = ref[0, first].log() + ref[1 + first, second].log()
        reward = float(first != second) + 0.3 * second
        regularized = reward - 0.2 * (lp - reference_lp)
        probability = lp.exp()
        objective = objective + probability * regularized
        surrogate = surrogate + probability.detach() * regularized.detach() * lp
    exact = torch.autograd.grad(objective, logits, retain_graph=True)[0]
    score_function = torch.autograd.grad(surrogate, logits)[0]
    torch.testing.assert_close(exact, score_function, atol=1e-10, rtol=1e-10)


def test_data_pipeline_uses_train_vocabulary_and_preserves_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/raw").mkdir(parents=True)
    rows = []
    for uid in [1, 2]:
        for t in range(1, 101):
            rows.append(
                dict(
                    uid=uid,
                    timestamp=t,
                    item_id=(999 if t >= 90 else 10 + t % 2),
                    is_organic=0,
                    played_ratio_pct=120 if t % 7 == 0 else 40,
                    track_length_seconds=200,
                )
            )
    pd.DataFrame(rows).to_parquet("data/raw/listens.parquet")
    config = dict(
        seed=42,
        users=2,
        catalog_size=2,
        min_history=2,
        min_train_events=5,
        context_length=4,
        train_examples=50,
        eval_per_user=20,
        train_fraction=0.8,
        val_fraction=0.9,
    )
    prepare(config)
    catalog = np.load("data/processed/catalog.npz")
    assert 999 not in catalog["item_ids"]
    for name in ("train", "val", "test"):
        split = np.load(f"data/processed/{name}.npz")
        assert (split["history_end"] < split["time"]).all()
        assert (split["reward"] <= 1).all()
        last = (split["ctx"] != 0).sum(1) - 1
        expected = np.where(split["history_end"] % 7 == 0, 1.0, 0.4)
        np.testing.assert_allclose(
            split["feedback"][np.arange(len(last)), last], expected
        )
    test = np.load("data/processed/test.npz")
    assert (test["target"] == 1).all()
    assert (test["ctx"] == 1).any()


def test_config_guard_rejects_different_training_without_overwriting(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path

    from run import check_run_config, validate_config

    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/mac.json").read_text()
    )
    validate_config(config)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    snapshot = json.dumps(config)
    (tmp_path / "results/config.json").write_text(snapshot)
    check_run_config({**config, "device": "cpu"})
    with pytest.raises(ValueError, match="--work-dir"):
        check_run_config({**config, "seed": 123})
    assert (tmp_path / "results/config.json").read_text() == snapshot
    with pytest.raises(ValueError, match="Leave-one-out"):
        validate_config({**config, "rl_group": 1})


def test_continuation_completes_budget_across_short_epochs(tmp_path, monkeypatch):
    import json

    from src.train import train_base

    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    config = dict(
        seed=42,
        context_length=4,
        hidden_size=8,
        layers=1,
        heads=2,
        rl_lr=0.001,
        rl_batch=1,
        rl_group=2,
        rl_steps=5,
        batch_size=2,
        base_epochs=1,
        base_lr=0.001,
    )
    torch.save(GPTRec(23, config).state_dict(), "results/base.pt")
    data = {
        "ctx": np.array([[3, 4, 0, 0], [4, 5, 0, 0], [5, 6, 0, 0]]),
        "target": np.array([5, 6, 7]),
        "uid": np.array([1, 2, 3]),
    }
    train_base(
        config,
        {"train": data, "val": data},
        {"duration": np.ones(23)},
        torch.device("cpu"),
        continuation=True,
    )
    history = json.loads((tmp_path / "results/continued_training.json").read_text())
    assert history["steps"] == 5
    assert len(history["history"]) == 3


def test_frozen_reference_and_reward_leave_actor_gradients():
    from src.models import RewardModel

    config = dict(context_length=4, hidden_size=8, layers=1, heads=2)
    actor = GPTRec(23, config).eval()
    reference = GPTRec(23, config).eval().requires_grad_(False)
    reference.load_state_dict(actor.state_dict())
    reward_model = RewardModel(23, 8, 5.0).eval().requires_grad_(False)
    context = torch.tensor([[3, 4, 0, 0], [3, 4, 0, 0]])
    actions = torch.tensor([5, 7])
    logp = actor(context).log_softmax(-1)
    with torch.no_grad():
        rewards = reward_model(
            reference.encode(context),
            actions,
            torch.tensor([180.0, 220.0]),
            torch.full((2, 3), 0.5),
        )
    advantage = leave_one_out(rewards, 2)
    loss = -(advantage * logp.gather(1, actions[:, None]).squeeze(1)).mean()
    loss.backward()
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is None for p in reward_model.parameters())
    assert actor.backbone.wte.weight.grad.abs().sum() > 0
