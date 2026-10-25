"""GPT-2, модель отклика и обновление контекста."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model


class GPTRec(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        architecture = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config["context_length"],
            n_embd=config["hidden_size"],
            n_layer=config["layers"],
            n_head=config["heads"],
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            bos_token_id=2,
            eos_token_id=None,
            pad_token_id=0,
            use_cache=False,
        )
        architecture._attn_implementation = "eager"
        self.backbone = GPT2Model(architecture)

    def encode(self, contexts):
        mask = contexts.ne(0)
        hidden = self.backbone(
            input_ids=contexts, attention_mask=mask, use_cache=False
        ).last_hidden_state
        last_position = mask.sum(dim=1) - 1
        rows = torch.arange(len(contexts), device=contexts.device)
        return hidden[rows, last_position]

    def forward(self, contexts):
        logits = F.linear(self.encode(contexts), self.backbone.wte.weight)
        # Конечная маска служебных токенов предотвращает NaN в KL.
        special_tokens = torch.full_like(logits[:, :3], -1e9)
        return torch.cat((special_tokens, logits[:, 3:]), dim=1)


class RewardModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, duration_scale, no_history=False):
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, hidden_size)
        self.no_history = no_history
        self.duration_scale = float(duration_scale)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 3 + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, history, items, duration, feedback_summary):
        if self.no_history:
            history = torch.zeros_like(history)
            feedback_summary = torch.zeros_like(feedback_summary)
        candidate = self.item_embedding(items)
        duration_feature = torch.log1p(duration.float()).unsqueeze(-1)
        duration_feature = duration_feature / self.duration_scale
        features = torch.cat(
            [history, candidate, history * candidate, duration_feature, feedback_summary],
            dim=-1,
        )
        prediction = self.net(features).squeeze(-1)
        return prediction.sigmoid()


def summarize_feedback(context, feedback):
    """Среднее, последний отклик и доля прослушиваний от 50%."""
    mask = context.ne(0)
    length = mask.sum(1).clamp(min=1)
    mean = (feedback * mask).sum(1) / length
    last = feedback.gather(1, (length - 1)[:, None]).squeeze(1)
    positive = ((feedback >= 0.5) & mask).sum(1) / length
    return torch.stack((mean, last, positive), dim=1)


def append_feedback(context, feedback, prediction):
    """Добавить прогноз отклика в историю."""
    lengths = context.ne(0).sum(1)
    full = lengths == context.shape[1]
    result = feedback.clone()
    result[full, :-1] = feedback[full, 1:]
    position = torch.clamp(lengths, max=context.shape[1] - 1)
    result[torch.arange(len(context), device=context.device), position] = prediction
    return result


def append_item(context, action):
    """Добавить трек, сдвинув заполненное окно."""
    lengths = context.ne(0).sum(1)
    full = lengths == context.shape[1]
    result = context.clone()
    result[full, :-1] = context[full, 1:]
    position = torch.clamp(lengths, max=context.shape[1] - 1)
    result[torch.arange(len(context), device=context.device), position] = action
    return result


def load_policy(path, config, vocab_size, device):
    model = GPTRec(vocab_size, config).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def load_reward(path, catalog, config, device, no_history=False):
    model = RewardModel(
        len(catalog["duration"]),
        config["hidden_size"],
        catalog["duration_scale"],
        no_history,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()
