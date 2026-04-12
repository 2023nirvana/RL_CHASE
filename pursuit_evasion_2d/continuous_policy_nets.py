"""Continuous low-level policy networks for 2D control."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class _GaussianMixin:
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_dist(self, x: torch.Tensor) -> torch.distributions.Normal:
        mu = self._actor_mean(x)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        dist = self.get_dist(x)
        z = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, logp, entropy

    def evaluate_actions(self, x: torch.Tensor, action: torch.Tensor):
        dist = self.get_dist(x)
        z = self._atanh(action)
        logp = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy

    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self._actor_mean(x))


class GaussianActorCriticMLP(nn.Module, _GaussianMixin):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Iterable[int] = (128, 64)):
        super().__init__()
        actor_layers = []
        prev = obs_dim
        for h in hidden:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*actor_layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        critic_layers = []
        prev = obs_dim
        for h in hidden:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.critic = nn.Sequential(*critic_layers, nn.Linear(prev, 1))

    def _actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.mu_head(self.actor_body(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.critic(x).squeeze(-1)


class GaussianActorCriticRNN(nn.Module, _GaussianMixin):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        rnn_type: str = "lstm",
        encoder_hidden: int = 128,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.seq_len = 1
        if self.rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported rnn_type={rnn_type}")

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ReLU(),
        )
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=encoder_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.actor_head = nn.Sequential(nn.Linear(rnn_hidden, rnn_hidden), nn.ReLU(), nn.Linear(rnn_hidden, act_dim))
        self.critic_head = nn.Sequential(nn.Linear(rnn_hidden, rnn_hidden), nn.ReLU(), nn.Linear(rnn_hidden, 1))
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, seq_len, obs_dim = x.shape
        features = self.obs_encoder(x.reshape(batch * seq_len, obs_dim)).reshape(batch, seq_len, -1)
        out, _ = self.rnn(features)
        return out[:, -1, :]

    def _actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        last = self._encode_sequence(x)
        return self.actor_head(last)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        last = self._encode_sequence(x)
        return self.critic_head(last).squeeze(-1)


def build_continuous_actor_critic(
    obs_dim: int,
    act_dim: int,
    network_type: str = "mlp",
    hidden: Iterable[int] = (128, 64),
    seq_len: int = 8,
    rnn_hidden: int = 128,
    rnn_layers: int = 1,
) -> nn.Module:
    hidden = tuple(hidden)
    network_type = network_type.lower()
    if network_type == "mlp":
        return GaussianActorCriticMLP(obs_dim, act_dim, hidden=hidden)
    if network_type in {"lstm", "gru"}:
        encoder_hidden = hidden[0] if len(hidden) > 0 else 128
        model = GaussianActorCriticRNN(
            obs_dim,
            act_dim,
            rnn_type=network_type,
            encoder_hidden=encoder_hidden,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
        )
        model.seq_len = seq_len
        return model
    raise ValueError(f"Unsupported network_type={network_type}")