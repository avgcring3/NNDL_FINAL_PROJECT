from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.size(1)]


def zero_init_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class DemandTransformer(nn.Module):
    def __init__(self, config: ModelConfig, horizon_hours: int):
        super().__init__()
        self.max_zones = config.max_zones
        self.horizon_hours = horizon_hours
        self.value_projection = nn.Linear(1, config.d_model)
        self.zone_embedding = nn.Embedding(config.max_zones + 1, config.d_model)
        self.position = PositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, horizon_hours))
        zero_init_last_linear(self.head)

    def forward(self, demand_window: torch.Tensor, zone_ids: torch.Tensor) -> torch.Tensor:
        values = demand_window.unsqueeze(-1)
        zone_ids = zone_ids.clamp(min=0, max=self.max_zones)
        hidden = self.value_projection(values)
        hidden = hidden + self.zone_embedding(zone_ids).unsqueeze(1)
        hidden = self.position(hidden)
        encoded = self.encoder(hidden)
        residual = self.head(encoded[:, -1])
        base = demand_window[:, -1:].repeat(1, self.horizon_hours)
        return base + residual


class DemandMLP(nn.Module):
    def __init__(self, config: ModelConfig, lookback_hours: int, horizon_hours: int):
        super().__init__()
        self.max_zones = config.max_zones
        self.horizon_hours = horizon_hours
        self.zone_embedding = nn.Embedding(config.max_zones + 1, 8)
        self.network = nn.Sequential(
            nn.Linear(lookback_hours + 8, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, horizon_hours),
        )
        zero_init_last_linear(self.network)

    def forward(self, demand_window: torch.Tensor, zone_ids: torch.Tensor) -> torch.Tensor:
        zone_ids = zone_ids.clamp(min=0, max=self.max_zones)
        features = torch.cat([demand_window, self.zone_embedding(zone_ids)], dim=1)
        residual = self.network(features)
        base = demand_window[:, -1:].repeat(1, self.horizon_hours)
        return base + residual


def make_model(name: str, config: ModelConfig, lookback_hours: int, horizon_hours: int) -> nn.Module:
    if name == "transformer":
        return DemandTransformer(config, horizon_hours)
    if name == "mlp":
        return DemandMLP(config, lookback_hours, horizon_hours)
    raise ValueError(f"Unknown model: {name}")


def weighted_mae_loss(prediction: torch.Tensor, target: torch.Tensor, high_demand_weight: float) -> torch.Tensor:
    threshold = torch.quantile(target.detach().flatten(), 0.75)
    weights = torch.where(target >= threshold, high_demand_weight, 1.0)
    return torch.mean(torch.abs(prediction - target) * weights)
