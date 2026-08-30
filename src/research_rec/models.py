from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .config import ModelConfig


class FieldEmbedding(nn.Module):
    def __init__(self, field_dims: Sequence[int], embedding_dim: int):
        super().__init__()
        self.offsets = torch.tensor([0, *field_dims[:-1]]).cumsum(0)
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x + self.offsets.to(x.device))


def make_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float, output_dim: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(current, width), nn.ReLU(), nn.LayerNorm(width), nn.Dropout(dropout)])
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class MatrixFactorization(nn.Module):
    """Biased user-item MF; expects user and item as the first two fields."""

    def __init__(self, field_dims: Sequence[int], embedding_dim: int, **_: object):
        super().__init__()
        if len(field_dims) < 2:
            raise ValueError("MatrixFactorization requires user and item fields")
        self.user = nn.Embedding(field_dims[0], embedding_dim)
        self.item = nn.Embedding(field_dims[1], embedding_dim)
        self.user_bias = nn.Embedding(field_dims[0], 1)
        self.item_bias = nn.Embedding(field_dims[1], 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        interaction = (self.user(x[:, 0]) * self.item(x[:, 1])).sum(dim=1)
        return interaction + self.user_bias(x[:, 0]).squeeze(1) + self.item_bias(x[:, 1]).squeeze(1) + self.global_bias


class DeepFM(nn.Module):
    def __init__(self, field_dims: Sequence[int], embedding_dim: int, hidden_dims: Sequence[int], dropout: float, **_: object):
        super().__init__()
        self.offsets = torch.tensor([0, *field_dims[:-1]]).cumsum(0)
        self.linear = nn.Embedding(sum(field_dims), 1)
        self.embedding = FieldEmbedding(field_dims, embedding_dim)
        self.deep = make_mlp(len(field_dims) * embedding_dim, hidden_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        offsets = self.offsets.to(x.device)
        linear = self.linear(x + offsets).sum(dim=1).squeeze(1)
        summed = embedded.sum(dim=1)
        fm = 0.5 * (summed.square() - embedded.square().sum(dim=1)).sum(dim=1)
        deep = self.deep(embedded.flatten(start_dim=1)).squeeze(1)
        return linear + fm + deep


class CrossLayer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * torch.sum(x * self.weight, dim=1, keepdim=True) + self.bias + x


class DeepCrossNetwork(nn.Module):
    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        cross_layers: int,
        **_: object,
    ):
        super().__init__()
        self.embedding = FieldEmbedding(field_dims, embedding_dim)
        dimension = len(field_dims) * embedding_dim
        self.cross = nn.ModuleList(CrossLayer(dimension) for _ in range(cross_layers))
        deep_width = hidden_dims[-1] if hidden_dims else dimension
        self.deep = make_mlp(dimension, hidden_dims[:-1], dropout, output_dim=deep_width)
        self.output = nn.Linear(dimension + deep_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding(x).flatten(start_dim=1)
        crossed = x0
        for layer in self.cross:
            crossed = layer(x0, crossed)
        return self.output(torch.cat([crossed, self.deep(x0)], dim=1)).squeeze(1)


class MultiTaskDeepCrossNetwork(nn.Module):
    """Shared DCN representation with one click head and auxiliary feedback heads."""

    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        cross_layers: int,
        task_count: int,
        **_: object,
    ):
        super().__init__()
        if task_count < 2:
            raise ValueError("MultiTaskDeepCrossNetwork requires at least two tasks")
        self.embedding = FieldEmbedding(field_dims, embedding_dim)
        dimension = len(field_dims) * embedding_dim
        self.cross = nn.ModuleList(CrossLayer(dimension) for _ in range(cross_layers))
        deep_width = hidden_dims[-1] if hidden_dims else dimension
        self.deep = make_mlp(dimension, hidden_dims[:-1], dropout, output_dim=deep_width)
        shared_width = dimension + deep_width
        self.heads = nn.ModuleList(nn.Linear(shared_width, 1) for _ in range(task_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding(x).flatten(start_dim=1)
        crossed = x0
        for layer in self.cross:
            crossed = layer(x0, crossed)
        shared = torch.cat([crossed, self.deep(x0)], dim=1)
        return torch.cat([head(shared) for head in self.heads], dim=1)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mf": MatrixFactorization,
    "deepfm": DeepFM,
    "dcn": DeepCrossNetwork,
    "multitask_dcn": MultiTaskDeepCrossNetwork,
}


def build_model(config: ModelConfig, field_dims: Sequence[int], task_count: int = 1) -> nn.Module:
    model_class = MODEL_REGISTRY[config.name.lower()]
    return model_class(
        field_dims=field_dims,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        cross_layers=config.cross_layers,
        task_count=task_count,
    )
