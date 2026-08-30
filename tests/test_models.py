import pytest
import torch

from research_rec.config import ModelConfig
from research_rec.models import build_model


@pytest.mark.parametrize("name", ["mf", "deepfm", "dcn"])
def test_models_produce_one_logit_per_interaction(name):
    config = ModelConfig(name=name, embedding_dim=4, hidden_dims=[8, 4], dropout=0.0, cross_layers=2)
    model = build_model(config, [5, 7, 3])
    features = torch.tensor([[1, 2, 0], [2, 4, 1], [3, 1, 2]])
    output = model(features)
    assert output.shape == (3,)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_multitask_dcn_produces_one_logit_per_task():
    config = ModelConfig(name="multitask_dcn", embedding_dim=4, hidden_dims=[8, 4], dropout=0.0, cross_layers=2)
    model = build_model(config, [5, 7, 3], task_count=4)
    features = torch.tensor([[1, 2, 0], [2, 4, 1], [3, 1, 2]])
    output = model(features)
    assert output.shape == (3, 4)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
