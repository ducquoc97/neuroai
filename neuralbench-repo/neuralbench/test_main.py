# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import lightning.pytorch as pl
import torch
from torch import nn

from .main import Experiment


class _DummyLoss:
    def build(self, **kwargs) -> nn.Module:
        return nn.MSELoss()


class _DummyBrainModule:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_prepare_pl_module_seeds_before_and_after_model_build(monkeypatch) -> None:
    """Model construction should ignore prior RNG usage and reset training RNG."""
    seed = 123
    build_draws: list[torch.Tensor] = []
    post_draws: list[torch.Tensor] = []

    def fake_build_brain_model(**kwargs):
        del kwargs
        build_draws.append(torch.rand(4))
        _ = torch.rand(11)
        return nn.Identity(), 0, 0

    monkeypatch.setattr("neuralbench.main.build_brain_model", fake_build_brain_model)
    monkeypatch.setattr("neuralbench.main.BrainModule", _DummyBrainModule)

    experiment = Experiment.model_construct(
        brain_model_config=object(),
        downstream_model_wrapper=None,
        pretrained_weights_fname=None,
        _wandb_logger=None,
        target_scaler=None,
        compute_class_weights=False,
        loss=_DummyLoss(),
        lightning_optimizer_config=object(),
        metrics=[],
        test_full_metrics=[],
        test_full_retrieval_metrics=[],
        seed=seed,
        _brain_module=None,
    )

    for pre_draws in (3, 17):
        torch.manual_seed(999)
        _ = torch.rand(pre_draws)
        experiment.prepare_pl_module(train_loader=object())
        post_draws.append(torch.rand(4))

    pl.seed_everything(seed)
    expected = torch.rand(4)

    assert torch.allclose(build_draws[0], expected)
    assert torch.allclose(build_draws[1], expected)
    assert torch.allclose(post_draws[0], expected)
    assert torch.allclose(post_draws[1], expected)

