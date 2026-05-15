#!/usr/bin/env python3
"""Reproduce why seeding only after model construction is insufficient."""

# ruff: noqa: E402, I001

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for package_dir in (
    REPO_ROOT / "neuralbench-repo",
    REPO_ROOT / "neuraltrain-repo",
    REPO_ROOT / "neuralset-repo",
    REPO_ROOT / "neuralfetch-repo",
):
    sys.path.insert(0, str(package_dir))

import lightning.pytorch as pl
import torch
from torch import nn

from neuralbench.model_factory import build_brain_model
from neuralbench.modules import ChannelProjection, DownstreamWrapper


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.LazyLinear(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(start_dim=1))


class TinyConfig:
    def build(self, **kwargs) -> nn.Module:
        del kwargs
        return TinyEncoder()


class FakeBatch:
    def __init__(self) -> None:
        self.data = {
            "neuro": torch.randn(4, 8, 16),
            "target": torch.randn(4, 3),
        }


class FakeDataset:
    extractors: dict[str, object] = {}


class FakeLoader:
    def __init__(self) -> None:
        self.dataset = FakeDataset()
        self._batch = FakeBatch()

    def __iter__(self):
        yield self._batch


def parameter_digest(model: nn.Module) -> str:
    params = [param.detach().cpu().reshape(-1) for param in model.parameters()]
    flat = torch.cat(params)
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]


def build_digest(*, pre_draws: int, seed: int, seed_before_build: bool) -> str:
    torch.manual_seed(2024)
    _ = torch.rand(pre_draws)
    if seed_before_build:
        pl.seed_everything(seed)

    model, _, _ = build_brain_model(
        brain_model_config=TinyConfig(),
        downstream_model_wrapper=DownstreamWrapper(
            channel_adapter_config=ChannelProjection(n_target_channels=5),
            aggregation="flatten",
            probe_config="linear",
        ),
        pretrained_weights_fname=None,
        train_loader=FakeLoader(),
        val_loader=None,
        wandb_logger=None,
    )

    pl.seed_everything(seed)
    return parameter_digest(model)


def main() -> None:
    seed = 7
    late_a = build_digest(pre_draws=0, seed=seed, seed_before_build=False)
    late_b = build_digest(pre_draws=64, seed=seed, seed_before_build=False)
    early_a = build_digest(pre_draws=0, seed=seed, seed_before_build=True)
    early_b = build_digest(pre_draws=64, seed=seed, seed_before_build=True)

    print("Late seeding digests:")
    print(f"  pre_draws=0  -> {late_a}")
    print(f"  pre_draws=64 -> {late_b}")
    print("Early seeding digests:")
    print(f"  pre_draws=0  -> {early_a}")
    print(f"  pre_draws=64 -> {early_b}")

    if late_a == late_b:
        raise SystemExit("Expected late seeding to vary with prior RNG consumption.")
    if early_a != early_b:
        raise SystemExit("Expected early seeding to stabilize model construction.")

    print("Reproduction succeeded: only early seeding is invariant to prior RNG usage.")


if __name__ == "__main__":
    main()
