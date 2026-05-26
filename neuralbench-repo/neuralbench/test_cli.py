# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pytest
from exca import ConfDict

import neuralset as ns

from . import transforms as _transforms  # noqa: F401  — registers Step subclasses
from .experiment_config import prepare_task_configs
from .main import Data
from .registry import ALL_DATASETS, ALL_TASKS, DEFAULTS_DIR, load_yaml_config


def test_build_all_datasets() -> None:
    """Import-time _build_all_datasets parses every config with 'source' key."""
    assert len(ALL_TASKS) > 30
    total_studies = sum(
        len(studies) for tasks in ALL_DATASETS.values() for studies in tasks.values()
    )
    assert total_studies > 50


@pytest.mark.parametrize("dataset", [None, "schalk2004bci"])
def test_prepare_task_configs(dataset: str | None) -> None:
    """Merged config produces a valid Data with a Chain study.

    schalk2004bci uses =replace= which wipes the study dict; _restore_default_source
    must re-inject path and infra from the defaults.
    """
    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    grid = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))
    datasets: list[str | None] | None = [dataset] if dataset is not None else None
    configs = prepare_task_configs(
        config,
        grid,
        "eeg",
        "motor_imagery",
        use_task_grid=False,
        debug=False,
        force=False,
        prepare=False,
        download=False,
        models=[None],
        datasets=datasets,
    )
    data = Data(**configs[0]["data"])
    assert isinstance(data.study, ns.Chain)
    steps: tp.Any = data.study.steps
    assert isinstance(steps, dict)
    source: tp.Any = steps["source"]
    assert source.path is not None
    assert source.infra is not None


def test_run_benchmark_cli_help_smoke(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``neuralbench --help`` exits 0 and lists devices/tasks via the epilog.

    Smoke-tests that the CLI parser builds, the registry loads, and
    ``_format_datasets_epilog`` renders without crashing.
    """
    from .cli import run_benchmark_cli

    monkeypatch.setattr("sys.argv", ["neuralbench", "--help"])
    with pytest.raises(SystemExit) as exc:
        run_benchmark_cli()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "available datasets per task:" in out
    assert "eeg" in out

def test_run_benchmark_cli_local_flag_forwards_to_run_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from . import cli

    called: dict[str, tp.Any] = {}

    def fake_run_benchmark(**kwargs: tp.Any) -> list[dict[str, tp.Any]]:
        called.update(kwargs)
        return []

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        "sys.argv",
        ["neuralbench", "eeg", "audiovisual_stimulus", "--local"],
    )

    cli.run_benchmark_cli()

    assert called["local"] is True
    assert called["debug"] is False


def test_run_benchmark_local_preserves_official_config_and_runs_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from . import cli

    prepare_debug_flags: list[bool] = []
    aggregator_calls: dict[str, tp.Any] = {}

    def fake_load_yaml_config(path: tp.Any) -> dict[str, tp.Any]:
        if str(path).endswith("grid.yaml"):
            return {"seed": [1, 2, 3]}
        return {
            "infra": {"cluster": "slurm"},
            "data": {
                "neuro": {"infra": {"cluster": "slurm"}},
                "target": {"infra": {"cluster": "slurm"}},
            },
        }

    def fake_prepare_task_configs(
        config: ConfDict,
        grid: ConfDict,
        device: str,
        task_name: str,
        use_task_grid: bool,
        debug: bool,
        force: bool,
        prepare: bool,
        download: bool,
        models: list[str | None],
        datasets: list[str | None] | None,
        quiet: bool = False,
        retry: bool = False,
    ) -> list[ConfDict]:
        prepare_debug_flags.append(debug)
        return [config]

    class DummyAggregator:
        def __init__(self, experiments: list[ConfDict], debug: bool) -> None:
            aggregator_calls["experiments"] = experiments
            aggregator_calls["debug"] = debug

        def prepare(self) -> None:
            aggregator_calls["prepared"] = True

        def run(self, cached_only: bool = False) -> list[dict[str, tp.Any]]:
            aggregator_calls["cached_only"] = cached_only
            return []

    monkeypatch.setattr(cli, "load_yaml_config", fake_load_yaml_config)
    monkeypatch.setattr(cli, "_validate_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_warn_slurm_partition", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_resolve_tasks", lambda device, task: ["task_a"])
    monkeypatch.setattr(
        cli, "_expand_models", lambda model, device, task_name: [None]
    )
    monkeypatch.setattr(
        cli, "_resolve_datasets", lambda device, task_name, dataset: None
    )
    monkeypatch.setattr(cli, "prepare_task_configs", fake_prepare_task_configs)
    monkeypatch.setattr(
        "neuralbench.config_manager._ensure_initialized", lambda: None
    )
    monkeypatch.setattr("neuralbench.main.BenchmarkAggregator", DummyAggregator)

    cli.run_benchmark("eeg", "audiovisual_stimulus", local=True)

    assert prepare_debug_flags == [False]
    assert aggregator_calls["debug"] is True
    assert aggregator_calls["prepared"] is True
    experiment = aggregator_calls["experiments"][0]
    assert experiment["infra.cluster"] is None
    assert experiment["data.neuro.infra.cluster"] is None
    assert experiment["data.target.infra.cluster"] is None
