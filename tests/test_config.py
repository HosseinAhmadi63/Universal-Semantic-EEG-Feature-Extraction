from pathlib import Path

from useeg.config import (
    PAPER_DATASETS,
    config_hash,
    dataset_map,
    load_config,
    selected_dataset_names,
    with_selection,
)

ROOT = Path(__file__).resolve().parents[1]


def test_paper_config_contains_all_datasets() -> None:
    config = load_config(ROOT / "configs" / "paper.yaml")
    assert tuple(selected_dataset_names(config)) == PAPER_DATASETS
    assert set(dataset_map(config)) == set(PAPER_DATASETS)


def test_runtime_selection_changes_run_identity() -> None:
    config = load_config(ROOT / "configs" / "paper.yaml")
    selected = with_selection(config, ["bnci2014_001"], [1, 2])
    assert config_hash(config) != config_hash(selected)


def test_smoke_config_is_synthetic() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    specification = next(iter(dataset_map(config).values()))
    assert specification["moabb_class"] == "SyntheticEEG"
