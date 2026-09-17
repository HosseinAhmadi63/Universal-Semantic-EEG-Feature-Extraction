import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from useeg.data import cache_dataset, dataset_manifest, download_dataset, load_cached_dataset
from useeg.preprocessing import PreprocessingConfig


def _synthetic_config() -> dict[str, object]:
    return {
        "name": "synthetic",
        "moabb_class": "SyntheticEEG",
        "subjects": [1, 2],
        "selection": {"max_trials_per_class": 2},
        "labels": {"left_hand": 0, "right_hand": 1},
    }


def _preprocessing() -> PreprocessingConfig:
    return PreprocessingConfig(
        l_freq=None,
        h_freq=None,
        notch_frequency=None,
        ica_enabled=False,
    )


def test_synthetic_download_does_not_import_moabb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def guarded(name: str, package: str | None = None) -> object:
        if name == "moabb" or name.startswith("moabb."):
            raise AssertionError(name)
        return original(name, package)

    monkeypatch.setattr("useeg.data.importlib.import_module", guarded)
    monkeypatch.setattr("useeg.datasets.importlib.import_module", guarded)
    receipt = download_dataset(_synthetic_config(), tmp_path / "raw")
    assert receipt.exists()
    value = json.loads(receipt.read_text())
    assert value["paper_name"] == "SyntheticEEG"
    assert value["moabb_class"] == "SyntheticEEG"
    assert value["moabb_version"]
    assert value["created_at"] == value["downloaded_at"]
    modified = receipt.stat().st_mtime_ns
    assert download_dataset(_synthetic_config(), tmp_path / "raw") == receipt
    assert receipt.stat().st_mtime_ns == modified
    assert dataset_manifest("synthetic")["name"] == "synthetic"


def test_download_fallback_loads_one_subject_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[int]] = []

    class Dataset:
        subject_list = [1, 2]

        def get_data(self, subjects: list[int]) -> dict[int, object]:
            calls.append(subjects)
            return {}

    monkeypatch.setattr("useeg.data.create_dataset", lambda dataset: Dataset())
    monkeypatch.setattr("useeg.data._set_download_directory", lambda path: None)
    download_dataset(
        {"name": "bnci2014_001", "subjects": [1, 2]},
        tmp_path / "raw",
    )
    assert calls == [[1], [2]]


def test_cache_is_memmapped_keyed_resumable_and_lineage_complete(tmp_path: Path) -> None:
    config = _synthetic_config()
    receipt = download_dataset(config, tmp_path / "raw")
    first = cache_dataset(
        config,
        tmp_path / "raw",
        tmp_path / "processed",
        preprocessing=_preprocessing(),
        seed=42,
        run_key="smoke-key",
    )
    assert first.path == (tmp_path / "processed" / "smoke-key" / "synthetic").resolve()
    assert isinstance(first.X, np.memmap)
    assert isinstance(first.y, np.memmap)
    assert isinstance(first.metadata, np.memmap)
    assert first.X.shape == (8, 128, 8)
    assert first.y.tolist().count(0) == 4
    assert first.y.tolist().count(1) == 4
    assert set(first.metadata.dtype.names or ()) == {
        "subject",
        "session",
        "run",
        "trial",
        "window",
        "event_code",
        "label",
        "window_start",
        "valid_samples",
    }
    assert set(first.metadata["subject"].tolist()) == {"1", "2"}
    assert set(first.metadata["session"].tolist()) == {"0"}
    assert set(first.metadata["run"].tolist()) == {"0"}
    manifest_mtime = (first.path / "manifest.json").stat().st_mtime_ns
    second = cache_dataset(
        config,
        tmp_path / "raw",
        tmp_path / "processed",
        preprocessing=_preprocessing(),
        seed=42,
        run_key="smoke-key",
    )
    assert (second.path / "manifest.json").stat().st_mtime_ns == manifest_mtime
    assert second.manifest["validation"]["trial_counts_match"]
    assert len(second.manifest["shards"]) == 2
    for filename, expected in second.manifest["file_sha256"].items():
        assert hashlib.sha256((second.path / filename).read_bytes()).hexdigest() == expected
    assert second.manifest["source_receipt"] == {
        "path": "synthetic/download.json",
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }
    assert load_cached_dataset("synthetic", tmp_path / "processed").run_key == "smoke-key"
    assert dataset_manifest("synthetic", tmp_path / "processed")["counts"]["trials"] == 8


def test_force_rebuild_and_cache_key_collision(tmp_path: Path) -> None:
    config = _synthetic_config()
    cache_dataset(
        config,
        tmp_path / "raw",
        tmp_path / "processed",
        preprocessing=_preprocessing(),
        run_key="fixed",
    )
    changed = dict(config)
    changed["subjects"] = [1]
    with pytest.raises(ValueError):
        cache_dataset(
            changed,
            tmp_path / "raw",
            tmp_path / "processed",
            preprocessing=_preprocessing(),
            run_key="fixed",
        )
    rebuilt = cache_dataset(
        changed,
        tmp_path / "raw",
        tmp_path / "processed",
        preprocessing=_preprocessing(),
        run_key="fixed",
        force=True,
    )
    assert rebuilt.manifest["selected_subjects"] == [1]
    assert rebuilt.X.shape[0] == 4
