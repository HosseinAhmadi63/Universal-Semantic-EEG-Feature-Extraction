from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PAPER_DATASETS = (
    "bnci2014_001",
    "bnci2014_004",
    "lee2019_ssvep",
    "nakanishi2015",
    "bi2012",
    "bi2013a",
    "bi2014b",
    "bi2015a",
    "bi2015b",
    "bnci2014_008",
    "bnci2014_009",
    "sosulski2019",
)


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {source}")
    config = deepcopy(value)
    config["_config_path"] = str(source)
    config["_project_root"] = str(source.parent.parent)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "project",
        "seed",
        "device",
        "paths",
        "data",
        "datasets",
        "preprocessing",
        "autoencoder",
        "feature_extraction",
        "classifier",
        "evaluation",
        "analysis",
        "figures",
        "verification",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    datasets = config["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    names = [str(item.get("name", "")) for item in datasets]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("dataset names must be non-empty and unique")
    ordering = list(config["data"]["ordering"])
    if set(ordering) != set(names):
        raise ValueError("data.ordering must contain every configured dataset exactly once")
    for item in datasets:
        labels = item.get("labels")
        if not isinstance(labels, dict) or len(labels) < 2:
            raise ValueError(f"{item['name']}: labels must define at least two classes")
        encoded = sorted(int(value) for value in labels.values())
        if encoded != list(range(len(encoded))):
            raise ValueError(f"{item['name']}: labels must be contiguous from zero")
        attributes = item.get("paper_attributes", {})
        if int(attributes.get("channels", 0)) < 1:
            raise ValueError(f"{item['name']}: paper channel count must be positive")
        if float(attributes.get("trial_duration_seconds", 0.0)) <= 0.0:
            raise ValueError(f"{item['name']}: trial duration must be positive")
    preprocessing = config["preprocessing"]
    window = preprocessing["window"]
    if int(preprocessing["sfreq"]) != 128:
        raise ValueError("preprocessing.sfreq must be 128 for the paper protocol")
    if int(window["size_samples"]) != 128 or int(window["hop_samples"]) != 64:
        raise ValueError("paper windows require 128 samples and a 64-sample default hop")
    autoencoder = config["autoencoder"]
    fixed_autoencoder = {
        "d_model": 128,
        "nhead": 8,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 512,
        "loss": "mse",
    }
    for key, expected in fixed_autoencoder.items():
        if autoencoder[key] != expected:
            raise ValueError(f"autoencoder.{key} must be {expected!r}")
    if float(autoencoder["val_fraction"]) <= 0.0 or float(autoencoder["val_fraction"]) >= 1.0:
        raise ValueError("autoencoder.val_fraction must be between zero and one")
    classifier = config["classifier"]
    if list(classifier["filters"]) != [64, 128, 256, 512]:
        raise ValueError("classifier filters must be 64, 128, 256, and 512")
    if float(classifier["dropout"]) != 0.5:
        raise ValueError("classifier.dropout must be 0.5")
    if config["evaluation"]["mi"]["strategy"] != "leave_one_subject_out":
        raise ValueError("MI evaluation must use leave-one-subject-out")
    profile = str(config["project"].get("profile", "paper"))
    if profile == "paper":
        if tuple(ordering) != PAPER_DATASETS:
            raise ValueError("paper configuration must contain the 12 datasets in paper order")
        if int(config["evaluation"]["ssvep"]["folds"]) != 8:
            raise ValueError("paper SSVEP evaluation must use eight folds")
        if int(config["evaluation"]["erp"]["folds"]) != 8:
            raise ValueError("paper ERP evaluation must use eight folds")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_hash(config: dict[str, Any]) -> str:
    length = int(config.get("reproducibility", {}).get("config_hash_length", 12))
    payload = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def resolve_path(config: dict[str, Any], key: str) -> Path:
    configured = Path(config["paths"][key]).expanduser()
    if configured.is_absolute():
        return configured
    return (Path(config["_project_root"]) / configured).resolve()


def dataset_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in config["datasets"]}


def selected_dataset_names(
    config: dict[str, Any], requested: list[str] | tuple[str, ...] | None = None
) -> list[str]:
    configured = dataset_map(config)
    ordering = [name for name in config["data"]["ordering"] if configured[name]["enabled"]]
    if not requested or list(requested) == ["all"]:
        return ordering
    unknown = set(requested) - set(configured)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return [name for name in ordering if name in set(requested)]


def with_selection(
    config: dict[str, Any],
    datasets: list[str] | None = None,
    subjects: list[int] | None = None,
) -> dict[str, Any]:
    result = deepcopy(config)
    runtime = dict(result.get("runtime_selection", {}))
    if datasets and datasets != ["all"]:
        runtime["datasets"] = selected_dataset_names(result, datasets)
    if subjects:
        selected = sorted(set(int(subject) for subject in subjects))
        if any(subject < 1 for subject in selected):
            raise ValueError("subject identifiers must be positive")
        runtime["subjects"] = selected
    if runtime:
        result["runtime_selection"] = runtime
    return result


def processed_directory(config: dict[str, Any], dataset_name: str) -> Path:
    return resolve_path(config, "data_processed") / config_hash(config) / dataset_name


def run_directory(config: dict[str, Any], dataset_name: str | None = None) -> Path:
    root = resolve_path(config, "runs") / config_hash(config)
    return root if dataset_name is None else root / dataset_name


def generated_directory(config: dict[str, Any]) -> Path:
    return resolve_path(config, "publication_generated") / config_hash(config)
