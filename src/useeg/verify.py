from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import (
    PAPER_DATASETS,
    config_hash,
    dataset_map,
    public_config,
    resolve_path,
    validate_config,
)
from .datasets import get_dataset_spec
from .metrics import binary_classification_metrics, multiclass_classification_metrics
from .models import BinaryFocalLoss, build_autoencoder, build_classifier
from .splits import classification_folds, random_train_validation_split
from .utils import environment_record, sha256_file, write_json


def _study_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    rows = frame[frame["method"].str.lower() == "this study"]
    if len(rows) != 1:
        raise ValueError(f"{path.name}: expected one This study row")
    return rows.iloc[0]


def _verify_hashes(source: Path) -> list[dict[str, Any]]:
    manifest = pd.read_csv(source / "source_manifest.csv")
    rows: list[dict[str, Any]] = []
    for record in manifest.itertuples(index=False):
        path = source / record.file
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != record.sha256:
            raise ValueError(f"Hash mismatch for {record.file}: {actual}")
        rows.append({"file": record.file, "sha256": actual, "verified": True})
    return rows


def _verify_reported_values(source: Path, tolerance: float) -> dict[str, Any]:
    table_5 = _study_row(source / "table_5_bciciv_2a_accuracy.csv")
    table_6 = _study_row(source / "table_6_bciciv_2b_accuracy.csv")
    table_7 = _study_row(source / "table_7_ssvep_accuracy.csv")
    table_8 = _study_row(source / "table_8_p300_auc.csv")
    expected = {
        "bciciv_2a_accuracy": 83.50,
        "bciciv_2b_accuracy": 84.84,
        "lee2019_ssvep_accuracy": 98.41,
        "nakanishi2015_accuracy": 99.66,
        "erp_average_auc": 91.80,
    }
    actual = {
        "bciciv_2a_accuracy": float(table_5["average"]),
        "bciciv_2b_accuracy": float(table_6["average"]),
        "lee2019_ssvep_accuracy": float(table_7["lee2019_ssvep"]),
        "nakanishi2015_accuracy": float(table_7["nakanishi2015"]),
        "erp_average_auc": float(table_8["average"]),
    }
    for key, value in expected.items():
        if not np.isclose(actual[key], value, atol=tolerance, rtol=0.0):
            raise ValueError(f"Published value mismatch for {key}: {actual[key]}")
    subject_columns = [f"s{index:02d}" for index in range(1, 10)]
    if not np.isclose(
        np.mean([float(table_5[column]) for column in subject_columns]),
        float(table_5["average"]),
        atol=0.01,
    ):
        raise ValueError("Table 5 subject mean does not match its printed average")
    if not np.isclose(
        np.mean([float(table_6[column]) for column in subject_columns]),
        float(table_6["average"]),
        atol=0.01,
    ):
        raise ValueError("Table 6 subject mean does not match its printed average")
    erp_columns = [
        "bnci2014_008",
        "bnci2014_009",
        "bi2012",
        "bi2013a",
        "bi2014b",
        "bi2015a",
        "bi2015b",
        "sosulski2019",
    ]
    if not np.isclose(
        np.mean([float(table_8[column]) for column in erp_columns]),
        float(table_8["average"]),
        atol=0.01,
    ):
        raise ValueError("Table 8 dataset mean does not match its printed average")
    headline = pd.read_csv(source / "headline_results.csv")
    if len(headline) != 5:
        raise ValueError("headline_results.csv must contain five reported outcomes")
    return {"expected": expected, "actual": actual}


def _verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    specifications = dataset_map(config)
    profile = str(config["project"]["profile"])
    if profile == "paper" and tuple(config["data"]["ordering"]) != PAPER_DATASETS:
        raise ValueError("Paper dataset ordering is invalid")
    if profile == "paper":
        overrides = config["preprocessing"]["window"]["overrides"]
        expected_overrides = {
            "bnci2014_004",
            "nakanishi2015",
            "bnci2014_009",
            "sosulski2019",
        }
        if set(overrides) != expected_overrides:
            raise ValueError("Table 4 window overrides are incomplete")
        paper_names = {name: specifications[name]["paper_name"] for name in PAPER_DATASETS}
    else:
        paper_names = {name: value["paper_name"] for name, value in specifications.items()}
    return {
        "profile": profile,
        "datasets": list(config["data"]["ordering"]),
        "paper_names": paper_names,
        "sampling_rate_hz": int(config["preprocessing"]["sfreq"]),
        "window_samples": int(config["preprocessing"]["window"]["size_samples"]),
        "latent_dimensions": int(config["autoencoder"]["d_model"]),
        "transformer_total_layers": int(config["autoencoder"]["num_encoder_layers"])
        + int(config["autoencoder"]["num_decoder_layers"]),
        "attention_heads": int(config["autoencoder"]["nhead"]),
    }


def _verify_datasets(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in config["data"]["ordering"]:
        configured = dataset_map(config)[name]
        specification = get_dataset_spec(configured)
        rows.append(
            {
                "name": name,
                "paper_name": configured["paper_name"],
                "canonical_name": specification.name,
                "moabb_class": configured["moabb_class"],
                "paradigm": configured["paradigm"],
                "interval_seconds": list(specification.interval),
                "channels": int(configured["paper_attributes"]["channels"]),
                "classes": len(configured["labels"]),
            }
        )
    return {"count": len(rows), "datasets": rows}


def _verify_models(config: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        autoencoder = build_autoencoder(config, input_channels=3).eval()
        autoencoder_input = torch.zeros(1, 128, 3)
        reconstructed, latent = autoencoder(autoencoder_input, return_latent=True)
        multiclass = build_classifier(config, num_classes=4, binary=False).eval()
        binary = build_classifier(config, num_classes=2, binary=True).eval()
        classifier_input = torch.zeros(1, 1, 128, 128)
        multiclass_output = multiclass(classifier_input)
        binary_output = binary(classifier_input)
    expected = {
        "autoencoder_output": [1, 128, 3],
        "latent": [1, 128, 128],
        "multiclass_output": [1, 4],
        "binary_output": [1, 1],
    }
    observed = {
        "autoencoder_output": list(reconstructed.shape),
        "latent": list(latent.shape),
        "multiclass_output": list(multiclass_output.shape),
        "binary_output": list(binary_output.shape),
    }
    if observed != expected:
        raise ValueError(f"Model tensor contract mismatch: {observed}")
    focal = BinaryFocalLoss(alpha=0.5, gamma=float(config["classifier"]["focal_gamma"]))
    focal_value = float(focal(torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0], [1.0]])).item())
    if not np.isfinite(focal_value):
        raise ValueError("Focal loss produced a non-finite value")
    return {
        "tensor_shapes": observed,
        "autoencoder_parameters": sum(parameter.numel() for parameter in autoencoder.parameters()),
        "multiclass_classifier_parameters": sum(parameter.numel() for parameter in multiclass.parameters()),
        "binary_classifier_parameters": sum(parameter.numel() for parameter in binary.parameters()),
        "focal_loss_probe": focal_value,
    }


def _verify_splits_and_metrics(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(config["seed"])
    training, validation = random_train_validation_split(40, 0.1, seed)
    labels = np.tile(np.arange(4, dtype=np.int64), 18)
    subjects = np.repeat(np.arange(1, 10, dtype=np.int64), 8)
    mi_folds = classification_folds("mi", labels, subjects, 0, seed)
    tested = np.sort(np.concatenate([fold["test"] for fold in mi_folds]))
    if not np.array_equal(tested, np.arange(len(labels))):
        raise ValueError("MI folds do not cover every sample exactly once")
    binary = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    multiclass = multiclass_classification_metrics(
        [0, 1, 2],
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
    )
    if binary["auc_percent"] != 100.0 or multiclass["accuracy_percent"] != 100.0:
        raise ValueError("Metric contract failed its perfect-prediction probe")
    split_contract = {
        "random_split_train_count": int(len(training)),
        "random_split_validation_count": int(len(validation)),
        "mi_fold_count": len(mi_folds),
        "mi_test_coverage": int(len(tested)),
    }
    metric_contract = {
        "binary_auc_percent_probe": float(binary["auc_percent"]),
        "multiclass_accuracy_percent_probe": float(multiclass["accuracy_percent"]),
        "erp_positive_label": int(config["evaluation"]["roc_auc"]["binary_positive_label"]),
        "multiclass_auc_strategy": config["evaluation"]["roc_auc"]["multiclass_strategy"],
    }
    return split_contract, metric_contract


def verify_repository(config: dict[str, Any], write_report: bool = True) -> dict[str, Any]:
    source = resolve_path(config, "publication_source")
    tolerance = float(config["verification"]["numeric_tolerance"])
    key = config_hash(config)
    protocol = _verify_protocol(config)
    source_hashes = _verify_hashes(source)
    reported_values = _verify_reported_values(source, tolerance)
    dataset_contract = _verify_datasets(config)
    model_contract = _verify_models(config)
    split_contract, metric_contract = _verify_splits_and_metrics(config)
    preprocessing_contract = {
        "operation_order": list(config["preprocessing"]["operation_order"]),
        "sampling_rate_hz": int(config["preprocessing"]["sfreq"]),
        "window_samples": int(config["preprocessing"]["window"]["size_samples"]),
        "hop_samples": int(config["preprocessing"]["window"]["hop_samples"]),
        "standardization": dict(config["preprocessing"]["standardization"]),
    }
    output_contract = {
        "processed": f"data/processed/{key}/<dataset>",
        "runs": f"results/runs/{key}/<dataset>",
        "publication": f"results/publication/generated/{key}",
        "verification": f"results/verification/{key}/verification.json",
    }
    checks = [
        {"name": "configuration_valid", "passed": True, "observed": protocol["profile"]},
        {"name": "dataset_contract", "passed": True, "observed": dataset_contract["count"]},
        {"name": "publication_source_integrity", "passed": True, "observed": len(source_hashes)},
        {"name": "reported_value_transcription", "passed": True, "observed": reported_values["actual"]},
        {"name": "model_tensor_contract", "passed": True, "observed": model_contract["tensor_shapes"]},
        {"name": "split_contract", "passed": True, "observed": split_contract},
        {"name": "metric_contract", "passed": True, "observed": metric_contract},
        {"name": "run_key_stability", "passed": key == config_hash(config), "observed": key},
        {"name": "output_contract", "passed": True, "observed": output_contract},
    ]
    report = {
        "status": "passed",
        "run_key": key,
        "configuration": public_config(config),
        "source_integrity": {"directory": str(source), "files": source_hashes},
        "dataset_contract": dataset_contract,
        "preprocessing_contract": preprocessing_contract,
        "model_contract": model_contract,
        "split_contract": split_contract,
        "metric_contract": metric_contract,
        "output_contract": output_contract,
        "environment": environment_record(),
        "checks": checks,
        "protocol": protocol,
        "source_hashes": source_hashes,
        "reported_values": reported_values,
    }
    if write_report:
        output = resolve_path(config, "verification") / key
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "verification.json", report)
    return report


def verify_json(config: dict[str, Any]) -> str:
    return json.dumps(verify_repository(config), indent=2, sort_keys=True)
