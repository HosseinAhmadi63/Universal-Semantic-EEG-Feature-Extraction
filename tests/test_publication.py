import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from useeg.config import config_hash, generated_directory, load_config, run_directory
from useeg.publication import build_publication_outputs

ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publication_config(tmp_path: Path) -> dict[str, object]:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["paths"] = {
        "data_raw": str(tmp_path / "data" / "raw"),
        "data_processed": str(tmp_path / "data" / "processed"),
        "runs": str(tmp_path / "results" / "runs"),
        "publication_source": str(tmp_path / "results" / "publication" / "source"),
        "publication_generated": str(tmp_path / "results" / "publication" / "generated"),
        "verification": str(tmp_path / "results" / "verification"),
    }
    config["data"]["ordering"] = ["bnci2014_001"]
    config["datasets"][0]["name"] = "bnci2014_001"
    config["datasets"][0]["paper_name"] = "BCICIV_2a"
    source = Path(config["paths"]["publication_source"])
    shutil.copytree(ROOT / "results" / "publication" / "source", source)
    return config


def _write_inputs(config: dict[str, object]) -> list[Path]:
    classification = run_directory(config, "bnci2014_001") / "classification"
    classification.mkdir(parents=True, exist_ok=True)
    complete = classification / "complete.json"
    _write_json(
        complete,
        {
            "dataset": "bnci2014_001",
            "paradigm": "mi",
            "folds": 2,
            "samples": 16,
            "primary_metric": "accuracy_percent",
            "mean": 82.5,
            "standard_deviation": 2.5,
            "minimum": 80.0,
            "maximum": 85.0,
        },
    )
    fold_metrics = classification / "fold_metrics.csv"
    pd.DataFrame(
        [
            {
                "dataset": "bnci2014_001",
                "fold": 1,
                "fold_name": "subject_01",
                "held_out_subject": 1,
                "train_samples": 6,
                "validation_samples": 2,
                "test_samples": 8,
                "best_epoch": 1,
                "accuracy": 0.8,
                "accuracy_percent": 80.0,
                "macro_ovr_auc": 0.9,
                "macro_ovr_auc_percent": 90.0,
            },
            {
                "dataset": "bnci2014_001",
                "fold": 2,
                "fold_name": "subject_02",
                "held_out_subject": 2,
                "train_samples": 6,
                "validation_samples": 2,
                "test_samples": 8,
                "best_epoch": 1,
                "accuracy": 0.85,
                "accuracy_percent": 85.0,
                "macro_ovr_auc": 0.95,
                "macro_ovr_auc_percent": 95.0,
            },
        ]
    ).to_csv(fold_metrics, index=False)
    analysis = generated_directory(config) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    classification_summary = analysis / "classification_summary.csv"
    pd.DataFrame(
        [
            {
                "dataset": "bnci2014_001",
                "paper_name": "BCICIV_2a",
                "paradigm": "mi",
                "primary_metric": "accuracy_percent",
                "generated_mean_percent": 82.5,
                "generated_standard_deviation": 2.5,
                "folds": 2,
                "samples": 16,
                "reported_percent": 83.5,
                "difference_percentage_points": -1.0,
            }
        ]
    ).to_csv(classification_summary, index=False)
    analysis_complete = analysis / "complete.json"
    _write_json(
        analysis_complete,
        {"available": ["bnci2014_001"], "datasets": ["bnci2014_001"]},
    )
    return [complete, fold_metrics, classification_summary, analysis_complete]


def test_build_publication_outputs_creates_hashed_tables_without_mutating_sources(
    tmp_path: Path,
) -> None:
    config = _publication_config(tmp_path)
    inputs = _write_inputs(config)
    source = Path(config["paths"]["publication_source"])
    source_bytes = {path.name: path.read_bytes() for path in sorted(source.glob("*.csv"))}
    output = build_publication_outputs(config)
    assert output == generated_directory(config)
    summary_path = output / "tables" / "generated_dataset_summary.csv"
    comparison_path = output / "tables" / "paper_comparison.csv"
    manifest_path = output / "manifest.json"
    assert summary_path.is_file()
    assert comparison_path.is_file()
    assert manifest_path.is_file()
    summary = pd.read_csv(summary_path)
    assert {
        "dataset",
        "paper_name",
        "paradigm",
        "primary_metric",
        "unit",
        "generated_value",
        "generated_standard_deviation",
        "generated_minimum",
        "generated_maximum",
        "folds",
        "samples",
    }.issubset(summary.columns)
    row = summary.loc[summary["dataset"] == "bnci2014_001"].iloc[0]
    assert row["paper_name"] == "BCICIV_2a"
    assert row["primary_metric"] == "accuracy"
    assert row["unit"] == "percent"
    assert row["generated_value"] == 82.5
    assert row["folds"] == 2
    assert row["samples"] == 16
    comparison = pd.read_csv(comparison_path)
    assert {
        "generated_value",
        "paper_value",
        "difference",
        "metric",
        "unit",
        "dataset",
        "subject",
        "fold",
        "source_file",
    }.issubset(comparison.columns)
    aggregate = comparison.loc[
        (comparison["dataset"] == "bnci2014_001")
        & (comparison["source_file"] == "table_5_bciciv_2a_accuracy.csv")
        & comparison["subject"].isna()
    ].iloc[0]
    assert aggregate["generated_value"] == 82.5
    assert aggregate["paper_value"] == 83.5
    assert aggregate["difference"] == -1.0
    assert aggregate["metric"] == "accuracy"
    assert aggregate["unit"] == "percent"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_key"] == config_hash(config)
    serialized_manifest = json.dumps(manifest, sort_keys=True)
    for path in inputs:
        assert _digest(path) in serialized_manifest
    for name, content in source_bytes.items():
        assert hashlib.sha256(content).hexdigest() in serialized_manifest
        assert (source / name).read_bytes() == content
