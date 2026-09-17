from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import config_hash, dataset_map, generated_directory, resolve_path, run_directory
from .utils import sha256_file, write_csv, write_json

ERP_DATASETS = (
    "bnci2014_008",
    "bnci2014_009",
    "bi2012",
    "bi2013a",
    "bi2014b",
    "bi2015a",
    "bi2015b",
    "sosulski2019",
)


def _complete_record(config: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    path = run_directory(config, dataset_name) / "classification" / "complete.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_summary(config: dict[str, Any]) -> pd.DataFrame:
    specifications = dataset_map(config)
    analysis_path = generated_directory(config) / "analysis" / "classification_summary.csv"
    analyzed = pd.read_csv(analysis_path).set_index("dataset") if analysis_path.exists() else None
    rows: list[dict[str, Any]] = []
    for dataset_name in config["data"]["ordering"]:
        complete = _complete_record(config, dataset_name)
        if complete is None:
            continue
        specification = specifications[dataset_name]
        analyzed_row = analyzed.loc[dataset_name] if analyzed is not None and dataset_name in analyzed.index else None
        primary_metric = str(complete["primary_metric"])
        metric = "roc_auc" if primary_metric == "auc_percent" else "accuracy"
        rows.append(
            {
                "dataset": dataset_name,
                "paper_name": str(specification["paper_name"]),
                "paradigm": str(specification["paradigm"]),
                "primary_metric": metric,
                "unit": "percent",
                "folds": int(complete["folds"]),
                "samples": int(complete["samples"]),
                "generated_value": float(
                    analyzed_row["generated_mean_percent"]
                    if analyzed_row is not None
                    else complete["mean"]
                ),
                "generated_standard_deviation": float(
                    analyzed_row["generated_standard_deviation"]
                    if analyzed_row is not None
                    else complete["standard_deviation"]
                ),
                "generated_minimum": float(complete["minimum"]),
                "generated_maximum": float(complete["maximum"]),
            }
        )
    columns = [
        "dataset",
        "paper_name",
        "paradigm",
        "primary_metric",
        "unit",
        "folds",
        "samples",
        "generated_value",
        "generated_standard_deviation",
        "generated_minimum",
        "generated_maximum",
    ]
    return pd.DataFrame(rows, columns=columns)


def _study_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    selected = frame[frame["method"].astype(str).str.lower() == "this study"]
    if len(selected) != 1:
        raise ValueError(f"{path.name}: expected one This study row")
    return selected.iloc[0]


def _comparison_row(
    dataset: str,
    metric: str,
    generated_value: float,
    paper_value: float,
    source_file: str,
    subject: str | int = "",
    fold: str | int = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "subject": subject,
        "fold": fold,
        "metric": metric,
        "unit": "percent",
        "generated_value": float(generated_value),
        "paper_value": float(paper_value),
        "difference": float(generated_value - paper_value),
        "source_file": source_file,
    }


def _paper_comparison(config: dict[str, Any], summary: pd.DataFrame) -> pd.DataFrame:
    source = resolve_path(config, "publication_source")
    tables = {
        "bnci2014_001": ("table_5_bciciv_2a_accuracy.csv", _study_row(source / "table_5_bciciv_2a_accuracy.csv"), "average"),
        "bnci2014_004": ("table_6_bciciv_2b_accuracy.csv", _study_row(source / "table_6_bciciv_2b_accuracy.csv"), "average"),
        "lee2019_ssvep": ("table_7_ssvep_accuracy.csv", _study_row(source / "table_7_ssvep_accuracy.csv"), "lee2019_ssvep"),
        "nakanishi2015": ("table_7_ssvep_accuracy.csv", _study_row(source / "table_7_ssvep_accuracy.csv"), "nakanishi2015"),
    }
    table_8_name = "table_8_p300_auc.csv"
    table_8 = _study_row(source / table_8_name)
    for dataset_name in ERP_DATASETS:
        tables[dataset_name] = (table_8_name, table_8, dataset_name)
    rows: list[dict[str, Any]] = []
    for record in summary.itertuples(index=False):
        if record.dataset not in tables:
            continue
        source_file, paper_row, column = tables[record.dataset]
        rows.append(
            _comparison_row(
                record.dataset,
                record.primary_metric,
                record.generated_value,
                float(paper_row[column]),
                source_file,
            )
        )
        if record.dataset not in {"bnci2014_001", "bnci2014_004"}:
            continue
        fold_path = run_directory(config, record.dataset) / "classification" / "fold_metrics.csv"
        if not fold_path.exists():
            continue
        folds = pd.read_csv(fold_path)
        for fold_record in folds.itertuples(index=False):
            subject = int(fold_record.held_out_subject)
            paper_column = f"s{subject:02d}"
            rows.append(
                _comparison_row(
                    record.dataset,
                    "accuracy",
                    float(fold_record.accuracy_percent),
                    float(paper_row[paper_column]),
                    source_file,
                    subject=subject,
                    fold=int(fold_record.fold),
                )
            )
    available_erp = summary[summary["dataset"].isin(ERP_DATASETS)]
    if len(available_erp) == len(ERP_DATASETS):
        generated_average = float(available_erp["generated_value"].mean())
        rows.append(
            _comparison_row(
                "eight_dataset_average",
                "roc_auc",
                generated_average,
                float(table_8["average"]),
                table_8_name,
            )
        )
    columns = [
        "dataset",
        "subject",
        "fold",
        "metric",
        "unit",
        "generated_value",
        "paper_value",
        "difference",
        "source_file",
    ]
    return pd.DataFrame(rows, columns=columns)


def _manifest(config: dict[str, Any], root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path == root / "manifest.json":
            continue
        files.append(
            {
                "file": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    source = resolve_path(config, "publication_source")
    source_manifest = pd.read_csv(source / "source_manifest.csv")
    publication_sources = [
        {"file": str(record.file), "sha256": str(record.sha256)}
        for record in source_manifest.itertuples(index=False)
    ]
    publication_sources.append(
        {
            "file": "source_manifest.csv",
            "sha256": sha256_file(source / "source_manifest.csv"),
        }
    )
    project = Path(config["_project_root"])
    inputs = []
    for dataset_name in config["data"]["ordering"]:
        classification = run_directory(config, dataset_name) / "classification"
        for name in ("complete.json", "fold_metrics.csv"):
            path = classification / name
            if not path.exists():
                continue
            try:
                displayed = path.resolve().relative_to(project.resolve()).as_posix()
            except ValueError:
                displayed = path.as_posix()
            inputs.append(
                {
                    "file": displayed,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "schema_version": 1,
        "run_key": config_hash(config),
        "profile": str(config["project"]["profile"]),
        "files": files,
        "inputs": inputs,
        "publication_sources": publication_sources,
    }


def build_publication_outputs(config: dict[str, Any]) -> Path:
    root = generated_directory(config)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = _dataset_summary(config)
    comparison = _paper_comparison(config, summary)
    write_csv(tables / "generated_dataset_summary.csv", summary)
    write_csv(tables / "paper_comparison.csv", comparison)
    write_json(root / "manifest.json", _manifest(config, root))
    return root


__all__ = ["build_publication_outputs"]
