from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from .config import dataset_map, processed_directory, run_directory
from .metrics import binary_classification_metrics, multiclass_classification_metrics
from .models import (
    HierarchicalEEGAutoencoder,
    build_autoencoder,
    build_classifier,
)
from .splits import (
    classification_folds,
    random_train_validation_split,
    stratified_train_validation_split,
)
from .training import (
    ArrayDataset,
    load_checkpoint,
    predict_classifier,
    reconstruction_errors,
    resolve_device,
    train_autoencoder,
    train_classifier,
)
from .utils import environment_record, write_csv, write_json, write_npz


def _reset_output(output: Path, force: bool) -> None:
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _history_frame(history: dict[str, Any]) -> pd.DataFrame:
    length = len(history["train_loss"])
    return pd.DataFrame(
        {
            "epoch": np.arange(1, length + 1),
            "train_loss": history["train_loss"],
            "validation_loss": history["validation_loss"],
            "learning_rate": history["learning_rate"],
        }
    )


def _load_processed(
    config: dict[str, Any], dataset_name: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    source = processed_directory(config, dataset_name)
    required = (source / "X.npy", source / "y.npy", source / "metadata.npy")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Processed dataset is incomplete: {missing}")
    x = np.load(source / "X.npy", mmap_mode="r")
    y = np.load(source / "y.npy", mmap_mode="r")
    metadata = pd.DataFrame.from_records(np.load(source / "metadata.npy", mmap_mode="r"))
    metadata["subject"] = pd.to_numeric(metadata["subject"], errors="raise").astype(np.int64)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if len(x) != len(y) or len(x) != len(metadata):
        raise RuntimeError(f"{dataset_name}: processed arrays and metadata are misaligned")
    return x, y, metadata, manifest


def train_dataset_autoencoder(
    config: dict[str, Any], dataset_name: str, force: bool = False
) -> Path:
    output = run_directory(config, dataset_name) / "autoencoder"
    complete = output / "complete.json"
    if complete.exists() and not force:
        return output
    _reset_output(output, force)
    x, _, _, manifest = _load_processed(config, dataset_name)
    if x.ndim != 3 or x.shape[1] != 128:
        raise ValueError(f"{dataset_name}: expected N x 128 x C processed windows")
    seed = int(config["seed"])
    train_indices, validation_indices = random_train_validation_split(
        len(x), float(config["autoencoder"]["val_fraction"]), seed
    )
    write_npz(output / "split_indices.npz", train=train_indices, validation=validation_indices)
    dataset = ArrayDataset(x)
    train_data = Subset(dataset, train_indices.tolist())
    validation_data = Subset(dataset, validation_indices.tolist())
    model = build_autoencoder(config, input_channels=int(x.shape[2]))
    checkpoint = output / "best.pt"
    history = train_autoencoder(
        model,
        train_data,
        validation_data,
        config=config,
        device=config["device"],
        checkpoint_path=checkpoint,
    )
    write_csv(output / "history.csv", _history_frame(history))
    errors = reconstruction_errors(
        model,
        validation_data,
        batch_size=int(config["autoencoder"]["batch_size"]),
        device=config["device"],
    )
    np.save(output / "validation_reconstruction_errors.npy", errors.astype(np.float32))
    threshold = float(config["autoencoder"]["reconstruction_mse_threshold"])
    summary = {
        "dataset": dataset_name,
        "samples": int(len(x)),
        "channels": int(x.shape[2]),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "best_epoch": int(history["best_epoch"]) + 1,
        "epochs_ran": int(history["epochs_ran"]),
        "best_validation_loss": float(history["best_validation_loss"]),
        "validation_mse": float(errors.mean()),
        "validation_mse_std": float(errors.std()),
        "threshold": threshold,
        "passes_threshold": bool(errors.mean() < threshold),
        "source_manifest": manifest,
        "environment": environment_record(),
    }
    write_json(complete, summary)
    return output


def _load_autoencoder(
    config: dict[str, Any], dataset_name: str, input_channels: int
) -> HierarchicalEEGAutoencoder:
    checkpoint = run_directory(config, dataset_name) / "autoencoder" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint}")
    model = build_autoencoder(config, input_channels=input_channels)
    load_checkpoint(checkpoint, model, map_location="cpu")
    return model


def extract_dataset_features(
    config: dict[str, Any], dataset_name: str, force: bool = False
) -> Path:
    output = run_directory(config, dataset_name) / "features"
    complete = output / "complete.json"
    if complete.exists() and not force:
        return output
    _reset_output(output, force)
    x, y, metadata, _ = _load_processed(config, dataset_name)
    model = _load_autoencoder(config, dataset_name, int(x.shape[2]))
    device = resolve_device(config["device"])
    model.to(device)
    model.eval()
    sample_count = len(x)
    latent_shape = (sample_count, 128, 128)
    latent_temporary = output / "latent_maps.npy.tmp"
    vector_temporary = output / "window_vectors.npy.tmp"
    latent_maps = np.lib.format.open_memmap(
        latent_temporary, mode="w+", dtype=np.float32, shape=latent_shape
    )
    window_vectors = np.lib.format.open_memmap(
        vector_temporary, mode="w+", dtype=np.float32, shape=(sample_count, 128)
    )
    batch_size = int(config["autoencoder"]["batch_size"])
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            stop = min(sample_count, start + batch_size)
            batch = torch.as_tensor(np.array(x[start:stop], copy=True), dtype=torch.float32, device=device)
            latent = model.encode(batch)
            latent_array = latent.cpu().numpy().astype(np.float32, copy=False)
            latent_maps[start:stop] = latent_array
            window_vectors[start:stop] = latent_array.mean(axis=1)
    latent_maps.flush()
    window_vectors.flush()
    del latent_maps, window_vectors
    latent_temporary.replace(output / "latent_maps.npy")
    vector_temporary.replace(output / "window_vectors.npy")
    vectors = np.load(output / "window_vectors.npy", mmap_mode="r")
    subject_rows: list[dict[str, Any]] = []
    for subject in sorted(metadata["subject"].unique().tolist()):
        selector = metadata["subject"].to_numpy() == subject
        vector = np.asarray(vectors[selector]).mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        row: dict[str, Any] = {"subject": int(subject), "windows": int(selector.sum())}
        row.update({f"feature_{index:03d}": float(value) for index, value in enumerate(vector)})
        subject_rows.append(row)
    write_csv(output / "subject_vectors.csv", pd.DataFrame(subject_rows))
    write_json(
        complete,
        {
            "dataset": dataset_name,
            "samples": sample_count,
            "latent_map_shape": list(latent_shape),
            "window_vector_shape": [sample_count, 128],
            "subjects": len(subject_rows),
            "labels": sorted(np.unique(y).astype(int).tolist()),
        },
    )
    return output


def _metric_row(
    paradigm: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    if paradigm == "erp":
        metrics = binary_classification_metrics(y_true, probabilities)
        return (
            {
                "accuracy": float(metrics["accuracy"]),
                "accuracy_percent": float(metrics["accuracy_percent"]),
                "auc": float(metrics["auc"]),
                "auc_percent": float(metrics["auc_percent"]),
            },
            np.asarray(metrics["predictions"], dtype=np.int64),
        )
    metrics = multiclass_classification_metrics(y_true, probabilities)
    return (
        {
            "accuracy": float(metrics["accuracy"]),
            "accuracy_percent": float(metrics["accuracy_percent"]),
            "macro_ovr_auc": float(metrics["macro_ovr_auc"]),
            "macro_ovr_auc_percent": float(metrics["macro_ovr_auc_percent"]),
        },
        np.asarray(metrics["predictions"], dtype=np.int64),
    )


def classify_dataset(
    config: dict[str, Any], dataset_name: str, force: bool = False
) -> Path:
    output = run_directory(config, dataset_name) / "classification"
    complete = output / "complete.json"
    if complete.exists() and not force:
        return output
    _reset_output(output, force)
    _, y_mmap, metadata, _ = _load_processed(config, dataset_name)
    y = np.asarray(y_mmap, dtype=np.int64)
    latent_path = run_directory(config, dataset_name) / "features" / "latent_maps.npy"
    if not latent_path.exists():
        raise FileNotFoundError(f"Latent maps not found: {latent_path}")
    latent = np.load(latent_path, mmap_mode="r")
    spec = dataset_map(config)[dataset_name]
    paradigm = str(spec["paradigm"])
    evaluation = config["evaluation"][paradigm]
    folds = classification_folds(
        paradigm,
        y,
        metadata["subject"].to_numpy(),
        int(evaluation.get("folds", 0)),
        int(evaluation.get("random_state", config["seed"])),
    )
    classes = len(spec["labels"])
    binary = paradigm == "erp"
    probabilities = (
        np.full(len(y), np.nan, dtype=np.float32)
        if binary
        else np.full((len(y), classes), np.nan, dtype=np.float32)
    )
    predictions = np.full(len(y), -1, dtype=np.int64)
    fold_assignments = np.full(len(y), -1, dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    full_dataset = ArrayDataset(latent, y)
    for fold in folds:
        fold_number = int(fold["fold"])
        fold_output = output / "folds" / f"fold_{fold_number:02d}"
        fold_output.mkdir(parents=True, exist_ok=True)
        training_indices, validation_indices = stratified_train_validation_split(
            np.asarray(fold["train"]),
            y,
            float(config["classifier"]["val_fraction"]),
            int(config["seed"]) + fold_number,
        )
        test_indices = np.asarray(fold["test"], dtype=np.int64)
        write_npz(
            fold_output / "indices.npz",
            train=training_indices,
            validation=validation_indices,
            test=test_indices,
        )
        model = build_classifier(config, num_classes=classes, binary=binary)
        history = train_classifier(
            model,
            Subset(full_dataset, training_indices.tolist()),
            None,
            Subset(full_dataset, validation_indices.tolist()),
            None,
            config=config,
            device=config["device"],
            checkpoint_path=fold_output / "best.pt",
        )
        write_csv(fold_output / "history.csv", _history_frame(history))
        prediction = predict_classifier(
            model,
            Subset(ArrayDataset(latent), test_indices.tolist()),
            batch_size=int(config["classifier"]["batch_size"]),
            device=config["device"],
        )
        fold_probabilities = np.asarray(prediction["probabilities"], dtype=np.float32)
        probabilities[test_indices] = fold_probabilities
        metric, fold_predictions = _metric_row(paradigm, y[test_indices], fold_probabilities)
        predictions[test_indices] = fold_predictions
        fold_assignments[test_indices] = fold_number
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "fold": fold_number,
            "fold_name": fold["name"],
            "held_out_subject": fold["held_out_subject"],
            "train_samples": int(len(training_indices)),
            "validation_samples": int(len(validation_indices)),
            "test_samples": int(len(test_indices)),
            "best_epoch": int(history["best_epoch"]) + 1,
        }
        row.update(metric)
        metric_rows.append(row)
    if np.any(fold_assignments < 0) or np.any(predictions < 0) or np.any(~np.isfinite(probabilities)):
        raise RuntimeError(f"{dataset_name}: cross-validation did not produce complete predictions")
    metrics_frame = pd.DataFrame(metric_rows)
    write_csv(output / "fold_metrics.csv", metrics_frame)
    assignment_frame = metadata.copy()
    assignment_frame["fold"] = fold_assignments
    assignment_frame["target"] = y
    assignment_frame["prediction"] = predictions
    write_csv(output / "fold_assignments.csv", assignment_frame)
    write_npz(
        output / "predictions.npz",
        targets=y,
        predictions=predictions,
        probabilities=probabilities,
        folds=fold_assignments,
    )
    primary = "auc_percent" if binary else "accuracy_percent"
    summary = {
        "dataset": dataset_name,
        "paradigm": paradigm,
        "folds": len(folds),
        "samples": len(y),
        "primary_metric": primary,
        "mean": float(metrics_frame[primary].mean()),
        "standard_deviation": float(metrics_frame[primary].std(ddof=0)),
        "minimum": float(metrics_frame[primary].min()),
        "maximum": float(metrics_frame[primary].max()),
    }
    write_json(complete, summary)
    return output


def run_ablation(config: dict[str, Any], force: bool = False) -> Path:
    ablation = config["analysis"]["ablation"]
    dataset_name = str(ablation["dataset"])
    output = run_directory(config, dataset_name) / "ablation"
    complete = output / "complete.json"
    if complete.exists() and not force:
        return output
    _reset_output(output, force)
    x, _, _, _ = _load_processed(config, dataset_name)
    train_indices, validation_indices = random_train_validation_split(
        len(x), float(ablation["evaluation_fraction"]), int(config["seed"])
    )
    dataset = ArrayDataset(x)
    rows: list[dict[str, Any]] = []
    for model_spec in ablation["configurations"]:
        role = str(model_spec["role"])
        model = HierarchicalEEGAutoencoder(
            input_channels=int(x.shape[2]),
            window_size=128,
            d_model=int(config["autoencoder"]["d_model"]),
            num_heads=int(model_spec["attention_heads"]),
            encoder_layers=int(model_spec["encoder_layers"]),
            decoder_layers=int(model_spec["decoder_layers"]),
            ffn_dim=int(config["autoencoder"]["dim_feedforward"]),
            dropout=float(config["autoencoder"]["attention_dropout"]),
        )
        model_output = output / role
        model_output.mkdir(parents=True, exist_ok=True)
        training = Subset(dataset, train_indices.tolist())
        validation = Subset(dataset, validation_indices.tolist())
        history = train_autoencoder(
            model,
            training,
            validation,
            config=config,
            device=config["device"],
            checkpoint_path=model_output / "best.pt",
        )
        write_csv(model_output / "history.csv", _history_frame(history))
        errors = reconstruction_errors(
            model,
            validation,
            batch_size=int(config["autoencoder"]["batch_size"]),
            device=config["device"],
        ).astype(np.float32)
        np.save(model_output / "reconstruction_errors.npy", errors)
        rows.append(
            {
                "configuration": role,
                "total_layers": int(model_spec["total_layers"]),
                "encoder_layers": int(model_spec["encoder_layers"]),
                "decoder_layers": int(model_spec["decoder_layers"]),
                "attention_heads": int(model_spec["attention_heads"]),
                "validation_samples": int(len(errors)),
                "mean_mse": float(errors.mean()),
                "standard_deviation": float(errors.std()),
                "best_epoch": int(history["best_epoch"]) + 1,
            }
        )
    write_csv(output / "summary.csv", pd.DataFrame(rows))
    write_npz(output / "split_indices.npz", train=train_indices, validation=validation_indices)
    write_json(
        complete,
        {
            "dataset": dataset_name,
            "validation_samples": int(len(validation_indices)),
            "configurations": rows,
            "reported_evaluation_count": int(ablation["reported_evaluation_count"]),
        },
    )
    return output


def run_dataset_pipeline(
    config: dict[str, Any], dataset_name: str, force: bool = False
) -> Path:
    train_dataset_autoencoder(config, dataset_name, force=force)
    extract_dataset_features(config, dataset_name, force=force)
    classify_dataset(config, dataset_name, force=force)
    return run_directory(config, dataset_name)
