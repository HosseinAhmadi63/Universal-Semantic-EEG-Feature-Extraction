from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import (
    dataset_map,
    generated_directory,
    resolve_path,
    run_directory,
)
from .experiment import _load_autoencoder, _load_processed
from .metrics import anova_feature_statistics, feature_correlation_matrix, greedy_correlation_filter
from .training import resolve_device
from .utils import write_csv, write_json, write_npz


def _available(config: dict[str, Any], dataset_name: str) -> bool:
    root = run_directory(config, dataset_name)
    return (root / "features" / "complete.json").exists()


def _analysis_output(config: dict[str, Any], force: bool) -> Path:
    output = generated_directory(config) / "analysis"
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _load_feature_data(
    config: dict[str, Any], dataset_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x, y, metadata, _ = _load_processed(config, dataset_name)
    feature_root = run_directory(config, dataset_name) / "features"
    vectors = np.load(feature_root / "window_vectors.npy", mmap_mode="r")
    if not (len(x) == len(y) == len(vectors) == len(metadata)):
        raise RuntimeError(f"{dataset_name}: analysis inputs are misaligned")
    return x, y, vectors, metadata


def _deterministic_stratified_indices(
    groups: np.ndarray, cap: int, seed: int
) -> np.ndarray:
    values = np.asarray(groups)
    if len(values) <= cap:
        return np.arange(len(values), dtype=np.int64)
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(values, return_counts=True)
    quotas = np.maximum(1, np.floor(cap * counts / counts.sum()).astype(int))
    while quotas.sum() > cap:
        reducible = np.flatnonzero(quotas > 1)
        quotas[reducible[np.argmax(quotas[reducible])]] -= 1
    while quotas.sum() < cap:
        room = counts - quotas
        quotas[np.argmax(room)] += 1
    selected: list[np.ndarray] = []
    for group, quota in zip(unique, quotas, strict=True):
        candidates = np.flatnonzero(values == group)
        selected.append(np.sort(rng.choice(candidates, size=int(quota), replace=False)))
    return np.sort(np.concatenate(selected).astype(np.int64))


def analyze_classification(
    config: dict[str, Any], dataset_names: list[str], output: Path
) -> pd.DataFrame:
    specifications = dataset_map(config)
    rows: list[dict[str, Any]] = []
    for name in dataset_names:
        complete = run_directory(config, name) / "classification" / "complete.json"
        if not complete.exists():
            continue
        summary = json.loads(complete.read_text())
        row = {
            "dataset": name,
            "paper_name": specifications[name]["paper_name"],
            "paradigm": specifications[name]["paradigm"],
            "primary_metric": summary["primary_metric"],
            "generated_mean_percent": summary["mean"],
            "generated_standard_deviation": summary["standard_deviation"],
            "folds": summary["folds"],
            "samples": summary["samples"],
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    source = pd.read_csv(resolve_path(config, "publication_source") / "headline_results.csv")
    source = source.rename(columns={"dataset": "paper_name", "value": "reported_percent"})
    frame = frame.merge(source[["paper_name", "reported_percent"]], on="paper_name", how="left")
    frame["difference_percentage_points"] = (
        frame["generated_mean_percent"] - frame["reported_percent"]
    )
    write_csv(output / "classification_summary.csv", frame)
    return frame


def analyze_correlations(config: dict[str, Any], output: Path) -> None:
    settings = config["analysis"]["correlation"]
    dataset_name = str(settings["dataset"])
    if not _available(config, dataset_name):
        return
    _, y, vectors, metadata = _load_feature_data(config, dataset_name)
    correlation_output = output / "correlation"
    correlation_output.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []
    selected_subjects = set(int(value) for value in settings["selected_subjects"])
    for subject in sorted(metadata["subject"].unique().tolist()):
        selector = metadata["subject"].to_numpy() == subject
        subject_vectors = np.asarray(vectors[selector], dtype=np.float64)
        correlation = np.nan_to_num(feature_correlation_matrix(subject_vectors))
        np.save(correlation_output / f"subject_{int(subject):02d}.npy", correlation.astype(np.float32))
        if int(subject) not in selected_subjects:
            continue
        for threshold in settings["thresholds"]:
            retained, _ = greedy_correlation_filter(subject_vectors, float(threshold))
            selection_rows.append(
                {
                    "subject": int(subject),
                    "threshold": float(threshold),
                    "retained_count": int(len(retained)),
                    "retained_indices_zero_based": " ".join(str(int(value)) for value in retained),
                    "retained_ids_one_based": " ".join(str(int(value) + 1) for value in retained),
                }
            )
    write_csv(output / "feature_selection.csv", pd.DataFrame(selection_rows))
    anova = config["analysis"]["anova"]
    subject = int(anova["subject"])
    selector = metadata["subject"].to_numpy() == subject
    subject_vectors = np.asarray(vectors[selector], dtype=np.float64)
    subject_labels = np.asarray(y[selector], dtype=np.int64)
    anova_rows: list[dict[str, Any]] = []
    for threshold in anova["thresholds"]:
        retained, selected = greedy_correlation_filter(subject_vectors, float(threshold))
        f_statistics, p_values = anova_feature_statistics(selected, subject_labels)
        for feature, statistic, p_value in zip(retained, f_statistics, p_values, strict=True):
            anova_rows.append(
                {
                    "subject": subject,
                    "threshold": float(threshold),
                    "feature_index_zero_based": int(feature),
                    "feature_id_one_based": int(feature) + 1,
                    "f_statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": bool(p_value < float(anova["alpha"])),
                }
            )
    write_csv(output / "anova.csv", pd.DataFrame(anova_rows))


def analyze_clustering(config: dict[str, Any], output: Path) -> None:
    dataset_name = str(config["analysis"]["pca"]["dataset"])
    if not _available(config, dataset_name):
        return
    _, y, vectors, metadata = _load_feature_data(config, dataset_name)
    sample_cap = min(int(config["analysis"]["tsne"]["sample_cap"]), 5000)
    indices = _deterministic_stratified_indices(np.asarray(y), sample_cap, int(config["seed"]))
    selected = np.asarray(vectors[indices], dtype=np.float64)
    scaled = StandardScaler().fit_transform(selected)
    pca_settings = config["analysis"]["pca"]
    pca = PCA(
        n_components=int(pca_settings["components"]),
        whiten=bool(pca_settings["whiten"]),
        random_state=int(pca_settings["random_state"]),
    )
    coordinates = pca.fit_transform(scaled)
    kmeans_settings = config["analysis"]["kmeans"]
    kmeans = KMeans(
        n_clusters=int(kmeans_settings["clusters"]),
        n_init=int(kmeans_settings["n_init"]),
        random_state=int(kmeans_settings["random_state"]),
    )
    clusters = kmeans.fit_predict(scaled)
    clustering = pd.DataFrame(
        {
            "sample_index": indices,
            "subject": metadata.iloc[indices]["subject"].to_numpy(),
            "target": np.asarray(y[indices], dtype=np.int64),
            "cluster": clusters,
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
        }
    )
    write_csv(output / "pca_kmeans.csv", clustering)
    sample_scores = silhouette_samples(scaled, clusters)
    write_csv(
        output / "silhouette.csv",
        pd.DataFrame(
            {
                "sample_index": indices,
                "cluster": clusters,
                "silhouette": sample_scores,
            }
        ),
    )
    write_json(
        output / "clustering_summary.json",
        {
            "dataset": dataset_name,
            "samples": int(len(indices)),
            "silhouette_score": float(silhouette_score(scaled, clusters)),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "kmeans_inertia": float(kmeans.inertia_),
        },
    )
    hierarchy_cap = min(len(indices), 2000)
    hierarchy_indices = _deterministic_stratified_indices(
        np.asarray(y[indices]), hierarchy_cap, int(config["seed"])
    )
    hierarchy_matrix = linkage(
        scaled[hierarchy_indices],
        method=str(config["analysis"]["hierarchy"]["linkage"]),
        metric=str(config["analysis"]["hierarchy"]["metric"]),
    )
    np.save(output / "hierarchy_linkage.npy", hierarchy_matrix)
    write_npz(
        output / "hierarchy_samples.npz",
        sample_indices=indices[hierarchy_indices],
        targets=np.asarray(y[indices[hierarchy_indices]], dtype=np.int64),
    )


def analyze_tsne(config: dict[str, Any], output: Path) -> None:
    settings = config["analysis"]["tsne"]
    for dataset_name in config["analysis"]["representative_datasets"]["tsne"]:
        if not _available(config, dataset_name):
            continue
        x, _, vectors, metadata = _load_feature_data(config, dataset_name)
        indices = _deterministic_stratified_indices(
            metadata["subject"].to_numpy(),
            int(settings["sample_cap"]),
            int(settings["random_state"]),
        )
        raw = np.asarray(x[indices], dtype=np.float32).reshape(len(indices), -1)
        latent = np.asarray(vectors[indices], dtype=np.float32)
        coordinates: dict[str, np.ndarray] = {}
        for representation, matrix in (("raw", raw), ("latent", latent)):
            scaled = StandardScaler().fit_transform(matrix)
            components = min(
                int(settings["pre_pca_components"]), scaled.shape[1], scaled.shape[0] - 1
            )
            reduced = PCA(n_components=components, random_state=int(settings["random_state"])).fit_transform(
                scaled
            )
            perplexity = min(float(settings["perplexity"]), max(1.0, len(indices) - 1.0))
            coordinates[representation] = TSNE(
                n_components=int(settings["components"]),
                perplexity=perplexity,
                learning_rate=settings["learning_rate"],
                init=str(settings["init"]),
                max_iter=int(settings["iterations"]),
                random_state=int(settings["random_state"]),
            ).fit_transform(reduced)
        frame = pd.DataFrame(
            {
                "sample_index": indices,
                "subject": metadata.iloc[indices]["subject"].to_numpy(),
                "raw_x": coordinates["raw"][:, 0],
                "raw_y": coordinates["raw"][:, 1],
                "latent_x": coordinates["latent"][:, 0],
                "latent_y": coordinates["latent"][:, 1],
            }
        )
        write_csv(output / f"tsne_{dataset_name}.csv", frame)


def analyze_roc(config: dict[str, Any], output: Path) -> None:
    dataset_name = "bnci2014_001"
    prediction_path = run_directory(config, dataset_name) / "classification" / "predictions.npz"
    if not prediction_path.exists():
        return
    with np.load(prediction_path) as saved:
        targets = saved["targets"].astype(np.int64)
        probabilities = saved["probabilities"].astype(np.float64)
    rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    for class_index in range(probabilities.shape[1]):
        binary = (targets == class_index).astype(np.int64)
        false_positive, true_positive, thresholds = roc_curve(binary, probabilities[:, class_index])
        area = float(np.trapezoid(true_positive, false_positive))
        auc_rows.append({"class": class_index + 1, "generated_auc": area})
        for fpr, tpr, threshold in zip(false_positive, true_positive, thresholds, strict=True):
            rows.append(
                {
                    "class": class_index + 1,
                    "false_positive_rate": float(fpr),
                    "true_positive_rate": float(tpr),
                    "threshold": float(threshold),
                }
            )
    write_csv(output / "roc_bciciv_2a.csv", pd.DataFrame(rows))
    reported = pd.read_csv(resolve_path(config, "publication_source") / "figure_7_class_auc.csv")
    comparison = pd.DataFrame(auc_rows).merge(reported, on="class", how="left")
    comparison["difference"] = comparison["generated_auc"] - comparison["reported_auc"]
    write_csv(output / "roc_bciciv_2a_auc.csv", comparison)


def analyze_reconstruction_examples(config: dict[str, Any], output: Path) -> None:
    rng = np.random.default_rng(int(config["figures"]["reconstruction_examples"]["random_state"]))
    originals: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    device = resolve_device(config["device"])
    for entry in config["figures"]["reconstruction_examples"]["datasets"]:
        dataset_name = str(entry["name"])
        if not _available(config, dataset_name):
            continue
        x, _, _, metadata = _load_feature_data(config, dataset_name)
        subject = int(entry["subject"])
        channel = int(entry["channel"]) - 1
        candidates = np.flatnonzero(metadata["subject"].to_numpy() == subject)
        count = min(int(config["figures"]["reconstruction_examples"]["count_per_dataset"]), len(candidates))
        selected = np.sort(rng.choice(candidates, size=count, replace=False))
        model = _load_autoencoder(config, dataset_name, int(x.shape[2])).to(device)
        model.eval()
        batch = torch.as_tensor(np.array(x[selected], copy=True), dtype=torch.float32, device=device)
        with torch.no_grad():
            reconstructed = model(batch).cpu().numpy()
        for example, (sample_index, original, reconstruction) in enumerate(
            zip(selected, np.asarray(x[selected]), reconstructed, strict=True), 1
        ):
            originals.append(original[:, channel].astype(np.float32))
            reconstructions.append(reconstruction[:, channel].astype(np.float32))
            rows.append(
                {
                    "dataset": dataset_name,
                    "paper_name": dataset_map(config)[dataset_name]["paper_name"],
                    "subject": subject,
                    "channel_one_based": channel + 1,
                    "example": example,
                    "sample_index": int(sample_index),
                    "mse": float(np.mean(np.square(reconstruction - original))),
                }
            )
    if rows:
        write_npz(
            output / "reconstruction_examples.npz",
            originals=np.stack(originals),
            reconstructions=np.stack(reconstructions),
        )
        write_csv(output / "reconstruction_examples.csv", pd.DataFrame(rows))


def run_analysis(
    config: dict[str, Any], dataset_names: list[str], force: bool = False
) -> Path:
    output = _analysis_output(config, force)
    analyze_classification(config, dataset_names, output)
    analyze_correlations(config, output)
    analyze_clustering(config, output)
    analyze_tsne(config, output)
    analyze_roc(config, output)
    analyze_reconstruction_examples(config, output)
    write_json(
        output / "complete.json",
        {
            "datasets": dataset_names,
            "available": [name for name in dataset_names if _available(config, name)],
        },
    )
    return output
