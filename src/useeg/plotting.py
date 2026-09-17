from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram

from .config import generated_directory, run_directory
from .publication import build_publication_outputs
from .utils import sha256_file


def _save(config: dict[str, Any], figure: Figure, output: Path, stem: str) -> list[Path]:
    saved: list[Path] = []
    for file_format in config["figures"]["formats"]:
        path = output / f"{stem}.{file_format}"
        figure.savefig(path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
        saved.append(path)
    plt.close(figure)
    return saved


def _display_path(config: dict[str, Any], path: Path) -> str:
    project = Path(config["_project_root"])
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _figure_sources(config: dict[str, Any], root: Path, path: Path) -> list[str]:
    stem = path.stem
    analysis = root / "analysis"
    if stem == "figure_01_architecture":
        return ["configuration:autoencoder", "configuration:classifier"]
    if stem.startswith(("figure_02_", "figure_03_", "figure_04_")):
        dataset_name = stem.split("_", 2)[2].removesuffix("_learning")
        return [_display_path(config, run_directory(config, dataset_name) / "autoencoder" / "history.csv")]
    mappings = {
        "figure_05_reconstruction": [
            analysis / "reconstruction_examples.csv",
            analysis / "reconstruction_examples.npz",
        ],
        "figure_06_feature_correlations": [analysis / "correlation"],
        "figure_07_bciciv_2a_roc": [
            analysis / "roc_bciciv_2a.csv",
            analysis / "roc_bciciv_2a_auc.csv",
        ],
        "figure_08_raw_latent_tsne": [analysis / f"tsne_{name}.csv" for name in config["analysis"]["representative_datasets"]["tsne"]],
        "figure_09_filtered_correlations": [
            analysis / "feature_selection.csv",
            analysis / "correlation",
        ],
        "figure_10_anova": [analysis / "anova.csv"],
        "figure_11_hierarchical_clustering": [analysis / "hierarchy_linkage.npy"],
        "figure_12_pca_kmeans": [analysis / "pca_kmeans.csv"],
        "figure_13_silhouette": [
            analysis / "silhouette.csv",
            analysis / "clustering_summary.json",
        ],
        "figure_14_ablation": [
            run_directory(config, str(config["analysis"]["ablation"]["dataset"])) / "ablation" / "summary.csv"
        ],
    }
    resolved: list[str] = []
    for source in mappings.get(stem, []):
        if source.is_dir():
            resolved.extend(
                _display_path(config, candidate)
                for candidate in sorted(source.rglob("*"))
                if candidate.is_file()
            )
        elif source.exists():
            resolved.append(_display_path(config, source))
    if stem == "figure_14_ablation":
        ablation = run_directory(
            config, str(config["analysis"]["ablation"]["dataset"])
        ) / "ablation"
        resolved.extend(
            _display_path(config, candidate)
            for candidate in sorted(ablation.glob("*/reconstruction_errors.npy"))
        )
    return sorted(set(resolved))


def _architecture(config: dict[str, Any], output: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(16, 6))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 6)
    axis.axis("off")
    boxes = [
        (0.3, 2.1, 2.0, 1.8, "EEG\n128 × C"),
        (2.8, 2.1, 2.2, 1.8, "Conv1D encoder\n64 → 128\nLayerNorm"),
        (5.5, 2.1, 2.1, 1.8, "Dense 128\n+ sinusoidal\nposition"),
        (8.1, 2.1, 2.2, 1.8, "Transformer\nencoder\n2 × 8 heads"),
        (10.8, 2.1, 2.2, 1.8, "Self-attention\ndecoder\n2 × 8 heads"),
        (13.5, 2.1, 2.2, 1.8, "ConvTranspose\n128 → 64\nEEG 128 × C"),
    ]
    for x, y, width, height, text in boxes:
        axis.add_patch(
            plt.Rectangle((x, y), width, height, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=2)
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11)
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        axis.annotate(
            "",
            xy=(right[0], 3.0),
            xytext=(left[0] + left[2], 3.0),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#0f172a"},
        )
    axis.text(9.2, 1.1, "Latent semantic map: 128 × 128", ha="center", fontsize=12)
    axis.set_title("Hierarchical dual autoencoder for semantic EEG representation", fontsize=16)
    return _save(config, figure, output, "figure_01_architecture")


def _learning_curves(config: dict[str, Any], output: Path) -> list[Path]:
    saved: list[Path] = []
    figure_numbers = (2, 3, 4)
    for figure_number, dataset_name in zip(
        figure_numbers,
        config["analysis"]["representative_datasets"]["learning_curves"],
        strict=False,
    ):
        path = run_directory(config, dataset_name) / "autoencoder" / "history.csv"
        if not path.exists():
            continue
        history = pd.read_csv(path)
        figure, loss_axis = plt.subplots(figsize=(8, 5))
        learning_axis = loss_axis.twinx()
        loss_axis.plot(history["epoch"], history["train_loss"], label="Training loss", lw=2)
        loss_axis.plot(history["epoch"], history["validation_loss"], label="Validation loss", lw=2)
        learning_axis.step(
            history["epoch"], history["learning_rate"], where="post", color="#16a34a", label="Learning rate"
        )
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Mean squared error")
        learning_axis.set_ylabel("Learning rate")
        learning_axis.set_yscale("log")
        handles_left, labels_left = loss_axis.get_legend_handles_labels()
        handles_right, labels_right = learning_axis.get_legend_handles_labels()
        loss_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")
        loss_axis.set_title(dataset_name)
        saved.extend(
            _save(config, figure, output, f"figure_{figure_number:02d}_{dataset_name}_learning")
        )
    return saved


def _reconstruction(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    csv_path = analysis / "reconstruction_examples.csv"
    array_path = analysis / "reconstruction_examples.npz"
    if not csv_path.exists() or not array_path.exists():
        return []
    metadata = pd.read_csv(csv_path)
    with np.load(array_path) as saved:
        originals = saved["originals"]
        reconstructions = saved["reconstructions"]
    rows = len(metadata)
    figure, axes = plt.subplots(rows, 1, figsize=(10, max(3, 2.4 * rows)), squeeze=False)
    time = np.arange(originals.shape[1]) / 128.0
    for index, row in metadata.iterrows():
        axis = axes[index, 0]
        axis.plot(time, originals[index], lw=1.5, label="Original")
        axis.plot(time, reconstructions[index], lw=1.3, ls="--", label="Reconstructed")
        axis.set_title(
            f"{row['paper_name']} · S{int(row['subject']):02d} · C{int(row['channel_one_based'])} · MSE {row['mse']:.4f}"
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Standardized amplitude")
    axes[0, 0].legend(loc="upper right")
    figure.tight_layout()
    return _save(config, figure, output, "figure_05_reconstruction")


def _correlation(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    correlation_root = analysis / "correlation"
    files = sorted(correlation_root.glob("subject_*.npy"))
    if not files:
        return []
    columns = 3
    rows = int(np.ceil(len(files) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), squeeze=False)
    image = None
    for axis, path in zip(axes.ravel(), files, strict=False):
        image = axis.imshow(np.load(path), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        axis.set_title(path.stem.replace("subject_", "Subject A"))
        axis.set_xlabel("Feature")
        axis.set_ylabel("Feature")
    for axis in axes.ravel()[len(files) :]:
        axis.axis("off")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Pearson correlation")
    return _save(config, figure, output, "figure_06_feature_correlations")


def _roc(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    path = analysis / "roc_bciciv_2a.csv"
    if not path.exists():
        return []
    curves = pd.read_csv(path)
    auc_path = analysis / "roc_bciciv_2a_auc.csv"
    areas = pd.read_csv(auc_path).set_index("class") if auc_path.exists() else pd.DataFrame()
    figure, axis = plt.subplots(figsize=(7, 7))
    for class_id, group in curves.groupby("class"):
        area = float(areas.loc[class_id, "generated_auc"]) if not areas.empty else float("nan")
        axis.plot(
            group["false_positive_rate"],
            group["true_positive_rate"],
            lw=2,
            label=f"Class {int(class_id)} · AUC {area:.4f}",
        )
    axis.plot([0, 1], [0, 1], ls="--", color="black", lw=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("BCICIV_2a aggregate one-vs-rest ROC")
    axis.legend(loc="lower right")
    return _save(config, figure, output, "figure_07_bciciv_2a_roc")


def _tsne(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    datasets = config["analysis"]["representative_datasets"]["tsne"]
    available = [(name, analysis / f"tsne_{name}.csv") for name in datasets]
    available = [(name, path) for name, path in available if path.exists()]
    if not available:
        return []
    figure, axes = plt.subplots(len(available), 2, figsize=(13, 5 * len(available)), squeeze=False)
    for row, (dataset_name, path) in enumerate(available):
        frame = pd.read_csv(path)
        for column, representation in enumerate(("raw", "latent")):
            axis = axes[row, column]
            scatter = axis.scatter(
                frame[f"{representation}_x"],
                frame[f"{representation}_y"],
                c=frame["subject"],
                cmap="tab20",
                s=8,
                alpha=0.75,
            )
            axis.set_title(f"{dataset_name} · {representation}")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(scatter, ax=axis, label="Subject")
    figure.tight_layout()
    return _save(config, figure, output, "figure_08_raw_latent_tsne")


def _filtered_correlations(
    config: dict[str, Any], analysis: Path, output: Path
) -> list[Path]:
    selection_path = analysis / "feature_selection.csv"
    if not selection_path.exists():
        return []
    selections = pd.read_csv(selection_path)
    subjects = [int(value) for value in config["analysis"]["correlation"]["selected_subjects"]]
    thresholds = [float(value) for value in config["analysis"]["correlation"]["thresholds"]]
    figure, axes = plt.subplots(len(subjects), len(thresholds), figsize=(15, 4 * len(subjects)), squeeze=False)
    image = None
    for row, subject in enumerate(subjects):
        correlation = np.load(analysis / "correlation" / f"subject_{subject:02d}.npy")
        for column, threshold in enumerate(thresholds):
            record = selections[
                (selections["subject"] == subject)
                & np.isclose(selections["threshold"], threshold)
            ].iloc[0]
            retained = np.fromstring(str(record["retained_indices_zero_based"]), sep=" ", dtype=int)
            filtered = correlation[np.ix_(retained, retained)]
            image = axes[row, column].imshow(
                filtered, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto"
            )
            axes[row, column].set_title(f"A{subject:02d} · r={threshold:.2f} · {len(retained)} features")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.65)
    return _save(config, figure, output, "figure_09_filtered_correlations")


def _anova(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    path = analysis / "anova.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    thresholds = sorted(frame["threshold"].unique(), reverse=True)
    common = set(frame[frame["threshold"] == thresholds[0]]["feature_id_one_based"])
    for threshold in thresholds[1:]:
        common &= set(frame[frame["threshold"] == threshold]["feature_id_one_based"])
    features = sorted(common)
    if not features:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    positions = np.arange(len(features), dtype=float)
    width = 0.8 / len(thresholds)
    for index, threshold in enumerate(thresholds):
        selected = frame[
            np.isclose(frame["threshold"], threshold)
            & frame["feature_id_one_based"].isin(features)
        ].set_index("feature_id_one_based").loc[features]
        offset = (index - (len(thresholds) - 1) / 2) * width
        axes[0].bar(positions + offset, selected["f_statistic"], width, label=f"r={threshold:.2f}")
        axes[1].bar(positions + offset, selected["p_value"], width, label=f"r={threshold:.2f}")
    for axis in axes:
        axis.set_xticks(positions, [f"F{feature}" for feature in features])
        axis.legend()
    axes[0].set_ylabel("F statistic")
    axes[1].set_ylabel("p value")
    axes[1].set_yscale("log")
    for level in (0.05, 0.01, 0.001):
        axes[1].axhline(level, ls="--", lw=1)
    figure.tight_layout()
    return _save(config, figure, output, "figure_10_anova")


def _hierarchy(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    path = analysis / "hierarchy_linkage.npy"
    if not path.exists():
        return []
    matrix = np.load(path)
    threshold = float(config["analysis"]["hierarchy"]["distance_threshold"])
    figure, axis = plt.subplots(figsize=(13, 6))
    dendrogram(matrix, color_threshold=threshold, no_labels=True, ax=axis)
    axis.axhline(threshold, color="black", ls="--", lw=1.5)
    axis.set(xlabel="Semantic feature sample", ylabel="Ward distance")
    axis.set_title("Hierarchical clustering of BCICIV_2a semantic features")
    return _save(config, figure, output, "figure_11_hierarchical_clustering")


def _pca(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    path = analysis / "pca_kmeans.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    figure, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(frame["pc1"], frame["pc2"], c=frame["cluster"], cmap="tab10", s=10)
    axis.set(xlabel="Principal component 1", ylabel="Principal component 2")
    axis.set_title("PCA of latent semantic features with K-means clusters")
    figure.colorbar(scatter, ax=axis, label="Cluster")
    return _save(config, figure, output, "figure_12_pca_kmeans")


def _silhouette(config: dict[str, Any], analysis: Path, output: Path) -> list[Path]:
    path = analysis / "silhouette.csv"
    summary_path = analysis / "clustering_summary.json"
    if not path.exists() or not summary_path.exists():
        return []
    frame = pd.read_csv(path)
    summary = pd.read_json(summary_path, typ="series")
    figure, axis = plt.subplots(figsize=(8, 7))
    lower = 10
    for cluster in sorted(frame["cluster"].unique()):
        values = np.sort(frame.loc[frame["cluster"] == cluster, "silhouette"].to_numpy())
        upper = lower + len(values)
        axis.fill_betweenx(np.arange(lower, upper), 0, values, alpha=0.75)
        axis.text(-0.05, lower + len(values) / 2, str(int(cluster)))
        lower = upper + 10
    axis.axvline(float(summary["silhouette_score"]), color="red", ls="--")
    axis.set(xlabel="Silhouette coefficient", ylabel="Cluster")
    axis.set_yticks([])
    axis.set_title("Silhouette analysis of semantic-feature clusters")
    return _save(config, figure, output, "figure_13_silhouette")


def _ablation(config: dict[str, Any], output: Path) -> list[Path]:
    dataset_name = str(config["analysis"]["ablation"]["dataset"])
    root = run_directory(config, dataset_name) / "ablation"
    summary_path = root / "summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    figure, axes = plt.subplots(1, len(summary), figsize=(6 * len(summary), 5), squeeze=False)
    for axis, row in zip(axes[0], summary.itertuples(index=False), strict=True):
        errors = np.load(root / row.configuration / "reconstruction_errors.npy")
        axis.hist(errors, bins=30, color="#2563eb", alpha=0.85)
        axis.set_title(
            f"{int(row.total_layers)} layers, {int(row.attention_heads)} heads\nmean={row.mean_mse:.4f}"
        )
        axis.set_xlabel("Reconstruction MSE")
        axis.set_ylabel("Frequency")
    figure.tight_layout()
    return _save(config, figure, output, "figure_14_ablation")


def make_figures(config: dict[str, Any]) -> Path:
    plt.style.use(str(config["figures"]["style"]))
    root = generated_directory(config)
    analysis = root / "analysis"
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    saved.extend(_architecture(config, output))
    saved.extend(_learning_curves(config, output))
    saved.extend(_reconstruction(config, analysis, output))
    saved.extend(_correlation(config, analysis, output))
    saved.extend(_roc(config, analysis, output))
    saved.extend(_tsne(config, analysis, output))
    saved.extend(_filtered_correlations(config, analysis, output))
    saved.extend(_anova(config, analysis, output))
    saved.extend(_hierarchy(config, analysis, output))
    saved.extend(_pca(config, analysis, output))
    saved.extend(_silhouette(config, analysis, output))
    saved.extend(_ablation(config, output))
    records = []
    for path in saved:
        parameters = {
            "dpi": int(config["figures"]["dpi"]),
            "format": path.suffix.lstrip("."),
            "seed": int(config["seed"]),
            "style": str(config["figures"]["style"]),
        }
        records.append(
            {
                "file": path.relative_to(root).as_posix(),
                "source_data": json.dumps(_figure_sources(config, root, path), separators=(",", ":")),
                "plotting_parameters": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(
        records,
        columns=["file", "source_data", "plotting_parameters", "sha256"],
    ).to_csv(output / "manifest.csv", index=False)
    build_publication_outputs(config)
    return output
