from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    silhouette_samples,
    silhouette_score,
)


def _array(values: Any, dtype: Any | None = None) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def per_window_mse(original: Any, reconstructed: Any) -> np.ndarray:
    original_array = _array(original, np.float64)
    reconstructed_array = _array(reconstructed, np.float64)
    if original_array.shape != reconstructed_array.shape:
        raise ValueError("original and reconstructed arrays must have the same shape")
    if original_array.ndim < 2:
        raise ValueError("inputs must include a sample axis and at least one feature axis")
    axes = tuple(range(1, original_array.ndim))
    return np.mean(np.square(original_array - reconstructed_array), axis=axes)


def reconstruction_metrics(
    original: Any,
    reconstructed: Any,
    threshold: float = 0.01,
) -> dict[str, Any]:
    errors = per_window_mse(original, reconstructed)
    if errors.size == 0:
        raise ValueError("inputs are empty")
    mean_error = float(errors.mean())
    return {
        "mse": mean_error,
        "mean": mean_error,
        "standard_deviation": float(errors.std(ddof=0)),
        "median": float(np.median(errors)),
        "minimum": float(errors.min()),
        "maximum": float(errors.max()),
        "threshold": float(threshold),
        "passes_threshold": bool(mean_error < threshold),
        "per_window_mse": errors,
    }


def accuracy_percent(y_true: Any, y_pred: Any) -> float:
    true_array = _array(y_true).reshape(-1)
    prediction_array = _array(y_pred).reshape(-1)
    if true_array.shape != prediction_array.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(100.0 * accuracy_score(true_array, prediction_array))


def binary_classification_metrics(
    y_true: Any,
    probabilities: Any,
    threshold: float = 0.5,
) -> dict[str, Any]:
    true_array = _array(y_true, np.int64).reshape(-1)
    probability_array = _array(probabilities, np.float64)
    if probability_array.ndim == 2:
        if probability_array.shape[1] != 2:
            raise ValueError("binary probability matrices must have two columns")
        probability_array = probability_array[:, 1]
    probability_array = probability_array.reshape(-1)
    if true_array.shape != probability_array.shape:
        raise ValueError("targets and probabilities must have the same sample count")
    if np.unique(true_array).size != 2:
        raise ValueError("ROC-AUC requires both binary classes")
    predictions = (probability_array >= threshold).astype(np.int64)
    auc = float(roc_auc_score(true_array, probability_array))
    accuracy = float(accuracy_score(true_array, predictions))
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        true_array,
        probability_array,
    )
    return {
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "auc": auc,
        "auc_percent": 100.0 * auc,
        "threshold": float(threshold),
        "predictions": predictions,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "roc_thresholds": roc_thresholds,
    }


def multiclass_classification_metrics(
    y_true: Any,
    probabilities: Any,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    true_array = _array(y_true, np.int64).reshape(-1)
    probability_array = _array(probabilities, np.float64)
    if probability_array.ndim != 2:
        raise ValueError("multiclass probabilities must be a matrix")
    if true_array.shape[0] != probability_array.shape[0]:
        raise ValueError("targets and probabilities must have the same sample count")
    class_labels = np.asarray(
        list(labels) if labels is not None else list(range(probability_array.shape[1])),
        dtype=np.int64,
    )
    if class_labels.size != probability_array.shape[1]:
        raise ValueError("labels must match the probability columns")
    predictions = class_labels[probability_array.argmax(axis=1)]
    accuracy = float(accuracy_score(true_array, predictions))
    per_class_auc: dict[int, float] = {}
    roc_curves: dict[int, dict[str, np.ndarray]] = {}
    for column, label in enumerate(class_labels):
        binary_targets = (true_array == label).astype(np.int64)
        if np.unique(binary_targets).size < 2:
            auc = float("nan")
            false_positive_rate = np.asarray([], dtype=np.float64)
            true_positive_rate = np.asarray([], dtype=np.float64)
            thresholds = np.asarray([], dtype=np.float64)
        else:
            auc = float(roc_auc_score(binary_targets, probability_array[:, column]))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(
                binary_targets,
                probability_array[:, column],
            )
        per_class_auc[int(label)] = auc
        roc_curves[int(label)] = {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "thresholds": thresholds,
        }
    finite_auc = np.asarray(
        [value for value in per_class_auc.values() if np.isfinite(value)],
        dtype=np.float64,
    )
    macro_auc = float(finite_auc.mean()) if finite_auc.size else float("nan")
    return {
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "macro_ovr_auc": macro_auc,
        "macro_ovr_auc_percent": 100.0 * macro_auc,
        "per_class_auc": per_class_auc,
        "per_class_auc_percent": {
            label: 100.0 * value for label, value in per_class_auc.items()
        },
        "predictions": predictions,
        "roc_curves": roc_curves,
    }


def subject_accuracy(
    y_true: Any,
    y_pred: Any,
    subject_ids: Any,
) -> dict[Any, float]:
    true_array = _array(y_true).reshape(-1)
    prediction_array = _array(y_pred).reshape(-1)
    subjects = _array(subject_ids).reshape(-1)
    if not (true_array.size == prediction_array.size == subjects.size):
        raise ValueError("all inputs must have the same sample count")
    return {
        subject: accuracy_percent(true_array[subjects == subject], prediction_array[subjects == subject])
        for subject in np.unique(subjects)
    }


def aggregate_probabilities_by_group(
    probabilities: Any,
    y_true: Any,
    group_ids: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability_array = _array(probabilities, np.float64)
    true_array = _array(y_true, np.int64).reshape(-1)
    groups = _array(group_ids).reshape(-1)
    if probability_array.shape[0] != true_array.size or groups.size != true_array.size:
        raise ValueError("all inputs must have the same sample count")
    unique_groups, first_indices = np.unique(groups, return_index=True)
    order = np.argsort(first_indices)
    unique_groups = unique_groups[order]
    grouped_probabilities: list[np.ndarray] = []
    grouped_targets: list[int] = []
    for group in unique_groups:
        selector = groups == group
        labels = np.unique(true_array[selector])
        if labels.size != 1:
            raise ValueError("each group must contain one class label")
        grouped_probabilities.append(np.mean(probability_array[selector], axis=0))
        grouped_targets.append(int(labels[0]))
    return (
        unique_groups,
        np.asarray(grouped_targets, dtype=np.int64),
        np.asarray(grouped_probabilities, dtype=np.float64),
    )


def subject_semantic_vectors(
    window_features: Any,
    subject_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    features = _array(window_features, np.float64)
    subjects = _array(subject_ids).reshape(-1)
    if features.ndim != 2 or features.shape[0] != subjects.size:
        raise ValueError("window_features must be N x D and align with subject_ids")
    unique_subjects = np.unique(subjects)
    vectors = np.stack([features[subjects == subject].mean(axis=0) for subject in unique_subjects])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, np.finfo(np.float64).eps)
    return unique_subjects, vectors


def feature_correlation_matrix(features: Any) -> np.ndarray:
    feature_array = _array(features, np.float64)
    if feature_array.ndim != 2:
        raise ValueError("features must be an N x D matrix")
    return np.corrcoef(feature_array, rowvar=False)


def greedy_correlation_filter(
    features: Any,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    feature_array = _array(features, np.float64)
    if feature_array.ndim != 2:
        raise ValueError("features must be an N x D matrix")
    correlation = np.abs(feature_correlation_matrix(feature_array))
    retained: list[int] = []
    for candidate in range(feature_array.shape[1]):
        if all(correlation[candidate, previous] <= threshold for previous in retained):
            retained.append(candidate)
    retained_array = np.asarray(retained, dtype=np.int64)
    return retained_array, feature_array[:, retained_array]


def anova_feature_statistics(
    features: Any,
    labels: Any,
) -> tuple[np.ndarray, np.ndarray]:
    feature_array = _array(features, np.float64)
    label_array = _array(labels).reshape(-1)
    if feature_array.ndim != 2 or feature_array.shape[0] != label_array.size:
        raise ValueError("features and labels are not aligned")
    f_statistics, p_values = f_classif(feature_array, label_array)
    return np.asarray(f_statistics), np.asarray(p_values)


def silhouette_metrics(features: Any, cluster_labels: Any) -> dict[str, Any]:
    feature_array = _array(features, np.float64)
    label_array = _array(cluster_labels).reshape(-1)
    if feature_array.ndim != 2 or feature_array.shape[0] != label_array.size:
        raise ValueError("features and cluster_labels are not aligned")
    overall = float(silhouette_score(feature_array, label_array))
    sample_scores = silhouette_samples(feature_array, label_array)
    per_cluster = {
        label: float(sample_scores[label_array == label].mean())
        for label in np.unique(label_array)
    }
    return {
        "silhouette_score": overall,
        "per_cluster": per_cluster,
        "sample_scores": sample_scores,
    }


def summarize_folds(fold_metrics: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if not fold_metrics:
        raise ValueError("fold_metrics is empty")
    common_keys = set.intersection(*(set(fold.keys()) for fold in fold_metrics))
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(common_keys):
        values = np.asarray([fold[key] for fold in fold_metrics], dtype=np.float64)
        if values.ndim == 1 and np.all(np.isfinite(values)):
            summary[key] = {
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    return summary
