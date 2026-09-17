import numpy as np

from useeg.metrics import (
    aggregate_probabilities_by_group,
    anova_feature_statistics,
    binary_classification_metrics,
    feature_correlation_matrix,
    greedy_correlation_filter,
    multiclass_classification_metrics,
    per_window_mse,
    reconstruction_metrics,
    subject_accuracy,
    subject_semantic_vectors,
    summarize_folds,
)


def test_reconstruction_metrics_are_per_window() -> None:
    original = np.zeros((2, 2, 2), dtype=np.float32)
    reconstructed = np.asarray(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    errors = per_window_mse(original, reconstructed)
    metrics = reconstruction_metrics(original, reconstructed, threshold=3.0)
    np.testing.assert_allclose(errors, [1.0, 4.0])
    assert metrics["mse"] == 2.5
    assert metrics["passes_threshold"]


def test_binary_and_multiclass_paper_metrics() -> None:
    binary = binary_classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )
    multiclass = multiclass_classification_metrics(
        [0, 1, 2, 0, 1, 2],
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
    )
    assert binary["accuracy_percent"] == 100.0
    assert binary["auc_percent"] == 100.0
    assert multiclass["accuracy_percent"] == 100.0
    assert multiclass["macro_ovr_auc_percent"] == 100.0
    assert set(multiclass["per_class_auc"]) == {0, 1, 2}


def test_group_aggregation_and_subject_accuracy() -> None:
    group_ids, targets, probabilities = aggregate_probabilities_by_group(
        [0.1, 0.3, 0.7, 0.9],
        [0, 0, 1, 1],
        ["a", "a", "b", "b"],
    )
    np.testing.assert_array_equal(group_ids, ["a", "b"])
    np.testing.assert_array_equal(targets, [0, 1])
    np.testing.assert_allclose(probabilities, [0.2, 0.8])
    accuracies = subject_accuracy(
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 2, 2],
    )
    assert accuracies == {1: 100.0, 2: 50.0}


def test_subject_vectors_correlation_filter_and_anova() -> None:
    features = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 1.0],
            [5.0, 5.0, 0.0],
        ]
    )
    subjects, vectors = subject_semantic_vectors(features, [1, 1, 1, 2, 2, 2])
    correlation = feature_correlation_matrix(features)
    retained, filtered = greedy_correlation_filter(features, threshold=0.8)
    f_statistics, p_values = anova_feature_statistics(features, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(subjects, [1, 2])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert correlation.shape == (3, 3)
    np.testing.assert_array_equal(retained, [0, 2])
    assert filtered.shape == (6, 2)
    assert f_statistics.shape == (3,)
    assert p_values.shape == (3,)


def test_fold_summary() -> None:
    summary = summarize_folds(
        [
            {"accuracy_percent": 80.0, "auc_percent": 90.0},
            {"accuracy_percent": 100.0, "auc_percent": 94.0},
        ]
    )
    assert summary["accuracy_percent"]["mean"] == 90.0
    assert summary["auc_percent"]["standard_deviation"] == 2.0
