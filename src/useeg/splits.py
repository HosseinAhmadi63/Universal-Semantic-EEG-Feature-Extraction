from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def random_train_validation_split(
    size: int, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(size)
    val_size = min(size - 1, max(1, int(round(size * val_fraction))))
    validation = np.sort(permutation[:val_size].astype(np.int64))
    training = np.sort(permutation[val_size:].astype(np.int64))
    return training, validation


def stratified_train_validation_split(
    indices: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[selected]
    if len(selected) < 2:
        raise ValueError("at least two selected samples are required")
    counts = np.bincount(y)
    stratify = y if len(counts) > 1 and np.all(counts[counts > 0] >= 2) else None
    training, validation = train_test_split(
        selected,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.sort(training.astype(np.int64)), np.sort(validation.astype(np.int64))


def classification_folds(
    paradigm: str,
    labels: np.ndarray,
    subjects: np.ndarray,
    folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subjects)
    if len(y) != len(subject_ids):
        raise ValueError("labels and subjects must have equal lengths")
    indices = np.arange(len(y), dtype=np.int64)
    result: list[dict[str, Any]] = []
    if paradigm == "mi":
        for fold_index, subject in enumerate(sorted(np.unique(subject_ids).tolist()), 1):
            test = indices[subject_ids == subject]
            train = indices[subject_ids != subject]
            if not len(train) or not len(test):
                raise ValueError("each MI fold requires train and test samples")
            result.append(
                {
                    "fold": fold_index,
                    "name": f"subject_{int(subject):02d}",
                    "held_out_subject": int(subject),
                    "train": train,
                    "test": test,
                }
            )
        return result
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_index, (train, test) in enumerate(splitter.split(indices, y), 1):
        result.append(
            {
                "fold": fold_index,
                "name": f"fold_{fold_index:02d}",
                "held_out_subject": None,
                "train": train.astype(np.int64),
                "test": test.astype(np.int64),
            }
        )
    return result
