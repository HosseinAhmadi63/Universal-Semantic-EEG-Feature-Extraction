import numpy as np

from useeg.splits import (
    classification_folds,
    random_train_validation_split,
    stratified_train_validation_split,
)


def test_random_split_is_deterministic_and_complete() -> None:
    first = random_train_validation_split(20, 0.1, 42)
    second = random_train_validation_split(20, 0.1, 42)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert set(np.concatenate(first).tolist()) == set(range(20))


def test_stratified_inner_split_preserves_classes() -> None:
    labels = np.tile(np.arange(4), 10)
    train, validation = stratified_train_validation_split(
        np.arange(len(labels)), labels, 0.2, 42
    )
    assert set(labels[train]) == {0, 1, 2, 3}
    assert set(labels[validation]) == {0, 1, 2, 3}


def test_mi_folds_hold_out_each_subject() -> None:
    labels = np.tile([0, 1], 6)
    subjects = np.repeat([1, 2, 3], 4)
    folds = classification_folds("mi", labels, subjects, 8, 42)
    assert len(folds) == 3
    for fold in folds:
        held_out = fold["held_out_subject"]
        assert set(subjects[fold["test"]]) == {held_out}
        assert held_out not in set(subjects[fold["train"]])


def test_stratified_folds_cover_every_sample_once() -> None:
    labels = np.tile([0, 1], 16)
    subjects = np.repeat(np.arange(1, 9), 4)
    folds = classification_folds("erp", labels, subjects, 8, 42)
    tested = np.concatenate([fold["test"] for fold in folds])
    assert np.array_equal(np.sort(tested), np.arange(len(labels)))
