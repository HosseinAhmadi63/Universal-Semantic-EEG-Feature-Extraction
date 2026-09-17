from pathlib import Path

import numpy as np
import torch

from useeg.models import HierarchicalEEGAutoencoder, SemanticFeatureClassifier
from useeg.training import (
    evaluate_autoencoder,
    evaluate_classifier,
    extract_semantic_features,
    load_checkpoint,
    predict_classifier,
    reconstruction_errors,
    split_dataset,
    train_autoencoder,
    train_classifier,
)


def _autoencoder() -> HierarchicalEEGAutoencoder:
    return HierarchicalEEGAutoencoder(
        input_channels=2,
        window_size=8,
        d_model=8,
        num_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        ffn_dim=16,
        input_layout="batch_time_channels",
    )


def test_autoencoder_training_features_and_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(12, 8, 2)).astype(np.float32)
    train, validation = split_dataset(data, validation_fraction=0.25, seed=42)
    checkpoint = tmp_path / "autoencoder.pt"
    model = _autoencoder()
    history = train_autoencoder(
        model,
        train,
        validation,
        config={
            "seed": 42,
            "device": "cpu",
            "autoencoder": {
                "batch_size": 4,
                "max_epochs": 2,
                "early_stopping_patience": 2,
                "lr": 1e-3,
                "scheduler": {"factor": 0.5, "patience": 1},
            },
        },
        checkpoint_path=checkpoint,
    )
    assert history["epochs_ran"] == 2
    assert len(history["learning_rate"]) == 2
    assert checkpoint.exists()
    restored = _autoencoder()
    payload = load_checkpoint(checkpoint, restored)
    assert payload["epoch"] == history["best_epoch"]
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(first.detach().cpu(), second.detach().cpu())
    errors = reconstruction_errors(model, validation, batch_size=4, device="cpu")
    evaluation = evaluate_autoencoder(model, validation, batch_size=4, device="cpu")
    features = extract_semantic_features(model, validation, batch_size=4, device="cpu")
    assert errors.shape == (3,)
    assert evaluation["mse"] == np.mean(errors)
    assert features["latent_maps"].shape == (3, 8, 8)
    assert features["window_features"].shape == (3, 8)


def test_binary_classifier_training_prediction_and_evaluation() -> None:
    rng = np.random.default_rng(7)
    train_data = rng.normal(size=(12, 1, 8, 8)).astype(np.float32)
    validation_data = rng.normal(size=(8, 1, 8, 8)).astype(np.float32)
    train_targets = np.asarray([0, 1] * 6, dtype=np.int64)
    validation_targets = np.asarray([0, 1] * 4, dtype=np.int64)
    model = SemanticFeatureClassifier(
        num_classes=2,
        binary=True,
        filters=(4, 6, 8, 10),
        dropout=0.0,
    )
    history = train_classifier(
        model,
        train_data,
        train_targets,
        validation_data,
        validation_targets,
        config={
            "seed": 42,
            "device": "cpu",
            "classifier": {
                "batch_size": 4,
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "focal_alpha": "training_fold_negative_fraction",
            },
        },
    )
    prediction = predict_classifier(model, validation_data, batch_size=4, device="cpu")
    evaluation = evaluate_classifier(
        model,
        validation_data,
        validation_targets,
        batch_size=4,
        device="cpu",
    )
    assert history["epochs_ran"] == 1
    assert history["focal_alpha"] == 0.5
    assert prediction["probabilities"].shape == (8,)
    assert prediction["predictions"].shape == (8,)
    assert 0.0 <= evaluation["auc"] <= 1.0
