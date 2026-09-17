from __future__ import annotations

import copy
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset

from .models import BinaryFocalLoss, HierarchicalEEGAutoencoder, SemanticFeatureClassifier


class ArrayDataset(Dataset[Any]):
    def __init__(
        self,
        inputs: Any,
        targets: Any | None = None,
        input_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not hasattr(inputs, "__len__") or not hasattr(inputs, "__getitem__"):
            raise TypeError("inputs must support indexing")
        if targets is not None and len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        self.inputs = inputs
        self.targets = targets
        self.input_dtype = input_dtype

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Any:
        input_value = self.inputs[index]
        if isinstance(input_value, Tensor):
            inputs = input_value.to(dtype=self.input_dtype)
        else:
            input_array = np.asarray(input_value)
            if not input_array.flags.writeable:
                input_array = input_array.copy()
            inputs = torch.as_tensor(input_array, dtype=self.input_dtype)
        if self.targets is None:
            return inputs
        target_value = self.targets[index]
        targets = (
            target_value
            if isinstance(target_value, Tensor)
            else torch.as_tensor(np.asarray(target_value))
        )
        return inputs, targets


def set_deterministic(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return resolved


def _merge_config(config: Mapping[str, Any] | None, section: str) -> dict[str, Any]:
    values = dict(config or {})
    nested = values.get(section)
    if isinstance(nested, Mapping):
        values.update(nested)
    training_values = values.get("training")
    if isinstance(training_values, Mapping):
        values.update(training_values)
    optimizer_values = values.get("optimizer")
    if isinstance(optimizer_values, Mapping):
        for key, value in optimizer_values.items():
            values.setdefault(key, value)
        if "epsilon" in optimizer_values:
            values["optimizer_epsilon"] = optimizer_values["epsilon"]
    scheduler_values = values.get("scheduler")
    if isinstance(scheduler_values, Mapping):
        if "factor" in scheduler_values:
            values["lr_factor"] = scheduler_values["factor"]
        if "patience" in scheduler_values:
            values["lr_patience"] = scheduler_values["patience"]
        if "threshold" in scheduler_values:
            values["scheduler_threshold"] = scheduler_values["threshold"]
        if "threshold_mode" in scheduler_values:
            values["scheduler_threshold_mode"] = scheduler_values["threshold_mode"]
        if "cooldown" in scheduler_values:
            values["cooldown"] = scheduler_values["cooldown"]
        if "min_lr" in scheduler_values:
            values["min_lr"] = scheduler_values["min_lr"]
    early_values = values.get("early_stopping")
    if isinstance(early_values, Mapping):
        if "patience" in early_values:
            values["early_stopping_patience"] = early_values["patience"]
        if "min_delta" in early_values:
            values["min_delta"] = early_values["min_delta"]
    return values


def _first(config: Mapping[str, Any], names: Sequence[str], default: Any) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


def as_dataset(data: Any, targets: Any | None = None) -> Dataset[Any]:
    if isinstance(data, Dataset):
        if targets is not None:
            raise ValueError("targets must be embedded in a torch Dataset")
        return data
    return ArrayDataset(data, targets)


def make_data_loader(
    data: Any,
    targets: Any | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        as_dataset(data, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )


def split_dataset(
    data: Any,
    targets: Any | None = None,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Subset[Any], Subset[Any]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    dataset = as_dataset(data, targets)
    sample_count = len(dataset)
    if sample_count < 2:
        raise ValueError("at least two samples are required")
    validation_count = max(1, int(round(sample_count * validation_fraction)))
    validation_count = min(validation_count, sample_count - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator).tolist()
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def _mapping_value(batch: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def _unpack_batch(batch: Any) -> tuple[Any, Any | None, Any | None]:
    if isinstance(batch, Mapping):
        inputs = _mapping_value(batch, ("inputs", "input", "x", "eeg", "features", "latent"))
        targets = _mapping_value(batch, ("targets", "target", "y", "label", "labels"))
        mask = _mapping_value(batch, ("key_padding_mask", "padding_mask", "mask"))
        if inputs is None:
            raise KeyError("batch mapping does not contain inputs")
        return inputs, targets, mask
    if isinstance(batch, tuple | list):
        if len(batch) == 0:
            raise ValueError("empty batch")
        return batch[0], batch[1] if len(batch) > 1 else None, None
    return batch, None, None


def _move_inputs(
    inputs: Any,
    mask: Any | None,
    device: torch.device,
) -> tuple[Tensor, Tensor | None]:
    input_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    mask_tensor = None
    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    return input_tensor, mask_tensor


def _state_to_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    epoch: int | None = None,
    history: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": _state_to_cpu(model),
        "epoch": epoch,
        "history": dict(history or {}),
        "extra": dict(extra or {}),
    }
    if hasattr(model, "configuration"):
        payload["model_config"] = model.configuration()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError("invalid checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return dict(payload)


def _training_components(
    model: nn.Module,
    values: Mapping[str, Any],
) -> tuple[Adam, ReduceLROnPlateau]:
    learning_rate = float(_first(values, ("learning_rate", "lr"), 1e-4))
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(values.get("betas", (0.9, 0.999))),
        eps=float(_first(values, ("optimizer_epsilon", "epsilon", "eps"), 1e-8)),
        weight_decay=float(values.get("weight_decay", 0.0)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(_first(values, ("lr_factor", "factor"), 0.5)),
        patience=int(_first(values, ("lr_patience", "scheduler_patience"), 3)),
        threshold=float(values.get("scheduler_threshold", 1e-4)),
        threshold_mode=str(values.get("scheduler_threshold_mode", "rel")),
        cooldown=int(values.get("cooldown", 0)),
        min_lr=float(values.get("min_lr", 1e-7)),
    )
    return optimizer, scheduler


def _loader_settings(
    values: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "batch_size": int(values.get("batch_size", 64)),
        "num_workers": int(values.get("num_workers", 0)),
        "pin_memory": bool(values.get("pin_memory", device.type == "cuda")),
    }


def _autoencoder_epoch(
    model: HierarchicalEEGAutoencoder,
    loader: DataLoader[Any],
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        inputs, _, mask = _unpack_batch(batch)
        input_tensor, mask_tensor = _move_inputs(inputs, mask, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstructed = model(input_tensor, key_padding_mask=mask_tensor)
            if not isinstance(reconstructed, Tensor):
                raise TypeError("autoencoder returned an invalid value")
            loss = criterion(reconstructed, input_tensor)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = input_tensor.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("data loader is empty")
    return total_loss / total_samples


def train_autoencoder(
    model: HierarchicalEEGAutoencoder,
    train_data: Any,
    validation_data: Any,
    config: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    values = _merge_config(config, "autoencoder")
    seed = int(values.get("seed", 42))
    deterministic = bool(values.get("deterministic", True))
    set_deterministic(seed, deterministic)
    resolved_device = resolve_device(device if device is not None else values.get("device"))
    model.to(resolved_device)
    settings = _loader_settings(values, resolved_device)
    train_loader = make_data_loader(
        train_data,
        batch_size=settings["batch_size"],
        shuffle=True,
        seed=seed,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    validation_loader = make_data_loader(
        validation_data,
        batch_size=settings["batch_size"],
        shuffle=False,
        seed=seed,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    optimizer, scheduler = _training_components(model, values)
    criterion = nn.MSELoss(reduction="mean")
    max_epochs = int(_first(values, ("max_epochs", "epochs"), 100))
    patience = int(_first(values, ("early_stopping_patience", "patience"), 5))
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and early-stopping patience must be positive")
    min_delta = float(values.get("min_delta", 0.0))
    history: dict[str, Any] = {"train_loss": [], "validation_loss": [], "learning_rate": []}
    best_state = _state_to_cpu(model)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_scheduler_state = copy.deepcopy(scheduler.state_dict())
    best_validation_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    stopped_epoch = max_epochs - 1
    for epoch in range(max_epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _autoencoder_epoch(
            model,
            train_loader,
            resolved_device,
            criterion,
            optimizer,
        )
        with torch.no_grad():
            validation_loss = _autoencoder_epoch(
                model,
                validation_loader,
                resolved_device,
                criterion,
                None,
            )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["learning_rate"].append(learning_rate)
        improved = validation_loss < best_validation_loss - min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler.step(validation_loss)
        if improved:
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scheduler_state = copy.deepcopy(scheduler.state_dict())
        if bad_epochs >= patience:
            stopped_epoch = epoch
            break
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)
    scheduler.load_state_dict(best_scheduler_state)
    model.to(resolved_device)
    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = best_validation_loss
    history["stopped_epoch"] = stopped_epoch
    history["epochs_ran"] = len(history["train_loss"])
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best_epoch,
            history=history,
            extra={"best_validation_loss": best_validation_loss},
        )
    return history


def reconstruction_errors(
    model: HierarchicalEEGAutoencoder,
    data: Any,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    num_workers: int = 0,
) -> np.ndarray:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    loader = make_data_loader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs, _, mask = _unpack_batch(batch)
            input_tensor, mask_tensor = _move_inputs(inputs, mask, resolved_device)
            reconstructed = model(input_tensor, key_padding_mask=mask_tensor)
            if not isinstance(reconstructed, Tensor):
                raise TypeError("autoencoder returned an invalid value")
            batch_errors = (reconstructed - input_tensor).pow(2).flatten(1).mean(dim=1)
            errors.append(batch_errors.cpu().numpy())
    model.train(was_training)
    if not errors:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(errors).astype(np.float64, copy=False)


def evaluate_autoencoder(
    model: HierarchicalEEGAutoencoder,
    data: Any,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    threshold: float = 0.01,
) -> dict[str, Any]:
    errors = reconstruction_errors(model, data, batch_size=batch_size, device=device)
    if errors.size == 0:
        raise ValueError("evaluation data is empty")
    return {
        "mse": float(errors.mean()),
        "reconstruction_errors": errors,
        "threshold": float(threshold),
        "passes_threshold": bool(errors.mean() < threshold),
    }


def extract_semantic_features(
    model: HierarchicalEEGAutoencoder,
    data: Any,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    include_latent_maps: bool = True,
    normalize_window_features: bool = False,
) -> dict[str, np.ndarray]:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    loader = make_data_loader(
        data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=resolved_device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    latent_maps: list[np.ndarray] = []
    window_features: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs, _, mask = _unpack_batch(batch)
            input_tensor, mask_tensor = _move_inputs(inputs, mask, resolved_device)
            latent = model.encode(input_tensor, key_padding_mask=mask_tensor)
            if mask_tensor is None:
                features = latent.mean(dim=1)
            else:
                valid = (~mask_tensor).to(dtype=latent.dtype).unsqueeze(-1)
                features = (latent * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            if normalize_window_features:
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            if include_latent_maps:
                latent_maps.append(latent.cpu().numpy())
            window_features.append(features.cpu().numpy())
    model.train(was_training)
    result = {
        "window_features": np.concatenate(window_features) if window_features else np.empty((0, model.d_model)),
    }
    if include_latent_maps:
        result["latent_maps"] = (
            np.concatenate(latent_maps)
            if latent_maps
            else np.empty((0, model.window_size, model.d_model))
        )
    return result


def _classifier_loss(
    model: SemanticFeatureClassifier,
    values: Mapping[str, Any],
    training_dataset: Dataset[Any],
) -> nn.Module:
    if not model.binary:
        return nn.CrossEntropyLoss()
    alpha_value = values.get("focal_alpha")
    balance_tokens = {"balanced", "training_fold_negative_fraction", "negative_fraction"}
    if alpha_value is None or str(alpha_value).lower() in balance_tokens:
        label_array = _dataset_targets(training_dataset).astype(np.float64, copy=False)
        if label_array.size == 0:
            raise ValueError("classifier training data is empty")
        positive_count = float((label_array == 1).sum())
        negative_count = float((label_array == 0).sum())
        alpha = negative_count / max(positive_count + negative_count, 1.0)
        if positive_count == 0.0 or negative_count == 0.0:
            alpha = 0.5
    else:
        alpha = float(alpha_value)
    return BinaryFocalLoss(
        alpha=alpha,
        gamma=float(values.get("focal_gamma", 2.0)),
        reduction="mean",
    )


def _dataset_targets(dataset: Dataset[Any]) -> np.ndarray:
    if isinstance(dataset, ArrayDataset) and dataset.targets is not None:
        return np.asarray(dataset.targets).reshape(-1)
    if isinstance(dataset, Subset):
        parent_targets = _dataset_targets(dataset.dataset)
        return parent_targets[np.asarray(dataset.indices, dtype=np.int64)]
    tensors = getattr(dataset, "tensors", None)
    if isinstance(tensors, tuple) and len(tensors) > 1:
        return _array_from_tensor(tensors[1]).reshape(-1)
    labels: list[np.ndarray] = []
    for index in range(len(dataset)):
        _, target, _ = _unpack_batch(dataset[index])
        if target is None:
            raise ValueError("dataset does not contain targets")
        labels.append(_array_from_tensor(target).reshape(-1))
    if not labels:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(labels)


def _array_from_tensor(values: Any) -> np.ndarray:
    if isinstance(values, Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _classifier_epoch(
    model: SemanticFeatureClassifier,
    loader: DataLoader[Any],
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        inputs, targets, _ = _unpack_batch(batch)
        if targets is None:
            raise ValueError("classifier batch does not contain targets")
        input_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        if model.binary:
            target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device).reshape(-1, 1)
        else:
            target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device).reshape(-1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(input_tensor)
            loss = criterion(logits, target_tensor)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = input_tensor.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("data loader is empty")
    return total_loss / total_samples


def train_classifier(
    model: SemanticFeatureClassifier,
    train_data: Any,
    train_targets: Any | None,
    validation_data: Any,
    validation_targets: Any | None,
    config: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    values = _merge_config(config, "classifier")
    seed = int(values.get("seed", 42))
    deterministic = bool(values.get("deterministic", True))
    set_deterministic(seed, deterministic)
    resolved_device = resolve_device(device if device is not None else values.get("device"))
    model.to(resolved_device)
    settings = _loader_settings(values, resolved_device)
    training_dataset = as_dataset(train_data, train_targets)
    validation_dataset = as_dataset(validation_data, validation_targets)
    train_loader = make_data_loader(
        training_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        seed=seed,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    validation_loader = make_data_loader(
        validation_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        seed=seed,
        num_workers=settings["num_workers"],
        pin_memory=settings["pin_memory"],
    )
    optimizer, scheduler = _training_components(model, values)
    criterion = _classifier_loss(model, values, training_dataset)
    max_epochs = int(_first(values, ("max_epochs", "epochs"), 100))
    patience = int(_first(values, ("early_stopping_patience", "patience"), 5))
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and early-stopping patience must be positive")
    min_delta = float(values.get("min_delta", 0.0))
    history: dict[str, Any] = {"train_loss": [], "validation_loss": [], "learning_rate": []}
    best_state = _state_to_cpu(model)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_scheduler_state = copy.deepcopy(scheduler.state_dict())
    best_validation_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    stopped_epoch = max_epochs - 1
    for epoch in range(max_epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _classifier_epoch(
            model,
            train_loader,
            resolved_device,
            criterion,
            optimizer,
        )
        with torch.no_grad():
            validation_loss = _classifier_epoch(
                model,
                validation_loader,
                resolved_device,
                criterion,
                None,
            )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["learning_rate"].append(learning_rate)
        improved = validation_loss < best_validation_loss - min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler.step(validation_loss)
        if improved:
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scheduler_state = copy.deepcopy(scheduler.state_dict())
        if bad_epochs >= patience:
            stopped_epoch = epoch
            break
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)
    scheduler.load_state_dict(best_scheduler_state)
    model.to(resolved_device)
    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = best_validation_loss
    history["stopped_epoch"] = stopped_epoch
    history["epochs_ran"] = len(history["train_loss"])
    if isinstance(criterion, BinaryFocalLoss):
        history["focal_alpha"] = criterion.alpha
        history["focal_gamma"] = criterion.gamma
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best_epoch,
            history=history,
            extra={"best_validation_loss": best_validation_loss},
        )
    return history


def predict_classifier(
    model: SemanticFeatureClassifier,
    data: Any,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    loader = make_data_loader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    logits_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs, _, _ = _unpack_batch(batch)
            input_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=resolved_device)
            logits = model(input_tensor)
            probabilities = (
                torch.sigmoid(logits).reshape(-1)
                if model.binary
                else torch.softmax(logits, dim=1)
            )
            logits_batches.append(logits.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())
    model.train(was_training)
    if not logits_batches:
        output_width = 1 if model.binary else model.num_classes
        return {
            "logits": np.empty((0, output_width), dtype=np.float32),
            "probabilities": np.empty(0 if model.binary else (0, output_width), dtype=np.float32),
            "predictions": np.empty(0, dtype=np.int64),
        }
    logits_array = np.concatenate(logits_batches)
    probabilities_array = np.concatenate(probability_batches)
    predictions = (
        (probabilities_array >= 0.5).astype(np.int64)
        if model.binary
        else probabilities_array.argmax(axis=1).astype(np.int64)
    )
    return {
        "logits": logits_array,
        "probabilities": probabilities_array,
        "predictions": predictions,
    }


def evaluate_classifier(
    model: SemanticFeatureClassifier,
    data: Any,
    targets: Any | None = None,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    from .metrics import binary_classification_metrics, multiclass_classification_metrics

    dataset = as_dataset(data, targets)
    prediction = predict_classifier(model, dataset, batch_size=batch_size, device=device)
    y_true = _dataset_targets(dataset).astype(np.int64, copy=False)
    if model.binary:
        metrics = binary_classification_metrics(y_true, prediction["probabilities"])
    else:
        metrics = multiclass_classification_metrics(y_true, prediction["probabilities"])
    return {**prediction, **metrics}
