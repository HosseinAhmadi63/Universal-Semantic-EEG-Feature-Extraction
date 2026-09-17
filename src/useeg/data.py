from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from .datasets import (
    DatasetSpec,
    create_dataset,
    expected_trials_for_subject,
    get_dataset_spec,
    selected_subjects,
)
from .preprocessing import ArrayRaw, PreprocessingConfig, RunWindows, preprocess_run

CACHE_SCHEMA_VERSION = 1
METADATA_DTYPE = np.dtype(
    [
        ("subject", "U32"),
        ("session", "U64"),
        ("run", "U128"),
        ("trial", "i8"),
        ("window", "i4"),
        ("event_code", "i8"),
        ("label", "U64"),
        ("window_start", "i4"),
        ("valid_samples", "i4"),
    ]
)


@dataclass(frozen=True)
class CachedDataset:
    X: NDArray[np.float32]
    y: NDArray[np.int64]
    metadata: NDArray[Any]
    manifest: dict[str, Any]
    path: Path

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["channel_names"])

    @property
    def label_names(self) -> dict[int, str]:
        return {int(key): str(value) for key, value in self.manifest["label_names"].items()}

    @property
    def run_key(self) -> str:
        return str(self.manifest["run_key"])

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _save_array(path: Path, value: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _versions() -> dict[str, str]:
    return {
        "mne": _package_version("mne"),
        "moabb": _package_version("moabb"),
        "numpy": np.__version__,
        "scikit-learn": _package_version("scikit-learn"),
        "scipy": _package_version("scipy"),
    }


def _digest(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_config(dataset: str | DatasetSpec | Mapping[str, Any]) -> dict[str, Any]:
    return dict(dataset) if isinstance(dataset, Mapping) else {"name": get_dataset_spec(dataset).name}


def _uses_synthetic_source(dataset: str | DatasetSpec | Mapping[str, Any]) -> bool:
    if get_dataset_spec(dataset).name == "synthetic":
        return True
    if not isinstance(dataset, Mapping):
        return False
    value = "".join(character.lower() for character in str(dataset.get("moabb_class", "")) if character.isalnum())
    return value == "syntheticeeg"


def _requested_subjects(config: Mapping[str, Any], subjects: str | int | Sequence[int] | None) -> str | int | Sequence[int] | None:
    return subjects if subjects is not None else config.get("subjects", "all")


def _normalize_run_key(run_key: str) -> str:
    value = str(run_key).strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("run_key must contain only letters, digits, dots, underscores, or hyphens")
    return value


def _set_download_directory(path: Path) -> None:
    module = importlib.import_module("moabb")
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(module, "set_download_dir"):
        module.set_download_dir(str(path))
    else:
        os.environ["MNE_DATA"] = str(path)


def download_dataset(
    dataset: str | DatasetSpec | Mapping[str, Any],
    raw_dir: str | Path,
    subjects: str | int | Sequence[int] | None = None,
    force: bool = False,
    accept: bool = True,
) -> Path:
    spec = get_dataset_spec(dataset)
    config = _dataset_config(dataset)
    synthetic_source = _uses_synthetic_source(dataset)
    root = Path(raw_dir).expanduser().resolve()
    receipt = root / spec.name / "download.json"
    instance = create_dataset(dataset)
    chosen = selected_subjects(instance, _requested_subjects(config, subjects))
    if receipt.exists() and not force:
        try:
            existing = _read_json(receipt)
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
        if (
            existing.get("dataset") == spec.name
            and tuple(int(value) for value in existing.get("subjects", ())) == chosen
            and bool(existing.get("synthetic")) == synthetic_source
            and existing.get("moabb_version") == _package_version("moabb")
        ):
            return receipt
    if not synthetic_source:
        _set_download_directory(root)
        if hasattr(instance, "download"):
            available = inspect.signature(instance.download).parameters
            candidates: dict[str, Any] = {
                "subject_list": list(chosen),
                "path": str(root),
                "force_update": bool(force),
                "update_path": False,
                "accept": bool(accept),
                "verbose": "WARNING",
            }
            instance.download(**{key: value for key, value in candidates.items() if key in available})
        else:
            for subject in chosen:
                instance.get_data(subjects=[subject])
    timestamp = _utc_now()
    source_class = config.get("moabb_class") or ("SyntheticEEG" if synthetic_source else spec.moabb_class)
    value = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": spec.name,
        "paper_name": str(config.get("paper_name", spec.paper_name)),
        "moabb_class": source_class,
        "moabb_version": _package_version("moabb"),
        "created_at": timestamp,
        "downloaded_at": timestamp,
        "subjects": list(chosen),
        "force_update": bool(force),
        "synthetic": synthetic_source,
        "versions": _versions(),
    }
    _write_json(receipt, value)
    return receipt


def _resolve_labels(spec: DatasetSpec, config: Mapping[str, Any]) -> tuple[dict[int, int], dict[int, str]]:
    configured = config.get("labels")
    if not isinstance(configured, Mapping):
        return spec.event_to_class, spec.label_names
    normalized: dict[str, tuple[str, int]] = {}
    for label, index in configured.items():
        key = str(label)
        try:
            normalized[f"number:{float(key):.8g}"] = (key, int(index))
        except ValueError:
            normalized[f"text:{key.lower()}"] = (key, int(index))
    event_to_class: dict[int, int] = {}
    label_names: dict[int, str] = {}
    for event_label, event_code in spec.event_id:
        try:
            lookup = f"number:{float(event_label):.8g}"
        except ValueError:
            lookup = f"text:{event_label.lower()}"
        if lookup not in normalized:
            raise ValueError(f"configured labels omit {event_label} for {spec.name}")
        configured_label, class_index = normalized[lookup]
        event_to_class[event_code] = class_index
        label_names[class_index] = configured_label
    expected_indices = list(range(len(label_names)))
    if sorted(label_names) != expected_indices:
        raise ValueError("class indices must be contiguous and start at zero")
    return event_to_class, label_names


def _synthetic_subject(spec: DatasetSpec, subject: int, seed: int) -> dict[str, dict[str, ArrayRaw]]:
    sfreq = int(spec.source_sfreq)
    events_by_class = 4
    event_count = events_by_class * len(spec.event_id)
    trial_samples = int(round(spec.duration * sfreq))
    event_span = max(spec.interval[1], 0.0) - min(spec.interval[0], 0.0)
    spacing = int(round(event_span * sfreq)) + max(sfreq // 2, 1)
    first_event = int(round(max(1.0, -spec.interval[0] + 0.5) * sfreq))
    sample_count = first_event + (event_count - 1) * spacing + int(round(max(spec.interval[1], 0.0) * sfreq)) + sfreq
    generator = np.random.default_rng(seed + subject * 1009)
    data = generator.normal(0.0, 0.7e-6, size=(spec.channels + 1, sample_count))
    events = np.zeros((event_count, 3), dtype=np.int64)
    timeline = np.arange(trial_samples, dtype=np.float64) / sfreq
    event_values = list(spec.event_id)
    for trial in range(event_count):
        class_index = trial % len(event_values)
        code = event_values[class_index][1]
        sample = first_event + trial * spacing
        events[trial] = (sample, 0, code)
        trial_start = sample + int(round(spec.interval[0] * sfreq))
        frequency = 8.0 + class_index * 2.0
        waveform = np.sin(2.0 * np.pi * frequency * timeline)
        data[class_index % spec.channels, trial_start : trial_start + trial_samples] += 4.0e-6 * waveform
        data[2 : min(5, spec.channels), trial_start : trial_start + trial_samples] += 1.2e-6 * waveform
        blink = np.exp(-((timeline - 0.2) ** 2) / 0.002)
        data[spec.channels, trial_start : trial_start + trial_samples] += 18.0e-6 * blink
    standard_names = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz")
    eeg_names = standard_names[: spec.channels] if spec.channels <= len(standard_names) else tuple(f"EEG{index + 1:03d}" for index in range(spec.channels))
    names = tuple(eeg_names) + ("EOGvu",)
    types = tuple("eeg" for _ in range(spec.channels)) + ("misc",)
    raw = ArrayRaw(data=data, sfreq=float(sfreq), ch_names=names, ch_types=types, events=events)
    return {"0": {"0": raw}}


def _subject_data(instance: Any, spec: DatasetSpec, subject: int, raw_dir: Path, seed: int, synthetic_source: bool) -> Mapping[Any, Any]:
    if synthetic_source:
        return _synthetic_subject(spec, subject, seed)
    _set_download_directory(raw_dir)
    result = instance.get_data(subjects=[subject])
    if subject in result:
        return result[subject]
    if str(subject) in result:
        return result[str(subject)]
    if len(result) == 1:
        return next(iter(result.values()))
    raise KeyError(f"subject {subject} absent from loaded data")


def _session_allowed(spec: DatasetSpec, session: Any, selection: Mapping[str, Any]) -> bool:
    requested = selection.get("sessions")
    if requested is None:
        return True
    requested_strings = {str(value) for value in requested}
    if str(session) in requested_strings:
        return True
    if spec.name == "lee2019_ssvep":
        try:
            return str(int(str(session)) + 1) in requested_strings
        except ValueError:
            return False
    return False


def _iter_runs(data: Mapping[Any, Any], spec: DatasetSpec, selection: Mapping[str, Any]) -> Iterator[tuple[str, str, Any]]:
    for session in sorted(data, key=str):
        if not _session_allowed(spec, session, selection):
            continue
        runs = data[session]
        for run in sorted(runs, key=str):
            yield str(session), str(run), runs[run]


def _shard_name(subject: int, session: str, run: str) -> str:
    identity = f"{subject}\0{session}\0{run}"
    return f"subject-{subject:03d}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def _metadata(result: RunWindows, subject: int, session: str, run: str) -> NDArray[Any]:
    metadata = np.empty(len(result.y), dtype=METADATA_DTYPE)
    metadata["subject"] = str(subject)
    metadata["session"] = session
    metadata["run"] = run
    metadata["trial"] = result.trial_ids
    metadata["window"] = result.window_ids
    metadata["event_code"] = result.event_codes
    metadata["label"] = result.labels
    metadata["window_start"] = result.window_starts
    metadata["valid_samples"] = result.valid_samples
    return metadata


def _trial_limit(selection: Mapping[str, Any], label: str, class_index: int) -> int | None:
    value = selection.get("max_trials_per_class")
    if value is None:
        return None
    if isinstance(value, Mapping):
        selected = value.get(label, value.get(str(class_index)))
        return None if selected is None else int(selected)
    return int(value)


def _limit_result(
    result: RunWindows,
    subject: int,
    selection: Mapping[str, Any],
    trial_counts: dict[tuple[int, int], int],
) -> NDArray[np.bool_]:
    keep = np.zeros(len(result.y), dtype=bool)
    if not len(result.y):
        return keep
    previous_trial: int | None = None
    accept = False
    for index, trial in enumerate(result.trial_ids):
        class_index = int(result.y[index])
        if previous_trial != int(trial):
            label = str(result.labels[index])
            limit = _trial_limit(selection, label, class_index)
            key = (subject, class_index)
            accept = limit is None or trial_counts.get(key, 0) < limit
            if accept:
                trial_counts[key] = trial_counts.get(key, 0) + 1
            previous_trial = int(trial)
        keep[index] = accept
    return keep


def _all_limits_reached(
    subject: int,
    selection: Mapping[str, Any],
    label_names: Mapping[int, str],
    trial_counts: Mapping[tuple[int, int], int],
) -> bool:
    if "max_trials_per_class" not in selection:
        return False
    for class_index, label in label_names.items():
        limit = _trial_limit(selection, label, class_index)
        if limit is None or trial_counts.get((subject, class_index), 0) < limit:
            return False
    return True


def _write_shard(
    path: Path,
    result: RunWindows,
    subject: int,
    session: str,
    run: str,
    keep: NDArray[np.bool_],
    fingerprint: str,
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    X = result.X[keep]
    y = result.y[keep]
    metadata = _metadata(result, subject, session, run)[keep]
    _save_array(path / "X.npy", X)
    _save_array(path / "y.npy", y)
    _save_array(path / "metadata.npy", metadata)
    retained_trials = len({int(value) for value in metadata["trial"]})
    record = {
        "status": "complete",
        "fingerprint": fingerprint,
        "subject": subject,
        "session": session,
        "run": run,
        "windows": int(len(y)),
        "retained_trials": retained_trials,
        "source_trials": int(result.source_trials),
        "dropped_boundary_trials": int(result.dropped_boundary_trials),
        "dropped_annotation_trials": int(result.dropped_annotation_trials),
        "channel_names": list(result.channel_names),
        "ica_excluded": list(result.ica_excluded),
        "x_shape": list(X.shape),
        "x_dtype": str(X.dtype),
        "class_window_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(y, return_counts=True), strict=True)
        },
    }
    _write_json(path / "manifest.json", record)
    return record


def _valid_shard(path: Path, fingerprint: str) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        record = _read_json(manifest_path)
        if record.get("status") != "complete" or record.get("fingerprint") != fingerprint:
            return None
        X = np.load(path / "X.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(path / "y.npy", mmap_mode="r", allow_pickle=False)
        metadata = np.load(path / "metadata.npy", mmap_mode="r", allow_pickle=False)
        if X.shape[0] != y.shape[0] or y.shape[0] != metadata.shape[0] or list(X.shape) != record.get("x_shape"):
            return None
        return record
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _update_counts_from_shard(path: Path, subject: int, counts: dict[tuple[int, int], int]) -> None:
    y = np.load(path / "y.npy", mmap_mode="r", allow_pickle=False)
    metadata = np.load(path / "metadata.npy", mmap_mode="r", allow_pickle=False)
    seen: set[tuple[str, str, int, int]] = set()
    for index in range(len(y)):
        key = (str(metadata["session"][index]), str(metadata["run"][index]), int(metadata["trial"][index]), int(y[index]))
        if key not in seen:
            seen.add(key)
            count_key = (subject, int(y[index]))
            counts[count_key] = counts.get(count_key, 0) + 1


def _align_indices(source: Sequence[str], target: Sequence[str]) -> NDArray[np.int64]:
    if tuple(source) == tuple(target):
        return np.arange(len(target), dtype=np.int64)
    if set(source) != set(target) or len(source) != len(target):
        raise ValueError(f"inconsistent EEG channels: {tuple(source)} versus {tuple(target)}")
    source_positions = {name: index for index, name in enumerate(source)}
    return np.asarray([source_positions[name] for name in target], dtype=np.int64)


def _merge_shards(cache_dir: Path, records: Sequence[dict[str, Any]]) -> tuple[tuple[str, ...], int]:
    nonempty = [record for record in records if int(record["windows"]) > 0]
    if not nonempty:
        raise ValueError("preprocessing produced no windows")
    channel_names = tuple(nonempty[0]["channel_names"])
    window_samples = int(nonempty[0]["x_shape"][1])
    total = sum(int(record["windows"]) for record in records)
    x_temporary = cache_dir / "X.npy.tmp"
    y_temporary = cache_dir / "y.npy.tmp"
    metadata_temporary = cache_dir / "metadata.npy.tmp"
    X_output = open_memmap(x_temporary, mode="w+", dtype=np.float32, shape=(total, window_samples, len(channel_names)))
    y_output = open_memmap(y_temporary, mode="w+", dtype=np.int64, shape=(total,))
    metadata_output = open_memmap(metadata_temporary, mode="w+", dtype=METADATA_DTYPE, shape=(total,))
    cursor = 0
    for record in records:
        count = int(record["windows"])
        if not count:
            continue
        shard = cache_dir / "shards" / str(record["shard"])
        X = np.load(shard / "X.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(shard / "y.npy", mmap_mode="r", allow_pickle=False)
        metadata = np.load(shard / "metadata.npy", mmap_mode="r", allow_pickle=False)
        indices = _align_indices(record["channel_names"], channel_names)
        X_output[cursor : cursor + count] = X[:, :, indices]
        y_output[cursor : cursor + count] = y
        metadata_output[cursor : cursor + count] = metadata
        cursor += count
    X_output.flush()
    y_output.flush()
    metadata_output.flush()
    del X_output, y_output, metadata_output
    os.replace(x_temporary, cache_dir / "X.npy")
    os.replace(y_temporary, cache_dir / "y.npy")
    os.replace(metadata_temporary, cache_dir / "metadata.npy")
    return channel_names, total


def _aggregate_counts(cache_dir: Path, label_names: Mapping[int, str]) -> dict[str, Any]:
    y = np.load(cache_dir / "y.npy", mmap_mode="r", allow_pickle=False)
    metadata = np.load(cache_dir / "metadata.npy", mmap_mode="r", allow_pickle=False)
    window_counts = {
        str(label_names[int(value)]): int(count)
        for value, count in zip(*np.unique(y, return_counts=True), strict=True)
    }
    trial_keys: set[tuple[str, str, str, int, int]] = set()
    subject_counts: dict[str, dict[str, int]] = {}
    session_counts: dict[str, int] = {}
    run_counts: dict[str, int] = {}
    for index in range(len(metadata)):
        subject = str(metadata["subject"][index])
        session = str(metadata["session"][index])
        run = str(metadata["run"][index])
        trial = int(metadata["trial"][index])
        class_index = int(y[index])
        key = (subject, session, run, trial, class_index)
        if key in trial_keys:
            continue
        trial_keys.add(key)
        label = str(label_names[class_index])
        subject_counts.setdefault(subject, {})[label] = subject_counts.setdefault(subject, {}).get(label, 0) + 1
        session_key = f"{subject}/{session}"
        run_key = f"{subject}/{session}/{run}"
        session_counts[session_key] = session_counts.get(session_key, 0) + 1
        run_counts[run_key] = run_counts.get(run_key, 0) + 1
    trial_counts: dict[str, int] = {}
    for counts in subject_counts.values():
        for label, count in counts.items():
            trial_counts[label] = trial_counts.get(label, 0) + count
    return {
        "windows": int(len(y)),
        "trials": len(trial_keys),
        "windows_per_class": window_counts,
        "trials_per_class": trial_counts,
        "trials_per_subject_and_class": subject_counts,
        "trials_per_subject_and_session": session_counts,
        "trials_per_subject_session_and_run": run_counts,
    }


def _validation(
    spec: DatasetSpec,
    subjects: Sequence[int],
    event_to_class: Mapping[int, int],
    label_names: Mapping[int, str],
    selection: Mapping[str, Any],
    counts: Mapping[str, Any],
    channel_count: int,
) -> dict[str, Any]:
    expected_by_subject: dict[str, dict[str, int]] = {}
    strict = True
    for subject in subjects:
        expected = expected_trials_for_subject(spec, subject)
        if expected is None:
            strict = False
            continue
        converted: dict[str, int] = {}
        for source_label, expected_count in expected.items():
            event_code = spec.event_mapping[source_label]
            class_index = event_to_class[event_code]
            output_label = label_names[class_index]
            limit = _trial_limit(selection, output_label, class_index)
            converted[output_label] = min(expected_count, limit) if limit is not None else expected_count
        expected_by_subject[str(subject)] = converted
    observed = counts["trials_per_subject_and_class"]
    mismatches: dict[str, Any] = {}
    if strict:
        for subject in subjects:
            expected = expected_by_subject[str(subject)]
            actual = observed.get(str(subject), {})
            if actual != expected:
                mismatches[str(subject)] = {"expected": expected, "observed": actual}
    return {
        "strict_trial_counts_available": strict,
        "trial_counts_match": None if not strict else not mismatches,
        "trial_count_mismatches": mismatches,
        "expected_trials_per_subject_and_class": expected_by_subject,
        "expected_rule": spec.expected_rule,
        "channel_count_expected": spec.channels,
        "channel_count_observed": channel_count,
        "channel_count_matches": channel_count == spec.channels,
        "paper_reported_total_trials": spec.paper_reported_total_trials,
    }


def _latest_file(processed_dir: Path) -> Path:
    return processed_dir / "latest.json"


def _record_latest(processed_dir: Path, dataset: str, run_key: str) -> None:
    path = _latest_file(processed_dir)
    value = _read_json(path) if path.exists() else {}
    value[dataset] = run_key
    _write_json(path, value)


def cache_dataset(
    dataset: str | DatasetSpec | Mapping[str, Any],
    raw_dir: str | Path,
    processed_dir: str | Path,
    subjects: str | int | Sequence[int] | None = None,
    preprocessing: PreprocessingConfig | Mapping[str, Any] | None = None,
    seed: int = 42,
    force: bool = False,
    run_key: str | None = None,
) -> CachedDataset:
    spec = get_dataset_spec(dataset)
    dataset_config = _dataset_config(dataset)
    synthetic_source = _uses_synthetic_source(dataset)
    instance = create_dataset(dataset)
    chosen = selected_subjects(instance, _requested_subjects(dataset_config, subjects))
    settings = preprocessing if isinstance(preprocessing, PreprocessingConfig) else PreprocessingConfig.from_mapping(preprocessing, spec, seed)
    event_to_class, label_names = _resolve_labels(spec, dataset_config)
    selection = dataset_config.get("selection", {})
    selection = dict(selection) if isinstance(selection, Mapping) else {}
    fingerprint_input = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": spec.manifest(),
        "dataset_config": dataset_config,
        "subjects": list(chosen),
        "preprocessing": settings.manifest(),
        "event_to_class": event_to_class,
        "label_names": label_names,
        "seed": seed,
        "versions": _versions(),
    }
    fingerprint = _digest(fingerprint_input)
    resolved_run_key = _normalize_run_key(run_key or fingerprint[:12])
    processed_root = Path(processed_dir).expanduser().resolve()
    cache_dir = processed_root / resolved_run_key / spec.name
    manifest_path = cache_dir / "manifest.json"
    if force and cache_dir.exists():
        shutil.rmtree(cache_dir)
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing.get("fingerprint") != fingerprint:
            raise ValueError(f"cache key {resolved_run_key} already belongs to a different configuration")
        if existing.get("status") == "complete":
            _record_latest(processed_root, spec.name, resolved_run_key)
            return load_cached_dataset(spec.name, processed_root, resolved_run_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "shards").mkdir(parents=True, exist_ok=True)
    building = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "building",
        "dataset": spec.name,
        "run_key": resolved_run_key,
        "fingerprint": fingerprint,
        "started_at": _utc_now(),
        "selected_subjects": list(chosen),
        "preprocessing": settings.manifest(),
    }
    _write_json(manifest_path, building)
    records: list[dict[str, Any]] = []
    trial_counts: dict[tuple[int, int], int] = {}
    raw_root = Path(raw_dir).expanduser().resolve()
    for subject in chosen:
        subject_data = _subject_data(instance, spec, subject, raw_root, seed, synthetic_source)
        for session, run, raw in _iter_runs(subject_data, spec, selection):
            if _all_limits_reached(subject, selection, label_names, trial_counts):
                break
            shard_name = _shard_name(subject, session, run)
            shard_path = cache_dir / "shards" / shard_name
            record = _valid_shard(shard_path, fingerprint)
            if record is None:
                result = preprocess_run(raw, spec, settings, event_to_class, label_names)
                keep = _limit_result(result, subject, selection, trial_counts)
                record = _write_shard(shard_path, result, subject, session, run, keep, fingerprint)
            else:
                _update_counts_from_shard(shard_path, subject, trial_counts)
            record = dict(record)
            record["shard"] = shard_name
            records.append(record)
    channel_names, total_windows = _merge_shards(cache_dir, records)
    counts = _aggregate_counts(cache_dir, label_names)
    receipt = raw_root / spec.name / "download.json"
    source_receipt = (
        {
            "path": f"{spec.name}/download.json",
            "sha256": _file_digest(receipt),
        }
        if receipt.exists()
        else None
    )
    file_sha256 = {
        "X.npy": _file_digest(cache_dir / "X.npy"),
        "y.npy": _file_digest(cache_dir / "y.npy"),
        "metadata.npy": _file_digest(cache_dir / "metadata.npy"),
    }
    final_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete",
        "dataset": spec.name,
        "paper_name": spec.paper_name,
        "run_key": resolved_run_key,
        "fingerprint": fingerprint,
        "completed_at": _utc_now(),
        "selected_subjects": list(chosen),
        "dataset_spec": spec.manifest(),
        "dataset_config": dataset_config,
        "preprocessing": settings.manifest(),
        "event_id": spec.event_mapping,
        "event_to_class": {str(key): value for key, value in event_to_class.items()},
        "label_names": {str(key): value for key, value in label_names.items()},
        "channel_names": list(channel_names),
        "layout": "windows_time_channels",
        "files": {"X": "X.npy", "y": "y.npy", "metadata": "metadata.npy"},
        "file_sha256": file_sha256,
        "source_receipt": source_receipt,
        "shapes": {"X": [total_windows, settings.window_samples, len(channel_names)], "y": [total_windows], "metadata": [total_windows]},
        "dtypes": {"X": "float32", "y": "int64", "metadata": METADATA_DTYPE.descr},
        "counts": counts,
        "validation": _validation(
            spec,
            chosen,
            event_to_class,
            label_names,
            selection,
            counts,
            len(channel_names),
        ),
        "source_trials_seen": sum(int(record["source_trials"]) for record in records),
        "dropped_boundary_trials": sum(int(record["dropped_boundary_trials"]) for record in records),
        "dropped_annotation_trials": sum(int(record["dropped_annotation_trials"]) for record in records),
        "shards": records,
        "versions": _versions(),
    }
    _write_json(manifest_path, final_manifest)
    _record_latest(processed_root, spec.name, resolved_run_key)
    return load_cached_dataset(spec.name, processed_root, resolved_run_key)


def _resolve_cache_path(dataset: str, processed_dir: Path, run_key: str | None) -> Path:
    if run_key is not None:
        return processed_dir / _normalize_run_key(run_key) / dataset
    latest = _latest_file(processed_dir)
    if latest.exists():
        value = _read_json(latest)
        if dataset in value:
            return processed_dir / _normalize_run_key(str(value[dataset])) / dataset
    candidates = sorted(processed_dir.glob(f"*/{dataset}/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    complete = [path.parent for path in candidates if _read_json(path).get("status") == "complete"]
    if not complete:
        raise FileNotFoundError(f"no complete cache found for {dataset}")
    return complete[0]


def load_cached_dataset(
    dataset: str | DatasetSpec | Mapping[str, Any],
    processed_dir: str | Path,
    run_key: str | None = None,
    mmap_mode: str | None = "r",
) -> CachedDataset:
    spec = get_dataset_spec(dataset)
    root = Path(processed_dir).expanduser().resolve()
    path = _resolve_cache_path(spec.name, root, run_key)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"cache is not complete: {path}")
    X = np.load(path / str(manifest["files"]["X"]), mmap_mode=mmap_mode, allow_pickle=False)
    y = np.load(path / str(manifest["files"]["y"]), mmap_mode=mmap_mode, allow_pickle=False)
    metadata = np.load(path / str(manifest["files"]["metadata"]), mmap_mode=mmap_mode, allow_pickle=False)
    expected_shape = tuple(manifest["shapes"]["X"])
    if X.shape != expected_shape or y.shape != (X.shape[0],) or metadata.shape != (X.shape[0],):
        raise ValueError(f"cache files do not match manifest: {path}")
    return CachedDataset(X=X, y=y, metadata=metadata, manifest=manifest, path=path)


def dataset_manifest(
    dataset: str | DatasetSpec | Mapping[str, Any],
    processed_dir: str | Path | None = None,
    run_key: str | None = None,
) -> dict[str, Any]:
    spec = get_dataset_spec(dataset)
    if processed_dir is None:
        return spec.manifest()
    path = _resolve_cache_path(spec.name, Path(processed_dir).expanduser().resolve(), run_key)
    return _read_json(path / "manifest.json")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "METADATA_DTYPE",
    "CachedDataset",
    "cache_dataset",
    "dataset_manifest",
    "download_dataset",
    "load_cached_dataset",
]
