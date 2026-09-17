from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    paper_name: str
    moabb_class: str | None
    paradigm: str
    interval: tuple[float, float]
    event_id: tuple[tuple[str, int], ...]
    class_indices: tuple[tuple[str, int], ...]
    bandpass: tuple[float, float]
    subjects: tuple[int, ...]
    channels: int
    source_sfreq: float
    sessions: int | None
    segmentation: str
    constructor_kwargs: tuple[tuple[str, Any], ...] = ()
    expected_trials: tuple[tuple[str, int], ...] = ()
    expected_rule: str = "fixed_per_subject"
    paper_reported_total_trials: int | None = None

    @property
    def duration(self) -> float:
        return self.interval[1] - self.interval[0]

    @property
    def event_mapping(self) -> dict[str, int]:
        return dict(self.event_id)

    @property
    def class_mapping(self) -> dict[str, int]:
        return dict(self.class_indices)

    @property
    def event_to_class(self) -> dict[int, int]:
        classes = self.class_mapping
        return {code: classes[label] for label, code in self.event_id}

    @property
    def label_names(self) -> dict[int, str]:
        return {index: label for label, index in self.class_indices}

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.constructor_kwargs)

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["duration"] = self.duration
        value["event_id"] = self.event_mapping
        value["class_indices"] = self.class_mapping
        value["event_to_class"] = self.event_to_class
        value["constructor_kwargs"] = self.kwargs
        value["expected_trials"] = dict(self.expected_trials)
        return value


def _subjects(count: int) -> tuple[int, ...]:
    return tuple(range(1, count + 1))


PUBLIC_DATASET_NAMES = (
    "bnci2014_001",
    "bnci2014_004",
    "lee2019_ssvep",
    "nakanishi2015",
    "bi2012",
    "bi2013a",
    "bi2014b",
    "bi2015a",
    "bi2015b",
    "bnci2014_008",
    "bnci2014_009",
    "sosulski2019",
)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "bnci2014_001": DatasetSpec(
        "bnci2014_001",
        "BCICIV_2a",
        "BNCI2014_001",
        "mi",
        (2.0, 6.0),
        (("left_hand", 1), ("right_hand", 2), ("feet", 3), ("tongue", 4)),
        (("left_hand", 0), ("right_hand", 1), ("feet", 2), ("tongue", 3)),
        (6.0, 32.0),
        _subjects(9),
        22,
        250.0,
        2,
        "overlapping_full_windows",
        expected_trials=(("left_hand", 144), ("right_hand", 144), ("feet", 144), ("tongue", 144)),
        paper_reported_total_trials=62208,
    ),
    "bnci2014_004": DatasetSpec(
        "bnci2014_004",
        "BCICIV_2b",
        "BNCI2014_004",
        "mi",
        (3.0, 7.5),
        (("left_hand", 1), ("right_hand", 2)),
        (("left_hand", 0), ("right_hand", 1)),
        (6.0, 32.0),
        _subjects(9),
        3,
        250.0,
        5,
        "nonoverlapping_full_windows_and_right_padded_tail",
        expected_trials=(("left_hand", 360), ("right_hand", 360)),
        paper_reported_total_trials=32400,
    ),
    "lee2019_ssvep": DatasetSpec(
        "lee2019_ssvep",
        "Lee2019_SSVEP",
        "Lee2019_SSVEP",
        "ssvep",
        (0.0, 4.0),
        (("12.0", 1), ("8.57", 2), ("6.67", 3), ("5.45", 4)),
        (("5.45", 0), ("6.67", 1), ("8.57", 2), ("12.0", 3)),
        (4.0, 14.0),
        _subjects(54),
        62,
        1000.0,
        2,
        "overlapping_full_windows",
        constructor_kwargs=(("train_run", True), ("test_run", False), ("resting_state", False), ("sessions", (1, 2))),
        expected_trials=(("5.45", 50), ("6.67", 50), ("8.57", 50), ("12.0", 50)),
    ),
    "nakanishi2015": DatasetSpec(
        "nakanishi2015",
        "Nakanishi2015",
        "Nakanishi2015",
        "ssvep",
        (0.15, 4.3),
        (("9.25", 1), ("11.25", 2), ("13.25", 3), ("9.75", 4), ("11.75", 5), ("13.75", 6), ("10.25", 7), ("12.25", 8), ("14.25", 9), ("10.75", 10), ("12.75", 11), ("14.75", 12)),
        (("9.25", 0), ("9.75", 1), ("10.25", 2), ("10.75", 3), ("11.25", 4), ("11.75", 5), ("12.25", 6), ("12.75", 7), ("13.25", 8), ("13.75", 9), ("14.25", 10), ("14.75", 11)),
        (8.0, 16.0),
        _subjects(9),
        8,
        256.0,
        1,
        "nonoverlapping_full_windows_and_right_padded_tail",
        expected_trials=tuple((label, 15) for label in ("9.25", "9.75", "10.25", "10.75", "11.25", "11.75", "12.25", "12.75", "13.25", "13.75", "14.25", "14.75")),
    ),
    "bi2012": DatasetSpec(
        "bi2012",
        "BI2012",
        "BI2012",
        "erp",
        (0.0, 1.0),
        (("Target", 2), ("NonTarget", 1)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(25),
        16,
        128.0,
        1,
        "overlapping_full_windows",
        constructor_kwargs=(("Training", True), ("Online", False)),
        expected_trials=(("NonTarget", 640), ("Target", 128)),
    ),
    "bi2013a": DatasetSpec(
        "bi2013a",
        "BI2013a",
        "BI2013a",
        "erp",
        (0.0, 1.0),
        (("Target", 33285), ("NonTarget", 33286)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(24),
        16,
        512.0,
        None,
        "overlapping_full_windows",
        constructor_kwargs=(("NonAdaptive", True), ("Adaptive", False), ("Training", True), ("Online", False)),
        expected_trials=(("NonTarget", 400), ("Target", 80)),
        expected_rule="subjects_1_to_7_have_eight_sessions_others_one",
    ),
    "bi2014b": DatasetSpec(
        "bi2014b",
        "BI2014b",
        "BI2014b",
        "erp",
        (0.0, 1.0),
        (("Target", 2), ("NonTarget", 1)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(38),
        32,
        512.0,
        1,
        "overlapping_full_windows",
        expected_trials=(("NonTarget", 200), ("Target", 40)),
    ),
    "bi2015a": DatasetSpec(
        "bi2015a",
        "BI2015a",
        "BI2015a",
        "erp",
        (0.0, 1.0),
        (("Target", 2), ("NonTarget", 1)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(43),
        32,
        512.0,
        3,
        "overlapping_full_windows",
        expected_trials=(("NonTarget", 4131), ("Target", 825)),
        expected_rule="paper_reported_approximate_counts",
    ),
    "bi2015b": DatasetSpec(
        "bi2015b",
        "BI2015b",
        "BI2015b",
        "erp",
        (0.0, 1.0),
        (("Target", 2), ("NonTarget", 1)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(44),
        32,
        512.0,
        1,
        "overlapping_full_windows",
        expected_trials=(("NonTarget", 2160), ("Target", 480)),
    ),
    "bnci2014_008": DatasetSpec(
        "bnci2014_008",
        "BNCI2014_008",
        "BNCI2014_008",
        "erp",
        (0.0, 1.0),
        (("Target", 2), ("NonTarget", 1)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(8),
        8,
        256.0,
        1,
        "overlapping_full_windows",
        expected_trials=(("NonTarget", 3500), ("Target", 700)),
    ),
    "bnci2014_009": DatasetSpec(
        "bnci2014_009",
        "BNCI2014_009",
        "BNCI2014_009",
        "erp",
        (0.0, 0.8),
        (("Target", 2), ("NonTarget", 1)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(10),
        16,
        256.0,
        3,
        "single_right_padded_window",
        expected_trials=(("NonTarget", 1440), ("Target", 288)),
    ),
    "sosulski2019": DatasetSpec(
        "sosulski2019",
        "Sosulski2019",
        "Sosulski2019",
        "erp",
        (-0.2, 1.0),
        (("Target", 21), ("NonTarget", 1)),
        (("NonTarget", 0), ("Target", 1)),
        (0.1, 30.0),
        _subjects(13),
        31,
        1000.0,
        1,
        "nonoverlapping_full_windows_and_right_padded_tail",
        constructor_kwargs=(("use_soas_as_sessions", False), ("load_soa_60", True), ("reject_non_iid", False), ("interval", (-0.2, 1.0))),
        expected_trials=(("NonTarget", 7500), ("Target", 1500)),
    ),
    "synthetic": DatasetSpec(
        "synthetic",
        "SyntheticEEG",
        None,
        "mi",
        (0.0, 1.0),
        (("left_hand", 1), ("right_hand", 2)),
        (("left_hand", 0), ("right_hand", 1)),
        (6.0, 32.0),
        (1, 2, 3, 4),
        8,
        128.0,
        1,
        "overlapping_full_windows",
        expected_trials=(("left_hand", 8), ("right_hand", 8)),
    ),
}


def _normalized_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


_ALIASES: dict[str, str] = {}
for _name, _spec in DATASET_REGISTRY.items():
    for _alias in (_name, _spec.paper_name, _spec.moabb_class or ""):
        if _alias:
            _ALIASES[_normalized_name(_alias)] = _name
_ALIASES.update({"bciciv2a": "bnci2014_001", "bciciv2b": "bnci2014_004"})


class SyntheticDataset:
    def __init__(self, spec: DatasetSpec | None = None) -> None:
        resolved = spec or DATASET_REGISTRY["synthetic"]
        self.subject_list = list(resolved.subjects)
        self.code = "SyntheticEEG"
        self.event_id = resolved.event_mapping
        self.interval = list(resolved.interval)


def get_dataset_spec(dataset: str | DatasetSpec | Mapping[str, Any]) -> DatasetSpec:
    if isinstance(dataset, DatasetSpec):
        return dataset
    name = dataset.get("name", dataset.get("moabb_class", "")) if isinstance(dataset, Mapping) else dataset
    canonical = _ALIASES.get(_normalized_name(str(name)))
    if canonical is None:
        raise KeyError(f"unknown dataset: {name}")
    return DATASET_REGISTRY[canonical]


def list_dataset_specs(include_synthetic: bool = False) -> tuple[DatasetSpec, ...]:
    names = PUBLIC_DATASET_NAMES + (("synthetic",) if include_synthetic else ())
    return tuple(DATASET_REGISTRY[name] for name in names)


def create_dataset(dataset: str | DatasetSpec | Mapping[str, Any]) -> Any:
    spec = get_dataset_spec(dataset)
    synthetic_source = isinstance(dataset, Mapping) and _normalized_name(str(dataset.get("moabb_class", ""))) == "syntheticeeg"
    if spec.name == "synthetic" or synthetic_source:
        return SyntheticDataset(spec)
    values = spec.kwargs
    if isinstance(dataset, Mapping):
        values.update(dict(dataset.get("constructor_kwargs", {})))
        selection = dataset.get("selection", {})
        if spec.name == "lee2019_ssvep" and isinstance(selection, Mapping) and "sessions" in selection:
            values["sessions"] = tuple(int(value) for value in selection["sessions"])
    module = importlib.import_module("moabb.datasets")
    constructor = getattr(module, str(spec.moabb_class))
    return constructor(**values)


def selected_subjects(dataset: Any, requested: str | int | Sequence[int] | None = None) -> tuple[int, ...]:
    available = tuple(int(subject) for subject in dataset.subject_list)
    if requested is None or isinstance(requested, str) and requested == "all":
        return available
    if isinstance(requested, str):
        raise ValueError("subjects must be integers or 'all'")
    values = (int(requested),) if isinstance(requested, Integral) else tuple(int(subject) for subject in requested)
    unknown = sorted(set(values).difference(available))
    if unknown:
        raise ValueError(f"subjects unavailable for dataset: {unknown}")
    if len(values) != len(set(values)):
        raise ValueError("subjects must be unique")
    return values


def select_subjects(dataset: Any, requested: str | int | Sequence[int] | None = None) -> tuple[int, ...]:
    return selected_subjects(dataset, requested)


def expected_trials_for_subject(spec: DatasetSpec, subject: int) -> dict[str, int] | None:
    if not spec.expected_trials or spec.expected_rule == "paper_reported_approximate_counts":
        return None
    multiplier = 8 if spec.expected_rule == "subjects_1_to_7_have_eight_sessions_others_one" and subject <= 7 else 1
    return {label: count * multiplier for label, count in spec.expected_trials}


__all__ = [
    "DATASET_REGISTRY",
    "PUBLIC_DATASET_NAMES",
    "DatasetSpec",
    "SyntheticDataset",
    "create_dataset",
    "expected_trials_for_subject",
    "get_dataset_spec",
    "list_dataset_specs",
    "select_subjects",
    "selected_subjects",
]
