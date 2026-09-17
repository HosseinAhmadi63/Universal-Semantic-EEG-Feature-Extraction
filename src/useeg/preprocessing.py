from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .datasets import DatasetSpec, get_dataset_spec


@dataclass(frozen=True)
class PreprocessingConfig:
    sfreq: float = 128.0
    l_freq: float | None = None
    h_freq: float | None = None
    filter_method: str = "fir"
    fir_design: str = "firwin"
    fir_window: str = "hamming"
    filter_phase: str = "zero-double"
    filter_length: str | int = "auto"
    l_trans_bandwidth: str | float = "auto"
    h_trans_bandwidth: str | float = "auto"
    filter_pad: str = "reflect_limited"
    notch_frequency: float | None = 50.0
    notch_width: float = 0.25
    notch_transition_bandwidth: float = 1.0
    ica_enabled: bool = True
    ica_method: str = "fastica"
    ica_n_components: float | int | None = 0.99
    ica_max_iter: str | int = "auto"
    ica_random_state: int = 42
    reject_by_annotation: bool = True
    window_samples: int = 128
    hop_samples: int = 64
    window_mode: str = "overlapping_full_windows"
    include_partial_tail: bool = False
    pad_value: float = 0.0
    epsilon: float = 1e-8

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        dataset: str | DatasetSpec | Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> PreprocessingConfig:
        source = dict(value or {})
        spec = get_dataset_spec(dataset) if dataset is not None else None
        filters = source.get("filters", {})
        filters = filters if isinstance(filters, Mapping) else {}
        band = filters.get(spec.name, filters.get(spec.paradigm, {})) if spec is not None else {}
        band = band if isinstance(band, Mapping) else {}
        notch = source.get("notch", {})
        notch = notch if isinstance(notch, Mapping) else {}
        ica = source.get("ica", {})
        ica = ica if isinstance(ica, Mapping) else {}
        window = source.get("window", {})
        window = window if isinstance(window, Mapping) else {}
        overrides = window.get("overrides", {})
        overrides = overrides if isinstance(overrides, Mapping) else {}
        override = overrides.get(spec.name, {}) if spec is not None else {}
        override = override if isinstance(override, Mapping) else {}
        standardization = source.get("standardization", {})
        standardization = standardization if isinstance(standardization, Mapping) else {}
        default_band = spec.bandpass if spec is not None else (None, None)
        mode = override.get("mode", window.get("mode", spec.segmentation if spec is not None else "overlapping_full_windows"))
        random_state = ica.get("random_state", seed if seed is not None else 42)
        return cls(
            sfreq=float(source.get("sfreq", source.get("target_sfreq", 128.0))),
            l_freq=_optional_float(band.get("l_freq", source.get("l_freq", default_band[0]))),
            h_freq=_optional_float(band.get("h_freq", source.get("h_freq", default_band[1]))),
            filter_method=str(filters.get("method", "fir")),
            fir_design=str(filters.get("design", "firwin")),
            fir_window=str(filters.get("window", "hamming")),
            filter_phase=str(filters.get("phase", "zero-double")),
            filter_length=filters.get("filter_length", "auto"),
            l_trans_bandwidth=filters.get("l_trans_bandwidth", "auto"),
            h_trans_bandwidth=filters.get("h_trans_bandwidth", "auto"),
            filter_pad=str(filters.get("pad", "reflect_limited")),
            notch_frequency=_optional_float(notch.get("frequency_hz", source.get("notch_frequency", 50.0))),
            notch_width=float(notch.get("notch_width_hz", 0.25)),
            notch_transition_bandwidth=float(notch.get("transition_bandwidth_hz", 1.0)),
            ica_enabled=bool(ica.get("enabled", source.get("ica_enabled", True))),
            ica_method=str(ica.get("method", "fastica")),
            ica_n_components=ica.get("n_components", 0.99),
            ica_max_iter=ica.get("max_iter", "auto"),
            ica_random_state=int(random_state),
            reject_by_annotation=bool(ica.get("reject_by_annotation", source.get("reject_by_annotation", True))),
            window_samples=int(window.get("size_samples", source.get("window_samples", 128))),
            hop_samples=int(override.get("hop_samples", window.get("hop_samples", source.get("hop_samples", 64)))),
            window_mode=str(mode),
            include_partial_tail=bool(override.get("include_partial_tail", window.get("include_partial_tail", False))),
            pad_value=float(window.get("pad_value", 0.0)),
            epsilon=float(standardization.get("epsilon", source.get("epsilon", 1e-8))),
        )

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArrayRaw:
    data: NDArray[np.floating[Any]]
    sfreq: float
    ch_names: tuple[str, ...]
    ch_types: tuple[str, ...]
    events: NDArray[np.integer[Any]]
    first_samp: int = 0

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("data must have shape channels by samples")
        if len(self.ch_names) != self.data.shape[0] or len(self.ch_types) != self.data.shape[0]:
            raise ValueError("channel metadata does not match data")
        if self.events.ndim != 2 or self.events.shape[1] != 3:
            raise ValueError("events must have shape events by 3")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")


@dataclass(frozen=True)
class SegmentedTrial:
    windows: NDArray[np.float32]
    starts: NDArray[np.int64]
    valid_samples: NDArray[np.int32]


@dataclass(frozen=True)
class RunWindows:
    X: NDArray[np.float32]
    y: NDArray[np.int64]
    event_codes: NDArray[np.int64]
    labels: NDArray[np.str_]
    trial_ids: NDArray[np.int64]
    window_ids: NDArray[np.int32]
    window_starts: NDArray[np.int32]
    valid_samples: NDArray[np.int32]
    channel_names: tuple[str, ...]
    source_trials: int
    dropped_boundary_trials: int
    dropped_annotation_trials: int
    ica_excluded: tuple[int, ...]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def zscore_windows(
    windows: NDArray[np.floating[Any]],
    epsilon: float = 1e-8,
    time_axis: int = 1,
) -> NDArray[np.float32]:
    values = np.asarray(windows, dtype=np.float64)
    means = values.mean(axis=time_axis, keepdims=True)
    standard_deviations = values.std(axis=time_axis, keepdims=True)
    scaled = (values - means) / (standard_deviations + epsilon)
    return scaled.astype(np.float32, copy=False)


def segment_trial(
    trial: NDArray[np.floating[Any]],
    mode: str = "overlapping_full_windows",
    window_samples: int = 128,
    hop_samples: int = 64,
    include_partial_tail: bool = False,
    pad_value: float = 0.0,
) -> SegmentedTrial:
    values = np.asarray(trial)
    if values.ndim != 2:
        raise ValueError("trial must have shape channels by samples")
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("window and hop sizes must be positive")
    sample_count = values.shape[1]
    if mode == "single_right_padded_window":
        starts = [0]
    elif mode == "nonoverlapping_full_windows_and_right_padded_tail":
        starts = list(range(0, sample_count - window_samples + 1, window_samples))
        consumed = len(starts) * window_samples
        if consumed < sample_count or not starts:
            starts.append(consumed)
    elif mode == "overlapping_full_windows":
        starts = list(range(0, sample_count - window_samples + 1, hop_samples))
        if not starts and sample_count > 0:
            starts = [0]
        if include_partial_tail and starts:
            tail = starts[-1] + window_samples
            if tail < sample_count:
                starts.append(starts[-1] + hop_samples)
    else:
        raise ValueError(f"unknown window mode: {mode}")
    windows = np.full((len(starts), window_samples, values.shape[0]), pad_value, dtype=np.float64)
    valid = np.zeros(len(starts), dtype=np.int32)
    for index, start in enumerate(starts):
        length = max(0, min(window_samples, sample_count - start))
        if length:
            windows[index, :length] = values[:, start : start + length].T
        valid[index] = length
    return SegmentedTrial(windows.astype(np.float32), np.asarray(starts, dtype=np.int64), valid)


def _reference_type(name: str, current: str) -> str:
    lowered = name.lower().replace("-", "").replace("_", "")
    if current.lower() == "eog" or "eog" in lowered:
        return "eog"
    if current.lower() == "ecg" or "ecg" in lowered or "ekg" in lowered:
        return "ecg"
    return current.lower()


def _array_filter(data: NDArray[np.float64], config: PreprocessingConfig) -> NDArray[np.float64]:
    values = data.copy()
    sample_count = values.shape[1]
    nyquist = config.sfreq / 2.0
    if sample_count < 8:
        return values
    filter_active = config.l_freq is not None or config.h_freq is not None
    notch_active = config.notch_frequency is not None and 0 < config.notch_frequency < nyquist and (config.h_freq is None or config.h_freq >= config.notch_frequency)
    if not filter_active and not notch_active:
        return values
    from scipy.signal import filtfilt, firwin

    if filter_active:
        low = config.l_freq
        high = config.h_freq
        transition = low if low is not None else max(nyquist - float(high), 1.0)
        suggested = max(31, int(round(3.3 * config.sfreq / max(float(transition), 0.5))))
        maximum = max(3, (sample_count - 2) // 3)
        taps = min(suggested, maximum)
        taps -= 1 - taps % 2
        taps = max(taps, 3)
        if low is not None and high is not None:
            cutoff: float | list[float] = [low, high]
            pass_zero = False
        elif low is not None:
            cutoff = low
            pass_zero = False
        else:
            cutoff = float(high)
            pass_zero = True
        coefficients = firwin(taps, cutoff, pass_zero=pass_zero, fs=config.sfreq, window=config.fir_window)
        values = filtfilt(coefficients, [1.0], values, axis=1, padlen=min(3 * (taps - 1), sample_count - 1))
    frequency = config.notch_frequency
    if notch_active and frequency is not None:
        maximum = max(3, (sample_count - 2) // 3)
        taps = min(max(31, int(round(3.3 * config.sfreq))), maximum)
        taps -= 1 - taps % 2
        taps = max(taps, 3)
        half_width = config.notch_width / 2.0
        coefficients = firwin(taps, [frequency - half_width, frequency + half_width], pass_zero="bandstop", fs=config.sfreq, window=config.fir_window)
        values = filtfilt(coefficients, [1.0], values, axis=1, padlen=min(3 * (taps - 1), sample_count - 1))
    return values


def _array_ica(
    eeg: NDArray[np.float64],
    references: NDArray[np.float64],
    config: PreprocessingConfig,
) -> tuple[NDArray[np.float64], tuple[int, ...]]:
    if not config.ica_enabled or references.size == 0 or eeg.shape[0] < 2:
        return eeg, ()
    from sklearn.decomposition import FastICA

    centered = eeg.T - eeg.T.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if isinstance(config.ica_n_components, float) and 0 < config.ica_n_components < 1:
        cumulative = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        component_count = int(np.searchsorted(cumulative, config.ica_n_components) + 1)
    elif config.ica_n_components is None:
        component_count = eeg.shape[0]
    else:
        component_count = int(config.ica_n_components)
    component_count = min(max(component_count, 1), eeg.shape[0])
    max_iter = 1000 if config.ica_max_iter == "auto" else int(config.ica_max_iter)
    estimator = FastICA(
        n_components=component_count,
        whiten="unit-variance",
        random_state=config.ica_random_state,
        max_iter=max_iter,
    )
    sources = estimator.fit_transform(eeg.T)
    excluded: list[int] = []
    for component in range(sources.shape[1]):
        for reference in references:
            if np.std(reference) > 0 and abs(float(np.corrcoef(sources[:, component], reference)[0, 1])) >= 0.3:
                excluded.append(component)
                break
    excluded = sorted(set(excluded))
    if not excluded:
        return eeg, ()
    cleaned = sources.copy()
    cleaned[:, excluded] = 0.0
    return estimator.inverse_transform(cleaned).T, tuple(excluded)


def _prepare_array(
    raw: ArrayRaw,
    config: PreprocessingConfig,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int, tuple[str, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    data = np.asarray(raw.data, dtype=np.float64)
    events = np.asarray(raw.events, dtype=np.int64).copy()
    first_samp = int(raw.first_samp)
    if not np.isclose(raw.sfreq, config.sfreq):
        from scipy.signal import resample_poly

        fraction = Fraction(config.sfreq / raw.sfreq).limit_denominator(10000)
        data = resample_poly(data, fraction.numerator, fraction.denominator, axis=1)
        new_first = int(round(first_samp * config.sfreq / raw.sfreq))
        events[:, 0] = new_first + np.rint((events[:, 0] - first_samp) * config.sfreq / raw.sfreq).astype(np.int64)
        first_samp = new_first
    types = tuple(
        _reference_type(name, kind)
        for name, kind in zip(raw.ch_names, raw.ch_types, strict=True)
    )
    eeg_indices = [index for index, kind in enumerate(types) if kind == "eeg"]
    reference_indices = [index for index, kind in enumerate(types) if kind in {"eog", "ecg"}]
    if not eeg_indices:
        raise ValueError("run contains no EEG channels")
    eeg = _array_filter(data[eeg_indices], config)
    references = data[reference_indices] if reference_indices else np.empty((0, data.shape[1]))
    eeg, excluded = _array_ica(eeg, references, config)
    names = tuple(raw.ch_names[index] for index in eeg_indices)
    return eeg, events, first_samp, names, excluded, ()


def _find_mne_events(raw: Any, spec: DatasetSpec) -> NDArray[np.int64]:
    import mne

    try:
        events = mne.find_events(raw, shortest_event=1, verbose="ERROR")
    except (ValueError, RuntimeError):
        events = np.empty((0, 3), dtype=np.int64)
    valid_codes = set(spec.event_mapping.values())
    events = np.asarray(events, dtype=np.int64)
    if events.size:
        events = events[np.isin(events[:, 2], list(valid_codes))]
    if not events.size and len(raw.annotations):
        events, _ = mne.events_from_annotations(raw, event_id=spec.event_mapping, verbose="ERROR")
        events = events[np.isin(events[:, 2], list(valid_codes))]
    if not events.size:
        raise ValueError(f"no configured events found for {spec.name}")
    return events


def _annotation_intervals(raw: Any) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        if str(description).lower().startswith(("bad", "edge")):
            start = int(round((float(onset) - float(raw.first_time)) * float(raw.info["sfreq"])))
            stop = start + int(round(float(duration) * float(raw.info["sfreq"])))
            intervals.append((start, stop))
    return tuple(intervals)


def _prepare_mne(
    raw: Any,
    spec: DatasetSpec,
    config: PreprocessingConfig,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int, tuple[str, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    import mne

    work = raw.copy().load_data()
    updates: dict[str, str] = {}
    for name, kind in zip(work.ch_names, work.get_channel_types(), strict=True):
        resolved = _reference_type(name, kind)
        if resolved in {"eog", "ecg"} and resolved != kind:
            updates[name] = resolved
    if updates:
        work.set_channel_types(updates, verbose="ERROR")
    events = _find_mne_events(work, spec)
    if not np.isclose(float(work.info["sfreq"]), config.sfreq):
        work, events = work.resample(config.sfreq, events=events, npad="auto", verbose="ERROR")
    eeg_picks = mne.pick_types(work.info, eeg=True, exclude="bads")
    if not len(eeg_picks):
        raise ValueError("run contains no usable EEG channels")
    if config.l_freq is not None or config.h_freq is not None:
        work.filter(
            config.l_freq,
            config.h_freq,
            picks=eeg_picks,
            method=config.filter_method,
            phase=config.filter_phase,
            fir_window=config.fir_window,
            fir_design=config.fir_design,
            filter_length=config.filter_length,
            l_trans_bandwidth=config.l_trans_bandwidth,
            h_trans_bandwidth=config.h_trans_bandwidth,
            pad=config.filter_pad,
            verbose="ERROR",
        )
    if config.notch_frequency is not None and config.notch_frequency < config.sfreq / 2.0:
        work.notch_filter(
            freqs=[config.notch_frequency],
            picks=eeg_picks,
            method=config.filter_method,
            phase=config.filter_phase,
            fir_window=config.fir_window,
            fir_design=config.fir_design,
            filter_length=config.filter_length,
            notch_widths=config.notch_width,
            trans_bandwidth=config.notch_transition_bandwidth,
            pad=config.filter_pad,
            verbose="ERROR",
        )
    excluded: tuple[int, ...] = ()
    eog_names = [
        name
        for name, kind in zip(work.ch_names, work.get_channel_types(), strict=True)
        if kind == "eog"
    ]
    ecg_names = [
        name
        for name, kind in zip(work.ch_names, work.get_channel_types(), strict=True)
        if kind == "ecg"
    ]
    if config.ica_enabled and (eog_names or ecg_names) and len(eeg_picks) > 1:
        fit_raw = work.copy()
        if config.l_freq is None or config.l_freq < 1.0:
            fit_raw.filter(
                1.0,
                config.h_freq,
                picks=eeg_picks,
                method=config.filter_method,
                phase=config.filter_phase,
                fir_window=config.fir_window,
                fir_design=config.fir_design,
                filter_length=config.filter_length,
                pad=config.filter_pad,
                verbose="ERROR",
            )
        estimator = mne.preprocessing.ICA(
            n_components=config.ica_n_components,
            method=config.ica_method,
            random_state=config.ica_random_state,
            max_iter=config.ica_max_iter,
            verbose="ERROR",
        )
        estimator.fit(fit_raw, picks=eeg_picks, reject_by_annotation=config.reject_by_annotation, verbose="ERROR")
        detected: list[int] = []
        if eog_names:
            try:
                indices, _ = estimator.find_bads_eog(fit_raw, ch_name=eog_names, verbose="ERROR")
                detected.extend(indices)
            except (RuntimeError, ValueError):
                pass
        if ecg_names:
            try:
                indices, _ = estimator.find_bads_ecg(fit_raw, ch_name=ecg_names[0], verbose="ERROR")
                detected.extend(indices)
            except (RuntimeError, ValueError):
                pass
        excluded = tuple(sorted(set(int(index) for index in detected)))
        if excluded:
            estimator.apply(work, exclude=list(excluded), verbose="ERROR")
    annotation_intervals = _annotation_intervals(work) if config.reject_by_annotation else ()
    names = tuple(work.ch_names[index] for index in eeg_picks)
    data = work.get_data(picks=eeg_picks).astype(np.float64, copy=False)
    return data, np.asarray(events, dtype=np.int64), int(work.first_samp), names, excluded, annotation_intervals


def _overlaps(start: int, stop: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start < interval_stop and stop > interval_start for interval_start, interval_stop in intervals)


def preprocess_run(
    raw: Any,
    dataset: str | DatasetSpec | Mapping[str, Any],
    config: PreprocessingConfig | Mapping[str, Any] | None = None,
    event_to_class: Mapping[int, int] | None = None,
    label_names: Mapping[int, str] | None = None,
) -> RunWindows:
    spec = get_dataset_spec(dataset)
    settings = config if isinstance(config, PreprocessingConfig) else PreprocessingConfig.from_mapping(config, spec)
    if isinstance(raw, ArrayRaw):
        data, events, first_samp, channel_names, excluded, annotation_intervals = _prepare_array(raw, settings)
    else:
        data, events, first_samp, channel_names, excluded, annotation_intervals = _prepare_mne(raw, spec, settings)
    class_map = dict(event_to_class or spec.event_to_class)
    names = dict(label_names or spec.label_names)
    valid_events = events[np.isin(events[:, 2], list(class_map))]
    expected_samples = int(round(spec.duration * settings.sfreq))
    offset = int(round(spec.interval[0] * settings.sfreq))
    windows: list[NDArray[np.float32]] = []
    classes: list[int] = []
    codes: list[int] = []
    labels: list[str] = []
    trials: list[int] = []
    window_ids: list[int] = []
    starts: list[int] = []
    valid_lengths: list[int] = []
    dropped_boundary = 0
    dropped_annotation = 0
    for trial_id, event in enumerate(valid_events):
        start = int(event[0]) - first_samp + offset
        stop = start + expected_samples
        if start < 0 or stop > data.shape[1]:
            dropped_boundary += 1
            continue
        if _overlaps(start, stop, annotation_intervals):
            dropped_annotation += 1
            continue
        segmented = segment_trial(
            data[:, start:stop],
            mode=settings.window_mode,
            window_samples=settings.window_samples,
            hop_samples=settings.hop_samples,
            include_partial_tail=settings.include_partial_tail,
            pad_value=settings.pad_value,
        )
        normalized = zscore_windows(segmented.windows, epsilon=settings.epsilon, time_axis=1)
        class_index = int(class_map[int(event[2])])
        for window_id in range(normalized.shape[0]):
            windows.append(normalized[window_id])
            classes.append(class_index)
            codes.append(int(event[2]))
            labels.append(str(names[class_index]))
            trials.append(trial_id)
            window_ids.append(window_id)
            starts.append(int(segmented.starts[window_id]))
            valid_lengths.append(int(segmented.valid_samples[window_id]))
    shape = (0, settings.window_samples, len(channel_names))
    X = np.stack(windows).astype(np.float32, copy=False) if windows else np.empty(shape, dtype=np.float32)
    return RunWindows(
        X=X,
        y=np.asarray(classes, dtype=np.int64),
        event_codes=np.asarray(codes, dtype=np.int64),
        labels=np.asarray(labels, dtype="U64"),
        trial_ids=np.asarray(trials, dtype=np.int64),
        window_ids=np.asarray(window_ids, dtype=np.int32),
        window_starts=np.asarray(starts, dtype=np.int32),
        valid_samples=np.asarray(valid_lengths, dtype=np.int32),
        channel_names=channel_names,
        source_trials=len(valid_events),
        dropped_boundary_trials=dropped_boundary,
        dropped_annotation_trials=dropped_annotation,
        ica_excluded=excluded,
    )


__all__ = [
    "ArrayRaw",
    "PreprocessingConfig",
    "RunWindows",
    "SegmentedTrial",
    "preprocess_run",
    "segment_trial",
    "zscore_windows",
]
