import numpy as np

from useeg.preprocessing import (
    ArrayRaw,
    PreprocessingConfig,
    preprocess_run,
    segment_trial,
    zscore_windows,
)


def test_table_four_segmentation_counts_and_padding() -> None:
    ordinary = segment_trial(np.ones((2, 512)), "overlapping_full_windows", 128, 64)
    bnci009 = segment_trial(np.ones((2, 102)), "single_right_padded_window", 128, 128, True)
    sosulski = segment_trial(
        np.ones((2, 154)),
        "nonoverlapping_full_windows_and_right_padded_tail",
        128,
        128,
        True,
    )
    nakanishi = segment_trial(
        np.ones((2, 531)),
        "nonoverlapping_full_windows_and_right_padded_tail",
        128,
        128,
        True,
    )
    bci2b = segment_trial(
        np.ones((2, 576)),
        "nonoverlapping_full_windows_and_right_padded_tail",
        128,
        128,
        True,
    )
    assert ordinary.windows.shape == (7, 128, 2)
    assert bnci009.windows.shape == (1, 128, 2)
    assert sosulski.windows.shape == (2, 128, 2)
    assert nakanishi.windows.shape == (5, 128, 2)
    assert bci2b.windows.shape == (5, 128, 2)
    assert bnci009.valid_samples.tolist() == [102]
    assert sosulski.valid_samples.tolist() == [128, 26]
    assert nakanishi.valid_samples.tolist() == [128, 128, 128, 128, 19]
    assert bci2b.valid_samples.tolist() == [128, 128, 128, 128, 64]
    assert np.all(bnci009.windows[0, 102:] == 0.0)


def test_per_window_channel_standardization() -> None:
    windows = np.asarray(
        [
            [[1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0]],
            [[10.0, -2.0], [12.0, -2.0], [14.0, -2.0], [16.0, -2.0]],
        ]
    )
    standardized = zscore_windows(windows, epsilon=1e-8)
    np.testing.assert_allclose(standardized[:, :, 0].mean(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(standardized[:, :, 0].std(axis=1), 1.0, atol=1e-7)
    np.testing.assert_array_equal(standardized[:, :, 1], 0.0)
    assert standardized.dtype == np.float32


def test_array_run_crops_after_continuous_processing_and_keeps_lineage(monkeypatch) -> None:
    generator = np.random.default_rng(4)
    data = generator.normal(size=(3, 420))
    events = np.asarray([[40, 0, 1], [230, 0, 2]], dtype=np.int64)
    raw = ArrayRaw(
        data=data,
        sfreq=128.0,
        ch_names=("C3", "C4", "EOGvu"),
        ch_types=("eeg", "eeg", "misc"),
        events=events,
    )
    config = PreprocessingConfig(
        l_freq=None,
        h_freq=None,
        notch_frequency=None,
        ica_enabled=False,
    )
    observed: dict[str, int] = {}

    def continuous_filter(values, settings):
        observed["samples"] = values.shape[1]
        return values

    monkeypatch.setattr("useeg.preprocessing._array_filter", continuous_filter)
    result = preprocess_run(raw, "synthetic", config)
    assert observed["samples"] == 420
    assert result.X.shape == (2, 128, 2)
    assert result.y.tolist() == [0, 1]
    assert result.trial_ids.tolist() == [0, 1]
    assert result.window_ids.tolist() == [0, 0]
    assert result.channel_names == ("C3", "C4")
    np.testing.assert_allclose(result.X.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(result.X.std(axis=1), 1.0, atol=1e-6)


def test_config_resolves_dataset_filter_and_window_override() -> None:
    mapping = {
        "sfreq": 128,
        "filters": {
            "ssvep": {"l_freq": 4.0, "h_freq": 14.0},
            "nakanishi2015": {"l_freq": 8.0, "h_freq": 16.0},
            "phase": "zero-double",
        },
        "window": {
            "size_samples": 128,
            "hop_samples": 64,
            "overrides": {
                "nakanishi2015": {
                    "mode": "nonoverlapping_full_windows_and_right_padded_tail",
                    "hop_samples": 128,
                    "include_partial_tail": True,
                }
            },
        },
        "standardization": {"epsilon": 1e-8},
    }
    config = PreprocessingConfig.from_mapping(mapping, "nakanishi2015", seed=7)
    assert (config.l_freq, config.h_freq) == (8.0, 16.0)
    assert config.filter_phase == "zero-double"
    assert config.window_mode == "nonoverlapping_full_windows_and_right_padded_tail"
    assert config.hop_samples == 128
    assert config.include_partial_tail
    assert config.epsilon == 1e-8
