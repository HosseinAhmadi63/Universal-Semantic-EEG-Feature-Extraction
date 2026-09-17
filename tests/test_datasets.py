from types import SimpleNamespace

import pytest

from useeg.datasets import (
    DATASET_REGISTRY,
    PUBLIC_DATASET_NAMES,
    SyntheticDataset,
    create_dataset,
    expected_trials_for_subject,
    get_dataset_spec,
    list_dataset_specs,
    selected_subjects,
)


def test_public_registry_is_complete_and_exact() -> None:
    assert len(PUBLIC_DATASET_NAMES) == 12
    assert tuple(spec.name for spec in list_dataset_specs()) == PUBLIC_DATASET_NAMES
    assert set(PUBLIC_DATASET_NAMES).issubset(DATASET_REGISTRY)
    assert get_dataset_spec("BCICIV_2a").name == "bnci2014_001"
    assert get_dataset_spec("BNCI2014_004").interval == (3.0, 7.5)
    assert get_dataset_spec("Lee2019_SSVEP").kwargs["test_run"] is False
    assert get_dataset_spec("nakanishi2015").event_mapping["14.75"] == 12
    assert get_dataset_spec("bi2013a").event_mapping == {"Target": 33285, "NonTarget": 33286}
    assert get_dataset_spec("sosulski2019").interval == (-0.2, 1.0)


def test_version_specific_constructor_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict[str, object]] = {}

    def factory(name: str):
        def construct(**kwargs: object) -> object:
            calls[name] = kwargs
            return SimpleNamespace(subject_list=[1])

        return construct

    module = SimpleNamespace(
        BI2012=factory("BI2012"),
        BI2013a=factory("BI2013a"),
        Lee2019_SSVEP=factory("Lee2019_SSVEP"),
        Sosulski2019=factory("Sosulski2019"),
    )
    monkeypatch.setattr("useeg.datasets.importlib.import_module", lambda name: module)
    create_dataset("bi2012")
    create_dataset("bi2013a")
    create_dataset("lee2019_ssvep")
    create_dataset("sosulski2019")
    assert calls["BI2012"] == {"Training": True, "Online": False}
    assert calls["BI2013a"] == {
        "NonAdaptive": True,
        "Adaptive": False,
        "Training": True,
        "Online": False,
    }
    assert calls["Lee2019_SSVEP"] == {
        "train_run": True,
        "test_run": False,
        "resting_state": False,
        "sessions": (1, 2),
    }
    assert calls["Sosulski2019"] == {
        "use_soas_as_sessions": False,
        "load_soa_60": True,
        "reject_non_iid": False,
        "interval": (-0.2, 1.0),
    }


def test_synthetic_source_never_imports_moabb(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_import(name: str) -> object:
        raise AssertionError(name)

    monkeypatch.setattr("useeg.datasets.importlib.import_module", reject_import)
    direct = create_dataset("synthetic")
    substituted = create_dataset({"name": "bnci2014_001", "moabb_class": "SyntheticEEG"})
    assert isinstance(direct, SyntheticDataset)
    assert isinstance(substituted, SyntheticDataset)
    assert substituted.subject_list == list(range(1, 10))


def test_subject_selection_and_variable_expected_counts() -> None:
    dataset = SimpleNamespace(subject_list=[1, 2, 3])
    assert selected_subjects(dataset, "all") == (1, 2, 3)
    assert selected_subjects(dataset, [3, 1]) == (3, 1)
    with pytest.raises(ValueError):
        selected_subjects(dataset, [4])
    first = expected_trials_for_subject(get_dataset_spec("bi2013a"), 1)
    eighth = expected_trials_for_subject(get_dataset_spec("bi2013a"), 8)
    assert first == {"NonTarget": 3200, "Target": 640}
    assert eighth == {"NonTarget": 400, "Target": 80}
