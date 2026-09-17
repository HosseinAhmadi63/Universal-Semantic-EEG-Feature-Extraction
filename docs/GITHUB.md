# GitHub repository contract

The intended public repository is:

`https://github.com/HosseinAhmadi63/Universal-Semantic-EEG-Feature-Extraction`

The default branch is `main`. The repository description is:

> Complete deterministic Python and PyCharm reproduction pipeline for universal task-independent semantic EEG feature extraction across MI, SSVEP, and P300 datasets.

Recommended repository topics are `eeg`, `bci`, `semantic-features`, `representation-learning`, `transformer`, `autoencoder`, `moabb`, `pytorch`, and `reproducible-research`.

## Tracked content

The public repository includes source code, tests, frozen configuration, documentation, shared PyCharm configurations, paper-source CSV files, citation metadata, and continuous integration. It excludes raw EEG, processed windows, model checkpoints, generated semantic features, generated figures, verification reports, local environments, editor state, and credentials.

## Release identity

The initial release is `v1.0.0`, matching `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md`. `CITATION.cff` cites the final Journal of Neural Engineering article and DOI. The software uses the MIT License; article and dataset rights remain separate.

## Continuous integration

The workflow installs Python 3.11 and the pinned environment, runs Ruff, executes the test suite, verifies the paper configuration and publication sources, and runs the deterministic synthetic smoke pipeline. It does not download external EEG datasets or run full model training.

## Publication-source integrity

The files under `results/publication/source/` are direct article transcriptions. `source_manifest.csv` records their SHA-256 hashes. A public release retains these files unchanged and writes all fresh results under ignored generated paths.
