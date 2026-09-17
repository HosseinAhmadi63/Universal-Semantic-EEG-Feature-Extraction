# Universal Semantic EEG Feature Extraction

[![Tests](https://github.com/HosseinAhmadi63/Universal-Semantic-EEG-Feature-Extraction/actions/workflows/tests.yml/badge.svg)](https://github.com/HosseinAhmadi63/Universal-Semantic-EEG-Feature-Extraction/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the complete Python and PyCharm reproduction pipeline for:

> **Universal semantic feature extraction from EEG signals: a task-independent framework**  
> Hossein Ahmadi and Luca Mesin  
> *Journal of Neural Engineering*, volume 22, issue 3, article 036003, 2025  
> [https://doi.org/10.1088/1741-2552/add08f](https://doi.org/10.1088/1741-2552/add08f)

The pipeline downloads twelve public motor-imagery, steady-state visually evoked potential, and P300 datasets through MOABB; applies the paper's paradigm-specific preprocessing and segmentation; trains one hierarchical convolutional and Transformer autoencoder per dataset; extracts window-level and subject-level 128-dimensional semantic representations; evaluates the reported downstream classifiers; runs correlation, ANOVA, PCA, K-means, hierarchical clustering, silhouette, t-SNE, and ablation analyses; and generates paper-equivalent tables and figures.

If you use this repository, its code, or its results, cite the article above. Machine-readable citation metadata are in [CITATION.cff](CITATION.cff).


## Frozen pipeline

| Stage | Repository implementation |
|---|---|
| Datasets | BCICIV_2a, BCICIV_2b, Lee2019_SSVEP, Nakanishi2015, BI2012, BI2013a, BI2014b, BI2015a, BI2015b, BNCI2014_008, BNCI2014_009, and Sosulski2019 through MOABB 1.4.3 |
| Resampling | 128 Hz |
| Band-pass | MI 6-32 Hz; ERP 0.1-30 Hz; Lee2019_SSVEP 4-14 Hz; Nakanishi2015 8-16 Hz |
| Line-noise removal | 50 Hz linear-phase FIR notch |
| Artifact handling | FastICA per raw run; EOG- and ECG-correlated components removed only when matching reference channels exist |
| Segmentation | 1 s windows; 50% overlap ordinarily; the four Table 4 datasets use the stated non-overlapping and right-padding rules |
| Normalization | Per-window, per-channel z-score across time |
| Autoencoder | Conv1d 64/128, two-layer eight-head Transformer encoder, two-layer self-attention-only Transformer decoder, ConvTranspose1d 128/64, latent width 128 |
| Autoencoder training | Per dataset, random 90:10 window split, Adam at `1e-4`, batch 64, MSE, 100 epochs, early stopping, plateau scheduler |
| Semantic feature | Full `128 x 128` latent map; temporal mean gives a 128-value window vector; subject mean followed by L2 normalization gives a subject vector |
| Classifier | Four-block 2D CNN with 64/128/256/512 filters, spatial attention, global average pooling, dropout 0.5 |
| Evaluation | Subject LOSO for MI; shuffled stratified eight-fold window evaluation for SSVEP and ERP |
| Metrics | Accuracy for MI and SSVEP; ROC-AUC for ERP |
| Analyses | Correlation thresholds 0.80/0.60/0.40, ANOVA, PCA, K-means, Ward hierarchy, silhouette, t-SNE, and BI2015b Transformer ablation |

## Repository contents

```text
.github/workflows/tests.yml                 Continuous integration
.run/                                      Shared PyCharm run configurations
configs/paper.yaml                         Complete frozen publication profile
configs/smoke.yaml                         Reduced deterministic integration profile
data/README.md                             Dataset identities, selections, and local paths
docs/IMPLEMENTATION_DETAILS.md             Full protocol and frozen decisions
docs/PAPER_TO_CODE.md                      Article component to implementation map
docs/PYCHARM.md                            PyCharm interpreter and launcher setup
docs/REPRODUCIBILITY.md                    Determinism and traceability contract
docs/RESULT_SCHEMA.md                      Generated artifact definitions
main.py                                    Single PyCharm and command-line entry point
scripts/                                   Direct stage launchers
src/useeg/                                 Installable implementation package
tests/                                     Scientific and implementation checks
results/publication/source/                Immutable values transcribed from the paper
results/publication/generated/<run-key>/   Fresh tables and paper-equivalent figures
results/runs/<run-key>/                    Models, features, predictions, and ablation runs
```

## Installation

Python 3.11 is the reference interpreter.

macOS or Linux:

```bash
git clone https://github.com/HosseinAhmadi63/Universal-Semantic-EEG-Feature-Extraction.git
cd Universal-Semantic-EEG-Feature-Extraction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
git clone https://github.com/HosseinAhmadi63/Universal-Semantic-EEG-Feature-Extraction.git
cd Universal-Semantic-EEG-Feature-Extraction
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Conda:

```bash
conda env create -f environment.yml
conda activate useeg
```

## Verify the installation

The verifier checks configuration invariants, paper-source checksums, dataset definitions, architecture dimensions, loss functions, deterministic settings, and reference table schemas without training the full models.

```bash
python main.py verify --config configs/paper.yaml
python -m pytest
```

Run the reduced end-to-end profile on CPU:

```bash
python main.py run --config configs/smoke.yaml --force --verbose
```

Smoke outputs use their own configuration key and cannot overwrite paper-profile artifacts.

## PyCharm

Open the repository root in PyCharm, create a Python 3.11 virtual environment at `.venv`, install the pinned requirements, and select one of the committed configurations in `.run/`. The principal configurations are **USEEG Verify**, **USEEG Smoke**, **USEEG Complete Reproduction**, and **USEEG Figures**. Every launcher uses the repository root as its working directory and the selected project interpreter.

Detailed setup and all stage launchers are listed in [docs/PYCHARM.md](docs/PYCHARM.md).

## Complete reproduction

Run the complete ordered pipeline:

```bash
python main.py run --config configs/paper.yaml --verbose
```

The ordered stages are:

```bash
python main.py download --config configs/paper.yaml --verbose
python main.py preprocess --config configs/paper.yaml --verbose
python main.py train-autoencoder --config configs/paper.yaml --verbose
python main.py extract --config configs/paper.yaml --verbose
python main.py classify --config configs/paper.yaml --verbose
python main.py analyze --config configs/paper.yaml --verbose
python main.py figures --config configs/paper.yaml --verbose
```

The pipeline is resumable. Completed keyed stages are reused, and processed caches additionally validate their configuration fingerprint and array contract. `--force` replaces artifacts only for the selected configuration key.

## Outputs

The normalized configuration and dataset selection receive a deterministic twelve-character SHA-256 key.

```text
data/raw/<dataset>/download.json
data/processed/<run-key>/<dataset>/X.npy
data/processed/<run-key>/<dataset>/y.npy
data/processed/<run-key>/<dataset>/metadata.npy
data/processed/<run-key>/<dataset>/manifest.json
results/runs/<run-key>/<dataset>/autoencoder/history.csv
results/runs/<run-key>/<dataset>/autoencoder/best.pt
results/runs/<run-key>/<dataset>/features/latent_maps.npy
results/runs/<run-key>/<dataset>/features/window_vectors.npy
results/runs/<run-key>/<dataset>/features/subject_vectors.csv
results/runs/<run-key>/<dataset>/classification/fold_assignments.csv
results/runs/<run-key>/<dataset>/classification/fold_metrics.csv
results/runs/<run-key>/<dataset>/classification/predictions.npz
results/publication/generated/<run-key>/analysis/
results/publication/generated/<run-key>/tables/
results/publication/generated/<run-key>/figures/*.{png,pdf,svg}
results/publication/generated/<run-key>/tables/paper_comparison.csv
results/verification/<run-key>/verification.json
```

The complete schemas are in [docs/RESULT_SCHEMA.md](docs/RESULT_SCHEMA.md).


## Data and licenses

No EEG recording is committed. MOABB downloads the original public datasets into `data/raw/`, and processed arrays are stored below `data/processed/<run-key>/`. The original dataset licenses, access conditions, and citation requirements remain controlling. [data/README.md](data/README.md) identifies every dataset and the exact repository selection applied here.

The paper profile records `data.accept_dataset_terms: true`. Running its download stage confirms acceptance of provider terms presented by MOABB.

## Citation

```bibtex
@article{Ahmadi2025Universal,
  author  = {Ahmadi, Hossein and Mesin, Luca},
  title   = {Universal semantic feature extraction from EEG signals: a task-independent framework},
  journal = {Journal of Neural Engineering},
  year    = {2025},
  volume  = {22},
  number  = {3},
  pages   = {036003},
  doi     = {10.1088/1741-2552/add08f}
}
```

## License

The original source code in this repository is released under the [MIT License](LICENSE). The article, public EEG datasets, and downloaded MOABB archives have separate licenses and terms.
