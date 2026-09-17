# PyCharm setup

## Open the project

1. Start PyCharm and choose **Open**.
2. Select the `Universal-Semantic-EEG-Feature-Extraction` directory containing `main.py` and `pyproject.toml`.
3. Allow project indexing to complete.

## Create the interpreter

The reference interpreter is Python 3.11.

1. Open **Settings | Project | Python Interpreter**.
2. Choose **Add Interpreter | Add Local Interpreter | Virtualenv**.
3. Select Python 3.11 as the base interpreter.
4. Set the environment location to `.venv` inside the repository.
5. Open the PyCharm terminal at the repository root and run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

No environment variables or manual source-root settings are required.

## Shared run configurations

The `.run/` directory supplies project-level launchers:

| PyCharm configuration | Equivalent command | Principal output |
|---|---|---|
| `USEEG Verify` | `python main.py verify --config configs/paper.yaml` | `results/verification/<run-key>/verification.json` |
| `USEEG Smoke` | `python main.py run --config configs/smoke.yaml --force --verbose` | Reduced keyed run |
| `USEEG Download` | `python main.py download --config configs/paper.yaml --verbose` | `data/raw/` |
| `USEEG Preprocess` | `python main.py preprocess --config configs/paper.yaml --verbose` | `data/processed/<run-key>/` |
| `USEEG Train Autoencoder` | `python main.py train-autoencoder --config configs/paper.yaml --verbose` | Autoencoder histories and checkpoints |
| `USEEG Extract Features` | `python main.py extract --config configs/paper.yaml --verbose` | Latent maps and feature vectors |
| `USEEG Classify` | `python main.py classify --config configs/paper.yaml --verbose` | Fold assignments, predictions, and metrics |
| `USEEG Analyze` | `python main.py analyze --config configs/paper.yaml --verbose` | Correlation, ANOVA, clustering, and ablation outputs |
| `USEEG Figures` | `python main.py figures --config configs/paper.yaml --verbose` | Generated publication figures |
| `USEEG Complete Reproduction` | `python main.py run --config configs/paper.yaml --verbose` | Complete ordered paper run |

PyCharm imports these configurations automatically. Every configuration uses `$PROJECT_DIR$` as its working directory, the selected project interpreter, and unbuffered Python output.

## Debugging

Breakpoints can be placed directly in `src/useeg/` because the package is installed in editable mode. Stage commands reuse completed keyed upstream artifacts, so debugging one downstream stage does not repeat completed downloads or preprocessing. `--force` replaces the selected stage artifacts for its run key.

The smoke configuration is the fast integration route. It uses deterministic eight-channel synthetic EEG on CPU, two synthetic subjects, four trials per class, one training epoch, and reduced analysis sample counts while preserving the production tensor interfaces.

## Data location

MOABB archives are stored below `data/raw/`. Processed arrays and results remain inside the repository under hash-isolated ignored directories. The repository root must remain the working directory so all configured relative paths resolve consistently.
