# Result schema

## Path identity

`<run-key>` is the first twelve hexadecimal characters of the SHA-256 digest of the normalized effective configuration. Dataset and subject selections are part of the digest. All generated files use UTF-8, NumPy `.npy` or `.npz`, CSV without an index column, or JSON with sorted keys.

## Download receipts

```text
data/raw/<dataset>/download.json
```

Required fields:

| Field | Meaning |
|---|---|
| `schema_version` | Receipt schema version |
| `dataset` | Configuration key |
| `paper_name` | Name printed in the article |
| `moabb_class` | Pinned MOABB dataset class |
| `subjects` | Selected subject identifiers |
| `moabb_version` | MOABB version used to access the data |
| `created_at` | UTC acquisition timestamp |
| `downloaded_at` | UTC acquisition timestamp retained for explicit download lineage |
| `force_update` | Whether the acquisition was explicitly refreshed |
| `synthetic` | Whether the receipt describes the local synthetic integration source |
| `versions` | MNE, MOABB, NumPy, scikit-learn, and SciPy versions |

## Processed datasets

```text
data/processed/<run-key>/<dataset>/X.npy
data/processed/<run-key>/<dataset>/y.npy
data/processed/<run-key>/<dataset>/metadata.npy
data/processed/<run-key>/<dataset>/manifest.json
```

`X.npy` is float32 with shape `N x 128 x C`, where `N` is the number of windows and `C` is the selected EEG channel count. `y.npy` is int64 with shape `N`. Labels are contiguous from zero according to `configs/paper.yaml`.

`metadata.npy` is a structured NumPy array containing one record per window:

| Column | Meaning |
|---|---|
| `subject` | MOABB subject identifier |
| `session` | Source session identifier |
| `run` | Source run identifier |
| `trial` | Trial identifier within the source run |
| `window` | Window identifier within the trial |
| `event_code` | Integer event code in the source recording |
| `label` | Canonical event name |
| `window_start` | Window start in the cropped 128 Hz trial |
| `valid_samples` | Non-padding samples in the window |

`manifest.json` records the run key, dataset selection, source receipt hash, preprocessing parameters, ICA exclusions, shapes, dtypes, label counts, subjects, sessions, runs, and SHA-256 digests.

## Autoencoder

```text
results/runs/<run-key>/<dataset>/autoencoder/split_indices.npz
results/runs/<run-key>/<dataset>/autoencoder/history.csv
results/runs/<run-key>/<dataset>/autoencoder/best.pt
results/runs/<run-key>/<dataset>/autoencoder/validation_reconstruction_errors.npy
results/runs/<run-key>/<dataset>/autoencoder/complete.json
```

`split_indices.npz` contains int64 arrays `train` and `validation`. `history.csv` columns are `epoch`, `train_loss`, `validation_loss`, and `learning_rate`. Reconstruction errors are float32 per-window MSE values. `complete.json` records sample counts, channel count, best epoch, best validation loss, mean and standard deviation of validation MSE, threshold status, source manifest, and environment.

## Semantic features

```text
results/runs/<run-key>/<dataset>/features/latent_maps.npy
results/runs/<run-key>/<dataset>/features/window_vectors.npy
results/runs/<run-key>/<dataset>/features/subject_vectors.csv
results/runs/<run-key>/<dataset>/features/complete.json
```

`latent_maps.npy` is float32 with shape `N x 128 x 128`. `window_vectors.npy` is float32 with shape `N x 128` and is the temporal mean of each latent map. `subject_vectors.csv` has `subject`, `windows`, and columns `feature_000` through `feature_127`; each feature row has unit L2 norm unless its unnormalized vector is zero.

## Classification

```text
results/runs/<run-key>/<dataset>/classification/folds/fold_<NN>/indices.npz
results/runs/<run-key>/<dataset>/classification/folds/fold_<NN>/history.csv
results/runs/<run-key>/<dataset>/classification/folds/fold_<NN>/best.pt
results/runs/<run-key>/<dataset>/classification/fold_metrics.csv
results/runs/<run-key>/<dataset>/classification/fold_assignments.csv
results/runs/<run-key>/<dataset>/classification/predictions.npz
results/runs/<run-key>/<dataset>/classification/complete.json
```

Each fold `indices.npz` contains disjoint int64 `train`, `validation`, and `test` arrays. `fold_metrics.csv` contains:

| Column | Meaning |
|---|---|
| `dataset` | Dataset key |
| `fold` | One-based fold number |
| `fold_name` | Held-out subject or fold label |
| `held_out_subject` | MI subject; empty for SSVEP/ERP |
| `train_samples` | Outer-train samples after inner validation split |
| `validation_samples` | Inner validation samples |
| `test_samples` | Outer-test samples |
| `best_epoch` | One-based selected training epoch |
| `accuracy` | Fraction in `[0,1]` |
| `accuracy_percent` | Percentage in `[0,100]` |
| `auc` | ERP ROC-AUC fraction |
| `auc_percent` | ERP ROC-AUC percentage |
| `macro_ovr_auc` | MI/SSVEP macro one-vs-rest AUC fraction |
| `macro_ovr_auc_percent` | MI/SSVEP macro one-vs-rest AUC percentage |

Only columns applicable to the paradigm are populated. `fold_assignments.csv` extends processed metadata with `fold`, `target`, and `prediction`. `predictions.npz` contains `targets`, `predictions`, `probabilities`, and `folds`. Binary probabilities have shape `N`; multiclass probabilities have shape `N x K`.

`complete.json` records paradigm, fold count, sample count, primary metric, mean, population standard deviation, minimum, and maximum.

## Ablation

```text
results/runs/<run-key>/bi2015b/ablation/<role>/history.csv
results/runs/<run-key>/bi2015b/ablation/<role>/best.pt
results/runs/<run-key>/bi2015b/ablation/<role>/reconstruction_errors.npy
results/runs/<run-key>/bi2015b/ablation/summary.csv
results/runs/<run-key>/bi2015b/ablation/split_indices.npz
results/runs/<run-key>/bi2015b/ablation/complete.json
```

`summary.csv` columns are `configuration`, `total_layers`, `encoder_layers`, `decoder_layers`, `attention_heads`, `validation_samples`, `mean_mse`, `standard_deviation`, and `best_epoch`.

## Analyses

```text
results/publication/generated/<run-key>/analysis/classification_summary.csv
results/publication/generated/<run-key>/analysis/reconstruction_examples.csv
results/publication/generated/<run-key>/analysis/reconstruction_examples.npz
results/publication/generated/<run-key>/analysis/correlation/subject_<NN>.npy
results/publication/generated/<run-key>/analysis/feature_selection.csv
results/publication/generated/<run-key>/analysis/anova.csv
results/publication/generated/<run-key>/analysis/pca_kmeans.csv
results/publication/generated/<run-key>/analysis/hierarchy_linkage.npy
results/publication/generated/<run-key>/analysis/hierarchy_samples.npz
results/publication/generated/<run-key>/analysis/silhouette.csv
results/publication/generated/<run-key>/analysis/clustering_summary.json
results/publication/generated/<run-key>/analysis/tsne_<dataset>.csv
results/publication/generated/<run-key>/analysis/roc_bciciv_2a.csv
results/publication/generated/<run-key>/analysis/roc_bciciv_2a_auc.csv
results/publication/generated/<run-key>/analysis/complete.json
```

`feature_selection.csv` contains subject, threshold, retained count, and retained zero- and one-based feature IDs. `anova.csv` contains subject, threshold, feature IDs, F-statistic, p-value, and significance. PCA, K-means, hierarchy, silhouette, and t-SNE outputs preserve the sample indices and targets or subjects used for display.

## Publication output

```text
results/publication/generated/<run-key>/tables/generated_dataset_summary.csv
results/publication/generated/<run-key>/tables/paper_comparison.csv
results/publication/generated/<run-key>/figures/figure_<NN>_<name>.png
results/publication/generated/<run-key>/figures/figure_<NN>_<name>.pdf
results/publication/generated/<run-key>/figures/figure_<NN>_<name>.svg
results/publication/generated/<run-key>/manifest.json
```

`paper_comparison.csv` uses separate columns for `generated_value`, `paper_value`, `difference`, `metric`, `unit`, `dataset`, `subject`, `fold`, and `source_file`. Paper percentages remain percentages; generated fractional metrics are converted before comparison.

`generated_dataset_summary.csv` contains dataset key, paper name, paradigm, primary metric, unit, folds, samples, generated mean, population standard deviation, minimum, and maximum. The root manifest records the run key, profile, relative path, byte size, and SHA-256 digest for every generated analysis, table, and figure, plus the immutable source digests.

The figure manifest records every image's relative path, source-data paths, plotting parameters, and SHA-256 digest. Raster figures use the configured DPI; PDF and SVG files are vector outputs.

## Verification report

```text
results/verification/<run-key>/verification.json
```

The report contains `status`, `run_key`, `configuration`, `source_integrity`, `dataset_contract`, `preprocessing_contract`, `model_contract`, `split_contract`, `metric_contract`, `output_contract`, `environment`, and `checks`. Every check has a stable name, pass/fail status, and observed value. A failed required check produces a nonzero process exit status.
