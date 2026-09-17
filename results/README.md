# Results

Results are divided into immutable paper sources and generated artifacts.

## Publication sources

`results/publication/source/` contains values transcribed from the published article:

- Tables 5 and 6: subject-level and average MI accuracy percentages.
- Table 7: SSVEP accuracy percentages.
- Table 8: P300 ROC-AUC percentages.
- Abstract headline values.
- Section 6.2 reconstruction extremes.
- Figure 5 example reconstruction errors.
- Figure 7 class-wise AUC values.
- Figure 9 retained-feature counts.
- Figure 10 ANOVA values.
- Figure 14 evaluated Transformer configurations.

`source_manifest.csv` records a SHA-256 digest, article location, and measurement unit for every source file. The verification command checks these files before analysis.

## Generated results

Fresh artifacts are written under a deterministic configuration key:

```text
results/runs/<run-key>/
results/publication/generated/<run-key>/
results/verification/<run-key>/
```

`results/runs/` contains checkpoints, histories, semantic features, fold assignments, predictions, metrics, and ablation runs. `results/publication/generated/` contains numerical analyses, publication tables, comparisons with the transcribed sources, and paper-equivalent figures. `results/verification/` contains machine-readable validation reports.

Generated artifacts never replace `results/publication/source/`. A smoke run, subject subset, dataset subset, or changed parameter set receives a different key and cannot overwrite the paper-profile run.

Detailed columns, units, shapes, and provenance fields are defined in [../docs/RESULT_SCHEMA.md](../docs/RESULT_SCHEMA.md).
