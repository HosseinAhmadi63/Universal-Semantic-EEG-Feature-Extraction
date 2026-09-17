# Paper-to-code map

| Article component | Configuration | Implementation | Entry point | Principal output |
|---|---|---|---|---|
| Tables 1-3 datasets | `datasets`, `data` | `src/useeg/datasets.py`, `src/useeg/data.py` | `download` | Download receipts and dataset manifests |
| Section 3.3 preprocessing | `preprocessing` | `src/useeg/preprocessing.py` | `preprocess` | Processed arrays, metadata, and manifests |
| Table 4 segmentation | `preprocessing.window.overrides` | `src/useeg/preprocessing.py` | `preprocess` | Window lineage and padding counts |
| Sections 3.1 and 4.1 CNN encoder | `autoencoder.encoder_*` | `src/useeg/models.py` | `train-autoencoder` | Autoencoder checkpoint and history |
| Sections 4.1-4.2 Transformer encoder | `autoencoder.d_model`, `nhead`, `num_encoder_layers`, `dim_feedforward` | `src/useeg/models.py` | `train-autoencoder` | Latent encoder map |
| Sections 4.2-4.3 Transformer and CNN decoder | `autoencoder.num_decoder_layers`, `decoder_*` | `src/useeg/models.py` | `train-autoencoder` | Reconstructions and validation MSE |
| Section 3.4 training | `autoencoder` optimization fields | `src/useeg/training.py`, `src/useeg/experiment.py` | `train-autoencoder` | Best checkpoint, split, and learning history |
| Equations 19-22 semantic features | `feature_extraction` | `src/useeg/experiment.py` | `extract` | Latent maps, window vectors, subject vectors |
| Section 3.5 downstream CNN | `classifier` | `src/useeg/models.py`, `src/useeg/training.py` | `classify` | Fold checkpoints and histories |
| Section 3.5 evaluation | `evaluation` | `src/useeg/splits.py`, `src/useeg/metrics.py`, `src/useeg/experiment.py` | `classify` | Fold assignments, predictions, metrics |
| Figures 2-4 learning curves | `analysis.representative_datasets.learning_curves` | `src/useeg/plotting.py` | `figures` | Learning-curve figures |
| Figure 5 reconstruction | `figures.reconstruction_examples` | `src/useeg/analysis.py`, `src/useeg/plotting.py` | `analyze`, `figures` | Example data and reconstruction figure |
| Figure 6 correlation matrices | `analysis.correlation` | `src/useeg/analysis.py`, `src/useeg/plotting.py` | `analyze`, `figures` | Subject matrices and heatmaps |
| Figure 7 aggregate ROC | `evaluation.roc_auc` | `src/useeg/metrics.py`, `src/useeg/analysis.py`, `src/useeg/plotting.py` | `classify`, `analyze`, `figures` | Out-of-fold ROC data and figure |
| Figure 8 t-SNE | `analysis.tsne` | `src/useeg/analysis.py`, `src/useeg/plotting.py` | `analyze`, `figures` | Raw and latent embeddings |
| Figures 9-10 feature filtering and ANOVA | `analysis.correlation`, `analysis.anova` | `src/useeg/analysis.py`, `src/useeg/plotting.py` | `analyze`, `figures` | Retained features, F-statistics, p-values |
| Figures 11-13 clustering | `analysis.pca`, `kmeans`, `hierarchy`, `silhouette` | `src/useeg/analysis.py`, `src/useeg/plotting.py` | `analyze`, `figures` | Cluster assignments and plots |
| Figure 14 ablation | `analysis.ablation` | `src/useeg/experiment.py`, `src/useeg/plotting.py` | `analyze`, `figures` | Configuration errors and ablation plot |
| Tables 5-8 paper values | `verification.compare_publication_tables` | `results/publication/source/*.csv`, `src/useeg/publication.py` | `verify`, `figures` | Immutable sources and generated comparison |

`main.py` delegates to `src/useeg/cli.py`. The direct files in `scripts/` delegate to the same command functions, so PyCharm, the installed `useeg` command, and individual scripts execute the same implementation.

The exact operational choices needed to turn the article into executable code are in [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md). Output columns and tensor shapes are in [RESULT_SCHEMA.md](RESULT_SCHEMA.md).
