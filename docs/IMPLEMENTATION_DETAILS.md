# Implementation details

## Scope

The repository implements the method described in *Universal semantic feature extraction from EEG signals: a task-independent framework*. 

## Dataset resolution

The paper names twelve public datasets. They resolve to MOABB 1.4.3 classes as follows:

| Paper name | MOABB class | Paradigm |
|---|---|---|
| BCICIV_2a | `BNCI2014_001` | Motor imagery |
| BCICIV_2b | `BNCI2014_004` | Motor imagery |
| Lee2019_SSVEP | `Lee2019_SSVEP` | SSVEP |
| Nakanishi2015 | `Nakanishi2015` | SSVEP |
| BI2012 | `BI2012` | P300 |
| BI2013a | `BI2013a` | P300 |
| BI2014b | `BI2014b` | P300 |
| BI2015a | `BI2015a` | P300 |
| BI2015b | `BI2015b` | P300 |
| BNCI2014_008 | `BNCI2014_008` | P300 |
| BNCI2014_009 | `BNCI2014_009` | P300 |
| Sosulski2019 | `Sosulski2019` | P300 |

Lee2019_SSVEP uses the training run, sessions 1 and 2, with the test run and resting-state data disabled. BI2012 uses training data. BI2013a uses non-adaptive training data. Sosulski2019 uses the 60 ms stimulus-onset asynchrony, does not convert SOAs into sessions, retains non-IID trials, and loads the -0.2 to 1.0 s interval.

The repository starts at the interval declared by the MOABB dataset and crops to the paper duration. Dataset manifests record actual subjects, sessions, runs, events, and counts obtained from the pinned MOABB release. Printed counts in Tables 1-3 remain documentary targets and do not override the downloaded dataset structure.

The MI totals in Table 1 depend on treating the printed trials-per-class figures as repeated within the listed session and run structure. They should not be used as array-allocation constants. The BI2015b discussion reports 116,160 total trials, which equals 44 subjects multiplied by 2,160 non-target plus 480 target trials. The repository derives actual counts from the loaded event records and reports any difference from the table.

## Preprocessing order

Each raw run is processed independently in this order:

1. Resample each continuous raw run to 128 Hz.
2. Apply the paradigm-specific band-pass filter to the continuous run.
3. Apply the 50 Hz notch filter to the continuous run.
4. Fit and apply ICA to the continuous run.
5. Crop each event at the MOABB-declared start for the paper-reported duration.
6. Segment into 128-sample windows.
7. Standardize every window and channel across its 128 time samples.

### Filters

The passbands are 6-32 Hz for MI, 0.1-30 Hz for ERP, 4-14 Hz for Lee2019_SSVEP, and 8-16 Hz for Nakanishi2015. Band-pass and notch operations use MNE FIR filters with `firwin`, a Hamming window, `zero-double` phase, automatic filter length and transition bandwidths, and reflect-limited padding. The notch is centered at 50 Hz with a 0.25 Hz notch width and 1 Hz transition bandwidth.


### ICA

FastICA is fitted once per raw run after filtering. It retains enough principal components to explain 99% variance, uses random seed 42, and uses MNE's automatic iteration limit. EOG- and ECG-correlated components are removed through the pinned MNE detection routines only when compatible reference channels exist. No component is removed solely from kurtosis or visual inspection, and no component is removed when reference channels are absent.


### Windowing and Table 4

The general method specifies 1 s windows with 50% overlap. At 128 Hz this gives 128 samples and a 64-sample hop. Only complete windows are kept under the general rule.

Table 4 gives a conflicting dataset-specific segmentation description. The publication profile gives Table 4 precedence for its four datasets:

| Dataset | Frozen rule |
|---|---|
| BNCI2014_009 | One 0.8 s segment right-padded to 1 s |
| Sosulski2019 | Consecutive non-overlapping 1 s windows plus the 0.2 s residual right-padded to 1 s |
| Nakanishi2015 | Four consecutive non-overlapping 1 s windows plus the 0.15 s residual right-padded to 1 s |
| BCICIV_2b | Four consecutive non-overlapping 1 s windows plus the 0.5 s residual right-padded to 1 s |

This choice follows the explicit Table 4 durations while retaining all samples. It means the four overrides do not use the general 50% overlap rule.

### Standardization

Every segmented window is independently standardized for each channel across time. The transformation is `(x - mean) / max(std, 1e-8)`. This is the narrowest literal interpretation of the paper's statement that signals were z-score normalized per channel. It prevents statistics from one subject or classifier fold from entering another window, but it also removes window-level channel offsets and scales.

## Hierarchical autoencoder

### Input and convolutional encoder

Processed arrays and the public model interface use `batch x 128 time samples x channels`. The convolutional encoder transposes this to channel-first form internally. Two same-padded temporal convolutions map channels to 64 and then 128 feature channels. Both use kernel size 3 and ReLU. The tensor is transposed back to `batch x 128 time positions x 128 features`, normalized with LayerNorm epsilon `1e-5`, and projected by a 128-unit linear layer. Fixed sinusoidal positional encoding is added.

### Transformer encoder

The encoder contains two post-normalization blocks. Each block uses eight-head self-attention at model width 128 and a position-wise feed-forward network of width 512. Attention and feed-forward dropout are zero because the article does not state Transformer dropout. Residual connections and LayerNorm follow the equations in Section 4. The encoder output `Z` has shape `batch x 128 x 128`.

### Transformer decoder

The decoder contains two blocks with the same dimensions and self-attention structure. It has no encoder-decoder cross-attention. 


### Convolutional decoder

A 128-unit linear layer precedes two same-padded transposed temporal convolutions with 128 and 64 output channels, kernel size 3, and ReLU. LayerNorm is applied to the 64 decoded features at each time position. A linear kernel-size-one convolution maps 64 features back to the dataset's EEG channel count. The public output is transposed back to the original `batch x 128 time samples x channels` shape.

### Autoencoder training

One autoencoder is trained for each dataset. All processed windows for that dataset are available before the downstream classifier cross-validation. The autoencoder therefore follows a transductive feature-learning interpretation rather than being refitted inside every classifier fold.

Windows are shuffled with seed 42 and divided 90:10 into training and validation sets. Training uses MSE, Adam with learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, no weight decay, batch size 64, and at most 100 epochs. Early stopping has patience 5 and restores the best validation checkpoint. ReduceLROnPlateau uses factor 0.5, patience 3, relative threshold `1e-4`, and minimum learning rate `1e-7`.


The paper treats dataset-level MSE below 0.01 as acceptable. Figure 5 contains an individual example error of 0.0179; the repository distinguishes per-window errors from the dataset-level threshold.

## Semantic representations

Three representations are saved:

- The complete encoder map `Z`, with shape `128 x 128` per window.
- The temporal mean of `Z`, a 128-value vector per window.
- The mean of all window vectors for one subject, L2-normalized to unit length.

The first two follow Equations 19 and 20. The third follows Equations 21 and 22. The complete latent map is retained because the article's downstream CNN is described as learning spatial-temporal patterns and does not state that classification uses only the temporal mean.

## Downstream classifier

The complete latent map is treated as a single-channel `128 x 128` image. Four same-padded 3 x 3 convolutional blocks use 64, 128, 256, and 512 filters with batch normalization and ReLU. Two-by-two max pooling follows the first three blocks. After the fourth block, channel-wise average and maximum maps are concatenated and passed through a same-padded 7 x 7 convolution with sigmoid activation to form a spatial-attention mask. Global average pooling, dropout 0.5, and the output layer complete the classifier.

MI and SSVEP use one logit per class with cross-entropy and softmax probabilities. ERP uses one logit with binary focal loss. Focal gamma is 2.0. Alpha is the negative-class fraction calculated from the training portion of each fold.

Classifier optimization reuses the autoencoder training settings: Adam at `1e-4`, batch 64, at most 100 epochs, 10% stratified inner validation, patience-5 early stopping with best-checkpoint restoration, and the same plateau scheduler.

## Evaluation

Motor-imagery datasets use leave-one-subject-out evaluation. The SSVEP and ERP datasets use shuffled, stratified eight-fold cross-validation over windows with seed 42. Training windows inside each classifier fold receive a separate stratified 10% validation split. MI and SSVEP are evaluated with window-level accuracy. ERP is evaluated with window-level ROC-AUC using Target as the positive class.

The autoencoder is trained before classifier folds, so classifier test-fold windows contribute to unsupervised representation learning. This matches the most direct reading of the two-stage workflow but is transductive. The hashed configuration and verification report record that scope.

## Analyses

### Reconstruction and learning curves

Learning curves are generated for BCICIV_2b, BI2015a, and Lee2019_SSVEP. Reconstruction examples use BCICIV_2b subject 6 channel 2, BNCI2014_008 subject 4 channel 3, and Nakanishi2015 subject 7 channel 6. The article identifies these subjects and channels. The repository samples three windows without replacement using seed 42.

### Correlation filtering and ANOVA

For each BCICIV_2a subject, Pearson correlation is calculated across window-level 128-value features, producing a `128 x 128` matrix. At thresholds 0.80, 0.60, and 0.40, absolute correlations are traversed through the upper triangle in ascending feature order. When a pair exceeds the threshold, the higher-indexed feature is removed. 

One-way `f_classif` ANOVA uses class labels. Figure 10 prints different F-statistics and p-values for the same seven feature IDs at three correlation thresholds. Plain column filtering cannot change a retained feature's univariate F-statistic when the underlying samples remain identical. The printed values are preserved as immutable publication targets, while generated ANOVA values are computed from the actual retained columns and are not forced to match those numbers.

### PCA, clustering, silhouette, and hierarchy

BCICIV_2a window vectors are projected to two dimensions with PCA. K-means uses four clusters, 20 initializations, and seed 42. Ward hierarchical clustering uses Euclidean distance, and the displayed cut threshold is 7.5. Silhouette values use Euclidean distance.

### t-SNE

The representative datasets are BCICIV_2b, Sosulski2019, and Nakanishi2015. Raw inputs are flattened standardized windows; latent inputs are temporal-mean vectors. Points are colored by subject as described in the article. A deterministic subject-stratified sample of at most 5,000 windows is reduced to at most 50 PCA dimensions, then embedded with two-dimensional t-SNE, perplexity 30, PCA initialization, automatic learning rate, 1,000 iterations, and seed 42.

### Transformer ablation

The BI2015b ablation evaluates the configurations `(4 layers, 8 heads)`, `(4 layers, 4 heads)`, and `(2 layers, 8 heads)`. Four layers means two encoder plus two decoder blocks; two layers means one plus one. A deterministic 10% window sample is used for evaluation. 
