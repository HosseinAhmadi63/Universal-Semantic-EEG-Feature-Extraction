# Configuration files

`paper.yaml` is the only publication reproduction profile. It contains the twelve paper datasets, the fixed preprocessing and segmentation rules, the complete autoencoder and classifier architectures, evaluation procedures, analysis settings, and publication checks.

`smoke.yaml` preserves the same tensor interfaces and deterministic seed while using generated eight-channel EEG for two synthetic subjects, four trials per class, one training epoch, disabled ICA, and reduced analysis workloads. The paper profile retains the complete FastICA procedure. The smoke profile is an integration check rather than a paper result and requires no external download.

Every run receives the first twelve hexadecimal characters of the SHA-256 digest of its normalized configuration and dataset selection. Processed data and result directories use that key, preventing a smoke run or a subset run from overwriting a complete paper run.
