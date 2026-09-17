# Data

The repository downloads public EEG recordings through MOABB 1.4.3. Raw archives are stored below `data/raw/`; deterministic processed arrays are stored below `data/processed/<run-key>/`. Neither directory is committed.

| Configuration key | Paper name | MOABB class | Paradigm | Repository selection |
|---|---|---|---|---|
| `bnci2014_001` | BCICIV_2a | `BNCI2014_001` | MI | All nine subjects; left hand, right hand, feet, tongue |
| `bnci2014_004` | BCICIV_2b | `BNCI2014_004` | MI | All nine subjects; left hand and right hand |
| `lee2019_ssvep` | Lee2019_SSVEP | `Lee2019_SSVEP` | SSVEP | Training run, sessions 1 and 2, no test run, no resting state |
| `nakanishi2015` | Nakanishi2015 | `Nakanishi2015` | SSVEP | All nine subjects and twelve stimulation frequencies |
| `bi2012` | BI2012 | `BI2012` | P300 | Training data only |
| `bi2013a` | BI2013a | `BI2013a` | P300 | Non-adaptive training data only |
| `bi2014b` | BI2014b | `BI2014b` | P300 | Complete public dataset |
| `bi2015a` | BI2015a | `BI2015a` | P300 | Complete public dataset |
| `bi2015b` | BI2015b | `BI2015b` | P300 | Complete public dataset |
| `bnci2014_008` | BNCI2014_008 | `BNCI2014_008` | P300 | Complete public dataset |
| `bnci2014_009` | BNCI2014_009 | `BNCI2014_009` | P300 | Complete public dataset |
| `sosulski2019` | Sosulski2019 | `Sosulski2019` | P300 | 60 ms SOA, SOAs not treated as sessions, non-IID trials retained, interval -0.2 to 1.0 s |

The event interval starts at the interval declared by the corresponding MOABB dataset and is cropped to the duration printed in Tables 1-3 of the article. 

The canonical label mapping is fixed in `configs/paper.yaml`. MI labels are left hand, right hand, feet, and tongue for BCICIV_2a and left hand/right hand for BCICIV_2b. SSVEP labels follow ascending stimulus frequency. ERP labels are `NonTarget=0` and `Target=1`.

Every processed dataset directory contains arrays, metadata, and a manifest recording source identity, preprocessing parameters, software versions, configuration hash, shape, label counts, subjects, sessions, runs, and file digests. Download receipts record the MOABB class and local acquisition time without copying private paths into publication outputs.

The datasets are not relicensed by this repository. Their original provider terms and required citations apply.
