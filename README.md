# GPCR Binder Classifier 2.0

Predicts whether a protein–ligand pair is an active binder (activity ≤ 50 nM) from
BindingDB or Papyrus data. Ligands can be represented with MoLFormer-XL embeddings,
Morgan/Avalon fingerprints, RDKit physicochemical descriptors, or any concatenation
of those. Proteins are embedded with ESM-2. Ligand and protein vectors are cached
in LMDB stores named after the components that produced them. Models are trained
with a cold-protein split and scored by classification metrics (AUROC, AUPRC,
precision, recall, F1).

## Setup

```bash
pip install -r requirements.txt
```

The default embedding models are downloaded automatically from the Hugging Face
Hub on first use:

- Protein: `facebook/esm2_t33_650M_UR50D` (1280-d)
- Ligand: `ibm-research/MoLFormer-XL-both-10pct` (768-d, loaded with `trust_remote_code=True`)

[Papyrus](https://doi.org/10.1186/s13321-022-00672-x) support requires
`papyrus-scripts` (included in `requirements.txt`).

## Data sources

### BindingDB (default)

Training data is read from `data/train/BindingDB_all_prepared.csv`. During
preparation:

- Salts are stripped and SMILES canonicalized (RDKit largest-fragment).
- Activity values (IC50, EC50, Ki, Kd, in nM) are converted to
  `pActivity = -log10(value_nM * 1e-9)`.
- Rows with multiple assay values are exploded into one example per assay type.
- Censored values (`>` / `<`) and rows with missing SMILES/sequence/activity are
  dropped.
- `pH` (missing imputed to 7.4) and `Temp (C)` (parsed, clipped, missing imputed
  to 25.0 C) are used as scalar features. Assay type is one-hot encoded.

Labels are stored as binder / non-binder (50 nM cutoff,
`config.ACTIVITY_THRESHOLD_NM`) when features are built. Rows with an optional
`Activity Label` column (Papyrus binary builds) use that class directly.

### Papyrus

Build a BindingDB-compatible prepared CSV from the [Papyrus](https://zenodo.org/records/13787634)
dataset with `build_papyrus.py`. The script downloads (if needed), streams, filters
to exact Ki/Kd/IC50/EC50/**Other** values (`type_other` included), joins protein
sequences, and writes a CSV the training pipeline can consume.

**Papyrus++** (default, high-quality reproducible rows):

```bash
python build_papyrus.py --subset plusplus
# output: data/train/Papyrus_pp_prepared.csv
```

**Full Papyrus** quantitative build (`without_stereochemistry`; multi-GB
download, ~2.4M exact Ki/Kd/IC50/EC50/Other rows after filters):

```bash
python build_papyrus.py --subset full
# output: data/train/Papyrus_full_prepared.csv

# optional: restrict to higher-quality tiers
python build_papyrus.py --subset full --min-quality high
```

**Full + binary** (`Activity_class` rows, mostly inactive `N`, low quality).
Quant rows are written first; reuse `--limit` to cap total size (recommended —
the binary tail is ~56M rows):

```bash
python build_papyrus.py --subset full --include-binary --limit 5000000
# default output: data/train/Papyrus_full_binary_prepared.csv
# (~2.4M quant + ~2.6M binary when limit=5M)
```

Binary rows are encoded as `Other (nM)` sentinel values plus an `Activity Label`
column (`0`/`1`). Override the path with `--output` if needed.

Smoke test:

```bash
python build_papyrus.py --subset plusplus --limit 10000
# Full quant only (small limits never reach the binary region — that starts
# after ~2–3M quantitative rows):
python build_papyrus.py --subset full --limit 10000 --no-download
# Quant + binary requires --limit above the quantitative row count, e.g.:
python build_papyrus.py --subset full --include-binary --limit 3000000 --no-download
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--include-binary` | Append Papyrus `Activity_class` rows (`--subset full` only) |
| `--no-download` | Skip download; assume Papyrus data is already present |
| `--disk-margin 0` | Relax papyrus-scripts free-disk check (default) |
| `--chunk-size 500000` | Rows per streaming chunk |
| `--output PATH` | Override output CSV path |
| `--limit N` | Cap total output rows (quant written before binary) |

Train or grid-search on Papyrus by passing `--csv`. **Always use
`--rebuild-features`** after changing assay vocabulary (the assay one-hot is
now 5-d including `Other`) or switching prepared CSVs:

```bash
python train.py --csv data/train/Papyrus_pp_prepared.csv --rebuild-features
python grid_search.py --csv data/train/Papyrus_full_prepared.csv --rebuild-features
python grid_search.py --csv data/train/Papyrus_full_binary_prepared.csv --rebuild-features
```

Each CSV gets its own feature-cache signature and cold-protein splits under
`cache/features/` and `data/splits/`.

## Ligand representations

`--ligand-model` accepts a Hugging Face SMILES transformer id, a reserved RDKit
token, or a **comma-separated combination** (concatenated in that order):

| Token | Features |
|---|---|
| `morgan` | Morgan/ECFP fingerprint (radius 2, 2048 bits) |
| `avalon` | Avalon fingerprint (512 bits) |
| `descriptors` | Full RDKit `Descriptors.descList` (~217 physicochemical features) |
| `molformer` | Alias for `ibm-research/MoLFormer-XL-both-10pct` (768-d) |
| any HF model id | Mean-pooled SMILES transformer embedding |

Examples:

```bash
# Fingerprints only
python train.py --ligand-model morgan --rebuild-features

# RDKit-only stack
python train.py --ligand-model morgan,avalon,descriptors --rebuild-features

# Hybrid: Morgan + MoLFormer
python train.py --ligand-model morgan,molformer --rebuild-features

# Explicit HF id (default when --ligand-model is omitted)
python train.py --ligand-model ibm-research/MoLFormer-XL-both-10pct
```

Each component is cached independently under `cache/ligand__*.lmdb`, so adding
MoLFormer to an existing Morgan cache reuses the Morgan store.

## Feature vector

`concat(protein_emb, ligand_repr, assay_onehot[5], pH[1], temp[1])`

Assay one-hot order is fixed in `config.ASSAY_TYPES`: IC50, EC50, Ki, Kd, Other.

With the default MoLFormer ligand representation this is
`concat(protein[1280], ligand[768], assay[4], pH, temp)`. With combined
representations, `ligand` is the concatenation of each selected component.

## Training

```bash
# Default: BindingDB, random forest, 80/20 cold-protein split
python train.py

# Papyrus++
python train.py --csv data/train/Papyrus_pp_prepared.csv --rebuild-features

# MLP with custom hyperparameters
python train.py --model mlp --epochs 30 --hidden-dim 2048 --learning-rate 5e-4

# Random forest memory/accuracy tuning
python train.py --model rf --n-estimators 400 --rf-batch-trees 25 --rf-shard-rows 50000

# Quick smoke test on a subset of raw rows
python train.py --limit 5000
```

After training, accuracy, precision, recall, F1, AUROC, and AUPRC are printed
and the model is saved to `models/` as a joblib file. Splits are saved under
`data/splits/` and reused automatically when the dataset and seed match.

MLP early stopping uses a **cold-protein holdout** carved from the training split
(~5% of train proteins), not a random row split.

## Grid search

```bash
# BindingDB (default)
python grid_search.py

# Papyrus++
python grid_search.py --csv data/train/Papyrus_pp_prepared.csv --rebuild-features

# Include random-forest baselines
python grid_search.py --include-rf
```

Iterates over MLP hyperparameter combinations (and optionally RF baselines) using
an 80/10/10 cold-protein split, prints each candidate's validation AUROC as it
finishes, and saves the best model to `models/`.

## Prediction

```bash
# Mode 1: single spreadsheet with SMILES + protein sequence columns
python predict.py --spreadsheet input.csv

# Mode 2: one ligand input x one protein input (all pairwise combinations)
python predict.py --ligand "CCO" --protein sequences.fasta
python predict.py --ligand ligands.smi --protein "MKT...SEQ"

# Mode 3: a folder of single-spreadsheet inputs
python predict.py --spreadsheet-dir ./inputs/

# Mode 4: a folder of ligand inputs x a folder of protein inputs
python predict.py --ligand-dir ./ligands/ --protein-dir ./proteins/

# Override the assumed assay / conditions (defaults: Ki, pH 7.4, 25 C)
python predict.py --spreadsheet input.csv --assay IC50 --pH 6.5 --temp 37
```

Outputs include a `P(Active)` column: the predicted probability that activity is
at most 50 nM (matching the training threshold). Spreadsheet outputs are named
`*_predictions.<ext>`; pairwise/folder modes write a combined CSV. Prediction
reloads the ligand representation from the saved model's metadata.

## Project layout

```
config.py            Central configuration and defaults.
build_papyrus.py     Download Papyrus and write a prepared training CSV.
src/data_prep.py     Streaming cleaning + label preparation.
src/lmdb_cache.py    Embedding cache with model-derived DB names.
src/embeddings.py    ESM-2 and MoLFormer-XL embedders.
src/ligand_repr.py   Fingerprints, descriptors, composite ligand reps.
src/featurize.py     Memmapped feature-matrix assembly.
src/splits.py        Cold-protein splits with save/reuse.
src/models.py        Warm-start RF + torch MLP classifiers.
src/metrics.py       Classification metrics (AUROC, AUPRC, F1, …).
src/io_utils.py      Spreadsheet / SMILES / SDF / FASTA IO.
train.py             Train one model.
grid_search.py       Search model types and hyperparameters.
predict.py           Four prediction modes.
```
