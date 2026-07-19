# GPCR Binder Classifier 2.0

Predicts whether a protein–ligand pair is an active binder (activity ≤ 50 nM) from
BindingDB or Papyrus data. Ligands default to Morgan fingerprints + RDKit
descriptors (MoLFormer-XL and other reps are available via ``--ligand-model``).
Proteins are embedded with ESM-2. Ligand and protein vectors are cached
in LMDB stores named after the components that produced them. Outer folds use
independent ``--test-split`` / ``--validation-split`` strategies (both default to
**cold-protein**); test is carved first, then validation from the remainder.
Evaluation reports AUROC/AUPRC plus known/unknown protein and scaffold breakouts.
By default the feature vector is protein + ligand only (assay type, pH, and
temperature are optional).

## Setup

```bash
pip install -r requirements.txt
```

The default embedding models are downloaded automatically from the Hugging Face
Hub on first use:

- Protein: `facebook/esm2_t33_650M_UR50D` (1280-d)
- Ligand (default): Morgan (2048-d) + RDKit descriptors (217-d). Optional HF ligand
  model: `ibm-research/MoLFormer-XL-both-10pct` (768-d, `trust_remote_code=True`)
  via `--ligand-model molformer`.

[Papyrus](https://doi.org/10.1186/s13321-022-00672-x) support requires
`papyrus-scripts` and `pyarrow` (both in `requirements.txt`).

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
  to 25.0 C) are available as optional scalar features when
  `--include-assay-context` is set. Assay type is one-hot encoded in that mode.

Labels are stored as binder / non-binder (50 nM cutoff,
`config.ACTIVITY_THRESHOLD_NM`) when features are built. Rows with an optional
`Activity Label` column (Papyrus binary builds) use that class directly.

### Papyrus

Build a BindingDB-compatible prepared **Parquet** file from the
[Papyrus](https://zenodo.org/records/13787634) dataset with `build_papyrus.py`.
The script downloads (if needed), streams, filters to exact Ki/Kd/IC50/EC50/**Other**
values (`type_other` included), joins protein sequences, carries publication
`Year`, and writes Parquet the training pipeline can consume.

**Papyrus++** (default, high-quality reproducible rows):

```bash
python build_papyrus.py --subset plusplus
# output: data/train/Papyrus_pp_prepared.parquet
```

**Full Papyrus** quantitative build (`without_stereochemistry`; multi-GB
download, ~2.4M exact Ki/Kd/IC50/EC50/Other rows after filters):

```bash
python build_papyrus.py --subset full
# output: data/train/Papyrus_full_prepared.parquet

# optional: restrict to higher-quality tiers
python build_papyrus.py --subset full --min-quality high
```

**Full + binary** (`Activity_class` rows, mostly inactive `N`, low quality).
Quant rows are written first; reuse `--limit` to cap total size (recommended —
the binary tail is ~56M rows):

```bash
python build_papyrus.py --subset full --include-binary --limit 5000000
# default output: data/train/Papyrus_full_binary_prepared.parquet
# (~2.4M quant + ~2.6M binary when limit=5M)
```

Binary rows are encoded as `Other (nM)` sentinel values plus an `Activity Label`
column (`0`/`1`). A `Year` column is included when Papyrus provides it. Override
the path with `--output` if needed (must end in `.parquet`).

Older CSV prepared builds are no longer the default; rebuild with
`build_papyrus.py` to get Year + Parquet for time splits.

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
| `--output PATH` | Override output Parquet path |
| `--limit N` | Cap total output rows (quant written before binary) |
| `--resume` | Resume the binary pass from an existing valid prepared Parquet |

Train or grid-search on Papyrus by passing `--data` (CSV or Parquet prepared
paths). Use `--rebuild-features` after changing row-level settings such as the
activity cutoff or assay context. Switching prepared files selects another
readable dataset cache directory automatically:

```bash
python train.py --data data/train/Papyrus_pp_prepared.parquet --rebuild-features
python grid_search.py --data data/train/Papyrus_full_prepared.parquet --rebuild-features
python grid_search.py --data data/train/Papyrus_full_binary_prepared.parquet --rebuild-features
```

Dataset caches are grouped under
`cache/datasets/<prepared-file-stem>/`. The row layout is validated against the
resolved source path, file size, modification time, cutoff, limit, and assay
context before reuse.

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
# Fingerprints only (adds morgan.npy without deleting existing embeddings)
python train.py --ligand-model morgan

# RDKit-only stack
python train.py --ligand-model morgan,avalon,descriptors

# Hybrid: Morgan + MoLFormer
python train.py --ligand-model morgan,molformer

# Explicit HF id (default when --ligand-model is omitted)
python train.py --ligand-model ibm-research/MoLFormer-XL-both-10pct
```

Each component is cached independently under `cache/ligand__*.lmdb`, so adding
MoLFormer to an existing Morgan cache reuses the Morgan store. Dataset-local
matrices are also component-specific, so combined representations concatenate
the selected matrices at runtime rather than storing another combined copy.

## Feature storage

The expanded per-activity-row `X.dat` and `split_*.dat` files are no longer
created. Each prepared dataset has one browseable snapshot:

```text
cache/datasets/Papyrus_full_binary_prepared/
  features/
    protein_ids.npy
    ligand_ids.npy
    activity_labels.npy
    scaffold_groups.npy
    years.npy
    protein_embeddings/
      ESM650.npy
    ligand_embeddings/
      MolFormerXL.npy
      morgan.npy
    meta.json
  splits/
    double_cold__test20pct__seed42.npz
```

The embedding `.npy` files contain one vector per unique entity and remain
memory-mappable. `FeatureView` gathers only the current MLP minibatch or random
forest shard into RAM. Disk use therefore scales primarily with unique proteins
and ligands, not activity rows. For 10 million rows with roughly one million
unique 768-dimensional ligands, the ligand snapshot is about 2.9 GiB instead
of roughly 76 GiB for a 2048-dimensional `X.dat`; no additional 61 GiB 80%
training split copy is written.

The global LMDB stores remain the durable, reusable source of embeddings for
other datasets and prediction. Existing LMDBs require no migration. After a
successful rebuild, obsolete `cache/features/`, `data/splits/`, `X.dat`, and
`split_*.dat` artifacts may be deleted manually.

## Feature vector

**Default** (assay context off):

`concat(protein_emb, ligand_repr)`

**With `--include-assay-context`:**

`concat(protein_emb, ligand_repr, assay_onehot[5], pH[1], temp[1])`

Assay one-hot order is fixed in `config.ASSAY_TYPES`: IC50, EC50, Ki, Kd, Other.

With the default Morgan + descriptors ligand representation and assay context off this is
`concat(protein[1280], ligand[768])`. With assay context enabled it becomes
`concat(protein[1280], ligand[768], assay[5], pH, temp)`. With combined
fingerprints, `ligand` is the concatenation of each selected component.

During feature build, Bemis–Murcko scaffold ids and publication years are stored
(`scaffold_groups.npy`, `years.npy`). Empty or failed scaffolds get a unique
per-row orphan id so they never merge incorrectly.

## Splits

Outer train/val/test folds are controlled by two flags (both default to
``protein`` = cold-protein):

| Strategy | Meaning |
|----------|---------|
| `protein` | Train/val/test share no proteins |
| `double-cold` | Share neither proteins nor Murcko scaffolds |
| `time` | Publication-year cutoffs from `--val-fraction` / `--test-fraction` (missing years → train) |

**Composition:** test is always carved with `--test-split` first; when a
validation fold is used, `--validation-split` carves val from the remainder.
`train.py` merges the val fold into train (two-way fit/test); `grid_search.py`
keeps val for model selection.

```bash
# Default: cold-protein test and val
python grid_search.py --data data/train/Papyrus_full_prepared.parquet

# Temporal test + cold-protein validation (select for protein transfer)
python grid_search.py \
  --data data/train/Papyrus_full_prepared.parquet \
  --test-split time \
  --validation-split protein \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --rebuild-features
```

Time folds require a Papyrus Parquet rebuild that includes `Year` and a feature
rebuild that writes `years.npy`.

## Training

```bash
# Default: BindingDB, random forest, cold-protein test, protein+ligand only
python train.py

# Include assay type / pH / temperature features
python train.py --include-assay-context --rebuild-features

# Papyrus++
python train.py --data data/train/Papyrus_pp_prepared.parquet --rebuild-features

# Temporal test fold (Papyrus with Year); val merged into train in train.py
python train.py --data data/train/Papyrus_pp_prepared.parquet \
  --test-split time --rebuild-features

# MLP with custom hyperparameters
python train.py --model mlp --epochs 30 --hidden-dim 2048 --learning-rate 5e-4

# MLP without inverse-frequency BCE class weights (on by default)
python train.py --model mlp --no-class-weights

# Random forest memory/accuracy tuning
python train.py --model rf --n-estimators 400 --rf-batch-trees 25 --rf-shard-rows 50000

# Quick smoke test on a subset of raw rows
python train.py --limit 5000
```

After training, accuracy, precision, recall, F1, AUROC, and AUPRC are printed,
followed by **novelty breakouts** on the test set (known/unknown protein,
known/unknown scaffold, and the 2×2 cells relative to train). The model is
saved to `models/` as a joblib file. Splits are saved under the matching
`cache/datasets/<stem>/splits/` directory and reused only when their embedded
row-layout signature and seed match.

MLP early stopping carves a holdout from the **training** split (~5% of train
rows by target). It prefers a **double-cold** holdout (no protein or scaffold
overlap with the fit set). If that holdout is too small — common when the
protein–scaffold graph forms a giant connected component — it falls back to a
**cold-protein** holdout. This inner ES carve is independent of
`--test-split` / `--validation-split` (e.g. a temporal test fold does not make
early stopping itself a future-year holdout).

By default the MLP uses **class weights**: BCE `pos_weight = n_neg / n_pos` on
the fit rows (inverse class frequency). Disable with `--no-class-weights`.

## Grid search

```bash
# BindingDB (default: cold-protein test + val)
python grid_search.py

# Papyrus++
python grid_search.py --data data/train/Papyrus_pp_prepared.parquet --rebuild-features

# Temporal test + cold-protein val (select for protein transfer)
python grid_search.py --data data/train/Papyrus_pp_prepared.parquet \
  --test-split time --validation-split protein --rebuild-features

# Include assay context
python grid_search.py --include-assay-context --rebuild-features

# Include random-forest baselines
python grid_search.py --include-rf

# Disable MLP class weights for the whole grid
python grid_search.py --no-class-weights
```

Iterates over MLP hyperparameter combinations (and optionally RF baselines)
using nested `--test-split` / `--validation-split` folds (both default to
cold-protein), prints each candidate's validation AUROC as it finishes,
evaluates the best model on test with novelty breakouts, and saves to `models/`.
MLP candidates inherit early stopping and `class_weights` (BCE
`pos_weight = n_neg / n_pos`) from `config.MLP_DEFAULTS` unless overridden.

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

# Override the assumed assay / conditions (defaults: Ki, pH 7.4, 25 C).
# These are only used in the feature vector when the model was trained with
# --include-assay-context; otherwise they are ignored at featurization time.
python predict.py --spreadsheet input.csv --assay IC50 --pH 6.5 --temp 37
```

Outputs include a `P(Active)` column: the predicted probability that activity is
at most 50 nM (matching the training threshold). Spreadsheet outputs are named
`*_predictions.<ext>`; pairwise/folder modes write a combined CSV. Prediction
reloads the ligand representation and assay-context setting from the saved
model's metadata.

## Ligand multilabel (experimental)

Isolated ligand-centric **family** and **target** multilabel models under
`src/multilabel/`. They do **not** change the pair binder pipeline
(`train.py` / `grid_search.py` / `predict.py` / `src/models.py`). Delete
`src/multilabel/`, `build_papyrus_multilabel.py`, `train_multilabel.py`,
`predict_multilabel.py`, and `models/multilabel/` to remove the stack.

Defaults:

- **Family vocab:** ChEMBL protein `Classification` level-2 tokens
- **Target vocab:** `target_id`s with ≥10 active ligands, top 1024 by count
- **Positives:** same binder rule as pair training; negatives are implicit zeros
- **Splits:** nested ``--test-split`` / ``--validation-split`` with
  ``scaffold`` (Murcko-cold, default) or ``time`` (percentage year cutoffs);
  test is carved first, then val from the remainder (merged into train by
  ``train_multilabel``). Per-ligand year is the **max** among its active
  annotations (undated → train)

```bash
# Build prepared ligand tables + vocab sidecars from pair prepared Parquet
python build_papyrus_multilabel.py \
  --activity-source data/train/Papyrus_full_binary_prepared.parquet

# Train (scaffold-cold test + val defaults)
python train_multilabel.py --task family --rebuild-features
python train_multilabel.py --task target --rebuild-features

# Temporal test + scaffold-cold validation carve (val merged into train)
python train_multilabel.py --task family \
  --test-split time --validation-split scaffold \
  --val-fraction 0.1 --test-fraction 0.2 --rebuild-features

# Predict (full vocab columns, or --top-k)
python predict_multilabel.py --model models/multilabel/family_multilabel__test-scaffold__val-scaffold.joblib \
  --ligand "CCO" --top-k 10 --output family_preds.csv
```

## Project layout

```
config.py            Central configuration and defaults.
build_papyrus.py     Download Papyrus and write prepared Parquet (+ Year).
src/data_prep.py     Streaming cleaning + label preparation (CSV / Parquet).
src/lmdb_cache.py    Embedding cache with model-derived DB names.
src/embeddings.py    ESM-2 and MoLFormer-XL embedders.
src/ligand_repr.py   Fingerprints, descriptors, composite ligand reps.
src/featurize.py     Compact feature snapshots + on-demand FeatureView.
src/splits.py        Nested test/val strategies (protein, double-cold, time).
src/models.py        Warm-start RF + torch MLP classifiers.
src/metrics.py       Classification metrics + novelty breakouts.
src/io_utils.py      Spreadsheet / SMILES / SDF / FASTA IO.
train.py             Train one model.
grid_search.py       Search model types and hyperparameters.
predict.py           Four prediction modes.

# Experimental ligand multilabel (deletable)
src/multilabel/              Isolated family/target multilabel package.
build_papyrus_multilabel.py  Build ligand prepared tables + vocab.
train_multilabel.py          Train family or target multilabel MLP.
predict_multilabel.py        Ligand → multilabel probability vector.
```
