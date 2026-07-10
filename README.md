# GPCR Binding Affinity Predictor 2.0

Predicts protein-ligand binding affinity (pActivity) from BindingDB data.
Ligands are embedded with MoLFormer-XL and proteins with ESM-2; both embeddings
are cached in LMDB stores named after the models that produced them. Models are
trained with a cold-protein split and scored by RMSE.

## Setup

```bash
pip install -r requirements.txt
```

The default embedding models are downloaded automatically from the Hugging Face
Hub on first use:

- Protein: `facebook/esm2_t33_650M_UR50D` (1280-d)
- Ligand: `ibm-research/MoLFormer-XL-both-10pct` (768-d, loaded with `trust_remote_code=True`)

## Data preparation

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

## Feature vector

`concat(protein_emb[1280], ligand_emb[384], assay_onehot[4], pH[1], temp[1])`

## Training

```bash
# Default: random forest with warm-start incremental training, 80/20 cold split
python train.py

# Deep-learning model with custom hyperparameters
python train.py --model mlp --epochs 30 --hidden-dim 2048 --learning-rate 5e-4

# Random forest memory/accuracy tuning
python train.py --model rf --n-estimators 400 --rf-batch-trees 25 --rf-shard-rows 50000

# Quick smoke test on a subset
python train.py --limit 5000
```

After training, MAE, RMSE and R^2 are printed and the model is saved to
`models/` as a joblib file. Splits are saved under `data/splits/` and reused
automatically when the dataset and seed match.

## Grid search

```bash
python grid_search.py
```

Iterates over model types and hyperparameter combinations using an 80/10/10
cold-protein split, prints each model's validation metrics as it is trained, and
saves the best model to `models/`.

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

Outputs preserve the input spreadsheet extension and are named
`*_predictions.<ext>`; pairwise/folder modes write a combined CSV.

## Project layout

```
config.py            Central configuration and defaults.
src/data_prep.py     Streaming cleaning + label preparation.
src/lmdb_cache.py    Embedding cache with model-derived DB names.
src/embeddings.py    ESM-2 and MoLFormer-XL embedders.
src/featurize.py     Memmapped feature-matrix assembly.
src/splits.py        Cold-protein splits with save/reuse.
src/models.py        Warm-start RF + torch MLP regressors.
src/metrics.py       MAE / RMSE / R^2.
src/io_utils.py      Spreadsheet / SMILES / SDF / FASTA IO.
train.py             Train one model.
grid_search.py       Search model types and hyperparameters.
predict.py           Four prediction modes.
```
