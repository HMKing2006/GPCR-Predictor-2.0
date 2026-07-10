"""Source package for the GPCR binding-affinity predictor.

Modules:
    data_prep: Streaming cleaning and label preparation from the BindingDB CSV.
    lmdb_cache: On-disk embedding cache keyed by model identity.
    embeddings: MoLFormer-XL (ligand) and ESM-2 (protein) embedders.
    featurize: Assemble memmapped feature matrices from cached embeddings.
    splits: Cold-protein train/test/validation splitting with reuse.
    models: Random-forest and MLP regressors with a shared interface.
    metrics: Regression metric helpers.
    io_utils: Readers/writers for spreadsheet, SMILES, SDF and FASTA formats.
"""
