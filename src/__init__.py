"""Source package for the GPCR binder classifier.

Modules:
    data_prep: Streaming cleaning and label preparation from the BindingDB CSV.
    lmdb_cache: On-disk embedding cache keyed by model identity.
    embeddings: MoLFormer-XL (ligand) and ESM-2 (protein) embedders.
    ligand_repr: Morgan/Avalon fingerprints, RDKit descriptors, and composite
        ligand representations.
    featurize: Build compact snapshots and gather feature batches on demand.
    splits: Double-cold and temporal splitting with validated reuse.
    models: Random-forest and MLP classifiers with a shared interface.
    metrics: Classification metric helpers.
    io_utils: Readers/writers for spreadsheet, SMILES, SDF and FASTA formats.
"""
