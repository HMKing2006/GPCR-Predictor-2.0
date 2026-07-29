"""Compare sequence-based pair models on the NCATS experimental HTS benchmark.

Screens each named model against every benchmark target on Sytravon and Genesis,
writes a top-N% hitlist per target/library, then scores those hitlists against
the experimental HTS outcomes in
``data/benchmark/experimental-GPCR-predictor-performance.xlsx``.

A compound counts as an experimental hit when ``Max Resp`` exceeds +30 (agonist
readouts) or falls below -30 for the GPR17 antagonist readout, after restricting
to compounds that are actually present in the screened library.

Example:
    python compare_sequence_models.py \\
        --model MD_1uM=models/mlp_512x2_time_morgan_descriptors.joblib \\
        --model MAD_10uM=models/mlp_512x2_time_morgan_avalon_descriptors_10uM.joblib
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import pandas as pd

import config
from src.screen_library import DEFAULT_LIGAND_MAP, ScreenLibrary
from src.uniprot_search import fetch_sequence

DEFAULT_WORKBOOK: str = os.path.join(
    config.DATA_DIR, "benchmark", "experimental-GPCR-predictor-performance.xlsx"
)
DEFAULT_OUT_DIR: str = os.path.join(
    config.DATA_DIR, "benchmark", "sequence_model_comparison"
)

TARGETS: dict[str, str] = {
    "GALR1": "P47211",
    "GPR151": "Q8TDV0",
    "GHSR": "Q92847",
    "RXFP2": "Q8WXD0",
    "GPR17": "Q13304",
    "RXFP1": "Q9HBX9",
    "TSHR": "P16473",
    "PAC1R": "P41586",
}

LIBRARIES: dict[str, str] = {"Sytravon": "Sytravon44K", "Genesis": "Genesis124K"}

_NCGC_RE = re.compile(r"(NCGC\d+)")


@dataclass(frozen=True)
class BenchmarkSheet:
    """One experimental HTS result sheet in the benchmark workbook.

    Attributes:
        target: Target gene symbol.
        library: Screening library name (``Sytravon`` or ``Genesis``).
        sheet: Worksheet name inside the benchmark workbook.
        polarity: ``1`` if hits have ``Max Resp`` > +30, ``-1`` if < -30.
        status: ``ok``, ``caveat``, or ``subset_only`` denominator quality flag.
        note: Human-readable caveat, empty when the denominator is clean.
    """

    target: str
    library: str
    sheet: str
    polarity: int
    status: str
    note: str


BENCHMARK_SHEETS: tuple[BenchmarkSheet, ...] = (
    BenchmarkSheet("GALR1", "Sytravon", "GALR1-STY", 1, "ok", ""),
    BenchmarkSheet("GALR1", "Genesis", "GALR1-Gene", 1, "ok", ""),
    BenchmarkSheet("GPR151", "Sytravon", "GPR151-STY", 1, "ok", ""),
    BenchmarkSheet(
        "GPR151",
        "Genesis",
        "GPR151-miniG",
        1,
        "subset_only",
        "GPR151-miniG is not a full Genesis HTS denominator; enrichment not comparable",
    ),
    BenchmarkSheet("GHSR", "Sytravon", "GHSR-S", 1, "ok", ""),
    BenchmarkSheet("RXFP2", "Sytravon", "RXFP2-S", 1, "ok", ""),
    BenchmarkSheet(
        "RXFP2",
        "Genesis",
        "RXFP2-G",
        1,
        "caveat",
        "RXFP2-G may not be a complete hits-only Genesis screen",
    ),
    BenchmarkSheet("GPR17", "Sytravon", "GPR17-S", -1, "ok", ""),
)


def normalize_ncgc(value: object) -> Optional[str]:
    """Extract the bare NCGC identifier from a sample ID.

    Args:
        value: Raw cell value such as ``"NCGC00106796-01"``.

    Returns:
        The ``NCGC…`` stem, or ``None`` when no identifier is present.
    """
    match = _NCGC_RE.search(str(value))
    return match.group(1) if match else None


def load_library_ids(ligand_map_path: str) -> dict[str, set[str]]:
    """Collect the compound IDs belonging to each screening library.

    Args:
        ligand_map_path: Path to ``ligand_id_map.tsv``.

    Returns:
        Mapping from dataset name (e.g. ``Sytravon44K``) to its NCGC ID set.
    """
    frame = pd.read_csv(ligand_map_path, sep="\t")
    out: dict[str, set[str]] = {}
    for dataset, group in frame.groupby("dataset"):
        ids = {normalize_ncgc(v) for v in group["ligand_id"]}
        ids.discard(None)
        out[str(dataset)] = ids  # type: ignore[assignment]
    return out


def load_experimental_hits(
    workbook_path: str, library_ids: dict[str, set[str]]
) -> dict[tuple[str, str], set[str]]:
    """Read experimental hit sets for every benchmark target/library pair.

    Args:
        workbook_path: Path to the benchmark workbook.
        library_ids: Output of :func:`load_library_ids`.

    Returns:
        Mapping from ``(target, library)`` to the set of hit NCGC IDs that are
        present in that library.
    """
    hits: dict[tuple[str, str], set[str]] = {}
    for spec in BENCHMARK_SHEETS:
        frame = pd.read_excel(workbook_path, sheet_name=spec.sheet)
        response = pd.to_numeric(frame["Max Resp"], errors="coerce")
        mask = response > 30 if spec.polarity > 0 else response < -30
        ids = {normalize_ncgc(v) for v in frame.loc[mask, "Sample ID"]}
        ids.discard(None)
        hits[(spec.target, spec.library)] = ids & library_ids[LIBRARIES[spec.library]]
    return hits


def screen_model(
    name: str,
    model_path: str,
    out_dir: str,
    *,
    top_pct: float,
    ligand_map_path: str,
    skip_existing: bool,
) -> list[dict[str, object]]:
    """Screen every benchmark target with one model and write top-N% hitlists.

    Args:
        name: Short label for the model (used as the output subdirectory).
        model_path: Path to the saved joblib pair model.
        out_dir: Root comparison output directory.
        top_pct: Percentage of each library to keep (e.g. ``5.0``).
        ligand_map_path: Path to ``ligand_id_map.tsv``.
        skip_existing: If ``True``, reuse hitlists already written to disk.

    Returns:
        One manifest record per target/library pair.
    """
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    expected = [
        (target, library, os.path.join(model_dir, f"{target}_top5pct_{library}.xlsx"))
        for target in TARGETS
        for library in LIBRARIES
    ]
    screener: Optional[ScreenLibrary] = None
    if not (skip_existing and all(os.path.isfile(path) for _, _, path in expected)):
        screener = ScreenLibrary(model_path, ligand_map_path, verbose=True)
        screener.warm_ligands()

    metadata = (
        screener.predictor.model.metadata
        if screener is not None
        else _metadata_only(model_path)
    )

    records: list[dict[str, object]] = []
    try:
        for target, uniprot in TARGETS.items():
            scored: Optional[pd.DataFrame] = None
            for library, dataset in LIBRARIES.items():
                path = os.path.join(model_dir, f"{target}_top5pct_{library}.xlsx")
                if skip_existing and os.path.isfile(path):
                    top = pd.read_excel(path)
                else:
                    if scored is None:
                        assert screener is not None
                        print(f"[{name}] screening {target} ({uniprot})", flush=True)
                        scored = screener.screen(fetch_sequence(uniprot).sequence)
                    subset = scored[scored["dataset"] == dataset]
                    keep = math.ceil(len(subset) * top_pct / 100.0)
                    top = subset.head(keep).reset_index(drop=True)
                    top = pd.DataFrame(
                        {
                            "rank": range(1, len(top) + 1),
                            "target": target,
                            "uniprot": uniprot,
                            "ligand_id": top["ID"],
                            "dataset": top["dataset"],
                            "Ligand SMILES": top["SMILES"],
                            "P_Active": top["P(Active)"],
                        }
                    )
                    top.to_excel(path, index=False)

                records.append(
                    {
                        "model": name,
                        "model_file": os.path.basename(model_path),
                        "threshold_nm": metadata.get("activity_threshold_nm"),
                        "ligand_model": metadata.get("ligand_model"),
                        "target": target,
                        "uniprot": uniprot,
                        "library": library,
                        "dataset": dataset,
                        "n_library": _library_size(ligand_map_path, dataset),
                        "n_top5pct": len(top),
                        "P_Active_min": float(top["P_Active"].min()),
                        "P_Active_max": float(top["P_Active"].max()),
                        "P_Active_cutoff": float(top["P_Active"].min()),
                        "file": os.path.relpath(path, config.PROJECT_ROOT),
                    }
                )
    finally:
        if screener is not None:
            screener.close()
    return records


def _metadata_only(model_path: str) -> dict[str, object]:
    """Read a model's metadata without building embedders or caches.

    Args:
        model_path: Path to the saved joblib pair model.

    Returns:
        The model metadata dictionary.
    """
    import joblib

    bundle = joblib.load(model_path)
    return dict(bundle.get("metadata", {}))


_LIBRARY_SIZES: dict[str, int] = {}


def _library_size(ligand_map_path: str, dataset: str) -> int:
    """Return the number of compounds in a screening library.

    Args:
        ligand_map_path: Path to ``ligand_id_map.tsv``.
        dataset: Dataset name (e.g. ``Sytravon44K``).

    Returns:
        Compound count for ``dataset``.
    """
    if not _LIBRARY_SIZES:
        frame = pd.read_csv(ligand_map_path, sep="\t", usecols=["dataset"])
        _LIBRARY_SIZES.update(frame["dataset"].value_counts().to_dict())
    return int(_LIBRARY_SIZES[dataset])


def compute_enrichment(
    manifest: pd.DataFrame,
    hits: dict[tuple[str, str], set[str]],
    out_dir: str,
) -> pd.DataFrame:
    """Score each hitlist against the experimental HTS outcomes.

    Args:
        manifest: Combined manifest records from :func:`screen_model`.
        hits: Output of :func:`load_experimental_hits`.
        out_dir: Root comparison output directory (used to resolve hitlist paths).

    Returns:
        One row per model/target/library with baseline and top-list hit rates,
        enrichment, and experimental recovery.
    """
    spec_by_key = {(s.target, s.library): s for s in BENCHMARK_SHEETS}
    rows: list[dict[str, object]] = []
    for record in manifest.to_dict("records"):
        key = (str(record["target"]), str(record["library"]))
        if key not in hits:
            continue
        spec = spec_by_key[key]
        hit_ids = hits[key]
        top = pd.read_excel(
            os.path.join(out_dir, str(record["model"]), f"{key[0]}_top5pct_{key[1]}.xlsx")
        )
        top_ids = {normalize_ncgc(v) for v in top["ligand_id"]}
        top_ids.discard(None)
        n_library = int(record["n_library"])
        n_top = int(record["n_top5pct"])
        n_found = len(top_ids & hit_ids)
        baseline = 100.0 * len(hit_ids) / n_library
        top_rate = 100.0 * n_found / n_top
        rows.append(
            {
                "model": record["model"],
                "target": key[0],
                "library": key[1],
                "n_library": n_library,
                "n_top5pct": n_top,
                "n_experimental_hits_in_library": len(hit_ids),
                "n_hits_in_top5pct": n_found,
                "baseline_hitrate_pct": baseline,
                "top5_hitrate_pct": top_rate,
                "enrichment": None if spec.status == "subset_only" else top_rate / baseline,
                "experimental_recovery_pct": 100.0 * n_found / len(hit_ids),
                "status": spec.status,
                "note": spec.note,
            }
        )
    return pd.DataFrame(rows)


def compute_overlap(manifest: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Measure how much the models' top lists agree with each other.

    Args:
        manifest: Combined manifest records from :func:`screen_model`.
        out_dir: Root comparison output directory.

    Returns:
        One row per target/library/model-pair with intersection size, union
        size, percentage overlap of the top lists, and Jaccard index.
    """
    models = list(dict.fromkeys(manifest["model"].astype(str)))
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for library in LIBRARIES:
            lists: dict[str, set[str]] = {}
            for model in models:
                path = os.path.join(out_dir, model, f"{target}_top5pct_{library}.xlsx")
                if not os.path.isfile(path):
                    continue
                ids = {normalize_ncgc(v) for v in pd.read_excel(path)["ligand_id"]}
                ids.discard(None)
                lists[model] = ids
            for model_a, model_b in combinations(lists, 2):
                set_a, set_b = lists[model_a], lists[model_b]
                inter = len(set_a & set_b)
                union = len(set_a | set_b)
                rows.append(
                    {
                        "target": target,
                        "library": library,
                        "model_a": model_a,
                        "model_b": model_b,
                        "intersection": inter,
                        "union": union,
                        "overlap_of_top5_pct": 100.0 * inter / len(set_a),
                        "jaccard": inter / union if union else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def write_summary(
    out_dir: str,
    manifest: pd.DataFrame,
    enrichment: pd.DataFrame,
    overlap: pd.DataFrame,
) -> str:
    """Write the comparison CSVs and a multi-sheet summary workbook.

    Args:
        out_dir: Root comparison output directory.
        manifest: Combined manifest records.
        enrichment: Output of :func:`compute_enrichment`.
        overlap: Output of :func:`compute_overlap`.

    Returns:
        Path to the summary workbook.
    """
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    enrichment.to_csv(os.path.join(out_dir, "experimental_enrichment.csv"), index=False)
    overlap.to_csv(os.path.join(out_dir, "top5_overlap.csv"), index=False)

    scored = enrichment[enrichment["status"] != "subset_only"]
    pivot = scored.pivot_table(
        index=["target", "library"], columns="model", values="enrichment"
    ).reset_index()
    per_model = (
        scored.groupby("model")
        .agg(
            n_comparisons=("enrichment", "size"),
            mean_enrichment=("enrichment", "mean"),
            median_enrichment=("enrichment", "median"),
            n_above_1x=("enrichment", lambda s: int((s > 1.0).sum())),
            total_hits_recovered=("n_hits_in_top5pct", "sum"),
        )
        .reset_index()
        .sort_values("mean_enrichment", ascending=False)
    )

    path = os.path.join(out_dir, "model_comparison_summary.xlsx")
    with pd.ExcelWriter(path) as writer:
        per_model.to_excel(writer, sheet_name="per_model", index=False)
        pivot.to_excel(writer, sheet_name="enrichment_pivot", index=False)
        enrichment.to_excel(writer, sheet_name="enrichment", index=False)
        overlap.to_excel(writer, sheet_name="top5_overlap", index=False)
        manifest.to_excel(writer, sheet_name="manifest", index=False)
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Model to include, e.g. MD_1uM=models/foo.joblib (repeatable).",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--workbook", default=DEFAULT_WORKBOOK)
    parser.add_argument("--ligand-map", default=DEFAULT_LIGAND_MAP)
    parser.add_argument("--top-pct", type=float, default=5.0)
    parser.add_argument(
        "--rescreen",
        action="store_true",
        help="Recompute hitlists even when they already exist on disk.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the model comparison end to end.

    Returns:
        None.

    Raises:
        ValueError: If a ``--model`` argument is not ``NAME=PATH``.
    """
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    records: list[dict[str, object]] = []
    for entry in args.model:
        if "=" not in entry:
            raise ValueError(f"--model expects NAME=PATH, got {entry!r}")
        name, path = entry.split("=", 1)
        records.extend(
            screen_model(
                name.strip(),
                path.strip(),
                args.out_dir,
                top_pct=args.top_pct,
                ligand_map_path=args.ligand_map,
                skip_existing=not args.rescreen,
            )
        )

    manifest = pd.DataFrame(records)
    library_ids = load_library_ids(args.ligand_map)
    hits = load_experimental_hits(args.workbook, library_ids)
    enrichment = compute_enrichment(manifest, hits, args.out_dir)
    overlap = compute_overlap(manifest, args.out_dir)
    summary = write_summary(args.out_dir, manifest, enrichment, overlap)
    print(f"[compare] wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
