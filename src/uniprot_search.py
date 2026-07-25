"""UniProt Swiss-Prot search and sequence fetch helpers for the screening GUI.

Uses the public UniProt REST API with a small JSON disk cache under
``cache/uniprot/`` so repeated accession lookups skip the network.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import config

_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"
_HTTP_RETRIES = 5
_HTTP_BACKOFF_S = 1.5
_CACHE_DIR = os.path.join(config.CACHE_DIR, "uniprot")
_SEQUENCE_CACHE_PATH = os.path.join(_CACHE_DIR, "sequences.json")
_USER_AGENT = "GPCR-Predictor-2.0/uniprot_search"


@dataclass(frozen=True)
class UniProtHit:
    """One Swiss-Prot search suggestion.

    Attributes:
        accession: Primary UniProt accession (e.g. ``P25929``).
        gene: Preferred gene symbol when available.
        protein_name: Recommended protein name.
        organism: Organism scientific name.
        label: Human-readable label for UI dropdowns.
    """

    accession: str
    gene: str
    protein_name: str
    organism: str
    label: str


@dataclass(frozen=True)
class UniProtSequence:
    """Resolved Swiss-Prot entry with sequence.

    Attributes:
        accession: Primary UniProt accession.
        gene: Preferred gene symbol when available.
        protein_name: Recommended protein name.
        organism: Organism scientific name.
        sequence: Amino-acid sequence.
    """

    accession: str
    gene: str
    protein_name: str
    organism: str
    sequence: str


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context, preferring certifi CA roots when installed.

    Returns:
        An :class:`ssl.SSLContext` suitable for HTTPS API calls.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_json(url: str, *, timeout: float = 60.0) -> Any:
    """GET JSON from ``url`` with retries.

    Args:
        url: Request URL.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON payload.

    Raises:
        urllib.error.URLError: When all retries fail.
        json.JSONDecodeError: When the response is not valid JSON.
    """
    hdrs = {"Accept": "application/json", "User-Agent": _USER_AGENT}
    ctx = _ssl_context()
    last_error: Optional[BaseException] = None
    for attempt in range(_HTTP_RETRIES):
        req = urllib.request.Request(url, headers=hdrs, method="GET")
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 or exc.code >= 500:
                time.sleep(_HTTP_BACKOFF_S * (2**attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(_HTTP_BACKOFF_S * (2**attempt))
    assert last_error is not None
    raise last_error


def _load_json_cache(path: str) -> dict[str, Any]:
    """Load a JSON object cache from disk.

    Args:
        path: Cache file path.

    Returns:
        Parsed mapping, or an empty dict when the file is absent.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Cache {path!r} must contain a JSON object.")
    return payload


def _save_json_cache(path: str, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON object cache.

    Args:
        path: Destination path.
        payload: Mapping to serialize.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)


def _preferred_gene(entry: Mapping[str, Any]) -> str:
    """Extract the preferred gene name from a UniProt JSON entry.

    Args:
        entry: One UniProtKB result object.

    Returns:
        Gene symbol, or an empty string when missing.
    """
    genes = entry.get("genes") or []
    if not genes:
        return ""
    first = genes[0] or {}
    gene = first.get("geneName") or {}
    return str(gene.get("value") or "").strip()


def _protein_name(entry: Mapping[str, Any]) -> str:
    """Extract the recommended protein name from a UniProt JSON entry.

    Args:
        entry: One UniProtKB result object.

    Returns:
        Protein name, or an empty string when missing.
    """
    description = entry.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full = recommended.get("fullName") or {}
    name = str(full.get("value") or "").strip()
    if name:
        return name
    submission = description.get("submissionNames") or []
    if submission:
        full = (submission[0] or {}).get("fullName") or {}
        return str(full.get("value") or "").strip()
    return ""


def _hit_label(gene: str, protein_name: str, accession: str, organism: str) -> str:
    """Build a compact UI label for a search hit.

    Args:
        gene: Gene symbol.
        protein_name: Protein name.
        accession: UniProt accession.
        organism: Organism name.

    Returns:
        Display string such as ``NPY1R — Neuropeptide Y receptor type 1 (P25929)``.
    """
    head = gene or protein_name or accession
    middle = protein_name if gene and protein_name else ""
    if middle and middle != head:
        base = f"{head} — {middle} ({accession})"
    else:
        base = f"{head} ({accession})"
    if organism and organism.lower() != "homo sapiens":
        return f"{base} [{organism}]"
    return base


def search_swissprot(
    query: str,
    *,
    size: int = 10,
    human_only: bool = True,
) -> list[UniProtHit]:
    """Search reviewed Swiss-Prot entries by name, gene, or accession text.

    Args:
        query: Free-text search string from the user.
        size: Maximum number of suggestions to return.
        human_only: If ``True``, restrict to ``organism_id:9606``.

    Returns:
        A list of :class:`UniProtHit` suggestions (may be empty).
    """
    text = str(query or "").strip()
    if len(text) < 2:
        return []
    clauses = ["(reviewed:true)"]
    if human_only:
        clauses.append("(organism_id:9606)")
    # Escape characters that break UniProt query syntax.
    safe = text.replace("(", " ").replace(")", " ").replace('"', " ").strip()
    if not safe:
        return []
    clauses.append(f"({safe})")
    params = urllib.parse.urlencode(
        {
            "query": " AND ".join(clauses),
            "fields": "accession,protein_name,gene_names,organism_name",
            "size": str(max(1, min(int(size), 25))),
            "format": "json",
        }
    )
    payload = _http_json(f"{_SEARCH_URL}?{params}")
    hits: list[UniProtHit] = []
    for entry in payload.get("results") or []:
        accession = str(entry.get("primaryAccession") or "").strip().upper()
        if not accession:
            continue
        gene = _preferred_gene(entry)
        protein_name = _protein_name(entry)
        organism = str((entry.get("organism") or {}).get("scientificName") or "").strip()
        hits.append(
            UniProtHit(
                accession=accession,
                gene=gene,
                protein_name=protein_name,
                organism=organism,
                label=_hit_label(gene, protein_name, accession, organism),
            )
        )
    return hits


def fetch_sequence(accession: str) -> UniProtSequence:
    """Fetch the amino-acid sequence for a UniProt accession (disk-cached).

    Args:
        accession: UniProt primary accession.

    Returns:
        A :class:`UniProtSequence` with a non-empty sequence.

    Raises:
        ValueError: If the accession is empty or cannot be resolved.
        urllib.error.URLError: On unrecoverable network failure.
    """
    key = str(accession or "").strip().upper()
    if not key:
        raise ValueError("Accession is required.")

    cache = _load_json_cache(_SEQUENCE_CACHE_PATH)
    cached = cache.get(key)
    if isinstance(cached, dict) and cached.get("sequence"):
        return UniProtSequence(
            accession=key,
            gene=str(cached.get("gene") or ""),
            protein_name=str(cached.get("protein_name") or ""),
            organism=str(cached.get("organism") or ""),
            sequence=str(cached["sequence"]),
        )

    params = urllib.parse.urlencode(
        {
            "accessions": key,
            "format": "json",
            "fields": "accession,sequence,organism_name,protein_name,gene_names",
        }
    )
    payload = _http_json(f"{_ACCESSIONS_URL}?{params}")
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"No UniProt entry found for accession {key!r}.")
    entry = results[0]
    sequence = str((entry.get("sequence") or {}).get("value") or "").strip()
    if not sequence:
        raise ValueError(f"UniProt entry {key!r} has no sequence.")
    gene = _preferred_gene(entry)
    protein_name = _protein_name(entry)
    organism = str((entry.get("organism") or {}).get("scientificName") or "").strip()
    record = {
        "gene": gene,
        "protein_name": protein_name,
        "organism": organism,
        "sequence": sequence,
    }
    cache[key] = record
    _save_json_cache(_SEQUENCE_CACHE_PATH, cache)
    return UniProtSequence(
        accession=key,
        gene=gene,
        protein_name=protein_name,
        organism=organism,
        sequence=sequence,
    )
