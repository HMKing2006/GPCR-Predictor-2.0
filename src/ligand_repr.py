"""Ligand representation featurizers and multi-component composition.

Supports reserved RDKit tokens (``morgan``, ``avalon``, ``descriptors``), the
``molformer`` alias for the default Hugging Face ligand model, arbitrary HF
SMILES transformers, and comma-separated combinations thereof. Each component
writes to its own LMDB cache store; composites concatenate vectors in CLI order.
"""

from __future__ import annotations

from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

import config
from src.embeddings import HFEmbedder, ligand_embedder
from src.lmdb_cache import EmbeddingCache

_RDKIT_BATCH: int = 10_000
_FLUSH_EVERY: int = 5


@runtime_checkable
class LigandComponent(Protocol):
    """Protocol for a single ligand-feature component with LMDB caching."""

    cache_id: str
    dim: int

    def ensure_cached(
        self,
        smiles: Iterable[str],
        cache: EmbeddingCache,
        verbose: bool = True,
    ) -> int:
        """Compute and persist features for any uncached SMILES."""

    def vectors_for(self, smiles: Sequence[str], cache: EmbeddingCache) -> np.ndarray:
        """Return a stacked ``(n, dim)`` feature matrix for ``smiles``."""


def _bitvect_to_float32(bitvect: object, n_bits: int) -> np.ndarray:
    """Convert an RDKit ExplicitBitVect to a float32 0/1 array.

    Args:
        bitvect: An RDKit bit vector supporting ``GetOnBits()``.
        n_bits: Total length of the bit vector.

    Returns:
        A ``float32`` array of shape ``(n_bits,)``.
    """
    out = np.zeros(n_bits, dtype=np.float32)
    on_bits = bitvect.GetOnBits()  # type: ignore[attr-defined]
    if on_bits:
        out[list(on_bits)] = 1.0
    return out


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string, returning ``None`` on failure.

    Args:
        smiles: Canonical SMILES.

    Returns:
        An RDKit mol, or ``None`` if parsing fails.
    """
    return Chem.MolFromSmiles(smiles)


class MorganFingerprintFeaturizer:
    """ECFP-style Morgan fingerprint ligand featurizer."""

    def __init__(
        self,
        radius: int = config.MORGAN_RADIUS,
        n_bits: int = config.MORGAN_N_BITS,
    ) -> None:
        """Configure Morgan fingerprint parameters.

        Args:
            radius: Morgan radius (ECFP diameter is ``2 * radius``).
            n_bits: Fingerprint bit length.
        """
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.cache_id: str = f"morgan_r{self.radius}_b{self.n_bits}"
        self.dim: int = self.n_bits
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )

    def _featurize_one(self, smiles: str) -> np.ndarray:
        """Compute a Morgan fingerprint for one SMILES.

        Args:
            smiles: Canonical SMILES.

        Returns:
            A ``float32`` vector of length ``n_bits`` (zeros if parsing fails).
        """
        mol = _mol_from_smiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = self._generator.GetFingerprint(mol)
        return _bitvect_to_float32(fp, self.n_bits)

    def ensure_cached(
        self,
        smiles: Iterable[str],
        cache: EmbeddingCache,
        verbose: bool = True,
    ) -> int:
        """Compute and persist Morgan fingerprints for uncached SMILES.

        Args:
            smiles: Iterable of canonical SMILES.
            cache: Ligand embedding cache for this component.
            verbose: If ``True``, print progress.

        Returns:
            The number of newly computed fingerprints.
        """
        todo = cache.missing(smiles)
        total = len(todo)
        if total == 0:
            if verbose:
                print(f"[morgan] all {len(cache)} fingerprints cached; nothing to do")
            return 0
        if verbose:
            print(f"[morgan] computing {total} fingerprints")
        pending: dict[str, np.ndarray] = {}
        done = 0
        batches_since_flush = 0
        for start in range(0, total, _RDKIT_BATCH):
            batch = todo[start : start + _RDKIT_BATCH]
            for s in batch:
                pending[s] = self._featurize_one(s)
            done += len(batch)
            batches_since_flush += 1
            if batches_since_flush >= _FLUSH_EVERY:
                cache.put_many(pending)
                pending.clear()
                batches_since_flush = 0
                if verbose:
                    print(f"[morgan] {done}/{total} fingerprinted", flush=True)
        cache.put_many(pending)
        if verbose:
            print(f"[morgan] finished: {done}/{total} fingerprinted", flush=True)
        return done

    def vectors_for(self, smiles: Sequence[str], cache: EmbeddingCache) -> np.ndarray:
        """Look up cached Morgan fingerprints for ``smiles``.

        Args:
            smiles: Ordered SMILES list.
            cache: Ligand embedding cache for this component.

        Returns:
            A ``float32`` array of shape ``(len(smiles), dim)``.

        Raises:
            KeyError: If any SMILES is missing from the cache.
        """
        got = cache.get_many(smiles)
        out = np.empty((len(smiles), self.dim), dtype=np.float32)
        for i, s in enumerate(smiles):
            vec = got.get(s)
            if vec is None:
                raise KeyError(f"Morgan fingerprint missing from cache for {s!r}")
            out[i] = vec
        return out


class AvalonFingerprintFeaturizer:
    """Avalon bit-fingerprint ligand featurizer."""

    def __init__(self, n_bits: int = config.AVALON_N_BITS) -> None:
        """Configure Avalon fingerprint parameters.

        Args:
            n_bits: Fingerprint bit length.
        """
        self.n_bits = int(n_bits)
        self.cache_id: str = f"avalon_b{self.n_bits}"
        self.dim: int = self.n_bits

    def _featurize_one(self, smiles: str) -> np.ndarray:
        """Compute an Avalon fingerprint for one SMILES.

        Args:
            smiles: Canonical SMILES.

        Returns:
            A ``float32`` vector of length ``n_bits`` (zeros if parsing fails).
        """
        mol = _mol_from_smiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.n_bits)
        return _bitvect_to_float32(fp, self.n_bits)

    def ensure_cached(
        self,
        smiles: Iterable[str],
        cache: EmbeddingCache,
        verbose: bool = True,
    ) -> int:
        """Compute and persist Avalon fingerprints for uncached SMILES.

        Args:
            smiles: Iterable of canonical SMILES.
            cache: Ligand embedding cache for this component.
            verbose: If ``True``, print progress.

        Returns:
            The number of newly computed fingerprints.
        """
        todo = cache.missing(smiles)
        total = len(todo)
        if total == 0:
            if verbose:
                print(f"[avalon] all {len(cache)} fingerprints cached; nothing to do")
            return 0
        if verbose:
            print(f"[avalon] computing {total} fingerprints")
        pending: dict[str, np.ndarray] = {}
        done = 0
        batches_since_flush = 0
        for start in range(0, total, _RDKIT_BATCH):
            batch = todo[start : start + _RDKIT_BATCH]
            for s in batch:
                pending[s] = self._featurize_one(s)
            done += len(batch)
            batches_since_flush += 1
            if batches_since_flush >= _FLUSH_EVERY:
                cache.put_many(pending)
                pending.clear()
                batches_since_flush = 0
                if verbose:
                    print(f"[avalon] {done}/{total} fingerprinted", flush=True)
        cache.put_many(pending)
        if verbose:
            print(f"[avalon] finished: {done}/{total} fingerprinted", flush=True)
        return done

    def vectors_for(self, smiles: Sequence[str], cache: EmbeddingCache) -> np.ndarray:
        """Look up cached Avalon fingerprints for ``smiles``.

        Args:
            smiles: Ordered SMILES list.
            cache: Ligand embedding cache for this component.

        Returns:
            A ``float32`` array of shape ``(len(smiles), dim)``.

        Raises:
            KeyError: If any SMILES is missing from the cache.
        """
        got = cache.get_many(smiles)
        out = np.empty((len(smiles), self.dim), dtype=np.float32)
        for i, s in enumerate(smiles):
            vec = got.get(s)
            if vec is None:
                raise KeyError(f"Avalon fingerprint missing from cache for {s!r}")
            out[i] = vec
        return out


class RDKitDescriptorFeaturizer:
    """Full RDKit ``Descriptors.descList`` physicochemical descriptor set."""

    def __init__(self) -> None:
        """Build a calculator over the full RDKit descriptor list."""
        self._names: list[str] = [name for name, _ in Descriptors.descList]
        self._calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self._names)
        self.cache_id: str = "rdkit_descriptors"
        self.dim: int = len(self._names)

    def _featurize_one(self, smiles: str) -> np.ndarray:
        """Compute RDKit descriptors for one SMILES.

        Args:
            smiles: Canonical SMILES.

        Returns:
            A ``float32`` vector of length ``dim`` (zeros if parsing fails;
            non-finite values replaced with 0).
        """
        mol = _mol_from_smiles(smiles)
        if mol is None:
            return np.zeros(self.dim, dtype=np.float32)
        values = self._calculator.CalcDescriptors(mol)
        arr = np.asarray(values, dtype=np.float32)
        arr[~np.isfinite(arr)] = 0.0
        return arr

    def ensure_cached(
        self,
        smiles: Iterable[str],
        cache: EmbeddingCache,
        verbose: bool = True,
    ) -> int:
        """Compute and persist RDKit descriptors for uncached SMILES.

        Args:
            smiles: Iterable of canonical SMILES.
            cache: Ligand embedding cache for this component.
            verbose: If ``True``, print progress.

        Returns:
            The number of newly computed descriptor vectors.
        """
        todo = cache.missing(smiles)
        total = len(todo)
        if total == 0:
            if verbose:
                print(f"[descriptors] all {len(cache)} vectors cached; nothing to do")
            return 0
        if verbose:
            print(f"[descriptors] computing {total} descriptor vectors ({self.dim}-d)")
        pending: dict[str, np.ndarray] = {}
        done = 0
        batches_since_flush = 0
        for start in range(0, total, _RDKIT_BATCH):
            batch = todo[start : start + _RDKIT_BATCH]
            for s in batch:
                pending[s] = self._featurize_one(s)
            done += len(batch)
            batches_since_flush += 1
            if batches_since_flush >= _FLUSH_EVERY:
                cache.put_many(pending)
                pending.clear()
                batches_since_flush = 0
                if verbose:
                    print(f"[descriptors] {done}/{total} computed", flush=True)
        cache.put_many(pending)
        if verbose:
            print(f"[descriptors] finished: {done}/{total} computed", flush=True)
        return done

    def vectors_for(self, smiles: Sequence[str], cache: EmbeddingCache) -> np.ndarray:
        """Look up cached descriptor vectors for ``smiles``.

        Args:
            smiles: Ordered SMILES list.
            cache: Ligand embedding cache for this component.

        Returns:
            A ``float32`` array of shape ``(len(smiles), dim)``.

        Raises:
            KeyError: If any SMILES is missing from the cache.
        """
        got = cache.get_many(smiles)
        out = np.empty((len(smiles), self.dim), dtype=np.float32)
        for i, s in enumerate(smiles):
            vec = got.get(s)
            if vec is None:
                raise KeyError(f"Descriptors missing from cache for {s!r}")
            out[i] = vec
        return out


class HFLigandComponent:
    """Hugging Face SMILES-transformer ligand component."""

    def __init__(
        self,
        model_id: str,
        device: object = None,
    ) -> None:
        """Wrap an HF ligand embedder.

        Args:
            model_id: Hugging Face model identifier.
            device: Optional torch device for embedding.
        """
        self.model_id = model_id
        self.cache_id: str = model_id
        self._embedder: HFEmbedder = ligand_embedder(model_id, device=device)  # type: ignore[arg-type]
        self.dim: int = self._embedder.dim

    def ensure_cached(
        self,
        smiles: Iterable[str],
        cache: EmbeddingCache,
        verbose: bool = True,
    ) -> int:
        """Compute and persist HF embeddings for uncached SMILES.

        Args:
            smiles: Iterable of canonical SMILES.
            cache: Ligand embedding cache for this component.
            verbose: If ``True``, print progress.

        Returns:
            The number of newly computed embeddings.
        """
        return self._embedder.ensure_cached(smiles, cache, verbose=verbose)

    def vectors_for(self, smiles: Sequence[str], cache: EmbeddingCache) -> np.ndarray:
        """Look up cached HF embeddings for ``smiles``.

        Args:
            smiles: Ordered SMILES list.
            cache: Ligand embedding cache for this component.

        Returns:
            A ``float32`` array of shape ``(len(smiles), dim)``.

        Raises:
            KeyError: If any SMILES is missing from the cache.
        """
        got = cache.get_many(smiles)
        out = np.empty((len(smiles), self.dim), dtype=np.float32)
        for i, s in enumerate(smiles):
            vec = got.get(s)
            if vec is None:
                raise KeyError(f"HF embedding missing from cache for {s!r} ({self.model_id})")
            out[i] = vec
        return out


class CompositeLigandFeaturizer:
    """Ordered concatenation of one or more ligand components."""

    def __init__(self, components: Sequence[LigandComponent], canonical_spec: str) -> None:
        """Store ordered components and the canonical representation string.

        Args:
            components: Non-empty sequence of ligand components.
            canonical_spec: Normalized comma-separated representation string.

        Raises:
            ValueError: If ``components`` is empty.
        """
        if not components:
            raise ValueError("CompositeLigandFeaturizer requires at least one component.")
        self.components: list[LigandComponent] = list(components)
        self.canonical_spec: str = canonical_spec
        self.cache_id: str = "+".join(c.cache_id for c in self.components)
        self.dim: int = sum(c.dim for c in self.components)

    @property
    def component_dims(self) -> dict[str, int]:
        """Mapping from each component's cache id to its dimensionality.

        Returns:
            Ordered dict-like mapping of ``cache_id -> dim``.
        """
        return {c.cache_id: c.dim for c in self.components}

    def ensure_cached(
        self,
        smiles: Iterable[str],
        verbose: bool = True,
    ) -> int:
        """Ensure every component has cached vectors for ``smiles``.

        Opens one LMDB store per component.

        Args:
            smiles: Iterable of canonical SMILES.
            verbose: If ``True``, print progress per component.

        Returns:
            Total newly computed vectors across all components.
        """
        entities = list(dict.fromkeys(smiles))
        total_new = 0
        for component in self.components:
            with EmbeddingCache("ligand", component.cache_id) as cache:
                total_new += component.ensure_cached(entities, cache, verbose=verbose)
        return total_new

    def vectors_for(self, smiles: Sequence[str]) -> np.ndarray:
        """Concatenate component vectors for ``smiles`` in component order.

        Args:
            smiles: Ordered SMILES list.

        Returns:
            A ``float32`` array of shape ``(len(smiles), total_dim)``.
        """
        if not smiles:
            return np.zeros((0, self.dim), dtype=np.float32)
        blocks: list[np.ndarray] = []
        for component in self.components:
            with EmbeddingCache("ligand", component.cache_id) as cache:
                blocks.append(component.vectors_for(smiles, cache))
        return np.concatenate(blocks, axis=1)

    def vectors_dict(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        """Return a mapping from SMILES to concatenated ligand vectors.

        Args:
            smiles: SMILES list (duplicates tolerated).

        Returns:
            Mapping from each unique SMILES to its concatenated ``float32`` vector.
        """
        uniq = list(dict.fromkeys(smiles))
        matrix = self.vectors_for(uniq)
        return {s: matrix[i] for i, s in enumerate(uniq)}


def _canonical_token(token: str) -> str:
    """Normalize a single ligand-representation token.

    Args:
        token: One comma-separated piece of ``--ligand-model``.

    Returns:
        The canonical token string (``molformer`` expands to the default HF id).

    Raises:
        ValueError: If the token is empty or unrecognized.
    """
    t = token.strip()
    if not t:
        raise ValueError("Empty ligand representation token in --ligand-model.")
    lower = t.lower()
    if lower == "molformer":
        return config.DEFAULT_LIGAND_MODEL
    if lower in ("morgan", "avalon", "descriptors"):
        return lower
    # Hugging Face model ids typically contain "/"; also allow bare ids that are
    # not reserved tokens (backward-compatible with a single HF model string).
    if lower in config.LIGAND_REPR_TOKENS:
        return lower
    if "/" in t or t == config.DEFAULT_LIGAND_MODEL:
        return t
    # Accept any other non-reserved string as an HF model id (current behavior).
    return t


def canonical_ligand_repr(spec: str) -> str:
    """Normalize a ligand-representation spec for signatures and metadata.

    Args:
        spec: Raw ``--ligand-model`` value (possibly comma-separated).

    Returns:
        A canonical comma-joined string with aliases expanded.
    """
    tokens = [_canonical_token(part) for part in spec.split(",")]
    return ",".join(tokens)


def _build_component(token: str, device: object = None) -> LigandComponent:
    """Construct a ligand component from a canonical token.

    Args:
        token: Canonical token from :func:`_canonical_token`.
        device: Optional torch device for HF components.

    Returns:
        A configured ligand component.

    Raises:
        ValueError: If the token cannot be mapped to a component.
    """
    lower = token.lower()
    if lower == "morgan":
        return MorganFingerprintFeaturizer()
    if lower == "avalon":
        return AvalonFingerprintFeaturizer()
    if lower == "descriptors":
        return RDKitDescriptorFeaturizer()
    return HFLigandComponent(token, device=device)


def parse_ligand_repr(
    spec: str,
    device: object = None,
) -> CompositeLigandFeaturizer:
    """Parse a comma-separated ``--ligand-model`` value into a composite featurizer.

    Args:
        spec: Ligand representation string (e.g. ``"morgan"``,
            ``"morgan,avalon,molformer"``, or a HF model id).
        device: Optional torch device forwarded to HF components.

    Returns:
        A :class:`CompositeLigandFeaturizer` with components in CLI order.

    Raises:
        ValueError: If the spec is empty or contains invalid tokens.
    """
    if not spec or not str(spec).strip():
        raise ValueError("--ligand-model must be a non-empty representation spec.")
    canonical = canonical_ligand_repr(spec)
    tokens = canonical.split(",")
    components = [_build_component(t, device=device) for t in tokens]
    return CompositeLigandFeaturizer(components, canonical_spec=canonical)
