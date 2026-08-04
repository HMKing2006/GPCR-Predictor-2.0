"""On-disk LMDB caches for embeddings and Murcko scaffolds.

Embeddings are expensive to compute, so protein and ligand vectors are persisted
in LMDB stores. Each store's filename is derived from the model that produced
its contents, so switching either embedding model transparently references (or
creates) a different store and never mixes vectors from different models.

Bemis-Murcko scaffolds are cached separately in ``scaffold__murcko.lmdb`` as
UTF-8 strings keyed by SMILES.

Keys are SHA-1 hashes of the entity string (protein sequence or canonical
SMILES), keeping key sizes bounded regardless of sequence/SMILES length.
Embedding values are raw ``float32`` bytes decoded with
:func:`numpy.frombuffer`.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, Iterator, Optional

import lmdb
import numpy as np

import config

# 64 GiB sparse map; LMDB only consumes actual written pages on most systems.
_DEFAULT_MAP_SIZE: int = 64 * 1024**3
# Murcko scaffolds are short strings; a 1 GiB map is ample headroom.
_SCAFFOLD_MAP_SIZE: int = 1 * 1024**3
SCAFFOLD_CACHE_MODEL: str = "murcko"
SCAFFOLD_ORPHAN_SENTINEL: str = "__orphan__"


def sanitize_model_id(model_id: str) -> str:
    """Turn a model identifier into a filesystem-safe token.

    Args:
        model_id: A Hugging Face model id such as ``"facebook/esm2_t33_650M_UR50D"``.

    Returns:
        A string with every run of non-alphanumeric characters replaced by a
        single underscore.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", model_id).strip("_")


def cache_path(modality: str, model_id: str, cache_dir: str = config.CACHE_DIR) -> str:
    """Build the LMDB path for a modality/model pair.

    Args:
        modality: ``"protein"``, ``"ligand"``, or ``"scaffold"``.
        model_id: The embedding model identifier or scaffold cache id.
        cache_dir: Directory in which cache stores live.

    Returns:
        The absolute path of the LMDB store, e.g.
        ``cache/protein__esm2_t33_650M_UR50D.lmdb``.
    """
    return os.path.join(cache_dir, f"{modality}__{sanitize_model_id(model_id)}.lmdb")


def _key(entity: str) -> bytes:
    """Hash an entity string into a fixed-length LMDB key.

    Args:
        entity: Protein sequence or canonical SMILES.

    Returns:
        The SHA-1 digest of the UTF-8 encoded entity.
    """
    return hashlib.sha1(entity.encode("utf-8")).digest()


class EmbeddingCache:
    """A persistent, model-scoped store of embedding vectors.

    The cache may be used as a context manager to guarantee the underlying LMDB
    environment is closed.
    """

    def __init__(
        self,
        modality: str,
        model_id: str,
        cache_dir: str = config.CACHE_DIR,
        map_size: int = _DEFAULT_MAP_SIZE,
        readonly: bool = False,
        expected_dim: Optional[int] = None,
    ) -> None:
        """Open (or create) an LMDB store for one modality/model pair.

        Args:
            modality: Either ``"protein"`` or ``"ligand"``.
            model_id: The embedding model identifier used to name the store.
            cache_dir: Directory in which cache stores live.
            map_size: Maximum size in bytes the store may grow to.
            readonly: If ``True``, open without write access (and without
                creating the store when it is missing).
            expected_dim: Optional vector length. Reads and writes with another
                length raise ``ValueError`` instead of silently propagating a
                corrupt or incompatible cache entry.
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.path: str = cache_path(modality, model_id, cache_dir)
        self.modality: str = modality
        self.model_id: str = model_id
        self.expected_dim: Optional[int] = (
            None if expected_dim is None else int(expected_dim)
        )
        self._env: lmdb.Environment = lmdb.open(
            self.path,
            map_size=map_size,
            subdir=True,
            readonly=readonly,
            lock=not readonly,
            readahead=False,
            meminit=False,
            create=not readonly,
        )

    def __enter__(self) -> "EmbeddingCache":
        """Enter the runtime context and return the cache instance.

        Returns:
            This :class:`EmbeddingCache`.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the LMDB environment on context exit.

        Args:
            exc_type: Exception type if raised in the block, else ``None``.
            exc: Exception instance if raised, else ``None``.
            tb: Traceback if an exception was raised, else ``None``.
        """
        self.close()

    def close(self) -> None:
        """Close the underlying LMDB environment.

        Returns:
            None.
        """
        self._env.close()

    def contains(self, entity: str) -> bool:
        """Report whether an entity already has a cached embedding.

        Args:
            entity: Protein sequence or canonical SMILES.

        Returns:
            ``True`` if a vector is stored for ``entity``.
        """
        with self._env.begin(write=False) as txn:
            return txn.get(_key(entity)) is not None

    def get(self, entity: str) -> Optional[np.ndarray]:
        """Fetch a single cached embedding.

        Args:
            entity: Protein sequence or canonical SMILES.

        Returns:
            The stored ``float32`` vector, or ``None`` on a cache miss.
        """
        with self._env.begin(write=False) as txn:
            raw = txn.get(_key(entity))
        if raw is None:
            return None
        return self._decode(raw, entity)

    def get_many(self, entities: Iterable[str]) -> dict[str, np.ndarray]:
        """Fetch several embeddings in one read transaction.

        Args:
            entities: Iterable of entity strings.

        Returns:
            A mapping from each present entity to its vector. Missing entities
            are simply absent from the mapping.
        """
        result: dict[str, np.ndarray] = {}
        with self._env.begin(write=False) as txn:
            for entity in entities:
                raw = txn.get(_key(entity))
                if raw is not None:
                    result[entity] = self._decode(raw, entity)
        return result

    def _decode(self, raw: bytes, entity: str) -> np.ndarray:
        """Decode and validate one cached vector.

        Args:
            raw: Raw LMDB value bytes.
            entity: Entity string used only to identify malformed entries.

        Returns:
            A read-only ``float32`` view over ``raw``.

        Raises:
            ValueError: If byte length is not valid float32 data or the vector
                length differs from ``expected_dim``.
        """
        if len(raw) % np.dtype(np.float32).itemsize != 0:
            raise ValueError(
                f"Malformed {self.modality} cache entry for {_key(entity).hex()}: "
                f"{len(raw)} bytes is not valid float32 data."
            )
        vector = np.frombuffer(raw, dtype=np.float32)
        if self.expected_dim is not None and vector.size != self.expected_dim:
            raise ValueError(
                f"Incompatible {self.modality} cache entry for {_key(entity).hex()}: "
                f"expected {self.expected_dim} values, found {vector.size} "
                f"in {self.path}."
            )
        return vector

    def missing(self, entities: Iterable[str]) -> list[str]:
        """Return the entities that are not yet cached.

        Args:
            entities: Iterable of entity strings (may contain duplicates).

        Returns:
            A de-duplicated, order-preserving list of entities with no vector.
        """
        result: list[str] = []
        already: set[str] = set()
        with self._env.begin(write=False) as txn:
            for entity in entities:
                if entity in already:
                    continue
                already.add(entity)
                if txn.get(_key(entity)) is None:
                    result.append(entity)
        return result

    def put_many(self, vectors: dict[str, np.ndarray]) -> None:
        """Persist a batch of embeddings in a single write transaction.

        Args:
            vectors: Mapping from entity string to its ``float32`` vector.

        Returns:
            None.
        """
        if not vectors:
            return
        with self._env.begin(write=True) as txn:
            for entity, vec in vectors.items():
                arr = np.asarray(vec, dtype=np.float32)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Embedding for {entity!r} must be one-dimensional; "
                        f"got shape {arr.shape}."
                    )
                if self.expected_dim is not None and arr.size != self.expected_dim:
                    raise ValueError(
                        f"Embedding for {entity!r} has {arr.size} values; "
                        f"expected {self.expected_dim}."
                    )
                txn.put(_key(entity), arr.tobytes(), overwrite=True)

    def __len__(self) -> int:
        """Count the stored embeddings.

        Returns:
            The number of key/value entries in the store.
        """
        with self._env.begin(write=False) as txn:
            return int(txn.stat()["entries"])


class ScaffoldCache:
    """A persistent store of Bemis-Murcko scaffold strings keyed by SMILES.

    Values are UTF-8 scaffold SMILES, or :data:`SCAFFOLD_ORPHAN_SENTINEL` when
    RDKit cannot derive a scaffold. Row-specific orphan keys are expanded at
    index time so split semantics stay unchanged.
    """

    def __init__(
        self,
        cache_dir: str = config.CACHE_DIR,
        map_size: int = _SCAFFOLD_MAP_SIZE,
        readonly: bool = False,
    ) -> None:
        """Open (or create) the global Murcko scaffold LMDB store.

        Args:
            cache_dir: Directory in which cache stores live.
            map_size: Maximum size in bytes the store may grow to.
            readonly: If ``True``, open without write access (and without
                creating the store when it is missing).
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.path: str = cache_path("scaffold", SCAFFOLD_CACHE_MODEL, cache_dir)
        self._env: lmdb.Environment = lmdb.open(
            self.path,
            map_size=map_size,
            subdir=True,
            readonly=readonly,
            lock=not readonly,
            readahead=False,
            meminit=False,
            create=not readonly,
        )

    def __enter__(self) -> "ScaffoldCache":
        """Enter the runtime context and return the cache instance.

        Returns:
            This :class:`ScaffoldCache`.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the LMDB environment on context exit.

        Args:
            exc_type: Exception type if raised in the block, else ``None``.
            exc: Exception instance if raised, else ``None``.
            tb: Traceback if an exception was raised, else ``None``.
        """
        self.close()

    def close(self) -> None:
        """Close the underlying LMDB environment.

        Returns:
            None.
        """
        self._env.close()

    def contains(self, smiles: str) -> bool:
        """Report whether a SMILES already has a cached scaffold value.

        Args:
            smiles: Canonical ligand SMILES.

        Returns:
            ``True`` if a value is stored for ``smiles``.
        """
        with self._env.begin(write=False) as txn:
            return txn.get(_key(smiles)) is not None

    def get(self, smiles: str) -> Optional[str]:
        """Fetch a single cached scaffold value.

        Args:
            smiles: Canonical ligand SMILES.

        Returns:
            The stored scaffold string or orphan sentinel, or ``None`` on miss.
        """
        with self._env.begin(write=False) as txn:
            raw = txn.get(_key(smiles))
        if raw is None:
            return None
        return raw.decode("utf-8")

    def get_many(self, smiles_list: Iterable[str]) -> dict[str, str]:
        """Fetch several scaffold values in one read transaction.

        Args:
            smiles_list: Iterable of canonical ligand SMILES strings.

        Returns:
            Mapping from each present SMILES to its stored value.
        """
        result: dict[str, str] = {}
        with self._env.begin(write=False) as txn:
            for smiles in smiles_list:
                raw = txn.get(_key(smiles))
                if raw is not None:
                    result[smiles] = raw.decode("utf-8")
        return result

    def missing(self, smiles_list: Iterable[str]) -> list[str]:
        """Return SMILES strings that are not yet cached.

        Args:
            smiles_list: Iterable of canonical ligand SMILES (may contain
                duplicates).

        Returns:
            De-duplicated, order-preserving list of SMILES with no stored value.
        """
        result: list[str] = []
        already: set[str] = set()
        with self._env.begin(write=False) as txn:
            for smiles in smiles_list:
                if smiles in already:
                    continue
                already.add(smiles)
                if txn.get(_key(smiles)) is None:
                    result.append(smiles)
        return result

    def put_many(self, scaffolds: dict[str, str]) -> None:
        """Persist a batch of scaffold values in one write transaction.

        Args:
            scaffolds: Mapping from SMILES to scaffold SMILES or
                :data:`SCAFFOLD_ORPHAN_SENTINEL`.

        Returns:
            None.
        """
        if not scaffolds:
            return
        with self._env.begin(write=True) as txn:
            for smiles, scaffold in scaffolds.items():
                txn.put(_key(smiles), scaffold.encode("utf-8"), overwrite=True)

    def __len__(self) -> int:
        """Count the stored scaffold entries.

        Returns:
            The number of key/value entries in the store.
        """
        with self._env.begin(write=False) as txn:
            return int(txn.stat()["entries"])
