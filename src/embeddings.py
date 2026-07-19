"""Protein (ESM-2) and ligand (MoLFormer-XL / ChemBERTa) embedders.

Both embedders wrap a Hugging Face transformer, run masked mean-pooling over the
token embeddings, and expose a cache-aware path that only computes vectors for
entities missing from an :class:`~src.lmdb_cache.EmbeddingCache`, persisting new
vectors in batches so long runs are resumable.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import config
from src.lmdb_cache import EmbeddingCache


def select_device() -> torch.device:
    """Choose the best available torch device.

    Returns:
        ``cuda`` if available, otherwise Apple ``mps``, otherwise ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache(device: torch.device) -> None:
    """Return cached-but-unused allocator blocks to the OS.

    PyTorch's MPS (and CUDA) caching allocators retain freed blocks for reuse.
    With variable-length batches this cache grows without bound because each new
    tensor size allocates a fresh block, so the reserved footprint climbs toward
    an out-of-memory kill even though live tensor usage is small. Emptying the
    cache between batches caps the reserved footprint.

    Args:
        device: The torch device whose allocator cache should be emptied.

    Returns:
        None.
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean-pool token embeddings into one vector per sequence.

    Args:
        last_hidden: Tensor of shape ``(batch, seq_len, dim)``.
        attention_mask: Tensor of shape ``(batch, seq_len)`` with 1 for real
            tokens and 0 for padding.

    Returns:
        Tensor of shape ``(batch, dim)`` of mean-pooled embeddings.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class HFEmbedder:
    """A batched, cache-aware Hugging Face sequence embedder."""

    def __init__(
        self,
        model_id: str,
        modality: str,
        max_length: Optional[int] = None,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        trust_remote_code: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Load the tokenizer and model and place it on the target device.

        Args:
            model_id: Hugging Face model identifier.
            modality: Either ``"protein"`` or ``"ligand"``; used to name the
                cache store.
            max_length: Optional maximum token length (sequences are truncated).
            batch_size: Number of sequences per forward pass.
            device: Torch device to run on; auto-selected when ``None``.
            trust_remote_code: Whether to allow custom Hub modeling code (needed
                for models such as MoLFormer that ship their own architecture).
            model_kwargs: Extra keyword arguments forwarded to
                ``AutoModel.from_pretrained`` (e.g. ``deterministic_eval``).
        """
        self.model_id: str = model_id
        self.modality: str = modality
        self.max_length: Optional[int] = max_length
        self.batch_size: int = batch_size
        self.device: torch.device = device if device is not None else select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = (
            AutoModel.from_pretrained(
                model_id, trust_remote_code=trust_remote_code, **(model_kwargs or {})
            )
            .to(self.device)
            .eval()
        )

    @property
    def dim(self) -> int:
        """Embedding dimensionality reported by the model config.

        Returns:
            The model's hidden size.
        """
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        """Embed a single batch of sequences.

        Args:
            sequences: List of raw sequences/SMILES (length <= ``batch_size``
                recommended).

        Returns:
            A ``float32`` array of shape ``(len(sequences), dim)``.
        """
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        output = self.model(**encoded)
        pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        return pooled.detach().to("cpu", dtype=torch.float32).numpy()

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Embed a list of sequences over multiple batches.

        Args:
            sequences: List of raw sequences/SMILES.

        Returns:
            A ``float32`` array of shape ``(len(sequences), dim)``.
        """
        if not sequences:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(sequences), self.batch_size):
            chunks.append(self.embed_batch(sequences[start : start + self.batch_size]))
        return np.concatenate(chunks, axis=0)

    def ensure_cached(
        self,
        entities: Iterable[str],
        cache: EmbeddingCache,
        verbose: bool = True,
        flush_every: int = 20,
    ) -> int:
        """Compute and persist embeddings for any uncached entities.

        Args:
            entities: Iterable of entity strings (duplicates are ignored).
            cache: The embedding cache to read from and write to.
            verbose: If ``True``, print periodic progress.
            flush_every: Number of batches between cache write transactions.

        Returns:
            The number of newly computed and stored embeddings.
        """
        todo = cache.missing(entities)
        total = len(todo)
        if total == 0:
            if verbose:
                print(f"[{self.modality}] all {cache.__len__()} embeddings cached; nothing to do")
            return 0
        # Group similar-length sequences together so each padded batch is as
        # uniform as possible, minimizing distinct tensor shapes (and thus
        # allocator fragmentation) as well as wasted padding compute.
        todo.sort(key=len)
        if verbose:
            print(f"[{self.modality}] computing {total} new embeddings on {self.device}")

        pending: dict[str, np.ndarray] = {}
        done = 0
        batches_since_flush = 0
        for start in range(0, total, self.batch_size):
            batch = todo[start : start + self.batch_size]
            vecs = self.embed_batch(batch)
            for entity, vec in zip(batch, vecs):
                pending[entity] = vec
            # Release cached allocator blocks so the reserved device footprint
            # stays bounded across many variable-length batches (fixes the MPS
            # driver-memory growth that caused the OOM kill).
            empty_device_cache(self.device)
            done += len(batch)
            batches_since_flush += 1
            if batches_since_flush >= flush_every:
                cache.put_many(pending)
                pending.clear()
                batches_since_flush = 0
                if verbose:
                    print(f"[{self.modality}] {done}/{total} embedded", flush=True)
        cache.put_many(pending)
        if verbose:
            print(f"[{self.modality}] finished: {done}/{total} embedded", flush=True)
        return done


def protein_embedder(
    model_id: str = config.DEFAULT_PROTEIN_MODEL,
    device: Optional[torch.device] = None,
) -> HFEmbedder:
    """Construct the default protein (ESM-2) embedder.

    Args:
        model_id: ESM-2 model identifier.
        device: Torch device; auto-selected when ``None``.

    Returns:
        A configured :class:`HFEmbedder` for proteins.
    """
    return HFEmbedder(
        model_id=model_id,
        modality="protein",
        max_length=config.MAX_PROTEIN_LEN,
        batch_size=config.PROTEIN_BATCH_SIZE,
        device=device,
    )


def ligand_embedder(
    model_id: str = config.MOLFORMER_MODEL_ID,
    device: Optional[torch.device] = None,
) -> HFEmbedder:
    """Construct a Hugging Face ligand (SMILES) embedder.

    Args:
        model_id: SMILES transformer model identifier (default: MoLFormer-XL).
        device: Torch device; auto-selected when ``None``.

    Returns:
        A configured :class:`HFEmbedder` for ligands.
    """
    trust_remote_code = model_id in config.TRUST_REMOTE_CODE_MODELS
    # MoLFormer's linear attention is stochastic unless deterministic_eval is set,
    # which would make cached embeddings unstable across runs.
    model_kwargs = {"deterministic_eval": True} if trust_remote_code else None
    return HFEmbedder(
        model_id=model_id,
        modality="ligand",
        max_length=None,
        batch_size=config.LIGAND_BATCH_SIZE,
        device=device,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs,
    )
