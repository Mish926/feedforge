"""Content leg: ViT embeddings of movie posters -> FAISS -> candidates.

Poster images are embedded with a pretrained ViT (torchvision ViT-B/16 by
default, classifier head removed, CLS representation used). Embeddings
are L2-normalized so FAISS inner-product search is cosine similarity.

Items with no poster (TMDB misses) simply have no row in the index; the
system degrades gracefully to collaborative-only signal for them, which
is documented behavior, not an accident.

The encoder is injected as a callable (batch of tensors -> batch of
vectors) so tests can substitute a stub and the real ViT is only loaded
when actually embedding, keeping test runs offline and fast.

Content-based candidate generation: embed-lookup the user's last N
history items, average their vectors (a cheap, standard user-content
profile), and retrieve nearest unseen items from FAISS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    import faiss
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install faiss-cpu") from e


# -- embedding -------------------------------------------------------------

def load_vit_encoder(device: str = "cpu") -> Callable:
    """Returns encode(batch_tensor) -> np.ndarray using torchvision
    ViT-B/16 (ImageNet weights, head removed). Kept inside a function so
    importing this module never triggers a weights download."""
    import torch
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).to(device).eval()
    model.heads = torch.nn.Identity()  # CLS representation, 768-d
    preprocess = weights.transforms()

    @torch.no_grad()
    def encode(images) -> np.ndarray:  # images: list[PIL.Image]
        batch = torch.stack([preprocess(img) for img in images]).to(device)
        return model(batch).cpu().numpy()

    return encode


def embed_posters(
    poster_dir: str | Path,
    encoder: Callable,
    batch_size: int = 32,
    out_path: str | Path = "data/content_embeddings.npz",
) -> tuple[np.ndarray, np.ndarray]:
    """Embed every {movie_id}.jpg in poster_dir.

    Saves and returns (item_ids, embeddings) with embeddings
    L2-normalized. item_ids are ORIGINAL MovieLens ids; mapping to dense
    ids happens where the split's item_id_map is available.
    """
    from PIL import Image

    paths = sorted(Path(poster_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(f"no posters in {poster_dir}")

    ids, vecs = [], []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]
        emb = encoder(images)
        vecs.append(emb)
        ids.extend(int(p.stem) for p in chunk)

    embeddings = np.concatenate(vecs).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    item_ids = np.array(ids, dtype=np.int64)
    np.savez(out_path, item_ids=item_ids, embeddings=embeddings)
    return item_ids, embeddings


# -- FAISS index -----------------------------------------------------------

class ContentIndex:
    """Cosine-similarity item index over content embeddings.

    IndexFlatIP is exact search; at 3,700 items approximate search would
    be pure overhead. The FAISS row id -> dense item id mapping is kept
    explicitly because not every item has an embedding.
    """

    def __init__(self, dense_ids: np.ndarray, embeddings: np.ndarray):
        assert len(dense_ids) == len(embeddings)
        self.dense_ids = dense_ids.astype(np.int64)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._row_of = {int(d): i for i, d in enumerate(self.dense_ids)}
        self._emb = embeddings

    @classmethod
    def from_npz(cls, npz_path: str | Path, item_id_map: dict) -> "ContentIndex":
        """Load embeddings saved by embed_posters and translate original
        MovieLens ids to the dense ids used by the rest of the system.
        Items absent from item_id_map (filtered from interactions) are
        dropped."""
        data = np.load(npz_path)
        orig_ids, emb = data["item_ids"], data["embeddings"]
        keep = [i for i, oid in enumerate(orig_ids) if int(oid) in item_id_map]
        dense = np.array([item_id_map[int(orig_ids[i])] for i in keep], dtype=np.int64)
        return cls(dense, emb[keep])

    def vector_of(self, dense_id: int) -> np.ndarray | None:
        row = self._row_of.get(int(dense_id))
        return None if row is None else self._emb[row]

    def candidates_for_history(
        self,
        history: Sequence[int],
        k: int = 100,
        profile_items: int = 10,
    ) -> list[int]:
        """Content candidates for a user: mean embedding of their last
        profile_items history items with content vectors, then top-k
        nearest unseen items. Returns [] if no history item has content
        (graceful degradation to collaborative-only)."""
        vecs = [v for item in history[-profile_items:]
                if (v := self.vector_of(item)) is not None]
        if not vecs:
            return []
        profile = np.mean(vecs, axis=0, keepdims=True).astype(np.float32)
        profile /= np.linalg.norm(profile) + 1e-12

        seen = set(int(i) for i in history)
        # Over-fetch to survive the seen-filter
        _, rows = self.index.search(profile, min(k + len(seen), self.index.ntotal))
        out = []
        for row in rows[0]:
            item = int(self.dense_ids[row])
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out
