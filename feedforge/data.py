"""Data pipeline for FeedForge.

MovieLens-1M -> per-user temporally-ordered interaction sequences ->
leave-one-out split -> masked-LM training dataset for BERT4Rec.

Split protocol: per-user leave-one-out (last interaction is the test
target, second-to-last is validation, everything before is training),
with each user's sequence sorted by timestamp. This is the protocol the
BERT4Rec paper and most published baselines use, which keeps our numbers
comparable to the literature. The deliberate departure from the paper is
in evaluation (see evaluate.py): we rank the target against ALL items,
not 100 sampled negatives, because sampled metrics are inconsistent with
true ranking (Krichene & Rendle, KDD 2020) and inflate Recall@10 by
roughly 3x on ML-1M.

Item IDs are remapped to a dense 1..n_items range. ID 0 is reserved for
padding; n_items + 1 is the [MASK] token.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD = 0


@dataclass
class SplitData:
    """Per-user sequences after leave-one-out splitting."""

    train: list[list[int]]      # per user: s[:-2]
    valid_target: list[int]     # per user: s[-2]
    test_target: list[int]      # per user: s[-1]
    n_items: int                # dense item count (ids are 1..n_items)
    item_id_map: dict           # original id -> dense id
    user_ids: list = None       # original user ids, aligned with train rows

    @property
    def mask_token(self) -> int:
        return self.n_items + 1

    @property
    def vocab_size(self) -> int:
        return self.n_items + 2  # + PAD + MASK


def load_movielens_1m(path: str | Path) -> pd.DataFrame:
    """Load ml-1m/ratings.dat (UserID::MovieID::Rating::Timestamp).

    Every rating is treated as an implicit positive interaction, the
    standard protocol for this benchmark.
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def build_sequences(df: pd.DataFrame, min_seq_len: int = 5) -> SplitData:
    """Interactions dataframe -> leave-one-out split sequences.

    Users with fewer than min_seq_len interactions are dropped (need at
    least train material + valid + test). ML-1M users all have >= 20
    ratings so this only matters for other datasets.
    """
    # Dense item ids: 1..n_items (0 reserved for padding)
    unique_items = sorted(df["item_id"].unique())
    item_id_map = {orig: i + 1 for i, orig in enumerate(unique_items)}
    df = df.assign(item=df["item_id"].map(item_id_map))

    # Stable sort by timestamp within user preserves within-second order
    df = df.sort_values(["user_id", "timestamp"], kind="stable")

    train, valid_t, test_t, user_ids = [], [], [], []
    for uid, group in df.groupby("user_id", sort=False):
        seq = group["item"].tolist()
        if len(seq) < min_seq_len:
            continue
        train.append(seq[:-2])
        valid_t.append(seq[-2])
        test_t.append(seq[-1])
        user_ids.append(uid)

    return SplitData(
        train=train,
        valid_target=valid_t,
        test_target=test_t,
        n_items=len(unique_items),
        item_id_map=item_id_map,
        user_ids=user_ids,
    )


class MLMSequenceDataset(Dataset):
    """BERT4Rec training dataset: randomly mask items in each sequence.

    Masking is done inside __getitem__, so every epoch sees different
    masks over the same sequences -- this is the data augmentation that
    makes the cloze objective work.

    Following the paper's training detail: with probability
    last_item_prob, instead of random masking we mask ONLY the final
    item, which matches the inference-time pattern (append [MASK] at the
    end, predict it) and measurably improves next-item metrics.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        n_items: int,
        max_len: int = 200,
        mask_prob: float = 0.2,
        last_item_prob: float = 0.1,
        seed: int | None = None,
    ):
        self.sequences = sequences
        self.n_items = n_items
        self.mask_token = n_items + 1
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.last_item_prob = last_item_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx][-self.max_len:]

        tokens: list[int] = []
        labels: list[int] = []

        if self.rng.random() < self.last_item_prob and len(seq) > 1:
            # Inference-pattern sample: mask only the last position
            tokens = list(seq[:-1]) + [self.mask_token]
            labels = [PAD] * (len(seq) - 1) + [seq[-1]]
        else:
            for item in seq:
                if self.rng.random() < self.mask_prob:
                    tokens.append(self.mask_token)
                    labels.append(item)
                else:
                    tokens.append(item)
                    labels.append(PAD)
            # Degenerate case: nothing got masked -> mask the last item
            if all(l == PAD for l in labels):
                tokens[-1] = self.mask_token
                labels[-1] = seq[-1]

        # Left-pad to max_len (recent items sit at the end, matching the
        # positional embedding usage at inference)
        pad_n = self.max_len - len(tokens)
        tokens = [PAD] * pad_n + tokens
        labels = [PAD] * pad_n + labels
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def inference_batch(
    sequences: list[list[int]], mask_token: int, max_len: int
) -> torch.Tensor:
    """Build the inference input: each sequence + [MASK] appended, left-padded."""
    batch = []
    for seq in sequences:
        s = list(seq[-(max_len - 1):]) + [mask_token]
        batch.append([PAD] * (max_len - len(s)) + s)
    return torch.tensor(batch, dtype=torch.long)
