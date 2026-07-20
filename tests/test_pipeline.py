"""End-to-end tests on synthetic data with learnable structure.

Synthetic world: items form a cycle, and each user walks it (next item =
current + 1 mod n, with occasional random jumps). A working BERT4Rec
must learn the transition and beat the popularity baseline by a wide
margin within a few epochs; a broken pipeline (masking bug, padding bug,
eval bug) can't. This validates the entire path -- dataframe ->
build_sequences -> MLM dataset -> model -> full-ranking eval -- without
needing the real dataset or a GPU.
"""

import random

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from feedforge.data import (
    PAD,
    MLMSequenceDataset,
    build_sequences,
    inference_batch,
)
from feedforge.evaluate import (
    full_ranking_metrics,
    popularity_baseline,
    sampled_metrics,
)
from feedforge.model import BERT4Rec

N_ITEMS = 60
N_USERS = 300
SEQ_LEN = 30
MAX_LEN = 32


def synthetic_df(seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for user in range(N_USERS):
        item = rng.randint(1, N_ITEMS)
        for t in range(SEQ_LEN):
            rows.append({"user_id": user, "item_id": item, "rating": 5, "timestamp": t})
            if rng.random() < 0.1:  # occasional jump = noise
                item = rng.randint(1, N_ITEMS)
            else:  # the learnable rule
                item = item % N_ITEMS + 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def split():
    return build_sequences(synthetic_df())


def test_split_shapes(split):
    assert len(split.train) == N_USERS
    assert len(split.valid_target) == N_USERS
    assert len(split.test_target) == N_USERS
    assert all(len(s) == SEQ_LEN - 2 for s in split.train)
    assert split.n_items == N_ITEMS


def test_masking_produces_valid_training_pairs(split):
    ds = MLMSequenceDataset(split.train, n_items=split.n_items, max_len=MAX_LEN, seed=0)
    tokens, labels = ds[0]
    assert tokens.shape == (MAX_LEN,) and labels.shape == (MAX_LEN,)
    masked = tokens == split.mask_token
    assert masked.any(), "at least one position must be masked"
    # Every masked position carries its true item as the label
    assert (labels[masked] > 0).all()
    # Every unmasked position has PAD label (no loss there)
    assert (labels[~masked] == PAD).all()


def test_inference_batch_layout(split):
    batch = inference_batch(split.train[:4], split.mask_token, MAX_LEN)
    assert batch.shape == (4, MAX_LEN)
    assert (batch[:, -1] == split.mask_token).all(), "MASK must be the final token"


def test_end_to_end_learning(split):
    """The pipeline must learn the cyclic rule and crush the popularity
    baseline on full-ranking metrics."""
    torch.manual_seed(0)
    ds = MLMSequenceDataset(
        split.train, n_items=split.n_items, max_len=MAX_LEN, mask_prob=0.3, seed=0
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = BERT4Rec(
        vocab_size=split.vocab_size, max_len=MAX_LEN, d_model=32, n_heads=2,
        n_layers=1, d_ff=64,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(15):
        model.train()
        for tokens, labels in loader:
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    metrics = full_ranking_metrics(
        model, split.train, split.valid_target, split.n_items,
        split.mask_token, MAX_LEN,
    )
    baseline = popularity_baseline(
        split.train, split.train, split.valid_target, split.n_items
    )
    # The rule is deterministic 90% of the time; a working model should
    # be far above both chance (10/60 = 0.17) and popularity.
    assert metrics["recall@10"] > 0.5, f"model failed to learn: {metrics}"
    assert metrics["recall@10"] > 2 * baseline["recall@10"], (
        f"model {metrics} vs baseline {baseline}"
    )


def test_sampled_metrics_inflate(split):
    """Demonstrate the Krichene & Rendle effect the project is built to
    avoid: on an UNTRAINED model, sampled evaluation still reports a
    materially higher recall than full ranking (both are near-random,
    but the sampled candidate pool is 101 items vs the full catalog)."""
    torch.manual_seed(1)
    model = BERT4Rec(
        vocab_size=split.vocab_size, max_len=MAX_LEN, d_model=16, n_heads=2,
        n_layers=1, d_ff=32,
    )
    full = full_ranking_metrics(
        model, split.train, split.valid_target, split.n_items,
        split.mask_token, MAX_LEN, ks=(10,),
    )
    sampled = sampled_metrics(
        model, split.train, split.valid_target, split.n_items,
        split.mask_token, MAX_LEN, n_negatives=20,
    )
    assert sampled["sampled20_recall@10"] > 1.5 * full["recall@10"]


def test_filter_seen_actually_filters(split):
    """A target that already appears in the history must be unrankable,
    i.e. never counted as a hit, guarding the filter-seen logic."""
    torch.manual_seed(2)
    model = BERT4Rec(
        vocab_size=split.vocab_size, max_len=MAX_LEN, d_model=16, n_heads=2,
        n_layers=1, d_ff=32,
    )
    hist = [split.train[0]]
    seen_target = [hist[0][0]]  # target is in the history
    m = full_ranking_metrics(
        model, hist, seen_target, split.n_items, split.mask_token, MAX_LEN, ks=(10,)
    )
    assert m["recall@10"] == 0.0
